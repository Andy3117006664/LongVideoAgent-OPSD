"""Reloadable local scorer for lagged/EMA privileged OT-OPSD.

This service keeps the frozen base model in a reserved GPU process and loads
only small PEFT/LoRA adapter snapshots for the current student and the
lagged/EMA teacher.  A trainer publishes a JSON manifest (or calls
``POST /reload``) after an actor update.  The service never calls a remote
provider and fails closed on incomplete adapter directories.

The standard ``e_obs`` field remains the observation-induced change in the
teacher--student gap,

    (D_plus - D_minus),  D = log pi_T - log pi_S.

For privileged OPSD, ``e_priv`` is additionally returned.  It is the same
lagged teacher's likelihood lift on the *same next action* when a
training-only, answer-free visual description ``z`` is supplied:

    e_priv = log pi_T(a | h + z) - log pi_T(a | h).

Only ``e_priv`` should be used by the privileged OPSD advantage hook; ``e_obs``
is retained as a diagnostic/control signal.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("online_opsd_server")


def _load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        logger.exception("failed to load JSON: %s", path)
        return {}


def _sha256_file(path: str | None) -> str | None:
    """Return a short digest for cache provenance, without loading frames."""

    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).split()).encode("utf-8")).hexdigest()


class ReloadablePairedScorer:
    """Thread-safe holder that swaps a fully constructed paired scorer."""

    def __init__(
        self,
        *,
        base_model: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        privileged_cache_path: str = "",
        privileged_max_tokens: int = 192,
    ) -> None:
        self.base_model = str(base_model)
        self.device_name = str(device)
        self.dtype = str(dtype)
        self.trust_remote_code = bool(trust_remote_code)
        self.privileged_cache_path = str(privileged_cache_path or "")
        self.privileged_cache = _load_json(self.privileged_cache_path)
        self.privileged_cache_sha256 = _sha256_file(self.privileged_cache_path)
        self.privileged_max_tokens = max(1, int(privileged_max_tokens))
        self._lock = threading.RLock()
        self._scorer = None
        self._state: dict[str, Any] = {
            "status": "initializing",
            "version": 0,
            "student_adapter": None,
            "teacher_adapter": None,
            "base_model": self.base_model,
            "privileged_cache": self.privileged_cache_path,
            "privileged_cache_sha256": self.privileged_cache_sha256,
        }

    @staticmethod
    def _normalise_adapter(path: Any) -> str | None:
        if path is None or str(path).strip() in {"", "none", "null"}:
            return None
        candidate = Path(str(path)).expanduser().resolve()
        # Do not import project code before the first reload; keep the service
        # usable with a clean base model for a smoke test.
        if not (candidate.is_dir() and (candidate / "READY").is_file()):
            raise FileNotFoundError(f"adapter is not atomically ready: {candidate}")
        if not (candidate / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter config missing: {candidate}")
        if not ((candidate / "adapter_model.safetensors").is_file() or (candidate / "adapter_model.bin").is_file()):
            raise FileNotFoundError(f"adapter weights missing: {candidate}")
        return str(candidate)

    @staticmethod
    def _version_int(value: Any) -> int | None:
        """Parse the monotone snapshot version used by trainer reloads."""

        if value is None or str(value).strip() == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"snapshot version must be an integer, got {value!r}")

    def _build_scorer(self, student_adapter: str | None, teacher_adapter: str | None):
        # Importing this module only after the service starts avoids imposing
        # torch/Transformers on ordinary LongVideoAgent workers.
        try:
            from .online_ot_scorer import OnlinePairedScorer
        except ImportError:
            from online_ot_scorer import OnlinePairedScorer

        return OnlinePairedScorer(
            student_model=self.base_model,
            teacher_model=self.base_model,
            device=self.device_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            student_adapter=student_adapter,
            teacher_adapter=teacher_adapter,
        )

    def reload(
        self,
        *,
        student_adapter: Any = None,
        teacher_adapter: Any = None,
        version: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        student = self._normalise_adapter(student_adapter)
        teacher = self._normalise_adapter(teacher_adapter)
        requested_version = self._version_int(version)
        logger.info("loading OPSD pair student=%s teacher=%s", student, teacher)
        # Keep the scorer (and its frozen base weights) resident.  Once the
        # first adapter pair is loaded, OnlinePairedScorer.reload uses PEFT's
        # in-place load_adapter/set_adapter path, so each refresh transfers
        # only the tiny LoRA files instead of a second full 3B model pair.
        with self._lock:
            current_version = self._version_int(self._state.get("version"))
            if requested_version is not None and current_version is not None:
                if requested_version < current_version:
                    raise ValueError(
                        f"stale snapshot version {requested_version}; current scorer version is {current_version}"
                    )
                if requested_version == current_version:
                    # Retries of the same publication are idempotent only when
                    # they name the exact same adapter pair.  This prevents a
                    # delayed worker from replacing a committed snapshot under
                    # an old step number.
                    current_student = str(self._state.get("student_adapter") or "")
                    current_teacher = str(self._state.get("teacher_adapter") or "")
                    if current_student or current_teacher:
                        if current_student != str(student or "") or current_teacher != str(teacher or ""):
                            raise ValueError(
                                f"snapshot version {requested_version} already committed with a different adapter pair"
                            )
            if self._scorer is None:
                new_scorer = self._build_scorer(student, teacher)
                if metadata:
                    if metadata.get("student_step") is not None:
                        new_scorer.student_step = metadata.get("student_step")
                    if metadata.get("teacher_step") is not None:
                        new_scorer.teacher_step = metadata.get("teacher_step")
                    new_scorer.snapshot_mode = str(
                        metadata.get("snapshot_mode")
                        or metadata.get("mode")
                        or ("ema_actor" if metadata.get("ema_decay") is not None else "lagged_actor")
                    )
                    if metadata.get("ema_decay") is not None:
                        new_scorer.ema_decay = metadata.get("ema_decay")
                self._scorer = new_scorer
                reload_kind = "initial"
            else:
                result = self._scorer.reload(
                    student_model=self.base_model,
                    teacher_model=self.base_model,
                    student_adapter=student,
                    teacher_adapter=teacher,
                    base_model=self.base_model,
                    student_step=(None if metadata is None else metadata.get("student_step")),
                    teacher_step=(None if metadata is None else metadata.get("teacher_step")),
                    snapshot_mode=(
                        (metadata.get("snapshot_mode") or metadata.get("mode"))
                        if metadata
                        else "lagged_actor"
                    ),
                    ema_decay=(None if metadata is None else metadata.get("ema_decay")),
                )
                reload_kind = result.get("reload", "pair")
            state = dict(self._state)
            state.update(
                {
                    "status": "ready",
                    "version": requested_version if requested_version is not None else int(state.get("version", 0)) + 1,
                    "student_adapter": student,
                    "teacher_adapter": teacher,
                    "student_adapter_sha256": self._adapter_digest(student),
                    "teacher_adapter_sha256": self._adapter_digest(teacher),
                    "base_model": self.base_model,
                    "privileged_cache": self.privileged_cache_path,
                    "reload": reload_kind,
                }
            )
            if metadata:
                state["metadata"] = metadata
            # Mirror the scorer's immutable snapshot provenance in /health;
            # this is useful when the trainer and GPU-7 service are restarted
            # independently and a stale manifest is accidentally replayed.
            if self._scorer is not None:
                try:
                    state.update(
                        {
                            "student_step": self._scorer.student_step,
                            "teacher_step": self._scorer.teacher_step,
                            "snapshot_mode": self._scorer.snapshot_mode,
                            "ema_decay": self._scorer.ema_decay,
                            "tokenizer_fingerprint": self._scorer.tokenizer_fingerprint,
                        }
                    )
                except Exception:
                    logger.debug("could not copy scorer provenance into state", exc_info=True)
            self._state = state
            return dict(self._state)

    @staticmethod
    def _adapter_digest(path: str | None) -> str | None:
        if not path:
            return None
        digest = hashlib.sha256()
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            file_path = Path(path) / name
            if not file_path.is_file():
                continue
            digest.update(name.encode("utf-8"))
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()[:24]

    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def scorer(self):
        with self._lock:
            if self._scorer is None:
                raise RuntimeError("paired scorer is not ready")
            return self._scorer

    def maybe_manifest_reload(self, manifest_path: str) -> bool:
        payload = _load_json(manifest_path)
        if not payload:
            return False
        version = payload.get("version", payload.get("student_step"))
        parsed_version = self._version_int(version)
        with self._lock:
            current_version = self._version_int(self._state.get("version"))
            if parsed_version is not None and current_version is not None and parsed_version == current_version:
                same_pair = (
                    str(self._state.get("student_adapter") or "") == str(self._normalise_adapter(payload.get("student_adapter")) or "")
                    and str(self._state.get("teacher_adapter") or "") == str(self._normalise_adapter(payload.get("teacher_adapter")) or "")
                )
                if same_pair:
                    return False
        self.reload(
            student_adapter=payload.get("student_adapter"),
            teacher_adapter=payload.get("teacher_adapter"),
            version=version,
            metadata=payload,
        )
        return True

    def privileged_entry(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Return a validated, answer-free visual ``z`` entry for one clip.

        The cache is intentionally keyed by the rollout's *post-tool* clip.
        ``z`` is an additional teacher-only view of that same clip; it is
        never copied into the student observation.  We reject obvious answer
       /option strings so a malformed cache cannot turn the signal into a
        hindsight label.
        """

        clip = str(record.get("current_vid_after") or record.get("current_vid") or "").strip()
        if not clip:
            return None
        cache = self.privileged_cache
        clips = cache.get("clips") if isinstance(cache, dict) else None
        entry = clips.get(clip) if isinstance(clips, dict) else None
        if entry is None and isinstance(cache, dict):
            entry = cache.get(clip)
        if isinstance(entry, dict):
            text = entry.get("z_text") or entry.get("observation") or entry.get("text") or entry.get("description") or ""
            # Do not infer answer-freeness from a bare description.  A
            # privileged cache is a training-only label source, so missing
            # provenance must fail closed; producers should set this flag at
            # the cache top level or on every clip entry.
            answer_free = entry.get("z_answer_free", cache.get("z_answer_free", False))
            answer_free_ok = answer_free is True or str(answer_free).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
            if not answer_free_ok:
                return None
            frame_sha = entry.get("frame_sha256")
            frame_paths = entry.get("frame_paths")
        else:
            text = entry or ""
            frame_sha = None
            frame_paths = None
            answer_free = cache.get("z_answer_free", False) if isinstance(cache, dict) else False
            answer_free_ok = answer_free is True or str(answer_free).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
            if not answer_free_ok:
                return None
        text = str(text).strip()
        if not text or text.startswith("["):
            return None
        # Cache generation is expected to be generic and answer-free.  These
        # checks are deliberately conservative; a rejected row becomes a
        # miss rather than an un-auditable training label.
        forbidden = (
            r"<\s*answer\b",
            r"\bground\s*truth\b",
            r"\bcorrect\s+(?:answer|option)\b",
            r"\b(?:a[0-4]|option\s*[a-e])\s*[:=]",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in forbidden):
            return None
        if isinstance(frame_sha, list):
            frame_sha = [str(value) for value in frame_sha if str(value).strip()]
        elif frame_sha:
            frame_sha = [str(frame_sha)]
        else:
            frame_sha = []
        return {
            "clip": clip,
            "text": text,
            "z_sha256": _sha256_text(text),
            "frame_sha256": frame_sha,
            "frame_paths": frame_paths if isinstance(frame_paths, list) else [],
        }

    def score_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        scorer = self.scorer()
        prompt_ids = payload.get("prompt_ids")
        response_ids = payload.get("response_ids")
        response_mask = payload.get("response_mask")
        records = payload.get("records")
        if not all(isinstance(value, list) for value in (prompt_ids, response_ids, response_mask, records)):
            raise ValueError("prompt/response/mask/records must be lists")
        if len(response_ids) != len(response_mask):
            raise ValueError("response_ids and response_mask lengths differ")
        if not all(isinstance(record, dict) for record in records):
            raise ValueError("records must contain JSON objects")
        # The formal privileged OPSD arm only needs the teacher's two
        # likelihood evaluations.  Keep the four-forward gap shift as an
        # explicit diagnostic/control (`standard_gap=true`) so the pilot does
        # not pay for it accidentally.
        standard_gap = bool(payload.get("standard_gap", False))
        if standard_gap:
            hits, _, status = scorer.score_records(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                records=records,
                reduction=str(payload.get("reduction", "mean")),
                batch_size=max(1, int(payload.get("batch_size", 2))),
                max_records=max(0, int(payload.get("max_records", 0))),
            )
        else:
            hits, status = 0, "privileged_only"
        privileged_hits = 0
        if bool(payload.get("privileged", True)):
            privileged_hits = self._attach_privileged(
                records,
                prompt_ids,
                response_ids,
                response_mask,
                scorer,
                batch_size=max(1, int(payload.get("batch_size", 2))),
            )
        return {
            "ok": True,
            "status": str(status),
            "hits": int(hits),
            "standard_hits": int(hits),
            "privileged_hits": int(privileged_hits),
            "hits_total": int(max(hits, privileged_hits)),
            "privileged_misses": int(max(0, len(records) - privileged_hits))
            if bool(payload.get("privileged", True))
            else 0,
            "misses": int(max(0, len(records) - max(hits, privileged_hits))),
            "records": records,
            "scorer_state": self.state(),
        }

    def _attach_privileged(self, records, prompt_ids, response_ids, response_mask, scorer, *, batch_size: int = 2) -> int:
        """Attach e_priv on the same post-observation next-action span."""

        try:
            import torch
        except Exception:
            return 0
        tokenizer = scorer.teacher_tokenizer
        pad_id = scorer.pad_id
        contexts_plain: list[list[int]] = []
        contexts_priv: list[list[int]] = []
        targets: list[list[int]] = []
        eligible: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        response = [int(x) for x in response_ids]
        prompt = [int(x) for x in prompt_ids]
        for record in records:
            if not isinstance(record, dict):
                continue
            # Reuse the paired scorer's strict action/evidence/span contract.
            # Privileged evidence is an independent arm and may supplement a
            # static/ordinary ``e_obs`` hit.  Do not overwrite an already
            # attached privileged score, however; this also makes retries
            # idempotent when a client times out after the server committed.
            if record.get("ot_privileged_hit") is True or str(record.get("ot_privileged_hit") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }:
                continue
            if not scorer._eligible(
                record,
                response,
                [int(x) for x in response_mask],
                reject_existing=False,
            ):
                continue
            try:
                o0 = int(record["observation_response_start"])
                o1 = int(record["observation_response_end"])
                t0 = int(record["next_action_response_start"])
                t1 = int(record["next_action_response_end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= o0 < o1 <= t0 < t1 <= len(response)):
                continue
            if len(response_mask) != len(response):
                continue
            try:
                if any(float(value) != 0.0 for value in response_mask[o0:o1]):
                    continue
                if any(float(value) != 1.0 for value in response_mask[t0:t1]):
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            entry = self.privileged_entry(record)
            if entry is None:
                continue
            z = entry["text"]
            marker = "\n<privileged_visual>" + z + "</privileged_visual>\n"
            z_ids = tokenizer.encode(marker, add_special_tokens=False)
            if not z_ids:
                continue
            z_ids = z_ids[: self.privileged_max_tokens]
            plain_context = prompt + response[:o1]
            contexts_plain.append(plain_context)
            # ``z`` is an *additional* teacher-only view.  Retain the exact
            # deployment observation and append a fixed sentinel, so e_priv
            # isolates the value of privileged visual evidence rather than
            # conflating it with an observation replacement.
            contexts_priv.append(plain_context + z_ids)
            targets.append(response[t0:t1])
            eligible.append((record, entry, len(z_ids)))
        if not eligible:
            return 0
        # We need the teacher's likelihood on both contexts.  Reuse the
        # scorer's private batch helper to preserve exact left-padding and
        # logits-to-keep semantics.
        # Resolve the helper normally for both package and script launches.
        try:
            from .online_ot_scorer import _score_batch
        except ImportError:
            from online_ot_scorer import _score_batch
        gains: list[float] = []
        bs = max(1, min(8, int(batch_size), len(eligible)))
        with scorer._forward_lock:
            for start in range(0, len(eligible), bs):
                sl = slice(start, start + bs)
                plain = _score_batch(scorer.teacher, contexts_plain[sl], targets[sl], device=scorer.device, pad_id=pad_id)
                priv = _score_batch(scorer.teacher, contexts_priv[sl], targets[sl], device=scorer.device, pad_id=pad_id)
                for (record, entry, z_len), p, q in zip(eligible[sl], plain, priv, strict=True):
                    if not p or not q:
                        continue
                    d = float(sum(q) / len(q) - sum(p) / len(p))
                    if not torch.isfinite(torch.tensor(d)):
                        continue
                    record["e_priv"] = d
                    record["e_priv_orientation"] = "teacher_priv_minus_deploy"
                    record["ot_privileged_hit"] = True
                    record["ot_privileged_source"] = "local_visual_cache"
                    record["ot_privileged_cache"] = self.privileged_cache_path
                    record["ot_privileged_cache_sha256"] = self.privileged_cache_sha256
                    record["ot_privileged_z_sha256"] = entry["z_sha256"]
                    record["ot_privileged_frame_sha256"] = entry["frame_sha256"]
                    record["ot_privileged_frame_paths"] = entry["frame_paths"]
                    record["ot_privileged_clip"] = entry["clip"]
                    record["ot_privileged_tokens"] = z_len
                    record["ot_privileged_context_mode"] = "append_after_deploy_observation"
                    record["ot_privileged_student_step"] = getattr(scorer, "student_step", None)
                    record["ot_privileged_teacher_step"] = getattr(scorer, "teacher_step", None)
                    record["ot_privileged_snapshot_mode"] = getattr(scorer, "snapshot_mode", "")
                    record["ot_privileged_teacher_model"] = getattr(scorer, "teacher_model_path", "")
                    record["ot_privileged_student_adapter"] = getattr(scorer, "student_adapter_path", "")
                    record["ot_privileged_teacher_adapter"] = getattr(scorer, "teacher_adapter_path", "")
                    record["ot_privileged_tokenizer_fingerprint"] = getattr(scorer, "tokenizer_fingerprint", "")
                    gains.append(d)
        return len(gains)


def create_app(
    *,
    base_model: str,
    device: str = "cuda:0",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    privileged_cache_path: str = "",
    privileged_max_tokens: int = 192,
    initial_student_adapter: str | None = None,
    initial_teacher_adapter: str | None = None,
    manifest: str = "",
    manifest_poll_seconds: float = 2.0,
):
    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:
        raise RuntimeError("online_opsd_server requires fastapi and uvicorn") from exc

    holder = ReloadablePairedScorer(
        base_model=base_model,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        privileged_cache_path=privileged_cache_path,
        privileged_max_tokens=privileged_max_tokens,
    )
    # Keep startup cheap when the trainer will publish its first actor
    # snapshot immediately.  If explicit initial adapters were supplied, load
    # them now; otherwise /health reports ``initializing`` until /reload.
    if initial_student_adapter or initial_teacher_adapter:
        holder.reload(
            student_adapter=initial_student_adapter,
            teacher_adapter=initial_teacher_adapter,
            version=0,
        )

    app = FastAPI(title="LongVideoAgent local OT-OPSD scorer")

    @app.get("/health")
    def health() -> dict[str, Any]:
        state = holder.state()
        return {"ok": state.get("status") == "ready", **state}

    @app.get("/state")
    def state() -> dict[str, Any]:
        return {"ok": True, **holder.state()}

    @app.post("/reload")
    def reload(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        try:
            state = holder.reload(
                student_adapter=payload.get("student_adapter"),
                teacher_adapter=payload.get("teacher_adapter"),
                version=payload.get("version", payload.get("student_step")),
                metadata=payload,
            )
            return {"ok": True, **state}
        except Exception as exc:
            logger.exception("reload failed")
            raise HTTPException(status_code=400, detail=f"reload failed: {type(exc).__name__}: {exc}") from exc

    @app.post("/score")
    def score(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return holder.score_payload(payload)
        except Exception as exc:
            logger.exception("paired scoring request failed")
            records = payload.get("records", []) if isinstance(payload, dict) else []
            return {
                "ok": True,
                "status": f"error:{type(exc).__name__}",
                "hits": 0,
                "standard_hits": 0,
                "privileged_hits": 0,
                "privileged_misses": len(records),
                "misses": len(records),
                "records": records,
            }

    if manifest:
        def watch() -> None:
            last_error = ""
            while True:
                try:
                    holder.maybe_manifest_reload(manifest)
                    last_error = ""
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    if msg != last_error:
                        logger.exception("manifest reload failed")
                        last_error = msg
                time.sleep(max(0.25, float(manifest_poll_seconds)))
        thread = threading.Thread(target=watch, name="opsd-manifest-watcher", daemon=True)
        thread.start()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--privileged-cache", default="")
    parser.add_argument("--privileged-max-tokens", type=int, default=192)
    parser.add_argument("--initial-student-adapter", default="")
    parser.add_argument("--initial-teacher-adapter", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--manifest-poll-seconds", type=float, default=2.0)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("online_opsd_server requires uvicorn") from exc
    app = create_app(
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        privileged_cache_path=args.privileged_cache,
        privileged_max_tokens=args.privileged_max_tokens,
        initial_student_adapter=args.initial_student_adapter or None,
        initial_teacher_adapter=args.initial_teacher_adapter or None,
        manifest=args.manifest,
        manifest_poll_seconds=args.manifest_poll_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
