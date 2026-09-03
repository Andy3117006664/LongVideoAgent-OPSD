"""Optional in-process paired scorer for LongVideoAgent rollouts.

The regular OT cache is deliberately strict and therefore only accepts a
record whose response hash was produced by the same rollout.  This module is
the online alternative: after a rollout has finished (and all next-action
spans are known), it evaluates the *actual* response under literal prefixes
``h_minus = prompt + response[:obs_start]`` and
``h_plus = prompt + response[:obs_end]``.  Teacher/student log-probabilities
are reduced to ``e_obs = D_plus - D_minus`` and attached to that record.
Here ``D`` is the teacher-minus-student gap on the next action; ``plus`` is
the exact prefix after the observation and ``minus`` is the prefix before it.
The signed value is therefore the observation-induced change in the gap (the
same orientation as ``ot-opd-paired-v2``), not an absolute ``gap-closing``
metric.

It is opt-in because loading two causal LMs inside a rollout worker consumes
substantial memory.  No Transformers/Torch import happens until the scorer is
requested.  Failures are returned to the caller and never produce a proxy
signal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_ACTION_TYPES = frozenset(
    {
        "request_grounding",
        "visual_query",
        "search",
        "seek",
        "query",
        "acquire",
    }
)


def _tokenizer_signature(tokenizer: Any) -> str:
    """Return a cheap compatibility signature for two tokenizers."""

    probes = ("hello", "<reasoning>", "<search>x</search>", "你好")
    payload = {
        "vocab_size": int(len(tokenizer)),
        "ids": [tokenizer.encode(text, add_special_tokens=False) for text in probes],
        "bos": getattr(tokenizer, "bos_token_id", None),
        "eos": getattr(tokenizer, "eos_token_id", None),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:24]


def _reduce(values: Sequence[float], reduction: str) -> float:
    if not values:
        raise ValueError("cannot reduce an empty log-probability vector")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("non-finite log-probability")
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"unsupported paired-score reduction: {reduction!r}")
    if reduction == "sum":
        return float(sum(float(value) for value in values))
    return float(sum(float(value) for value in values) / len(values))


def _is_adapter_dir(value: str | os.PathLike[str] | None) -> bool:
    """Whether ``value`` looks like a published PEFT adapter snapshot."""

    if not value:
        return False
    path = Path(value).expanduser()
    return path.is_dir() and (path / "READY").is_file() and (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    )


def _resolve_model_spec(
    model: str | None,
    *,
    adapter: str | None,
    base_model: str | None,
) -> tuple[str, str | None]:
    """Resolve a full HF model or a LoRA adapter + shared base model."""

    model = str(model or "").strip()
    adapter = str(adapter or "").strip()
    base_model = str(base_model or "").strip()
    # An adapter can be passed in the historical ``student_model`` slot.
    if not adapter and _is_adapter_dir(model):
        adapter, model = model, ""
    if adapter:
        if not _is_adapter_dir(adapter):
            raise FileNotFoundError(f"LoRA adapter is not a published READY snapshot: {adapter}")
        if not base_model:
            try:
                config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
                base_model = str(config.get("base_model_name_or_path") or "").strip()
            except Exception:
                base_model = ""
        if not base_model:
            raise ValueError("base_model is required when loading a LoRA adapter snapshot")
        return base_model, adapter
    if not model:
        raise ValueError("a full model path or adapter snapshot is required")
    return model, None


def _score_batch(
    model: Any,
    contexts: list[list[int]],
    targets: list[list[int]],
    *,
    device: Any,
    pad_id: int,
) -> list[list[float]]:
    """Score variable-length ``target`` continuations for one model."""

    import torch

    if not contexts or len(contexts) != len(targets):
        return []
    if any(not context or not target for context, target in zip(contexts, targets, strict=True)):
        raise ValueError("empty context/target in paired scorer")

    sequences = [context + target for context, target in zip(contexts, targets, strict=True)]
    max_len = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), max_len), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    context_lengths: list[int] = []
    target_lengths: list[int] = []
    for row, (context, target) in enumerate(zip(contexts, targets, strict=True)):
        sequence = context + target
        offset = max_len - len(sequence)
        ids[row, offset:] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention[row, offset:] = 1
        context_lengths.append(len(context))
        target_lengths.append(len(target))

    # Explicitly reset positions after left padding.  Several decoder-only
    # model implementations default to absolute positions over the padded
    # tensor, which would shift shorter rows and make their log-probabilities
    # differ from the corresponding unpadded prefix.  The cumulative mask
    # gives every real token the same position it would have in isolation.
    position_ids = attention.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention == 0, 0)

    # The target is always the final continuation, so only the logits at the
    # end of each sequence are needed.  Recent Transformers models expose
    # ``logits_to_keep``; use it when available to avoid materialising a
    # vocab-sized tensor for every prompt position.  The fallback keeps the
    # same semantics for older model classes.
    max_target_len = max(target_lengths)
    keep = min(max_len, max_target_len + 1)
    with torch.inference_mode():
        try:
            logits = model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=keep,
            ).logits
        except TypeError:
            logits = model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=position_ids,
                use_cache=False,
            ).logits

    if logits.ndim != 3:
        raise ValueError(f"paired scorer expected [batch, sequence, vocab] logits, got {tuple(logits.shape)}")
    logits_len = int(logits.shape[1])
    if logits_len not in {max_len, keep}:
        raise ValueError(
            f"paired scorer returned {logits_len} positions; expected full={max_len} or kept={keep}"
        )

    result: list[list[float]] = []
    for row, (context_len, target_len) in enumerate(zip(context_lengths, target_lengths, strict=True)):
        offset = max_len - (context_len + target_len)
        # Position p predicts token p+1; the first target token is therefore
        # read from context_len - 1.
        positions = torch.arange(
            offset + context_len - 1,
            offset + context_len + target_len - 1,
            device=device,
        )
        token_ids = ids[row, offset + context_len : offset + context_len + target_len]
        if logits_len != max_len:
            # ``logits_to_keep`` retains the final ``keep`` sequence
            # positions.  Left padding means every row's real sequence ends
            # at ``max_len``, so the absolute positions map directly to this
            # tail window.
            positions = positions - (max_len - logits_len)
            if int(positions.min()) < 0 or int(positions.max()) >= logits_len:
                raise ValueError("paired scorer target positions fell outside logits_to_keep window")
        selected = logits[row, positions].float()
        token_logits = selected.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        values = (token_logits - torch.logsumexp(selected, dim=-1)).detach().cpu().tolist()
        result.append([float(value) for value in values])
    return result


def _matched_null_tokens(tokenizer: Any, length: int) -> list[int]:
    """Build a deterministic observation-null span with exactly ``length`` tokens.

    The historical scorer deletes the observation. Formal transition credit
    instead replaces it with a fixed marker sequence, preserving token count
    and positions for a schema-matched counterfactual.
    """
    length = max(0, int(length))
    if length == 0:
        return []
    marker = tokenizer.encode("<observation_mask>", add_special_tokens=False)
    if not marker:
        marker_id = getattr(tokenizer, "unk_token_id", None)
        if marker_id is None:
            marker_id = getattr(tokenizer, "eos_token_id", None)
        if marker_id is None:
            marker_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        marker = [int(marker_id)]
    return [int(marker[index % len(marker)]) for index in range(length)]


class OnlinePairedScorer:
    """Process-local singleton holding one teacher and one student model."""

    _instances: dict[tuple[Any, ...], "OnlinePairedScorer"] = {}
    _errors: dict[tuple[Any, ...], str] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(
        cls,
        *,
        student_model: str | None = None,
        teacher_model: str | None = None,
        student_adapter: str | None = None,
        teacher_adapter: str | None = None,
        base_model: str | None = None,
        device: str = "cuda:0",
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> "OnlinePairedScorer":
        """Load/reuse a scorer for one explicit model/device tuple."""

        key = (
            str(student_model or ""),
            str(teacher_model or ""),
            str(student_adapter or ""),
            str(teacher_adapter or ""),
            str(base_model or ""),
            str(device),
            str(dtype),
            bool(trust_remote_code),
        )
        with cls._instances_lock:
            if key in cls._instances:
                return cls._instances[key]
            if key in cls._errors:
                raise RuntimeError(cls._errors[key])
            try:
                scorer = cls(
                    student_model=student_model,
                    teacher_model=teacher_model,
                    student_adapter=student_adapter,
                    teacher_adapter=teacher_adapter,
                    base_model=base_model,
                    device=device,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:
                message = f"online paired scorer initialization failed: {type(exc).__name__}: {exc}"
                cls._errors[key] = message
                raise RuntimeError(message) from exc
            cls._instances[key] = scorer
            return scorer

    def __init__(
        self,
        *,
        student_model: str | None = None,
        teacher_model: str | None = None,
        student_adapter: str | None = None,
        teacher_adapter: str | None = None,
        base_model: str | None = None,
        student_step: int | None = None,
        teacher_step: int | None = None,
        snapshot_mode: str = "fixed",
        ema_decay: float | None = None,
        device: str = "cuda:0",
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        # Imports are intentionally local: normal LongVideoAgent runs do not
        # require a second pair of models or the Transformers dependency.
        import torch

        self._forward_lock = threading.RLock()
        self._dtype_name = str(dtype)
        self._trust_remote_code = bool(trust_remote_code)
        self._dtype_obj = None
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested scorer device is unavailable: {self.device}")
        if dtype == "auto":
            if self.device.type == "cuda":
                bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                dtype_obj = torch.bfloat16 if bf16_supported else torch.float16
            else:
                dtype_obj = torch.float32
        else:
            try:
                dtype_obj = getattr(torch, str(dtype))
            except AttributeError as exc:
                raise ValueError(f"unsupported scorer dtype: {dtype}") from exc
        self._dtype_obj = dtype_obj
        self.student = None
        self.teacher = None
        self.student_tokenizer = None
        self.teacher_tokenizer = None
        self.student_model_path = ""
        self.teacher_model_path = ""
        self.student_adapter_path = ""
        self.teacher_adapter_path = ""
        self.base_model_path = ""
        self.student_step = student_step
        self.teacher_step = teacher_step
        self.snapshot_mode = str(snapshot_mode or "fixed")
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        self.null_mode = str(os.getenv("OT_OPD_NULL_MODE", "literal") or "literal").strip().lower()
        if self.null_mode in {"matched", "schema_matched", "matched_length", "mask"}:
            self.null_mode = "matched_mask"
        elif self.null_mode not in {"literal", "matched_mask"}:
            self.null_mode = "literal"
        self._student_adapter_name = None
        self._teacher_adapter_name = None
        self._load_pair(
            student_model=student_model,
            teacher_model=teacher_model,
            student_adapter=student_adapter,
            teacher_adapter=teacher_adapter,
            base_model=base_model,
            trust_remote_code=trust_remote_code,
        )

    @staticmethod
    def _load_one(
        *,
        source_model: str,
        adapter_path: str | None,
        dtype_obj: Any,
        device: Any,
        trust_remote_code: bool,
        adapter_name: str,
    ) -> tuple[Any, Any]:
        """Load one HF model, optionally wrapping it in a PEFT adapter."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source_model, use_fast=True, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            source_model,
            torch_dtype=dtype_obj,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except Exception as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("LoRA scorer snapshots require peft") from exc
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                adapter_name=adapter_name,
                is_trainable=False,
            )
        model = model.to(device).eval()
        return tokenizer, model

    def _load_pair(
        self,
        *,
        student_model: str | None,
        teacher_model: str | None,
        student_adapter: str | None,
        teacher_adapter: str | None,
        base_model: str | None,
        trust_remote_code: bool,
    ) -> None:
        """Load and validate a complete student/teacher pair."""

        student_source, student_adapter = _resolve_model_spec(
            student_model, adapter=student_adapter, base_model=base_model
        )
        teacher_source, teacher_adapter = _resolve_model_spec(
            teacher_model, adapter=teacher_adapter, base_model=base_model or student_source
        )
        student_name = "student_initial"
        teacher_name = "teacher_initial"
        student_tokenizer, student = self._load_one(
            source_model=student_source,
            adapter_path=student_adapter,
            dtype_obj=self._dtype_obj,
            device=self.device,
            trust_remote_code=trust_remote_code,
            adapter_name=student_name,
        )
        try:
            teacher_tokenizer, teacher = self._load_one(
                source_model=teacher_source,
                adapter_path=teacher_adapter,
                dtype_obj=self._dtype_obj,
                device=self.device,
                trust_remote_code=trust_remote_code,
                adapter_name=teacher_name,
            )
        except Exception:
            del student
            try:
                import gc
                import torch

                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            raise
        student_fp = _tokenizer_signature(student_tokenizer)
        teacher_fp = _tokenizer_signature(teacher_tokenizer)
        if student_fp != teacher_fp:
            raise ValueError("teacher/student tokenizer signatures differ; tokenwise pairing is invalid")
        if student_tokenizer.pad_token_id is None:
            if student_tokenizer.eos_token_id is None:
                raise ValueError("student tokenizer has neither pad_token_id nor eos_token_id")
            student_tokenizer.pad_token = student_tokenizer.eos_token

        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.tokenizer_fingerprint = student_fp
        self.pad_id = int(student_tokenizer.pad_token_id)
        self.student = student
        self.teacher = teacher
        self.student_model_path = str(student_model or student_source)
        self.teacher_model_path = str(teacher_model or teacher_source)
        self.student_adapter_path = str(student_adapter or "")
        self.teacher_adapter_path = str(teacher_adapter or "")
        self.base_model_path = str(base_model or (student_source if student_adapter else ""))
        self._student_adapter_name = student_name if student_adapter else None
        self._teacher_adapter_name = teacher_name if teacher_adapter else None

    def snapshot_state(self) -> dict[str, Any]:
        """Return JSON-safe model/snapshot provenance for ``/health`` and logs."""

        return {
            "student_model": self.student_model_path,
            "teacher_model": self.teacher_model_path,
            "student_adapter": self.student_adapter_path or None,
            "teacher_adapter": self.teacher_adapter_path or None,
            "base_model": self.base_model_path or None,
            "student_step": self.student_step,
            "teacher_step": self.teacher_step,
            "snapshot_mode": self.snapshot_mode,
            "ema_decay": self.ema_decay,
            "null_mode": self.null_mode,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }

    def _try_swap_adapter(self, model: Any, adapter_path: str, adapter_name: str, old_name: str | None) -> bool:
        """Load a tiny adapter into an existing PEFT model when supported.

        PEFT has kept ``load_adapter``/``set_adapter`` stable, but older
        releases do not expose ``delete_adapter``.  In that case the old
        adapter remains resident (still correct, but with a small memory
        leak), and the caller can choose a full reload at the next refresh.
        """

        load_adapter = getattr(model, "load_adapter", None)
        set_adapter = getattr(model, "set_adapter", None)
        if not callable(load_adapter) or not callable(set_adapter):
            return False
        try:
            try:
                load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
            except TypeError:
                load_adapter(adapter_path, adapter_name=adapter_name)
            set_adapter(adapter_name)
            delete_adapter = getattr(model, "delete_adapter", None)
            if old_name and callable(delete_adapter):
                try:
                    delete_adapter(old_name)
                except Exception:
                    logger.debug("PEFT delete_adapter(%s) failed; retaining old adapter", old_name, exc_info=True)
            return True
        except Exception:
            logger.debug("in-place PEFT adapter swap failed", exc_info=True)
            return False

    def reload(
        self,
        *,
        student_model: str | None = None,
        teacher_model: str | None = None,
        student_adapter: str | None = None,
        teacher_adapter: str | None = None,
        base_model: str | None = None,
        student_step: int | None = None,
        teacher_step: int | None = None,
        snapshot_mode: str = "lagged_actor",
        ema_decay: float | None = None,
    ) -> dict[str, Any]:
        """Atomically switch scorer weights to a new actor snapshot pair.

        Adapter-only refreshes use PEFT's in-place loader and therefore keep
        the base model resident on reserved GPU 7.  If that API is missing or
        the tokenizer/base changes, a serialized full pair reload is used;
        old models are released before loading the new pair to avoid a
        transient 3B+3B+3B+3B GPU allocation.  Any failure raises and leaves
        the old pair untouched for in-place swaps, or leaves the service
        unavailable for a full reload (the HTTP endpoint reports an error and
        the rollout path fails closed).
        """

        # Resolve adapter-vs-full paths before acquiring the forward lock so
        # malformed/incomplete snapshots cannot block scoring.
        next_student_source, next_student_adapter = _resolve_model_spec(
            student_model or self.student_model_path,
            adapter=student_adapter or self.student_adapter_path or None,
            base_model=base_model or self.base_model_path or None,
        )
        next_teacher_source, next_teacher_adapter = _resolve_model_spec(
            teacher_model or self.teacher_model_path,
            adapter=teacher_adapter or self.teacher_adapter_path or None,
            base_model=base_model or self.base_model_path or next_student_source,
        )
        next_base = str(base_model or (next_student_source if next_student_adapter else self.base_model_path or ""))

        with self._forward_lock:
            # Fast path: same base/tokenizer, adapter snapshots only.  Each
            # model gets a unique adapter name so an in-flight scorer never
            # observes a partially loaded tensor set.
            same_base = bool(next_student_adapter and next_teacher_adapter and self.student_adapter_path and self.teacher_adapter_path)
            same_base = same_base and os.path.abspath(next_student_source) == os.path.abspath(self.base_model_path or next_student_source)
            same_base = same_base and os.path.abspath(next_teacher_source) == os.path.abspath(self.base_model_path or next_teacher_source)
            if same_base:
                import uuid

                student_name = f"student_{student_step if student_step is not None else 'next'}_{uuid.uuid4().hex[:8]}"
                teacher_name = f"teacher_{teacher_step if teacher_step is not None else 'next'}_{uuid.uuid4().hex[:8]}"
                student_ok = self._try_swap_adapter(
                    self.student, next_student_adapter, student_name, self._student_adapter_name
                )
                teacher_ok = self._try_swap_adapter(
                    self.teacher, next_teacher_adapter, teacher_name, self._teacher_adapter_name
                )
                if student_ok and teacher_ok:
                    self.student_model_path = str(student_model or next_student_source)
                    self.teacher_model_path = str(teacher_model or next_teacher_source)
                    self.student_adapter_path = str(next_student_adapter)
                    self.teacher_adapter_path = str(next_teacher_adapter)
                    self.base_model_path = next_base
                    self.student_step = self.student_step if student_step is None else student_step
                    self.teacher_step = self.teacher_step if teacher_step is None else teacher_step
                    self.snapshot_mode = str(snapshot_mode or self.snapshot_mode or "lagged_actor")
                    self.ema_decay = self.ema_decay if ema_decay is None else float(ema_decay)
                    self._student_adapter_name = student_name
                    self._teacher_adapter_name = teacher_name
                    return {"ok": True, "reload": "adapter_in_place", **self.snapshot_state()}
                # If one side loaded successfully, retain correctness by
                # falling through to a full pair reload.  The next full load
                # replaces both models; no mixed pair is reported to callers.

            # Slow path: release the old pair *before* loading to keep peak
            # GPU memory bounded.  Do not keep model references in a backup
            # tuple: doing so would leave the old 2x model pair live while a
            # new pair is constructed (and can OOM a reserved scorer GPU).
            old_meta = (
                self.student_model_path,
                self.teacher_model_path,
                self.student_adapter_path,
                self.teacher_adapter_path,
                self.base_model_path,
                self._student_adapter_name,
                self._teacher_adapter_name,
            )
            old_student = self.student
            old_teacher = self.teacher
            self.student = None
            self.teacher = None
            self.student_tokenizer = None
            self.teacher_tokenizer = None
            del old_student, old_teacher
            try:
                import gc
                import torch

                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                self._load_pair(
                    student_model=student_model or next_student_source,
                    teacher_model=teacher_model or next_teacher_source,
                    student_adapter=next_student_adapter,
                    teacher_adapter=next_teacher_adapter,
                    base_model=base_model or next_base or None,
                    trust_remote_code=self._trust_remote_code,
                )
            except Exception:
                (
                    self.student_model_path,
                    self.teacher_model_path,
                    self.student_adapter_path,
                    self.teacher_adapter_path,
                    self.base_model_path,
                    self._student_adapter_name,
                    self._teacher_adapter_name,
                ) = old_meta
                # The previous model objects were deliberately destroyed to
                # bound memory.  Leave the service unavailable on a failed
                # full reload; the holder/HTTP layer reports the error and
                # rollouts fail closed instead of pretending the stale pair
                # is still active.
                self.student = None
                self.teacher = None
                raise
            self.student_step = self.student_step if student_step is None else student_step
            self.teacher_step = self.teacher_step if teacher_step is None else teacher_step
            self.snapshot_mode = str(snapshot_mode or self.snapshot_mode or "lagged_actor")
            self.ema_decay = self.ema_decay if ema_decay is None else float(ema_decay)
            return {"ok": True, "reload": "pair", **self.snapshot_state()}

    @staticmethod
    def _error_text(value: Any) -> bool:
        text = str(value or "").strip().lower()
        if not text:
            return False
        if text in {"error", "api_error", "cache_miss", "local_cache_miss"}:
            return True
        return any(
            marker in text
            for marker in (
                "local_cache_miss",
                "local_vlm_error",
                "cache_miss",
                "vision llm call failed",
                "grounding api call failed",
                "api_error",
            )
        )

    @staticmethod
    def _eligible(
        record: Any,
        response_ids: list[int],
        response_mask: list[int],
        *,
        reject_existing: bool = True,
    ) -> bool:
        """Validate one rollout transition for paired scoring.

        ``reject_existing`` is true for the ordinary four-forward
        observation-gap scorer: a static/online hit must not be overwritten.
        The privileged OPSD arm uses the same span/evidence checks but may
        run independently of an existing ``e_obs`` cache entry, hence it
        passes ``reject_existing=False``.  Keeping this switch here avoids a
        second, subtly different eligibility implementation in the server.
        """
        if not isinstance(record, dict):
            return False
        if str(record.get("action_type") or "").strip().lower() not in _ACTION_TYPES:
            return False
        if reject_existing and (record.get("ot_cache_hit") or record.get("ot_online_hit")):
            return False
        if record.get("evidence_valid") is not True and str(record.get("evidence_valid")).lower() not in {
            "1",
            "true",
            "yes",
        }:
            return False
        if str(record.get("tool_status") or "").strip().lower() not in {"cache_hit", "api_ok"}:
            return False
        if OnlinePairedScorer._error_text(record.get("observation_text")):
            return False
        try:
            a0 = int(record["assistant_response_start"])
            a1 = int(record["assistant_response_end"])
            o0 = int(record["observation_response_start"])
            o1 = int(record["observation_response_end"])
            t0 = int(record["next_action_response_start"])
            t1 = int(record["next_action_response_end"])
        except (KeyError, TypeError, ValueError):
            return False
        if not (0 <= a0 < a1 <= o0 < o1 <= t0 < t1 <= len(response_ids)):
            return False
        if len(response_mask) != len(response_ids):
            return False
        # Tool observations must be masked out; source and next action tokens
        # must remain policy tokens.  This is the same contract as strict
        # static-cache attachment.
        try:
            def mask_is(values: Sequence[int], expected: int) -> bool:
                for value in values:
                    numeric = float(value)
                    if not math.isfinite(numeric) or numeric != float(int(numeric)) or int(numeric) != expected:
                        return False
                return True

            if not mask_is(response_mask[o0:o1], 0):
                return False
            if not mask_is(response_mask[a0:a1], 1):
                return False
            if not mask_is(response_mask[t0:t1], 1):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        return True

    def score_records(
        self,
        *,
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        response_mask: Sequence[int],
        records: Sequence[dict[str, Any]],
        reduction: str = "mean",
        batch_size: int = 2,
        max_records: int = 0,
    ) -> tuple[int, int, str]:
        """Attach online effects and return ``(hits, skipped, status)``."""

        prompt = [int(value) for value in prompt_ids]
        response = [int(value) for value in response_ids]
        mask = [int(value) for value in response_mask]
        eligible: list[tuple[int, list[int], list[int], list[int]]] = []
        skipped = 0
        for index, record in enumerate(records):
            if not self._eligible(record, response, mask):
                continue
            if max_records > 0 and len(eligible) >= max_records:
                skipped += 1
                continue
            o0 = int(record["observation_response_start"])
            o1 = int(record["observation_response_end"])
            t0 = int(record["next_action_response_start"])
            t1 = int(record["next_action_response_end"])
            minus = prompt + response[:o0]
            if self.null_mode == "matched_mask":
                minus += _matched_null_tokens(self.teacher_tokenizer, o1 - o0)
            plus = prompt + response[:o1]
            eligible.append((index, minus, plus, response[t0:t1]))

        if not eligible:
            return 0, skipped, "no_eligible_records"
        batch_size = max(1, int(batch_size))
        hits = 0
        try:
            with self._forward_lock:
                for start in range(0, len(eligible), batch_size):
                    chunk = eligible[start : start + batch_size]
                    minus_contexts = [item[1] for item in chunk]
                    plus_contexts = [item[2] for item in chunk]
                    targets = [item[3] for item in chunk]
                    student_minus = _score_batch(
                        self.student,
                        minus_contexts,
                        targets,
                        device=self.device,
                        pad_id=self.pad_id,
                    )
                    student_plus = _score_batch(
                        self.student,
                        plus_contexts,
                        targets,
                        device=self.device,
                        pad_id=self.pad_id,
                    )
                    teacher_minus = _score_batch(
                        self.teacher,
                        minus_contexts,
                        targets,
                        device=self.device,
                        pad_id=self.pad_id,
                    )
                    teacher_plus = _score_batch(
                        self.teacher,
                        plus_contexts,
                        targets,
                        device=self.device,
                        pad_id=self.pad_id,
                    )
                    for local, (index, _, _, _) in enumerate(chunk):
                        d_minus_vec = [
                            float(teacher_minus[local][j] - student_minus[local][j])
                            for j in range(len(targets[local]))
                        ]
                        d_plus_vec = [
                            float(teacher_plus[local][j] - student_plus[local][j])
                            for j in range(len(targets[local]))
                        ]
                        d_minus = _reduce(d_minus_vec, reduction)
                        d_plus = _reduce(d_plus_vec, reduction)
                        # Canonical ot-opd-paired-v2 orientation: the signed
                        # change in the teacher--student gap after observing
                        # the tool result.
                        effect = float(d_plus - d_minus)
                        if not math.isfinite(effect):
                            raise ValueError("non-finite online observation effect")
                        record = records[index]
                        record["D_plus"] = d_plus
                        record["D_minus"] = d_minus
                        record["e_obs"] = effect
                        record["e_obs_orientation"] = "plus_minus"
                        record["ot_null_mode"] = self.null_mode
                        record["ot_null_span_tokens"] = int(record["observation_response_end"]) - int(record["observation_response_start"])
                        # Preserve the historical acquisition diagnostic under
                        # an explicit name; formal OPSD shaping consumes
                        # ``e_obs`` on the next-action span instead.
                        record["e_gap_shift"] = effect
                        record["e_gap_shift_orientation"] = "plus_minus"
                        record["ot_online_hit"] = True
                        record["ot_source"] = "online"
                        record["ot_online_status"] = "scored"
                        record["ot_student_model"] = self.student_model_path
                        record["ot_teacher_model"] = self.teacher_model_path
                        record["ot_student_adapter"] = self.student_adapter_path or None
                        record["ot_teacher_adapter"] = self.teacher_adapter_path or None
                        record["ot_student_step"] = self.student_step
                        record["ot_teacher_step"] = self.teacher_step
                        record["ot_snapshot_mode"] = self.snapshot_mode
                        record["ot_ema_decay"] = self.ema_decay
                        record["ot_tokenizer_fingerprint"] = self.tokenizer_fingerprint
                        record["ot_reduction"] = reduction
                        hits += 1
        except Exception as exc:
            # Remove any partial writes so a failed batch cannot leave a
            # seemingly valid prefix of the online cache.
            for index, _, _, _ in eligible:
                record = records[index]
                if record.get("ot_source") == "online":
                    record.pop("D_plus", None)
                    record.pop("D_minus", None)
                    record.pop("e_obs", None)
                    record.pop("e_obs_orientation", None)
                    record.pop("ot_online_hit", None)
                    record.pop("ot_source", None)
                    record.pop("ot_student_model", None)
                    record.pop("ot_teacher_model", None)
                    record.pop("ot_tokenizer_fingerprint", None)
                    record.pop("ot_reduction", None)
                    record["ot_online_status"] = "error"
            logger.exception("online paired scorer failed")
            return 0, len(eligible) + skipped, f"error:{type(exc).__name__}"
        return hits, skipped, "scored"


__all__ = ["OnlinePairedScorer"]
