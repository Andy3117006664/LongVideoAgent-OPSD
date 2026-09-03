import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


# Keep this set in sync with ``videoagent/verl_ext/ot_opd.py``.  Only an
# acquisition action can receive an observation-effect label; answer and
# malformed turns are never cache targets.
_OT_ACTION_TYPES = frozenset(
    {
        "request_grounding",
        "visual_query",
        "search",
        "seek",
        "query",
        "acquire",
    }
)


@register("longvideoagent_multiturn")
class LongVideoAgentLoop(AgentLoopBase):
    """LongVideoAgent multi-turn rollout implemented on verl_new AgentLoop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.response_length = int(self.rollout_config.response_length)
        self.max_tool_response_length = int(self.rollout_config.multi_turn.max_tool_response_length)

        custom_cfg = self.rollout_config.get("custom", {}) or {}
        video_cfg = custom_cfg.get("videoagent", {}) if hasattr(custom_cfg, "get") else {}

        def cfg(name: str, default: Any) -> Any:
            if hasattr(video_cfg, "get"):
                return video_cfg.get(name, default)
            return default

        def cfg_bool(name: str, default: bool = False) -> bool:
            value = cfg(name, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        # Offline mode is intended for deterministic smoke/canary runs when
        # video frames or external API credentials are not available.  It
        # substitutes subtitle-backed observations and never calls a tool API.
        self.offline_mock = cfg_bool("offline_mock", False)
        # ``offline_cache`` is the scientific local-tools mode: grounding and
        # vision responses are looked up from versioned local JSON caches and a
        # cache miss is fail-closed (no hidden HTTP fallback).
        self.offline_cache = cfg_bool("offline_cache", False)

        max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        if max_assistant_turns is None:
            max_assistant_turns = cfg("max_turns", 1)
        self.max_assistant_turns = int(max_assistant_turns)

        max_user_turns = self.rollout_config.multi_turn.max_user_turns
        if max_user_turns is None:
            max_user_turns = self.max_assistant_turns
        self.max_user_turns = int(max_user_turns)

        max_obs_from_data = self.config.data.get("max_obs_length", self.response_length)
        self.max_obs_length = int(cfg("max_obs_length", max_obs_from_data))

        self.base_frame_dir = str(cfg("base_frame_dir", "../bbt_frames"))
        self.subs_path = str(cfg("subs_path", "../Tvqa_data/all_episodes_subtitles_by_clips.json"))
        self.bbox_json_path = str(cfg("bbox_json_path", "bbox_annotations.json"))
        self.observation_cache_path = str(cfg("observation_cache_path", ""))
        self.grounding_cache_path = str(cfg("grounding_cache_path", ""))
        self.ot_cache_path = str(cfg("ot_cache_path", ""))
        self.ot_cache_strict = cfg_bool("ot_cache_strict", True)
        # New caches carry an explicit sign tag.  Keep legacy caches usable by
        # default (they historically followed the builder's plus-minus
        # convention), while allowing a pilot to require the tag after
        # rebuilding all records.
        self.ot_cache_require_orientation = cfg_bool("ot_cache_require_orientation", False)
        # Hash-index replay is an explicit opt-in for asynchronous rollout
        # workers whose rollout_n labels can be permuted between runs.  It
        # still requires stable row/action/video identities plus exact
        # response/observation/target hashes; only the bookkeeping rollout_n
        # field is ignored when selecting a cache entry.
        self.ot_cache_hash_lookup = cfg_bool("ot_cache_hash_lookup", False)
        # Optional in-process paired scorer. It is disabled by default; when
        # enabled, each worker lazily holds one local teacher/student pair and
        # scores the actual sampled prefixes before returning the rollout.
        self.ot_online_score = cfg_bool("ot_online_score", False)
        self.ot_online_student_model = str(cfg("ot_online_student_model", ""))
        self.ot_online_teacher_model = str(cfg("ot_online_teacher_model", ""))
        # LoRA snapshots are the lightweight formal OT-OPSD path.  When set,
        # the HTTP scorer keeps the shared base model on its reserved GPU and
        # swaps only these tiny adapter directories at trainer refreshes.
        self.ot_online_base_model = str(cfg("ot_online_base_model", ""))
        self.ot_online_student_adapter = str(cfg("ot_online_student_adapter", ""))
        self.ot_online_teacher_adapter = str(cfg("ot_online_teacher_adapter", ""))
        self.ot_online_device = str(cfg("ot_online_device", "cuda:0"))
        self.ot_online_dtype = str(cfg("ot_online_dtype", "auto"))
        self.ot_online_batch_size = int(cfg("ot_online_batch_size", 2))
        self.ot_online_max_records = int(cfg("ot_online_max_records", 0))
        self.ot_online_reduction = str(cfg("ot_online_reduction", "mean"))
        self.ot_online_privileged = cfg_bool("ot_online_privileged", False)
        self.ot_online_standard_gap = cfg_bool(
            "ot_online_standard_gap", not self.ot_online_privileged
        )
        # Optional localhost scorer.  Rollout workers are commonly CPU Ray
        # actors, so a reserved-GPU HTTP service is safer than loading a
        # second model pair in every worker.  Empty keeps the in-process
        # scorer path (also opt-in) unchanged.
        self.ot_online_url = str(cfg("ot_online_url", ""))
        self.ot_online_timeout = float(cfg("ot_online_timeout", 120.0))
        self.grounding_cache_strict = cfg_bool("grounding_cache_strict", True)
        # A strict observation cache must not silently substitute a generic
        # clip description for a query-conditioned result.  Keep the option
        # separate so a generic observation-bank experiment can opt in
        # explicitly and report that protocol.
        self.observation_cache_strict = cfg_bool("observation_cache_strict", self.offline_cache)
        # A one-prompt-per-clip observation bank is useful as a cheap
        # plumbing control, but it is not query-conditioned evidence. Strict
        # replay rejects such entries unless the generic-bank ablation is
        # explicitly enabled.
        self.observation_cache_allow_generic = cfg_bool(
            "observation_cache_allow_generic", not self.observation_cache_strict
        )

        self.vision_model = str(cfg("vision_model", "gpt-4o"))
        self.grounding_model = str(cfg("grounding_model", "grok-4-fast-reasoning"))
        self.grounding_temperature = float(cfg("grounding_temperature", 0.6))
        self.grounding_max_tokens = int(cfg("grounding_max_tokens", 512))

        self.frame_start = int(cfg("frame_start", 1))
        self.frame_end = int(cfg("frame_end", 180))
        self.frame_step = int(cfg("frame_step", 15))

        shared_api_key = cfg("api_key", None)
        grounding_api = cfg("grounding_api", shared_api_key or os.getenv("qdd_api"))
        vision_api = cfg("vision_api", shared_api_key or os.getenv("aliyun_api"))

        grounding_base_url = str(cfg("grounding_base_url", "https://api2.aigcbest.top/v1"))
        vision_base_url = str(cfg("vision_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

        self.observation_cache = self._load_json_safely(self.observation_cache_path)
        self.grounding_cache = self._load_json_safely(self.grounding_cache_path)
        self.ot_cache = self._load_json_safely(self.ot_cache_path)

        if self.offline_mock or self.offline_cache:
            self.grounding_client = None
            self.vision_client = None
        else:
            try:
                from openai import OpenAI
            except Exception as exc:
                raise ImportError("LongVideoAgentLoop requires `openai` package, please install it.") from exc

            self.grounding_client = OpenAI(api_key=grounding_api, base_url=grounding_base_url)
            self.vision_client = OpenAI(api_key=vision_api, base_url=vision_base_url)

        self.clip_subtitles = self._load_json_safely(self.subs_path)
        self.bbox_cache = self._load_json_safely(self.bbox_json_path)

    @staticmethod
    def _load_json_safely(path: str) -> dict:
        if not path:
            return {}
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("JSON file not found: %s", path)
            return {}
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.exception("Failed to load json: %s", path)
            return {}

    @staticmethod
    def _extract_episode_prefix(clip_name: str) -> str:
        if not clip_name:
            return ""
        # LongTVQA uses both six-character BBT IDs (``s05e06``) and
        # show-prefixed IDs (e.g. ``friends_s09e14_seg02_clip_20``).  Splitting
        # at the segment marker preserves the full episode key; the six-char
        # fallback is retained for legacy clip names without ``_seg``.
        if "_seg" in clip_name:
            return clip_name.split("_seg", 1)[0]
        return clip_name[:6]

    def _build_subtitles_for_episode(self, episode_prefix: str) -> str:
        if not episode_prefix:
            return ""
        matched = {k: v for k, v in self.clip_subtitles.items() if str(k).startswith(episode_prefix)}
        if not matched:
            return ""
        parts = [f"<{clip_key}>{subtitle}</{clip_key}>" for clip_key, subtitle in sorted(matched.items())]
        return "\n".join(parts)

    def _get_clip_subtitle(self, clip_name: str) -> str:
        if not clip_name:
            return ""
        return str(self.clip_subtitles.get(clip_name, ""))

    def _get_bbox_content(self, vid: str) -> str:
        if not vid:
            return "{}"
        return json.dumps(self.bbox_cache.get(vid, {}))

    @staticmethod
    def _normalize_cache_key(text: Any) -> str:
        return " ".join(str(text or "").strip().split()).lower()

    @staticmethod
    def _is_tool_error_text(text: Any) -> bool:
        """Recognize sentinel failures without rejecting normal prose.

        Cache builders use bracketed sentinels; matching those prefixes (and
        exact short status words) avoids treating a legitimate description
        containing a word such as ``error`` as failed evidence.
        """
        value = str(text or "").strip().lower()
        if not value:
            return False
        if value in {"error", "api_error", "cache_miss", "local_cache_miss"}:
            return True
        # These are unambiguous sentinel phrases and may be wrapped by an
        # observation protocol tag such as ``<information>...</information>``.
        return any(
            marker in value
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
    def _cache_payload(cache: dict, section: str) -> dict:
        value = cache.get(section, cache) if isinstance(cache, dict) else {}
        return value if isinstance(value, dict) else {}

    def _lookup_grounding_cache(self, question_data: dict[str, str]) -> str:
        """Look up a qid/UID keyed local grounding result, then question text."""

        cache = self.grounding_cache
        if not isinstance(cache, dict) or not cache:
            return ""
        if self.grounding_cache_strict and not (
            str(question_data.get("_sample_uid") or "").strip()
            or str(question_data.get("_qid") or "").strip()
        ):
            return ""
        by_uid = self._cache_payload(cache, "by_uid")
        by_qid = self._cache_payload(cache, "by_qid")
        by_question = self._cache_payload(cache, "by_question")
        for key in (question_data.get("_sample_uid"), question_data.get("_qid")):
            if key is None or str(key) not in by_uid and str(key) not in by_qid:
                continue
            value = by_uid.get(str(key), by_qid.get(str(key), ""))
            if isinstance(value, dict):
                value = value.get("clip") or value.get("predicted_clip") or ""
            if value:
                return str(value).strip()
        qkey = self._normalize_cache_key(question_data.get("q", ""))
        if self.grounding_cache_strict:
            return ""
        value = by_question.get(qkey, cache.get(qkey, ""))
        if isinstance(value, dict):
            value = value.get("clip") or value.get("predicted_clip") or ""
        return str(value or "").strip()

    def _lookup_observation_cache(self, query: str, vid: str) -> str:
        """Return a deterministic cached visual description for ``vid``."""

        cache = self.observation_cache
        if not isinstance(cache, dict) or not cache:
            return ""
        clips = self._cache_payload(cache, "clips")
        entry = clips.get(str(vid), cache.get(str(vid), ""))
        if isinstance(entry, dict):
            # Prefer a query-specific entry when available, otherwise the
            # generic clip observation generated with the fixed prompt.
            qkey = self._normalize_cache_key(query)
            by_query = entry.get("by_query")
            if isinstance(by_query, dict):
                if qkey in by_query:
                    entry = by_query[qkey]
                elif self.observation_cache_strict:
                    # Do not let a generic description masquerade as the
                    # answer to a different query in a strict replay.
                    return ""
                else:
                    entry = entry.get("observation") or entry.get("text") or entry.get("description") or ""
            else:
                if self.observation_cache_strict and not self.observation_cache_allow_generic:
                    # Generic per-clip descriptions were not conditioned on
                    # this query; accepting them as strict evidence would
                    # invalidate the observation-effect interpretation.
                    return ""
                entry = entry.get("observation") or entry.get("text") or entry.get("description") or ""
            if isinstance(entry, dict):
                entry = entry.get("observation") or entry.get("text") or entry.get("description") or ""
        elif self.observation_cache_strict and not self.observation_cache_allow_generic:
            # Legacy flat ``{clip: text}`` caches are necessarily generic
            # unless they carry an explicit query namespace.
            return ""
        return str(entry or "").strip()

    def _attach_ot_cache(
        self,
        records: list[dict[str, Any]],
        extra_info: dict[str, Any],
        response_ids: Optional[list[int]] = None,
        response_mask: Optional[list[int]] = None,
    ) -> tuple[int, int]:
        """Attach precomputed D+/D− to matching acquisition records.

        This is deliberately an explicit *offline replay* path.  The cache is
        keyed by stable source UID/qid and turn; stale/missing entries are
        counted rather than replaced by a proxy signal.  All counters are
        local to this call (agent-loop workers run many samples concurrently),
        so one trajectory cannot overwrite another trajectory's metrics.

        ``response_ids`` and ``response_mask`` are passed after truncation so
        strict mode can reject stale offsets and targets that contain tool or
        padding tokens.  The older two-argument call remains usable for loose
        mode, but strict mode fails closed when these arrays are unavailable.
        """

        hits = 0
        misses = 0
        if not isinstance(records, list):
            return hits, misses

        def _is_acquisition(record: Any) -> bool:
            return bool(
                isinstance(record, dict)
                and str(record.get("action_type") or "").strip().lower() in _OT_ACTION_TYPES
            )

        def _clear_attachment(record: Any) -> None:
            """Remove any signal left by an earlier replay attempt."""
            if not _is_acquisition(record):
                return
            record["ot_cache_hit"] = False
            record.pop("ot_cache_key", None)
            record.pop("D_plus", None)
            record.pop("D_minus", None)
            record.pop("e_obs", None)
            record.pop("e_obs_orientation", None)

        cache = self.ot_cache
        if not isinstance(cache, dict) or not cache:
            # Report coverage even when the cache file is absent.  Returning
            # (0, 0) would make a completely unpaired run look as if it had
            # no acquisition turns to evaluate.
            misses = sum(
                int(_is_acquisition(record))
                for record in records
            )
            for record in records:
                _clear_attachment(record)
            return hits, misses
        entries = cache.get("records", cache)
        if not isinstance(entries, dict):
            misses = sum(
                int(_is_acquisition(record))
                for record in records
            )
            for record in records:
                _clear_attachment(record)
            return hits, misses

        # The identity in ``extra_info`` is the source row identity.  Do not
        # use ``str(None)`` as a real qid: it would create collisions across
        # rows whose qid is absent.
        uid = str(extra_info.get("sample_uid") or extra_info.get("uid") or "").strip()
        qid_value = extra_info.get("qid")
        qid = "" if qid_value is None else str(qid_value).strip()
        rollout_n = str(extra_info.get("rollout_n", "0")).strip()
        response_len = len(response_ids) if response_ids is not None else None

        def miss() -> None:
            nonlocal misses
            misses += 1

        def _payload_identity_matches(
            record: dict[str, Any], payload: dict[str, Any], turn: str, *, ignore_rollout_n: bool = False
        ) -> bool:
            """Validate cache identity/provenance fields in strict mode."""

            if not self.ot_cache_strict:
                return True
            # Strict cache records are expected to carry all of these fields.
            # Requiring them avoids accepting a legacy question-only cache
            # whose values happen to share a key with this trajectory.
            record_uid = str(record.get("sample_uid") or record.get("uid") or uid).strip()
            record_qid = record.get("qid")
            if record_qid is None:
                record_qid = qid
            record_qid = "" if record_qid is None else str(record_qid).strip()
            record_rollout_n = str(record.get("rollout_n", rollout_n)).strip()
            # The outer row identity and the per-turn copy must agree.  A
            # mismatching turn record is stale even if a qid-only key happens
            # to resolve to a payload.
            if uid and record_uid and record_uid != uid:
                return False
            if qid and record_qid and record_qid != qid:
                return False
            if not ignore_rollout_n and record_rollout_n != rollout_n:
                return False
            expected = {
                "sample_uid": record_uid,
                "qid": record_qid,
                "turn": turn,
                "rollout_n": record_rollout_n,
                "action_type": str(record.get("action_type") or "").strip().lower(),
            }
            # At least one stable row identity is mandatory; an empty UID/qid
            # cannot be made unique by a turn number alone.
            if not expected["sample_uid"] and not expected["qid"]:
                return False
            for key, value in expected.items():
                if key == "rollout_n" and ignore_rollout_n:
                    continue
                # A stable identity is required, but qid is legitimately
                # absent in some LongTVQA rows when sample_uid is present.
                # In that case both sides may omit qid; do not turn the
                # empty placeholder into a cross-row identity.
                if key == "qid" and not value and key not in payload:
                    continue
                if key not in payload:
                    return False
                actual = str(payload.get(key) if payload.get(key) is not None else "").strip()
                if key == "action_type":
                    matches = actual.lower() == value.lower()
                else:
                    # IDs and clip names are opaque, case-sensitive tokens;
                    # accepting a case-folded match can pair two videos or
                    # rows that only differ by naming convention.
                    matches = actual == value
                if not matches:
                    return False
            expected_vid = str(record.get("current_vid_before") or record.get("current_vid") or "").strip()
            if not expected_vid:
                return False
            payload_vid = payload.get("current_vid")
            if payload_vid is None:
                payload_vid = payload.get("current_vid_before")
            if "current_vid" not in payload and "current_vid_before" not in payload:
                return False
            if str(payload_vid if payload_vid is not None else "").strip() != expected_vid:
                return False
            return True

        for record in records:
            if not isinstance(record, dict):
                continue
            action_type = str(record.get("action_type") or "").strip().lower()
            # Non-acquisition turns are not cache targets and should not count
            # as misses in coverage metrics.
            if action_type not in _OT_ACTION_TYPES:
                continue
            # Clear stale attachment state when a record is replayed more than
            # once; only a fully validated entry below may set it to true.
            _clear_attachment(record)

            turn = str(record.get("turn", "")).strip()
            if self.ot_cache_strict and not turn:
                miss()
                continue
            exact_candidates = [
                f"{uid}|{rollout_n}|{turn}" if uid else "",
                f"{qid}|{rollout_n}|{turn}" if qid else "",
            ]
            candidates = exact_candidates if self.ot_cache_strict else [
                *exact_candidates,
                f"{uid}|{turn}" if uid else "",
                f"{qid}|{turn}" if qid else "",
                uid,
                qid,
            ]
            payload = None
            matched_key = ""
            for key in candidates:
                if key and key in entries:
                    payload = entries[key]
                    matched_key = key
                    break
            # Optional strict hash-index fallback.  Async workers may assign
            # rollout_n differently across otherwise identical runs.  A
            # response/observation/target hash match is stronger than that
            # bookkeeping label, so permit the alias only when explicitly
            # enabled and retain all other identity/provenance checks below.
            hash_alias = False
            selected_hashes_match = isinstance(payload, dict) and all(
                key in payload and str(payload.get(key)) == str(record.get(key, ""))
                for key in ("response_sha256", "obs_sha256", "target_sha256")
            )
            if self.ot_cache_hash_lookup and not selected_hashes_match and response_ids is not None:
                # Do not let a stale exact rollout_n key block the stronger
                # hash-index lookup.  This occurs when async scheduling
                # reuses/permutates rollout_n labels across repeated runs.
                payload = None
                matched_key = ""
                try:
                    o0 = int(record.get("observation_response_start"))
                    o1 = int(record.get("observation_response_end"))
                    t0 = int(record.get("next_action_response_start"))
                    t1 = int(record.get("next_action_response_end"))
                    a0 = int(record.get("assistant_response_start"))
                    a1 = int(record.get("assistant_response_end"))
                    if 0 <= a0 < a1 <= o0 < o1 <= t0 < t1 <= len(response_ids):
                        response_hash = hashlib.sha256(
                            json.dumps(response_ids, separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
                        obs_hash = hashlib.sha256(
                            json.dumps(response_ids[o0:o1], separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
                        target_hash = hashlib.sha256(
                            json.dumps(response_ids[t0:t1], separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
                        action = str(record.get("action_type") or "").strip().lower()
                        expected_uid = uid
                        expected_qid = qid
                        expected_vid = str(record.get("current_vid_before") or record.get("current_vid") or "").strip()
                        for alias_key, candidate in entries.items():
                            if not isinstance(candidate, dict):
                                continue
                            if str(candidate.get("sample_uid") or "").strip() != expected_uid:
                                continue
                            if expected_qid and str(candidate.get("qid") or "").strip() != expected_qid:
                                continue
                            if str(candidate.get("turn") or "").strip() != turn:
                                continue
                            if str(candidate.get("action_type") or "").strip().lower() != action:
                                continue
                            if str(candidate.get("current_vid") or candidate.get("current_vid_before") or "").strip() != expected_vid:
                                continue
                            if candidate.get("response_sha256") != response_hash:
                                continue
                            if candidate.get("obs_sha256") != obs_hash or candidate.get("target_sha256") != target_hash:
                                continue
                            payload = candidate
                            matched_key = str(alias_key)
                            hash_alias = True
                            break
                except (TypeError, ValueError, OverflowError):
                    payload = None
            if not isinstance(payload, dict):
                miss()
                continue

            # A cache made for a different sampled response is stale.  The
            # rollout loop records hashes after truncation; require all hashes
            # present in the payload to agree before copying a signal.
            hash_keys = ("response_sha256", "obs_sha256", "target_sha256")
            if self.ot_cache_strict and any(key not in payload or not record.get(key) for key in hash_keys):
                miss()
                continue
            if any(key in payload and str(payload[key]) != str(record.get(key, "")) for key in hash_keys):
                miss()
                continue
            if not _payload_identity_matches(record, payload, turn, ignore_rollout_n=hash_alias):
                miss()
                continue

            # An evidence generated by a cache miss/error must never become a
            # valid paired observation merely because its span/hash exists.
            statuses = {
                str(value).strip().lower()
                for value in (record.get("tool_status"), payload.get("tool_status"))
                if value is not None and str(value).strip()
            }
            cache_flags = [
                value
                for value in (
                    record.get("cache_hit"),
                    record.get("tool_cache_hit"),
                    payload.get("cache_hit"),
                    payload.get("tool_cache_hit"),
                )
                if value is not None
            ]
            evidence_values = [
                value
                for value in (record.get("evidence_valid"), payload.get("evidence_valid"))
                if value is not None
            ]
            bad_markers = (
                "local_cache_miss",
                "local_vlm_error",
                "vision_llm_call_failed",
                "grounding_api_call_failed",
                "api_error",
                "cache_miss",
                "no_frames",
                "error",
            )
            if self.ot_cache_strict and (
                not statuses
                or not evidence_values
                or any(
                    value is not True and str(value).strip().lower() not in {"1", "true", "yes"}
                    for value in evidence_values
                )
            ):
                miss()
                continue
            if self.ot_cache_strict and len(statuses) > 1:
                # Both the trajectory and cache may carry provenance.  A
                # disagreement (e.g. `cache_hit` vs `api_ok`) is stale data,
                # even though each status is individually admissible.
                miss()
                continue
            if self.ot_cache_strict and not statuses.issubset({"cache_hit", "api_ok"}):
                # Mock/proxy output is useful for smoke tests but must not be
                # treated as evidence for a paired OT label.
                miss()
                continue
            if any(marker in candidate for candidate in statuses for marker in bad_markers):
                miss()
                continue
            if self.ot_cache_strict and cache_flags:
                normalized_flags = {
                    str(value).strip().lower() in {"1", "true", "yes", "y"} for value in cache_flags
                }
                if len(normalized_flags) > 1:
                    miss()
                    continue
                cache_flag = next(iter(normalized_flags))
                if "cache_hit" in statuses and not cache_flag:
                    miss()
                    continue
                if cache_flag and any(marker in candidate for candidate in statuses for marker in bad_markers):
                    miss()
                    continue
            observation_texts = {
                str(value).strip().lower()
                for value in (record.get("observation_text"), payload.get("observation_text"))
                if value is not None and str(value).strip()
            }
            if self.ot_cache_strict and not str(record.get("observation_text") or "").strip():
                miss()
                continue
            if self.ot_cache_strict and any(self._is_tool_error_text(candidate) for candidate in observation_texts):
                miss()
                continue

            try:
                o0 = int(record.get("observation_response_start"))
                o1 = int(record.get("observation_response_end"))
                t0 = int(record.get("next_action_response_start"))
                t1 = int(record.get("next_action_response_end"))
                a0 = int(record.get("assistant_response_start"))
                a1 = int(record.get("assistant_response_end"))
                # If the cache stores explicit offsets, they must describe
                # exactly the same trajectory spans.  Hash equality alone is
                # insufficient when repeated tokens make two offsets look
                # identical.
                for key, expected_span in (
                    ("obs_start", o0),
                    ("obs_end", o1),
                    ("target_start", t0),
                    ("target_end", t1),
                    ("assistant_start", a0),
                    ("assistant_end", a1),
                ):
                    if key in payload and int(payload[key]) != expected_span:
                        raise ValueError(f"cache span mismatch for {key}")
                if response_len is None and self.ot_cache_strict:
                    raise ValueError("response_ids unavailable in strict cache attach")
                upper = response_len if response_len is not None else max(a1, o1, t1)
                if not (0 <= a0 < a1 <= o0 < o1 <= t0 < t1 <= upper):
                    raise ValueError("invalid response-relative span")
                if response_mask is not None:
                    if response_len is not None and len(response_mask) != response_len:
                        raise ValueError("response mask and response ids have different lengths")
                    if len(response_mask) < t1 or len(response_mask) < a1:
                        raise ValueError("response mask shorter than target/action span")
                    def _mask_is(values: Sequence[int], expected: int) -> bool:
                        for value in values:
                            try:
                                numeric = float(value)
                            except (TypeError, ValueError, OverflowError):
                                return False
                            if not math.isfinite(numeric) or numeric != float(int(numeric)) or int(numeric) != expected:
                                return False
                        return True

                    if not _mask_is(response_mask[o0:o1], 0):
                        raise ValueError("observation span contains policy tokens")
                    if not _mask_is(response_mask[t0:t1], 1):
                        raise ValueError("target span contains tool/padding tokens")
                    if not _mask_is(response_mask[a0:a1], 1):
                        raise ValueError("source action span contains tool/padding tokens")
                elif self.ot_cache_strict:
                    raise ValueError("response_mask unavailable in strict cache attach")
            except (TypeError, ValueError, OverflowError):
                miss()
                continue

            def _finite_scalar(value: Any) -> float | None:
                # The strict cache format stores already-reduced scalar
                # means.  Accept a one-element list for JSON convenience but
                # reject vectors here rather than accidentally summing token
                # lengths (the hook itself has a separate vector path).
                if isinstance(value, (list, tuple)):
                    if len(value) != 1:
                        return None
                    value = value[0]
                try:
                    result = float(value)
                except (TypeError, ValueError):
                    return None
                return result if math.isfinite(result) else None

            # Canonical paired-cache orientation (ot-opd-paired-v2):
            # D_plus is the teacher--student gap after the observation and
            # D_minus is the gap before it.  Keep e_obs as the signed change
            # in that gap; never silently negate an explicitly incompatible
            # cache because its provenance would become unauditable.
            orientation = payload.get(
                "e_obs_orientation",
                payload.get("orientation", cache.get("e_obs_orientation")),
            )
            if self.ot_cache_strict and getattr(self, "ot_cache_require_orientation", False) and orientation is None:
                miss()
                continue
            if orientation is not None:
                normalized_orientation = str(orientation).strip().lower().replace("-", "_")
                if normalized_orientation not in {"plus_minus", "plusminus", "dplus_minus_dminus"}:
                    miss()
                    continue
            d_plus = _finite_scalar(payload.get("D_plus", payload.get("d_plus")))
            d_minus = _finite_scalar(payload.get("D_minus", payload.get("d_minus")))
            e_obs = _finite_scalar(payload.get("e_obs"))
            if d_plus is not None and d_minus is not None:
                computed = d_plus - d_minus
                if e_obs is not None and not math.isclose(computed, e_obs, rel_tol=1e-4, abs_tol=1e-5):
                    miss()
                    continue
                record["D_plus"] = d_plus
                record["D_minus"] = d_minus
                record["e_obs"] = computed
                record["e_obs_orientation"] = "plus_minus"
            elif e_obs is not None:
                # A scalar without D+/D− cannot be re-derived.  Retain
                # backwards compatibility for legacy caches by assuming the
                # historical builder orientation, while tagging the record
                # so downstream hooks never have to guess.
                record["e_obs"] = e_obs
                record["e_obs_orientation"] = "plus_minus"
            else:
                miss()
                continue

            record["ot_cache_key"] = matched_key
            record["ot_cache_hit"] = True
            hits += 1

        return hits, misses

    def _attach_online_ot_scores(
        self,
        records: list[dict[str, Any]],
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        response_mask: Sequence[int],
    ) -> tuple[int, int, str]:
        """Score the actual rollout with an opt-in local teacher/student pair.

        Static cache attachment is attempted first.  This method only handles
        records that remain unmatched, so stochastic rollout text does not
        silently receive a signal from another trajectory.  The scorer is
        process-global/lazy and therefore does not load models when the flag
        is disabled (the default).
        """

        self._last_online_privileged_hits = 0
        if not getattr(self, "ot_online_score", False):
            return 0, 0, "disabled"
        # The ordinary e_obs arm only needs transitions not already covered
        # by a strict static/online cache.  The privileged OPSD arm is
        # independent, so when enabled it also receives evidence-valid rows
        # that already have e_obs; the server will run the same span checks
        # and attach e_priv without replacing the existing diagnostic.
        targets = []
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("action_type") or "").strip().lower() not in _OT_ACTION_TYPES:
                continue
            need_standard = not bool(record.get("ot_cache_hit")) and not bool(record.get("ot_online_hit"))
            need_privileged = bool(getattr(self, "ot_online_privileged", False)) and not bool(
                record.get("ot_privileged_hit")
            )
            if need_standard or need_privileged:
                targets.append(record)
        if not targets:
            return 0, 0, "no_targets"
        student_model = str(getattr(self, "ot_online_student_model", ""))
        teacher_model = str(getattr(self, "ot_online_teacher_model", ""))
        scorer_url = str(getattr(self, "ot_online_url", "")).strip()
        if scorer_url:
            try:
                import urllib.request

                # Advertise the single supported wire orientation so an
                # endpoint cannot silently return a negated legacy score.
                for target in targets:
                    target.setdefault("e_obs_orientation", "plus_minus")
                request_payload = {
                    "prompt_ids": [int(x) for x in prompt_ids],
                    "response_ids": [int(x) for x in response_ids],
                    "response_mask": [int(x) for x in response_mask],
                    "records": targets,
                    "reduction": str(getattr(self, "ot_online_reduction", "mean")),
                    "batch_size": int(getattr(self, "ot_online_batch_size", 1)),
                    "max_records": int(getattr(self, "ot_online_max_records", 0)),
                    "privileged": bool(getattr(self, "ot_online_privileged", False)),
                    "standard_gap": bool(getattr(self, "ot_online_standard_gap", True)),
                }
                endpoint = scorer_url.rstrip("/") + "/score"
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=float(getattr(self, "ot_online_timeout", 120.0))) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict) or payload.get("ok") is not True:
                    raise RuntimeError(f"scorer returned invalid payload: {payload!r}")
                returned = payload.get("records")
                if not isinstance(returned, list) or len(returned) != len(targets):
                    raise RuntimeError("scorer returned a different record count")
                def _finite_remote(value: Any) -> float | None:
                    if isinstance(value, (list, tuple)):
                        if len(value) != 1:
                            return None
                        value = value[0]
                    try:
                        result = float(value)
                    except (TypeError, ValueError, OverflowError):
                        return None
                    return result if math.isfinite(result) else None

                # Validate every remote result before mutating a trajectory.
                # The endpoint is an optional optimization service, not a
                # trusted source of labels; identity, spans, provenance and
                # the D+/D−/e_obs invariant must still hold locally.
                validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
                # A remote scorer must echo the trajectory contract rather
                # than return a sparse score object.  The hashes alone are not
                # sufficient: repeated token spans can occur in different
                # rows, turns, clips, or rollout replicas.
                required_echo = (
                    "turn",
                    "rollout_n",
                    "action_type",
                    "response_sha256",
                    "obs_sha256",
                    "target_sha256",
                    "assistant_response_start",
                    "assistant_response_end",
                    "observation_response_start",
                    "observation_response_end",
                    "next_action_response_start",
                    "next_action_response_end",
                    "tool_status",
                    "evidence_valid",
                    "observation_text",
                    "e_obs_orientation",
                )
                privileged_validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
                for target, scored in zip(targets, returned, strict=True):
                    if not isinstance(scored, dict):
                        continue
                    # Privileged OPSD is independent of the standard
                    # teacher/student gap.  Retain a valid teacher-only
                    # visual gain even when e_obs itself was skipped.
                    priv_hit = scored.get("ot_privileged_hit")
                    priv_hit = priv_hit is True or str(priv_hit).strip().lower() in {"1", "true", "yes", "y"}
                    if priv_hit:
                        priv_source = str(scored.get("ot_privileged_source") or "").strip().lower()
                        priv_orientation = str(scored.get("e_priv_orientation") or "").strip().lower().replace("-", "_")
                        target_evidence = target.get("evidence_valid")
                        priv_evidence_ok = target_evidence is True or str(target_evidence).strip().lower() in {
                            "1",
                            "true",
                            "yes",
                        }
                        target_status = str(target.get("tool_status") or "").strip().lower()
                        priv_evidence_ok = (
                            priv_evidence_ok
                            and target_status in {"cache_hit", "api_ok"}
                            and not self._is_tool_error_text(target.get("observation_text"))
                        )
                        try:
                            priv_value = float(scored.get("e_priv"))
                            pa0 = int(target["assistant_response_start"])
                            pa1 = int(target["assistant_response_end"])
                            po0 = int(target["observation_response_start"])
                            po1 = int(target["observation_response_end"])
                            pt0 = int(target["next_action_response_start"])
                            pt1 = int(target["next_action_response_end"])
                            priv_masks_ok = (
                                0 <= pa0 < pa1 <= po0 < po1 <= pt0 < pt1 <= len(response_ids)
                                and len(response_mask) == len(response_ids)
                                and all(float(value) == 1.0 for value in response_mask[pa0:pa1])
                                and all(float(value) == 0.0 for value in response_mask[po0:po1])
                                and all(float(value) == 1.0 for value in response_mask[pt0:pt1])
                            )
                        except (KeyError, TypeError, ValueError, OverflowError):
                            priv_masks_ok = False
                            priv_value = float("nan")
                        priv_echo_ok = all(
                            key in target
                            and key in scored
                            and str(scored.get(key)) == str(target.get(key))
                            for key in required_echo
                        )
                        target_uid = str(target.get("sample_uid") or "").strip()
                        target_qid = "" if target.get("qid") is None else str(target.get("qid")).strip()
                        uid_ok = (not target_uid or str(scored.get("sample_uid") or "").strip() == target_uid) and (
                            not target_qid
                            or str(scored.get("qid") if scored.get("qid") is not None else "").strip() == target_qid
                        )
                        target_vid = str(target.get("current_vid") or target.get("current_vid_before") or "").strip()
                        scored_vid = str(scored.get("current_vid") or scored.get("current_vid_before") or "").strip()
                        expected_priv_clip = str(
                            target.get("current_vid_after")
                            or target.get("current_vid")
                            or target.get("current_vid_before")
                            or ""
                        ).strip()
                        scored_priv_clip = str(scored.get("ot_privileged_clip") or "").strip()
                        privileged_cache = str(scored.get("ot_privileged_cache") or "").strip()
                        privileged_cache_sha = str(scored.get("ot_privileged_cache_sha256") or "").strip()
                        privileged_z_sha = str(scored.get("ot_privileged_z_sha256") or "").strip()
                        try:
                            privileged_tokens = int(scored.get("ot_privileged_tokens", 0))
                        except (TypeError, ValueError, OverflowError):
                            privileged_tokens = 0
                        target_after = str(target.get("current_vid_after") or "").strip()
                        scored_after = str(scored.get("current_vid_after") or "").strip()
                        if (
                            priv_source == "local_visual_cache"
                            and priv_orientation in {"teacher_priv_minus_deploy", "teacher_privileged_minus_deploy"}
                            and math.isfinite(priv_value)
                            and priv_masks_ok
                            and priv_evidence_ok
                            and priv_echo_ok
                            and uid_ok
                            and bool(target_vid)
                            and scored_vid == target_vid
                            and bool(expected_priv_clip)
                            and scored_priv_clip == expected_priv_clip
                            and bool(privileged_cache)
                            and bool(privileged_cache_sha)
                            and bool(privileged_z_sha)
                            and privileged_tokens > 0
                            and str(scored.get("ot_privileged_context_mode") or "")
                            == "append_after_deploy_observation"
                            and (not target_after or scored_after == target_after)
                        ):
                            privileged_validated.append(
                                (
                                    target,
                                    {
                                        "e_priv": priv_value,
                                        "e_priv_orientation": "teacher_priv_minus_deploy",
                                        "ot_privileged_hit": True,
                                        "ot_privileged_source": priv_source,
                                        **{
                                            key: scored[key]
                                            for key in (
                                                "ot_privileged_cache",
                                                "ot_privileged_cache_sha256",
                                                "ot_privileged_z_sha256",
                                                "ot_privileged_frame_sha256",
                                                "ot_privileged_frame_paths",
                                                "ot_privileged_clip",
                                                "ot_privileged_tokens",
                                                "ot_privileged_context_mode",
                                                "ot_privileged_teacher_step",
                                                "ot_privileged_student_step",
                                                "ot_privileged_snapshot_mode",
                                                "ot_privileged_teacher_model",
                                                "ot_privileged_student_adapter",
                                                "ot_privileged_teacher_adapter",
                                                "ot_privileged_tokenizer_fingerprint",
                                            )
                                            if key in scored
                                        },
                                    },
                                )
                            )
                    # Mirror the local scorer's eligibility contract before
                    # trusting a remote result.  Invalid/mocked/error
                    # observations must remain misses even if the service
                    # returns finite numbers.
                    if target.get("evidence_valid") is not True and str(target.get("evidence_valid")).strip().lower() not in {
                        "1",
                        "true",
                        "yes",
                    }:
                        continue
                    if str(target.get("tool_status") or "").strip().lower() not in {"cache_hit", "api_ok"}:
                        continue
                    if self._is_tool_error_text(target.get("observation_text")):
                        continue
                    try:
                        a0 = int(target["assistant_response_start"])
                        a1 = int(target["assistant_response_end"])
                        o0 = int(target["observation_response_start"])
                        o1 = int(target["observation_response_end"])
                        t0 = int(target["next_action_response_start"])
                        t1 = int(target["next_action_response_end"])
                    except (KeyError, TypeError, ValueError, OverflowError):
                        continue
                    if not (0 <= a0 < a1 <= o0 < o1 <= t0 < t1 <= len(response_ids)):
                        continue
                    if len(response_mask) != len(response_ids):
                        continue
                    try:
                        mask_values = [float(value) for value in response_mask]
                        if any(
                            not math.isfinite(value) or value not in {0.0, 1.0}
                            for value in mask_values
                        ):
                            continue
                        if any(mask_values[index] != 0.0 for index in range(o0, o1)):
                            continue
                        if any(mask_values[index] != 1.0 for index in (*range(a0, a1), *range(t0, t1))):
                            continue
                    except (TypeError, ValueError, OverflowError):
                        continue
                    # A hit must be an explicit boolean/string true and carry
                    # an unambiguous successful provenance status.
                    hit_value = scored.get("ot_online_hit")
                    hit = hit_value is True or str(hit_value).strip().lower() in {"1", "true", "yes", "y"}
                    status = str(scored.get("ot_online_status") or "").strip().lower()
                    source = str(scored.get("ot_source") or "").strip().lower()
                    if not hit or source != "online" or status not in {"scored", "remote_scored"}:
                        continue
                    if self._is_tool_error_text(status) or self._is_tool_error_text(scored.get("observation_text")):
                        continue
                    # Require every span/provenance field and compare it to
                    # the exact local target before accepting any number.  A
                    # missing field is a miss, not an invitation to infer it.
                    if any(
                        key not in target
                        or key not in scored
                        or str(scored.get(key)) != str(target.get(key))
                        for key in required_echo
                    ):
                        continue
                    if any(
                        not str(target.get(key) or "").strip()
                        for key in ("response_sha256", "obs_sha256", "target_sha256")
                    ):
                        continue

                    # Echo every stable row identity that exists locally.  A
                    # qid is optional only when sample_uid is present; if both
                    # are available, both must match.
                    target_uid = str(target.get("sample_uid") or "").strip()
                    target_qid = "" if target.get("qid") is None else str(target.get("qid")).strip()
                    if not target_uid and not target_qid:
                        continue
                    if target_uid:
                        if "sample_uid" not in scored or str(scored.get("sample_uid") or "").strip() != target_uid:
                            continue
                    if target_qid:
                        if "qid" not in scored or str(scored.get("qid") if scored.get("qid") is not None else "").strip() != target_qid:
                            continue

                    # current_vid is part of the cache identity.  Accept the
                    # historical `current_vid_before` alias, but require the
                    # endpoint to echo at least one and reject disagreement.
                    target_vid = str(target.get("current_vid") or target.get("current_vid_before") or "").strip()
                    scored_has_vid = "current_vid" in scored or "current_vid_before" in scored
                    scored_vid = str(scored.get("current_vid") or scored.get("current_vid_before") or "").strip()
                    if not target_vid or not scored_has_vid or scored_vid != target_vid:
                        continue
                    if any(
                        key in target
                        and (
                            key not in scored
                            or str(scored.get(key)).strip().lower()
                            != str(target.get(key)).strip().lower()
                        )
                        for key in ("cache_hit", "tool_cache_hit")
                    ):
                        continue
                    d_plus = _finite_remote(scored.get("D_plus", scored.get("d_plus")))
                    d_minus = _finite_remote(scored.get("D_minus", scored.get("d_minus")))
                    e_obs = _finite_remote(scored.get("e_obs"))
                    if d_plus is None or d_minus is None or e_obs is None:
                        continue
                    orientation = scored.get("e_obs_orientation", scored.get("orientation"))
                    if orientation is not None:
                        normalized_orientation = str(orientation).strip().lower().replace("-", "_")
                        if normalized_orientation not in {"plus_minus", "plusminus", "dplus_minus_dminus"}:
                            continue
                    if not math.isclose(d_plus - d_minus, e_obs, rel_tol=1e-4, abs_tol=1e-5):
                        continue
                    # Require the endpoint to echo the exact rollout hashes
                    # when they are present locally.  This prevents a service
                    # from accidentally pairing a score from another sample.
                    if any(
                        key in target
                        and str(target.get(key) or "")
                        and (key not in scored or str(scored.get(key)) != str(target.get(key)))
                        for key in ("response_sha256", "obs_sha256", "target_sha256")
                    ):
                        continue
                    validated.append(
                        (
                            target,
                            {
                                "D_plus": d_plus,
                                "D_minus": d_minus,
                                "e_obs": e_obs,
                                "e_obs_orientation": "plus_minus",
                                "ot_online_hit": True,
                                "ot_source": "online",
                                "ot_online_status": status,
                                **{
                                    key: scored[key]
                                    for key in (
                                        "ot_student_model",
                                        "ot_teacher_model",
                                        "ot_student_adapter",
                                        "ot_teacher_adapter",
                                        "ot_student_step",
                                        "ot_teacher_step",
                                        "ot_snapshot_mode",
                                        "ot_ema_decay",
                                        "ot_tokenizer_fingerprint",
                                        "ot_reduction",
                                        "ot_null_mode",
                                        "ot_null_span_tokens",
                                    )
                                    if key in scored
                                },
                            },
                        )
                    )
                for target, updates in validated:
                    target.update(updates)
                for target, updates in privileged_validated:
                    target.update(updates)
                standard_hits = len(validated)
                privileged_hits = len(privileged_validated)
                hits = max(standard_hits, privileged_hits)
                misses = max(0, len(targets) - hits)
                status = str(payload.get("status", "scored"))
                if privileged_validated and status in {"no_eligible_records", "privileged_only"}:
                    status = "privileged_scored"
                self._last_online_privileged_hits = privileged_hits
                return hits, misses, status
            except Exception as exc:
                logger.exception("remote online paired scorer unavailable")
                return 0, len(targets), f"error:{type(exc).__name__}"
        if not student_model or not teacher_model:
            return 0, len(targets), "missing_model_path"
        try:
            try:
                from .online_ot_scorer import OnlinePairedScorer
            except ImportError:
                # A few launcher paths load this file as a top-level module;
                # retain an absolute fallback for that deployment shape.
                from videoagent.verl_ext.online_ot_scorer import OnlinePairedScorer

            scorer = OnlinePairedScorer.get(
                student_model=student_model,
                teacher_model=teacher_model,
                student_adapter=(str(getattr(self, "ot_online_student_adapter", "")) or None),
                teacher_adapter=(str(getattr(self, "ot_online_teacher_adapter", "")) or None),
                base_model=(str(getattr(self, "ot_online_base_model", "")) or None),
                device=str(getattr(self, "ot_online_device", "cuda:0")),
                dtype=str(getattr(self, "ot_online_dtype", "auto")),
            )
            hits, skipped, status = scorer.score_records(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                records=targets,
                reduction=str(getattr(self, "ot_online_reduction", "mean")),
                batch_size=int(getattr(self, "ot_online_batch_size", 2)),
                max_records=int(getattr(self, "ot_online_max_records", 0)),
            )
            # Every unmatched acquisition is either scored or a failed/skipped
            # target; report the complement rather than hiding ineligible rows.
            misses = max(0, len(targets) - hits)
            return hits, misses, status
        except Exception as exc:
            logger.exception("online paired scorer unavailable")
            return 0, len(targets), f"error:{type(exc).__name__}"

    def _convert_image_to_data_url(self, image_path: Path) -> Optional[str]:
        if not image_path.is_file():
            return None
        try:
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                encoded = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return None

    def _query_vision(self, query: str, vid: str) -> tuple[str, bool, str]:
        """Run the vision tool and return ``(text, cache_hit, status)``.

        Status is deliberately explicit so a cache miss or API error cannot
        be mistaken for valid evidence by the paired-OT path.
        """
        if self.offline_cache:
            cached = self._lookup_observation_cache(query, vid)
            if cached:
                if self._is_tool_error_text(cached):
                    return cached, False, "cache_error"
                return cached, True, "cache_hit"
            return f"[LOCAL_CACHE_MISS vision clip={vid}]", False, "cache_miss"
        if self.offline_mock:
            subtitle = self._get_clip_subtitle(vid).strip()
            if len(subtitle) > 1800:
                subtitle = subtitle[:1800]
            return (
                f"Offline visual proxy for {vid}. Subtitle evidence: {subtitle or 'no subtitle available'}",
                False,
                "offline_mock",
            )

        frame_nums = list(range(self.frame_start, self.frame_end + 1, self.frame_step))
        messages_content = []
        for fn in frame_nums:
            image_path = Path(self.base_frame_dir, vid, f"{fn:05d}.jpg")
            image_url = self._convert_image_to_data_url(image_path)
            if image_url:
                messages_content.append({"type": "image_url", "image_url": {"url": image_url}})

        if not messages_content:
            return "No visual frames found for the current clip.", False, "no_frames"

        messages_content.append(
            {
                "type": "text",
                "text": (
                    f"Images 1-{len(messages_content)} are sampled frames from one clip.\n"
                    "Focus on key objects, actions, scene transitions and event clues.\n"
                    f"Question: {query}"
                ),
            }
        )

        try:
            resp = self.vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": messages_content}],
            )
            time.sleep(0.1)
            text = resp.choices[0].message.content or ""
            if not text.strip():
                return "", False, "api_empty"
            return text, False, "api_ok"
        except Exception as exc:
            return f"Vision LLM call failed: {exc}", False, "api_error"

    @staticmethod
    def _postprocess_grounding_response(response: str) -> str:
        clip_match = re.search(r"<clip>.*?</clip>", response, re.DOTALL)
        if clip_match:
            return response[: clip_match.end()]
        return response

    @staticmethod
    def _extract_clip_tag(text: str) -> str:
        match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _query_grounding(self, question_data: dict[str, str], sub_block: str, current_vid: str) -> tuple[str, bool, str]:
        """Run grounding and return ``(clip, cache_hit, status)``."""
        if self.offline_cache:
            cached = self._lookup_grounding_cache(question_data)
            if cached:
                if self._is_tool_error_text(cached):
                    return current_vid or "", False, "cache_error"
                return cached, True, "cache_hit"
            # Preserve the current clip for protocol continuity, but mark the
            # result as a miss.  The caller records this as invalid evidence;
            # it must not receive an OT label.
            return current_vid or "", False, "cache_miss"
        if self.offline_mock:
            # Keep the transition deterministic and local.  Prefer the current
            # clip when it has a subtitle; otherwise choose the first labelled
            # clip in the cached episode block.
            if current_vid and self._get_clip_subtitle(current_vid).strip():
                return current_vid, False, "offline_mock"
            match = re.search(r"<([^>]+)>", sub_block or "")
            return (match.group(1).strip() if match else (current_vid or "")), False, "offline_mock"

        if not sub_block:
            sub_block = self._build_subtitles_for_episode(self._extract_episode_prefix(current_vid))

        prompt = f"""
Question: {question_data.get("q", "")}
Options:
a0: {question_data.get("a0", "")}
a1: {question_data.get("a1", "")}
a2: {question_data.get("a2", "")}
a3: {question_data.get("a3", "")}
a4: {question_data.get("a4", "")}
Subtitles: {sub_block}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>.
Based on the question and subtitles, locate the most relevant clip label.
{current_vid} may be wrong. Return answer as <clip>clip_label</clip>.
"""
        try:
            raw_response = self.grounding_client.chat.completions.create(
                model=self.grounding_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.grounding_temperature,
                max_tokens=self.grounding_max_tokens,
            ).choices[0].message.content
            cleaned = self._postprocess_grounding_response(raw_response or "")
            clip = self._extract_clip_tag(cleaned)
            return clip, False, "api_ok" if clip else "api_empty"
        except Exception:
            logger.exception("Grounding API call failed.")
            return "", False, "api_error"

    @staticmethod
    def _truncate_to_first_action(response_text: str) -> str:
        for pattern in (
            r"<search>.*?</search>",
            r"<request_grounding>.*?</request_grounding>",
            r"<answer>.*?</answer>",
        ):
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return response_text[: match.end()]
        return response_text

    @staticmethod
    def _parse_action(response_text: str) -> tuple[str, str]:
        search_match = re.search(r"<search>(.*?)</search>", response_text, re.DOTALL)
        if search_match:
            return "search", search_match.group(1).strip()
        grounding_match = re.search(r"<request_grounding>(.*?)</request_grounding>", response_text, re.DOTALL)
        if grounding_match:
            return "request_grounding", grounding_match.group(1).strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if answer_match:
            return "answer", answer_match.group(1).strip()
        return "invalid", ""

    @staticmethod
    def _extract_video_id(extra_info: dict[str, Any]) -> str:
        for key in ("predicted_vid", "vid_name", "video_id"):
            value = extra_info.get(key)
            if value:
                return str(value)
        return "default_vid"

    @staticmethod
    def _build_question_data(extra_info: dict[str, Any]) -> dict[str, str]:
        question_data = {
            "q": str(extra_info.get("original_question", "")),
            "_sample_uid": str(extra_info.get("sample_uid") or extra_info.get("uid") or ""),
            "_qid": "" if extra_info.get("qid") is None else str(extra_info.get("qid")),
        }
        choices = extra_info.get("choices", {})
        if isinstance(choices, dict):
            for i in range(5):
                question_data[f"a{i}"] = str(choices.get(str(i), ""))
        return question_data

    def _execute_action(
        self,
        action_type: str,
        content: str,
        current_vid: str,
        question_data: dict[str, str],
        episode_sub_block: str,
    ) -> tuple[str, bool, str, dict[str, Any]]:
        stats: dict[str, Any] = {
            "valid_action": 0,
            "is_vision": 0,
            "is_grounding": 0,
            "tool_cache_hit": 0,
            "cache_hit": 0,
            "tool_status": "not_run",
            "evidence_valid": False,
        }

        if action_type == "answer":
            stats["tool_status"] = "answer"
            return f"\n<answer>{content}</answer>", True, current_vid, stats

        if action_type == "search":
            stats.update({"valid_action": 1, "is_vision": 1})
            vision_response, cache_hit, tool_status = self._query_vision(content, current_vid)
            if tool_status in {"cache_hit", "api_ok"} and self._is_tool_error_text(vision_response):
                # A backend may return an error sentinel with HTTP 200; do
                # not let that text count as evidence merely because the
                # request itself succeeded.
                tool_status = "cache_error" if cache_hit else "api_error"
                cache_hit = False
            stats.update(
                {
                    "tool_cache_hit": int(cache_hit),
                    "cache_hit": int(cache_hit),
                    "tool_status": tool_status,
                    # API success and a verified cache hit are admissible
                    # evidence; mock/miss/error strings are diagnostics only.
                    "evidence_valid": bool(tool_status in {"cache_hit", "api_ok"}),
                }
            )
            bbox_content = self._get_bbox_content(current_vid)
            observation = (
                f"\n<information>Bounding BOX:\n{bbox_content.strip()}\n"
                f"Visual Description:\n{vision_response.strip()}</information>\n"
            )
            return observation, False, current_vid, stats

        if action_type == "request_grounding":
            stats.update({"valid_action": 1, "is_grounding": 1})
            predicted_clip, cache_hit, tool_status = self._query_grounding(
                question_data, episode_sub_block, current_vid
            )
            if tool_status in {"cache_hit", "api_ok"} and self._is_tool_error_text(predicted_clip):
                tool_status = "cache_error" if cache_hit else "api_error"
                cache_hit = False
            stats.update(
                {
                    "tool_cache_hit": int(cache_hit),
                    "cache_hit": int(cache_hit),
                    "tool_status": tool_status,
                    "evidence_valid": bool(tool_status in {"cache_hit", "api_ok"}),
                }
            )
            # Keep the state machine moving on a miss/error, but preserve the
            # status bit so strict paired replay rejects this transition.
            predicted_clip = predicted_clip or current_vid
            subtitle = self._get_clip_subtitle(predicted_clip)
            observation = f"\n<New_clip>{predicted_clip} + {subtitle}</New_clip>\n"
            return observation, False, predicted_clip, stats

        stats["tool_status"] = "invalid_action"
        return (
            "\nMy action is not correct. I need to search, request grounding, or answer.\n",
            False,
            current_vid,
            stats,
        )

    def _append_tokens(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        token_ids: list[int],
        mask_value: int,
    ) -> bool:
        remain = self.response_length - len(response_mask)
        if remain <= 0:
            return False
        clipped = token_ids[:remain]
        if not clipped:
            return False
        prompt_ids.extend(clipped)
        response_ids.extend(clipped)
        response_mask.extend([mask_value] * len(clipped))
        return True

    def _action_token_prefix(self, token_ids: Any) -> tuple[list[int], str]:
        """Keep the exact sampled token prefix through the first action tag.

        Re-tokenizing decoded text can change whitespace and special-token
        boundaries.  Incremental decoding is slower only for the short action
        prefix and preserves the IDs used by the rollout for paired replay.
        """

        raw = [int(x) for x in (token_ids or [])]
        prefix: list[int] = []
        decoded = ""
        for token in raw:
            prefix.append(token)
            decoded = self.tokenizer.decode(prefix, skip_special_tokens=True)
            if re.search(r"</(?:search|request_grounding|answer)>", decoded, re.DOTALL):
                return prefix, decoded
        return raw, decoded or self.tokenizer.decode(raw, skip_special_tokens=True)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        response_ids: list[int] = []
        response_mask: list[int] = []
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        rollout_n = int(kwargs.get("rollout_n", 0) or 0)
        global_step = int(kwargs.get("step", 0) or 0)

        current_vid = self._extract_video_id(extra_info)
        question_data = self._build_question_data(extra_info)
        episode_sub_block = str(extra_info.get("episode_sub_block", ""))

        assistant_turns = 0
        user_turns = 0
        valid_action_count = 0
        vision_count = 0
        grounding_count = 0
        action_history: list[str] = []
        turn_records: list[dict[str, Any]] = []

        while assistant_turns < self.max_assistant_turns and len(response_mask) < self.response_length:
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                )

            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            assistant_ids, decoded = self._action_token_prefix(output.token_ids)
            processed_text = self._truncate_to_first_action(decoded).strip()
            if not processed_text:
                processed_text = decoded.strip()
            if not processed_text:
                break

            assistant_start = len(response_ids)
            if not assistant_ids:
                assistant_ids = self.tokenizer.encode(processed_text, add_special_tokens=False)
            if not assistant_ids:
                assistant_ids = list(output.token_ids)
            if not self._append_tokens(prompt_ids, response_ids, response_mask, assistant_ids, mask_value=1):
                break

            assistant_turns += 1
            action_type, action_content = self._parse_action(processed_text)
            action_history.append(action_type)
            turn_record: dict[str, Any] = {
                "turn": assistant_turns,
                "action_type": action_type,
                "action_text": processed_text,
                "action_content": action_content,
                "current_vid_before": current_vid,
                # Alias used by cache producers; ``current_vid_before`` is
                # retained for backwards-compatible trajectory readers.
                "current_vid": current_vid,
                "assistant_response_start": assistant_start,
                "assistant_response_end": len(response_ids),
                "assistant_token_ids": list(response_ids[assistant_start:]),
                "observation_response_start": None,
                "observation_response_end": None,
                "observation_text": "",
                "observation_token_ids": [],
                "sample_uid": str(extra_info.get("sample_uid") or extra_info.get("uid") or ""),
                "qid": extra_info.get("qid"),
                "rollout_n": rollout_n,
                "global_step": global_step,
                "tool_cache_hit": 0,
                "cache_hit": 0,
                "tool_status": "not_run",
                "evidence_valid": False,
            }
            if action_type == "answer":
                turn_records.append(turn_record)
                break

            if user_turns >= self.max_user_turns:
                turn_records.append(turn_record)
                break

            with simple_timer("tool_calls", metrics):
                observation, should_stop, next_vid, action_stats = await self.loop.run_in_executor(
                    None,
                    lambda: self._execute_action(
                        action_type=action_type,
                        content=action_content,
                        current_vid=current_vid,
                        question_data=question_data,
                        episode_sub_block=episode_sub_block,
                    ),
                )

            current_vid = next_vid
            valid_action_count += int(action_stats["valid_action"])
            vision_count += int(action_stats["is_vision"])
            grounding_count += int(action_stats["is_grounding"])

            # Persist tool provenance on the same record as the intervention.
            # Paired e_obs replay must be able to distinguish a verified local
            # observation/API response from a cache miss, mock, or error text.
            turn_record.update(
                {
                    "tool_cache_hit": int(action_stats.get("tool_cache_hit", 0)),
                    "cache_hit": int(action_stats.get("cache_hit", action_stats.get("tool_cache_hit", 0))),
                    "tool_status": str(action_stats.get("tool_status", "unknown")),
                    "evidence_valid": bool(action_stats.get("evidence_valid", False)),
                }
            )

            if should_stop:
                break

            observation_ids = self.tokenizer.encode(observation, add_special_tokens=False)
            if self.max_obs_length > 0:
                observation_ids = observation_ids[: self.max_obs_length]
            if self.max_tool_response_length > 0:
                observation_ids = observation_ids[: self.max_tool_response_length]

            observation_start = len(response_ids)
            appended_observation = True
            if observation_ids:
                appended_observation = self._append_tokens(
                    prompt_ids, response_ids, response_mask, observation_ids, mask_value=0
                )
            turn_record.update(
                {
                    "observation_response_start": observation_start,
                    "observation_response_end": len(response_ids),
                    "observation_text": observation,
                    "assistant_token_ids": list(response_ids[assistant_start:turn_record["assistant_response_end"]]),
                    "observation_token_ids": list(response_ids[observation_start:]),
                    "current_vid_after": current_vid,
                }
            )
            turn_records.append(turn_record)
            if observation_ids and not appended_observation:
                break
            user_turns += 1

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = -1

        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]

        # The next assistant span is the action target conditioned on the
        # preceding acquisition/observation transition.  Storing offsets (and
        # the exact textual observation) makes later paired replay possible
        # without re-running video tools.
        for i, record in enumerate(turn_records[:-1]):
            nxt = turn_records[i + 1]
            record["next_action_response_start"] = nxt.get("assistant_response_start")
            record["next_action_response_end"] = nxt.get("assistant_response_end")

        # Hash the exact unpadded response and intervention spans.  These are
        # used to reject an offline cache generated from a different sample,
        # tokenizer, truncation boundary, or rollout repeat.
        response_blob = json.dumps(response_ids, separators=(",", ":")).encode("utf-8")
        response_hash = hashlib.sha256(response_blob).hexdigest()
        for record in turn_records:
            record["response_sha256"] = response_hash
            try:
                o0 = int(record.get("observation_response_start"))
                o1 = int(record.get("observation_response_end"))
                t0 = int(record.get("next_action_response_start"))
                t1 = int(record.get("next_action_response_end"))
                a0 = int(record.get("assistant_response_start"))
                a1 = int(record.get("assistant_response_end"))
                # Validate before slicing: Python accepts negative slice
                # bounds and could otherwise hash a different span while
                # making a stale cache appear to match.
                if not (
                    0 <= o0 <= o1 <= t0 < t1 <= len(response_ids)
                    and 0 <= a0 < a1 <= o0
                ):
                    raise ValueError("invalid response-relative hash span")
                record["obs_sha256"] = hashlib.sha256(
                    json.dumps(response_ids[o0:o1], separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                record["target_sha256"] = hashlib.sha256(
                    json.dumps(response_ids[t0:t1], separators=(",", ":")).encode("utf-8")
                ).hexdigest()
            except (TypeError, ValueError, OverflowError):
                record["obs_sha256"] = ""
                record["target_sha256"] = ""

        ot_cache_hits, ot_cache_misses = self._attach_ot_cache(
            turn_records,
            {**extra_info, "rollout_n": rollout_n},
            response_ids=response_ids,
            response_mask=response_mask,
        )
        prompt_only_len = max(0, len(prompt_ids) - len(response_mask))
        # Static hashes are preferred for reproducibility; optional online
        # scoring fills only unmatched records using the exact sampled text.
        # This keeps stochastic rollouts from borrowing an e_obs value from a
        # different response while avoiding a second tool call.
        ot_online_hits, ot_online_misses, ot_online_status = self._attach_online_ot_scores(
            turn_records,
            prompt_ids=prompt_ids[:prompt_only_len],
            response_ids=response_ids,
            response_mask=response_mask,
        )
        # Keep privileged coverage separate from the ordinary four-forward
        # e_obs counters: a row may legitimately have e_obs from a static
        # cache and e_priv from the visual arm in the same request.
        ot_privileged_hits = sum(
            int(
                bool(record.get("ot_privileged_hit"))
                and str(record.get("e_priv_orientation") or "").strip().lower().replace("-", "_")
                in {"teacher_priv_minus_deploy", "teacher_privileged_minus_deploy"}
            )
            for record in turn_records
            if isinstance(record, dict)
        )
        tool_cache_hits = sum(int(record.get("tool_cache_hit", 0)) for record in turn_records)
        tool_cache_misses = sum(
            int(
                str(record.get("tool_status", "")).strip().lower()
                in {
                    "cache_miss",
                    "local_cache_miss",
                    "cache_error",
                    "api_error",
                    "api_empty",
                    "no_frames",
                    "invalid_action",
                }
            )
            for record in turn_records
            if str(record.get("action_type", "")).strip().lower() in _OT_ACTION_TYPES
        )
        valid_evidence_count = sum(int(bool(record.get("evidence_valid", False))) for record in turn_records)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids[:prompt_only_len],
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "predicted_vid": current_vid,
                "valid_action_count": valid_action_count,
                "vision_count": vision_count,
                "grounding_count": grounding_count,
                "tool_used": int((vision_count + grounding_count) > 0),
                "action_history": action_history,
                "request_id": request_id,
                "turn_records": turn_records,
                "tool_cache_hits": int(tool_cache_hits),
                "tool_cache_misses": int(tool_cache_misses),
                "cache_hits": int(tool_cache_hits),
                "valid_evidence_count": int(valid_evidence_count),
                "offline_mock": int(self.offline_mock),
                "offline_cache": int(self.offline_cache),
                "sample_uid": str(extra_info.get("sample_uid") or extra_info.get("uid") or ""),
                "qid": extra_info.get("qid"),
                "rollout_n": rollout_n,
                "global_step": global_step,
                # These are local return values from this trajectory's cache
                # attach; never read mutable worker-level counters because
                # concurrent rollouts can otherwise overwrite one another.
                "ot_cache_hits": int(ot_cache_hits),
                "ot_cache_misses": int(ot_cache_misses),
                "ot_online_hits": int(ot_online_hits),
                "ot_online_misses": int(ot_online_misses),
                "ot_online_status": str(ot_online_status),
                "ot_privileged_hits": int(ot_privileged_hits),
                "ot_online_privileged_hits": int(getattr(self, "_last_online_privileged_hits", 0)),
            },
        )
        return output
