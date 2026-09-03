"""Minimal, opt-in OT-OPD advantage shaping for LongVideoAgent.

The rollout loop stores ``turn_records`` as an object-valued non-tensor field.
This module deliberately does *not* infer an observation effect from reward or
from the observed trajectory.  A strict OT-OPD effect must be produced by a
paired replay/cache as ``e_obs = D(h_plus, u_next) - D(h_minus, u_next)`` and
attached to the corresponding acquisition record (or supplied in
``non_tensor_batch['ot_e_obs']``).  For convenience, a record may instead
carry ``D_plus``/``D_minus`` or the four cached teacher/student log-probability
terms, which are reduced to the same scalar without another model call.

The hook is disabled by default (alpha=0), so importing/calling it preserves
the stock trainer exactly.  In the formal OPSD setting, set
``target_span=next_action`` so the effect is applied to the policy tokens that
follow the observation; ``target_span=acquisition`` remains available as the
legacy acquisition/e_gap_shift diagnostic.  Observation and padding spans are
always excluded.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


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
_SIGNAL_KEYS = (
    "e_obs",
    "e_priv",
    "ot_e_obs",
    "observation_effect",
    "observation_delta",
    "proxy_e_obs",
    "obs_sensitivity",
)
_PROXY_KEYS = frozenset({"proxy_e_obs", "obs_sensitivity"})
_CANONICAL_E_OBS_ORIENTATION = "plus_minus"


def _has_canonical_orientation(record: Mapping[str, Any]) -> bool:
    """Reject explicitly tagged legacy sign conventions.

    Older records omitted the tag; those are interpreted using the original
    builder convention (``D_plus - D_minus``).  An explicit incompatible tag
    is never auto-negated because doing so would hide a producer mismatch.
    """

    raw = record.get("e_obs_orientation", record.get("orientation"))
    if raw is None or str(raw).strip() == "":
        return True
    normalized = str(raw).strip().lower().replace("-", "_")
    return normalized in {"plus_minus", "plusminus", "dplus_minus_dminus"}


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from DictConfig/BaseConfig/dict without assuming its type."""

    if cfg is None:
        return default
    try:
        if hasattr(cfg, "get"):
            value = cfg.get(key, default)
        else:
            value = getattr(cfg, key, default)
    except Exception:
        return default
    return default if value is None else value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_records(value: Any) -> list[Mapping[str, Any]]:
    """Normalize one object-valued row to a list of record mappings."""

    if isinstance(value, Mapping):
        nested = value.get("turn_records")
        if nested is not None:
            return _as_records(nested)
        return [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records: list[Mapping[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                records.append(item)
        return records
    return []


def _row_value(field: Any, row: int, batch_size: int) -> Any:
    """Get one row from an object/list field while tolerating scalar fields."""

    if field is None:
        return None
    if isinstance(field, np.ndarray):
        if field.ndim == 0:
            return field.item()
        if row < len(field):
            return field[row]
        return None
    if isinstance(field, Sequence) and not isinstance(field, (str, bytes, bytearray, Mapping)):
        if len(field) == batch_size:
            return field[row]
    return field


def _finite_float(value: Any) -> float | None:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).size == 1:
        value = np.asarray(value).reshape(-1)[0]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_sum(value: Any) -> float | None:
    """Sum a cached scalar/vector of log-probability gaps, rejecting NaN/Inf."""

    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return _finite_float(value)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return float(arr.sum())


def _paired_record_signal(record: Mapping[str, Any]) -> float | None:
    """Compute strict e_obs from cached plus/minus teacher–student gaps.

    Accepted cache layouts are either ``D_plus``/``D_minus`` (already reduced
    teacher-minus-student gaps) or four log-probability fields:
    ``teacher_plus_logp``, ``teacher_minus_logp``, ``student_plus_logp`` and
    ``student_minus_logp``.  Values may be scalars or per-token vectors.  No
    model call is made here.
    """

    if not _has_canonical_orientation(record):
        return None

    d_plus = _finite_sum(record.get("D_plus", record.get("d_plus")))
    d_minus = _finite_sum(record.get("D_minus", record.get("d_minus")))
    if d_plus is not None and d_minus is not None:
        # Canonical ot-opd-paired-v2 orientation is the signed change in the
        # teacher--student gap after vs. before the observation.
        return d_plus - d_minus

    teacher_plus = _finite_sum(record.get("teacher_plus_logp"))
    teacher_minus = _finite_sum(record.get("teacher_minus_logp"))
    student_plus = _finite_sum(record.get("student_plus_logp"))
    student_minus = _finite_sum(record.get("student_minus_logp"))
    if None not in (teacher_plus, teacher_minus, student_plus, student_minus):
        return (teacher_plus - student_plus) - (teacher_minus - student_minus)
    return None


def _record_signal(record: Mapping[str, Any], signal_key: str, allow_proxy: bool) -> float | None:
    # ``e_priv`` has its own teacher-privileged orientation and is valid even
    # when a co-located legacy e_obs diagnostic carries an incompatible sign
    # tag.  Ordinary e_obs/raw-gap signals still require the canonical
    # plus-minus orientation.
    privileged_signal = signal_key in {
        "e_priv",
        "ot_e_priv",
        "privileged_observation_effect",
        "privileged_visual_gain",
    }
    if not privileged_signal and not _has_canonical_orientation(record):
        return None
    if privileged_signal:
        # Teacher-only privileged visual gain is accepted only when the local
        # scorer marked it as a successful answer-free cache lookup.  This
        # prevents a stale scalar e_priv from being mistaken for OPSD credit.
        hit = record.get("ot_privileged_hit")
        if not (hit is True or str(hit).strip().lower() in {"1", "true", "yes", "y"}):
            return None
        orientation = str(record.get("e_priv_orientation") or "").strip().lower().replace("-", "_")
        if orientation not in {"teacher_priv_minus_deploy", "teacher_privileged_minus_deploy"}:
            return None
        value = _finite_float(record.get(signal_key, record.get("e_priv")))
        return value
    keys = (signal_key,) if signal_key else _SIGNAL_KEYS
    if signal_key not in _SIGNAL_KEYS:
        keys = (signal_key, *_SIGNAL_KEYS)
    for key in keys:
        if key in _PROXY_KEYS and not allow_proxy:
            continue
        value = _finite_float(record.get(key))
        if value is not None:
            return value
    # If a paired replay stored raw teacher/student log-probability terms, form
    # the strict observation effect locally without rerunning either model.
    return _paired_record_signal(record)


def _row_signal_map(value: Any) -> dict[int, float]:
    """Parse optional row-level e_obs as {turn_index: value}."""

    if isinstance(value, Mapping):
        result: dict[int, float] = {}
        for key, item in value.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            val = _finite_float(item)
            if val is not None:
                result[idx] = val
        return result
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = {}
        for idx, item in enumerate(value):
            val = _finite_float(item)
            if val is not None:
                result[idx] = val
        return result
    return {}


def _resolve_settings(config: Any) -> dict[str, Any]:
    """Resolve optional config, falling back to environment variables.

    We intentionally read ``config.get('ot_opd')`` rather than requiring a new
    AlgoConfig dataclass field.  This keeps the patch compatible with stock
    verl configs; environment variables are useful for a pilot without YAML
    schema changes.
    """

    section = _get(config, "ot_opd", {}) or {}
    alpha = _get(section, "alpha", None)
    if alpha is None:
        alpha = _env_float("OT_OPD_ALPHA", 0.0)
    clip = _get(section, "clip", None)
    if clip is None:
        clip = _env_float("OT_OPD_CLIP", 2.0)
    normalize = _get(section, "normalize", None)
    if normalize is None:
        normalize = _env_bool("OT_OPD_NORMALIZE", True)
    allow_proxy = _get(section, "allow_proxy", None)
    if allow_proxy is None:
        allow_proxy = _env_bool("OT_OPD_ALLOW_PROXY", False)
    signal_key = str(_get(section, "signal_key", os.getenv("OT_OPD_SIGNAL_KEY", "e_obs")))
    require_next = _get(section, "require_next_action", None)
    if require_next is None:
        require_next = _env_bool("OT_OPD_REQUIRE_NEXT_ACTION", True)
    per_token = _get(section, "per_token", None)
    if per_token is None:
        per_token = _env_bool("OT_OPD_PER_TOKEN", True)
    target_span = str(_get(section, "target_span", os.getenv("OT_OPD_TARGET_SPAN", "acquisition")) or "acquisition")
    target_span = target_span.strip().lower().replace("-", "_")
    if target_span in {"next", "nextaction", "next_action_tokens"}:
        target_span = "next_action"
    if target_span not in {"acquisition", "next_action"}:
        target_span = "acquisition"
    mode = str(_get(section, "mode", os.getenv("OT_OPD_MODE", "additive")) or "additive")
    mode = mode.strip().lower().replace("-", "_")
    if mode in {"scale", "sign_preserving", "sign_preserving_multiplier", "multiplicative"}:
        mode = "multiplier"
    if mode not in {"additive", "multiplier"}:
        mode = "additive"
    negate_signal = _get(section, "negate_signal", None)
    if negate_signal is None:
        negate_signal = _env_bool("OT_OPD_NEGATE_SIGNAL", False)
    multiplier_bound = _get(section, "multiplier_bound", None)
    if multiplier_bound is None:
        multiplier_bound = _env_float("OT_OPD_MULTIPLIER_BOUND", 0.5)
    try:
        multiplier_bound = min(abs(float(multiplier_bound)), 0.99)
    except (TypeError, ValueError):
        multiplier_bound = 0.5
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        alpha = 0.0
    try:
        clip = abs(float(clip))
    except (TypeError, ValueError):
        clip = 2.0
    return {
        "alpha": alpha,
        "clip": clip,
        "normalize": bool(normalize),
        "allow_proxy": bool(allow_proxy),
        "signal_key": signal_key,
        "require_next_action": bool(require_next),
        "per_token": bool(per_token),
        "target_span": target_span,
        "mode": mode,
        "multiplier_bound": multiplier_bound,
        "negate_signal": bool(negate_signal),
    }


def apply_ot_opd_advantage(data: Any, config: Any = None) -> tuple[Any, dict[str, float]]:
    """Add opt-in action-only OT-OPD shaping to already computed advantages.

    Returns ``(data, metrics)``.  With the default alpha=0, missing fields, or
    no valid cached effects, this is a no-op and returns an empty metrics dict.
    The function is deliberately conservative: a record must contain an
    acquisition action, valid response-relative offsets, and (when
    ``target_span=next_action``) a following action span.  This prevents
    crediting answer/observation tokens or stale offsets after response
    truncation.
    """

    settings = _resolve_settings(config)
    alpha = settings["alpha"]
    if alpha == 0.0:
        return data, {}
    if not hasattr(data, "batch") or not hasattr(data, "non_tensor_batch"):
        return data, {}
    if "advantages" not in data.batch or "responses" not in data.batch:
        return data, {}

    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"] if "response_mask" in data.batch.keys() else None
    if response_mask is None:
        response_mask = torch.ones_like(advantages)
    if advantages.ndim != 2 or response_mask.shape != advantages.shape:
        return data, {"ot_opd/skipped_shape": 1.0}
    batch_size, response_len = advantages.shape

    nt = data.non_tensor_batch
    records_field = nt.get("turn_records")
    if records_field is None:
        # Some reward-loop paths retain all rollout extras under this wrapper.
        extras_field = nt.get("tool_extra_fields")
        if extras_field is not None:
            records_field = []
            for row in range(batch_size):
                extra = _row_value(extras_field, row, batch_size)
                records_field.append(extra.get("turn_records") if isinstance(extra, Mapping) else None)
    if records_field is None:
        return data, {"ot_opd/skipped_no_records": float(batch_size)}

    row_effects = nt.get("ot_e_obs")
    row_priv_effects = nt.get("ot_e_priv", nt.get("e_priv"))
    row_proxy_effects = nt.get("proxy_e_obs")
    row_orientations = nt.get("ot_e_obs_orientation", nt.get("e_obs_orientation"))
    row_priv_orientations = nt.get("ot_e_priv_orientation", nt.get("e_priv_orientation"))
    orientation_misses = 0
    # Keep the source record alongside each span so we can write the exact
    # normalized signal ``z`` and snapshot provenance into rollout JSONL.
    candidates: list[tuple[int, int, int, float, Mapping[str, Any]]] = []
    raw_values: list[float] = []

    for row in range(batch_size):
        records = _as_records(_row_value(records_field, row, batch_size))
        external = _row_signal_map(
            _row_value(row_priv_effects if settings["signal_key"] == "e_priv" else row_effects, row, batch_size)
        )
        row_orientation = _row_value(row_orientations, row, batch_size)
        row_orientation_valid = row_orientation is None or _has_canonical_orientation(
            {"e_obs_orientation": row_orientation}
        )
        if settings["signal_key"] == "e_priv":
            row_priv_orientation = _row_value(row_priv_orientations, row, batch_size)
            normalized_priv_orientation = (
                ""
                if row_priv_orientation is None
                else str(row_priv_orientation).strip().lower().replace("-", "_")
            )
            # A row-level privileged scalar has no record-level provenance;
            # require an explicit orientation rather than guessing whether it
            # is teacher-minus-deploy or its negation.
            if normalized_priv_orientation not in {
                "teacher_priv_minus_deploy",
                "teacher_privileged_minus_deploy",
            }:
                row_priv_orientation_valid = False
            else:
                row_priv_orientation_valid = True
        else:
            row_priv_orientation_valid = True
        if not row_orientation_valid or not row_priv_orientation_valid:
            # A row-level scalar has no D+/D− pair from which to infer a sign;
            # reject an explicitly incompatible producer instead of silently
            # negating it.
            external = {}
            orientation_misses += 1
        if settings["allow_proxy"] and row_orientation_valid:
            # A strict cached value wins over an explicitly enabled proxy.
            for idx, value in _row_signal_map(_row_value(row_proxy_effects, row, batch_size)).items():
                external.setdefault(idx, value)
        for record_idx, record in enumerate(records):
            action_type = str(record.get("action_type", "")).strip().lower()
            if action_type not in _ACTION_TYPES:
                continue
            if settings["require_next_action"]:
                next_start = record.get("next_action_response_start")
                next_end = record.get("next_action_response_end")
                try:
                    if next_start is None or next_end is None or int(next_end) <= int(next_start):
                        continue
                    # The next action must occur after the observation that
                    # this acquisition produced.  This guards against stale
                    # offsets and accidental credit leakage into the same turn.
                    observation_end = record.get("observation_response_end")
                    if observation_end is not None and int(next_start) < int(observation_end):
                        continue
                except (TypeError, ValueError):
                    continue
            value = _record_signal(record, settings["signal_key"], settings["allow_proxy"])
            # An explicit e_priv run must not silently consume a row-level
            # e_obs scalar from the legacy cache.  This is especially
            # important when both arms are logged in the same batch.
            if value is None and settings["signal_key"] != "e_priv":
                value = external.get(record_idx)
            if value is None and settings["signal_key"] != "e_priv":
                # Also accept one-based turn numbering in external maps.
                value = external.get(int(record.get("turn", record_idx)) - 1)
            if value is None:
                continue
            if settings["negate_signal"] and settings["signal_key"] != "e_priv":
                # Formal eta orientation: eta = D_minus - D_plus = -e_obs.
                value = -float(value)
            if settings["target_span"] == "next_action":
                # Formal OPSD target: score/shape the policy tokens emitted
                # *after* the observation.  Never put credit on the tool text
                # itself or on the acquisition that requested it.
                try:
                    start = int(record.get("next_action_response_start"))
                    end = int(record.get("next_action_response_end"))
                    observation_end = int(record.get("observation_response_end"))
                except (TypeError, ValueError):
                    continue
                if not (0 <= start < end <= response_len and observation_end <= start):
                    continue
            else:
                # Legacy acquisition/e_gap_shift diagnostic.  This branch is
                # retained for backwards-compatible ablations only.
                try:
                    start = int(record.get("assistant_response_start"))
                    end = int(record.get("assistant_response_end"))
                except (TypeError, ValueError):
                    continue
                start = max(0, start)
                end = min(response_len, end)
                if end <= start:
                    continue
                observation_start = record.get("observation_response_start")
                if observation_start is not None:
                    try:
                        if end > int(observation_start):
                            continue
                    except (TypeError, ValueError):
                        continue
            # Keep the credit on policy tokens only.  The rollout loop marks
            # observations with response_mask=0; intersecting below provides a
            # second guard against malformed/shifted spans.
            try:
                if not bool(torch.all(response_mask[row, start:end] > 0).item()):
                    continue
            except Exception:
                continue
            candidates.append((row, start, end, float(value), record))
            raw_values.append(float(value))

    if not candidates:
        metrics = {"ot_opd/skipped_no_effect": 1.0}
        if orientation_misses:
            metrics["ot_opd/orientation_miss"] = float(orientation_misses)
        return data, metrics

    values = np.asarray(raw_values, dtype=np.float32)
    if settings["normalize"] and values.size > 1:
        mean = float(values.mean())
        std = float(values.std())
        if std > 1e-6:
            values = (values - mean) / std
        else:
            values = values - mean
    if settings["clip"] > 0:
        values = np.clip(values, -settings["clip"], settings["clip"])

    shaping = torch.zeros_like(advantages, dtype=advantages.dtype)
    multiplier_delta = torch.zeros_like(advantages, dtype=advantages.dtype)
    applied_tokens = 0
    applied_records = 0
    for (row, start, end, _, record), value in zip(candidates, values, strict=False):
        valid = response_mask[row, start:end].to(dtype=shaping.dtype)
        if valid.numel() == 0 or float(valid.sum().item()) <= 0:
            continue
        credit = float(value)
        # ``z`` is the normalized/clipped per-record signal before optional
        # per-token averaging.  Keep it on the record for auditability; all
        # teacher/student snapshot fields are copied by the rollout scorer.
        try:
            record["ot_opd_z"] = credit
            record["ot_opd_target_span"] = settings["target_span"]
            record["ot_opd_signal_key"] = settings["signal_key"]
            record["ot_opd_signal_orientation"] = "eta_negated_plus_minus" if settings["negate_signal"] else "plus_minus"
        except Exception:
            pass
        if settings["mode"] == "multiplier":
            # Sign-preserving multiplier: e_priv/e_obs can only reweight the
            # existing GRPO direction.  It cannot create an update when the
            # base advantage is zero or reverse a positive/negative update.
            span_adv = advantages[row, start:end]
            aligned = torch.sign(span_adv) * float(credit)
            strength = torch.as_tensor(float(alpha), dtype=shaping.dtype, device=shaping.device)
            multiplier = 1.0 + strength * torch.tanh(aligned)
            multiplier = torch.clamp(
                multiplier,
                1.0 - float(settings["multiplier_bound"]),
                1.0 + float(settings["multiplier_bound"]),
            )
            multiplier_delta[row, start:end] += (multiplier - 1.0) * span_adv * valid
        else:
            if settings["per_token"]:
                credit /= max(float(valid.sum().item()), 1.0)
            shaping[row, start:end] += valid * credit
        applied_tokens += int((valid > 0).sum().item())
        applied_records += 1

    if applied_records == 0:
        return data, {"ot_opd/skipped_mask": 1.0}
    if settings["mode"] == "multiplier":
        data.batch["advantages"] = advantages + multiplier_delta
    else:
        data.batch["advantages"] = advantages + alpha * shaping
    metrics = {
        "ot_opd/applied_records": float(applied_records),
        "ot_opd/applied_tokens": float(applied_tokens),
        "ot_opd/e_obs_mean": float(np.mean(values)),
        "ot_opd/e_obs_std": float(np.std(values)),
        "ot_opd/alpha": float(alpha),
        "ot_opd/target_next_action": float(settings["target_span"] == "next_action"),
        "ot_opd/target_acquisition": float(settings["target_span"] == "acquisition"),
        "ot_opd/z_mean": float(np.mean(values)),
        "ot_opd/z_std": float(np.std(values)),
        "ot_opd/mode_multiplier": float(settings["mode"] == "multiplier"),
        "ot_opd/multiplier_bound": float(settings["multiplier_bound"]),
        "ot_opd/negate_signal": float(settings["negate_signal"]),
    }
    if orientation_misses:
        metrics["ot_opd/orientation_miss"] = float(orientation_misses)
    return data, metrics


__all__ = ["apply_ot_opd_advantage"]
