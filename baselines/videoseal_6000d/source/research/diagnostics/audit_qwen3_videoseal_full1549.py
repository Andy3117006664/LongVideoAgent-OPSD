#!/usr/bin/env python3
"""Terminal audit for the fair full-LVBench Qwen3/VideoSEAL comparison.

The audit computes paired accuracy only after both frozen arms are exhausted.
Normally that means 1549 non-empty predictions per arm.  A separately frozen
shared-invalid policy may instead count the same repeatedly invalid UID as
incorrect in both arms; it never imputes an A-D prediction.
Before terminal readiness, ``--allow-incomplete-integrity-check`` may be used
to validate configuration, traces, and already-written trajectories without
producing any accuracy field.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_qwen3_planner_swap_pilot as base


EXPECTED_TOTAL = 1549
EXPECTED_UID_SHA256 = "2d248aef3c040839786ab0c284296822e9b1a451443d756c8a216cd1ce577183"
EXPECTED_DATASET_SHA256 = "64d1e5325dac13edc363a26b842bfcb5e13231e77778e8ca7566096e1e48e93c"
EXPECTED_INSPECTOR = "Qwen2.5-VL-7B-Instruct"
EXPECTED_SHARED_CONFIG: dict[str, Any] = {
    "inspector_model": EXPECTED_INSPECTOR,
    "planner_max_tokens": 2048,
    "planner_context_limit": 40960,
    "legacy_history_max_blocks": 0,
    "legacy_history_max_chars": 0,
    "max_steps": 16,
    "frame_budget": 64,
    "inspect_max_total_images": 64,
    "inspect_fps": 2.0,
    "tool_budget": 16,
    "retrieval_requested": "bm25",
    "retrieval_effective": "bm25",
    "strict_final_gate": True,
    "time_reference_used_online": False,
    "concurrency": 6,
}
ARM_CONFIG = {
    "qwen3": {
        **EXPECTED_SHARED_CONFIG,
        "planner_model": "Qwen3-8B",
        "planner_endpoint": "http://127.0.0.1:18107/v1",
        "inspector_endpoint": "http://127.0.0.1:18105/v1",
    },
    "videoseal": {
        **EXPECTED_SHARED_CONFIG,
        "planner_model": "VideoSEAL_8B",
        "planner_endpoint": "http://127.0.0.1:18104/v1",
        "inspector_endpoint": "http://127.0.0.1:18103/v1",
    },
}
FORBIDDEN_MODEL_INPUT_KEYS = {
    "time_reference",
    "ground_truth",
    "groundtruth",
    "gold_answer",
    "reference_span",
    "reference_spans",
}


def project_path(value: Path) -> Path:
    path = (value if value.is_absolute() else ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {path}") from exc
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_uids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate UID in {path}")
    return values


def terminal_status(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("state") == "complete"
        and value.get("total") == EXPECTED_TOTAL
        and value.get("completed") == EXPECTED_TOTAL
        and value.get("failed") == 0
        and value.get("remaining") == 0
    )


def valid_completed_result(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("state") == "completed"
        and not str(value.get("error") or "").strip()
        and base.answer_key(value.get("prediction")).upper() in {"A", "B", "C", "D"}
    )


def partial_completed_scope(
    expected: set[tuple[str, str]],
    qwen: Mapping[tuple[str, str], Mapping[str, Any]],
    videoseal: Mapping[tuple[str, str], Mapping[str, Any]],
) -> set[tuple[str, str]]:
    return {
        key
        for key in expected & set(qwen) & set(videoseal)
        if valid_completed_result(qwen[key]) and valid_completed_result(videoseal[key])
    }


def scoring_terminal_status(
    value: Mapping[str, Any], *, shared_invalid_count: int
) -> bool:
    completed = int(value.get("completed") or 0)
    failed = int(value.get("failed") or 0)
    return bool(
        value.get("state") in {"complete", "failed"}
        and value.get("total") == EXPECTED_TOTAL
        and completed + failed == EXPECTED_TOTAL
        and failed == shared_invalid_count
        and value.get("remaining") == 0
    )


def load_shared_invalid_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "sha256": None,
            "uids": set(),
            "minimum_failed_attempts_per_arm": 0,
            "errors": [],
        }
    errors: list[dict[str, Any]] = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "sha256": None,
            "uids": set(),
            "minimum_failed_attempts_per_arm": 0,
            "errors": [{"reason": f"shared_invalid_policy_unreadable:{exc}"}],
        }
    raw_uids = value.get("shared_invalid_uids")
    uids = {
        str(uid).strip()
        for uid in raw_uids
        if str(uid).strip()
    } if isinstance(raw_uids, list) else set()
    if value.get("evaluation") != "qwen3_videoseal_shared_invalid_terminal_policy":
        errors.append({"reason": "shared_invalid_policy_identity_mismatch"})
    if not uids:
        errors.append({"reason": "shared_invalid_policy_has_no_uids"})
    minimum_attempts = int(value.get("minimum_failed_attempts_per_arm") or 0)
    if minimum_attempts < 1:
        errors.append({"reason": "shared_invalid_minimum_attempts_invalid"})
    scoring = value.get("scoring") if isinstance(value.get("scoring"), Mapping) else {}
    required_scoring = {
        "invalid_counts_as_incorrect": True,
        "invalid_pair_counts_as_both_wrong": True,
        "prediction_imputation_forbidden": True,
        "ground_truth_must_not_be_used_to_repair_prediction": True,
    }
    for field, expected in required_scoring.items():
        if scoring.get(field) is not expected:
            errors.append({"reason": "shared_invalid_scoring_policy_mismatch", "field": field})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "uids": uids,
        "minimum_failed_attempts_per_arm": minimum_attempts,
        "errors": errors,
    }


def audit_status(
    path: Path,
    expected_config: Mapping[str, Any],
    *,
    shared_invalid_count: int = 0,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "terminal": False,
            "scoring_terminal": False,
            "snapshot": {},
            "config": {},
            "errors": [{"reason": f"status_unreadable:{exc}"}],
        }
    config = value.get("config") if isinstance(value.get("config"), Mapping) else {}
    for field, expected in expected_config.items():
        actual = config.get(field)
        if actual != expected:
            errors.append(
                {
                    "reason": "config_mismatch",
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return {
        "terminal": terminal_status(value),
        "scoring_terminal": scoring_terminal_status(
            value, shared_invalid_count=shared_invalid_count
        ),
        "snapshot": {
            key: value.get(key)
            for key in ("state", "total", "processed", "completed", "failed", "remaining", "updated_at")
        },
        "config": {field: config.get(field) for field in expected_config},
        "errors": errors,
    }


def forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().casefold()
            location = f"{prefix}.{key}" if prefix else key
            if key in FORBIDDEN_MODEL_INPUT_KEYS:
                found.append(location)
            found.extend(forbidden_keys(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def resolve_trajectory(run_root: Path, row: Mapping[str, Any]) -> Path | None:
    raw = str(row.get("trajectory_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = run_root / path
    try:
        path.resolve().relative_to(run_root.resolve())
    except ValueError:
        return None
    return path


def validate_shared_invalid_rows(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    attempts: Mapping[tuple[str, str], int],
    expected: set[tuple[str, str]],
    *,
    allowed_uids: set[str],
    minimum_attempts: int,
) -> dict[str, Any]:
    policy_keys = {key for key in expected if key[1] in allowed_uids}
    errors: list[dict[str, Any]] = []
    if len(policy_keys) != len(allowed_uids):
        errors.append(
            {
                "reason": "shared_invalid_uid_not_unique_in_manifest",
                "policy_uids": sorted(allowed_uids),
                "manifest_keys": [list(key) for key in sorted(policy_keys)],
            }
        )
    actual_invalid = {
        key
        for key in expected & set(rows)
        if rows[key].get("state") != "completed"
        or base.answer_key(rows[key].get("prediction")).upper() not in {"A", "B", "C", "D"}
    }
    if actual_invalid != policy_keys:
        errors.append(
            {
                "reason": "shared_invalid_set_mismatch",
                "expected": [list(key) for key in sorted(policy_keys)],
                "actual": [list(key) for key in sorted(actual_invalid)],
            }
        )
    attempt_counts: dict[str, int] = {}
    for key in sorted(policy_keys):
        row = rows.get(key) or {}
        count = int(attempts.get(key) or 0)
        attempt_counts[key[1]] = count
        if row.get("state") != "failed":
            errors.append({"uid": key[1], "reason": "shared_invalid_state_not_failed"})
        if str(row.get("prediction") or "").strip():
            errors.append({"uid": key[1], "reason": "shared_invalid_prediction_not_empty"})
        if count < minimum_attempts:
            errors.append(
                {
                    "uid": key[1],
                    "reason": "shared_invalid_attempts_below_minimum",
                    "minimum": minimum_attempts,
                    "actual": count,
                }
            )
    return {
        "policy_keys": policy_keys,
        "attempt_counts": attempt_counts,
        "errors": errors,
        "passed": not errors,
    }


def audit_arm(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    expected: set[tuple[str, str]],
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    run_root: Path,
    *,
    shared_invalid_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    shared_invalid_keys = shared_invalid_keys or set()
    errors: list[dict[str, Any]] = []
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    if missing:
        errors.append({"reason": "missing_result_keys", "count": len(missing), "keys": [list(x) for x in missing[:20]]})
    if extra:
        errors.append({"reason": "extra_result_keys", "count": len(extra), "keys": [list(x) for x in extra[:20]]})

    readable = 0
    total_steps = 0
    min_steps: int | None = None
    max_steps = 0
    actions: Counter[str] = Counter()
    online_time_reference_records = 0
    forced_fallback_trajectories = 0
    valid_prediction_count = 0
    accepted_shared_invalid: list[str] = []
    for key in sorted(expected & set(rows)):
        row = rows[key]
        uid = key[1]
        prediction = base.answer_key(row.get("prediction")).upper()
        if key in shared_invalid_keys:
            if row.get("state") == "failed" and not prediction:
                accepted_shared_invalid.append(uid)
            else:
                errors.append(
                    {
                        "uid": uid,
                        "reason": "shared_invalid_row_mismatch",
                        "state": row.get("state"),
                        "prediction": prediction,
                    }
                )
        else:
            if row.get("state") != "completed" or row.get("error"):
                errors.append(
                    {
                        "uid": uid,
                        "reason": "result_not_completed",
                        "state": row.get("state"),
                        "error": row.get("error"),
                    }
                )
            if prediction not in {"A", "B", "C", "D"}:
                errors.append({"uid": uid, "reason": "invalid_prediction", "prediction": prediction})
        if prediction in {"A", "B", "C", "D"}:
            valid_prediction_count += 1
        truth = base.answer_key((manifest.get(key) or {}).get("ground_truth"))
        row_truth = base.answer_key(row.get("ground_truth"))
        if not truth or row_truth != truth:
            errors.append(
                {
                    "uid": uid,
                    "reason": "result_truth_mismatch",
                    "manifest": truth,
                    "result": row_truth,
                }
            )

        trajectory_path = resolve_trajectory(run_root, row)
        if trajectory_path is None or not trajectory_path.is_file():
            errors.append({"uid": uid, "reason": "trajectory_missing_or_outside_run_root"})
            continue
        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"uid": uid, "reason": f"trajectory_unreadable:{exc}"})
            continue
        readable += 1
        if str(trajectory.get("uid") or "") != uid:
            errors.append({"uid": uid, "reason": "trajectory_uid_mismatch"})
        if str(trajectory.get("video_id") or "") != key[0]:
            errors.append({"uid": uid, "reason": "trajectory_video_id_mismatch"})
        if trajectory.get("time_reference") not in (None, ""):
            online_time_reference_records += 1
            errors.append({"uid": uid, "reason": "trajectory_exposes_time_reference"})
        steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
        if not steps:
            errors.append({"uid": uid, "reason": "trajectory_steps_missing"})
            continue
        count = len(steps)
        total_steps += count
        min_steps = count if min_steps is None else min(min_steps, count)
        max_steps = max(max_steps, count)
        if count > 17:
            errors.append({"uid": uid, "reason": "trajectory_length_exceeded", "steps": count})
        if count == 17:
            last = steps[-1] if isinstance(steps[-1], Mapping) else {}
            last_action = last.get("action") if isinstance(last.get("action"), Mapping) else {}
            last_observation = last.get("observation") if isinstance(last.get("observation"), Mapping) else {}
            if (
                last_action.get("name") == "visual_inspect"
                and last_observation.get("forced") is True
                and last_observation.get("mode") == "full_video"
            ):
                forced_fallback_trajectories += 1
            else:
                errors.append({"uid": uid, "reason": "invalid_seventeenth_step"})
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                errors.append({"uid": uid, "reason": "invalid_step", "step": index})
                continue
            prompt = str(step.get("chat_prompt") or "").casefold()
            if "[time reference:" in prompt or "time_reference" in prompt:
                online_time_reference_records += 1
                errors.append({"uid": uid, "reason": "chat_prompt_exposes_time_reference", "step": index})
            action = step.get("action") if isinstance(step.get("action"), Mapping) else {}
            name = str(action.get("name") or "").strip()
            if name:
                actions[name] += 1
            exposed = forbidden_keys(action.get("arguments"), "action.arguments")
            observation = step.get("observation") if isinstance(step.get("observation"), Mapping) else {}
            exposed.extend(forbidden_keys(observation.get("output"), "observation.output"))
            if exposed:
                errors.append(
                    {
                        "uid": uid,
                        "reason": "privileged_key_in_model_visible_tool_io",
                        "step": index,
                        "locations": exposed,
                    }
                )
    n = len(expected & set(rows))
    return {
        "expected": len(expected),
        "unique_results": len(rows),
        "missing": len(missing),
        "extra": len(extra),
        "readable_trajectories": readable,
        "online_time_reference_records": online_time_reference_records,
        "forced_full_video_fallback_trajectories": forced_fallback_trajectories,
        "valid_prediction_count": valid_prediction_count,
        "nonempty_prediction_coverage_percent": (
            100.0 * valid_prediction_count / len(expected) if expected else None
        ),
        "accepted_shared_invalid_uids": accepted_shared_invalid,
        "step_budget": {
            "total": total_steps,
            "mean": total_steps / n if n else None,
            "min": min_steps,
            "max": max_steps,
        },
        "tool_calls": dict(actions),
        "errors": errors,
        "passed": not errors,
    }


def audit_trace(
    path: Path,
    *,
    planner_model: str,
    planner_endpoint: str,
    inspector_endpoint: str,
) -> dict[str, Any]:
    report = base.scan_online_trace(
        path,
        expected_planner=planner_model,
        expected_inspector=EXPECTED_INSPECTOR,
        expected_planner_endpoint=planner_endpoint,
        expected_inspector_endpoint=inspector_endpoint,
    )
    token_caps: dict[str, Counter[int]] = {}
    token_errors: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            model = str(row.get("model") or "")
            cap = int(row.get("max_tokens"))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        token_caps.setdefault(model, Counter())[cap] += 1
        if model == planner_model and cap != 2048:
            token_errors.append({"line": line_number, "reason": "planner_token_cap_mismatch", "actual": cap})
        if model == EXPECTED_INSPECTOR and cap not in {2048, 4096}:
            token_errors.append({"line": line_number, "reason": "inspector_token_cap_mismatch", "actual": cap})
    report["max_tokens"] = {model: dict(counts) for model, counts in token_caps.items()}
    report["errors"] = list(report.get("errors") or []) + token_errors
    report["passed"] = bool(report.get("rows")) and not report["errors"]
    return report


def paired_comparison(
    qwen: Mapping[tuple[str, str], Mapping[str, Any]],
    videoseal: Mapping[tuple[str, str], Mapping[str, Any]],
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    expected: set[tuple[str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in sorted(expected):
        truth = base.answer_key(manifest[key].get("ground_truth"))
        q_prediction = base.answer_key(qwen[key].get("prediction"))
        v_prediction = base.answer_key(videoseal[key].get("prediction"))
        q_correct = bool(truth and q_prediction == truth)
        v_correct = bool(truth and v_prediction == truth)
        outcome = (
            "qwen3_fixed"
            if q_correct and not v_correct
            else "qwen3_broken"
            if v_correct and not q_correct
            else "both_correct"
            if q_correct
            else "both_wrong"
        )
        rows.append(
            {
                "video_id": key[0],
                "uid": key[1],
                "qwen3_prediction": q_prediction.upper(),
                "videoseal_prediction": v_prediction.upper(),
                "qwen3_correct": q_correct,
                "videoseal_correct": v_correct,
                "paired_outcome": outcome,
            }
        )
    n = len(rows)
    q_correct = sum(bool(row["qwen3_correct"]) for row in rows)
    v_correct = sum(bool(row["videoseal_correct"]) for row in rows)
    fixed = sum(row["paired_outcome"] == "qwen3_fixed" for row in rows)
    broken = sum(row["paired_outcome"] == "qwen3_broken" for row in rows)
    return {
        "n": n,
        "qwen3_correct": q_correct,
        "videoseal_correct": v_correct,
        "qwen3_accuracy_percent": 100.0 * q_correct / n if n else None,
        "videoseal_accuracy_percent": 100.0 * v_correct / n if n else None,
        "qwen3_delta_percentage_points": 100.0 * (q_correct - v_correct) / n if n else None,
        "qwen3_fixed": fixed,
        "qwen3_broken": broken,
        "qwen3_net": fixed - broken,
        "both_correct": sum(row["paired_outcome"] == "both_correct" for row in rows),
        "both_wrong": sum(row["paired_outcome"] == "both_wrong" for row in rows),
        "same_prediction": sum(row["qwen3_prediction"] == row["videoseal_prediction"] for row in rows),
        "exact_mcnemar_two_sided": base.exact_mcnemar(fixed, broken),
        "qwen3_raw_exceeds_videoseal": q_correct > v_correct,
        "qwen3_raw_exceeds_with_p_lt_0_05": q_correct > v_correct and base.exact_mcnemar(fixed, broken) < 0.05,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--uids", type=Path, required=True)
    parser.add_argument("--qwen-results", type=Path, required=True)
    parser.add_argument("--qwen-run-root", type=Path, required=True)
    parser.add_argument("--qwen-trace", type=Path, required=True)
    parser.add_argument("--qwen-status", type=Path, required=True)
    parser.add_argument("--videoseal-results", type=Path, required=True)
    parser.add_argument("--videoseal-run-root", type=Path, required=True)
    parser.add_argument("--videoseal-trace", type=Path, required=True)
    parser.add_argument("--videoseal-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-invalid-policy", type=Path)
    parser.add_argument("--allow-incomplete-integrity-check", action="store_true")
    args = parser.parse_args()
    paths = {
        name: project_path(value)
        for name, value in {
            "parquet": args.parquet,
            "uids": args.uids,
            "qwen_results": args.qwen_results,
            "qwen_run_root": args.qwen_run_root,
            "qwen_trace": args.qwen_trace,
            "qwen_status": args.qwen_status,
            "videoseal_results": args.videoseal_results,
            "videoseal_run_root": args.videoseal_run_root,
            "videoseal_trace": args.videoseal_trace,
            "videoseal_status": args.videoseal_status,
            "output": args.output,
        }.items()
    }
    shared_invalid_policy_path = (
        project_path(args.shared_invalid_policy)
        if args.shared_invalid_policy is not None
        else None
    )
    shared_invalid_policy = load_shared_invalid_policy(shared_invalid_policy_path)
    shared_invalid_uids = set(shared_invalid_policy["uids"])
    shared_invalid_count = len(shared_invalid_uids)

    uid_order = read_uids(paths["uids"])
    uid_hash = sha256_file(paths["uids"])
    dataset_hash = sha256_file(paths["parquet"])
    manifest = base.load_manifest(paths["parquet"])
    expected_uids = set(uid_order)
    expected = {key for key in manifest if key[1] in expected_uids}
    qwen, qwen_attempts, qwen_parse_errors = base.load_latest(paths["qwen_results"])
    videoseal, videoseal_attempts, videoseal_parse_errors = base.load_latest(paths["videoseal_results"])
    qwen = {key: value for key, value in qwen.items() if key[1] in expected_uids}
    videoseal = {key: value for key, value in videoseal.items() if key[1] in expected_uids}
    statuses = {
        "qwen3": audit_status(
            paths["qwen_status"],
            ARM_CONFIG["qwen3"],
            shared_invalid_count=shared_invalid_count,
        ),
        "videoseal": audit_status(
            paths["videoseal_status"],
            ARM_CONFIG["videoseal"],
            shared_invalid_count=shared_invalid_count,
        ),
    }
    shared_invalid_validation = {
        "qwen3": validate_shared_invalid_rows(
            qwen,
            qwen_attempts,
            expected,
            allowed_uids=shared_invalid_uids,
            minimum_attempts=int(
                shared_invalid_policy["minimum_failed_attempts_per_arm"]
            ),
        ),
        "videoseal": validate_shared_invalid_rows(
            videoseal,
            videoseal_attempts,
            expected,
            allowed_uids=shared_invalid_uids,
            minimum_attempts=int(
                shared_invalid_policy["minimum_failed_attempts_per_arm"]
            ),
        ),
    }
    strict_terminal_ready = bool(
        len(uid_order) == EXPECTED_TOTAL
        and len(expected) == EXPECTED_TOTAL
        and all(status["terminal"] for status in statuses.values())
        and len(qwen) == EXPECTED_TOTAL
        and len(videoseal) == EXPECTED_TOTAL
    )
    terminal_ready = bool(
        len(uid_order) == EXPECTED_TOTAL
        and len(expected) == EXPECTED_TOTAL
        and all(status["scoring_terminal"] for status in statuses.values())
        and len(qwen) == EXPECTED_TOTAL
        and len(videoseal) == EXPECTED_TOTAL
        and not shared_invalid_policy["errors"]
        and all(value["passed"] for value in shared_invalid_validation.values())
    )
    if not terminal_ready and not args.allow_incomplete_integrity_check:
        raise SystemExit("both arms must be terminal; use --allow-incomplete-integrity-check for a label-free structural check")

    scope = (
        expected
        if terminal_ready
        else partial_completed_scope(expected, qwen, videoseal)
    )
    shared_invalid_keys = {key for key in expected if key[1] in shared_invalid_uids}
    qwen_audit = audit_arm(
        qwen if terminal_ready else {k: qwen[k] for k in scope},
        scope,
        manifest,
        paths["qwen_run_root"],
        shared_invalid_keys=shared_invalid_keys & scope,
    )
    videoseal_audit = audit_arm(
        videoseal if terminal_ready else {k: videoseal[k] for k in scope},
        scope,
        manifest,
        paths["videoseal_run_root"],
        shared_invalid_keys=shared_invalid_keys & scope,
    )
    traces = {
        "qwen3": audit_trace(
            paths["qwen_trace"],
            planner_model="Qwen3-8B",
            planner_endpoint="http://127.0.0.1:18107/v1",
            inspector_endpoint="http://127.0.0.1:18105/v1",
        ),
        "videoseal": audit_trace(
            paths["videoseal_trace"],
            planner_model="VideoSEAL_8B",
            planner_endpoint="http://127.0.0.1:18104/v1",
            inspector_endpoint="http://127.0.0.1:18103/v1",
        ),
    }
    static_errors: list[dict[str, Any]] = []
    static_errors.extend(shared_invalid_policy["errors"])
    if uid_hash != EXPECTED_UID_SHA256:
        static_errors.append({"reason": "uid_hash_mismatch", "expected": EXPECTED_UID_SHA256, "actual": uid_hash})
    if dataset_hash != EXPECTED_DATASET_SHA256:
        static_errors.append({"reason": "dataset_hash_mismatch", "expected": EXPECTED_DATASET_SHA256, "actual": dataset_hash})
    if len(uid_order) != EXPECTED_TOTAL or len(expected) != EXPECTED_TOTAL:
        static_errors.append(
            {
                "reason": "population_count_mismatch",
                "uid_file": len(uid_order),
                "manifest_matches": len(expected),
            }
        )
    comparison = paired_comparison(qwen, videoseal, manifest, expected) if terminal_ready else None
    budget = None
    if terminal_ready:
        q_calls = sum(qwen_audit["tool_calls"].values())
        v_calls = sum(videoseal_audit["tool_calls"].values())
        budget = {
            "configured_limits_equal": all(
                statuses["qwen3"]["config"].get(field) == statuses["videoseal"]["config"].get(field)
                for field in EXPECTED_SHARED_CONFIG
            ),
            "qwen3_total_steps": qwen_audit["step_budget"]["total"],
            "videoseal_total_steps": videoseal_audit["step_budget"]["total"],
            "qwen3_total_tool_calls": q_calls,
            "videoseal_total_tool_calls": v_calls,
            "tool_call_delta_percent_vs_videoseal": 100.0 * (q_calls - v_calls) / v_calls if v_calls else None,
        }
    integrity_passed = bool(
        terminal_ready
        and not static_errors
        and qwen_parse_errors == 0
        and videoseal_parse_errors == 0
        and all(not status["errors"] for status in statuses.values())
        and all(value["passed"] for value in shared_invalid_validation.values())
        and qwen_audit["passed"]
        and videoseal_audit["passed"]
        and all(trace["passed"] for trace in traces.values())
        and comparison is not None
        and comparison["n"] == EXPECTED_TOTAL
        and bool(budget and budget["configured_limits_equal"])
    )
    partial_integrity_passed = bool(
        not static_errors
        and qwen_parse_errors == 0
        and videoseal_parse_errors == 0
        and all(not status["errors"] for status in statuses.values())
        and qwen_audit["passed"]
        and videoseal_audit["passed"]
        and all(trace["passed"] for trace in traces.values())
    )
    report = {
        "schema_version": 1,
        "evaluation": "qwen3_vs_videoseal_no_time_full1549_terminal_audit_v1",
        "terminal_ready": terminal_ready,
        "strict_terminal_ready": strict_terminal_ready,
        "shared_invalid_terminal_ready": terminal_ready and not strict_terminal_ready,
        "partial_integrity_only": not terminal_ready,
        "comparison": comparison,
        "budget": budget,
        "integrity": {
            "passed": integrity_passed,
            "partial_integrity_passed": partial_integrity_passed,
            "static_errors": static_errors,
            "uid_sha256": uid_hash,
            "dataset_sha256": dataset_hash,
            "registered_unique_uids": len(uid_order),
            "manifest_keys": len(expected),
            "paired_scope": len(scope),
            "qwen_parse_errors": qwen_parse_errors,
            "videoseal_parse_errors": videoseal_parse_errors,
            "qwen_attempts": sum(qwen_attempts.values()),
            "videoseal_attempts": sum(videoseal_attempts.values()),
            "qwen_duplicate_attempts": sum(max(0, count - 1) for count in qwen_attempts.values()),
            "videoseal_duplicate_attempts": sum(max(0, count - 1) for count in videoseal_attempts.values()),
            "statuses": statuses,
            "shared_invalid_validation": {
                label: {
                    "attempt_counts": value["attempt_counts"],
                    "errors": value["errors"],
                    "passed": value["passed"],
                }
                for label, value in shared_invalid_validation.items()
            },
            "qwen3": qwen_audit,
            "videoseal": videoseal_audit,
            "online_traces": traces,
        },
        "policy": {
            "candidate_models": ["Qwen3-8B", EXPECTED_INSPECTOR],
            "candidate_forbidden_components_used": [],
            "time_reference_used_online": False,
            "gold_used_only_after_terminal_readiness": terminal_ready,
            "same_caption_bm25_index": True,
            "same_runner_prompt_and_budget": True,
            "shared_invalid": {
                "path": shared_invalid_policy["path"],
                "sha256": shared_invalid_policy["sha256"],
                "uids": sorted(shared_invalid_uids),
                "minimum_failed_attempts_per_arm": shared_invalid_policy[
                    "minimum_failed_attempts_per_arm"
                ],
                "counted_as_incorrect_without_prediction_imputation": True,
            },
        },
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "terminal_ready": terminal_ready,
                "strict_terminal_ready": strict_terminal_ready,
                "partial_integrity_passed": partial_integrity_passed,
                "integrity_passed": integrity_passed,
                "paired_scope": len(scope),
                "comparison": comparison,
                "output": str(paths["output"]),
            },
            ensure_ascii=False,
        )
    )
    if terminal_ready:
        return 0 if integrity_passed else 2
    return 3 if partial_integrity_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
