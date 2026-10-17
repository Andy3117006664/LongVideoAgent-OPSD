#!/usr/bin/env python3
"""Offline audit for the public-Qwen planner-swap pilot.

The candidate is run with the VideoSEAL decoupled controller, BM25 retrieval,
and the public Qwen2.5-VL inspector. This script reads answers only after the
run is complete and keeps offline trajectory metadata separate from online
request traces.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_ONLINE = ("time_reference", "ground_truth", "groundtruth", "reference_span", "gold_answer")
EXPECTED_PLANNER = "Qwen3-8B"
EXPECTED_INSPECTOR = "Qwen2.5-VL-7B-Instruct"
EXPECTED_PLANNER_ENDPOINT = "http://127.0.0.1:18106/v1"
EXPECTED_INSPECTOR_ENDPOINT = "http://127.0.0.1:18101/v1"


def project_path(value: Path) -> Path:
    candidate = value if value.is_absolute() else ROOT / value
    return candidate.resolve()


def answer_key(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if len(text) == 1:
        return text
    match = re.match(r"^[\[(]?([a-z])[\]).:：\s]", text)
    return match.group(1) if match else text


def load_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    table = pq.read_table(path, columns=["extra_info"])
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in table["extra_info"].to_pylist():
        info = dict(raw or {})
        key = (str(info.get("video_id") or "").strip(), str(info.get("qa_uid") or "").strip())
        if all(key):
            output[key] = {
                "ground_truth": str(info.get("ground_truth") or "").strip(),
                "video_id": key[0],
                "uid": key[1],
            }
    return output


def load_latest(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[tuple[str, str]], int]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    attempts: Counter[tuple[str, str]] = Counter()
    parse_errors = 0
    if not path.is_file():
        return latest, attempts, 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(row, Mapping):
            parse_errors += 1
            continue
        key = (str(row.get("video_id") or row.get("video") or "").strip(), str(row.get("uid") or row.get("qa_uid") or "").strip())
        if all(key):
            attempts[key] += 1
            latest[key] = dict(row)
    return latest, attempts, parse_errors


def exact_mcnemar(fixed: int, broken: int) -> float:
    discordant = fixed + broken
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(fixed, broken) + 1)) / (2.0 ** discordant)
    return min(1.0, 2.0 * tail)


def scan_online_trace(
    path: Path,
    *,
    expected_planner: str = EXPECTED_PLANNER,
    expected_inspector: str = EXPECTED_INSPECTOR,
    expected_planner_endpoint: str = EXPECTED_PLANNER_ENDPOINT,
    expected_inspector_endpoint: str = EXPECTED_INSPECTOR_ENDPOINT,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    models: Counter[str] = Counter()
    urls: Counter[str] = Counter()
    tools_present: Counter[str] = Counter()
    rows = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        rows += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "reason": f"json:{exc}"})
            continue
        models[str(row.get("model") or "")] += 1
        urls[str(row.get("url") or "")] += 1
        tools_present[str(row.get("tools_present"))] += 1
        lowered = raw.casefold()
        for token in FORBIDDEN_ONLINE:
            if token in lowered:
                errors.append({"line": line_number, "reason": f"forbidden_online_token:{token}"})
        if row.get("event") != "request":
            errors.append({"line": line_number, "reason": "unexpected_event"})
        if row.get("tools_present") is not False or row.get("tool_choice_present") is not False:
            errors.append({"line": line_number, "reason": "tool_payload_present"})
    if models and set(models) != {expected_planner, expected_inspector}:
        errors.append({"reason": "unexpected_model_set", "models": dict(models)})
    if urls and set(urls) != {
        expected_planner_endpoint.rstrip("/") + "/chat/completions",
        expected_inspector_endpoint.rstrip("/") + "/chat/completions",
    }:
        errors.append({"reason": "unexpected_endpoint_set", "urls": dict(urls)})
    return {
        "rows": rows,
        "models": dict(models),
        "urls": dict(urls),
        "tools_present": dict(tools_present),
        "errors": errors,
        "passed": bool(rows) and not errors,
    }


def audit_candidate(
    candidate: Mapping[tuple[str, str], Mapping[str, Any]],
    expected: set[tuple[str, str]],
    run_root: Path,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    missing = sorted(expected - set(candidate))
    extra = sorted(set(candidate) - expected)
    if missing:
        errors.append({"reason": "missing_candidate_keys", "keys": missing})
    if extra:
        errors.append({"reason": "extra_candidate_keys", "keys": extra})
    trajectory_count = 0
    for key in sorted(expected & set(candidate)):
        row = candidate[key]
        uid = key[1]
        if row.get("state") != "completed" or row.get("error"):
            errors.append({"uid": uid, "reason": "candidate_not_completed", "state": row.get("state"), "error": row.get("error")})
        if not str(row.get("prediction") or "").strip():
            errors.append({"uid": uid, "reason": "empty_prediction"})
        trajectory = Path(str(row.get("trajectory_path") or ""))
        if not trajectory.is_absolute():
            trajectory = run_root / trajectory
        try:
            trajectory.resolve().relative_to(run_root.resolve())
        except ValueError:
            errors.append({"uid": uid, "reason": "trajectory_outside_run_root", "path": str(trajectory)})
            continue
        if not trajectory.is_file():
            errors.append({"uid": uid, "reason": "trajectory_missing", "path": str(trajectory)})
            continue
        trajectory_count += 1
        try:
            payload = json.loads(trajectory.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"uid": uid, "reason": f"trajectory_unreadable:{exc}"})
            continue
        steps = payload.get("steps") if isinstance(payload, Mapping) else None
        if not isinstance(steps, list) or not steps:
            errors.append({"uid": uid, "reason": "trajectory_steps_missing"})
        else:
            online_text = json.dumps(
                [
                    {field: step.get(field) for field in ("chat_prompt", "model_response", "action", "observation")}
                    for step in steps
                    if isinstance(step, Mapping)
                ],
                ensure_ascii=False,
            ).casefold()
            for token in FORBIDDEN_ONLINE:
                if token in online_text:
                    errors.append({"uid": uid, "reason": f"forbidden_trajectory_online_token:{token}"})
            if len(steps) > 17:
                errors.append({"uid": uid, "reason": "step_budget_exceeded", "steps": len(steps)})
    return {
        "expected": len(expected),
        "candidate_unique": len(candidate),
        "missing": missing,
        "extra": extra,
        "trajectory_readable": trajectory_count,
        "errors": errors,
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--uids", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--videoseal", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--expected-planner", default=EXPECTED_PLANNER)
    parser.add_argument("--expected-inspector", default=EXPECTED_INSPECTOR)
    parser.add_argument("--expected-planner-endpoint", default=EXPECTED_PLANNER_ENDPOINT)
    parser.add_argument("--expected-inspector-endpoint", default=EXPECTED_INSPECTOR_ENDPOINT)
    parser.add_argument("--expected-planner-max-tokens", type=int)
    parser.add_argument("--expected-planner-context-limit", type=int)
    parser.add_argument("--expected-history-max-blocks", type=int)
    parser.add_argument("--expected-history-max-chars", type=int)
    args = parser.parse_args()
    parquet = project_path(args.parquet)
    uid_file = project_path(args.uids)
    candidate_path = project_path(args.candidate)
    baseline_path = project_path(args.videoseal)
    trace_path = project_path(args.trace)
    run_root = project_path(args.run_root)
    manifest = load_manifest(parquet)
    expected_uids = {line.strip() for line in uid_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    expected = {key for key in manifest if key[1] in expected_uids}
    candidate, candidate_attempts, candidate_parse_errors = load_latest(candidate_path)
    baseline, baseline_attempts, baseline_parse_errors = load_latest(baseline_path)
    integrity = audit_candidate(candidate, expected, run_root)
    trace = scan_online_trace(
        trace_path,
        expected_planner=args.expected_planner,
        expected_inspector=args.expected_inspector,
        expected_planner_endpoint=args.expected_planner_endpoint,
        expected_inspector_endpoint=args.expected_inspector_endpoint,
    )
    status_errors: list[dict[str, Any]] = []
    status_snapshot: dict[str, Any] = {}
    if args.status is not None:
        status_path = project_path(args.status)
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            status = {}
            status_errors.append({"reason": f"status_unreadable:{exc}"})
        status_snapshot = {
            key: status.get(key)
            for key in ("state", "total", "completed", "failed", "remaining", "config")
        }
        for field, expected_value in (
            ("state", "complete"),
            ("total", len(expected)),
            ("completed", len(expected)),
            ("failed", 0),
            ("remaining", 0),
        ):
            if status.get(field) != expected_value:
                status_errors.append({
                    "reason": "status_mismatch",
                    "field": field,
                    "expected": expected_value,
                    "actual": status.get(field),
                })
        expected_config = {
            "planner_model": args.expected_planner,
            "inspector_model": args.expected_inspector,
            "planner_endpoint": args.expected_planner_endpoint,
            "inspector_endpoint": args.expected_inspector_endpoint,
        }
        for field, value in (
            ("planner_max_tokens", args.expected_planner_max_tokens),
            ("planner_context_limit", args.expected_planner_context_limit),
            ("legacy_history_max_blocks", args.expected_history_max_blocks),
            ("legacy_history_max_chars", args.expected_history_max_chars),
        ):
            if value is not None:
                expected_config[field] = value
        actual_config = status.get("config") or {}
        for field, expected_value in expected_config.items():
            if actual_config.get(field) != expected_value:
                status_errors.append({
                    "reason": "config_mismatch",
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_config.get(field),
                })
    paired = []
    missing_baseline = []
    for key in sorted(expected):
        c = candidate.get(key)
        b = baseline.get(key)
        if b is None or not str(b.get("prediction") or "").strip():
            missing_baseline.append(key)
            continue
        truth = manifest[key]["ground_truth"]
        cp = answer_key((c or {}).get("prediction"))
        bp = answer_key(b.get("prediction"))
        gt = answer_key(truth)
        paired.append({"key": list(key), "candidate_correct": cp == gt, "videoseal_correct": bp == gt})
    candidate_correct = sum(int(x["candidate_correct"]) for x in paired)
    baseline_correct = sum(int(x["videoseal_correct"]) for x in paired)
    fixed = sum(int(x["candidate_correct"] and not x["videoseal_correct"]) for x in paired)
    broken = sum(int(x["videoseal_correct"] and not x["candidate_correct"]) for x in paired)
    n = len(paired)
    comparison = {
        "n": n,
        "candidate_correct": candidate_correct,
        "videoseal_correct": baseline_correct,
        "candidate_accuracy_percent": 100.0 * candidate_correct / n if n else None,
        "videoseal_accuracy_percent": 100.0 * baseline_correct / n if n else None,
        "delta_percentage_points": 100.0 * (candidate_correct - baseline_correct) / n if n else None,
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "exact_mcnemar_two_sided": exact_mcnemar(fixed, broken),
        "missing_videoseal": [list(key) for key in missing_baseline],
    }
    integrity_pass = candidate_parse_errors == 0 and baseline_parse_errors == 0 and integrity["passed"] and trace["passed"] and not missing_baseline and not status_errors
    accepted = bool(
        integrity_pass
        and n >= len(expected)
        and candidate_correct > baseline_correct
        and fixed > broken
        and comparison["exact_mcnemar_two_sided"] <= 0.10
    )
    report = {
        "schema_version": 1,
        "evaluation": "qwen3_planner_swap_pilot_offline_audit_v1",
        "policy": {
            "online_models": {"planner": args.expected_planner, "inspector": args.expected_inspector},
            "online_endpoints": {"planner": args.expected_planner_endpoint, "inspector": args.expected_inspector_endpoint},
            "online_privileged_fields": [],
            "offline_fields_used_for_comparison": ["ground_truth"],
            "primary_comparator": "latest canonicalized VideoSEAL BM25 replay per UID",
        },
        "integrity": {
            "candidate_parse_errors": candidate_parse_errors,
            "videoseal_parse_errors": baseline_parse_errors,
            "candidate_attempts": sum(candidate_attempts.values()),
            "videoseal_attempts": sum(baseline_attempts.values()),
            "candidate": integrity,
            "online_trace": trace,
            "status_errors": status_errors,
            "status_snapshot": status_snapshot,
            "passed": integrity_pass,
        },
        "comparison_vs_videoseal": comparison,
        "accepted_for_full_run": accepted,
        "decision": "promote_only_if_paired_gate_passes" if accepted else "retain_pilot_without_full_run",
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integrity_passed": integrity_pass, "n": n, "candidate": candidate_correct, "videoseal": baseline_correct, "fixed": fixed, "broken": broken, "net": fixed - broken, "p": comparison["exact_mcnemar_two_sided"], "accepted_for_full_run": accepted, "output": str(output)}, ensure_ascii=False))
    return 0 if integrity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
