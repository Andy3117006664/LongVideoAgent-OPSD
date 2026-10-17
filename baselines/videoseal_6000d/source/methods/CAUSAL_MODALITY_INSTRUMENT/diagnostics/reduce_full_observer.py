#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--baseline-correct", type=int, default=793)
    parser.add_argument("--benchmark-size", type=int, default=1549)
    parser.add_argument("--funnel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = {(str(item["video_id"]), str(item["uid"])) for item in json.loads(args.protocol.read_text())["items"]}
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(args.shards)):
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = []
    for row in rows:
        key = (str(row.get("video_id")), str(row.get("uid")))
        if key in keyed:
            duplicates.append(key)
        keyed[key] = row
    valid = [keyed[key] for key in expected if keyed.get(key, {}).get("status") == "ok"]
    fixed = sum(r["incumbent"] != r["ground_truth_for_offline_reducer_only"] and r["prediction"] == r["ground_truth_for_offline_reducer_only"] for r in valid)
    broken = sum(r["incumbent"] == r["ground_truth_for_offline_reducer_only"] and r["prediction"] != r["ground_truth_for_offline_reducer_only"] for r in valid)
    report: dict[str, Any] = {
        "schema_version": "full-observer-reducer-v1",
        "coverage": {
            "expected": len(expected),
            "rows": len(rows),
            "unique": len(keyed),
            "valid": len(valid),
            "missing": len(expected - set(keyed)),
            "duplicates": len(duplicates),
        },
        "evaluated_subset": {
            "incumbent_correct": sum(r["incumbent"] == r["ground_truth_for_offline_reducer_only"] for r in valid),
            "observer_correct": sum(r["prediction"] == r["ground_truth_for_offline_reducer_only"] for r in valid),
            "fixed": fixed,
            "broken": broken,
            "net": fixed - broken,
            "changed": sum(r["prediction"] != r["incumbent"] for r in valid),
        },
        "full_overlay": {
            "baseline_correct": args.baseline_correct,
            "candidate_correct": args.baseline_correct + fixed - broken,
            "denominator": args.benchmark_size,
            "accuracy": (args.baseline_correct + fixed - broken) / args.benchmark_size,
        },
    }
    if args.funnel:
        funnel = {(str(r["video_id"]), str(r["uid"])): r for r in json.loads(args.funnel.read_text())["rows"]}
        target = [r for r in valid if funnel.get((str(r["video_id"]), str(r["uid"])), {}).get("terminal_stage") == "target_frames_reached_but_inspector_verdict_not_correct"]
        report["target_frame_failure_subset"] = {
            "n": len(target),
            "observer_correct": sum(r["prediction"] == r["ground_truth_for_offline_reducer_only"] for r in target),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["coverage"]["missing"] == 0 and not duplicates else 2


if __name__ == "__main__":
    raise SystemExit(main())
