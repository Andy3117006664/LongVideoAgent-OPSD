#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARMS = ("causal_orbit", "semantic_control", "hash_sham")


def summarize(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    failures = [row for row in rows if row.get("status") != "ok"]
    report: dict[str, Any] = {
        "schema_version": "causal-evidence-orbits-audit-v1",
        "coverage": {"expected": expected, "valid": len(valid), "failed": len(failures)},
        "failures": failures,
        "arms": {},
    }
    incumbent_correct = sum(
        row["incumbent_prediction"] == row["ground_truth_for_offline_reducer_only"]
        for row in valid
    )
    report["incumbent"] = {"correct_valid": incumbent_correct, "correct_strict": incumbent_correct}
    for arm in ARMS:
        correct = sum(
            row["arms"][arm]["prediction"] == row["ground_truth_for_offline_reducer_only"]
            for row in valid
        )
        fixed = sum(
            row["incumbent_prediction"] != row["ground_truth_for_offline_reducer_only"]
            and row["arms"][arm]["prediction"] == row["ground_truth_for_offline_reducer_only"]
            for row in valid
        )
        broken = sum(
            row["incumbent_prediction"] == row["ground_truth_for_offline_reducer_only"]
            and row["arms"][arm]["prediction"] != row["ground_truth_for_offline_reducer_only"]
            for row in valid
        )
        report["arms"][arm] = {
            "correct_valid": correct,
            "correct_strict": correct,
            "fixed": fixed,
            "broken": broken,
            "net": fixed - broken,
            "overrides": sum(bool(row["arms"][arm]["override_allowed"]) for row in valid),
        }
    candidate = report["arms"]["causal_orbit"]
    report["promotion_gate"] = {
        "all_valid": len(valid) == expected,
        "candidate_vs_incumbent_at_least_plus_2": candidate["correct_strict"] >= incumbent_correct + 2,
        "candidate_vs_semantic_at_least_plus_1": candidate["correct_strict"] >= report["arms"]["semantic_control"]["correct_strict"] + 1,
        "candidate_vs_sham_at_least_plus_1": candidate["correct_strict"] >= report["arms"]["hash_sham"]["correct_strict"] + 1,
        "protection": candidate["broken"] == 0 or (candidate["fixed"] >= 3 and candidate["broken"] <= 1),
    }
    report["promoted"] = all(report["promotion_gate"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=12)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = summarize(rows, args.expected)
    target = args.input.with_suffix(".summary.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["coverage"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

