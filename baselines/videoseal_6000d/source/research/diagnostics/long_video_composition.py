#!/usr/bin/env python3
"""Audit long-video agent traces for distributed-evidence composition behavior.

This is intentionally a diagnostic, not a reward function or a claim that a
trajectory is a compositional failure.  It uses only information already
emitted by the runner: tool actions, inspected spans, inspector verdicts, and
the final answer.  The output separates observable behavior from heuristic
``composition_candidate`` flags that require manual or judge-based review.

The script can run on the evaluation host where trajectory_path values point
to local files.  It does not modify the run directory.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)")
SHORT_TIME_RE = re.compile(r"(?P<m>\d+):(?P<s>\d{2}(?:\.\d+)?)")
RELATION_RE = re.compile(
    r"\b(after|before|between|first|last|order|respectively|how many times|"
    r"what happens|what happened|most likely|why|entire video|overall|"
    r"from front to back|at the beginning|at the end)\b",
    re.IGNORECASE,
)
QUESTION_TIME_RE = re.compile(
    r"(?P<a>\d{1,3}:\d{2}(?::\d{2})?)\s*(?:-|–|—|to|and)\s*"
    r"(?P<b>\d{1,3}:\d{2}(?::\d{2})?)",
    re.IGNORECASE,
)
QUESTION_POINT_TIME_RE = re.compile(r"\b(?:at|around|near)\s+(?P<t>\d{1,3}:\d{2}(?::\d{2})?)\b", re.IGNORECASE)


def parse_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = TIME_RE.search(text)
    if match:
        return 3600 * int(match.group("h")) + 60 * int(match.group("m")) + float(match.group("s"))
    match = SHORT_TIME_RE.search(text)
    if match:
        return 60 * int(match.group("m")) + float(match.group("s"))
    try:
        return float(text)
    except ValueError:
        return None


def parse_question_clock(value: str) -> float | None:
    """Parse LVBench question clocks; two fields are minute:second."""
    parts = value.split(":")
    try:
        if len(parts) == 2:
            return 60 * int(parts[0]) + float(parts[1])
        if len(parts) == 3:
            return 3600 * int(parts[0]) + 60 * int(parts[1]) + float(parts[2])
    except ValueError:
        return None
    return None


def question_time_targets(question: str) -> list[tuple[float, float]]:
    targets: list[tuple[float, float]] = []
    for match in QUESTION_TIME_RE.finditer(question):
        start, end = parse_question_clock(match.group("a")), parse_question_clock(match.group("b"))
        if start is not None and end is not None and end >= start:
            targets.append((start, end))
    for match in QUESTION_POINT_TIME_RE.finditer(question):
        point = parse_question_clock(match.group("t"))
        if point is not None:
            targets.append((point, point))
    return unique(targets)


def span_from_mapping(value: dict[str, Any]) -> tuple[float, float] | None:
    start = value.get("start_time", value.get("start"))
    end = value.get("end_time", value.get("end"))
    start_s, end_s = parse_seconds(start), parse_seconds(end)
    if start_s is None or end_s is None or end_s <= start_s:
        return None
    return (round(start_s, 3), round(end_s, 3))


def extract_spans(value: Any) -> list[tuple[float, float]]:
    """Extract structured spans without scraping free-form prose."""
    found: list[tuple[float, float]] = []
    if isinstance(value, dict):
        span = span_from_mapping(value)
        if span is not None:
            found.append(span)
        for key in ("spans", "useful_spans", "evidence_spans", "time_spans", "intervals"):
            if key in value:
                found.extend(extract_spans(value[key]))
    elif isinstance(value, list):
        for item in value:
            found.extend(extract_spans(item))
    return found


def unique(items: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def answer_key(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if len(text) == 1:
        return text
    match = re.match(r"^[\[(]?([a-z])[\]).:：\s]", text)
    return match.group(1) if match else text


def latest_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        key = (str(row.get("video_id", "")), str(row.get("uid", "")))
        if all(key):
            latest[key] = row
    return latest


def load_trajectory(row: dict[str, Any], run_root: Path | None) -> dict[str, Any] | None:
    raw_path = str(row.get("trajectory_path") or "")
    candidates = [Path(raw_path)] if raw_path else []
    if run_root is not None:
        video_id, uid = str(row.get("video_id", "")), str(row.get("uid", ""))
        candidates.extend(sorted((run_root / video_id).glob("*/trajectory.json"), reverse=True))
        candidates.extend(sorted((run_root / video_id.lstrip("-/")).glob("*/trajectory.json"), reverse=True))
    for candidate in unique(candidates):
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            expected_video = str(row.get("video_id", "")).strip()
            expected_uid = str(row.get("uid", "")).strip()
            actual_video = str(value.get("video_id", "")).strip()
            actual_uid = str(value.get("uid", "")).strip()
            # Never substitute another question from the same video when a
            # result row lacks a usable trajectory_path (e.g. recovered pred
            # files).  This guard is essential for outcome-linked audits.
            if expected_video and actual_video and expected_video != actual_video:
                continue
            if expected_uid and actual_uid and expected_uid != actual_uid:
                continue
            if expected_uid and not actual_uid:
                continue
            return value
    return None


def inspect_record(row: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    retrieve_calls = 0
    inspect_calls = 0
    inspect_spans: list[tuple[float, float]] = []
    inspect_sets: list[tuple[tuple[float, float], ...]] = []
    repeated_query = 0
    query_counts: Counter[str] = Counter()
    search_more = 0
    answer_verdicts: list[str] = []
    answer_letters: list[str] = []
    answer_span_counts: list[int] = []
    output_texts: list[str] = []
    events: list[dict[str, Any]] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        name = str(action.get("name") or step.get("tool_name") or "")
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        if name == "visual_retrieve":
            retrieve_calls += 1
            query = str(arguments.get("query") or "").strip().casefold()
            if query:
                repeated_query += int(query_counts[query] > 0)
                query_counts[query] += 1
            retrieve_output = (step.get("observation") or {}).get("output") if isinstance(step.get("observation"), dict) else None
            events.append({"kind": "retrieve", "spans": unique(extract_spans(retrieve_output))})
        if name != "visual_inspect":
            continue
        inspect_calls += 1
        observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        output = observation.get("output")
        spans = extract_spans(arguments)
        spans.extend(extract_spans(output))
        spans.extend(extract_spans(observation.get("metadata")))
        current = tuple(unique(spans))
        inspect_sets.append(current)
        inspect_spans.extend(current)
        text = json.dumps(output, ensure_ascii=False) if output is not None else ""
        output_texts.append(text)
        if "SEARCH_MORE" in text.upper():
            search_more += 1
            answer_verdicts.append("search_more")
            verdict = "search_more"
        elif re.search(r"['\"]?(?:answer|final_answer)['\"]?\s*[:=]\s*['\"]?\s*[A-D]", text, re.I):
            answer_verdicts.append("answer")
            answer_span_counts.append(len(current))
            match = re.search(r"['\"]?(?:answer|final_answer)['\"]?\s*[:=]\s*['\"]?\s*([A-D])", text, re.I)
            if match:
                answer_letters.append(match.group(1).upper())
            verdict = "answer"
        else:
            answer_verdicts.append("other")
            verdict = "other"
        events.append({"kind": "inspect", "spans": list(current), "verdict": verdict})

    unique_spans = unique(inspect_spans)
    duplicate_spans = len(inspect_spans) - len(unique_spans)
    repeated_inspection = sum(
        int(inspect_sets[i] and inspect_sets[i] == inspect_sets[i - 1])
        for i in range(1, len(inspect_sets))
    )
    search_more_transitions = 0
    stagnant_search_more_transitions = 0
    for i in range(1, len(inspect_sets)):
        previous_verdict = answer_verdicts[i - 1] if i - 1 < len(answer_verdicts) else "other"
        if previous_verdict != "search_more":
            continue
        search_more_transitions += 1
        previous_set, current_set = set(inspect_sets[i - 1]), set(inspect_sets[i])
        if current_set and current_set.issubset(previous_set):
            stagnant_search_more_transitions += 1
    repair_opportunities = 0
    missed_repair_opportunities = 0
    for i, event in enumerate(events):
        if event.get("kind") != "inspect" or event.get("verdict") != "search_more":
            continue
        previous_set = set(event.get("spans") or [])
        retrieved_after_failure: set[tuple[float, float]] = set()
        next_inspect: dict[str, Any] | None = None
        for later in events[i + 1 :]:
            if later.get("kind") == "retrieve":
                retrieved_after_failure.update(later.get("spans") or [])
            elif later.get("kind") == "inspect":
                next_inspect = later
                break
        if next_inspect is None:
            continue
        available_new = retrieved_after_failure - previous_set
        if available_new:
            repair_opportunities += 1
            chosen_new = set(next_inspect.get("spans") or []) - previous_set
            if not chosen_new:
                missed_repair_opportunities += 1
    final_output = output_texts[-1] if output_texts else ""
    final_search_more = bool(final_output and "SEARCH_MORE" in final_output.upper())
    correct = bool(row.get("correct"))
    question = str(row.get("question", trajectory.get("question", "")))
    multi_hop_language = bool(RELATION_RE.search(question))
    requested_targets = question_time_targets(question)
    requested_time_hit = None
    if requested_targets:
        def overlaps(span: tuple[float, float], target: tuple[float, float]) -> bool:
            if target[0] == target[1]:
                return span[0] <= target[0] <= span[1]
            return min(span[1], target[1]) > max(span[0], target[0])
        requested_time_hit = any(overlaps(span, target) for span in unique_spans for target in requested_targets)
    multi_span = any(len(item) >= 2 for item in inspect_sets)
    long_trace = len(steps) >= 8 or inspect_calls >= 4
    conflicting_answer_reports = len(set(answer_letters)) > 1
    full_video_inspect = any((span[1] - span[0]) >= 300 for span in unique_spans)
    answer_after_search_more = bool(answer_letters and search_more > 0)
    answer_on_multispan = any(count >= 2 for count in answer_span_counts)

    # This flag is deliberately conservative in wording: it identifies cases
    # worth reviewing, not labels them as proof of relation failure.
    composition_candidate = bool(
        not correct
        and multi_span
        and inspect_calls >= 2
        and len(unique_spans) >= 2
        and not final_search_more
    )
    return {
        "video_id": str(row.get("video_id", trajectory.get("video_id", ""))),
        "uid": str(row.get("uid", trajectory.get("uid", ""))),
        "question": str(row.get("question", trajectory.get("question", ""))),
        "question_type": str(row.get("question_type", "unknown")),
        "prediction": str(row.get("prediction", "")),
        "ground_truth": str(row.get("ground_truth", trajectory.get("groundtruth", ""))),
        "correct": correct,
        "steps": len(steps),
        "retrieve_calls": retrieve_calls,
        "inspect_calls": inspect_calls,
        "unique_inspected_spans": len(unique_spans),
        "total_inspected_spans": len(inspect_spans),
        "duplicate_span_count": duplicate_spans,
        "repeated_inspection_count": repeated_inspection,
        "search_more_transitions": search_more_transitions,
        "stagnant_search_more_transitions": stagnant_search_more_transitions,
        "repair_opportunities": repair_opportunities,
        "missed_repair_opportunities": missed_repair_opportunities,
        "missed_repair_rate": (
            missed_repair_opportunities / repair_opportunities if repair_opportunities else None
        ),
        "search_more_repair_rate": (
            (search_more_transitions - stagnant_search_more_transitions) / search_more_transitions
            if search_more_transitions else None
        ),
        "repeated_retrieval_count": repeated_query,
        "search_more_count": search_more,
        "final_search_more": final_search_more,
        "answer_letters": answer_letters,
        "conflicting_answer_reports": conflicting_answer_reports,
        "full_video_inspect": full_video_inspect,
        "answer_after_search_more": answer_after_search_more,
        "answer_on_multispan": answer_on_multispan,
        "multi_span_inspection": multi_span,
        "multi_hop_language": multi_hop_language,
        "question_time_targets": requested_targets,
        "requested_time_hit": requested_time_hit,
        "requested_time_miss": bool(requested_targets) and not bool(requested_time_hit),
        "long_trace": long_trace,
        "composition_candidate": composition_candidate,
    }


def rate(rows: list[dict[str, Any]], predicate) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict[str, Any]], missing: int, malformed: int) -> dict[str, Any]:
    def group_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(group),
            "accuracy": rate(group, lambda x: x["correct"]),
            "multi_span_rate": rate(group, lambda x: x["multi_span_inspection"]),
            "long_trace_rate": rate(group, lambda x: x["long_trace"]),
            "repeat_inspection_rate": rate(group, lambda x: x["repeated_inspection_count"] > 0),
            "stagnant_search_more_rate": rate(group, lambda x: x["stagnant_search_more_transitions"] > 0),
            "mean_search_more_repair_rate": (
                sum(x["search_more_repair_rate"] for x in group if x["search_more_repair_rate"] is not None)
                / sum(x["search_more_repair_rate"] is not None for x in group)
                if any(x["search_more_repair_rate"] is not None for x in group) else None
            ),
            "repair_opportunity_count": sum(x["repair_opportunities"] for x in group),
            "missed_repair_opportunity_count": sum(x["missed_repair_opportunities"] for x in group),
            "missed_repair_rate": (
                sum(x["missed_repair_opportunities"] for x in group)
                / sum(x["repair_opportunities"] for x in group)
                if sum(x["repair_opportunities"] for x in group) else None
            ),
            "conflicting_answer_rate": rate(group, lambda x: x["conflicting_answer_reports"]),
            "full_video_inspect_rate": rate(group, lambda x: x["full_video_inspect"]),
            "answer_after_search_more_rate": rate(group, lambda x: x["answer_after_search_more"]),
            "answer_on_multispan_rate": rate(group, lambda x: x["answer_on_multispan"]),
            "composition_candidate_rate": rate(group, lambda x: x["composition_candidate"]),
            "multi_hop_language_rate": rate(group, lambda x: x["multi_hop_language"]),
            "explicit_time_rate": rate(group, lambda x: bool(x["question_time_targets"])),
            "requested_time_miss_rate": rate(group, lambda x: x["requested_time_miss"]),
            "mean_inspect_calls": (sum(x["inspect_calls"] for x in group) / len(group)) if group else 0.0,
            "mean_unique_spans": (sum(x["unique_inspected_spans"] for x in group) / len(group)) if group else 0.0,
        }

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    return {
        "schema_version": 1,
        "definition": {
            "composition_candidate": "wrong final answer after at least two multi-span inspections with at least two unique spans and no final SEARCH_MORE; requires review and is not a ground-truth label",
            "gold_evidence_used": False,
            "counterfactual_intervention": False,
        },
        "coverage": {"result_keys": len(rows), "missing_trajectories": missing, "malformed_trajectories": malformed},
        "overall": group_stats(rows),
        "correct": group_stats([x for x in rows if x["correct"]]),
        "wrong": group_stats([x for x in rows if not x["correct"]]),
        "multi_hop_language": group_stats([x for x in rows if x["multi_hop_language"]]),
        "multi_hop_wrong": group_stats([x for x in rows if x["multi_hop_language"] and not x["correct"]]),
        "non_multi_hop_language": group_stats([x for x in rows if not x["multi_hop_language"]]),
        "by_question_type": {name: group_stats(group) for name, group in sorted(by_type.items())},
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    latest = latest_results(args.results)
    rows: list[dict[str, Any]] = []
    missing = malformed = 0
    for row in latest.values():
        trajectory = load_trajectory(row, args.run_root)
        if trajectory is None:
            missing += 1
            continue
        try:
            rows.append(inspect_record(row, trajectory))
        except (TypeError, ValueError, KeyError):
            malformed += 1

    report = summarize(rows, missing, malformed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("coverage", "overall", "correct", "wrong")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
