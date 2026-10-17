#!/usr/bin/env python3
"""Outcome-linked audit for long-video QA agent trajectories.

The audit is deliberately tied to the benchmark outcome instead of using
trace length or tool-call frequency as a proxy for failure.  It asks a narrow,
observable question for every trajectory:

    Did any *visual_inspect* call produce the benchmark-correct option, and
    did the agent preserve that decision in its final answer?

This yields an outcome matrix analogous to the correctness/groundedness table
used in evidence-alignment work:

    visual verdict reached  x  final answer correct

The visual verdict is not claimed to be a human-groundedness label.  It is an
internal evidence-exposure signal that can be audited directly from the
trajectory.  A later pixel-level judge may be attached to the emitted audit
queue, but it is never silently used as a training label.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .long_video_composition import extract_spans, latest_results, load_trajectory
except ImportError:  # Running the file directly from research/diagnostics.
    from long_video_composition import extract_spans, latest_results, load_trajectory


ANSWER_LINE_RE = re.compile(
    r"(?:^|\n)\s*answer\s*:\s*"
    r"(?P<answer>SEARCH_MORE|[A-D](?:\s*(?:,|/|and)\s*[A-D])*)",
    re.IGNORECASE,
)
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
TIME_RE = re.compile(r"(?P<h>\d{1,3}):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)")
SHORT_TIME_RE = re.compile(r"(?P<m>\d{1,4}):(?P<s>\d{2}(?:\.\d+)?)")
TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{1,3}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*"
    r"(?:-|–|—|to)\s*"
    r"(?P<end>\d{1,3}:\d{2}(?::\d{2}(?:\.\d+)?)?)"
)
UNCERTAINTY_MARKERS = (
    "not visible",
    "not clear",
    "no clear",
    "cannot determine",
    "insufficient",
    "unclear",
    "search_more",
)


def parse_seconds(value: Any) -> float | None:
    """Parse LVBench time strings without depending on a specific format."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = TIME_RE.fullmatch(text)
    if match:
        return 3600 * int(match.group("h")) + 60 * int(match.group("m")) + float(match.group("s"))
    match = SHORT_TIME_RE.fullmatch(text)
    if match:
        return 60 * int(match.group("m")) + float(match.group("s"))
    try:
        return float(text)
    except ValueError:
        return None


def parse_time_reference(value: Any) -> list[tuple[float, float]]:
    """Read one or more answer-bearing intervals from LVBench metadata."""
    if isinstance(value, (list, tuple)):
        intervals: list[tuple[float, float]] = []
        for item in value:
            intervals.extend(parse_time_reference(item))
        return intervals
    text = str(value or "").strip()
    intervals: list[tuple[float, float]] = []
    for match in TIME_RANGE_RE.finditer(text):
        start = parse_seconds(match.group("start"))
        end = parse_seconds(match.group("end"))
        if start is not None and end is not None and end >= start:
            intervals.append((round(start, 3), round(end, 3)))
    return list(dict.fromkeys(intervals))


def temporal_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Compute temporal IoU for two non-empty intervals."""
    first_start, first_end = first
    second_start, second_end = second
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def max_temporal_overlap(
    inspected: Iterable[tuple[float, float]],
    target: Iterable[tuple[float, float]],
) -> float | None:
    values = [temporal_iou(left, right) for left in inspected for right in target]
    return max(values) if values else None


def load_target_spans(path: Path) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Load answer-bearing intervals from raw or VideoSEAL LVBench parquet."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("pyarrow is required when --dataset is supplied") from exc

    table = pq.read_table(path)
    result: dict[tuple[str, str], list[tuple[float, float]]] = {}
    if "extra_info" in table.column_names:
        rows = table["extra_info"].to_pylist()
    else:
        rows = [
            {name: table[name][index].as_py() for name in table.column_names}
            for index in range(table.num_rows)
        ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        video = str(row.get("video_id") or row.get("video_key") or row.get("key") or "").strip()
        uid = str(row.get("qa_uid") or row.get("uid") or "").strip()
        spans = parse_time_reference(row.get("time_reference"))
        if video and uid and spans:
            result[(video, uid)] = spans
    return result


def option_set(value: Any) -> tuple[str, ...]:
    """Normalize an MCQ answer without treating prose as an answer."""
    text = str(value or "").strip().upper()
    if not text or "SEARCH_MORE" in text:
        return ()
    # Most result rows contain a single letter.  For inspector text, the
    # caller first extracts the Answer: field, so this remains conservative.
    if len(text) == 1 and text in "ABCD":
        return (text,)
    letters = tuple(dict.fromkeys(LETTER_RE.findall(text)))
    return letters if letters and len(text) <= 32 else ()


def answer_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("answer", "")
    return str(value or "").strip()


def parse_visual_answer(value: Any) -> dict[str, Any]:
    """Parse the structured inspector answer field and retain raw evidence."""
    text = answer_text(value)
    match = ANSWER_LINE_RE.search(text)
    if match:
        raw_answer = match.group("answer").strip()
    elif text.upper() == "SEARCH_MORE":
        raw_answer = "SEARCH_MORE"
    elif len(text.strip()) <= 8 and option_set(text):
        raw_answer = text
    else:
        raw_answer = ""
    evidence_match = re.search(
        r"(?:^|\n)\s*evidence\s*:\s*(.*?)(?=\n\s*(?:missing|confidence|next search|reminder)\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    confidence_match = re.search(r"(?:^|\n)\s*confidence\s*:\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
    missing_match = re.search(
        r"(?:^|\n)\s*missing\s*:\s*(.*?)(?=\n\s*(?:next search|reminder)\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    evidence = " ".join((evidence_match.group(1) if evidence_match else "").split())
    missing = " ".join((missing_match.group(1) if missing_match else "").split())
    confidence = float(confidence_match.group(1)) if confidence_match else None
    parsed_options = option_set(raw_answer)
    decisive = bool(parsed_options)
    return {
        "raw": " ".join(text.split()),
        "answer": raw_answer.upper(),
        "options": list(parsed_options),
        "decisive": decisive,
        "evidence": evidence,
        "missing": missing,
        "confidence": confidence,
        "uncertain_language": any(marker in text.casefold() for marker in UNCERTAINTY_MARKERS),
    }


def inspect_records(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        if str(action.get("name") or "") != "visual_inspect":
            continue
        observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        output = observation.get("output") if isinstance(observation.get("output"), dict) else {}
        parsed = parse_visual_answer(output.get("answer"))
        spans = extract_spans(action.get("arguments"))
        spans.extend(extract_spans(output))
        spans.extend(extract_spans(observation.get("metadata")))
        records.append(
            {
                "step": index,
                "spans": [list(span) for span in dict.fromkeys(spans)],
                "parsed": parsed,
            }
        )
    return records


def last_action_name(trajectory: dict[str, Any]) -> str:
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        name = str(action.get("name") or step.get("tool_name") or "").strip()
        if name:
            return name
    return ""


def benchmark_correct(prediction: Any, ground_truth: Any) -> bool:
    pred, gold = option_set(prediction), option_set(ground_truth)
    return bool(pred and gold and pred == gold)


def primary_failure(
    *,
    correct: bool,
    final_visual_inspect_decisive: bool,
    final_matches_visual_verdict: bool,
    decisive: list[dict[str, Any]],
    correct_visual: list[dict[str, Any]],
    final_inspect: dict[str, Any] | None,
) -> str:
    if correct:
        return "correct_with_visual_verdict" if correct_visual else "correct_without_correct_visual_verdict"
    if not final_visual_inspect_decisive:
        return "protocol_shortcut"
    if not decisive:
        return "search_or_coverage_failure"
    if not final_matches_visual_verdict:
        return "answer_mapping_or_format_error"
    if correct_visual and final_inspect and not set(final_inspect["parsed"]["options"]) == set(correct_visual[-1]["parsed"]["options"]):
        return "conflict_resolution_error"
    if correct_visual:
        return "evidence_discarded"
    return "visual_verdict_error"


def audit_row(row: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    prediction = str(row.get("prediction") or row.get("pred") or trajectory.get("answer") or "").strip()
    ground_truth = str(row.get("ground_truth") or trajectory.get("groundtruth") or "").strip()
    correct = benchmark_correct(prediction, ground_truth)
    inspections = inspect_records(trajectory)
    decisive = [item for item in inspections if item["parsed"]["decisive"]]
    correct_visual = [item for item in decisive if set(item["parsed"]["options"]) == set(option_set(ground_truth))]
    last_inspect = inspections[-1] if inspections else None
    final_inspect = last_inspect if last_inspect and last_inspect["parsed"]["decisive"] else None
    last_decisive = decisive[-1] if decisive else None
    last_name = last_action_name(trajectory)
    final_options = option_set(prediction)
    final_visual_inspect_decisive = bool(last_name == "visual_inspect" and final_inspect is not None)
    final_matches_visual_verdict = bool(
        final_visual_inspect_decisive
        and set(final_inspect["parsed"]["options"]) == set(final_options)
    )
    last_visual_verdict_correct = bool(
        last_decisive
        and set(last_decisive["parsed"]["options"]) == set(option_set(ground_truth))
    )
    all_spans = [span for item in inspections for span in item["spans"]]
    unique_spans = list(dict.fromkeys(tuple(span) for span in all_spans))
    conflict = bool(
        decisive
        and len({tuple(item["parsed"]["options"]) for item in decisive}) > 1
    )
    failure = primary_failure(
        correct=correct,
        final_visual_inspect_decisive=final_visual_inspect_decisive,
        final_matches_visual_verdict=final_matches_visual_verdict,
        decisive=decisive,
        correct_visual=correct_visual,
        final_inspect=final_inspect,
    )
    return {
        "video_id": str(row.get("video_id") or trajectory.get("video_id") or ""),
        "uid": str(row.get("uid") or trajectory.get("uid") or ""),
        "question": str(row.get("question") or trajectory.get("question") or ""),
        "question_type": str(row.get("question_type") or "unknown"),
        "prediction": prediction,
        "ground_truth": ground_truth,
        "correct": correct,
        "steps": len(trajectory.get("steps") or []),
        "last_action": last_name,
        "protocol_compliant": final_visual_inspect_decisive and final_matches_visual_verdict,
        "final_visual_inspect_decisive": final_visual_inspect_decisive,
        "final_matches_visual_verdict": final_matches_visual_verdict,
        "inspections": len(inspections),
        "decisive_inspections": len(decisive),
        "correct_visual_verdict": bool(correct_visual),
        "correct_visual_verdict_count": len(correct_visual),
        "last_visual_verdict_correct": last_visual_verdict_correct,
        "authority_preserved": bool(
            last_decisive and set(last_decisive["parsed"]["options"]) == set(final_options)
        ),
        "first_correct_visual_step": correct_visual[0]["step"] if correct_visual else None,
        "last_correct_visual_step": correct_visual[-1]["step"] if correct_visual else None,
        "last_visual_answer": final_inspect["parsed"]["answer"] if final_inspect else "",
        "last_visual_evidence": final_inspect["parsed"]["evidence"] if final_inspect else "",
        "last_visual_missing": final_inspect["parsed"]["missing"] if final_inspect else "",
        "last_visual_confidence": final_inspect["parsed"]["confidence"] if final_inspect else None,
        "visual_conflict": conflict,
        "unique_inspected_spans": len(unique_spans),
        "primary_failure": failure,
        # Keep the raw, auditable decisions.  These are never training labels.
        "visual_inspections": inspections,
        "decisive_visual_verdicts": decisive,
    }


def add_reachability(
    item: dict[str, Any],
    target_spans: dict[tuple[str, str], list[tuple[float, float]]] | None,
    *,
    overlap_threshold: float,
) -> dict[str, Any]:
    """Attach offline evidence reachability without changing the trajectory."""
    if target_spans is None:
        item.update({"target_spans_available": False, "max_target_tiou": None, "evidence_reached": None})
        return item
    key = (str(item.get("video_id") or ""), str(item.get("uid") or ""))
    target = target_spans.get(key, [])
    inspected = [
        tuple(span)
        for inspection in item.get("visual_inspections", [])
        for span in inspection.get("spans", [])
        if isinstance(span, list) and len(span) >= 2
    ]
    overlap = max_temporal_overlap(inspected, target)
    item.update(
        {
            "target_spans_available": bool(target),
            "max_target_tiou": overlap,
            "evidence_reached": bool(overlap is not None and overlap >= overlap_threshold),
        }
    )
    return item


def wilson_interval(correct: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = correct / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def bootstrap_difference(rows: list[dict[str, Any]], predicate, *, rounds: int = 2000, seed: int = 17) -> list[float] | None:
    group_a = [item for item in rows if predicate(item)]
    group_b = [item for item in rows if not predicate(item)]
    if not group_a or not group_b:
        return None
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(max(100, rounds)):
        sample_a = [rng.choice(group_a) for _ in group_a]
        sample_b = [rng.choice(group_b) for _ in group_b]
        diffs.append(
            sum(bool(item["correct"]) for item in sample_a) / len(sample_a)
            - sum(bool(item["correct"]) for item in sample_b) / len(sample_b)
        )
    diffs.sort()
    return [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]]


def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(item["correct"]) for item in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "accuracy_ci95": wilson_interval(correct, len(rows)),
    }


def conditional_accuracy(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    return group_stats([row for row in rows if predicate(row)])


def stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the ordered evidence-to-answer funnel.

    Rows without dataset target spans are excluded from reachability cells but
    remain in the ordinary outcome report.  This prevents missing annotation
    metadata from being silently counted as retrieval failure.
    """
    annotated = [row for row in rows if row.get("target_spans_available")]
    reached = [row for row in annotated if row.get("evidence_reached")]
    not_reached = [row for row in annotated if not row.get("evidence_reached")]
    reached_wrong_verdict = [row for row in reached if not row.get("last_visual_verdict_correct")]
    reached_correct_verdict = [row for row in reached if row.get("last_visual_verdict_correct")]
    # A correct verdict may have been overwritten by a later decisive verdict;
    # this is kept as a separate authority/conflict signal.
    ever_correct_later_wrong = [
        row for row in rows
        if row.get("correct_visual_verdict") and not row.get("last_visual_verdict_correct")
    ]
    return {
        "annotated_n": len(annotated),
        "missing_target_spans_n": len(rows) - len(annotated),
        "reachability_rate": len(reached) / len(annotated) if annotated else None,
        "visual_decision_rate_given_reached": (
            len(reached_correct_verdict) / len(reached) if reached else None
        ),
        "final_accuracy_given_reached": conditional_accuracy(rows, lambda row: row.get("evidence_reached") is True),
        "final_accuracy_given_not_reached": conditional_accuracy(rows, lambda row: row.get("evidence_reached") is False),
        "final_accuracy_given_last_correct_verdict": conditional_accuracy(rows, lambda row: row.get("last_visual_verdict_correct") is True),
        "final_accuracy_given_last_wrong_verdict": conditional_accuracy(rows, lambda row: row.get("last_visual_verdict_correct") is False),
        "funnel_cells": {
            "R0_C1": sum(row.get("evidence_reached") is False and row.get("correct") for row in annotated),
            "R0_C0": sum(row.get("evidence_reached") is False and not row.get("correct") for row in annotated),
            "R1_V0_C1": sum(row.get("evidence_reached") is True and not row.get("last_visual_verdict_correct") and row.get("correct") for row in annotated),
            "R1_V0_C0": sum(row.get("evidence_reached") is True and not row.get("last_visual_verdict_correct") and not row.get("correct") for row in annotated),
            "R1_V1_C1": sum(row.get("evidence_reached") is True and row.get("last_visual_verdict_correct") and row.get("correct") for row in annotated),
            "R1_V1_C0": sum(row.get("evidence_reached") is True and row.get("last_visual_verdict_correct") and not row.get("correct") for row in annotated),
        },
        "error_cells": {
            "not_reached_final_wrong": sum(not row.get("correct") for row in not_reached),
            "reached_wrong_verdict_final_wrong": sum(not row.get("correct") for row in reached_wrong_verdict),
            "reached_correct_verdict_final_wrong": sum(not row.get("correct") for row in reached_correct_verdict),
            "ever_correct_verdict_later_wrong_final_wrong": sum(not row.get("correct") for row in ever_correct_later_wrong),
        },
        "rows_with_ever_correct_later_wrong": len(ever_correct_later_wrong),
    }


def summarize(rows: list[dict[str, Any]], *, missing: int, malformed: int, bootstrap_rounds: int) -> dict[str, Any]:
    matrix = Counter((bool(item["correct_visual_verdict"]), bool(item["correct"])) for item in rows)
    compliant_rows = [item for item in rows if item["protocol_compliant"]]
    compliant_matrix = Counter((bool(item["correct_visual_verdict"]), bool(item["correct"])) for item in compliant_rows)
    by_failure = Counter(item["primary_failure"] for item in rows)
    conditional = {
        "correct_visual_verdict": group_stats([x for x in rows if x["correct_visual_verdict"]]),
        "no_correct_visual_verdict": group_stats([x for x in rows if not x["correct_visual_verdict"]]),
        "final_visual_inspect_decisive": group_stats([x for x in rows if x["final_visual_inspect_decisive"]]),
        "final_not_visual_inspect_decisive": group_stats([x for x in rows if not x["final_visual_inspect_decisive"]]),
        "final_matches_visual_verdict": group_stats([x for x in rows if x["final_matches_visual_verdict"]]),
        "final_disagrees_with_visual_verdict": group_stats([x for x in rows if not x["final_matches_visual_verdict"]]),
        "visual_conflict": group_stats([x for x in rows if x["visual_conflict"]]),
        "no_visual_conflict": group_stats([x for x in rows if not x["visual_conflict"]]),
    }
    for name, predicate in (
        ("correct_visual_verdict_accuracy_gap", lambda x: x["correct_visual_verdict"]),
        ("final_visual_authority_accuracy_gap", lambda x: x["final_matches_visual_verdict"]),
        ("visual_conflict_accuracy_gap", lambda x: x["visual_conflict"]),
    ):
        conditional[name + "_bootstrap_ci95"] = bootstrap_difference(
            rows, predicate, rounds=bootstrap_rounds
        )
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        by_type[item["question_type"]].append(item)
    return {
        "schema_version": 1,
        "definition": {
            "correct_visual_verdict": "At least one decisive visual_inspect output contains exactly the benchmark-correct option. This is an outcome-linked trajectory signal, not an independent human groundedness label.",
            "evidence_discarded": "Final answer is wrong even though a prior decisive visual_inspect output contained the benchmark-correct option and the final answer was authorized by visual_inspect.",
            "protocol_shortcut": "Final answer is wrong and the last executed tool was not a decisive visual_inspect matching the final answer.",
            "search_or_coverage_failure": "Final answer is wrong, was authorized by visual_inspect, and no decisive visual_inspect output was produced.",
            "visual_verdict_error": "Final answer is wrong, was authorized by visual_inspect, decisive inspection occurred, but no decisive visual verdict matched the benchmark answer.",
            "gold_answer_used_only_for_offline_audit": True,
            "counterfactual_intervention": False,
        },
        "coverage": {"result_keys": len(rows), "missing_trajectories": missing, "malformed_trajectories": malformed},
        "overall": group_stats(rows),
        "outcome_matrix": {
            "visual_reached_and_final_correct": matrix[(True, True)],
            "visual_reached_and_final_wrong": matrix[(True, False)],
            "visual_not_reached_and_final_correct": matrix[(False, True)],
            "visual_not_reached_and_final_wrong": matrix[(False, False)],
        },
        "protocol_compliant_subset": {
            "n": len(compliant_rows),
            "outcome_matrix": {
                "visual_reached_and_final_correct": compliant_matrix[(True, True)],
                "visual_reached_and_final_wrong": compliant_matrix[(True, False)],
                "visual_not_reached_and_final_correct": compliant_matrix[(False, True)],
                "visual_not_reached_and_final_wrong": compliant_matrix[(False, False)],
            },
            "overall": group_stats(compliant_rows),
            "failure_counts": dict(sorted(Counter(item["primary_failure"] for item in compliant_rows).items())),
        },
        "conditional_accuracy": conditional,
        "failure_counts": dict(sorted(by_failure.items())),
        "by_question_type": {
            name: {
                "overall": group_stats(group),
                "failure_counts": dict(sorted(Counter(item["primary_failure"] for item in group).items())),
                "visual_reached_rate": sum(item["correct_visual_verdict"] for item in group) / len(group) if group else None,
            }
            for name, group in sorted(by_type.items())
        },
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    matrix = report["outcome_matrix"]
    cond = report["conditional_accuracy"]
    lines = [
        "# Outcome-Linked Long-Video QA Audit",
        "",
        "This report uses final benchmark correctness and the trajectory's own decisive visual verdicts. It does not treat trace length or inspection count as a failure label.",
        "",
        "## Outcome matrix",
        "",
        "| Visual verdict matching gold | Final correct | Count | Interpretation |",
        "|---|---:|---:|---|",
        f"| yes | yes | {matrix['visual_reached_and_final_correct']} | evidence reached and retained |",
        f"| yes | no | {matrix['visual_reached_and_final_wrong']} | evidence reached but final answer lost/disagreed |",
        f"| no | yes | {matrix['visual_not_reached_and_final_correct']} | correct without a correct visual verdict; possible shortcut/luck |",
        f"| no | no | {matrix['visual_not_reached_and_final_wrong']} | no correct visual verdict reached |",
        "",
        "## Accuracy conditioned on outcome-linked signals",
        "",
        "| Group | n | Accuracy | 95% Wilson CI |",
        "|---|---:|---:|---:|",
    ]
    for name in ("correct_visual_verdict", "no_correct_visual_verdict", "final_visual_inspect_decisive", "final_not_visual_inspect_decisive", "final_matches_visual_verdict", "final_disagrees_with_visual_verdict", "visual_conflict", "no_visual_conflict"):
        item = cond[name]
        ci = item["accuracy_ci95"]
        ci_text = "-" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        accuracy_text = "-" if item["accuracy"] is None else f"{item['accuracy']:.3f}"
        lines.append(f"| {name} | {item['n']} | {accuracy_text} | {ci_text} |")
    lines.extend(["", "## Primary failure counts", "", "| Failure | Count |", "|---|---:|"])
    for name, count in report["failure_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## Protocol-compliant subset",
        "",
        f"Rows whose last visual inspection was decisive and whose final answer matched that inspection: **{report['protocol_compliant_subset']['n']}**.",
        "The `visual_reached_and_final_wrong` cell is the only cell that can justify an evidence-retention/state-integration claim. It must be independently pixel-audited before being presented as visual groundedness.",
        "",
    ])
    if report.get("stage_summary") is not None:
        stage = report["stage_summary"]

        def fmt(value: Any) -> str:
            return "-" if value is None else f"{100.0 * value:.1f}%"

        lines.extend([
            "## Evidence-to-answer funnel",
            "",
            "This offline funnel uses dataset `time_reference` only for diagnosis; it is never exposed to the agent.",
            "",
            f"- Annotated rows: **{stage['annotated_n']}**; rows without target metadata: **{stage['missing_target_spans_n']}**",
            f"- Evidence reachability `R`: **{fmt(stage['reachability_rate'])}**",
            f"- Correct last visual verdict given `R=1`: **{fmt(stage['visual_decision_rate_given_reached'])}**",
            f"- Accuracy given `R=1`: **{fmt(stage['final_accuracy_given_reached']['accuracy'])}**",
            f"- Accuracy given `R=0`: **{fmt(stage['final_accuracy_given_not_reached']['accuracy'])}**",
            f"- Accuracy given a correct last visual verdict: **{fmt(stage['final_accuracy_given_last_correct_verdict']['accuracy'])}**",
            "",
            "| Funnel cell | Count | Interpretation |",
            "|---|---:|---|",
            f"| `R=0, C=1` | {stage['funnel_cells']['R0_C1']} | correct without an annotated target hit |",
            f"| `R=0, C=0` | {stage['funnel_cells']['R0_C0']} | target evidence not reached and answer wrong |",
            f"| `R=1, V_last=0, C=1` | {stage['funnel_cells']['R1_V0_C1']} | correct despite a wrong/no final visual verdict |",
            f"| `R=1, V_last=0, C=0` | {stage['funnel_cells']['R1_V0_C0']} | reached target but visual decision failed |",
            f"| `R=1, V_last=1, C=1` | {stage['funnel_cells']['R1_V1_C1']} | reached target and converted correctly |",
            f"| `R=1, V_last=1, C=0` | {stage['funnel_cells']['R1_V1_C0']} | correct visual verdict not preserved |",
            "",
            "The method target must be selected from the largest wrong cell in this funnel, rather than from trace length or inspection count.",
            "",
        ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional raw or VideoSEAL LVBench parquet containing time_reference.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.05,
        help="tIoU threshold for offline evidence reachability.",
    )
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    args = parser.parse_args()

    latest = latest_results(args.results)
    target_spans = load_target_spans(args.dataset) if args.dataset else None
    rows: list[dict[str, Any]] = []
    missing = malformed = 0
    for row in latest.values():
        trajectory = load_trajectory(row, args.run_root)
        if trajectory is None:
            missing += 1
            continue
        try:
            audited = audit_row(row, trajectory)
            rows.append(add_reachability(audited, target_spans, overlap_threshold=args.overlap_threshold))
        except (TypeError, ValueError, KeyError, AttributeError):
            malformed += 1
    report = summarize(rows, missing=missing, malformed=malformed, bootstrap_rounds=args.bootstrap_rounds)
    report["stage_summary"] = stage_summary(rows) if target_spans is not None else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("coverage", "overall", "outcome_matrix", "conditional_accuracy", "failure_counts")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
