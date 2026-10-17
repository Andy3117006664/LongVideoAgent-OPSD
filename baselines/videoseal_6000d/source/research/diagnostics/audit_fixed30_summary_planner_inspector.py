#!/usr/bin/env python3
"""Offline funnel audit for the fixed-30 VideoSEAL experiment.

The online run never receives ``time_reference``.  This script joins it only
after inference to locate where answer-bearing windows were lost:

    retrieved caption -> summarizer text -> planner inspection -> inspector
    verdict -> final answer

The report deliberately keeps requested inspection-window reachability and
reconstructed sampled-frame reachability separate.  The latter replays the
frozen 2 FPS / 64-frame prefix sampler used by the formal run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq


TIME_RE = re.compile(r"(?<!\d)(?P<h>\d{1,3}):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)(?!\d)")
SHORT_TIME_RE = re.compile(r"(?<![\d:])(?P<m>\d{1,4}):(?P<s>\d{2}(?:\.\d+)?)(?![\d:])")
RANGE_RE = re.compile(
    r"(?P<start>\d{1,3}:\d{2}(?::\d{2}(?:\.\d+)?)?)"
    r"\s*[-\u2013\u2014]\s*"
    r"(?P<end>\d{1,3}:\d{2}(?::\d{2}(?:\.\d+)?)?)"
)
ANSWER_RE = re.compile(r"(?:^|\n)\s*Answer\s*[:\uff1a]\s*\(?([A-D])\)?", re.IGNORECASE)
BARE_ANSWER_RE = re.compile(r"^\s*\(?([A-D])\)?\s*$", re.IGNORECASE)


def parse_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = TIME_RE.fullmatch(text)
    if match:
        return 3600.0 * int(match.group("h")) + 60.0 * int(match.group("m")) + float(match.group("s"))
    match = SHORT_TIME_RE.fullmatch(text)
    if match:
        return 60.0 * int(match.group("m")) + float(match.group("s"))
    try:
        return float(text)
    except ValueError:
        return None


def hms(value: float) -> str:
    total = max(0, int(value))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def parse_references(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, (list, tuple)):
        found: list[tuple[float, float]] = []
        for item in value:
            found.extend(parse_references(item))
        return list(dict.fromkeys(found))
    found = []
    for match in RANGE_RE.finditer(str(value or "")):
        start = parse_seconds(match.group("start"))
        end = parse_seconds(match.group("end"))
        if start is not None and end is not None:
            found.append((min(start, end), max(start, end)))
    return list(dict.fromkeys(found))


def parse_span(item: Mapping[str, Any]) -> tuple[float, float] | None:
    start = parse_seconds(item.get("start_time"))
    end = parse_seconds(item.get("end_time"))
    if start is None or end is None:
        return None
    return min(start, end), max(start, end)


def extract_spans(value: Any) -> list[tuple[float, float]]:
    found: list[tuple[float, float]] = []
    if isinstance(value, Mapping):
        direct = parse_span(value)
        if direct is not None:
            found.append(direct)
        for key in ("spans", "useful_spans", "intervals", "windows"):
            if key in value:
                found.extend(extract_spans(value[key]))
    elif isinstance(value, list):
        for item in value:
            found.extend(extract_spans(item))
    return list(dict.fromkeys(found))


def intersects(left: tuple[float, float], right: tuple[float, float]) -> bool:
    # Match the established LVBench retrieval audit convention, including a
    # shared boundary between a 16-second index cell and the reference.
    return max(left[0], right[0]) <= min(left[1], right[1])


def any_intersection(spans: Iterable[tuple[float, float]], references: Iterable[tuple[float, float]]) -> bool:
    return any(intersects(span, reference) for span in spans for reference in references)


def relevance_block(summary: str) -> str:
    text = str(summary or "")
    start = text.casefold().find("span relevance:")
    if start < 0:
        # The lossless compactor emits one canonical Cxx line per candidate
        # instead of the released Span relevance / Evidence sections.
        return text if re.search(r"(?m)^\s*C\d{2}\s*\|", text) else ""
    start += len("span relevance:")
    end = text.casefold().find("evidence:", start)
    return text[start : end if end >= 0 else len(text)].strip()


def summary_mentioned_candidates(summary: str, candidates: list[tuple[float, float]]) -> list[int]:
    """Map the summarizer's relevance lines back to candidate indices.

    Most outputs repeat full candidate ranges.  Nine calls in the formal run
    emit only one timestamp per retained caption, so point-in-window matching
    is the necessary fallback for those calls.
    """

    block = relevance_block(summary)
    if not block:
        return []
    found: set[int] = set()
    ranges = []
    for match in RANGE_RE.finditer(block):
        start = parse_seconds(match.group("start"))
        end = parse_seconds(match.group("end"))
        if start is not None and end is not None:
            ranges.append((min(start, end), max(start, end)))
    for index, candidate in enumerate(candidates):
        if any(abs(candidate[0] - span[0]) < 1.0 and abs(candidate[1] - span[1]) < 1.0 for span in ranges):
            found.add(index)
    if ranges:
        return sorted(found)

    if "none found" in block.casefold():
        return []

    points = [parse_seconds(match.group(0)) for match in TIME_RE.finditer(block)]
    for point in (value for value in points if value is not None):
        containing = [
            index for index, candidate in enumerate(candidates)
            if candidate[0] <= point < candidate[1]
        ]
        if containing:
            # Windows are non-overlapping in the frozen semantic index.
            found.add(containing[0])
    return sorted(found)


def inspector_answer(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("answer")
    text = str(value or "").strip()
    match = ANSWER_RE.search(text) or BARE_ANSWER_RE.fullmatch(text)
    return match.group(1).upper() if match else ""


def tagged_final(text: str, *, last: bool) -> str:
    source = str(text or "")
    start = source.rfind("<final>") if last else source.find("<final>")
    if start < 0:
        return ""
    start += len("<final>")
    end = source.find("</final>", start)
    return source[start:end].strip() if end >= 0 else ""


def replay_sampled_timestamps(spans: list[tuple[float, float]], fps: float = 2.0, max_total: int = 64) -> list[float]:
    per_span = [
        [start + index / fps for index in range(max(1, int((end - start) * fps)))][:max_total]
        for start, end in spans
    ]
    if sum(len(values) for values in per_span) <= max_total:
        return [value for values in per_span for value in values]
    positions = [0] * len(per_span)
    selected: list[float] = []
    while len(selected) < max_total and any(positions[index] < len(values) for index, values in enumerate(per_span)):
        for index, values in enumerate(per_span):
            if len(selected) >= max_total:
                break
            position = positions[index]
            if position >= len(values):
                continue
            selected.append(values[position])
            positions[index] += 1
    return selected


def timestamps_hit(timestamps: Iterable[float], references: Iterable[tuple[float, float]]) -> bool:
    return any(start <= timestamp <= end for timestamp in timestamps for start, end in references)


def load_dataset(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    table = pq.read_table(path, columns=["extra_info"])
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for wrapper in table["extra_info"].to_pylist():
        row = wrapper.get("extra_info") if isinstance(wrapper, dict) and isinstance(wrapper.get("extra_info"), dict) else wrapper
        if not isinstance(row, dict):
            continue
        key = (str(row.get("video_id") or "").strip(), str(row.get("qa_uid") or "").strip())
        if all(key):
            rows[key] = dict(row)
    return rows


def load_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("state") != "completed":
            continue
        key = (str(row.get("video_id") or "").strip(), str(row.get("uid") or "").strip())
        if all(key):
            rows[key] = row
    return rows


def resolve_trajectory(run_root: Path, row: Mapping[str, Any]) -> Path | None:
    raw = Path(str(row.get("trajectory_path") or ""))
    candidates = [raw]
    if raw.is_absolute():
        try:
            marker = raw.parts.index(run_root.name)
            candidates.append(run_root.joinpath(*raw.parts[marker + 1 :]))
        except ValueError:
            pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    video_id = str(row.get("video_id") or "")
    uid = str(row.get("uid") or "")
    for candidate in sorted((run_root / video_id).glob("*/trajectory.json")):
        try:
            trajectory = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(trajectory.get("uid") or "") == uid:
            return candidate
    return None


def load_pair_cells(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["uid"]): str(row["cell"]) for row in value.get("rows") or []}


def load_uids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def caption_store(index_path: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if index_path not in cache:
        path = Path(index_path) / "semantic_captions.json"
        cache[index_path] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return cache[index_path]


def caption_for_span(store: Mapping[str, Any], span: tuple[float, float]) -> str:
    doc_id = f"{int(span[0])}_{int(span[1])}"
    value = store.get(doc_id)
    if not isinstance(value, Mapping):
        return ""
    return str(value.get("caption") or value.get("clip_caption") or value.get("ocr_text") or "").strip()


def audit_one(
    result: Mapping[str, Any],
    dataset_row: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    pair_cell: str,
    captions: Mapping[str, Any],
) -> dict[str, Any]:
    references = parse_references(dataset_row.get("time_reference"))
    ground_truth = str(result.get("ground_truth") or trajectory.get("groundtruth") or "").strip().upper()
    prediction = str(result.get("prediction") or trajectory.get("answer") or "").strip().upper()
    retrievals: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    planner_trace: list[dict[str, Any]] = []

    for step_index, step in enumerate(trajectory.get("steps") or []):
        if not isinstance(step, Mapping):
            continue
        action = step.get("action") if isinstance(step.get("action"), Mapping) else {}
        observation = step.get("observation") if isinstance(step.get("observation"), Mapping) else {}
        name = str(action.get("name") or "")
        planner_trace.append(
            {
                "step": step_index,
                "action_name": name,
                "action_arguments": action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {},
                "thinking": str(step.get("thinking") or ""),
                "model_response": str(step.get("model_response") or ""),
            }
        )
        if name == "visual_retrieve":
            output = observation.get("output") if isinstance(observation.get("output"), Mapping) else {}
            metadata = observation.get("metadata") if isinstance(observation.get("metadata"), Mapping) else {}
            candidates = extract_spans(output.get("useful_spans")) or extract_spans(metadata.get("spans"))
            summary = str(output.get("summary") or "")
            mentioned_indices = summary_mentioned_candidates(summary, candidates)
            mentioned = [candidates[index] for index in mentioned_indices]
            target_indices = [index for index, span in enumerate(candidates) if any(intersects(span, ref) for ref in references)]
            retrievals.append(
                {
                    "step": step_index,
                    "query": str((action.get("arguments") or {}).get("query") or ""),
                    "candidate_count": len(candidates),
                    "candidate_hit": bool(target_indices),
                    "target_candidate_indices": target_indices,
                    "target_candidates": [
                        {
                            "rank": index + 1,
                            "span": [candidates[index][0], candidates[index][1]],
                            "caption": caption_for_span(captions, candidates[index]),
                            "mentioned_by_summarizer": index in mentioned_indices,
                        }
                        for index in target_indices
                    ],
                    "summary_mentioned_count": len(mentioned),
                    "summary_target_hit": any_intersection(mentioned, references),
                    "relevance_block": relevance_block(summary),
                    "summary": summary,
                }
            )
        elif name == "visual_inspect":
            forced = observation.get("forced") is True
            arguments = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
            requested = extract_spans(arguments.get("spans"))
            output = observation.get("output")
            verdict = inspector_answer(output)
            sampled = [] if forced else replay_sampled_timestamps(requested)
            window_hit = any_intersection(requested, references) if not forced else False
            frame_hit = timestamps_hit(sampled, references) if not forced else False
            answer_text = str(output.get("answer") or "") if isinstance(output, Mapping) else str(output or "")
            inspections.append(
                {
                    "step": step_index,
                    "forced_full_video": forced,
                    "requested_spans": [[start, end] for start, end in requested],
                    "requested_window_hit": window_hit,
                    "reconstructed_frame_hit": frame_hit,
                    "verdict": verdict,
                    "verdict_correct": bool(verdict and verdict == ground_truth),
                    "context": str(arguments.get("context") or ""),
                    "answer_text": answer_text,
                }
            )

    local_inspections = [item for item in inspections if not item["forced_full_video"]]
    frame_hit_inspections = [item for item in local_inspections if item["reconstructed_frame_hit"]]
    correct_frame_hit = [item for item in frame_hit_inspections if item["verdict_correct"]]
    retrieved = any(item["candidate_hit"] for item in retrievals)
    summary_retained = any(item["summary_target_hit"] for item in retrievals)
    target_caption_available = any(
        target.get("caption")
        for retrieval in retrievals
        for target in retrieval["target_candidates"]
    )
    local_window_hit = any(item["requested_window_hit"] for item in local_inspections)
    local_frame_hit = any(item["reconstructed_frame_hit"] for item in local_inspections)
    inspector_correct_on_target = bool(correct_frame_hit)
    final_correct = bool(result.get("correct")) if isinstance(result.get("correct"), bool) else prediction == ground_truth
    inspections_by_step = {item["step"]: item for item in inspections}
    parser_shadow_events = []
    for item in planner_trace:
        previous = inspections_by_step.get(item["step"] - 1)
        if not previous or not previous["verdict"]:
            continue
        first = tagged_final(item["model_response"], last=False)
        last = tagged_final(item["model_response"], last=True)
        if (
            re.fullmatch(r"[A-D]", last)
            and last == previous["verdict"]
            and not re.fullmatch(r"[A-D]", first)
        ):
            parser_shadow_events.append(
                {
                    "step": item["step"],
                    "first_tag_parse": first,
                    "last_tag_parse": last,
                    "matches_ground_truth": last == ground_truth,
                }
            )

    if final_correct:
        terminal_stage = "final_correct"
    elif not retrieved:
        terminal_stage = "retrieval_miss"
    elif not summary_retained and not local_frame_hit:
        terminal_stage = "retrieved_but_summary_omitted_and_not_inspected"
    elif summary_retained and not local_frame_hit:
        terminal_stage = "summary_retained_but_planner_did_not_reach_target_frames"
    elif local_frame_hit and not inspector_correct_on_target:
        terminal_stage = "target_frames_reached_but_inspector_verdict_not_correct"
    elif inspector_correct_on_target and not final_correct:
        terminal_stage = "correct_target_inspector_verdict_not_preserved"
    else:
        terminal_stage = "other"

    return {
        "uid": str(result.get("uid") or ""),
        "video_id": str(result.get("video_id") or ""),
        "pair_cell": pair_cell,
        "question": str(trajectory.get("question") or result.get("question") or ""),
        "time_reference": [[start, end] for start, end in references],
        "ground_truth": ground_truth,
        "prediction": prediction,
        "final_correct": final_correct,
        "retrieval_calls": len(retrievals),
        "inspect_calls": len(local_inspections),
        "forced_full_video_calls": len(inspections) - len(local_inspections),
        "has_forced_full_video": any(item["forced_full_video"] for item in inspections),
        "first_retrieval_hit": bool(retrievals and retrievals[0]["candidate_hit"]),
        "first_summary_target_hit": bool(retrievals and retrievals[0]["summary_target_hit"]),
        "retrieval_hit": retrieved,
        "target_caption_available": target_caption_available,
        "summary_target_hit": summary_retained,
        "local_inspection_window_hit": local_window_hit,
        "local_inspection_frame_hit": local_frame_hit,
        "inspector_correct_on_target_frames": inspector_correct_on_target,
        "correct_target_inspector_verdict_but_final_wrong": inspector_correct_on_target and not final_correct,
        "correct_target_inspector_overwritten_by_forced": (
            inspector_correct_on_target
            and not final_correct
            and any(item["forced_full_video"] for item in inspections)
        ),
        "parser_shadowed_valid_final": bool(parser_shadow_events),
        "parser_shadowed_correct_final": any(item["matches_ground_truth"] for item in parser_shadow_events),
        "parser_shadow_events": parser_shadow_events,
        "terminal_stage": terminal_stage,
        "planner_trace": planner_trace,
        "retrievals": retrievals,
        "inspections": inspections,
    }


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    keys = (
        "final_correct",
        "has_forced_full_video",
        "first_retrieval_hit",
        "first_summary_target_hit",
        "retrieval_hit",
        "target_caption_available",
        "summary_target_hit",
        "local_inspection_window_hit",
        "local_inspection_frame_hit",
        "inspector_correct_on_target_frames",
        "correct_target_inspector_verdict_but_final_wrong",
        "correct_target_inspector_overwritten_by_forced",
        "parser_shadowed_valid_final",
        "parser_shadowed_correct_final",
    )
    counts = {key: sum(bool(row[key]) for row in rows) for key in keys}
    retrieved = [row for row in rows if row["retrieval_hit"]]
    summary_hit = [row for row in rows if row["summary_target_hit"]]
    frame_hit = [row for row in rows if row["local_inspection_frame_hit"]]
    counts.update(
        {
            "n": n,
            "summary_target_hit_given_retrieval_hit": (
                sum(bool(row["summary_target_hit"]) for row in retrieved) / len(retrieved) if retrieved else None
            ),
            "local_frame_hit_given_summary_target_hit": (
                sum(bool(row["local_inspection_frame_hit"]) for row in summary_hit) / len(summary_hit) if summary_hit else None
            ),
            "correct_inspector_given_local_frame_hit": (
                sum(bool(row["inspector_correct_on_target_frames"]) for row in frame_hit) / len(frame_hit) if frame_hit else None
            ),
            "retrieval_hit_summary_miss": sum(
                bool(row["retrieval_hit"]) and not bool(row["summary_target_hit"])
                for row in rows
            ),
            "summary_miss_but_local_frame_hit": sum(
                bool(row["retrieval_hit"])
                and not bool(row["summary_target_hit"])
                and bool(row["local_inspection_frame_hit"])
                for row in rows
            ),
            "summary_hit_but_local_frame_miss": sum(
                bool(row["summary_target_hit"])
                and not bool(row["local_inspection_frame_hit"])
                for row in rows
            ),
            "local_frame_hit_but_no_correct_inspector": sum(
                bool(row["local_inspection_frame_hit"])
                and not bool(row["inspector_correct_on_target_frames"])
                for row in rows
            ),
            "terminal_stages": dict(Counter(row["terminal_stage"] for row in rows)),
        }
    )
    return counts


def render_markdown(report: Mapping[str, Any]) -> str:
    overall = report["overall"]
    stage_names = {
        "final_correct": "最终答案正确",
        "target_frames_reached_but_inspector_verdict_not_correct": "已采到参考帧，但 inspector 未给出正确判定",
        "correct_target_inspector_verdict_not_preserved": "inspector 已给出正确判定，但最终系统未保留",
        "summary_retained_but_planner_did_not_reach_target_frames": "summary 已保留参考窗口，但 planner 未检查到参考帧",
        "retrieved_but_summary_omitted_and_not_inspected": "检索已命中，但 summary 省略且 planner 未检查",
        "retrieval_miss": "检索未命中参考窗口",
    }
    cell_names = {
        "both_correct": "两版均正确",
        "fixed": "fixed-30 修复",
        "broken": "fixed-30 破坏",
        "both_wrong": "两版均错误",
    }

    def frac(key: str) -> str:
        return f"{overall[key]}/{overall['n']} ({100.0 * overall[key] / overall['n']:.1f}%)"

    lines = [
        "# Fixed-30：summarizer -> planner -> inspector 漏斗审计",
        "",
        "`time_reference` 仅在推理结束后离线关联，从未提供给在线 agent。",
        "局部 inspector 路由统计不包含强制全视频兜底调用。",
        "",
        "| 到达阶段 | 题数 |",
        "|---|---:|",
        f"| 首次检索包含与参考区间重叠的 caption 窗口 | {frac('first_retrieval_hit')} |",
        f"| 首次 summarizer 相关性文本保留该窗口 | {frac('first_summary_target_hit')} |",
        f"| 任意一次检索包含该窗口 | {frac('retrieval_hit')} |",
        f"| 参考候选具有非空原始 caption | {frac('target_caption_available')} |",
        f"| 任意一次 summarizer 相关性文本保留该窗口 | {frac('summary_target_hit')} |",
        f"| Planner 请求与参考区间重叠的局部检查窗口 | {frac('local_inspection_window_hit')} |",
        f"| 重建出的局部 inspector 采样帧到达参考区间 | {frac('local_inspection_frame_hit')} |",
        f"| 参考帧上的 inspector 检查返回正确选项 | {frac('inspector_correct_on_target_frames')} |",
        f"| 最终答案正确 | {frac('final_correct')} |",
        "",
        "## 条件传递率",
        "",
        f"- 检索命中后，summarizer 保留参考窗口的比例：{overall['summary_target_hit_given_retrieval_hit']:.3f}",
        f"- summary 保留后，局部帧到达参考区间的比例：{overall['local_frame_hit_given_summary_target_hit']:.3f}",
        f"- 局部帧到达后，inspector 给出正确判定的比例：{overall['correct_inspector_given_local_frame_hit']:.3f}",
        f"- 已检索到但被所有 summary 省略：{overall['retrieval_hit_summary_miss']}",
        f"- summary 省略但 planner 仍检查到局部参考帧：{overall['summary_miss_but_local_frame_hit']}",
        f"- summary 保留但 planner 未检查到局部参考帧：{overall['summary_hit_but_local_frame_miss']}",
        f"- 局部参考帧已到达但 inspector 未给出正确判定：{overall['local_frame_hit_but_no_correct_inspector']}",
        f"- 正确 inspector 判定后来被强制兜底覆盖：{overall['correct_target_inspector_overwritten_by_forced']}",
        f"- 模型推理中较早出现的 `<final>` 字样遮蔽了有效 final：{overall['parser_shadowed_valid_final']}",
        "",
        "## 互斥终态分类",
        "",
    ]
    for key, value in sorted(overall["terminal_stages"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {stage_names.get(key, key)}（`{key}`）：{value}")
    lines.extend(["", "## 配对结果分组", ""])
    for cell, summary in report["by_pair_cell"].items():
        lines.append(
            f"- {cell_names.get(cell, cell)}（`{cell}`）：题数={summary['n']}，检索命中={summary['retrieval_hit']}，"
            f"summary 保留={summary['summary_target_hit']}，局部帧到达={summary['local_inspection_frame_hit']}，"
            f"inspector 正确={summary['inspector_correct_on_target_frames']}"
        )
    lines.extend(
        [
            "",
            "配套 JSON 保存了每题的原始目标 caption、summarizer 相关性文本、planner 检查上下文、inspector 原始回答和最终结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paired-report", type=Path)
    parser.add_argument("--uids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = load_results(args.results)
    selected_uids = load_uids(args.uids)
    if selected_uids is not None:
        results = {key: value for key, value in results.items() if key[1] in selected_uids}
    pair_cells = load_pair_cells(args.paired_report)
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for key, result in sorted(results.items(), key=lambda item: int(item[0][1])):
        dataset_row = dataset.get(key)
        trajectory_path = resolve_trajectory(args.run_root, result)
        if dataset_row is None or trajectory_path is None:
            errors.append(f"missing {'dataset row' if dataset_row is None else 'trajectory'}: {key[0]}::{key[1]}")
            continue
        references = parse_references(dataset_row.get("time_reference"))
        if not references:
            errors.append(f"missing time_reference: {key[0]}::{key[1]}")
            continue
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if trajectory.get("time_reference") not in (None, ""):
            errors.append(f"online time_reference leakage: {key[0]}::{key[1]}")
            continue
        store = caption_store(str(dataset_row.get("VISUAL_INDEX_DIR") or ""), cache)
        rows.append(audit_one(result, dataset_row, trajectory, pair_cells.get(key[1], ""), store))

    by_pair_cell = {
        cell: group_summary([row for row in rows if row["pair_cell"] == cell])
        for cell in ("both_correct", "fixed", "broken", "both_wrong")
        if any(row["pair_cell"] == cell for row in rows)
    }
    report = {
        "schema_version": 1,
        "policy": {
            "time_reference_used_online": False,
            "time_reference_used_offline_only": True,
            "candidate_hit": "any temporal intersection with a returned fixed-30 caption window",
            "summary_target_hit": "the summarizer Span relevance block names a reference-overlapping candidate window",
            "local_inspector": "visual_inspect calls excluding forced full-video fallback",
            "frame_hit": "replay of the frozen 2 FPS, 64-frame balanced-prefix sampler",
        },
        "integrity": {"results": len(results), "audited": len(rows), "errors": errors, "passed": len(rows) == len(results) and not errors},
        "overall": group_summary(rows),
        "by_pair_cell": by_pair_cell,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"integrity": report["integrity"], "overall": report["overall"], "by_pair_cell": by_pair_cell}, ensure_ascii=False, indent=2))
    return 0 if report["integrity"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
