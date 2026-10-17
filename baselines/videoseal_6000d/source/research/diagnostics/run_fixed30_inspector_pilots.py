#!/usr/bin/env python3
"""Paired inspector pilots on the frozen Fixed-30 LVBench run.

This script runs two small causal replays without invoking retrieval or the
planner:

1. context: replay the same frozen target-reaching packet with source,
   question-only, and caption-facts-only contexts;
2. sampling: replay the same frozen packet and source context with the current
   prefix sampler, global full-span coverage, and quota-preserving within-span
   coverage.

Reference intervals and benchmark answers are used only to freeze diagnostic
cases and reduce outcomes. They are never included in model requests and never
affect timestamps inside a frozen packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "baselines" / "VideoSEAL"))
sys.path.insert(0, str(ROOT / "research" / "diagnostics"))

from outcome_linked_audit import parse_visual_answer  # noqa: E402
from run_reference_oracle_ladder import (  # noqa: E402
    content_with_frames,
    request_chat,
)
from videoseal.prompts.visual_tool_prompts import (  # noqa: E402
    build_visual_inspect_last_step_mcq_prompt,
    build_visual_inspect_prompt,
)
from videoseal.utils.video.frames import (  # noqa: E402
    resize_images_max_long_side,
    sample_uniform_frames,
)


WRITE_LOCK = threading.Lock()
VALID = tuple("ABCDEFGH")


def stable_hash(*values: Any) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_path(value: Any) -> Path:
    text = str(value or "")
    marker = "/lvqa-experiments/"
    return ROOT / text.split(marker, 1)[1] if marker in text else Path(text)


def load_tasks(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    table = pq.read_table(path, columns=["extra_info"])
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in table["extra_info"].to_pylist():
        row = dict(raw or {})
        key = (
            str(row.get("video_id") or row.get("video_key") or "").strip(),
            str(row.get("qa_uid") or row.get("uid") or "").strip(),
        )
        if all(key):
            row["video_id"], row["uid"] = key
            output[key] = row
    return output


def resolve_video(task: dict[str, Any]) -> Path | None:
    candidates = (
        project_path(task.get("VIDEO_PATH")),
        ROOT / "datasets" / "LVBench" / "video_chunks" / "extracted" / str(task.get("video_file") or ""),
        ROOT / "datasets" / "LVBench" / "video_chunks" / "extracted" / f"{task.get('video_id')}.mp4",
    )
    for path in candidates:
        try:
            if str(path) and path.is_file():
                return path
        except OSError:
            continue
    return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"missing rows in {path}")
    return [row for row in rows if isinstance(row, dict)]


def answer_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    direct = re.fullmatch(r"\(?([A-H])\)?[.)]?", text)
    if direct:
        return direct.group(1)
    parsed = parse_visual_answer(text)
    options = [str(item).upper() for item in parsed.get("options") or []]
    return options[0] if len(options) == 1 else ""


def spans_of(inspection: dict[str, Any]) -> list[tuple[float, float]]:
    output = []
    for raw in inspection.get("requested_spans") or []:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        try:
            start, end = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            continue
        if end > start:
            output.append((start, end))
    return output


def natural_count(spans: Iterable[tuple[float, float]], fps: float) -> int:
    return sum(min(max(int((end - start) * fps), 1), 64) for start, end in spans)


def packet_prefix_timestamps(
    spans: list[tuple[float, float]], *, fps: float, cap: int
) -> tuple[list[float], list[int]]:
    per_span: list[list[float]] = []
    for start, end in spans:
        count = min(max(int((end - start) * fps), 1), cap)
        per_span.append([round(start + index / fps, 3) for index in range(count)])
    selected: list[list[float]] = [[] for _ in spans]
    indexes = [0] * len(spans)
    taken = 0
    while taken < cap and any(indexes[i] < len(per_span[i]) for i in range(len(spans))):
        for i in range(len(spans)):
            if taken >= cap:
                break
            index = indexes[i]
            if index >= len(per_span[i]):
                continue
            selected[i].append(per_span[i][index])
            indexes[i] += 1
            taken += 1
    # Production defaults to INSPECT_GLOBAL_ORDER=0: preserve planner span
    # order, then chronological order inside each span.
    timestamps = [value for values in selected for value in values]
    return timestamps, [len(values) for values in selected]


def proportional_uniform_timestamps(
    spans: list[tuple[float, float]], *, cap: int
) -> list[float]:
    if not spans or cap <= 0:
        return []
    widths = [max(end - start, 1e-3) for start, end in spans]
    allocation = [1] * len(spans)
    remaining = max(0, cap - len(spans))
    total = sum(widths)
    raw = [remaining * width / total for width in widths]
    floors = [int(math.floor(value)) for value in raw]
    allocation = [base + extra for base, extra in zip(allocation, floors)]
    leftover = remaining - sum(floors)
    order = sorted(
        range(len(spans)),
        key=lambda i: (raw[i] - floors[i], widths[i]),
        reverse=True,
    )
    for index in order[:leftover]:
        allocation[index] += 1
    return spread_with_allocation(spans, allocation, cap=cap)


def spread_with_allocation(
    spans: list[tuple[float, float]], allocation: list[int], *, cap: int
) -> list[float]:
    values: list[float] = []
    for (start, end), count in zip(spans, allocation):
        if count <= 0:
            continue
        if count == 1:
            values.append(start)
        else:
            values.extend(start + (end - start) * index / (count - 1) for index in range(count))
    return list(dict.fromkeys(round(value, 3) for value in values))[:cap]


def extract_production_packet(
    video: Path,
    spans: list[tuple[float, float]],
    *,
    fps: float,
    cap: int,
    directory: Path,
) -> tuple[list[list[Path]], list[list[float]], list[Path], list[float], list[int]]:
    """Mirror VisualInspectAliasTool's OpenCV extraction and prefix cap."""
    per_span_frames: list[list[Path]] = []
    per_span_ts: list[list[float]] = []
    # Formal Fixed-30 launcher: VISUAL_INSPECT_MAX_LONG_EDGE=1280.
    previous_edge = __import__("os").environ.get("FRAME_MAX_LONG_SIDE")
    __import__("os").environ["FRAME_MAX_LONG_SIDE"] = "1280"
    for index, (start, end) in enumerate(spans):
        paths, timestamps = sample_uniform_frames(
            str(video),
            start,
            end,
            fps=fps,
            max_frames=cap,
            output_dir=str(directory / f"span_{index:03d}"),
            subtitle_path=None,
        )
        per_span_frames.append([Path(path) for path in paths])
        per_span_ts.append([float(value) for value in timestamps])

    selected_frames: list[list[Path]] = [[] for _ in spans]
    selected_ts: list[list[float]] = [[] for _ in spans]
    indexes = [0] * len(spans)
    taken = 0
    while taken < cap and any(indexes[i] < len(per_span_frames[i]) for i in range(len(spans))):
        for i in range(len(spans)):
            if taken >= cap:
                break
            position = indexes[i]
            if position >= len(per_span_frames[i]):
                continue
            selected_frames[i].append(per_span_frames[i][position])
            selected_ts[i].append(per_span_ts[i][position])
            indexes[i] += 1
            taken += 1
    flat_frames = [path for group in selected_frames for path in group]
    flat_ts = [value for group in selected_ts for value in group]
    # Formal Fixed-30 launcher: INSPECT_GLOBAL_ORDER=1.
    ordered = sorted(zip(flat_ts, flat_frames), key=lambda item: item[0])
    flat_ts = [item[0] for item in ordered]
    flat_frames = [item[1] for item in ordered]
    if previous_edge is None:
        __import__("os").environ.pop("FRAME_MAX_LONG_SIDE", None)
    else:
        __import__("os").environ["FRAME_MAX_LONG_SIDE"] = previous_edge
    return (
        per_span_frames,
        per_span_ts,
        flat_frames,
        flat_ts,
        [len(group) for group in selected_frames],
    )


def spread_existing_packet(
    per_span_frames: list[list[Path]],
    per_span_ts: list[list[float]],
    allocation: list[int],
) -> tuple[list[Path], list[float]]:
    frames: list[Path] = []
    timestamps: list[float] = []
    for span_frames, span_ts, keep in zip(per_span_frames, per_span_ts, allocation):
        if keep <= 0 or not span_frames:
            continue
        if keep >= len(span_frames):
            indices = list(range(len(span_frames)))
        elif keep == 1:
            indices = [0]
        else:
            last = len(span_frames) - 1
            indices = []
            for index in range(keep):
                position = int(round(index * last / float(keep - 1)))
                if not indices or position != indices[-1]:
                    indices.append(position)
            for position in range(len(span_frames)):
                if len(indices) >= keep:
                    break
                if position not in indices:
                    indices.append(position)
            indices = sorted(indices[:keep])
        frames.extend(span_frames[index] for index in indices)
        timestamps.extend(span_ts[index] for index in indices)
    ordered = sorted(zip(timestamps, frames), key=lambda item: item[0])
    return [item[1] for item in ordered], [item[0] for item in ordered]


def apply_formal_dynamic_resize(frames: list[Path]) -> int:
    """Mirror the formal run's shared 20 Mi-pixel dynamic edge policy."""
    if not frames:
        return 0
    from PIL import Image

    with Image.open(frames[0]) as image:
        short = max(1, min(image.width, image.height))
        long = max(1, max(image.width, image.height))
    aspect = max(1.0, float(long) / float(short))
    per_frame = max(12544.0, 20971520.0 / float(len(frames)))
    edge = int(math.floor(math.sqrt(per_frame * aspect)))
    edge = max(32, (edge // 32) * 32)
    edge = min(1280, edge)
    if edge < 1280:
        resize_images_max_long_side([str(path) for path in frames], max_long_side=edge)
    return edge


def request_formal_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    content: list[dict[str, Any]],
    timeout: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"] or "").strip()


def proportional_allocation(spans: list[tuple[float, float]], cap: int) -> list[int]:
    if not spans:
        return []
    widths = [max(end - start, 1e-3) for start, end in spans]
    allocation = [1] * len(spans)
    remaining = max(0, cap - len(spans))
    total = sum(widths)
    raw = [remaining * width / total for width in widths]
    floors = [int(math.floor(value)) for value in raw]
    allocation = [base + extra for base, extra in zip(allocation, floors)]
    leftover = remaining - sum(floors)
    order = sorted(
        range(len(spans)),
        key=lambda i: (raw[i] - floors[i], widths[i]),
        reverse=True,
    )
    for index in order[:leftover]:
        allocation[index] += 1
    return allocation


def span_label(spans: list[tuple[float, float]]) -> str:
    def clock(value: float) -> str:
        total = max(0, int(round(value)))
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    return ", ".join(f"{clock(start)}–{clock(end)}" for start, end in spans)


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def factual_context(task: dict[str, Any], spans: list[tuple[float, float]]) -> str:
    index_dir = project_path(task.get("VISUAL_INDEX_DIR"))
    captions_path = index_dir / "semantic_captions.json"
    facts: list[tuple[float, str]] = []
    if captions_path.is_file():
        try:
            captions = json.loads(captions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            captions = {}
        if isinstance(captions, dict):
            for doc_id, metadata in captions.items():
                parts = str(doc_id).split("_")
                if len(parts) != 2 or not isinstance(metadata, dict):
                    continue
                try:
                    window = (float(parts[0]), float(parts[1]))
                except ValueError:
                    continue
                score = sum(overlap(window, span) for span in spans)
                if score <= 0:
                    continue
                caption = str(
                    metadata.get("caption")
                    or metadata.get("clip_caption")
                    or metadata.get("ocr_text")
                    or ""
                ).strip()
                if caption:
                    facts.append((score, re.sub(r"\s+", " ", caption)))
    facts.sort(key=lambda item: (-item[0], item[1]))
    if not facts:
        return (
            "No prior answer hypothesis is supplied. Use only directly visible facts "
            "in the frames and evaluate every option symmetrically."
        )
    joined = " ".join(text[:700] for _, text in facts[:2])[:1200]
    return (
        "Precomputed visual caption facts for these windows (treat as unverified and "
        "check against the pixels; do not assume any answer option): " + joined
    )


def choose_inspection(
    row: dict[str, Any], *, require_correct: bool, require_truncated: bool, fps: float
) -> dict[str, Any] | None:
    candidates = []
    for inspection in row.get("inspections") or []:
        if not isinstance(inspection, dict) or not inspection.get("reconstructed_frame_hit"):
            continue
        spans = spans_of(inspection)
        if not spans:
            continue
        if require_truncated and natural_count(spans, fps) <= 64:
            continue
        is_correct = bool(inspection.get("verdict_correct"))
        if require_correct != is_correct:
            continue
        candidates.append(inspection)
    if not candidates:
        return None
    # Prefer a decisive wrong option over SEARCH_MORE in the failure pool;
    # otherwise use the latest target-reaching call.
    if not require_correct:
        decisive = [item for item in candidates if answer_letter(item.get("verdict"))]
        if decisive:
            candidates = decisive
    return max(candidates, key=lambda item: int(item.get("step") or -1))


def freeze_cases(
    rows: list[dict[str, Any]],
    tasks: dict[tuple[str, str], dict[str, Any]],
    *,
    per_pool: int,
    require_truncated: bool,
    fps: float,
    seed: str,
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = {"failure": [], "protection": []}
    for row in rows:
        key = (str(row.get("video_id") or ""), str(row.get("uid") or ""))
        task = tasks.get(key)
        if task is None or resolve_video(task) is None:
            continue
        final_correct = bool(row.get("final_correct"))
        pool = "protection" if final_correct else "failure"
        if pool == "failure" and row.get("terminal_stage") != "target_frames_reached_but_inspector_verdict_not_correct":
            continue
        inspection = choose_inspection(
            row,
            require_correct=(pool == "protection"),
            require_truncated=require_truncated,
            fps=fps,
        )
        if inspection is None:
            continue
        category = "correct" if pool == "protection" else (
            "wrong_option" if answer_letter(inspection.get("verdict")) else "search_more"
        )
        pools[pool].append(
            {
                "key": key,
                "row": row,
                "task": task,
                "inspection": inspection,
                "category": category,
                "rank": stable_hash(seed, pool, key[0], key[1], inspection.get("step")),
            }
        )

    selected: list[dict[str, Any]] = []
    used_videos: set[str] = set()
    for pool in ("failure", "protection"):
        candidates = sorted(pools[pool], key=lambda item: item["rank"])
        # Failure selection alternates decisive wrong and SEARCH_MORE when possible.
        if pool == "failure":
            groups = {
                name: [item for item in candidates if item["category"] == name]
                for name in ("wrong_option", "search_more")
            }
            ordered = []
            for index in range(max(len(value) for value in groups.values())):
                for name in ("wrong_option", "search_more"):
                    if index < len(groups[name]):
                        ordered.append(groups[name][index])
            candidates = ordered
        count = 0
        for item in candidates:
            video_id = item["key"][0]
            if video_id in used_videos:
                continue
            used_videos.add(video_id)
            selected.append(item)
            count += 1
            if count >= per_pool:
                break
        if count < per_pool:
            raise RuntimeError(f"only selected {count}/{per_pool} cases for {pool}")
    return selected


def run_visual(
    *,
    task: dict[str, Any],
    spans: list[tuple[float, float]],
    context: str,
    frames: list[Path],
    actual: list[float],
    prompt_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if prompt_mode == "mcq":
        prompt = build_visual_inspect_last_step_mcq_prompt(
            span_label(spans),
            "Q1: " + str(task.get("question") or ""),
            context=context or None,
        )
    else:
        prompt = build_visual_inspect_prompt(
            span_label(spans),
            "Q1: " + str(task.get("question") or ""),
            context=context or None,
            prompt_type=0,
        )
    raw = request_formal_chat(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        content=content_with_frames(prompt, frames, actual),
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    prediction = answer_letter(raw)
    return {
        "prediction": prediction,
        "control": "answer" if prediction else "search_more_or_unparsed",
        "raw": str(raw or "")[:2400],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "frame_count": len(frames),
        "timestamps": actual,
        "prompt_mode": prompt_mode,
    }


def run_context_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    task, inspection = case["task"], case["inspection"]
    spans = spans_of(inspection)
    video = resolve_video(task)
    if video is None:
        raise FileNotFoundError(case["key"])
    prompt_mode = (
        "structured"
        if str(inspection.get("answer_text") or "").lstrip().lower().startswith("answer:")
        else "mcq"
    )
    contexts = {
        "source_context": str(inspection.get("context") or ""),
        "question_only": "",
        "factual_context": factual_context(task, spans),
    }
    order = list(contexts)
    shift = int(case["rank"], 16) % len(order)
    order = order[shift:] + order[:shift]
    with tempfile.TemporaryDirectory(prefix="fixed30-context-", dir=args.temp_root) as directory:
        _, _, frames, actual, _ = extract_production_packet(
            video,
            spans,
            fps=args.fps,
            cap=args.frames,
            directory=Path(directory),
        )
        if not frames:
            raise RuntimeError("no frames extracted")
        dynamic_edge = apply_formal_dynamic_resize(frames)
        arms = {
            name: run_visual(
                task=task,
                spans=spans,
                context=contexts[name],
                frames=frames,
                actual=actual,
                prompt_mode=prompt_mode,
                args=args,
            )
            for name in order
        }
    gold = str(case["row"].get("ground_truth") or task.get("ground_truth") or "").strip().upper()
    return {
        "schema_version": "fixed30-context-pilot-v1",
        "experiment": "context",
        "video_id": case["key"][0],
        "uid": case["key"][1],
        "pool": "protection" if case["row"].get("final_correct") else "failure",
        "failure_type": case["category"],
        "inspection_step": inspection.get("step"),
        "source_spans": [list(span) for span in spans],
        "source_context": contexts["source_context"],
        "factual_context": contexts["factual_context"],
        "arm_order": order,
        "prompt_mode": prompt_mode,
        "dynamic_edge": dynamic_edge,
        "arms": arms,
        "ground_truth_for_offline_reducer_only": gold,
        "gold_exposed_to_models": False,
        "time_reference_exposed_to_models": False,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "status": "ok",
    }


def run_sampling_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    task, inspection = case["task"], case["inspection"]
    spans = spans_of(inspection)
    video = resolve_video(task)
    if video is None:
        raise FileNotFoundError(case["key"])
    prompt_mode = (
        "structured"
        if str(inspection.get("answer_text") or "").lstrip().lower().startswith("answer:")
        else "mcq"
    )
    order = ["prefix64", "uniform64", "selective64"]
    shift = int(case["rank"], 16) % len(order)
    order = order[shift:] + order[:shift]
    arms: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="fixed30-sampling-", dir=args.temp_root) as directory:
        directory_path = Path(directory)
        per_span_frames, per_span_ts, prefix_frames, prefix_ts, quotas = extract_production_packet(
            video,
            spans,
            fps=args.fps,
            cap=args.frames,
            directory=directory_path / "decoded",
        )
        uniform_frames, uniform_ts = spread_existing_packet(
            per_span_frames,
            per_span_ts,
            proportional_allocation(spans, args.frames),
        )
        selective_frames, selective_ts = spread_existing_packet(
            per_span_frames,
            per_span_ts,
            quotas,
        )
        packets = {
            "prefix64": (prefix_frames, prefix_ts),
            "uniform64": (uniform_frames, uniform_ts),
            "selective64": (selective_frames, selective_ts),
        }
        dynamic_edges = {
            name: apply_formal_dynamic_resize(packets[name][0]) for name in packets
        }
        for name in order:
            frames, actual = packets[name]
            if not frames:
                raise RuntimeError(f"no frames extracted for {name}")
            arms[name] = run_visual(
                task=task,
                spans=spans,
                context=str(inspection.get("context") or ""),
                frames=frames,
                actual=actual,
                prompt_mode=prompt_mode,
                args=args,
            )
    gold = str(case["row"].get("ground_truth") or task.get("ground_truth") or "").strip().upper()
    return {
        "schema_version": "fixed30-sampling-pilot-v1",
        "experiment": "sampling",
        "video_id": case["key"][0],
        "uid": case["key"][1],
        "pool": "protection" if case["row"].get("final_correct") else "failure",
        "failure_type": case["category"],
        "inspection_step": inspection.get("step"),
        "source_spans": [list(span) for span in spans],
        "source_context": str(inspection.get("context") or ""),
        "natural_frame_count": natural_count(spans, args.fps),
        "prefix_quotas": quotas,
        "arm_order": order,
        "prompt_mode": prompt_mode,
        "dynamic_edges": dynamic_edges,
        "arms": arms,
        "ground_truth_for_offline_reducer_only": gold,
        "gold_exposed_to_models": False,
        "time_reference_exposed_to_models": False,
        "sampler_uses_question_options_gold_or_time_reference": False,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "status": "ok",
    }


def compare(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    fixed = broken = left_correct = right_correct = changed = 0
    for row in rows:
        gold = row["ground_truth_for_offline_reducer_only"]
        left_prediction = row["arms"][left]["prediction"]
        right_prediction = row["arms"][right]["prediction"]
        left_ok, right_ok = left_prediction == gold, right_prediction == gold
        left_correct += int(left_ok)
        right_correct += int(right_ok)
        fixed += int(not left_ok and right_ok)
        broken += int(left_ok and not right_ok)
        changed += int(left_prediction != right_prediction)
    return {
        "left": left,
        "right": right,
        "n": len(rows),
        "left_correct": left_correct,
        "right_correct": right_correct,
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "output_changed": changed,
        "delta_percentage_points": 100.0 * (fixed - broken) / len(rows) if rows else 0.0,
    }


def summarize(rows: list[dict[str, Any]], experiment: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get("experiment") == experiment and row.get("status") == "ok"]
    arms = list(valid[0]["arms"]) if valid else []
    output: dict[str, Any] = {
        "experiment": experiment,
        "n": len(valid),
        "pool_counts": dict(Counter(row["pool"] for row in valid)),
        "failure_type_counts": dict(Counter(row["failure_type"] for row in valid)),
        "arms": {},
        "paired": {},
        "by_pool": {},
    }
    for arm in arms:
        correct = sum(row["arms"][arm]["prediction"] == row["ground_truth_for_offline_reducer_only"] for row in valid)
        output["arms"][arm] = {
            "correct": correct,
            "total": len(valid),
            "accuracy_percent": 100.0 * correct / len(valid) if valid else None,
            "answer_count": sum(bool(row["arms"][arm]["prediction"]) for row in valid),
            "search_more_or_unparsed_count": sum(not row["arms"][arm]["prediction"] for row in valid),
        }
    baseline = "source_context" if experiment == "context" else "prefix64"
    for arm in arms:
        if arm != baseline:
            output["paired"][f"{baseline}_vs_{arm}"] = compare(valid, baseline, arm)
    for pool in ("failure", "protection"):
        subset = [row for row in valid if row["pool"] == pool]
        output["by_pool"][pool] = {
            "n": len(subset),
            "paired": {
                f"{baseline}_vs_{arm}": compare(subset, baseline, arm)
                for arm in arms
                if arm != baseline
            },
        }
    return output


def read_existing(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        key = (str(row.get("experiment") or ""), str(row.get("video_id") or ""), str(row.get("uid") or ""))
        if all(key):
            rows[key] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--experiment", choices=("context", "sampling", "both"), default="both")
    parser.add_argument("--per-pool", type=int, default=12)
    parser.add_argument("--base-url", default="http://127.0.0.1:18405/v1")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", default="fixed30-inspector-pilots-v1")
    parser.add_argument("--temp-root", type=Path, default=ROOT / ".tmp/diagnostics/fixed30_inspector_pilots")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.parquet)
    funnel_rows = load_rows(args.funnel)
    cases: list[tuple[str, dict[str, Any]]] = []
    if args.experiment in {"context", "both"}:
        cases.extend(
            ("context", case)
            for case in freeze_cases(
                funnel_rows,
                tasks,
                per_pool=args.per_pool,
                require_truncated=False,
                fps=args.fps,
                seed=args.seed + ":context",
            )
        )
    if args.experiment in {"sampling", "both"}:
        cases.extend(
            ("sampling", case)
            for case in freeze_cases(
                funnel_rows,
                tasks,
                per_pool=args.per_pool,
                require_truncated=True,
                fps=args.fps,
                seed=args.seed + ":sampling",
            )
        )

    existing = read_existing(args.output)
    completed = list(existing.values())
    pending = [
        (experiment, case)
        for experiment, case in cases
        if (experiment, case["key"][0], case["key"][1]) not in existing
    ]
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {}
            for experiment, case in pending:
                function = run_context_case if experiment == "context" else run_sampling_case
                futures[pool.submit(function, case, args)] = (experiment, case)
            for future in as_completed(futures):
                experiment, case = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "schema_version": "fixed30-inspector-pilot-failure-v1",
                        "experiment": experiment,
                        "video_id": case["key"][0],
                        "uid": case["key"][1],
                        "pool": "protection" if case["row"].get("final_correct") else "failure",
                        "failure_type": case["category"],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                with WRITE_LOCK:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    completed.append(row)
                print(
                    json.dumps(
                        {
                            "experiment": experiment,
                            "uid": row["uid"],
                            "pool": row.get("pool"),
                            "status": row["status"],
                            "elapsed_sec": row.get("elapsed_sec"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    payload = {
        "schema_version": "fixed30-inspector-pilots-summary-v1",
        "context": summarize(completed, "context"),
        "sampling": summarize(completed, "sampling"),
        "failed": [
            {key: row.get(key) for key in ("experiment", "video_id", "uid", "pool", "error")}
            for row in completed
            if row.get("status") != "ok"
        ],
        "integrity": {
            "gold_exposed_to_models": False,
            "time_reference_exposed_to_models": False,
            "timestamps_selected_using_time_reference": False,
            "one_question_per_video_within_each_experiment": True,
        },
    }
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not payload["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
