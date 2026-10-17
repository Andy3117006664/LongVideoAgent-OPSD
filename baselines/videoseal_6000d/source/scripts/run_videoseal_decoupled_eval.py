#!/usr/bin/env python3
"""Run the VideoSEAL Decoupled LVBench evaluation with resumable status files.

The VideoSEAL baseline remains untouched.  This launcher is deliberately a
thin orchestration layer around ``videoseal.runner.per_question_runner``: it
adds an append-only result stream and an atomic status snapshot for the local
dashboard, while keeping the model/tool implementation from the released
baseline.

It is intended to run on the evaluation host, but all paths are configurable
so a dry run can be performed locally as well.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baselines" / "VideoSEAL"
DEFAULT_PARQUET = PROJECT_ROOT / "datasets" / "LVBench" / "videoseal" / "lvbench_full_indexed.parquet"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "videoseal_lvbench"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> tuple[OrderedDict[str, dict[str, Any]], int]:
    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    parse_errors = 0
    if not path.is_file():
        return records, parse_errors
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except (TypeError, ValueError):
                parse_errors += 1
                continue
            if not isinstance(row, dict):
                parse_errors += 1
                continue
            key = result_key(row)
            if key:
                records[key] = row
    return records, parse_errors


def result_key(row: dict[str, Any]) -> str:
    video = str(row.get("video_id") or row.get("video") or "").strip()
    uid = str(row.get("uid") or row.get("qa_uid") or row.get("task_id") or "").strip()
    return f"{video}\x1f{uid}" if video and uid else ""


def answer_key(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if len(text) == 1:
        return text
    match = re.match(r"^[\[(]?([a-z])[\]).:：\s]", text)
    if match:
        return match.group(1)
    return text


def is_correct(prediction: Any, ground_truth: Any) -> bool:
    pred = answer_key(prediction)
    gt = answer_key(ground_truth)
    return bool(pred and gt and pred == gt)


def source_question_types(source: Path) -> dict[str, str]:
    """Load question types when the raw HF parquet is available."""
    if not source.is_file():
        return {}
    try:
        table = pq.read_table(source, columns=["uid", "question_type", "type"])
    except Exception:
        return {}
    cols = {name: table[name].to_pylist() for name in table.column_names}
    out: dict[str, str] = {}
    for idx, uid in enumerate(cols.get("uid", [])):
        value = cols.get("question_type", [None] * table.num_rows)[idx]
        if value in (None, ""):
            value = cols.get("type", [None] * table.num_rows)[idx]
        if uid not in (None, "") and value not in (None, ""):
            out[str(uid).strip()] = str(value).strip()
    return out


def load_tasks(parquet: Path, question_type_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    table = pq.read_table(parquet)
    if "extra_info" not in table.column_names:
        raise RuntimeError(f"{parquet} has no extra_info column")
    tasks: dict[str, dict[str, Any]] = {}
    for row in table["extra_info"].to_pylist():
        info = dict(row or {})
        video = str(info.get("video_id") or "").strip()
        uid = str(info.get("qa_uid") or "").strip()
        question = str(info.get("question") or "").strip()
        video_path = str(info.get("VIDEO_PATH") or "").strip()
        if not (video and uid and question and video_path):
            continue
        key = f"{video}\x1f{uid}"
        tasks[key] = {
            "video_id": video,
            "uid": uid,
            "question": question,
            "ground_truth": str(info.get("ground_truth") or "").strip(),
            "question_type": str(
                info.get("question_type") or info.get("type") or question_type_map.get(uid) or "unknown"
            ).strip(),
        }
    if not tasks:
        raise RuntimeError(f"no valid tasks found in {parquet}")
    return tasks


def harvest_prediction_files(run_root: Path, tasks: dict[str, dict[str, Any]]) -> OrderedDict[str, dict[str, Any]]:
    """Recover predictions written before a launcher interruption."""
    recovered: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for path in sorted(run_root.glob("*/preds/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        key = result_key(row)
        if key not in tasks or not str(row.get("pred") or "").strip():
            continue
        task = tasks[key]
        # A worker can flush its prediction file before the parent has written
        # or the mount has made its trajectory visible.  Such a row is not a
        # complete evaluation artifact: accepting it would make the status
        # look complete while the full-set audit cannot inspect the trace.
        _elapsed, _calls, trajectory = trajectory_details(run_root, task)
        if not trajectory:
            continue
        recovered[key] = {
            "uid": task["uid"],
            "video_id": task["video_id"],
            "question": task["question"],
            "question_type": task["question_type"],
            "prediction": str(row.get("pred") or "").strip(),
            "ground_truth": task["ground_truth"],
            "state": "completed",
            "correct": is_correct(row.get("pred"), task["ground_truth"]),
            "updated_at": utc_now(),
            "source": "recovered_pred_file",
            "trajectory_path": trajectory,
            "tool_calls": _calls,
            "elapsed_seconds": _elapsed,
        }
    return recovered


def trajectory_details(run_root: Path, task: dict[str, Any]) -> tuple[float | None, list[str], str]:
    base = run_root / task["video_id"]
    candidates = sorted(base.glob("*/trajectory.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    for path in candidates[:24]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if str(data.get("uid") or "").strip() != task["uid"]:
            continue
        calls: list[str] = []
        for step in data.get("steps") or []:
            action = step.get("action") if isinstance(step, dict) else None
            if isinstance(action, dict) and action.get("name"):
                calls.append(str(action["name"]))
        elapsed = data.get("elapsed_sec")
        try:
            elapsed_value = float(elapsed) if elapsed is not None else None
        except (TypeError, ValueError):
            elapsed_value = None
        return elapsed_value, calls, str(path)
    return None, [], ""


def group_breakdown(tasks: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]], field: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for key, task in tasks.items():
        name = str(task.get(field) or "unknown")
        bucket = buckets.setdefault(name, {"name": name, "total": 0, "processed": 0, "completed": 0, "answered": 0, "correct": 0, "failed": 0, "skipped": 0})
        bucket["total"] += 1
        row = records.get(key)
        state = str(row.get("state") or "") if row else ""
        if state in {"completed", "failed", "skipped"}:
            bucket["processed"] += 1
        if state == "completed":
            bucket["completed"] += 1
            if str(row.get("prediction") or "").strip():
                bucket["answered"] += 1
            if row.get("correct"):
                bucket["correct"] += 1
        elif state == "failed":
            bucket["failed"] += 1
        elif state == "skipped":
            bucket["skipped"] += 1
    for bucket in buckets.values():
        answered = int(bucket["answered"])
        bucket["accuracy_percent"] = (100.0 * int(bucket["correct"]) / answered) if answered else None
    return sorted(buckets.values(), key=lambda item: str(item["name"]))


def build_status(
    *,
    tasks: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    started_at: str,
    started_epoch: float,
    state: str,
    config: dict[str, Any],
    parse_errors: int,
    current_task: dict[str, Any] | None,
    recent: Iterable[dict[str, Any]],
    message: str = "",
) -> dict[str, Any]:
    total = len(tasks)
    completed = sum(1 for row in records.values() if row.get("state") == "completed")
    failed = sum(1 for row in records.values() if row.get("state") == "failed")
    skipped = sum(1 for row in records.values() if row.get("state") == "skipped")
    processed = min(total, completed + failed + skipped)
    answered = sum(1 for row in records.values() if row.get("state") == "completed" and str(row.get("prediction") or "").strip())
    correct = sum(1 for row in records.values() if row.get("state") == "completed" and row.get("correct"))
    now = time.time()
    elapsed = max(0.0, now - started_epoch)
    if state == "complete":
        elapsed = max(elapsed, 0.0)
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - processed)
    eta = remaining / rate if rate > 0 and remaining else (0.0 if state == "complete" else None)
    recent_rows = list(recent)[-12:]
    accuracy_answered = (100.0 * correct / answered) if answered else None
    accuracy_all_tasks = (100.0 * correct / total) if total else None
    return {
        "schema_version": 1,
        "state": state,
        "status": state,
        "label": "VideoSEAL Decoupled | LVBench Extreme",
        "model": str(config.get("planner_model") or "VideoSEAL_8B"),
        "inspector_model": str(config.get("inspector_model") or "Qwen2.5-VL-7B-Instruct"),
        "mode": "decoupled",
        "started_at": started_at,
        "updated_at": utc_now(),
        "elapsed_seconds": elapsed,
        "total": total,
        "processed": processed,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "remaining": remaining,
        "answered": answered,
        "correct": correct,
        # Keep ``accuracy_percent`` compatible with VideoSEAL's released
        # answered-only aggregator, while exposing the fixed-denominator
        # benchmark metric for an honest partial-run view and final audit.
        "accuracy_percent": accuracy_answered,
        "accuracy_answered_percent": accuracy_answered,
        "accuracy_all_tasks_percent": accuracy_all_tasks,
        "answered_denominator": answered,
        "all_tasks_denominator": total,
        "throughput_per_second": rate,
        "eta_seconds": eta,
        "parse_errors": parse_errors,
        "current_task": current_task,
        "recent_tasks": recent_rows,
        "by_video": group_breakdown(tasks, records, "video_id"),
        "by_type": group_breakdown(tasks, records, "question_type"),
        "dataset": {
            "name": "LVBench",
            "source": "lmms-eval/LVBench",
            "source_url": "https://huggingface.co/datasets/lmms-eval/LVBench",
            "paper": "LVBench: An Extreme Long Video Understanding Benchmark (arXiv:2406.08035)",
            "target_compatible": True,
            "qa_rows": total,
            "videos": len({task["video_id"] for task in tasks.values()}),
        },
        "config": config,
        "message": message,
    }


def env_default(name: str, value: str) -> str:
    return str(os.environ.get(name) or value)


def make_config(args: argparse.Namespace, retrieval_mode: str) -> dict[str, Any]:
    config = {
        "planner_model": env_default("AGENT_LLM_MODEL", "VideoSEAL_8B"),
        "inspector_model": env_default("VISUAL_INSPECT_MODEL", "Qwen2.5-VL-7B-Instruct"),
        "planner_endpoint": env_default("AGENT_LLM_API_BASE", "http://127.0.0.1:18100/v1"),
        "inspector_endpoint": env_default("VISUAL_INSPECT_API_BASE", "http://127.0.0.1:18101/v1"),
        "max_steps": int(args.max_steps),
        "inspect_max_total_images": int(env_default("INSPECT_MAX_TOTAL_IMAGES", "64")),
        "inspect_fps": float(env_default("INSPECT_FPS", "2")),
        "inspect_max_long_edge": int(env_default("INSPECT_MAX_LONG_EDGE", "1280")),
        "inspect_vlm_max_tokens": int(env_default("INSPECT_VLM_MAX_TOKENS", "4096")),
        "inspect_vlm_temperature": float(env_default("INSPECT_VLM_TEMPERATURE", "0.1")),
        # Keep runtime controls in the status snapshot so a resumed run can
        # be audited without relying on the launcher shell environment.
        "planner_timeout": float(env_default("AGENT_LLM_TIMEOUT", "600")),
        "planner_max_tokens": int(env_default("AGENT_LLM_MAX_TOKENS", "2048")),
        "planner_context_limit": int(env_default("AGENT_LLM_CONTEXT_LIMIT", "0")),
        "legacy_history_max_blocks": int(env_default("AGENT_LEGACY_HISTORY_MAX_BLOCKS", "0")),
        "legacy_history_max_chars": int(env_default("AGENT_LEGACY_HISTORY_MAX_CHARS", "0")),
        "time_reference_used_online": env_default("VIDEOSEAL_DISABLE_TIME_REFERENCE", "0") not in {"1", "true", "True", "yes", "on"},
        "mllm_timeout": int(env_default("MLLM_TIMEOUT", "300")),
        "mllm_retry_times": int(env_default("MLLM_RETRY_TIMES", "1")),
        "mllm_retry_delay": float(env_default("MLLM_RETRY_DELAY", "0")),
        "semantic_retrieve_topk": int(env_default("SEMANTIC_RETRIEVE_TOPK", "40")),
        "visual_retrieve_topk": int(env_default("VISUAL_RETRIEVE_TOPK", "30")),
        "retrieve_summary_enabled": env_default("RETRIEVE_SUMMARY_ENABLED", "1"),
        "retrieval_priority_channel": env_default(
            "RETRIEVAL_PRIORITY_CHANNEL", "0"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        "inspect_pending_slot": env_default(
            "INSPECT_PENDING_SLOT", "0"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        "visual_inspect_return_frame_metadata": env_default(
            "VISUAL_INSPECT_RETURN_FRAME_METADATA", "0"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        "visual_inspect_frame_selection": env_default(
            "VISUAL_INSPECT_FRAME_SELECTION",
            env_default("INSPECT_FRAME_SELECTION", "released_round_robin"),
        ),
        "retrieval_requested": str(args.retrieval_mode),
        "retrieval_effective": retrieval_mode,
        "concurrency": int(args.concurrency),
        "task_timeout_sec": float(args.task_timeout_sec),
        "frame_budget": int(env_default("INSPECT_MAX_TOTAL_IMAGES", "64")),
        "tool_budget": int(args.max_steps),
        "strict_final_gate": bool(args.strict_final_gate)
        or env_default("VIDEOSEAL_STRICT_FINAL_GATE", "0") in {"1", "true", "True"},
        "last_step_visual_inspect_prompt_mode": env_default(
            "AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE", "mcq"
        ),
        "paper_reference": "VideoSEAL ICML 2026 Decoupled reported LVBench accuracy: 55.1%",
    }
    if os.getenv("TIER_ENABLE"):
        config["method"] = "TIER"
        config["tier_packet_variant"] = env_default("TIER_PACKET_VARIANT", "tier")
        config["tier_fallback_topk"] = int(env_default("TIER_FALLBACK_TOPK", "16"))
        config["tier_retrieve_enabled"] = env_default("TIER_RETRIEVE_ENABLE", "0") in {"1", "true", "True"}
    if os.getenv("SAHR_ENABLE"):
        config["method"] = "SAHR"
        config["sahr_margin"] = float(env_default("SAHR_MARGIN", "5.0"))
    return config


def configure_environment(args: argparse.Namespace, retrieval_mode: str) -> dict[str, str]:
    env = dict(os.environ)
    defaults = {
        "PYTHONPATH": str(BASELINE_ROOT),
        "AGENT_LLM_BACKEND": "api",
        # The released VideoSEAL Decoupled launcher and the completed
        # reference run use the legacy completions payload.  The local vLLM
        # endpoint rejects the chat-messages payload with HTTP 400.
        "AGENT_API_USE_MESSAGES": "0",
        "AGENT_LLM_API_BASE": "http://127.0.0.1:18100/v1",
        "AGENT_LLM_API_KEY": "local-vllm",
        "AGENT_LLM_MODEL": "VideoSEAL_8B",
        "AGENT_LLM_MAX_TOKENS": "2048",
        "AGENT_LLM_CONTEXT_LIMIT": "0",
        "AGENT_LLM_TEMPERATURE": "0.0",
        # Planner prompts include the accumulated tool transcript.  With the
        # local vLLM service, 60 seconds causes false client retries under
        # normal multimodal load; the released launcher uses a long budget
        # for this API path as well.
        "AGENT_LLM_TIMEOUT": "600",
        "AGENT_MLLM_BACKEND": "openai",
        # Match the released VideoSEAL inference launcher for tool-side MLLM
        # calls.  A short client timeout causes queued visual requests to be
        # retried and can multiply load on the single inspector service.
        "MLLM_BACKEND": "openai",
        "MLLM_MAX_TOKENS": "4096",
        "MLLM_TIMEOUT": "300",
        "MLLM_RETRY_TIMES": "1",
        "MLLM_RETRY_DELAY": "0",
        "VISUAL_INSPECT_BACKEND": "openai",
        "VISUAL_INSPECT_API_BASE": "http://127.0.0.1:18101/v1",
        "VISUAL_INSPECT_API_KEY": "local-vllm",
        "VISUAL_INSPECT_MODEL": "Qwen2.5-VL-7B-Instruct",
        "VISUAL_RETRIEVE_SUM_BACKEND": "openai",
        "VISUAL_RETRIEVE_SUM_API_BASE": "http://127.0.0.1:18101/v1",
        "VISUAL_RETRIEVE_SUM_API_KEY": "local-vllm",
        "VISUAL_RETRIEVE_SUM_MODEL": "Qwen2.5-VL-7B-Instruct",
        "VISUAL_RETRIEVE_SUM_MAX_TOKENS": "2048",
        "VISUAL_RETRIEVE_SUM_TEMPERATURE": "0.1",
        "SEMANTIC_RETRIEVE_MIX": retrieval_mode,
        "SEMANTIC_RETRIEVE_TOPK": "40",
        "VISUAL_RETRIEVE_TOPK": "30",
        "RETRIEVE_SUMMARY_ENABLED": "1",
        "RETRIEVE_SUMMARY_MAX_SPANS": "100",
        "RETRIEVE_EMBED_WEIGHT": "0.8",
        "RETRIEVE_BM25_WEIGHT": "0.2",
        "INSPECT_GLOBAL_ORDER": "1",
        "VISUAL_RETRIEVE_RETURN_SPANS": "1",
        "VISUAL_RETRIEVE_LOG_SPANS": "1",
        "INSPECT_MAX_TOTAL_IMAGES": "64",
        "INSPECT_FPS": "2",
        "INSPECT_MAX_LONG_EDGE": "1280",
        "VISUAL_INSPECT_MAX_LONG_EDGE": "1280",
        "INSPECT_VLM_MAX_TOKENS": "4096",
        "INSPECT_VLM_TEMPERATURE": "0.1",
        "VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE": "1",
        "VISUAL_INSPECT_TOTAL_PIXELS": str(20480 * 32 * 32),
        "VISUAL_INSPECT_MIN_PIXELS": str(16 * 28 * 28),
        "VISUAL_INSPECT_EDGE_MULTIPLE": "32",
        "MAX_STEPS": str(args.max_steps),
        "AGENT_FORCE_LAST_STEP_VISUAL_INSPECT": "1",
        "AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK": "1",
        "AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK": "1",
        "AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE": "mcq",
        "AGENT_PARSE_FAILURE_FALLBACK_TO_C": "0",
        "CLEAN_FRAMES": "0",
        "CLEAN_FRAMES_ASYNC": "1",
        "TASK_TIMEOUT_SEC": str(args.task_timeout_sec),
        # The LVBench parquet stores absolute paths from the original OFS
        # mount.  Allow the worker to use the byte-identical project through
        # its stable NFS alias when that mount endpoint is unavailable.
        "LVQA_PATH_REMAP_FROM": "/path/to/lvqa-experiments",
        "LVQA_PATH_REMAP_TO": str(PROJECT_ROOT),
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    # ``auto`` has already inspected the local index layout.  Make the
    # effective choice explicit so an inherited shell variable cannot make
    # the metadata and the actual retriever disagree.
    env["SEMANTIC_RETRIEVE_MIX"] = retrieval_mode
    if args.strict_final_gate:
        env["VIDEOSEAL_STRICT_FINAL_GATE"] = "1"
    old_pp = env.get("PYTHONPATH", "")
    paths = [str(BASELINE_ROOT), str(PROJECT_ROOT)]
    env["PYTHONPATH"] = ":".join(paths + ([old_pp] if old_pp else []))
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "64")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "16")))
    parser.add_argument("--task-timeout-sec", type=float, default=float(os.environ.get("TASK_TIMEOUT_SEC", "1000")))
    parser.add_argument("--retrieval-mode", choices=["auto", "embed", "bm25"], default=os.environ.get("RETRIEVAL_MODE", "auto"))
    parser.add_argument(
        "--uid",
        action="append",
        default=[],
        help="Restrict the run to one or more QA uids (repeat the option for multiple uids).",
    )
    parser.add_argument(
        "--uids-file",
        type=Path,
        default=None,
        help="Restrict the run to UIDs listed one per line in a file (comments and blank lines are ignored).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Restrict the run to the first N tasks in parquet order; intended for isolated smoke runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict-final-gate",
        action="store_true",
        help="Enforce the declared VideoSEAL rule that <final> is accepted only after a decisive visual_inspect.",
    )
    parser.add_argument(
        "--require-dense-retrieval",
        action="store_true",
        help="Fail before evaluation when the semantic vector index is absent; prevents silent BM25 fallback.",
    )
    args = parser.parse_args()

    parquet = args.parquet.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not parquet.is_file():
        raise SystemExit(f"parquet not found: {parquet}")
    if args.concurrency < 1 or args.max_steps < 1 or args.task_timeout_sec <= 0:
        raise SystemExit("concurrency, max-steps, and task-timeout-sec must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive when provided")

    vector_exists = any((PROJECT_ROOT / "data" / "indexes" / "semantic" / "LVBench").glob("*/*_vectors.npy"))
    if args.require_dense_retrieval and not vector_exists:
        raise SystemExit(
            "dense retrieval required but no semantic vector index was found under "
            f"{PROJECT_ROOT / 'data' / 'indexes' / 'semantic' / 'LVBench'}"
        )
    retrieval_mode = "bm25" if args.retrieval_mode == "auto" and not vector_exists else ("embed" if args.retrieval_mode == "auto" else args.retrieval_mode)
    raw_source = PROJECT_ROOT / "datasets" / "LVBench" / "data" / "train-00000-of-00001.parquet"
    type_map = source_question_types(raw_source)
    tasks = load_tasks(parquet, type_map)
    requested_uids = {str(uid).strip() for uid in args.uid if str(uid).strip()}
    if args.uids_file is not None:
        uids_file = args.uids_file.expanduser().resolve()
        if not uids_file.is_file():
            raise SystemExit(f"uids-file not found: {uids_file}")
        requested_uids.update(
            line.strip()
            for line in uids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if requested_uids:
        tasks = {
            key: task
            for key, task in tasks.items()
            if task["uid"] in requested_uids
        }
        missing_uids = requested_uids - {task["uid"] for task in tasks.values()}
        if missing_uids:
            raise SystemExit(f"uid(s) not found in parquet: {', '.join(sorted(missing_uids))}")
    if args.limit is not None:
        tasks = dict(list(tasks.items())[: args.limit])
    if not tasks:
        raise SystemExit("task selection is empty")
    config = make_config(args, retrieval_mode)
    env = configure_environment(args, retrieval_mode)
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "eval_status.json"
    result_path = run_root / "results.jsonl"
    log_path = run_root / "eval.log"

    existing, parse_errors = read_jsonl(result_path)
    recovered = harvest_prediction_files(run_root, tasks)
    for key, row in recovered.items():
        # A previous runner may have written the per-task prediction before
        # the parent appended its aggregate event.  Persist such recovered
        # rows before constructing the in-memory record set, otherwise the
        # dashboard and the final JSONL can disagree after a resume.
        if key not in existing:
            append_jsonl(result_path, row)
        existing.setdefault(key, row)
    records: OrderedDict[str, dict[str, Any]] = OrderedDict((key, row) for key, row in existing.items() if key in tasks)
    started_at = utc_now()
    started_epoch = time.time()
    if status_path.is_file():
        try:
            old_status = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(old_status, dict):
                started_at = str(old_status.get("started_at") or started_at)
                old_start = old_status.get("started_epoch")
                if old_start is not None:
                    started_epoch = float(old_start)
        except (OSError, TypeError, ValueError):
            pass

    if args.dry_run:
        print(json.dumps({"tasks": len(tasks), "records": len(records), "pending": len(tasks) - len(records), "retrieval": retrieval_mode, "config": config}, ensure_ascii=False, indent=2))
        return 0

    # A task with a previous error is retried; successful/skipped records are
    # left alone so an interrupted run can continue without duplicate work.
    pending = [task for key, task in tasks.items() if key not in records or records[key].get("state") == "failed"]
    pending_uids = sorted({task["uid"] for task in pending})
    pending_file = run_root / "pending_uids.txt"
    pending_file.write_text("\n".join(pending_uids) + ("\n" if pending_uids else ""), encoding="utf-8")

    recent: list[dict[str, Any]] = list(records.values())[-12:]
    status = build_status(tasks=tasks, records=records, started_at=started_at, started_epoch=started_epoch, state="queued" if pending else "complete", config=config, parse_errors=parse_errors, current_task=None, recent=recent, message="queued" if pending else "all tasks already completed")
    status["started_epoch"] = started_epoch
    status["parquet"] = str(parquet)
    atomic_json(status_path, status)
    if not pending:
        return 0

    # ``pending`` is already an exact allow-list: it contains only tasks that
    # have never produced a terminal record or whose latest record is failed.
    # Do not pass ``--skip-existing`` here.  A worker can time out just after
    # writing a non-empty pred file; letting the baseline inspect that stale
    # file would incorrectly skip a task that this launcher has classified as
    # failed.  Successful tasks are excluded by ``pending`` itself.
    cmd = [
        sys.executable,
        "-m",
        "videoseal.runner.per_question_runner",
        "--parquet",
        str(parquet),
        "--save-runs",
        str(run_root),
        "--uids-file",
        str(pending_file),
        "--concurrency",
        str(args.concurrency),
        "--max-steps",
        str(args.max_steps),
        "--task-timeout-sec",
        str(args.task_timeout_sec),
        "--no-aggregate",
    ]
    started_child = time.time()
    current_task: dict[str, Any] | None = None
    recent_keys: list[str] = list(records.keys())[-12:]
    env["EVAL_RUN_ROOT"] = str(run_root)
    env["EVAL_MODE"] = "decoupled"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{utc_now()}] START mode=decoupled tasks={len(tasks)} pending={len(pending)} concurrency={args.concurrency} retrieval={retrieval_mode}\n")
        log.flush()
        atomic_json(status_path, build_status(tasks=tasks, records=records, started_at=started_at, started_epoch=started_epoch, state="running", config=config, parse_errors=parse_errors, current_task=None, recent=[records[k] for k in recent_keys if k in records], message="evaluation running") | {"started_epoch": started_epoch, "parquet": str(parquet)})
        child = subprocess.Popen(cmd, cwd=str(BASELINE_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def stop_child(_signum: int, _frame: Any) -> None:
            if child.poll() is None:
                child.terminate()

        signal.signal(signal.SIGTERM, stop_child)
        signal.signal(signal.SIGINT, stop_child)

        for line in child.stdout or []:
            log.write(line)
            log.flush()
            stripped = line.strip()
            try:
                event = json.loads(stripped)
            except (TypeError, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            uid = str(event.get("uid") or event.get("qa_uid") or "").strip()
            video = str(event.get("video_id") or event.get("video") or "").strip()
            if not uid:
                continue
            if not video:
                matches = [task for task in tasks.values() if task["uid"] == uid]
                video = matches[0]["video_id"] if len(matches) == 1 else ""
            key = f"{video}\x1f{uid}" if video else ""
            if key not in tasks:
                continue
            task = tasks[key]
            prediction = str(event.get("pred") or event.get("prediction") or "").strip()
            error = str(event.get("error") or "").strip()
            elapsed, calls, trajectory = trajectory_details(run_root, task)
            row = {
                "uid": uid,
                "video_id": video,
                "question": task["question"],
                "question_type": task["question_type"],
                "prediction": prediction,
                "ground_truth": task["ground_truth"],
                "state": "failed" if error or not prediction else "completed",
                "correct": bool(prediction) and is_correct(prediction, task["ground_truth"]),
                "elapsed_seconds": elapsed,
                "tool_calls": calls,
                "trajectory_path": trajectory,
                "error": error,
                "updated_at": utc_now(),
            }
            records[key] = row
            append_jsonl(result_path, row)
            recent_keys.append(key)
            recent_keys = recent_keys[-12:]
            current_task = None
            snapshot = build_status(tasks=tasks, records=records, started_at=started_at, started_epoch=started_epoch, state="running", config=config, parse_errors=parse_errors, current_task=current_task, recent=[records[k] for k in reversed(recent_keys) if k in records], message=f"processed {sum(1 for r in records.values() if r.get('state') in {'completed','failed','skipped'})}/{len(tasks)}")
            snapshot["started_epoch"] = started_epoch
            snapshot["parquet"] = str(parquet)
            atomic_json(status_path, snapshot)

        return_code = child.wait()
        # A worker can finish writing its prediction just before the parent
        # process exits; harvest once more before choosing the terminal state.
        for key, row in harvest_prediction_files(run_root, tasks).items():
            if key not in records or records[key].get("state") != "completed":
                records[key] = row
                append_jsonl(result_path, row)
        done = all(key in records and records[key].get("state") in {"completed", "skipped"} for key in tasks)
        terminal_state = "complete" if done and return_code == 0 else ("paused" if return_code in {130, -signal.SIGINT} else "failed")
        message = "evaluation complete" if terminal_state == "complete" else f"runner exited with code {return_code}"
        log.write(f"[{utc_now()}] END state={terminal_state} return_code={return_code}\n")
        log.flush()
        final_snapshot = build_status(tasks=tasks, records=records, started_at=started_at, started_epoch=started_epoch, state=terminal_state, config=config, parse_errors=parse_errors, current_task=None, recent=[records[k] for k in reversed(recent_keys) if k in records], message=message)
        final_snapshot["started_epoch"] = started_epoch
        final_snapshot["parquet"] = str(parquet)
        atomic_json(status_path, final_snapshot)
    return 0 if terminal_state == "complete" else (return_code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
