"""Fixed300 evaluation through the captured Planner's original API client.

One vLLM server can expose the base model and a trained LoRA under different
request model names. Run this entry point once for each arm, with identical
server/tool settings. Tasks execute in independent spawned processes because
the captured tools and task context use process environment variables.

This module neither imports the training policy/Torch nor reads or scores gold.
The unchanged conditional Observer wrapper is a separate subsequent stage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
from typing import Any, Callable

from .evaluate import (EvaluationTask, _find_trajectory, _prediction, _source_paths,
                       load_evaluation_tasks)
from .identities import resolve_identity_mapping


PLANNER_TOKENS = 4096
MAX_STEPS = 16


def _question_hash(entry: EvaluationTask) -> str:
    return hashlib.sha256(entry.task["question"].encode()).hexdigest()


def _empty_row(entry: EvaluationTask) -> dict:
    return {
        "ordinal": entry.ordinal, "video_id": entry.task["video_id"],
        "uid": entry.task["uid"], "input_uid": entry.input_uid,
        "input_video_id": entry.input_video_id,
        "question_sha256": _question_hash(entry),
        "prediction": "", "state": "failed", "trajectory_path": None,
    }


def _failure(entry: EvaluationTask, error_type: str, error: str,
             elapsed_seconds: float = 0.0) -> dict:
    return dict(_empty_row(entry), error_type=error_type, error=str(error)[:1000],
                elapsed_seconds=round(elapsed_seconds, 6),
                updated_at=datetime.now(timezone.utc).isoformat())


def native_api_rollout(task: dict, *, save_dir: Path, planner_endpoint: str,
                       planner_model: str):
    """Use the original MLLMClient; never manufacture Planner completions/tokens."""
    from .rollout import RolloutResult, _load_videoseal, _task_environment

    agent_class, build_tool_map = _load_videoseal()
    with _task_environment(task, temperature=0.0, max_new_tokens=PLANNER_TOKENS):
        # _task_environment normally points at a dummy endpoint for local HF
        # training. Only this new native-API path replaces those three values.
        os.environ["AGENT_LLM_API_BASE"] = planner_endpoint
        os.environ["AGENT_LLM_API_KEY"] = "local-vllm"
        os.environ["AGENT_LLM_MODEL"] = planner_model
        tools = {name: tool for name, tool in build_tool_map().items()
                 if name in ("visual_retrieve", "visual_inspect")}
        if set(tools) != {"visual_retrieve", "visual_inspect"}:
            raise RuntimeError("The captured baseline must expose both original visual tools")
        agent = agent_class(tools=tools, llm_backend="api")
        # Keep agent.llm unchanged: it owns original payloads, EOS behavior,
        # token usage, retry policy and the legacy tag-based action transport.
        result = agent.run(
            question=str(task["question"]), uid=str(task["uid"]),
            video_id=str(task["video_id"]), video_path=str(task["video_path"]),
            visual_index=task.get("visual_index") or None, groundtruth=None,
            max_steps=MAX_STEPS, save_dir=str(save_dir),
        )
        trajectory = asdict(agent.traj) if is_dataclass(agent.traj) else agent.traj
    if not isinstance(trajectory, dict) or not isinstance(result, dict):
        raise RuntimeError("The original agent did not return an audit trajectory/result")
    return RolloutResult([], trajectory, result)


def _api_task_worker(entry: EvaluationTask, output: Path, settings: dict,
                     result_queue) -> None:
    """A fresh process owns a single task, including its environment and timeout."""
    started = time.monotonic()
    row = _empty_row(entry)
    task = entry.task
    try:
        _, source = _source_paths()
        from .train import configure_baseline
        configure_baseline(source, inspector_endpoint=settings["inspector_endpoint"],
                           compactor_endpoint=settings["compactor_endpoint"])
        if not Path(task["video_path"]).is_file() or not Path(task["visual_index"]).exists():
            raise FileNotFoundError("Evaluation video/index is unavailable")
        rollout = native_api_rollout(
            task, save_dir=output / task["video_id"],
            planner_endpoint=settings["planner_endpoint"],
            planner_model=settings["planner_model"],
        )
        path = _find_trajectory(output, task, rollout.trajectory)
        row["trajectory_path"] = str(path) if path else None
        row["prediction"] = _prediction(rollout.result, task["question"])
        row["tool_calls"] = [
            step["action"]["name"] for step in rollout.trajectory.get("steps", [])
            if isinstance(step.get("action"), dict) and step["action"].get("name")
        ]
        if row["prediction"]:
            row["state"] = "completed"
        else:
            row.update(error_type="NoFinalPrediction", error="Original agent returned no final prediction")
    except Exception as exc:
        row.update(error_type=type(exc).__name__, error=str(exc)[:1000])
        try:
            path = _find_trajectory(output, task)
            row["trajectory_path"] = str(path) if path else None
        except Exception:
            pass
    row.update(elapsed_seconds=round(time.monotonic() - started, 6),
               updated_at=datetime.now(timezone.utc).isoformat())
    result_queue.put({"ordinal": entry.ordinal, "row": row})


def _validate_worker_row(entry: EvaluationTask, row: Any) -> dict:
    """The parent owns identity/denominator; never trust a worker's task labels."""
    expected = _empty_row(entry)
    if not isinstance(row, dict) or any(row.get(key) != expected[key]
                                       for key in ("ordinal", "uid", "video_id", "question_sha256")):
        return _failure(entry, "WorkerIdentityMismatch", "Worker UID/ordinal/video/question hash differs")
    if row.get("state") not in ("completed", "failed"):
        return _failure(entry, "WorkerProtocolError", "Worker returned an unsupported state")
    if not isinstance(row.get("prediction"), str):
        return _failure(entry, "WorkerProtocolError", "Worker prediction must be text")
    if row["state"] == "completed" and not row["prediction"].strip():
        return _failure(entry, "WorkerProtocolError", "Completed worker has no prediction")
    # Reassert these values rather than accepting aliases supplied by a worker.
    return dict(row, prediction=row["prediction"] if row["state"] == "completed" else "",
                ordinal=entry.ordinal, uid=expected["uid"], video_id=expected["video_id"],
                input_uid=expected["input_uid"], input_video_id=expected["input_video_id"],
                question_sha256=expected["question_sha256"])


def _publish_results(entries: list[EvaluationTask], rows: dict[int, dict], output: Path,
                     *, smoke: bool) -> dict:
    """Only a complete identity-checked reduction may become Observer input."""
    expected_ordinals = {entry.ordinal for entry in entries}
    if (len(expected_ordinals) != len(entries)
            or len({entry.task["uid"] for entry in entries}) != len(entries)):
        raise ValueError("Evaluation identities/ordinals must be unique")
    if not smoke and (len(entries) != 300 or expected_ordinals != set(range(300))):
        raise ValueError("Only the complete fixed300 may publish results.jsonl")
    if set(rows) != expected_ordinals:
        raise ValueError("Cannot publish incomplete or unexpected worker results")
    filename = "SMOKE_RESULTS.jsonl" if smoke else "results.jsonl"
    checked = [_validate_worker_row(entry, rows[entry.ordinal])
               for entry in sorted(entries, key=lambda e: e.ordinal)]
    for row in checked:
        row.update(fixed300_complete=not smoke, smoke=smoke)
    output.mkdir(parents=True, exist_ok=True)
    with (output / filename).open("x", encoding="utf-8") as handle:
        for row in checked:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    failures = sum(row["state"] != "completed" for row in checked)
    summary = {
        "state": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
        "selected_rows": len(entries), "result_rows": len(checked),
        "completed_rows": len(checked) - failures, "failed_rows": failures,
        "fixed300_complete": not smoke, "observer_compatible": not smoke,
        "result_file": filename, "planner_temperature": 0.0,
        "planner_max_new_tokens": PLANNER_TOKENS, "max_steps": MAX_STEPS,
        "engine": "original-api-client", "gold_used": False,
        "accuracy_evaluated": False, "conditional_observer_run": False,
    }
    (output / "EVALUATION_RESULT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _stop_process(process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=.5)


def evaluate_api_tasks(entries: list[EvaluationTask], output: Path, *, settings: dict,
                       workers: int = 8, task_timeout: float = 2400., smoke: bool = False,
                       worker_target: Callable | None = None) -> dict:
    """Supervise isolated task processes, retaining failed/timeout result rows.

    worker_target is a CPU-only test seam. Formal CLI always uses the captured
    native API worker. A partial journal never has the Observer's results name.
    """
    if workers < 1 or not math.isfinite(task_timeout) or task_timeout <= 0:
        raise ValueError("Positive worker count and task timeout required")
    expected = {entry.ordinal: entry for entry in entries}
    if (not entries or len(expected) != len(entries)
            or len({entry.task["uid"] for entry in entries}) != len(entries)):
        raise ValueError("Unique nonempty evaluation identities/ordinals required")
    if not smoke and (len(entries) != 300 or set(expected) != set(range(300))):
        raise ValueError("Formal evaluation requires the complete fixed300")
    output.mkdir(parents=True, exist_ok=True)
    if (output / ("SMOKE_RESULTS.jsonl" if smoke else "results.jsonl")).exists():
        raise FileExistsError("Refusing to overwrite evaluation predictions")
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    pending, active, rows = iter(entries), {}, {}
    exhausted = False
    target = worker_target or _api_task_worker
    journal = (output / "PARTIAL_RESULTS.jsonl").open("x", encoding="utf-8")

    def record(entry, row):
        checked = _validate_worker_row(entry, row)
        rows[entry.ordinal] = checked
        journal.write(json.dumps(checked, ensure_ascii=False, allow_nan=False) + "\n")
        journal.flush()

    try:
        while active or not exhausted:
            while not exhausted and len(active) < workers:
                try:
                    entry = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                process = context.Process(target=target, args=(entry, output, settings, result_queue))
                started = time.monotonic()
                try:
                    process.start()
                except Exception as exc:
                    record(entry, _failure(entry, "WorkerStartError", str(exc)))
                    continue
                active[entry.ordinal] = {"process": process, "entry": entry, "started": started,
                                         "exited_at": None}
            if not active:
                continue
            try:
                message = result_queue.get(timeout=.05)
            except queue.Empty:
                message = None
            if isinstance(message, dict) and message.get("ordinal") in active:
                ordinal = message["ordinal"]
                running = active.pop(ordinal)
                record(running["entry"], message.get("row"))
                running["process"].join(timeout=.1)
                _stop_process(running["process"])
            now = time.monotonic()
            for ordinal, running in list(active.items()):
                process, entry = running["process"], running["entry"]
                elapsed = now - running["started"]
                if elapsed > task_timeout:
                    _stop_process(process)
                    record(entry, _failure(entry, "TaskTimeout", f"timeout>{task_timeout}s", elapsed))
                    del active[ordinal]
                elif not process.is_alive():
                    # Let a queued final result reach the parent before
                    # classifying an exited worker as a result-less failure.
                    if running["exited_at"] is None:
                        running["exited_at"] = now
                    elif now - running["exited_at"] > .5:
                        process.join(timeout=0)
                        record(entry, _failure(entry, "WorkerExited", f"exitcode={process.exitcode}; no result", elapsed))
                        del active[ordinal]
    finally:
        for running in active.values():
            _stop_process(running["process"])
        journal.close()
        result_queue.close()
        result_queue.join_thread()
    return _publish_results(entries, rows, output, smoke=smoke)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--identity-mapping", type=Path,
                        help="answer-free fixed300 mapping; defaults to the archived public mapping")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planner-endpoint", required=True)
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--inspector-endpoint", required=True)
    parser.add_argument("--compactor-endpoint", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--task-timeout", type=float, default=2400.)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, choices=(MAX_STEPS,))
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or not math.isfinite(args.task_timeout) or args.task_timeout <= 0:
        parser.error("workers and task-timeout must be positive")
    args.tasks, args.output = args.tasks.expanduser().resolve(), args.output.expanduser().resolve()
    repo, source = _source_paths()
    mapping = resolve_identity_mapping(args.identity_mapping, repo=repo)
    args.identity_mapping = mapping.path
    entries = load_evaluation_tasks(args.tasks, mapping, smoke_limit=args.smoke_limit)
    args.output.mkdir(parents=True, exist_ok=False)
    from .train import configure_baseline
    baseline_environment = configure_baseline(
        source, inspector_endpoint=args.inspector_endpoint,
        compactor_endpoint=args.compactor_endpoint,
    )
    baseline_environment.update(AGENT_LLM_API_BASE=args.planner_endpoint,
                                AGENT_LLM_MODEL=args.planner_model,
                                AGENT_LLM_API_KEY="local-vllm")
    settings = {key: getattr(args, key) for key in ("planner_endpoint", "planner_model",
                                                   "inspector_endpoint", "compactor_endpoint")}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(mapping.metadata())
    config.update(
        task_manifest_sha256=hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        selected_ordinals=[entry.ordinal for entry in entries],
        fixed300_complete=args.smoke_limit is None, planner_temperature=0.0,
        planner_max_new_tokens=PLANNER_TOKENS, planner_context_limit=40960,
        planner_stop="original server/model EOS; no additional stop strings",
        planner_transport="original MLLMClient; legacy tagged actions; chat/completions",
        engine="original-api-client", arm=args.planner_model,
        gold_used=False, accuracy_evaluated=False, conditional_observer_run=False,
        baseline_environment=baseline_environment,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    summary = evaluate_api_tasks(entries, args.output, settings=settings, workers=args.workers,
                                 task_timeout=args.task_timeout, smoke=args.smoke_limit is not None)
    print(json.dumps(summary), flush=True)
    return 0 if summary["failed_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
