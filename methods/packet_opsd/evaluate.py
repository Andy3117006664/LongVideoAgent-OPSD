"""Greedy, answer-free evaluation of a trained Planner on the fixed 300 tasks.

The original VideoSEAL agent, Inspector and compactor run unchanged. This
entry point produces search predictions, not accuracy or conditional Observer
predictions. A smoke subset deliberately uses SMOKE_RESULTS.jsonl so that the
fixed-300 Observer wrapper cannot mistake it for a complete search run.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

from .identities import IdentityMapping, bound_mapping, resolve_identity_mapping


TASK_FIELDS = {
    "uid", "video_id", "question", "video_path", "visual_index", "subtitle_path",
    "summary_dir", "summary_file", "benchmark", "video_duration_sec",
}


@dataclass(frozen=True)
class EvaluationTask:
    task: dict[str, Any]
    ordinal: int
    input_uid: str
    input_video_id: str


def _source_paths() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[2]
    source = repo / "baselines" / "videoseal_6000d" / "source"
    for path in (source, source / "baselines" / "VideoSEAL"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo, source


def load_evaluation_tasks(
    path: Path, mapping_path: Path | IdentityMapping, *, smoke_limit: int | None = None,
) -> list[EvaluationTask]:
    """Validate public identities and question hashes without any gold input.

    Bare qid, sample_uid and a runtime prefix ending in ``:qid`` identify the
    same task. Output UID is always mapping.qid, matching the published public
    parquet's qa_uid and conditional Observer. Video aliases are similarly
    normalized to mapping.video_id; arbitrary alternative videos are rejected.
    """
    mapping = bound_mapping(mapping_path).rows
    by_uid: dict[str, dict] = {}
    ordinals, canonical_uids = set(), set()
    for row in mapping:
        ordinal, qid = int(row["ordinal"]), str(row["qid"])
        if ordinal in ordinals or qid in canonical_uids:
            raise ValueError("Duplicate ordinal or qid in the published identity mapping")
        ordinals.add(ordinal)
        canonical_uids.add(qid)
        for alias in (qid, str(row.get("sample_uid") or qid)):
            if alias in by_uid and by_uid[alias]["qid"] != row["qid"]:
                raise ValueError("Ambiguous UID alias in the published identity mapping")
            by_uid[alias] = row
    if ordinals != set(range(300)):
        raise ValueError("The published mapping ordinals must cover 0 through 299")
    if smoke_limit is not None and not 1 <= smoke_limit < 300:
        raise ValueError("smoke_limit must be between 1 and 299")

    entries, seen = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        task = json.loads(line)
        if not isinstance(task, dict):
            raise ValueError(f"Task line {line_number} must be an object")
        unknown = set(task) - TASK_FIELDS
        if unknown:
            raise ValueError(f"Task line {line_number} has unexpected fields: {sorted(unknown)}")
        if not all(isinstance(task.get(key), str) and task[key].strip()
                   for key in ("uid", "video_id", "question", "video_path", "visual_index")):
            raise ValueError(f"Task line {line_number} is incomplete")
        input_uid, input_video = task["uid"].strip(), task["video_id"].strip()
        identity = by_uid.get(input_uid) or by_uid.get(input_uid.rsplit(":", 1)[-1])
        if identity is None:
            raise ValueError(f"Task line {line_number} is outside the fixed300 identities")
        qid = str(identity["qid"])
        if qid in seen:
            raise ValueError(f"Duplicate normalized evaluation UID at line {line_number}")
        aliases = {str(identity[key]) for key in ("video_id", "original_video_id") if identity.get(key)}
        if input_video not in aliases:
            raise ValueError(f"Task line {line_number} has the wrong video identity")
        question_hash = hashlib.sha256(task["question"].encode()).hexdigest()
        if question_hash != identity["question_sha256"]:
            raise ValueError(f"Task line {line_number} question hash differs from the fixed300 protocol")
        normalized = dict(task)
        normalized.update(uid=qid, video_id=str(identity["video_id"]))
        # File availability is checked per task during evaluation, so a missing
        # asset produces a failed result row instead of shrinking the denominator.
        for key in ("video_path", "visual_index", "subtitle_path", "summary_dir", "summary_file"):
            if normalized.get(key):
                normalized[key] = str(Path(normalized[key]).expanduser().resolve())
        seen.add(qid)
        entries.append(EvaluationTask(normalized, int(identity["ordinal"]), input_uid, input_video))
    if not entries:
        raise ValueError("Empty evaluation task manifest")
    if smoke_limit is None and seen != canonical_uids:
        raise ValueError("Formal evaluation requires all 300 unique fixed300 identities")
    entries.sort(key=lambda entry: entry.ordinal)
    return entries if smoke_limit is None else entries[:smoke_limit]


def _prediction(result: dict, question: str) -> str:
    # The same gold-free MCQ parsing as the original per-question runner.
    _source_paths()
    from videoseal.utils.agent.tool_agent_parsing import strip_multiple_choice_options
    from videoseal.utils.lvbench_io import parse_choice_letter, parse_choice_letter_smart

    text = str(result.get("final") or result.get("answer") or "").strip()
    if text and strip_multiple_choice_options(question.strip()) != question.strip():
        return (parse_choice_letter_smart(text, question) or parse_choice_letter(text) or text).strip().upper()
    return text


def _find_trajectory(output: Path, task: dict, trajectory: dict | None = None) -> Path | None:
    base = output / task["video_id"]
    if trajectory is not None:
        if str(trajectory.get("uid") or "") != task["uid"]:
            raise ValueError("Returned trajectory UID differs from the normalized evaluation task")
        if str(trajectory.get("video_id") or "") != task["video_id"]:
            raise ValueError("Returned trajectory video differs from the normalized evaluation task")
        run_id = str(trajectory.get("run_id") or "")
        if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            raise ValueError("Returned trajectory has an invalid run_id")
        target = base / run_id / "trajectory.json"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target.resolve()
    for candidate in sorted(base.glob("*/trajectory.json"), reverse=True):
        try:
            saved = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(saved.get("uid") or "") == task["uid"]:
            return candidate.resolve()
    return None


def evaluate_tasks(
    entries: list[EvaluationTask], policy: Any, output: Path, *,
    max_steps: int = 16, smoke: bool = False,
    run_task_fn: Callable | None = None, initial_error: dict[str, str] | None = None,
) -> dict:
    """Write one row per selected task, including failures; never score answers.

    ``run_task_fn`` is an optional test seam for CPU scripted checks. Runtime
    uses the same fresh-rollout adapter as training, with temperature zero.
    """
    if run_task_fn is None:
        from .rollout import run_task
        run_task_fn = run_task
    output.mkdir(parents=True, exist_ok=True)
    filename = "SMOKE_RESULTS.jsonl" if smoke else "results.jsonl"
    if not smoke and (len(entries) != 300 or {entry.ordinal for entry in entries} != set(range(300))):
        raise ValueError("Only a complete validated fixed300 run may produce results.jsonl")
    failures = 0
    with (output / filename).open("x", encoding="utf-8") as handle:
        for entry in entries:
            task, started = entry.task, time.monotonic()
            row = {
                "ordinal": entry.ordinal, "video_id": task["video_id"], "uid": task["uid"],
                "input_uid": entry.input_uid, "input_video_id": entry.input_video_id,
                "prediction": "", "state": "failed", "trajectory_path": None,
                "fixed300_complete": not smoke, "smoke": smoke,
            }
            try:
                if initial_error is not None:
                    row.update(initial_error)
                else:
                    if not Path(task["video_path"]).is_file() or not Path(task["visual_index"]).exists():
                        raise FileNotFoundError("Evaluation video/index is unavailable")
                    rollout = run_task_fn(
                        task, policy, save_dir=output / task["video_id"],
                        max_steps=max_steps, temperature=0.0, max_new_tokens=4096,
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
                path = _find_trajectory(output, task)
                row["trajectory_path"] = str(path) if path else None
            if row["state"] != "completed":
                failures += 1
            row.update(elapsed_seconds=round(time.monotonic() - started, 6),
                       updated_at=datetime.now(timezone.utc).isoformat())
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
    summary = {
        "state": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
        "selected_rows": len(entries), "result_rows": len(entries),
        "completed_rows": len(entries) - failures, "failed_rows": failures,
        "fixed300_complete": not smoke, "observer_compatible": not smoke,
        "result_file": filename, "planner_temperature": 0.0,
        "training_gold_used": False, "accuracy_evaluated": False,
        "conditional_observer_run": False,
    }
    (output / "EVALUATION_RESULT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _adapter_directory(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    if (path / "student" / "adapter_config.json").is_file():
        return path / "student"
    raise FileNotFoundError("--adapter must contain adapter_config.json or a saved student/ adapter")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--identity-mapping", type=Path,
                        help="answer-free fixed300 mapping; defaults to the archived public mapping")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inspector-endpoint", required=True)
    parser.add_argument("--compactor-endpoint", required=True)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=40960)
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args(argv)
    if args.max_steps < 1 or args.max_context < 1:
        parser.error("max-steps/max-context must be positive")
    if args.smoke_limit is not None and not 1 <= args.smoke_limit < 300:
        parser.error("smoke-limit must be between 1 and 299")
    for key in ("model", "adapter", "tasks", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    repo, source = _source_paths()
    mapping = resolve_identity_mapping(args.identity_mapping, repo=repo)
    args.identity_mapping = mapping.path
    entries = load_evaluation_tasks(args.tasks, mapping, smoke_limit=args.smoke_limit)
    args.output.mkdir(parents=True, exist_ok=False)
    from .train import configure_baseline
    baseline_environment = configure_baseline(
        source, inspector_endpoint=args.inspector_endpoint, compactor_endpoint=args.compactor_endpoint,
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(mapping.metadata())
    config.update({
        "task_manifest_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "selected_ordinals": [entry.ordinal for entry in entries],
        "fixed300_complete": args.smoke_limit is None,
        "planner_temperature": 0.0, "planner_max_new_tokens": 4096,
        "gold_used": False, "accuracy_evaluated": False,
        "baseline_environment": baseline_environment,
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    policy, initial_error = None, None
    try:
        from .policy import HFPolicy
        adapter = _adapter_directory(args.adapter)
        policy = HFPolicy.load(str(args.model), student_adapter=str(adapter), max_context=args.max_context)
    except Exception as exc:
        initial_error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
    summary = evaluate_tasks(
        entries, policy, args.output, max_steps=args.max_steps,
        smoke=args.smoke_limit is not None, initial_error=initial_error,
    )
    print(json.dumps(summary), flush=True)
    return 0 if summary["failed_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
