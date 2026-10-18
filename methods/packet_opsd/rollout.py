"""Fresh Planner rollouts through the captured 6000D VideoSEAL agent.

Only the Planner transport is replaced.  The captured tool implementations,
prompts, argument routing, Inspector feedback and termination code execute as
usual.  Inspector/compactor service settings must be supplied by the caller
using the published baseline configuration.  This synchronous adapter changes
process environment variables while a task runs; use one process per worker.

No gold fields are read.  Forced full-video fallback steps remain in the audit
trajectory but are not Planner actions and never become training records.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterator, Protocol


class Policy(Protocol):
    """Structural protocol; importing this module does not import torch."""

    def sample(
        self,
        messages: list[dict],
        *,
        adapter: str = "student",
        prefix: str = "",
        max_new_tokens: int,
        temperature: float,
    ) -> Any:
        """Return Completion(text, token_ids, prompt_ids, complete)."""
        ...


@dataclass
class RolloutResult:
    factual_records: list[dict[str, Any]]
    trajectory: dict[str, Any]
    result: dict[str, Any]


class _TrainingPrefixReady(BaseException):
    """Private control flow, caught only immediately outside the agent loop."""

    def __init__(self, eligible_records: int):
        self.eligible_records = eligible_records


class PlannerShim:
    """The original legacy ``generate_text`` interface backed by HFPolicy.

    ``calls`` retains the actual Completion objects rather than re-tokenizing
    the returned text.  Messages are exactly the original system/user text at
    that generation, including the tool schema and original history packing.
    """

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.calls: list[dict[str, Any]] = []
        self._last_usage: dict[str, int] | None = None
        self.before_generate: Callable[[], None] | None = None

    def generate_text(
        self,
        prompt: str,
        *,
        response_json: bool = False,
        with_usage: bool = False,
        system_message: str = "",
        **kwargs: Any,
    ) -> Any:
        if self.before_generate is not None:
            self.before_generate()
        if response_json:
            raise ValueError("The captured Planner uses the original text tool-call format.")
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        completion = self.policy.sample(
            deepcopy(messages),
            adapter="student",
            prefix="",
            max_new_tokens=int(kwargs["max_tokens"]),
            temperature=float(kwargs["temperature"]),
        )
        for field in ("text", "token_ids", "prompt_ids", "complete"):
            if not hasattr(completion, field):
                raise TypeError(f"Policy.sample returned a Completion without {field!r}")
        if not isinstance(completion.text, str):
            raise TypeError("Completion.text must be str")
        self.calls.append({"messages": deepcopy(messages), "completion": completion})
        self._last_usage = {
            "input_tokens": len(completion.prompt_ids),
            "output_tokens": len(completion.token_ids),
            "total_tokens": len(completion.prompt_ids) + len(completion.token_ids),
        }
        if with_usage:
            return {"text": completion.text, "usage": dict(self._last_usage)}
        return completion.text

    def get_last_usage(self) -> dict[str, int] | None:
        return dict(self._last_usage) if self._last_usage is not None else None

    def get_last_response(self) -> None:
        return None


def _load_videoseal() -> tuple[Any, Any]:
    """Import only the captured 6000D inference tree, never the old 3B loop."""
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "baselines" / "videoseal_6000d" / "source"
    baseline = source / "baselines" / "VideoSEAL"
    if not baseline.is_dir():
        raise FileNotFoundError(f"Captured VideoSEAL source not found: {baseline}")
    for entry in (str(source), str(baseline)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from videoseal.agents.tool_agent import SimpleToolAgent
    from videoseal.tools.tool_map import build_tool_map

    return SimpleToolAgent, build_tool_map


@contextmanager
def _task_environment(
    task: dict[str, Any], *, temperature: float, max_new_tokens: int
) -> Iterator[None]:
    # The legacy text branch is the published 6000D Planner path. The dummy
    # Planner endpoint is never contacted: its client is replaced before run().
    settings: dict[str, str | None] = {
        "AGENT_LLM_BACKEND": "api",
        "AGENT_API_USE_MESSAGES": "0",
        "AGENT_LLM_MODEL": os.environ.get("AGENT_LLM_MODEL") or "Qwen3-8B",
        "AGENT_LLM_API_BASE": "http://127.0.0.1:9/v1",
        "AGENT_LLM_API_KEY": "unused-local-policy",
        "AGENT_LLM_MAX_TOKENS": str(max_new_tokens),
        "AGENT_LLM_TEMPERATURE": str(temperature),
        "AGENT_LLM_TIMEOUT": os.environ.get("AGENT_LLM_TIMEOUT") or "600",
        "AGENT_LEGACY_HISTORY_MAX_BLOCKS": "0",
        "AGENT_LEGACY_HISTORY_MAX_CHARS": "0",
        "AGENT_PLANNER_NO_THINK": "0",
        "VIDEOSEAL_DISABLE_TIME_REFERENCE": "1",
        "VIDEOSEAL_STRICT_FINAL_GATE": "1",
        "VISUAL_INSPECT_EXCLUSIVE_OUTPUT": "1",
        "VISUAL_INSPECT_PRESERVE_UNCERTAINTY": "0",
        "VISUAL_INSPECT_RETURN_FRAME_METADATA": "1",
        "AGENT_FORCE_LAST_STEP_VISUAL_INSPECT": "1",
        "AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK": "1",
        "AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK": "1",
        "AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE": "mcq",
        "RETRIEVAL_PRIORITY_CHANNEL": "0",
        "INSPECT_PENDING_SLOT": "0",
        "VIDEOSEAL_VISUAL_RETRIEVE_CLASS": None,
        "VIDEOSEAL_VISUAL_INSPECT_CLASS": None,
        "SUMMARY_FILE": None,
        "SUMMARY_DIR": None,
        "SUBTITLE_PATH": None,
        "BENCHMARK": None,
        "VIDEO_DURATION_SEC": None,
        "SEMANTIC_INDEX": None,
        "VISUAL_INDEX_DIR": None,
    }
    for field, env_name in (
        ("subtitle_path", "SUBTITLE_PATH"),
        ("benchmark", "BENCHMARK"),
        ("video_duration_sec", "VIDEO_DURATION_SEC"),
        ("visual_index", "VISUAL_INDEX_DIR"),
        ("summary_dir", "SUMMARY_DIR"),
        ("summary_file", "SUMMARY_FILE"),
    ):
        if task.get(field) not in (None, ""):
            settings[env_name] = str(task[field])
    if settings["SUMMARY_FILE"]:
        settings["SUMMARY_DIR"] = str(Path(settings["SUMMARY_FILE"]).parent)
    previous = {name: os.environ.get(name) for name in settings}
    try:
        for name, value in settings.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_task(
    task: dict[str, Any],
    policy: Policy,
    save_dir: str | Path,
    *,
    max_steps: int = 16,
    temperature: float = 0.7,
    max_new_tokens: int = 4096,
    max_factual_records: int | None = None,
    record_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> RolloutResult:
    """Run one answer-free task with the current student Planner.

    Required task keys are ``question``, ``video_id``, ``uid``, ``video_path``;
    optional keys match the original runner (``visual_index``, subtitle/summary
    paths, benchmark, duration). Any other fields, including gold, are ignored.
    ``step_index`` is zero-based into ``trajectory['steps']``. Context overflow
    and policy errors propagate; this adapter never truncates the history.
    Completion.complete is retained for the trainer to reject truncated draws.

    The opt-in training prefix stops before the next Planner generation after
    enough records pass the caller's pure eligibility filter. The original
    Inspector may still return naturally first. This preserves each collected
    H/F prefix, but removes the old requirement that the *future* trajectory
    finish successfully: a later context/tool failure no longer discards that
    prefix. Evaluation and callers omitting the option keep the complete loop.
    At the final allowed step there is no next generation hook, so the original
    forced fallback still executes. No candidate outcome participates here.
    """
    for key in ("question", "video_id", "uid", "video_path"):
        if not str(task.get(key) or "").strip():
            raise ValueError(f"Task requires nonempty {key!r}")
    if max_steps < 1 or max_new_tokens < 1 or temperature < 0:
        raise ValueError("max_steps/max_new_tokens must be positive and temperature nonnegative")
    if max_factual_records is not None and max_factual_records < 1:
        raise ValueError("max_factual_records must be positive when supplied")
    agent_class, build_tool_map = _load_videoseal()
    shim = PlannerShim(policy)
    with _task_environment(task, temperature=temperature, max_new_tokens=max_new_tokens):
        tools = {
            name: tool
            for name, tool in build_tool_map().items()
            if name in ("visual_retrieve", "visual_inspect")
        }
        if set(tools) != {"visual_retrieve", "visual_inspect"}:
            raise RuntimeError("The captured baseline must expose both original visual tools")
        agent = agent_class(tools=tools, llm_backend="api")
        agent.llm = shim
        if max_factual_records is not None:
            def before_generate() -> None:
                records = _factual_records(_trajectory_dict(agent.traj), shim.calls)
                eligible = sum(record_filter(record) if record_filter else True for record in records)
                if eligible >= max_factual_records:
                    raise _TrainingPrefixReady(eligible)
            shim.before_generate = before_generate
        prefix_stop = None
        try:
            result = agent.run(
                question=str(task["question"]),
                uid=str(task["uid"]),
                video_id=str(task["video_id"]),
                video_path=str(task["video_path"]),
                visual_index=task.get("visual_index") or None,
                groundtruth=None,
                max_steps=int(max_steps),
                save_dir=str(save_dir),
            )
        except _TrainingPrefixReady as stopped:
            # Catch only our sentinel. Model/tool failures, cancellation and
            # unrelated BaseExceptions continue to propagate normally.
            agent.traj.answer = None
            agent.traj.note = "training prefix stopped before the next Planner generation; QA incomplete"
            agent._save_trajectory(str(save_dir))
            prefix_stop = {"partial": True, "eligible_records": stopped.eligible_records,
                           "max_factual_records": max_factual_records,
                           "reason": "training prefix ready before next Planner generation"}
            result = {"answer": None, "steps": len(agent.traj.steps), "run_id": agent.traj.run_id,
                      "note": agent.traj.note, "training_prefix_stop": prefix_stop}
        trajectory = _trajectory_dict(agent.traj)
        if prefix_stop is not None:
            # The extra marker is outside the original dataclass and does not
            # enter records' messages or observations / hindsight teacher H/F.
            trajectory["training_prefix_stop"] = prefix_stop
            path = Path(save_dir) / str(trajectory["run_id"]) / "trajectory.json"
            path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return RolloutResult(_factual_records(trajectory, shim.calls), trajectory, dict(result))


def _trajectory_dict(trajectory: Any) -> dict[str, Any]:
    trajectory = asdict(trajectory) if is_dataclass(trajectory) else deepcopy(trajectory)
    if not isinstance(trajectory, dict):
        raise RuntimeError("The original agent did not return an audit trajectory")
    return trajectory


def _factual_records(trajectory: dict[str, Any], calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One alignment contract for complete and training-prefix trajectories."""
    records: list[dict[str, Any]] = []
    call_index = 0
    for step_index, step in enumerate(trajectory.get("steps", [])):
        observation = step.get("observation")
        action = step.get("action")
        if isinstance(observation, dict) and observation.get("forced"):
            continue
        if call_index >= len(calls):
            raise RuntimeError("Original trajectory has a step without a Planner completion")
        call = calls[call_index]
        call_index += 1
        if step.get("model_response") != call["completion"].text:
            raise RuntimeError("Planner completion/trajectory alignment changed")
        if (
            isinstance(action, dict)
            and action.get("name") == "visual_inspect"
            and not action.get("rejected")
            and isinstance(observation, dict)
            and observation.get("name") == "visual_inspect"
            and not observation.get("rejected")
        ):
            records.append({
                "messages": deepcopy(call["messages"]),
                "completion": call["completion"],
                "action": deepcopy(action),
                "observation": deepcopy(observation),
                "step_index": step_index,
            })
    if call_index != len(calls):
        raise RuntimeError("Planner completion has no matching original trajectory step")
    return records
