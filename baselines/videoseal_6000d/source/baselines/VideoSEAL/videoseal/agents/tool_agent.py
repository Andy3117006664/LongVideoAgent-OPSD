from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import ipaddress

from videoseal.tools.base import Tool, ToolOutput
from videoseal.utils.agent.env import env_bool_strict, env_flag, require_env, require_float_env, require_int_env
from videoseal.utils.agent.tool_agent_parsing import extract_between, parse_thinking, parse_tool_call_qwen
from videoseal.utils.agent.tool_schema import filter_args_for_forward, sanitize_args_against_schema
from videoseal.utils.agent.trajectory import Step, Trajectory
from videoseal.utils.api.mllm import MLLMClient
from videoseal.utils.video.io import get_video_duration
from videoseal.utils.video.time import sec_to_hhmmss


def _is_local_or_private_api_base(api_base: str) -> bool:
    base = (api_base or "").strip()
    if not base:
        return False
    try:
        u = urlparse(base if "://" in base else f"http://{base}")
        host = (u.hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_unspecified)


def _bool_env_or_default(name: str, default: bool) -> bool:
    if name in os.environ:
        return env_bool_strict(name, default=default)
    return default


def _system_prompt() -> str:
    return """You are a helper that answers multi-step video questions by sequentially invoking functions. Your ONLY job is to retrieve and refine candidate time spans; the final answer content must come from visual_inspect, not from your own imagination.

Tool outputs are lossy and may miss details. Prefer improving coverage by:
- trying different queries,
- inspecting new / non-overlapping time spans,
rather than re-running the same tool on the same spans.

If you cannot find strong anchors after several retrieval attempts, you may switch to an exploration mode:
- randomly sample a few large, non-overlapping time spans across the video,
- send them to visual_inspect to discover where relevant evidence might be.

Each turn, do exactly one of the following:

* If further clues and evidence are needed:
  Output EXACTLY one tool call:
  <tool_call>{"name":"tool_name","arguments":{}}</tool_call>

* If ready to finalize:
  You may enter this branch ONLY if the immediately previous turn called visual_inspect and its output contains a clear, reliable verdict.
  Output:
  <final>final answer</final>

Final Format
- Multiple-choice -> ONLY the uppercase option letter(s), for example C. No extra words.

Rules
- One tool call only per turn. Exactly one tool call per turn, except final turn (no tool call).
- You are a retriever, not an answerer: never invent answers; never override visual_inspect.
- You MUST NOT conclude from any non-visual_inspect tool alone.
- You may output <final> ONLY if the immediately previous turn was a visual_inspect call AND it returned a clear, reliable verdict.
- If the last tool call is NOT visual_inspect, you MUST NOT output <final>; keep searching and then call visual_inspect.
- The final answer must match the most recent reliable visual_inspect verdict (prefer the latest decisive visual_inspect when multiple exist).
- Do NOT re-run the same tool on the same time spans just to get more info. Instead, broaden coverage via different queries and new spans, or use exploration mode with large non-overlapping spans.
- If visual_inspect explicitly indicates that more search is needed, you MUST NOT finalize.
- A good pattern is to alternate tools, for example: visual_retrieve -> visual_inspect -> visual_retrieve with a different query -> visual_inspect on different time spans -> final.
- As soon as you have plausible spans, it is better to call visual_inspect again, possibly with more detailed context or slightly refined spans, than to keep doing blind retrieval. Multiple visual_inspect calls with different time spans are encouraged.
- Always zero-pad time fields (HH:MM:SS).
- Hard constraint: end_time must be strictly greater than start_time.

Tool Calling Conventions
- visual_retrieve: {"name":"visual_retrieve","arguments":{"query":"..."}}.
  Use free-form natural language or short visual phrases related to the question to retrieve visually/scene-relevant regions and coarse time anchors from the unified visual index.
- visual_inspect: {"name":"visual_inspect","arguments":{"spans":[{"start_time":"HH:MM:SS","end_time":"HH:MM:SS"}],"context":"Restate the original question and the visual sub-questions. Explain briefly why these spans were chosen, and provide a short look-for checklist of disambiguating cues to verify within these spans."}}.
  Constraints: each span must satisfy end_time > start_time.
"""


def _parse_final_answer(text: str) -> Optional[str]:
    fin = extract_between(text or "", "<final>", "</final>")
    if fin:
        return fin.strip()
    return None


def _decisive_visual_answer(observation: Any) -> Optional[str]:
    """Return an option letter only for a successful, answer-bearing inspection.

    The released prompt defines a final answer as authoritative only after a
    clear visual verdict.  The old runtime enforced that rule in prose only,
    so this helper deliberately uses a conservative parser and never treats
    retrieval text or ``SEARCH_MORE`` as a verdict.
    """

    if not isinstance(observation, dict) or observation.get("ok") is not True:
        return None
    output = observation.get("output")
    if isinstance(output, dict):
        text = str(output.get("answer") or "").strip()
    else:
        text = str(output or "").strip()
    if not text or re.search(r"\bSEARCH_MORE\b", text, re.IGNORECASE):
        return None
    match = re.search(r"(?:^|\b)answer\s*[:：]?\s*\(?([A-D])\)?(?:\s|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if re.fullmatch(r"\(?[A-D]\)?", text, re.IGNORECASE):
        return text.strip("() ").upper()
    return None


def _tool_required_reminder(tool_names: List[str], *, api_messages: bool) -> str:
    tools = ", ".join(tool_names) if tool_names else "the available tools"
    if api_messages:
        return (
            "Your previous response was invalid: it did not contain a parsed tool call and did not contain <final>...</final>.\n"
            f"You must call one available tool now using the tool-calling interface: {tools}. Do not answer in plain text."
        )
    return (
        "Your previous response was invalid: it did not contain a parsed <tool_call>...</tool_call> and did not contain <final>...</final>.\n"
        f"You must use one available tool now: {tools}.\n"
        'Return exactly one tool call in this format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>. Do not answer in plain text.'
    )


def _legacy_history_text(blocks: List[str]) -> str:
    """Build an optional bounded legacy prompt for small-context planners.

    The released VideoSEAL planner uses a much larger context window than the
    public Qwen3 service used by the planner-only diagnostic.  The default is
    intentionally unbounded to preserve the released behavior; the two
    environment controls are opt-in and only used by the Qwen3 smoke launcher.
    """

    if not blocks:
        return ""
    selected = list(blocks)
    try:
        max_blocks = int(os.getenv("AGENT_LEGACY_HISTORY_MAX_BLOCKS", "0"))
    except (TypeError, ValueError):
        max_blocks = 0
    if max_blocks > 0:
        selected = selected[-max_blocks:]
    text = "\n".join(selected)
    try:
        max_chars = int(os.getenv("AGENT_LEGACY_HISTORY_MAX_CHARS", "0"))
    except (TypeError, ValueError):
        max_chars = 0
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    # Keep the most recent evidence, which contains the latest tool verdict,
    # and mark the omitted prefix so the planner does not mistake truncation
    # for an empty tool history.
    return "[older tool history omitted for context budget]\n" + text[-max_chars:]


class SimpleToolAgent:
    def __init__(
        self,
        *,
        tools: Dict[str, type[Tool]],
        llm_backend: str = "api",
        local_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.tools = tools
        self.system_prompt_override = system_prompt
        self.llm_backend = (os.getenv("AGENT_LLM_BACKEND") or llm_backend).lower()

        self.local_model = local_model
        if self.llm_backend == "vllm":
            self.local_model = self.local_model or os.getenv("VLLM_MODEL")
            if not self.local_model or not str(self.local_model).strip():
                raise RuntimeError("AGENT_LLM_BACKEND=vllm requires VLLM_MODEL (or pass local_model=...).")

        self.llm: Optional[MLLMClient] = None
        if self.llm_backend != "vllm":
            agent_api_model = require_env("AGENT_LLM_MODEL")
            agent_api_base = require_env("AGENT_LLM_API_BASE")
            agent_api_key = require_env("AGENT_LLM_API_KEY")
            agent_backend = (os.getenv("AGENT_MLLM_BACKEND") or "openai").strip()
            self.llm = MLLMClient(base_url=agent_api_base, api_key=agent_api_key, model=agent_api_model, backend=agent_backend)

        self._vllm = None
        self._tokenizer = None
        self._model_config = None

        self.traj: Optional[Trajectory] = None
        self._traj_start_monotonic: float | None = None
        self._reset_retrieval_routing_state()

        # API path: for local/private vLLM, messages mode is typically safe and enables OpenAI tool calling.
        if "AGENT_API_USE_MESSAGES" in os.environ:
            self._api_use_messages = env_bool_strict("AGENT_API_USE_MESSAGES", default=False)
        else:
            base = os.getenv("AGENT_LLM_API_BASE", "")
            self._api_use_messages = _is_local_or_private_api_base(base)

    def _reset_retrieval_routing_state(self) -> None:
        self._latest_priority_candidates: List[Dict[str, Any]] = []
        self._latest_raw_candidates: List[Dict[str, Any]] = []
        self._seen_inspection_spans: set[Tuple[str, str]] = set()
        self._last_inspector_state = "none"
        self._inspection_count = 0

    def _validate_tool_action(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[str]:
        """Return a rejection reason for an invalid agent-specific action.

        The released agent accepts every schema-valid tool call. Subclasses
        can use this hook to implement a dialogue protocol without replacing
        the Planner's action: a rejected call is shown back to the same
        Planner, which must choose a corrected action itself.
        """

        return None

    def _tool_schemas_for_step(
        self, schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return the tool schemas visible in the current dialogue state."""

        return list(schemas)

    def _planner_state_text(self) -> str:
        """Return an optional current-state reminder for the Planner."""

        return ""

    @staticmethod
    def _span_key(span: Any) -> Optional[Tuple[str, str]]:
        if not isinstance(span, dict):
            return None
        start = str(span.get("start_time") or "").strip()
        end = str(span.get("end_time") or "").strip()
        if not start or not end or start == end:
            return None
        return start, end

    def _route_visual_inspect_args(
        self, args: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if not env_flag("INSPECT_PENDING_SLOT", default=False):
            return dict(args or {}), None

        proposed_args = dict(args or {})
        proposed_spans = [
            dict(span)
            for span in (proposed_args.get("spans") or [])
            if self._span_key(span) is not None
        ]
        effective_args = dict(proposed_args)
        effective_args["spans"] = list(proposed_spans)
        log: Dict[str, Any] = {
            "planner_proposed_spans": proposed_spans,
            "effective_spans": list(proposed_spans),
            "routing_activated": False,
            "replacement_reason": "",
            "inserted_candidate_id": None,
            "replaced_span": None,
            "candidate_source": None,
            "candidate_rank": None,
        }

        if self._inspection_count == 0:
            log["replacement_reason"] = "first_inspection"
            return effective_args, log
        if self._last_inspector_state != "search_more":
            log["replacement_reason"] = "last_inspector_not_search_more"
            return effective_args, log
        if not proposed_spans:
            log["replacement_reason"] = "no_planner_spans"
            return effective_args, log

        proposed_keys = {
            key
            for span in proposed_spans
            if (key := self._span_key(span)) is not None
        }
        candidate: Optional[Dict[str, Any]] = None
        source = ""
        for pool_source, pool in (
            ("priority", self._latest_priority_candidates),
            ("raw", self._latest_raw_candidates),
        ):
            for item in pool:
                key = self._span_key(item)
                if (
                    key is None
                    or key in self._seen_inspection_spans
                    or key in proposed_keys
                ):
                    continue
                candidate = dict(item)
                source = pool_source
                break
            if candidate is not None:
                break
        if candidate is None:
            log["replacement_reason"] = "no_unseen_retrieval_candidate"
            return effective_args, log

        only_key = self._span_key(proposed_spans[0]) if len(proposed_spans) == 1 else None
        if len(proposed_spans) == 1 and only_key not in self._seen_inspection_spans:
            log["replacement_reason"] = "single_new_planner_span"
            return effective_args, log

        inserted = {
            "start_time": str(candidate.get("start_time") or "").strip(),
            "end_time": str(candidate.get("end_time") or "").strip(),
        }
        effective_spans = list(proposed_spans[:-1]) + [inserted]
        effective_args["spans"] = effective_spans
        log.update(
            {
                "effective_spans": effective_spans,
                "routing_activated": True,
                "replacement_reason": "search_more_pending_slot",
                "inserted_candidate_id": candidate.get("candidate_id"),
                "replaced_span": proposed_spans[-1],
                "candidate_source": source,
                "candidate_rank": candidate.get("rank"),
            }
        )
        return effective_args, log

    def _record_tool_result(
        self, tool_name: str, args: Dict[str, Any], out: ToolOutput
    ) -> None:
        if tool_name == "visual_retrieve" and out.error is None:
            metadata = out.metadata if isinstance(out.metadata, dict) else {}
            channel = metadata.get("retrieval_priority_channel")
            if isinstance(channel, dict):
                priority = channel.get("priority_spans")
                raw = channel.get("raw_candidates")
                self._latest_priority_candidates = [
                    dict(item) for item in priority or [] if isinstance(item, dict)
                ]
                self._latest_raw_candidates = [
                    dict(item) for item in raw or [] if isinstance(item, dict)
                ]
            return
        if tool_name != "visual_inspect":
            return

        self._inspection_count += 1
        spans: Any = None
        if out.error is None and isinstance(out.output, dict):
            spans = out.output.get("spans")
            answer_text = str(out.output.get("answer") or "").strip()
        else:
            answer_text = ""
        if not isinstance(spans, list):
            spans = args.get("spans") if isinstance(args, dict) else []
        for span in spans or []:
            key = self._span_key(span)
            if key is not None:
                self._seen_inspection_spans.add(key)

        if out.error is not None:
            self._last_inspector_state = "error"
        elif re.search(r"\bSEARCH_MORE\b", answer_text, re.IGNORECASE):
            self._last_inspector_state = "search_more"
        elif re.search(
            r"(?:^|\b)answer\s*[:：]?\s*\(?[A-D]\)?(?:\s|$)",
            answer_text,
            re.IGNORECASE,
        ) or re.fullmatch(r"\(?[A-D]\)?", answer_text, re.IGNORECASE):
            self._last_inspector_state = "decisive"
        else:
            self._last_inspector_state = "other"

    @staticmethod
    def _metadata_with_routing(
        metadata: Any, routing: Optional[Dict[str, Any]]
    ) -> Any:
        if routing is None:
            return metadata
        merged = dict(metadata) if isinstance(metadata, dict) else {}
        enriched = dict(routing)
        if isinstance(metadata, dict) and metadata.get("frame_count") is not None:
            enriched["frame_count"] = metadata.get("frame_count")
        merged["pending_slot_routing"] = enriched
        return merged

    def _ensure_vllm(self) -> None:
        if self._vllm is not None:
            return
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("vLLM backend requires transformers.") from exc
        try:
            from vllm import LLM  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("vLLM backend requires vllm.") from exc

        model = str(self.local_model or "").strip()
        if not model:
            raise RuntimeError("Missing VLLM_MODEL for local vLLM backend.")

        tp = int(os.getenv("VLLM_TENSOR_PARALLEL", "1"))
        gpu_mem = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.9"))
        self._vllm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._model_config = getattr(self._vllm, "llm_engine", None) and getattr(self._vllm.llm_engine, "model_config", None)

    def _inject_system_args(self, tool_name: str, model_args: Dict[str, Any]) -> Dict[str, Any]:
        injected = dict(model_args or {})
        # minimal injection for our two tools
        if tool_name == "visual_retrieve":
            injected.setdefault("video_id", getattr(self.traj, "video_id", "") if self.traj else "")
            injected.setdefault("original_question", getattr(self.traj, "question", "") if self.traj else "")
            vis = getattr(self, "_visual_index", None)
            if vis:
                injected.setdefault("index_path", vis)
        if tool_name == "visual_inspect":
            injected.setdefault("video_path", getattr(self, "_video_path", ""))
            injected.setdefault("questions", getattr(self.traj, "question", "") if self.traj else "")
            injected.setdefault("prompt_type", getattr(self, "_prompt_type", 0))
        return injected

    def _extract_tool_calls_from_last_response(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if self.llm is None:
            return {}, []
        data = self.llm.get_last_response()
        if not isinstance(data, dict):
            return {}, []
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return {}, []
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            return {}, []
        msg = choice0.get("message")
        if not isinstance(msg, dict):
            return {}, []
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
        return msg, tool_calls

    def _save_trajectory(self, save_dir: str) -> None:
        if self.traj is None:
            return
        if self._traj_start_monotonic is not None:
            self.traj.elapsed_sec = round(time.monotonic() - float(self._traj_start_monotonic), 6)
        self.traj.finished_at = datetime.now(timezone.utc).isoformat()
        out_dir = Path(save_dir) / self.traj.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trajectory.json").write_text(json.dumps(asdict(self.traj), ensure_ascii=False, indent=2), encoding="utf-8")

    def _fallback_answer_from_last_tool(self) -> Optional[str]:
        if self.traj is None:
            return None
        for st in reversed(self.traj.steps):
            obs = st.observation
            if not isinstance(obs, dict) or obs.get("ok") is not True:
                continue
            tool_name = str(obs.get("name") or "").strip() or (st.action.get("name") if isinstance(st.action, dict) else None) or ""
            if tool_name != "visual_inspect":
                continue
            # CRCC may deliberately expose a provisional visual answer as
            # SEARCH_MORE.  Preserve that incumbent only at horizon end; it
            # is metadata from the same inspection, never a new prediction.
            if env_flag("CRCC_PRESERVE_PROVISIONAL_FALLBACK", default=False):
                metadata = obs.get("metadata")
                if isinstance(metadata, dict):
                    crcc = metadata.get("crcc")
                    if isinstance(crcc, dict):
                        provisional = str(crcc.get("provisional_answer") or "").strip().upper()
                        if provisional in {"A", "B", "C", "D"}:
                            return provisional
            out = obs.get("output")
            if isinstance(out, dict):
                ans_text = str(out.get("answer") or "").strip()
            else:
                ans_text = str(out or "").strip()
            if not ans_text:
                continue
            return ans_text
        return None

    def _crcc_provisional_fallback(self) -> Optional[str]:
        """Return a CRCC incumbent only when its opt-in fallback is enabled."""

        if not env_flag("CRCC_PRESERVE_PROVISIONAL_FALLBACK", default=False) or self.traj is None:
            return None
        for st in reversed(self.traj.steps):
            obs = st.observation
            if not isinstance(obs, dict) or obs.get("ok") is not True:
                continue
            metadata = obs.get("metadata")
            if not isinstance(metadata, dict):
                continue
            crcc = metadata.get("crcc")
            if not isinstance(crcc, dict):
                continue
            provisional = str(crcc.get("provisional_answer") or "").strip().upper()
            if provisional in {"A", "B", "C", "D"}:
                return provisional
        return None

    def _strict_final_gate_enabled(self) -> bool:
        """Whether to enforce the visual-inspection termination contract."""

        return env_flag("VIDEOSEAL_STRICT_FINAL_GATE", default=False)

    def _last_step_has_decisive_visual_verdict(self) -> bool:
        """Check the immediately preceding trajectory step, not any old verdict."""

        return self._last_step_visual_verdict() is not None

    def _last_step_visual_verdict(self) -> Optional[str]:
        """Return the immediately preceding visual verdict, if it is decisive."""

        if self.traj is None or not self.traj.steps:
            return None
        step = self.traj.steps[-1]
        action = step.action if isinstance(step.action, dict) else {}
        if str(action.get("name") or "").strip() != "visual_inspect":
            return None
        return _decisive_visual_answer(step.observation)

    def _accept_final_answer(self, answer: Optional[str]) -> bool:
        """Apply the optional runtime contract before accepting ``<final>``."""

        if not answer:
            return False
        if not self._strict_final_gate_enabled():
            return True
        verdict = self._last_step_visual_verdict()
        final_match = re.fullmatch(r"\s*\(?([A-D])\)?\s*", str(answer), re.IGNORECASE)
        return bool(verdict and final_match and final_match.group(1).upper() == verdict)

    def _extract_answer_text(self, raw: object) -> str:
        if isinstance(raw, dict):
            raw = raw.get("answer")
        text = str(raw or "").strip()
        if not text:
            return ""
        unwrapped = _parse_final_answer(text)
        if unwrapped:
            return unwrapped.strip()
        return text

    def _auto_commit_exclusive_inspector(
        self,
        *,
        tool_name: str,
        payload: Dict[str, Any],
        save_dir: Optional[str],
        step_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Commit a normalized decisive inspector verdict without another LLM turn.

        When the inspector adapter has explicitly produced a mutually
        exclusive decisive state, asking the planner to repeat the same answer
        only adds latency and reintroduces the old answer/search ambiguity.
        SEARCH_MORE responses do not enter this path and continue through the
        normal planner loop.
        """

        if tool_name != "visual_inspect" or not env_flag("VISUAL_INSPECT_EXCLUSIVE_OUTPUT", default=False):
            return None
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        state = metadata.get("exclusive_inspector_output", {}).get("control_state") if isinstance(metadata, dict) else None
        if state != "decisive":
            return None
        verdict = _decisive_visual_answer(payload)
        if verdict is None or self.traj is None:
            return None
        self.traj.answer = verdict
        self.traj.note = "auto-committed decisive visual_inspect verdict"
        if save_dir:
            self._save_trajectory(save_dir)
        return {
            "final": verdict,
            "answer": verdict,
            "steps": int(step_idx),
            "note": self.traj.note,
            "run_id": self.traj.run_id,
        }

    def _forced_full_video_visual_inspect(self) -> Optional[str]:
        if self.traj is None:
            return None
        tool_cls = self.tools.get("visual_inspect")
        if tool_cls is None:
            return None

        video_path = str(getattr(self, "_video_path", "") or "").strip()
        if not video_path:
            raise RuntimeError("Forced visual_inspect fallback requires video_path.")

        duration = float(get_video_duration(video_path))
        if duration <= 0.0:
            raise RuntimeError(f"Forced visual_inspect fallback requires a valid video duration (video={video_path!r}).")

        max_total_raw = (os.getenv("INSPECT_MAX_TOTAL_IMAGES") or "").strip() or "64"
        max_total = int(max_total_raw)
        if max_total <= 0:
            raise ValueError(f"INSPECT_MAX_TOTAL_IMAGES must be a positive integer, got {max_total_raw!r}.")

        # Uniform sampling across the full video by setting fps so that duration * fps ~= max_total.
        fps = float(max_total) / float(duration)

        spans = [{"start_time": "00:00:00", "end_time": sec_to_hhmmss(int(math.ceil(duration)))}]

        args: Dict[str, Any] = {
            "video_path": video_path,
            "spans": spans,
            "questions": [self.traj.question],
            "prompt_type": int(getattr(self, "_prompt_type", 0)),
        }

        st = Step(chat_prompt=self.traj.question, model_response="", usage=None)
        st.action = {"name": "visual_inspect", "arguments": {"spans": spans}}

        restore_mode = os.getenv("VISUAL_INSPECT_PROMPT_MODE")
        restore_fps = os.getenv("INSPECT_FPS")
        try:
            os.environ["VISUAL_INSPECT_PROMPT_MODE"] = (os.getenv("AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE") or "mcq").strip() or "mcq"
            os.environ["INSPECT_FPS"] = str(fps)
            out = tool_cls()(**args)
        finally:
            if restore_mode is None:
                os.environ.pop("VISUAL_INSPECT_PROMPT_MODE", None)
            else:
                os.environ["VISUAL_INSPECT_PROMPT_MODE"] = restore_mode
            if restore_fps is None:
                os.environ.pop("INSPECT_FPS", None)
            else:
                os.environ["INSPECT_FPS"] = restore_fps

        ok = out.error is None
        payload = {
            "name": "visual_inspect",
            "ok": ok,
            "output": out.output if ok else None,
            "error": out.error,
            "forced": True,
            "mode": "full_video",
        }
        st.observation = payload
        self.traj.steps.append(st)

        if not ok:
            self.traj.note = f"max steps reached; forced full-video visual_inspect failed: {out.error}"
            return None

        return self._extract_answer_text(out.output)

    def run(
        self,
        *,
        question: str,
        uid: Optional[str] = None,
        time_reference: Optional[str] = None,
        video_id: str,
        video_path: str,
        visual_index: Optional[str] = None,
        groundtruth: Optional[str] = None,
        max_steps: int = 8,
        save_dir: Optional[str] = None,
        prompt_type: int = 0,
        initial_retrieval_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._traj_start_monotonic = time.monotonic()
        self._reset_retrieval_routing_state()
        self._video_path = str(video_path or "")
        self._visual_index = str(visual_index or "") if visual_index else None
        self._prompt_type = int(prompt_type)

        system = (self.system_prompt_override or _system_prompt()).strip()
        tool_names = list((self.tools or {}).keys())

        tool_schemas = [cls().json for cls in (self.tools or {}).values()]
        tools_schema_text = json.dumps(tool_schemas, ensure_ascii=False)

        clean_control = str(os.getenv("VIDEOSEAL_DISABLE_TIME_REFERENCE") or "").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        effective_time_reference = "" if clean_control else str(time_reference or "").strip()

        self.traj = Trajectory(
            system_prompt=system,
            tools_schema=tools_schema_text,
            question=str(question or "").strip(),
            time_reference=(effective_time_reference or None),
            uid=uid,
            groundtruth=(str(groundtruth).strip() if groundtruth else None),
            video_id=str(video_id or ""),
        )

        # History
        tool_history_blocks: List[str] = []
        api_history: List[Dict[str, Any]] = [{"role": "system", "content": system}]

        user_base = str(question or "").strip()
        if effective_time_reference:
            user_base = f"[Time reference: {effective_time_reference}]\n{user_base}"
        # Qwen3 can spend the whole planner output budget inside <think>
        # before emitting the required legacy tool call.  Keep this as an
        # opt-in suffix so the released behavior remains unchanged while a
        # no-thinking planner ablation can be run cleanly.
        if env_flag("AGENT_PLANNER_NO_THINK", default=False):
            user_base = f"{user_base}\n/no_think"
        if self.llm_backend != "vllm" and self._api_use_messages:
            api_history.append({"role": "user", "content": user_base})

        # Evidence-space scaling may choose a first retrieval query outside the
        # planner loop. Execute that query through the unchanged retrieval tool
        # and expose the real observation as step one; all later behavior stays
        # inside the native agent loop. The optional hook is inert for every
        # existing caller.
        seeded_steps = 0
        seed_query = str(initial_retrieval_query or "").strip()
        if seed_query:
            tool_name = "visual_retrieve"
            tool_cls = self.tools.get(tool_name)
            if tool_cls is None:
                raise RuntimeError("initial_retrieval_query requires visual_retrieve")
            tool = tool_cls()
            args = {"query": seed_query}
            injected = self._inject_system_args(tool_name, args)
            injected = sanitize_args_against_schema(tool, injected)
            injected = filter_args_for_forward(tool, injected)
            out = tool(**injected)
            self._record_tool_result(tool_name, args, out)
            ok = out.error is None
            payload = {
                "name": tool_name,
                "ok": ok,
                "output": out.output if ok else None,
                "error": out.error,
                "metadata": out.metadata,
                "seeded": True,
            }
            call_id = f"eviscale-seed-{uuid.uuid4()}"
            call_obj = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
            st = Step(
                chat_prompt=user_base,
                model_response=json.dumps({"seeded_tool_call": call_obj}, ensure_ascii=False),
                usage=None,
            )
            st.action = {"name": tool_name, "arguments": args}
            st.observation = payload
            self.traj.steps.append(st)
            if self.llm_backend != "vllm" and self._api_use_messages:
                api_history.append(
                    {"role": "assistant", "content": None, "tool_calls": [call_obj]}
                )
                api_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": out.to_string(),
                    }
                )
            else:
                visible = out.to_string() if ok else f"Error: {out.error}"
                tool_history_blocks.append(
                    f"<seeded_tool_call>{json.dumps({'name': tool_name, 'arguments': args}, ensure_ascii=False)}</seeded_tool_call>"
                    f"<tool_response>{visible}</tool_response>"
                )
            seeded_steps = 1

        for step_idx in range(seeded_steps + 1, int(max_steps) + 1):
            active_tool_schemas = self._tool_schemas_for_step(tool_schemas)
            active_tool_names = [
                str(schema.get("function", {}).get("name") or "")
                for schema in active_tool_schemas
                if isinstance(schema, dict)
            ]
            planner_state = self._planner_state_text().strip()
            state_suffix = (
                f"\n\n<planner_state>\n{planner_state}\n</planner_state>"
                if planner_state
                else ""
            )
            if self.llm_backend == "vllm":
                self._ensure_vllm()
                from vllm import SamplingParams  # type: ignore

                msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
                if tool_history_blocks:
                    msgs.append({"role": "user", "content": "\n".join(tool_history_blocks) + "\n\n" + user_base + state_suffix})
                else:
                    msgs.append({"role": "user", "content": user_base + state_suffix})
                prompt_text = self._tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)  # type: ignore[union-attr]
                max_new = int(os.getenv("VLLM_MAX_TOKENS", "1024"))
                temperature = float(os.getenv("VLLM_TEMPERATURE", "0.1"))
                outputs = self._vllm.generate([prompt_text], SamplingParams(max_tokens=max_new, temperature=temperature, top_p=0.95, n=1))
                resp_text = outputs[0].outputs[0].text
                usage = None
            else:
                if self.llm is None:
                    raise RuntimeError("Agent API client is not initialized.")

                tool_choice = None
                if max_steps > 1 and step_idx == max_steps - 1 and env_flag("AGENT_FORCE_LAST_STEP_VISUAL_INSPECT", default=False):
                    if "visual_inspect" in active_tool_names:
                        tool_choice = {"type": "function", "function": {"name": "visual_inspect"}}

                if self._api_use_messages:
                    request_history = list(api_history)
                    if planner_state:
                        request_history.append(
                            {
                                "role": "user",
                                "content": f"<planner_state>\n{planner_state}\n</planner_state>",
                            }
                        )
                    resp_text = self.llm.generate_chat(
                        request_history,
                        response_json=False,
                        max_tokens=require_int_env("AGENT_LLM_MAX_TOKENS"),
                        temperature=require_float_env("AGENT_LLM_TEMPERATURE"),
                        timeout=require_float_env("AGENT_LLM_TIMEOUT"),
                        tools=active_tool_schemas,
                        tool_choice=(tool_choice or "auto"),
                    )
                    usage = self.llm.get_last_usage() or None
                    msg, tool_calls = self._extract_tool_calls_from_last_response()
                    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.get("content", resp_text)}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    api_history.append(assistant_msg)

                    if tool_calls:
                        for tc in tool_calls:
                            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                            tool_name = str(fn.get("name") or "").strip()
                            call_id = str(tc.get("id") or str(uuid.uuid4()))
                            arg_s = fn.get("arguments")
                            if isinstance(arg_s, dict):
                                args = arg_s
                            elif isinstance(arg_s, str):
                                if not arg_s.strip():
                                    args = {}
                                else:
                                    try:
                                        raw_args = json.loads(arg_s)
                                    except json.JSONDecodeError as exc:
                                        raise RuntimeError(f"Invalid tool call arguments JSON for {tool_name!r}: {exc}") from exc
                                    if not isinstance(raw_args, dict):
                                        raise RuntimeError(
                                            f"Tool call arguments for {tool_name!r} must be a JSON object, got: {type(raw_args).__name__}"
                                        )
                                    args = raw_args
                            else:
                                args = {}

                            rejection = self._validate_tool_action(tool_name, args)
                            if rejection:
                                st = Step(
                                    chat_prompt=user_base,
                                    model_response=json.dumps(
                                        {"tool_calls": tool_calls}, ensure_ascii=False
                                    ),
                                    usage=usage,
                                )
                                st.thinking = parse_thinking(resp_text or "")
                                st.action = {
                                    "name": tool_name,
                                    "arguments": dict(args),
                                    "rejected": True,
                                }
                                payload = {
                                    "name": tool_name,
                                    "ok": False,
                                    "output": None,
                                    "error": f"protocol rejection: {rejection}",
                                    "rejected": True,
                                }
                                st.observation = payload
                                self.traj.steps.append(st)
                                api_history.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "content": json.dumps(
                                            payload, ensure_ascii=False
                                        ),
                                    }
                                )
                                continue

                            effective_args, routing = (
                                self._route_visual_inspect_args(args)
                                if tool_name == "visual_inspect"
                                else (dict(args), None)
                            )
                            st = Step(chat_prompt=user_base, model_response=json.dumps({"tool_calls": tool_calls}, ensure_ascii=False), usage=usage)
                            st.thinking = parse_thinking(resp_text or "")
                            st.action = {"name": tool_name, "arguments": effective_args}
                            if routing is not None:
                                st.action["routing"] = routing

                            tool_cls = self.tools.get(tool_name)
                            if tool_cls is None:
                                payload = {"name": tool_name, "ok": False, "output": None, "error": f"unknown tool: {tool_name}"}
                                api_history.append({"role": "tool", "tool_call_id": call_id, "content": f"Error: {payload['error']}"})
                                st.observation = payload
                                self.traj.steps.append(st)
                                continue

                            tool = tool_cls()
                            injected = self._inject_system_args(tool_name, effective_args)
                            injected = sanitize_args_against_schema(tool, injected)
                            injected = filter_args_for_forward(tool, injected)

                            # Optional prompt-mode override for last-step visual_inspect.
                            _set_vis_mode = (
                                tool_name == "visual_inspect"
                                and max_steps > 1
                                and step_idx == max_steps - 1
                                and env_flag("AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK", default=False)
                            )
                            _restore_vis_mode = None
                            if _set_vis_mode:
                                _restore_vis_mode = os.getenv("VISUAL_INSPECT_PROMPT_MODE")
                                os.environ["VISUAL_INSPECT_PROMPT_MODE"] = (os.getenv("AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE") or "mcq").strip()
                            try:
                                out = tool(**injected)
                            finally:
                                if _set_vis_mode:
                                    if _restore_vis_mode is None:
                                        os.environ.pop("VISUAL_INSPECT_PROMPT_MODE", None)
                                    else:
                                        os.environ["VISUAL_INSPECT_PROMPT_MODE"] = _restore_vis_mode

                            ok = out.error is None
                            self._record_tool_result(tool_name, effective_args, out)
                            payload = {
                                "name": tool_name,
                                "ok": ok,
                                "output": out.output if ok else None,
                                "error": out.error,
                                "metadata": self._metadata_with_routing(out.metadata, routing),
                            }
                            visible_txt = out.to_string()
                            api_history.append({"role": "tool", "tool_call_id": call_id, "content": visible_txt})
                            st.observation = payload
                            self.traj.steps.append(st)

                            committed = self._auto_commit_exclusive_inspector(
                                tool_name=tool_name,
                                payload=payload,
                                save_dir=save_dir,
                                step_idx=step_idx,
                            )
                            if committed is not None:
                                return committed

                        # continue after tool execution
                        continue

                    # No OpenAI tool calls. Accept only a tagged final answer; otherwise remind and retry.
                    st = Step(chat_prompt=user_base, model_response=resp_text or "", usage=usage)
                    st.thinking = parse_thinking(resp_text or "")
                    ans = _parse_final_answer(resp_text or "")
                    accepted = self._accept_final_answer(ans)
                    self.traj.steps.append(st)
                    if accepted:
                        self.traj.answer = ans.strip()
                        if save_dir:
                            self._save_trajectory(save_dir)
                        return {"final": self.traj.answer, "answer": self.traj.answer, "steps": step_idx, "run_id": self.traj.run_id}
                    if ans and self._strict_final_gate_enabled():
                        self.traj.note = "strict final gate rejected answer without an immediately preceding decisive visual_inspect"
                    else:
                        self.traj.note = "invalid response; reminded model to use a tool"
                    api_history.append(
                        {
                            "role": "user",
                            "content": _tool_required_reminder(active_tool_names or tool_names, api_messages=True),
                        }
                    )
                    continue

                # Legacy mode: tag-based tool calling (<tool_call>...</tool_call>).
                history_txt = _legacy_history_text(tool_history_blocks)
                tools_block = "<tools>\n" + "\n".join(json.dumps(s, ensure_ascii=False) for s in active_tool_schemas) + "\n</tools>"
                if history_txt:
                    user_prompt = tools_block + "\n\n" + history_txt + "\n\n" + user_base + state_suffix
                else:
                    user_prompt = tools_block + "\n\n" + user_base + state_suffix
                resp_text = self.llm.generate_text(
                    user_prompt,
                    response_json=False,
                    system_message=system.strip(),
                    max_tokens=require_int_env("AGENT_LLM_MAX_TOKENS"),
                    temperature=require_float_env("AGENT_LLM_TEMPERATURE"),
                    timeout=require_float_env("AGENT_LLM_TIMEOUT"),
                )
                usage = self.llm.get_last_usage() or None

            # Shared parsing for vllm backend and legacy API backend.
            resp_text = str(resp_text or "")
            st = Step(chat_prompt=user_base, model_response=resp_text, usage=usage)
            st.thinking = parse_thinking(resp_text)

            call = parse_tool_call_qwen(resp_text)
            if call:
                tool_name = str(call.get("name") or "").strip()
                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                rejection = self._validate_tool_action(tool_name, args)
                if rejection:
                    st.action = {
                        "name": tool_name,
                        "arguments": dict(args),
                        "rejected": True,
                    }
                    payload = {
                        "name": tool_name,
                        "ok": False,
                        "output": None,
                        "error": f"protocol rejection: {rejection}",
                        "rejected": True,
                    }
                    st.observation = payload
                    self.traj.steps.append(st)
                    tool_history_blocks.append(
                        f"<tool_response>{json.dumps(payload, ensure_ascii=False)}</tool_response>"
                    )
                    continue
                effective_args, routing = (
                    self._route_visual_inspect_args(args)
                    if tool_name == "visual_inspect"
                    else (dict(args), None)
                )
                st.action = {"name": tool_name, "arguments": effective_args}
                if routing is not None:
                    st.action["routing"] = routing

                tool_cls = self.tools.get(tool_name)
                if tool_cls is None:
                    payload = {"name": tool_name, "ok": False, "output": None, "error": f"unknown tool: {tool_name}"}
                    st.observation = payload
                    self.traj.steps.append(st)
                    tool_history_blocks.append(f"<tool_response>{json.dumps(payload, ensure_ascii=False)}</tool_response>")
                    continue

                tool = tool_cls()
                injected = self._inject_system_args(tool_name, effective_args)
                injected = sanitize_args_against_schema(tool, injected)
                injected = filter_args_for_forward(tool, injected)
                out = tool(**injected)
                ok = out.error is None
                self._record_tool_result(tool_name, effective_args, out)
                payload = {
                    "name": tool_name,
                    "ok": ok,
                    "output": out.output if ok else None,
                    "error": out.error,
                    "metadata": self._metadata_with_routing(out.metadata, routing),
                }
                st.observation = payload
                self.traj.steps.append(st)
                committed = self._auto_commit_exclusive_inspector(
                    tool_name=tool_name,
                    payload=payload,
                    save_dir=save_dir,
                    step_idx=step_idx,
                )
                if committed is not None:
                    return committed
                tool_history_blocks.append(f"<tool_response>{ToolOutput(name=tool_name, output=payload.get('output') if ok else out.to_string()).to_string()}</tool_response>")
                continue

            ans = _parse_final_answer(resp_text)
            accepted = self._accept_final_answer(ans)
            self.traj.steps.append(st)
            if accepted:
                self.traj.answer = ans.strip()
                if save_dir:
                    self._save_trajectory(save_dir)
                return {"final": self.traj.answer, "answer": self.traj.answer, "steps": step_idx, "run_id": self.traj.run_id}
            if ans and self._strict_final_gate_enabled():
                self.traj.note = "strict final gate rejected answer without an immediately preceding decisive visual_inspect"
            else:
                self.traj.note = "invalid response; reminded model to use a tool"
            tool_history_blocks.append(
                "<format_feedback>"
                + _tool_required_reminder(active_tool_names or tool_names, api_messages=False)
                + "</format_feedback>"
            )
            continue

        if env_flag("AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK", default=False):
            forced = self._forced_full_video_visual_inspect()
            if forced and (not self._strict_final_gate_enabled() or self._last_step_has_decisive_visual_verdict()):
                self.traj.answer = forced
                self.traj.note = "max steps reached; forced full-video visual_inspect fallback"
                if save_dir:
                    self._save_trajectory(save_dir)
                return {"final": self.traj.answer, "answer": self.traj.answer, "steps": int(max_steps), "note": self.traj.note, "run_id": self.traj.run_id}
            incumbent = self._crcc_provisional_fallback()
            if incumbent:
                self.traj.answer = incumbent
                self.traj.note = "max steps reached; CRCC preserved provisional visual answer"
                if save_dir:
                    self._save_trajectory(save_dir)
                return {"final": incumbent, "answer": incumbent, "steps": int(max_steps), "note": self.traj.note, "run_id": self.traj.run_id}
            if save_dir:
                self._save_trajectory(save_dir)
            return {"answer": None, "steps": int(max_steps), "note": self.traj.note or "max steps reached; forced visual_inspect fallback failed", "run_id": self.traj.run_id}

        fallback = self._fallback_answer_from_last_tool()
        if self._strict_final_gate_enabled():
            fallback = (
                _decisive_visual_answer(self.traj.steps[-1].observation)
                if self._last_step_has_decisive_visual_verdict() and self.traj and self.traj.steps
                else None
            )
        if fallback:
            self.traj.answer = str(fallback).strip()
            self.traj.note = "max steps reached (fallback from last visual_inspect)"
            if save_dir:
                self._save_trajectory(save_dir)
            return {"final": self.traj.answer, "answer": self.traj.answer, "steps": int(max_steps), "note": self.traj.note, "run_id": self.traj.run_id}

        self.traj.note = "max steps reached"
        if save_dir:
            self._save_trajectory(save_dir)
        return {"answer": None, "steps": int(max_steps), "note": self.traj.note, "run_id": self.traj.run_id}
