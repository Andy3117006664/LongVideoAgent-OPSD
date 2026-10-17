from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from videoseal.prompts.visual_tool_prompts import (
    build_visual_inspect_last_step_mcq_prompt,
    build_visual_inspect_prompt,
    build_visual_retrieve_summary_prompt,
)
from videoseal.tools.base import Tool, ToolOutput
from videoseal.utils.RAG.rag_query_bm25 import query_bm25
from videoseal.utils.RAG.rag_query_embed import query as query_embed
from videoseal.utils.agent.env import (
    build_mllm_client_from_env_prefix,
    env_flag,
    env_float_strict,
    env_int_first,
    env_int_strict,
)
from videoseal.utils.api.error_utils import format_exception_with_request_id
from videoseal.utils.retrieval.scoring import fuse_weighted_hits
from videoseal.utils.video.frames import resize_images_max_long_side, sample_uniform_frames
from videoseal.utils.video.time import sec_to_hhmmss, srt_timestamp_to_seconds
from videoseal.utils.video.tooling import (
    balanced_select_frames,
    cleanup_dir_tree,
    make_unique_run_dir,
    normalize_window_min_width,
    parse_hhmmss_spans_from_text,
    select_time_diverse_windows,
    video_id_from_path,
)
from videoseal.utils.env_paths import get_frames_root


FRAMES_ROOT = get_frames_root()


_INSPECT_ANSWER_RE = re.compile(
    r"(?im)^\s*Answer\s*[:：]\s*(?P<answer>SEARCH_MORE|[A-D](?:\s*,\s*[A-D])*)\s*$"
)


def _has_substantive_missing(text: str) -> bool:
    """Return whether an option-bearing response admits unresolved evidence.

    The inspector prompt permits ``Missing: None`` only for a complete
    verdict.  A non-empty Missing field is therefore a stronger uncertainty
    signal than the self-reported confidence, which is often saturated.
    """

    match = re.search(
        r"(?ims)^\s*Missing\s*[:：]\s*(.*?)(?=^\s*(?:Next\s+search|Reminder)\s*[:：]|\Z)",
        str(text or ""),
    )
    if match is None:
        return False
    value = re.sub(r"\s+", " ", match.group(1)).strip().casefold()
    return value not in {"", "none", "none.", "n/a", "na", "-", "nil"}


def normalize_visual_inspect_control_text(text: str) -> tuple[str, str]:
    """Make the inspector's answer/search control state mutually exclusive.

    The existing agent runtime already treats a leading option answer as a
    decisive verdict. Some VLM responses nevertheless append SEARCH_MORE-only
    fields, which gives the planner the opposite instruction. Canonicalizing
    only those option-bearing responses keeps the existing decision semantics
    while removing the contradictory control text at zero inference cost.
    """

    value = str(text or "").strip()
    match = _INSPECT_ANSWER_RE.search(value)
    if match is None:
        return value, "unparsed"
    answer = re.sub(r"\s+", "", match.group("answer").upper())
    if answer == "SEARCH_MORE":
        return value, "search_more"

    # Keep the inspector's uncertainty visible to the controller.  The
    # historical normalizer discarded Missing/Next search after an option
    # answer, which converted internally inconsistent responses into an
    # auto-committed decisive state.  This is opt-in so the released baseline
    # remains byte-for-byte compatible unless the experiment enables it.
    if env_flag("VISUAL_INSPECT_PRESERVE_UNCERTAINTY", default=False) and _has_substantive_missing(value):
        normalized = re.sub(
            r"(?im)^(\s*Answer\s*[:：]\s*)[A-D](?:\s*,\s*[A-D])*\s*$",
            r"\1SEARCH_MORE",
            value,
            count=1,
        )
        return normalized, "search_more"

    fields = [f"Answer: {answer}"]
    evidence = re.search(
        r"(?ims)^\s*Evidence\s*[:：]\s*(.*?)"
        r"(?=^\s*(?:Confidence|Missing|Next\s+search|Reminder)\s*[:：]|\Z)",
        value,
    )
    confidence = re.search(r"(?im)^\s*Confidence\s*[:：]\s*([^\r\n]+)", value)
    if evidence is not None and evidence.group(1).strip():
        fields.append(f"Evidence: {evidence.group(1).strip()}")
    if confidence is not None and confidence.group(1).strip():
        fields.append(f"Confidence: {confidence.group(1).strip()}")
    return "\n".join(fields), "decisive"


def _parse_doc_id_time_window(doc_id: str) -> tuple[int, int]:
    parts = str(doc_id).split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid doc_id {doc_id!r}; expected '<start>_<end>' seconds.")
    s_sec = int(float(parts[0]))
    e_sec = int(float(parts[1]))
    return s_sec, e_sec


class VisualRetrieveAliasTool(Tool):
    """Fine-grained retrieval over LVBench-style semantic index (unified semantic index)."""

    def __init__(self, name: str | None = None, description: str | None = None, function=None):
        super().__init__(
            name=name or "visual_retrieve",
            description=description
            or "Fine-grained retrieval over the visual (LVBench) semantic index; optionally summarizes top hits and returns useful spans.",
        )

    @property
    def json(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        }

    def _retrieve_hits(
        self,
        idx_dir: Path,
        query: str,
        k: int,
        original_question: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Return ranked caption hits.

        Kept as a method so retrieval-only ablations can replace ranking while
        reusing the released summarizer and output contract unchanged.
        """
        mix = (os.getenv("SEMANTIC_RETRIEVE_MIX") or "embed").lower()
        if mix == "bm25":
            return query_bm25(idx_dir, query, topk=k)
        if mix == "weighted":
            w_e = env_float_strict("RETRIEVE_EMBED_WEIGHT", 0.5)
            w_b = env_float_strict("RETRIEVE_BM25_WEIGHT", 0.5)
            e_hits = query_embed(idx_dir, query, topk=max(k, 1) * 2)
            b_hits = query_bm25(idx_dir, query, topk=max(k, 1) * 2)
            return fuse_weighted_hits(e_hits, b_hits, w_embed=w_e, w_bm25=w_b, top_k=k)
        return query_embed(idx_dir, query, topk=k)

    def forward(
        self,
        *,
        query: str,
        top_k: Optional[int] = None,
        video_id: Optional[str] = None,
        index_path: Optional[str] = None,
        original_question: Optional[str] = None,
    ) -> ToolOutput:
        try:
            if index_path:
                idx_dir = Path(index_path)
            else:
                env_dir = (os.getenv("VISUAL_INDEX_DIR") or os.getenv("SEMANTIC_INDEX") or "").strip()
                if not env_dir:
                    return ToolOutput(self.name, error="missing index_path and VISUAL_INDEX_DIR env for visual_retrieve")
                base = Path(env_dir)
                cand = (base / str(video_id)) if video_id else base
                idx_dir = cand if cand.exists() else base
            if not idx_dir.exists():
                return ToolOutput(self.name, error=f"index not found: {idx_dir}")

            k = env_int_first(("VISUAL_RETRIEVE_TOPK", "SEMANTIC_RETRIEVE_TOPK"), 100) if top_k is None else int(top_k)

            hits = self._retrieve_hits(idx_dir, query, k, original_question)

            caps_path = idx_dir / "semantic_captions.json"
            caps: Dict[str, Any] = {}
            if caps_path.exists():
                raw_caps = json.loads(caps_path.read_text(encoding="utf-8"))
                if not isinstance(raw_caps, dict):
                    raise RuntimeError(f"{caps_path} must be a JSON object mapping doc_id -> metadata.")
                caps = raw_caps

            entries: List[Tuple[str, int, int]] = []
            for doc_id, _score in hits:
                s_sec, e_sec = _parse_doc_id_time_window(doc_id)
                entries.append((doc_id, s_sec, e_sec))

            priority_channel = env_flag("RETRIEVAL_PRIORITY_CHANNEL", default=False)
            raw_candidates: List[Dict[str, Any]] = []
            if priority_channel:
                for rank, (doc_id, s_sec, e_sec) in enumerate(entries, start=1):
                    start_time = sec_to_hhmmss(s_sec)
                    end_time = sec_to_hhmmss(e_sec)
                    candidate: Dict[str, Any] = {
                        "candidate_id": f"bm25-r{rank}-{start_time}-{end_time}",
                        "source": "raw",
                        "query": str(query or "").strip(),
                        "rank": rank,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                    raw_candidates.append(candidate)

            min_gap = env_float_strict("RETRIEVE_MIN_TIME_GAP_SEC", 15.0)
            if min_gap < 0:
                min_gap = 0.0
            windows = [(float(s), float(e)) for (_, s, e) in entries]
            keep_indices = select_time_diverse_windows(windows, k=k, min_gap_sec=min_gap)

            out_items: List[Dict[str, Any]] = []
            for idx in keep_indices:
                doc_id, s_sec, e_sec = entries[idx]
                rec: Dict[str, Any] = {"start_time": sec_to_hhmmss(s_sec), "end_time": sec_to_hhmmss(e_sec)}
                meta = caps.get(doc_id)
                if isinstance(meta, dict):
                    cap = meta.get("caption") or meta.get("clip_caption") or meta.get("ocr_text")
                    if cap:
                        rec["caption"] = str(cap)
                out_items.append(rec)

            enabled_raw = os.getenv("VISUAL_RETRIEVE_SUMMARY_ENABLED") or os.getenv("RETRIEVE_SUMMARY_ENABLED") or "0"
            enabled = str(enabled_raw).strip().lower() in ("1", "true", "yes", "on")
            return_spans = env_flag("VISUAL_RETRIEVE_RETURN_SPANS", default=False)
            log_spans = env_flag("VISUAL_RETRIEVE_LOG_SPANS", default=False)
            log_meta = (
                {"spans": [{"start_time": it.get("start_time", ""), "end_time": it.get("end_time", "")} for it in out_items]} if log_spans else None
            )

            if not enabled:
                if return_spans:
                    return ToolOutput(self.name, output=out_items, metadata=log_meta)
                stripped: List[Dict[str, Any]] = []
                for it in out_items:
                    rec: Dict[str, Any] = {}
                    cap = str(it.get("caption") or "").strip()
                    if cap:
                        rec["caption"] = cap
                    stripped.append(rec)
                return ToolOutput(self.name, output=stripped, metadata=log_meta)

            max_useful = env_int_first(("VISUAL_RETRIEVE_SUMMARY_MAX_SPANS", "RETRIEVE_SUMMARY_MAX_SPANS"), 5)
            if max_useful <= 0:
                max_useful = 5

            cli = build_mllm_client_from_env_prefix("VISUAL_RETRIEVE_SUM")
            sum_max_tokens = int(os.getenv("VISUAL_RETRIEVE_SUM_MAX_TOKENS") or "800")
            sum_temperature = float(os.getenv("VISUAL_RETRIEVE_SUM_TEMPERATURE") or "0.0")

            focus_q = str(original_question or "").strip() or str(query or "").strip()
            prompt = build_visual_retrieve_summary_prompt(
                query_text=str(query or "").strip(),
                user_question=focus_q,
                candidates=out_items,
                max_useful=max_useful,
            )
            text = cli.generate_text(prompt, response_json=False, max_tokens=sum_max_tokens, temperature=sum_temperature) or ""
            # Preserve the released behavior unless the dual-channel ablation
            # is explicitly enabled. The old parser never matched a normal
            # HH:MM:SS span, so its fallback exposed the full candidate list.
            parsed = parse_hhmmss_spans_from_text(str(text)) if priority_channel else []
            allow = {(str(it.get("start_time") or "").strip(), str(it.get("end_time") or "").strip()) for it in out_items}
            want = [p for p in parsed if p in allow]

            useful_spans: List[Dict[str, str]] = []
            if want:
                wanted = set(want)
                for it in out_items:
                    s = str(it.get("start_time") or "").strip()
                    e = str(it.get("end_time") or "").strip()
                    if (s, e) in wanted:
                        useful_spans.append({"start_time": s, "end_time": e})
                        if len(useful_spans) >= max_useful:
                            break
            if not useful_spans and not priority_channel:
                useful_spans = [{"start_time": it.get("start_time", ""), "end_time": it.get("end_time", "")} for it in out_items[:max_useful]]

            summary = (str(text) or "").strip() or f"Condensed visual retrieval: selected {len(useful_spans)} spans; further verification may be needed."
            out_obj: Dict[str, Any] = {"summary": summary}
            if return_spans:
                out_obj["useful_spans"] = useful_spans
            if priority_channel:
                raw_by_span: Dict[Tuple[str, str], Dict[str, Any]] = {}
                for item in raw_candidates:
                    key = (str(item["start_time"]), str(item["end_time"]))
                    raw_by_span.setdefault(key, item)
                priority_spans: List[Dict[str, Any]] = []
                for span in useful_spans:
                    key = (str(span.get("start_time") or ""), str(span.get("end_time") or ""))
                    raw = raw_by_span.get(key, {})
                    priority_spans.append(
                        {
                            "candidate_id": raw.get("candidate_id", f"priority-{key[0]}-{key[1]}"),
                            "source": "priority",
                            "query": str(query or "").strip(),
                            "rank": raw.get("rank"),
                            "start_time": key[0],
                            "end_time": key[1],
                        }
                    )
                out_obj["priority_spans"] = priority_spans
                out_obj["raw_candidates"] = raw_candidates
                channel_meta = {
                    "query": str(query or "").strip(),
                    "priority_spans": priority_spans,
                    "raw_candidates": raw_candidates,
                }
                if log_meta is None:
                    log_meta = {}
                log_meta["retrieval_priority_channel"] = channel_meta
            if log_spans:
                if log_meta is None:
                    log_meta = {}
                log_meta["spans"] = useful_spans
            return ToolOutput(self.name, output=out_obj, metadata=log_meta)
        except Exception as e:
            return ToolOutput(self.name, error=format_exception_with_request_id(e))


class VisualInspectAliasTool(Tool):
    def __init__(self, name: str | None = None, description: str | None = None, function=None):
        super().__init__(
            name=name or "visual_inspect",
            description=description or "Execute fine-grained verification across specified time windows using a VLM (multi-span supported).",
        )

    @property
    def json(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spans": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"start_time": {"type": "string"}, "end_time": {"type": "string"}},
                                "required": ["start_time", "end_time"],
                            },
                        },
                        "context": {"type": "string", "description": "Optional short context from retrieval to guide inspection."},
                    },
                    "required": ["spans"],
                },
            },
        }

    def forward(
        self,
        *,
        video_path: str,
        spans: List[Dict[str, str]],
        questions: Any,
        context: Optional[str] = None,
        prompt_type: int = 0,
    ) -> ToolOutput:
        try:
            if not video_path:
                return ToolOutput(self.name, error="missing video_path")

            qtext_raw = ""
            qtext_tagged = ""
            if isinstance(questions, list):
                qs = [str(q).strip() for q in questions if str(q).strip()]
                qtext_raw = "\n".join(qs) if qs else ""
                qtext_tagged = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(qs)]) if qs else ""
            else:
                q = str(questions or "").strip()
                qtext_raw = q
                qtext_tagged = f"Q1: {q}" if q else ""

            fps = env_float_strict("INSPECT_FPS", 2.0)
            do_global_order = env_flag("INSPECT_GLOBAL_ORDER", default=False)
            max_total = env_int_strict("INSPECT_MAX_TOTAL_IMAGES", 48)
            base_max_edge = env_int_first(("VISUAL_INSPECT_MAX_LONG_EDGE", "INSPECT_MAX_LONG_EDGE"), 0)
            if base_max_edge > 0:
                os.environ["FRAME_MAX_LONG_SIDE"] = str(base_max_edge)

            win_list: List[tuple[float, float]] = []
            for sp in spans or []:
                s = float(srt_timestamp_to_seconds(str(sp.get("start_time") or "")))
                e = float(srt_timestamp_to_seconds(str(sp.get("end_time") or "")))
                ns, ne = normalize_window_min_width(video_path, s, e)
                win_list.append((ns, ne))
            if not win_list:
                return ToolOutput(self.name, error="no spans parsed")

            vid_id = video_id_from_path(video_path)
            frames_dir = make_unique_run_dir(FRAMES_ROOT, video_id=vid_id, prefix="vis")

            try:
                per_span_frames: List[List[str]] = []
                per_span_ts: List[List[float]] = []
                subtitle_path = (os.getenv("SUBTITLE_PATH") or "").strip() or None
                for (a, b) in win_list:
                    fpaths, ts = sample_uniform_frames(
                        video_path,
                        float(a),
                        float(b),
                        fps=fps,
                        max_frames=max_total,
                        output_dir=str(frames_dir),
                        subtitle_path=subtitle_path,
                    )
                    per_span_frames.append(fpaths)
                    per_span_ts.append(ts)

                frames_all, ts_all = balanced_select_frames(
                    per_span_frames=per_span_frames,
                    per_span_ts=per_span_ts,
                    span_windows=win_list,
                    max_total=max_total,
                    fps_fallback=fps,
                    global_order=bool(do_global_order),
                )
                if not frames_all:
                    return ToolOutput(self.name, error="no frames extracted for spans")

                dyn_edge_enabled = env_flag("VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE", default=False) or env_flag(
                    "INSPECT_DYNAMIC_MAX_LONG_EDGE", default=False
                )
                if dyn_edge_enabled:
                    total_pixels = env_int_first(("VISUAL_INSPECT_TOTAL_PIXELS", "INSPECT_TOTAL_PIXELS"), 0)
                    min_pixels = env_int_first(("VISUAL_INSPECT_MIN_PIXELS", "INSPECT_MIN_PIXELS"), 0)
                    edge_multiple = env_int_strict("VISUAL_INSPECT_EDGE_MULTIPLE", 32)
                    n_frames = len(frames_all)
                    if total_pixels > 0 and n_frames > 0:
                        per_frame = float(total_pixels) / float(n_frames)
                        if min_pixels > 0 and per_frame < float(min_pixels):
                            per_frame = float(min_pixels)

                        from PIL import Image

                        with Image.open(frames_all[0]) as im:
                            w0, h0 = im.size
                        short = max(1, min(int(w0), int(h0)))
                        long = max(1, max(int(w0), int(h0)))
                        aspect = float(long) / float(short) if short > 0 else 1.0
                        if aspect < 1.0:
                            aspect = 1.0

                        dyn_edge = int(math.floor(math.sqrt(per_frame * aspect)))
                        if edge_multiple > 1:
                            dyn_edge = max(edge_multiple, (dyn_edge // edge_multiple) * edge_multiple)
                        if base_max_edge > 0 and dyn_edge > base_max_edge:
                            dyn_edge = base_max_edge
                        if dyn_edge > 0 and not (base_max_edge > 0 and dyn_edge >= base_max_edge):
                            resize_images_max_long_side(frames_all, max_long_side=dyn_edge)
                            os.environ["FRAME_MAX_LONG_SIDE"] = str(dyn_edge)

                spans_label = ", ".join([f"{sec_to_hhmmss(a)}–{sec_to_hhmmss(b)}" for (a, b) in win_list])
                prompt_mode = (os.getenv("VISUAL_INSPECT_PROMPT_MODE") or "").strip().lower()
                if prompt_mode in {"question_only", "raw_question", "raw", "question"}:
                    prompt = qtext_raw
                elif prompt_mode in {"mcq", "mcq_last_step", "last_step_mcq", "laststep_mcq"}:
                    prompt = build_visual_inspect_last_step_mcq_prompt(spans_label, qtext_tagged, context=context)
                else:
                    prompt = build_visual_inspect_prompt(spans_label, qtext_tagged, context=context, prompt_type=int(prompt_type))

                cli = build_mllm_client_from_env_prefix("VISUAL_INSPECT")
                max_tok = env_int_strict("INSPECT_VLM_MAX_TOKENS", 800)
                temp = env_float_strict("INSPECT_VLM_TEMPERATURE", 0.0)
                raw_answer_text = str(cli.generate_images_paths(
                    frames_all,
                    prompt,
                    response_json=False,
                    timestamps=ts_all,
                    max_tokens=max_tok,
                    temperature=temp,
                ) or "").strip()
                exclusive_output = env_flag("VISUAL_INSPECT_EXCLUSIVE_OUTPUT", default=False)
                if exclusive_output and prompt_mode not in {"mcq", "mcq_last_step", "last_step_mcq", "laststep_mcq"}:
                    answer_text, control_state = normalize_visual_inspect_control_text(raw_answer_text)
                else:
                    answer_text, control_state = raw_answer_text, "disabled"
                output = {
                    "answer": answer_text,
                    "spans": [{"start_time": sec_to_hhmmss(a), "end_time": sec_to_hhmmss(b)} for (a, b) in win_list],
                }
                metadata = None
                # Optional audit-only instrumentation.  It is disabled by
                # default, so released VideoSEAL behavior and payloads stay
                # unchanged.  HSI enables it to record the actual final frame
                # timestamps and image sizes used by the inspector.
                if env_flag("VISUAL_INSPECT_RETURN_FRAME_METADATA", default=False):
                    actual_pixels = 0
                    try:
                        from PIL import Image

                        for frame_path in frames_all:
                            with Image.open(frame_path) as image:
                                actual_pixels += int(image.width) * int(image.height)
                    except Exception:
                        actual_pixels = 0
                    metadata = {
                        "frame_count": len(frames_all),
                        "timestamps": [float(value) for value in ts_all],
                        "frame_selection_mode": (
                            os.getenv("VISUAL_INSPECT_FRAME_SELECTION")
                            or os.getenv("INSPECT_FRAME_SELECTION")
                            or "released_round_robin"
                        ),
                        "actual_pixels": actual_pixels,
                        "max_tokens": max_tok,
                        "configured_total_pixels": env_int_first(
                            ("VISUAL_INSPECT_TOTAL_PIXELS", "INSPECT_TOTAL_PIXELS"), 0
                        ),
                    }
                if exclusive_output:
                    if metadata is None:
                        metadata = {}
                    metadata["exclusive_inspector_output"] = {
                        "control_state": control_state,
                        "changed": answer_text != raw_answer_text,
                        "raw_answer": raw_answer_text,
                    }
                return ToolOutput(self.name, output=output, metadata=metadata)
            finally:
                cleanup_dir_tree(frames_dir, enabled=True, label="FRAMES")
        except Exception as e:
            return ToolOutput(self.name, error=format_exception_with_request_id(e))
