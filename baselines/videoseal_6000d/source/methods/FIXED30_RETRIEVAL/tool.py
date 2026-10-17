"""A minimal fixed-30 retrieval-only ablation.

The adapter changes only the caption ranking used by ``visual_retrieve``.  It
leaves VideoSEAL's summarizer, tool output, planner, and inspector untouched.
No benchmark reference interval is read here.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np

from videoseal.tools.base import ToolOutput
from videoseal.tools.visual_tools import VisualRetrieveAliasTool
from videoseal.utils.RAG.rag_query_bm25 import query_bm25


_OPTION_RE = re.compile(r"(?m)^\s*\([A-Da-d]\)\s*(.+?)\s*$")
_TIME_TOKEN = r"(?:\d{1,3}:\d{2}:\d{2}(?:\.\d+)?|\d{1,4}:\d{2}(?:\.\d+)?)"
_TIME_RANGE_RE = re.compile(
    rf"(?P<start>{_TIME_TOKEN})\s*(?:-|–|—|to)\s*(?P<end>{_TIME_TOKEN})",
    re.IGNORECASE,
)
_TIME_POINT_RE = re.compile(rf"(?<![\d:])(?P<point>{_TIME_TOKEN})(?![\d:])")
_DENSE_ARRAY = "fixed30_bge_small.npy"
_DENSE_METADATA = "fixed30_bge_small.json"
_RRF_WEIGHTS = (0.5, 0.5, 0.5, 1.0, 0.5)
_RRF_CONSTANT = 60


def _question_parts(question: str) -> tuple[str, list[str]]:
    text = str(question or "")
    matches = list(_OPTION_RE.finditer(text))
    if not matches:
        return text.strip(), []
    return text[: matches[0].start()].strip(), [match.group(1).strip() for match in matches]


def _seconds(token: str) -> float:
    parts = [float(part) for part in str(token).split(":")]
    if len(parts) == 2:
        return 60.0 * parts[0] + parts[1]
    if len(parts) == 3:
        return 3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
    raise ValueError(f"invalid explicit time token: {token!r}")


def _doc_window(doc_id: str) -> tuple[float, float]:
    parts = str(doc_id).split("_")
    if len(parts) != 2:
        raise ValueError(f"invalid caption window id: {doc_id!r}")
    return float(parts[0]), float(parts[1])


def _explicit_intervals(stem: str) -> list[tuple[float, float]]:
    """Extract video times from the question stem, never from answer options."""
    intervals: list[tuple[float, float]] = []
    occupied: list[tuple[int, int]] = []
    for match in _TIME_RANGE_RE.finditer(stem):
        start = _seconds(match.group("start"))
        end = _seconds(match.group("end"))
        if end < start:
            start, end = end, start
        intervals.append((start, end))
        occupied.append(match.span())
    for match in _TIME_POINT_RE.finditer(stem):
        if any(left <= match.start() and match.end() <= right for left, right in occupied):
            continue
        point = _seconds(match.group("point"))
        intervals.append((point, point))
    return list(dict.fromkeys(intervals))


def _uniform_cap(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return values
    if limit <= 1:
        return values[: max(0, limit)]
    positions = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[int(position)] for position in positions]


def _pinned_doc_ids(doc_ids: list[str], question_stem: str, limit: int) -> list[str]:
    windows = [(doc_id, *_doc_window(doc_id)) for doc_id in doc_ids]
    pinned: list[str] = []
    for ref_start, ref_end in _explicit_intervals(question_stem):
        if ref_start == ref_end:
            matched = [doc_id for doc_id, start, end in windows if start <= ref_start < end]
            if not matched and windows:
                matched = [min(windows, key=lambda item: abs(0.5 * (item[1] + item[2]) - ref_start))[0]]
        else:
            matched = [
                doc_id
                for doc_id, start, end in windows
                if max(start, ref_start) <= min(end, ref_end)
            ]
        pinned.extend(matched)
    return _uniform_cap(list(dict.fromkeys(pinned)), limit)


def _dense_rank(doc_ids: list[str], document_vectors: np.ndarray, query_vector: np.ndarray) -> list[str]:
    if document_vectors.ndim != 2 or query_vector.ndim != 1:
        raise RuntimeError("fixed30 dense vectors have invalid dimensions")
    if len(doc_ids) != document_vectors.shape[0] or document_vectors.shape[1] != query_vector.shape[0]:
        raise RuntimeError(
            f"fixed30 dense shape mismatch: ids={len(doc_ids)} docs={document_vectors.shape} query={query_vector.shape}"
        )
    similarities = document_vectors @ query_vector
    order = np.argsort(-similarities, kind="stable")
    return [doc_ids[int(index)] for index in order]


def _option_max_rank(index_dir: Path, options: Iterable[str], pool_k: int) -> list[str]:
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for option in options:
        for rank, (doc_id, _score) in enumerate(query_bm25(index_dir, option, topk=pool_k), start=1):
            key = str(doc_id)
            best_rank[key] = min(best_rank.get(key, rank), rank)
            if key not in first_seen:
                first_seen[key] = seen_counter
                seen_counter += 1
    return sorted(best_rank, key=lambda doc_id: (best_rank[doc_id], first_seen[doc_id]))


def _rrf(rankings: Iterable[Iterable[str]], weights: Iterable[float]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            key = str(doc_id)
            scores[key] += float(weight) / (_RRF_CONSTANT + rank)
            if key not in first_seen:
                first_seen[key] = seen_counter
                seen_counter += 1
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]))


def _embed(texts: list[str]) -> list[np.ndarray]:
    endpoint = str(os.getenv("FIXED30_BGE_ENDPOINT") or "http://127.0.0.1:18109/embed").strip()
    timeout = float(os.getenv("FIXED30_BGE_TIMEOUT") or "30")
    payload = json.dumps({"texts": texts}, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urlerror.URLError) as exc:
        raise RuntimeError(f"fixed30 BGE query service failed at {endpoint}: {exc}") from exc
    vectors = value.get("embeddings") if isinstance(value, dict) else None
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise RuntimeError("fixed30 BGE query service returned an invalid batch")
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


class Fixed30VisualRetrieveTool(VisualRetrieveAliasTool):
    """Five-signal RRF with explicit-time pinning and a hard 30-window cap."""

    def __init__(self, name: str | None = None, description: str | None = None, function=None):
        super().__init__(name=name, description=description, function=function)
        self._fixed30_audit: dict[str, Any] = {}

    def _retrieve_hits(
        self,
        idx_dir: Path,
        query: str,
        k: int,
        original_question: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        hard_k = min(30, max(1, int(k)))
        question = str(original_question or "").strip()
        _stem, options = _question_parts(question)

        metadata_path = idx_dir / _DENSE_METADATA
        vectors_path = idx_dir / _DENSE_ARRAY
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        doc_ids = [str(item) for item in metadata.get("doc_ids") or []]
        document_vectors = np.load(vectors_path, mmap_mode="r")
        if not doc_ids or len(doc_ids) != document_vectors.shape[0]:
            raise RuntimeError(f"fixed30 dense index is incomplete: {idx_dir}")

        planner_vector, question_vector = _embed([str(query or "").strip(), question])
        configured_depth = int(os.getenv("FIXED30_SIGNAL_DEPTH") or "0")
        pool_k = len(doc_ids) if configured_depth <= 0 else min(len(doc_ids), configured_depth)

        planner_bm25 = [doc_id for doc_id, _ in query_bm25(idx_dir, query, topk=pool_k)]
        planner_bge = _dense_rank(doc_ids, document_vectors, planner_vector)[:pool_k]
        question_bm25 = [doc_id for doc_id, _ in query_bm25(idx_dir, question, topk=pool_k)]
        question_bge = _dense_rank(doc_ids, document_vectors, question_vector)[:pool_k]
        option_bm25 = _option_max_rank(idx_dir, options, pool_k)[:pool_k]

        fused = _rrf(
            (planner_bm25, planner_bge, question_bm25, question_bge, option_bm25),
            _RRF_WEIGHTS,
        )
        # The frozen protocol treats every explicit timestamp supplied by the
        # user as admissible evidence, including timestamp answer options in
        # "when" questions.  These are input-side candidate times, not the
        # benchmark-only reference interval.
        pinned = _pinned_doc_ids(doc_ids, question, hard_k)
        selected = list(dict.fromkeys(pinned + fused))[:hard_k]
        self._fixed30_audit = {
            "schema_version": 1,
            "method": "fixed30_five_signal_rrf",
            "gold_used": False,
            "weights": list(_RRF_WEIGHTS),
            "rrf_constant": _RRF_CONSTANT,
            "signal_depth": pool_k,
            "signal_sizes": {
                "planner_bm25": len(planner_bm25),
                "planner_bge": len(planner_bge),
                "question_bm25": len(question_bm25),
                "question_bge": len(question_bge),
                "option_bm25_max": len(option_bm25),
            },
            "explicit_time_intervals": _explicit_intervals(question),
            "pinned_doc_ids": pinned,
            "selected_doc_ids": selected,
            "selected_count": len(selected),
            "hard_cap": hard_k,
        }
        return [(doc_id, float(len(selected) - rank)) for rank, doc_id in enumerate(selected)]

    def forward(
        self,
        *,
        query: str,
        top_k: Optional[int] = None,
        video_id: Optional[str] = None,
        index_path: Optional[str] = None,
        original_question: Optional[str] = None,
    ) -> ToolOutput:
        output = super().forward(
            query=query,
            top_k=min(30, int(top_k)) if top_k is not None else 30,
            video_id=video_id,
            index_path=index_path,
            original_question=original_question,
        )
        metadata = dict(output.metadata) if isinstance(output.metadata, dict) else {}
        metadata["fixed30_retrieval"] = dict(self._fixed30_audit)
        output.metadata = metadata
        return output
