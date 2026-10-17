from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import CandidateSpan, LETTERS


def read_protocol(path: Path) -> list[tuple[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    items = value.get("items") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise ValueError("protocol must contain an items list")
    keys = [
        (str(item.get("video_id") or "").strip(), str(item.get("uid") or "").strip())
        for item in items
        if isinstance(item, dict)
    ]
    if any(not all(key) for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("protocol keys must be unique and non-empty")
    return keys


def read_baseline(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        key = (str(row.get("video_id") or ""), str(row.get("uid") or ""))
        prediction = str(row.get("prediction") or "").strip().upper()
        if all(key) and prediction in LETTERS and row.get("state") == "completed":
            output[key] = row
    return output


def _doc_span(doc_id: str) -> tuple[float, float]:
    pieces = str(doc_id).split("_")
    if len(pieces) != 2:
        raise ValueError(f"invalid fixed-30 doc id: {doc_id}")
    start, end = float(pieces[0]), float(pieces[1])
    if start < 0.0 or end <= start:
        raise ValueError(f"invalid fixed-30 span: {doc_id}")
    return start, end


def first_retrieval_candidates(trajectory_path: Path, limit: int = 8) -> tuple[CandidateSpan, ...]:
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    for step in trajectory.get("steps") or []:
        action = step.get("action") if isinstance(step, dict) else None
        observation = step.get("observation") if isinstance(step, dict) else None
        if not isinstance(action, dict) or action.get("name") != "visual_retrieve":
            continue
        if not isinstance(observation, dict) or observation.get("ok") is not True:
            continue
        metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
        retrieval = metadata.get("fixed30_retrieval") if isinstance(metadata.get("fixed30_retrieval"), dict) else {}
        doc_ids = retrieval.get("selected_doc_ids")
        if not isinstance(doc_ids, list) or not doc_ids:
            continue
        output = []
        for rank, doc_id in enumerate(doc_ids[:limit], start=1):
            start, end = _doc_span(str(doc_id))
            output.append(CandidateSpan(str(doc_id), start, end, rank))
        return tuple(output)
    raise ValueError("trajectory has no fixed-30 first-retrieval candidate list")

