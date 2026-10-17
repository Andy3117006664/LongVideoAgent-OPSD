#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import pyarrow.parquet as pq


AUDIO_QUERY = re.compile(
    r"\b(say|says|said|tell|tells|told|ask|asks|asked|mention|mentions|mentioned|"
    r"call|calls|called|apologi[sz]|announce|announces|announced|declare|declares|"
    r"declared|talk|talks|speak|speaks|hear|heard|name|named|how much|price|cost|"
    r"pay|paid)\b|^\s*(why|who)\b",
    re.IGNORECASE,
)
OLD_ROOT = "/path/to/legacy-project"


def project_path(value: Any, root: Path) -> Path:
    text = str(value or "")
    if text == OLD_ROOT or text.startswith(OLD_ROOT + "/"):
        text = str(root) + text[len(OLD_ROOT) :]
    return Path(text)


def selected_inspection(row: dict[str, Any]) -> dict[str, Any]:
    values = [
        item
        for item in row.get("inspections") or []
        if isinstance(item, dict)
        and not item.get("forced_full_video")
        and item.get("requested_spans")
    ]
    decisive = [item for item in values if str(item.get("verdict") or "").upper() in set("ABCD")]
    return (decisive or values or [{}])[-1]


def normalize_spans(item: dict[str, Any]) -> list[list[float]]:
    output: list[list[float]] = []
    for raw in item.get("requested_spans") or []:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        start, end = float(raw[0]), float(raw[1])
        if start >= 0.0 and end > start and [start, end] not in output:
            output.append([start, end])
    return output


def has_audio(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return bool(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--exclude-protocol", type=Path, action="append", default=[])
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--seed", default="cmi-outcome-blind-dev-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-filter", choices=("audio", "all"), default="audio")
    parser.add_argument("--require-audio", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    rows = json.loads(args.funnel.read_text(encoding="utf-8"))["rows"]
    tasks = {
        str(info.get("qa_uid") or ""): dict(info)
        for info in pq.read_table(args.parquet, columns=["extra_info"])["extra_info"].to_pylist()
    }
    blocked: set[str] = set()
    for path in args.exclude_protocol:
        blocked.update(
            str(item.get("video_id") or "")
            for item in json.loads(path.read_text(encoding="utf-8")).get("items") or []
        )

    audio_cache: dict[Path, bool] = {}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        video_id = str(row.get("video_id") or "")
        uid = str(row.get("uid") or "")
        question = str(row.get("question") or "")
        if video_id in blocked or uid not in tasks:
            continue
        if args.question_filter == "audio" and not AUDIO_QUERY.search(question):
            continue
        inspection = selected_inspection(row)
        spans = normalize_spans(inspection)
        if not spans:
            continue
        video = project_path(tasks[uid].get("VIDEO_PATH"), root)
        if video not in audio_cache:
            audio_cache[video] = video.is_file() and has_audio(video)
        if args.require_audio and not audio_cache[video]:
            continue
        candidates.append(
            {
                "video_id": video_id,
                "uid": uid,
                "spans": spans,
                "inspection_step": int(inspection.get("step") or 0),
                "rank": hashlib.sha256(f"{args.seed}:{video_id}:{uid}".encode()).hexdigest(),
            }
        )

    selected = []
    used = set(blocked)
    for item in sorted(candidates, key=lambda value: value["rank"]):
        if item["video_id"] in used:
            continue
        selected.append(item)
        used.add(item["video_id"])
        if len(selected) == args.size:
            break
    if len(selected) != args.size:
        raise RuntimeError(f"needed {args.size} items, found {len(selected)}")

    protocol = {
        "schema_version": "causal-modality-instrument-protocol-v1",
        "seed": args.seed,
        "selection": {
            "size": args.size,
            "reads_question_text": True,
            "reads_agent_requested_spans": True,
            "reads_gold": False,
            "reads_baseline_correctness": False,
            "reads_time_reference": False,
            "one_question_per_video": True,
            "requires_audio_stream": bool(args.require_audio),
        },
        "question_filter": AUDIO_QUERY.pattern if args.question_filter == "audio" else None,
        "excluded_protocols": [str(path) for path in args.exclude_protocol],
        "items": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": len(selected), "candidates": len(candidates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
