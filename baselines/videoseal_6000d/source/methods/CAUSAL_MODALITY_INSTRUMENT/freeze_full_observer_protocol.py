#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freeze_protocol import normalize_spans, selected_inspection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.funnel.read_text(encoding="utf-8"))["rows"]
    items = []
    for row in rows:
        inspection = selected_inspection(row)
        spans = normalize_spans(inspection)
        if not spans:
            continue
        uid = str(row["uid"])
        video_id = str(row["video_id"])
        items.append({
            "video_id": video_id,
            "uid": uid,
            "spans": spans,
            "inspection_step": int(inspection.get("step") or 0),
            "rank": hashlib.sha256(f"fixed30-full-observer-v1:{video_id}:{uid}".encode()).hexdigest(),
        })
    items.sort(key=lambda item: (item["video_id"], item["uid"]))
    protocol = {
        "schema_version": "fixed30-full-observer-protocol-v1",
        "selection": {
            "source": str(args.funnel),
            "reads_question_text": False,
            "reads_agent_requested_spans": True,
            "reads_gold": False,
            "reads_baseline_correctness": False,
            "reads_time_reference": False,
            "all_rows_with_nonforced_inspection_spans": True,
        },
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": len(items)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
