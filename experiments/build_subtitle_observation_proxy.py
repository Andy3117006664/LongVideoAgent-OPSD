#!/usr/bin/env python3
"""Build a clearly labelled deterministic subtitle observation bank.

This is only a plumbing/control arm while real frame/VLM assets are being
downloaded. It must not be reported as a visual-model result.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--subtitles", required=True)
ap.add_argument("--clips-file", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()
subs = json.loads(Path(a.subtitles).read_text(encoding="utf-8"))
clips = sorted({x.strip() for x in Path(a.clips_file).read_text().splitlines() if x.strip()})
rows = {}
for clip in clips:
    text = str(subs.get(clip, "")).strip()
    rows[clip] = {
        "observation": f"Subtitle-backed proxy for {clip}: {text}",
        "frame_count": 0,
        "frame_paths": [],
        "frame_sha256": [],
    }
out = {
    "schema_version": "subtitle-observation-proxy-v1",
    "model": "none-subtitle-proxy",
    "prompt": "not-visual; control-only",
    "frame_policy": {"start": 1, "end": 180, "step": 15},
    "count": len(rows),
    "clips": rows,
}
p = Path(a.output)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"wrote {p}: {len(rows)} clips")
