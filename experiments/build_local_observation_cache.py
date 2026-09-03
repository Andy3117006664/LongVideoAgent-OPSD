#!/usr/bin/env python3
"""Generate a deterministic per-clip visual observation bank via local vLLM.

The endpoint is expected to be on localhost/a private host and is never an
external provider.  One generic observation per clip keeps rollout lookups
cheap and makes the paired replay reproducible.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


FRAME_NUMS = list(range(1, 181, 15))


def image_data(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        enc = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{enc}", digest


def call_one(client, model: str, root: Path, clip: str, max_tokens: int, retries: int):
    paths = [root / clip / f"{n:05d}.jpg" for n in FRAME_NUMS]
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return clip, {"observation": "", "frame_count": 0, "frame_paths": [], "frame_sha256": []}
    content = []
    hashes = []
    for p in existing:
        try:
            url, digest = image_data(p)
            content.append({"type": "image_url", "image_url": {"url": url}})
            hashes.append(digest)
        except Exception:
            continue
    content.append({
        "type": "text",
        "text": "Describe the visible events, people, objects, actions, and scene changes across these sampled frames. Be factual and concise; do not infer anything not visible. Return one paragraph.",
    })
    last = None
    for attempt in range(max(1, retries)):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=max_tokens,
                seed=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            return clip, {
                "observation": text,
                "frame_count": len(hashes),
                "frame_paths": [str(p.relative_to(root)) for p in existing],
                "frame_sha256": hashes,
            }
        except Exception as exc:
            last = exc
            time.sleep(min(10, 2**attempt))
    return clip, {"observation": f"[LOCAL_VLM_ERROR {type(last).__name__}]", "frame_count": len(hashes), "frame_paths": [str(p.relative_to(root)) for p in existing], "frame_sha256": hashes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--clips-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    from openai import OpenAI

    clips = sorted({x.strip() for x in Path(args.clips_file).read_text().splitlines() if x.strip()})
    root = Path(args.frames_root)
    results = {}

    def worker(clip):
        client = OpenAI(api_key="local", base_url=args.base_url)
        return call_one(client, args.model, root, clip, args.max_tokens, args.retries)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(worker, clip): clip for clip in clips}
        for i, fut in enumerate(as_completed(futures), 1):
            clip, payload = fut.result()
            results[clip] = payload
            if i % 10 == 0 or i == len(clips):
                print(f"{i}/{len(clips)}", flush=True)
    out = {
        "schema_version": "local-observation-cache-v1",
        "model": args.model,
        "base_url": args.base_url,
        "frame_policy": {"start": 1, "end": 180, "step": 15, "format": "jpeg", "quality": 95},
        "prompt": "generic_visible_events_v1",
        "count": len(results),
        "clips": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}: {len(results)} clips")


if __name__ == "__main__":
    main()
