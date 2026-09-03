#!/usr/bin/env python3
"""Build a deterministic, no-API grounding cache from LongTVQA subtitles.

The retriever only sees the question/options and episode subtitles.  It never
uses ``occur_clip``; that field is retained solely for later evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")


def toks(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def uid(item: dict, source_name: str, idx: int) -> str:
    qid = item.get("qid")
    if qid is not None and str(qid).strip():
        base = f"qid:{str(qid).strip()}"
    else:
        raw = "\x1f".join([str(item.get("video_id") or item.get("vid_name") or item.get("occur_clip") or ""), str(item.get("q") or item.get("question") or ""), str(idx)])
        base = "row:" + hashlib.sha1(raw.encode()).hexdigest()[:20]
    return f"{source_name}:{base}"


def score(query: list[str], doc: list[str], idf: dict[str, float], avgdl: float) -> float:
    if not query or not doc:
        return -1e9
    q = Counter(query)
    d = Counter(doc)
    k1, b = 1.2, 0.75
    dl = len(doc)
    total = 0.0
    for term, qtf in q.items():
        tf = d.get(term, 0)
        if not tf:
            continue
        denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
        total += idf.get(term, 0.0) * (tf * (k1 + 1.0) / denom) * min(qtf, 2)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--clip-subtitles", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    clip_subs = json.loads(Path(args.clip_subtitles).read_text(encoding="utf-8"))
    if not isinstance(clip_subs, dict):
        raise ValueError("clip subtitle file must be a JSON object")
    by_episode: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for clip, text in clip_subs.items():
        clip = str(clip)
        episode = clip.split("_seg", 1)[0] if "_seg" in clip else clip[:6]
        by_episode[episode].append((clip, toks(str(text))))

    # Episode-local retrieval is what the grounding prompt permits.  Build a
    # global IDF only to downweight names/common subtitle filler.
    df = Counter()
    n_docs = 0
    lengths = []
    for docs in by_episode.values():
        for _, words in docs:
            n_docs += 1
            lengths.append(len(words))
            df.update(set(words))
    idf = {w: math.log((n_docs + 1.0) / (v + 0.5)) for w, v in df.items()}
    avgdl = sum(lengths) / max(len(lengths), 1)

    by_uid: dict[str, dict] = {}
    by_qid: dict[str, dict] = {}
    by_question: dict[str, dict] = {}
    total = 0
    for qpath in args.questions:
        source_name = Path(qpath).name
        rows = json.loads(Path(qpath).read_text(encoding="utf-8"))
        for idx, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            episode = str(item.get("episode_name") or item.get("show_name") or "").strip()
            if not episode:
                occ = str(item.get("occur_clip") or "")
                episode = occ.split("_seg", 1)[0] if "_seg" in occ else occ[:6]
            docs = by_episode.get(episode, [])
            query_text = " ".join(
                [str(item.get("q") or item.get("question") or "")]
                + [str(item.get(f"a{i}") or "") for i in range(5)]
            )
            query = toks(query_text)
            ranked = sorted(
                ((score(query, words, idf, avgdl), clip) for clip, words in docs),
                reverse=True,
            )
            clip = ranked[0][1] if ranked else ""
            payload = {
                "clip": clip,
                "score": ranked[0][0] if ranked else None,
                "retriever": "bm25-subtitles-v1",
                "used_ground_truth_clip": False,
            }
            suid = uid(item, source_name, idx)
            by_uid[suid] = payload
            if item.get("qid") is not None:
                by_qid[str(item["qid"])] = payload
            qkey = " ".join(str(item.get("q") or item.get("question") or "").split()).lower()
            if qkey:
                by_question[qkey] = payload
            total += 1

    out = {
        "schema_version": "local-grounding-cache-v1",
        "retriever": "bm25-subtitles-v1",
        "used_ground_truth_clip": False,
        "count": total,
        "by_uid": by_uid,
        "by_qid": by_qid,
        "by_question": by_question,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}: {total} rows, {len(by_uid)} uid entries")


if __name__ == "__main__":
    main()
