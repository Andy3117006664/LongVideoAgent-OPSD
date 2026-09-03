#!/usr/bin/env python3
"""Evaluate a text-only LongTVQA(+) control with full episode subtitles."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_prompt(row: dict, episode_subs: dict) -> str:
    ep = str(row.get("episode_name") or row.get("show_name") or "")
    subs = str(episode_subs.get(ep, ""))
    opts = "\n".join(f"a{i}: {row.get(f'a{i}', '')}" for i in range(5))
    return (
        "You are answering a multiple-choice question about a TV episode. "
        "Use only the episode subtitles below. Return exactly one option tag "
        "(a0, a1, a2, a3, or a4), with no explanation.\n\n"
        f"Question: {row.get('q', row.get('question', ''))}\n{opts}\n"
        f"Episode subtitles:\n{subs}\nAnswer:"
    )


def parse_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(a[0-4])\s*</answer>", text, re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b(a[0-4])\b", text, re.I)
    return m.group(1).lower() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--episode-subs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-input-tokens", type=int, default=24000)
    args = ap.parse_args()
    rows = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    subs = json.loads(Path(args.episode_subs).read_text(encoding="utf-8"))
    rows = [r for r in rows if isinstance(r, dict)]
    if args.limit > 0:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    results = []
    start = time.time()
    for i in range(0, len(rows), max(1, args.batch_size)):
        batch_rows = rows[i : i + max(1, args.batch_size)]
        prompts = [make_prompt(r, subs) for r in batch_rows]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_tokens)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        in_len = enc["input_ids"].shape[1]
        decoded = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
        for row, text in zip(batch_rows, decoded, strict=True):
            pred = parse_answer(text)
            gt = str(row.get("answer", "")).lower().strip()
            results.append({"qid": row.get("qid"), "prediction": pred, "ground_truth": gt, "raw": text})
        if (i // max(1, args.batch_size)) % 10 == 0:
            print(f"{min(i + len(batch_rows), len(rows))}/{len(rows)} elapsed={time.time()-start:.1f}s", flush=True)
    correct = sum(x["prediction"] == x["ground_truth"] for x in results)
    out = {
        "variant": "B0_text_only_full_episode_subtitles",
        "model": args.model,
        "count": len(results),
        "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"accuracy={out['accuracy']:.4f} ({correct}/{len(results)}) wrote {args.output}")


if __name__ == "__main__":
    main()
