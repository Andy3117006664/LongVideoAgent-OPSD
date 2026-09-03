#!/usr/bin/env python3
"""Small, deterministic LongTVQA+ observation-sensitivity probe.

This is deliberately a *proxy* diagnostic: it measures a single Qwen model's
teacher-forced answer log-probability change when a tool observation is
replaced by an equal-length sentinel.  It is not the teacher-student OPD gap
until a separate teacher checkpoint/API logit cache is supplied.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _pad(rows: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(x) for x in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, : len(row)] = 1
    return ids, mask


@torch.inference_mode()
def score_suffixes(model, rows: list[list[int]], suffix_spans: list[tuple[int, int]], pad_id: int, device: torch.device) -> list[float]:
    """Score one row at a time and ask Qwen to retain only suffix logits.

    Returning vocab logits for a 12k-token prompt would otherwise consume many
    GB per example.  Qwen2's ``logits_to_keep`` retains the final k positions;
    k+1 positions are requested so the first retained row predicts the first
    suffix token.  A full-logit fallback keeps this usable on older versions.
    """
    out: list[float] = []
    for row, (start, end) in zip(rows, suffix_spans):
        if end <= start or start <= 0:
            out.append(float("nan"))
            continue
        ids = torch.tensor(row, dtype=torch.long, device=device)[None, :]
        attn = torch.ones_like(ids)
        k = end - start
        try:
            logits = model(input_ids=ids, attention_mask=attn, logits_to_keep=k + 1).logits.float()[0]
            pred = logits[:k]
        except (TypeError, RuntimeError):
            logits = model(input_ids=ids, attention_mask=attn).logits.float()[0]
            pred = logits[start - 1 : end - 1]
        tok = ids[0, start:end]
        logp = torch.log_softmax(pred, dim=-1)
        out.append(float(logp.gather(-1, tok[:, None]).squeeze(-1).sum().cpu()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max_obs_tokens", type=int, default=96)
    args = ap.parse_args()

    t0 = time.time()
    ds = load_dataset("parquet", data_files=args.parquet, split="train")
    n = min(args.n, len(ds))
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    plus_rows: list[list[int]] = []
    minus_rows: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    records = []
    for i in range(n):
        row = ds[i]
        prompt_obj = row["prompt"]
        if isinstance(prompt_obj, str):
            prompt = prompt_obj
        else:
            prompt = prompt_obj[0].get("content", "")
        extra = row.get("extra_info") or {}
        gt = (row.get("reward_model") or {}).get("ground_truth", "a0")
        # Keep the action/observation transition explicit and deterministic.
        action = "\n<request_grounding>locate evidence for the answer</request_grounding>\n"
        subtitle = str(extra.get("episode_sub_block") or "")
        obs = subtitle[:2400]
        obs_ids = _ids(tok, "<observation>" + obs + "</observation>\n")[: args.max_obs_tokens]
        if not obs_ids:
            obs_ids = _ids(tok, " observation")
        # Equal-length counterfactual; use a neutral repeated lexical token,
        # preserving all positions and the suffix boundary.
        mask_one = _ids(tok, " [MASK]") or [tok.eos_token_id]
        mask_ids = (mask_one * ((len(obs_ids) + len(mask_one) - 1) // len(mask_one)))[: len(obs_ids)]
        suffix = f"<answer>{gt}</answer>"
        pre = _ids(tok, prompt + action)
        suf = _ids(tok, suffix)
        p = pre + obs_ids + suf
        m = pre + mask_ids + suf
        plus_rows.append(p)
        minus_rows.append(m)
        spans.append((len(pre) + len(obs_ids), len(p)))
        records.append({"row": i, "obs_tokens": len(obs_ids), "suffix_tokens": len(suf), "gt": gt})

    # Batched scoring; rows can differ in prefix length, but each pair has the
    # same observation and suffix positions.
    plus = score_suffixes(model, plus_rows, spans, tok.pad_token_id, device)
    minus = score_suffixes(model, minus_rows, spans, tok.pad_token_id, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    deltas = []
    for rec, a, b in zip(records, plus, minus):
        rec.update({"logp_plus": a, "logp_mask": b, "obs_sensitivity": a - b})
        deltas.append(a - b)
    finite = [x for x in deltas if math.isfinite(x)]
    summary = {
        "mode": "single_model_observation_sensitivity_proxy",
        "warning": "Not a teacher-student OPD gap; supply separate teacher/student logits for e_obs.",
        "n": len(finite),
        "mean": sum(finite) / len(finite) if finite else None,
        "median": sorted(finite)[len(finite) // 2] if finite else None,
        "positive_fraction": sum(x > 0 for x in finite) / len(finite) if finite else None,
        "elapsed_sec": time.time() - t0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["mode", "n", "mean", "median", "positive_fraction", "elapsed_sec", "gpu"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
