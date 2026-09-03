#!/usr/bin/env python3
"""Score exact-prefix paired trajectories for offline OT-OPD replay.

For every acquisition turn with a following action, this script evaluates the
same next-action tokens under (i) the prefix before the observation and (ii)
the prefix including the observation.  It computes teacher-minus-student
log-probability gaps using literal variable-length prefixes and stores a
versioned JSON cache.  No video tool is re-run and no ground-truth clip is
consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTION_TYPES = {"request_grounding", "visual_query", "search", "seek", "query", "acquire"}


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def model_fingerprint(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    for name in ("config.json", "generation_config.json", "adapter_config.json", "tokenizer_config.json", "tokenizer.json"):
        f = p / name
        if f.is_file():
            h.update(name.encode())
            h.update(f.read_bytes())
    h.update(str(p.resolve()).encode())
    return h.hexdigest()[:20]


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def unpad(entry: dict, key: str, valid_key: str, side: str) -> list[int]:
    aliases = {
        "prompt_ids": "prompts_ids",
        "response_ids": "responses_ids",
        "response_mask": "response_mask_ids",
    }
    values = [int(x) for x in (entry.get(key) or entry.get(aliases.get(key, "")) or [])]
    if valid_key in entry:
        n = int(entry[valid_key])
    else:
        n = len(values)
    n = max(0, min(n, len(values)))
    return values[-n:] if side == "left" else values[:n]


def valid_response(entry: dict, response: list[int]) -> tuple[list[int], list[int]]:
    # Response mask is right-padded.  If absent, treat all valid response
    # positions as generated; the turn offsets still provide strict bounds.
    mask = entry.get("response_mask") or entry.get("response_mask_ids")
    if mask is None:
        return response, [1] * len(response)
    mask = [int(x) for x in mask]
    n = min(len(response), len(mask))
    return response[:n], mask[:n]


def extract_records(entry: dict, prompt: list[int], response: list[int], response_mask: list[int]):
    records = entry.get("turn_records") or []
    if isinstance(records, dict):
        records = records.get("turn_records") or []
    if not isinstance(records, list):
        return []
    out = []
    for rec in records:
        if not isinstance(rec, dict) or str(rec.get("action_type", "")).lower() not in ACTION_TYPES:
            continue
        try:
            o0, o1 = int(rec["observation_response_start"]), int(rec["observation_response_end"])
            t0, t1 = int(rec["next_action_response_start"]), int(rec["next_action_response_end"])
            a0, a1 = int(rec["assistant_response_start"]), int(rec["assistant_response_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= o0 <= o1 <= t0 < t1 <= len(response) and 0 <= a0 < a1 <= len(response)):
            continue
        if not all(response_mask[t0:t1]):
            continue
        if a1 > o0:  # source action must end before its observation starts
            continue
        out.append((rec, prompt + response[:o0], prompt + response[:o1], response[t0:t1], (a0, a1), (o0, o1), (t0, t1)))
    return out


def score_batch(model, contexts: list[list[int]], targets: list[list[int]], device: torch.device, pad_id: int) -> list[list[float]]:
    if not contexts:
        return []
    seqs = [ctx + tgt for ctx, tgt in zip(contexts, targets, strict=True)]
    max_len = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros_like(ids)
    ctx_lens = []
    tgt_lens = []
    for i, (ctx, tgt) in enumerate(zip(contexts, targets, strict=True)):
        seq = ctx + tgt
        off = max_len - len(seq)
        ids[i, off:] = torch.tensor(seq, dtype=torch.long, device=device)
        attn[i, off:] = 1
        ctx_lens.append(len(ctx))
        tgt_lens.append(len(tgt))
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
    result: list[list[float]] = []
    for i, (ctx_len, tgt_len) in enumerate(zip(ctx_lens, tgt_lens, strict=True)):
        off = max_len - (ctx_len + tgt_len)
        # logits at position p predict token p+1; target starts at ctx_len.
        positions = torch.arange(off + ctx_len - 1, off + ctx_len + tgt_len - 1, device=device)
        token_ids = ids[i, off + ctx_len : off + ctx_len + tgt_len]
        vals = logp[i, positions, token_ids].detach().cpu().tolist()
        result.append([float(v) for v in vals])
    return result


def tokenizer_signature(tok) -> str:
    sample = ["hello", "<reasoning>", "<search>x</search>", "你好"]
    payload = {"vocab_size": len(tok), "ids": [tok.encode(x, add_special_tokens=False) for x in sample]}
    return sha(payload)[:20]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rollout-jsonl",
        required=True,
        nargs="+",
        help="One or more rollout JSONL files; exact duplicate trajectories are retained under hash-suffixed keys.",
    )
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reduction", choices=("mean", "sum"), default="mean")
    args = ap.parse_args()

    rows = []
    for rollout_path in args.rollout_jsonl:
        rows.extend(load_rows(rollout_path))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    tok_s = AutoTokenizer.from_pretrained(args.student_model, use_fast=True)
    tok_t = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    if tokenizer_signature(tok_s) != tokenizer_signature(tok_t):
        raise RuntimeError("student/teacher tokenizer signatures differ; tokenwise OT is invalid")
    if tok_s.pad_token_id is None:
        tok_s.pad_token = tok_s.eos_token
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    student = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()

    pending = []
    metadata = []
    skipped = {}
    for entry in rows:
        prompt = unpad(entry, "prompt_ids", "prompt_valid_len", "left")
        response = unpad(entry, "response_ids", "response_valid_len", "right")
        response, rmask = valid_response(entry, response)
        if not prompt or not response:
            skipped["missing_ids"] = skipped.get("missing_ids", 0) + 1
            continue
        for rec, ctx_m, ctx_p, target, src_span, obs_span, tgt_span in extract_records(entry, prompt, response, rmask):
            pending.append((ctx_m, ctx_p, target))
            metadata.append((entry, rec, response, src_span, obs_span, tgt_span))

    records = {}
    for start in range(0, len(pending), max(1, args.batch_size)):
        chunk = pending[start : start + max(1, args.batch_size)]
        minus_ctx = [x[0] for x in chunk]
        plus_ctx = [x[1] for x in chunk]
        targets = [x[2] for x in chunk]
        s_minus = score_batch(student, minus_ctx, targets, device, tok_s.pad_token_id)
        s_plus = score_batch(student, plus_ctx, targets, device, tok_s.pad_token_id)
        t_minus = score_batch(teacher, minus_ctx, targets, device, tok_s.pad_token_id)
        t_plus = score_batch(teacher, plus_ctx, targets, device, tok_s.pad_token_id)
        for j in range(len(chunk)):
            entry, rec, response, src_span, obs_span, tgt_span = metadata[start + j]
            dplus_vec = [a - b for a, b in zip(t_plus[j], s_plus[j], strict=True)]
            dminus_vec = [a - b for a, b in zip(t_minus[j], s_minus[j], strict=True)]
            reduce_fn = (lambda x: sum(x) / max(len(x), 1)) if args.reduction == "mean" else sum
            dplus, dminus = float(reduce_fn(dplus_vec)), float(reduce_fn(dminus_vec))
            suid = str(entry.get("sample_uid") or rec.get("sample_uid") or entry.get("qid") or "")
            qid = str(entry.get("qid") or rec.get("qid") or "")
            rn = int(entry.get("rollout_n", rec.get("rollout_n", 0)) or 0)
            turn = int(rec.get("turn", 0) or 0)
            key = f"{suid}|{rn}|{turn}"
            obs_ids = response[obs_span[0] : obs_span[1]]
            target_ids = response[tgt_span[0] : tgt_span[1]]
            payload = {
                "sample_uid": suid,
                "qid": qid,
                "rollout_n": rn,
                "turn": turn,
                "action_type": rec.get("action_type"),
                "current_vid": rec.get("current_vid_before"),
                "obs_start": obs_span[0],
                "obs_end": obs_span[1],
                "target_start": tgt_span[0],
                "target_end": tgt_span[1],
                "D_plus": dplus,
                "D_minus": dminus,
                "e_obs": dplus - dminus,
                "D_plus_tokens": dplus_vec,
                "D_minus_tokens": dminus_vec,
                "teacher_plus_logp": t_plus[j],
                "teacher_minus_logp": t_minus[j],
                "student_plus_logp": s_plus[j],
                "student_minus_logp": s_minus[j],
                "response_sha256": sha(response),
                "obs_sha256": sha(obs_ids),
                "target_sha256": sha(target_ids),
                "observation_text": str(rec.get("observation_text") or ""),
                "tool_status": rec.get("tool_status"),
                "evidence_valid": rec.get("evidence_valid"),
                "cache_hit": rec.get("cache_hit", rec.get("tool_cache_hit")),
                "student_fingerprint": model_fingerprint(args.student_model),
                "teacher_fingerprint": model_fingerprint(args.teacher_model),
                "tokenizer_fingerprint": tokenizer_signature(tok_s),
                "intervention": "literal_exact_prefix",
                "reduction": args.reduction,
                "temperature": 1.0,
            }
            # Keep independent trajectories that happen to share the same
            # asynchronous rollout_n label.  The primary key remains the
            # strict identity used by the fast path; a hash suffix preserves
            # additional exact variants for the opt-in hash-index lookup.
            if key in records:
                prior = records[key]
                if prior.get("response_sha256") != payload.get("response_sha256"):
                    key = f"{key}|sha:{payload['response_sha256'][:16]}"
            records[key] = payload

    out = {
        "schema_version": "ot-opd-paired-v2",
        "intervention": "literal_exact_prefix",
        "reduction": args.reduction,
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "student_fingerprint": model_fingerprint(args.student_model),
        "teacher_fingerprint": model_fingerprint(args.teacher_model),
        "tokenizer_fingerprint": tokenizer_signature(tok_s),
        "count": len(records),
        "skipped": skipped,
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}: {len(records)} paired records; skipped={skipped}")


if __name__ == "__main__":
    main()
