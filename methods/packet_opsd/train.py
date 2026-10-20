"""On-policy Planner OPSD with a metadata-only packet MMD auxiliary.

One optimizer update consumes only newly generated student trajectories. All
candidate draws precede the update; no replay epochs or stale-policy scores.
Frozen Inspector/compactor calls occur only inside the original agent rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

from .objective import (INSPECTION_PREFIX, candidate_spans, hindsight_messages,
                        symmetric_packet_coefficients)
from .identities import IdentityMapping, bound_mapping, resolve_identity_mapping
from .completion_regions import CompletionRegionError, completion_region


def teacher_adapter_name(reference: str) -> str:
    """Use one reference consistently for candidate generation and token KL."""
    if reference not in {"ema", "fixed-base"}:
        raise ValueError("unknown teacher reference")
    return "teacher" if reference == "ema" else "base"


def configure_baseline(source: Path, *, inspector_endpoint: str, compactor_endpoint: str):
    """The captured BM25/Inspector contract; only the Planner is trained."""
    values = {
        "AGENT_LLM_BACKEND": "api", "AGENT_API_USE_MESSAGES": "0",
        "AGENT_LLM_API_BASE": "http://127.0.0.1:1/v1", "AGENT_LLM_API_KEY": "local",
        "AGENT_LLM_MODEL": "Qwen3-8B", "AGENT_MLLM_BACKEND": "openai",
        "AGENT_LLM_MAX_TOKENS": "4096", "AGENT_LLM_CONTEXT_LIMIT": "40960",
        "AGENT_LLM_TEMPERATURE": "0.0", "AGENT_LLM_TIMEOUT": "600",
        "AGENT_LEGACY_HISTORY_MAX_BLOCKS": "0", "AGENT_LEGACY_HISTORY_MAX_CHARS": "0",
        "VIDEOSEAL_VISUAL_RETRIEVE_CLASS": "videoseal.tools.visual_tools:VisualRetrieveAliasTool",
        "VIDEOSEAL_DISABLE_TIME_REFERENCE": "1", "VIDEOSEAL_STRICT_FINAL_GATE": "1",
        "MLLM_BACKEND": "openai", "MLLM_MAX_TOKENS": "4096", "MLLM_TIMEOUT": "300",
        "MLLM_RETRY_TIMES": "1", "MLLM_RETRY_DELAY": "0",
        "VISUAL_INSPECT_BACKEND": "openai", "VISUAL_INSPECT_API_BASE": inspector_endpoint,
        "VISUAL_INSPECT_API_KEY": "local", "VISUAL_INSPECT_MODEL": "Qwen2.5-VL-7B-Instruct",
        "VISUAL_INSPECT_EXCLUSIVE_OUTPUT": "1", "VISUAL_INSPECT_PRESERVE_UNCERTAINTY": "0",
        "VISUAL_RETRIEVE_SUM_BACKEND": "openai",
        "VISUAL_RETRIEVE_SUM_API_BASE": compactor_endpoint,
        "VISUAL_RETRIEVE_SUM_API_KEY": "local", "VISUAL_RETRIEVE_SUM_MODEL": "Qwen3-8B",
        "VISUAL_RETRIEVE_SUM_MAX_TOKENS": "2048", "VISUAL_RETRIEVE_SUM_TEMPERATURE": "0.1",
        "VISUAL_RETRIEVE_SUMMARY_PROMPT_FILE": str(source / "research/prompts/fixed30_lossless_compactor_v1.txt"),
        "SEMANTIC_RETRIEVE_MIX": "bm25", "SEMANTIC_RETRIEVE_TOPK": "40",
        "VISUAL_RETRIEVE_TOPK": "30", "RETRIEVE_SUMMARY_ENABLED": "1",
        "RETRIEVE_SUMMARY_MAX_SPANS": "100", "VISUAL_RETRIEVE_RETURN_SPANS": "1",
        "VISUAL_RETRIEVE_LOG_SPANS": "1", "RETRIEVAL_PRIORITY_CHANNEL": "0", "INSPECT_PENDING_SLOT": "0",
        "INSPECT_GLOBAL_ORDER": "1", "INSPECT_MAX_TOTAL_IMAGES": "64",
        "VISUAL_INSPECT_MAX_TOTAL_IMAGES": "64", "INSPECT_FPS": "2",
        "INSPECT_MAX_LONG_EDGE": "1280", "VISUAL_INSPECT_MAX_LONG_EDGE": "1280",
        "VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE": "1", "VISUAL_INSPECT_TOTAL_PIXELS": "20971520",
        "VISUAL_INSPECT_MIN_PIXELS": "12544", "VISUAL_INSPECT_EDGE_MULTIPLE": "32",
        "INSPECT_VLM_MAX_TOKENS": "4096", "INSPECT_VLM_TEMPERATURE": "0.1",
        "VISUAL_INSPECT_RETURN_FRAME_METADATA": "1", "MAX_STEPS": "16",
        "AGENT_FORCE_LAST_STEP_VISUAL_INSPECT": "1", "AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK": "1",
        "AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK": "1",
        "AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE": "mcq", "AGENT_PARSE_FAILURE_FALLBACK_TO_C": "0",
        "CLEAN_FRAMES": "0", "CLEAN_FRAMES_ASYNC": "1",
    }
    for name in ("VISUAL_INSPECT_FRAME_SELECTION", "INSPECT_FRAME_SELECTION", "INSPECT_VLM_SYSTEM_PROMPT",
                 "INSPECT_VLM_SYSTEM_PROMPT_FILE", "VISUAL_INSPECT_PROMPT", "VISUAL_INSPECT_PROMPT_FILE",
                 "VISUAL_INSPECT_SYSTEM_PROMPT_FILE", "AGENT_PLANNER_NO_THINK"):
        os.environ.pop(name, None)
    os.environ.update(values)
    return values


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tasks(path: Path, heldout: Path | IdentityMapping, *, allow_shared_videos: bool = False) -> list[dict]:
    """Read an answer-free task manifest, rejecting gold fields and overlap."""
    held = bound_mapping(heldout).rows
    held_uids = {str(r[key]) for r in held for key in ("sample_uid", "qid") if r.get(key)}
    held_questions = {r["question_sha256"] for r in held if r.get("question_sha256")}
    held_videos = {str(r[key]) for r in held for key in ("original_video_id", "video_id") if r.get(key)}
    allowed = {"uid", "video_id", "question", "video_path", "visual_index", "subtitle_path",
               "summary_dir", "summary_file", "benchmark", "video_duration_sec"}
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if set(task) - allowed:
            raise ValueError(f"task manifest has unexpected fields: {sorted(set(task) - allowed)}")
        if not all(task.get(k) for k in ("uid", "video_id", "question", "video_path", "visual_index")):
            raise ValueError("incomplete training task")
        uid = str(task["uid"])
        if uid in held_uids or uid.rsplit(":", 1)[-1] in held_uids:
            raise ValueError("training UID overlaps the fixed300 evaluation")
        if hashlib.sha256(str(task["question"]).encode()).hexdigest() in held_questions:
            raise ValueError("training question overlaps the fixed300 evaluation")
        video_keys = {str(task["video_id"]), Path(task["video_path"]).stem,
                      Path(task["video_path"]).resolve().stem}
        if not allow_shared_videos and video_keys & held_videos:
            raise ValueError("training video overlaps the fixed300 evaluation")
        if not Path(task["video_path"]).is_file() or not Path(task["visual_index"]).exists():
            raise FileNotFoundError("training video/index is unavailable")
        result.append(task)
    if not result or len({str(t["uid"]) for t in result}) != len(result):
        raise ValueError("empty or duplicate-UID task manifest")
    return result


def media_spec(task):
    import cv2
    from .packets import MediaSpec
    cap = cv2.VideoCapture(str(task["video_path"]))
    try:
        if not cap.isOpened():
            raise ValueError("cannot open training media")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0 or min(width, height) <= 0:
            raise ValueError("complete positive FPS/frame-count/size metadata required")
        duration = count / fps
        # The original reader gives VIDEO_DURATION_SEC precedence over probe
        # duration. rollout.py sets that variable from this optional task field.
        try:
            override = float(task.get("video_duration_sec"))
        except (ValueError, TypeError):
            override = 0.0
        if override > 0:
            duration = override
        return MediaSpec(media_id=str(task["video_id"]), duration_s=duration,
                         source_fps=fps, width=width, height=height, frame_count=count)
    finally:
        cap.release()


def append(path: Path, row: dict):
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def factual_record_skip_reason(record: dict, media, sampler) -> str | None:
    """Pure, shared eligibility gate for prefix collection and training.

    Never write metrics or inspect candidate draws here: the prefix hook may
    check the same already-observed record before several Planner turns.
    """
    from .packets import build_action_packet, validate_factual_packet
    completion, observation = record["completion"], record["observation"]
    if (not completion.token_ids or not completion.complete
            or not observation.get("ok")):
        return "ineligible_factual_inspection"
    predicted = build_action_packet(media, record["action"]["arguments"].get("spans"), sampler)
    metadata = observation.get("metadata") or {}
    parity = validate_factual_packet(predicted, metadata.get("timestamps") or [])
    predicted_pixels = sum(f.final_width * f.final_height for f in predicted.frames)
    if not parity.accepted or metadata.get("actual_pixels") != predicted_pixels:
        return "packet_parity"
    return None


def capture_gradients(parameters):
    return [None if p.grad is None else p.grad.detach().clone() for p in parameters]


def gradient_increment(parameters, before):
    """Measure the contribution that survived accumulation in parameter dtype.

    A nonzero Python coefficient can still round away against existing FP32
    gradients. Measure actual accumulated differences, rather than claiming
    the total (usually token-KL-dominated) norm proves the packet term worked.
    """
    import torch
    with torch.no_grad():
        base, differences = [], []
        for parameter, old in zip(parameters, before):
            if old is not None:
                base.append(old.double().square().sum())
            if parameter.grad is not None:
                difference = parameter.grad.detach().float() if old is None else parameter.grad.detach().float() - old.float()
                differences.append(difference.double().square().sum())
        norm = float(torch.stack(differences).sum().sqrt()) if differences else 0.0
        base_norm = float(torch.stack(base).sum().sqrt()) if base else 0.0
    if not math.isfinite(norm) or not math.isfinite(base_norm):
        raise RuntimeError("nonfinite packet gradient diagnostics")
    return {"packet_gradient_norm": norm, "actual_accumulated_packet_delta_norm": norm,
            "gradient_before_packet_norm": base_norm,
            "packet_gradient_ratio": norm / base_norm if base_norm else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--identity-mapping", type=Path,
                        help="answer-free fixed300 mapping; defaults to the archived public mapping")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspector-endpoint", required=True)
    parser.add_argument("--compactor-endpoint", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--records-per-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=40960)
    parser.add_argument("--rollout-tokens", type=int, default=4096)
    parser.add_argument("--candidate-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--packet-weight", type=float, default=0.05)
    parser.add_argument("--packet-gradient-max-ratio", type=float, default=0.5)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--teacher-reference", choices=("ema", "fixed-base"), default="ema",
                        help="EMA adapter or frozen base with all adapters disabled; neither uses Gold")
    parser.add_argument("--kl-region", choices=("full", "action", "arguments"), default="full",
                        help="experimental original-token mask; all choices retain the full-N divisor")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--audit-candidate-probabilities", action="store_true")
    parser.add_argument("--collect-training-prefix", action="store_true",
                        help="stop training after enough eligible inspections, before the next Planner turn")
    overlap = parser.add_mutually_exclusive_group()
    overlap.add_argument("--allow-shared-videos-canary", action="store_true")
    overlap.add_argument("--training-split-policy", choices=("video-disjoint", "question-disjoint"),
                         default="video-disjoint")
    args = parser.parse_args(argv)
    teacher_adapter = teacher_adapter_name(args.teacher_reference)
    if (min(args.steps, args.max_attempts, args.records_per_step, args.candidate_tokens) < 1
            or args.temperature <= 0 or args.packet_weight < 0 or args.learning_rate <= 0
            or args.bandwidth <= 0 or not 0 <= args.ema_decay < 1
            or not math.isfinite(args.packet_gradient_max_ratio) or args.packet_gradient_max_ratio < 0
            or args.save_every < 1):
        parser.error("invalid training configuration")
    repo = Path(__file__).resolve().parents[2]
    mapping = resolve_identity_mapping(args.identity_mapping, repo=repo)
    args.identity_mapping = mapping.path
    source = repo / "baselines/videoseal_6000d/source"
    import sys
    sys.path[:0] = [str(source / "baselines/VideoSEAL"), str(source)]
    tasks = load_tasks(args.tasks, mapping,
                       allow_shared_videos=(args.allow_shared_videos_canary
                                            or args.training_split_policy == "question-disjoint"))
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_env = configure_baseline(source, inspector_endpoint=args.inspector_endpoint,
                                       compactor_endpoint=args.compactor_endpoint)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(mapping.metadata())
    config.update({"task_manifest_sha256": sha(args.tasks), "baseline_commit": "7676bac4fa06a3100e73cc728f53b6582cb97006",
                   "sampler_contract": "planned-frame-packet-approximation-with-factual-parity",
                   "training_gold_used": False, "candidate_tool_calls": 0,
                   "candidate_draws_per_record": {"student": 2, "teacher": 1},
                   "packet_estimator": "symmetric-score-function",
                   "token_kl_direction": "KL(student || stop-gradient teacher)",
                   "token_kl_denominator": "full-original-completion-length",
                   "teacher_information": "unchanged-natural-inspection-feedback-HF",
                   "teacher_adapter": teacher_adapter,
                   "teacher_saved_adapter_usage": ("active-EMA" if args.teacher_reference == "ema"
                                                   else "unused-in-fixed-base-mode"),
                   "kl_region_status": "experimental-ablation-not-validated-repair",
                   "rollout_collection": ("first-eligible-training-prefix" if args.collect_training_prefix
                                          else "complete-original-trajectory"),
                   "gradient_combination": "token-KL plus relative-norm-capped packet auxiliary",
                   "generalization_scope": ("new questions on potentially seen videos"
                                            if args.training_split_policy == "question-disjoint"
                                            or args.allow_shared_videos_canary
                                            else "held-out questions and videos"),
                   "baseline_environment": baseline_env})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    import torch
    from .packets import SamplerSpec, build_action_packet, gaussian_packet_kernel
    from .policy import ContextOverflow, HFPolicy
    from .gradients import bounded_packet_gradients
    from .rollout import run_task
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = HFPolicy.load(args.model, rank=args.rank, max_context=args.max_context)
    policy.audit_generation_scores = False
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.0)
    sampler = SamplerSpec()
    updates, attempts, total_records, nonzero_packet, packet_gradient_updates = 0, 0, 0, 0, 0
    task_order = list(tasks)
    random.shuffle(task_order)
    try:
        while updates < args.steps and attempts < args.max_attempts:
            task = task_order[attempts % len(task_order)]
            attempts += 1
            uid_hash = hashlib.sha256(str(task["uid"]).encode()).hexdigest()
            try:
                media = media_spec(task)
                rollout = run_task(task, policy, save_dir=args.output / "trajectories" / f"attempt-{attempts}",
                                   max_steps=args.max_steps, temperature=args.temperature,
                                   max_new_tokens=args.rollout_tokens,
                                   max_factual_records=(args.records_per_step if args.collect_training_prefix else None),
                                   record_filter=(lambda record: factual_record_skip_reason(record, media, sampler) is None)
                                   if args.collect_training_prefix else None)
                records = []
                for record in rollout.factual_records:
                    reason = factual_record_skip_reason(record, media, sampler)
                    if reason is not None:
                        if reason == "packet_parity":
                            append(args.output / "metrics.jsonl", {"event": "skip_packet_parity", "attempt": attempts,
                                                                   "uid_hash": uid_hash})
                        continue
                    try:
                        selection = completion_region(policy.tokenizer, record["completion"],
                            region=args.kl_region, eos_token_ids=policy._eos_token_ids())
                    except CompletionRegionError as exc:
                        append(args.output / "metrics.jsonl", {"event": "skip_kl_region", "attempt": attempts,
                            "uid_hash": uid_hash, "step_index": record["step_index"],
                            "kl_region": args.kl_region, "reason": str(exc)})
                        continue
                    # Keep training metadata outside the unchanged agent record.
                    selected_record = dict(record)
                    selected_record["_kl_selection"] = selection
                    records.append(selected_record)
                    if len(records) >= args.records_per_step:
                        break
                if not records:
                    append(args.output / "metrics.jsonl", {"event": "skip_no_eligible_inspection", "attempt": attempts,
                                                           "uid_hash": uid_hash})
                    continue
                # Fresh independent draws after freezing H/F; never reuse the
                # factual inspection action as an independent MMD sample.
                groups = []
                for record in records:
                    teacher_messages = hindsight_messages(record["messages"], record["observation"])
                    policy.audit_generation_scores = args.audit_candidate_probabilities
                    try:
                        a = policy.sample(record["messages"], prefix=INSPECTION_PREFIX,
                                          max_new_tokens=args.candidate_tokens, temperature=args.temperature)
                        ap = policy.sample(record["messages"], prefix=INSPECTION_PREFIX,
                                           max_new_tokens=args.candidate_tokens, temperature=args.temperature)
                        b = policy.sample(teacher_messages, adapter=teacher_adapter, prefix=INSPECTION_PREFIX,
                                          max_new_tokens=args.candidate_tokens, temperature=args.temperature)
                    finally:
                        policy.audit_generation_scores = False
                    append(args.output / "candidate_audit.private.jsonl", {
                        "attempt": attempts, "uid_hash": uid_hash, "step_index": record["step_index"],
                        "teacher_reference": args.teacher_reference,
                        "completions": [{"role": role, "text": c.text, "complete": c.complete,
                                         "token_count": len(c.token_ids),
                                         "stop_reason": getattr(c, "stop_reason", None),
                                         "generation_log_probability": getattr(c, "generation_log_probability", None)}
                                        for role, c in zip(("student", "independent_student", "teacher"), (a, ap, b))]})
                    packets = [build_action_packet(media, candidate_spans(c), sampler) for c in (a, ap, b)]
                    coefficients = symmetric_packet_coefficients(*packets, kernel=lambda x, y: gaussian_packet_kernel(x, y, bandwidth=args.bandwidth))
                    groups.append((record, teacher_messages, (a, ap, b), packets, coefficients))
                optimizer.zero_grad(set_to_none=True)
                values = []
                for record, teacher_messages, candidates, packets, coefficients in groups:
                    selection = record["_kl_selection"]
                    kl = policy.hindsight_kl(record["completion"], teacher_messages,
                        temperature=args.temperature, teacher_adapter=teacher_adapter, token_mask=selection.mask)
                    kl_value = float(kl.detach())
                    (kl / len(groups)).backward()
                    values.append({"reverse_kl": kl_value, "kl_region": selection.region,
                        "kl_full_tokens": selection.full_tokens, "kl_selected_tokens": selection.selected_tokens,
                        "kl_boundary_tokens_excluded": selection.boundary_tokens_excluded,
                        "teacher_reference": args.teacher_reference})
                # Compute the two objectives separately. Never infer the packet
                # gradient from a difference against a much larger accumulated
                # gradient, and do not change sequence scores to token means.
                before_packet = capture_gradients(policy.parameters())
                optimizer.zero_grad(set_to_none=True)
                for value_record, (record, teacher_messages, candidates, packets, coefficients) in zip(values, groups):
                    score_values = []
                    score_differences = []
                    applied_terms = []
                    for candidate, value in zip(candidates, (coefficients.first_value, coefficients.second_value)):
                        applied = bool(args.packet_weight and value != 0)
                        applied_terms.append(applied)
                        if applied or args.audit_candidate_probabilities:
                            with torch.set_grad_enabled(applied):
                                score = policy.sequence_log_prob(candidate, temperature=args.temperature)
                            score_values.append(float(score.detach()))
                            sampled_score = getattr(candidate, "generation_log_probability", None)
                            score_differences.append(None if sampled_score is None else float(score.detach()) - sampled_score)
                            if applied:
                                (args.packet_weight * value * score / len(groups)).backward()
                        else:
                            score_values.append(None)
                            score_differences.append(None)
                    value_record.update({"packet_coefficients": [coefficients.first_value, coefficients.second_value],
                                   "student_kernel": coefficients.student_similarity,
                                   "teacher_kernels": [coefficients.first_teacher_similarity, coefficients.second_teacher_similarity],
                                   "score_log_probabilities": score_values,
                                   "packet_score_terms_applied": applied_terms,
                                   "generation_log_probabilities": [getattr(c, "generation_log_probability", None) for c in candidates],
                                   "scoring_minus_generation_log_probabilities": score_differences,
                                   "candidate_complete": [c.complete for c in candidates],
                                   "candidate_stop_reasons": [getattr(c, "stop_reason", None) for c in candidates],
                                   "candidate_token_counts": [len(c.token_ids) for c in candidates],
                                   "candidate_valid": [p.valid for p in packets],
                                   "candidate_frame_counts": [len(p.frames) for p in packets]})
                increment = bounded_packet_gradients(policy.parameters(), before_packet,
                                                     max_ratio=args.packet_gradient_max_ratio)
                del before_packet
                increment["effective_packet_weight"] = args.packet_weight * increment["packet_gradient_scale"]
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
                if not math.isfinite(float(norm)):
                    raise RuntimeError("no finite nonzero student gradient")
                if float(norm) == 0:
                    append(args.output / "metrics.jsonl", {"event": "skip_zero_gradient", "attempt": attempts,
                                                           "uid_hash": uid_hash, "records": values, **increment})
                    continue
                optimizer.step()
                if args.teacher_reference == "ema":
                    policy.update_teacher(args.ema_decay)
                updates += 1
                total_records += len(groups)
                nonzero_packet += sum(any(v["packet_score_terms_applied"]) for v in values)
                packet_gradient_updates += int(increment["packet_gradient_norm"] > 0)
                row = {"event": "update", "step": updates, "attempt": attempts, "uid_hash": uid_hash,
                       "gradient_norm": float(norm), "records": values,
                       "candidate_tool_calls": 0, "teacher_reference": args.teacher_reference,
                       "kl_region": args.kl_region, "ema_updated": args.teacher_reference == "ema", **increment}
                append(args.output / "metrics.jsonl", row)
                print(json.dumps(row, allow_nan=False), flush=True)
                if updates % args.save_every == 0:
                    policy.save(args.output / f"step-{updates}", optimizer=optimizer, step=updates, config=config)
            except ContextOverflow as exc:
                optimizer.zero_grad(set_to_none=True)
                append(args.output / "metrics.jsonl", {"event": "skip_context_overflow", "attempt": attempts,
                                                       "uid_hash": uid_hash, "reason": str(exc)})
        if updates != args.steps:
            raise RuntimeError(f"only {updates}/{args.steps} updates in {attempts} on-policy attempts")
        if updates % args.save_every:
            policy.save(args.output / f"step-{updates}", optimizer=optimizer, step=updates, config=config)
        result = {"state": "COMPLETED", "updates": updates, "attempts": attempts,
                  "eligible_records": total_records, "nonzero_packet_updates": nonzero_packet,
                  "packet_gradient_updates": packet_gradient_updates,
                  "candidate_tool_calls": 0, "accuracy_evaluated": False,
                  "sampler_contract": config["sampler_contract"]}
        (args.output / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        return 0
    except Exception as exc:
        (args.output / "TRAIN_RESULT.json").write_text(json.dumps({"state": "FAILED", "updates": updates,
              "attempts": attempts, "error_type": type(exc).__name__, "error": str(exc)[:1000]}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
