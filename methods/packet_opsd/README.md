# Packet OPSD on the 6000D VideoSEAL baseline

Experimental training implementation on baseline `7676bac` (archived 51.33%
search / 53.00% after conditional Observer). Those scores belong to the
untrained baseline. No accuracy gain is implied by implementing this method.

## Learning contract

The current Qwen3-8B student generates fresh trajectories through the captured
BM25 / caption compactor / Inspector agent. The Inspector still answers and
terminates directly. No old 3B action language, eta scorer, fixed slot menu,
new evidence annotation, or counterfactual Inspector call is introduced.

For each naturally proposed, non-forced inspection, retain its original
pre-action history H, sampled completion, and natural next feedback F. A
stop-gradient EMA of the **same** student LoRA, sharing its frozen base, sees
H plus F. Dense reverse token KL on the factual completion prefixes supplies
ordinary hindsight OPSD. The teacher may see the Inspector's generated answer;
it never sees benchmark gold or any unobserved video image.

The auxiliary compares the distribution of complete frame packets after the
existing shared-budget packing operation S. Hold the original inspection-tool
prefix fixed, then sample its free JSON arguments twice from the current
student and once from the hindsight teacher. This is a **conditional argument
distribution**, not the full Planner distribution over rationales/tool types.

For independent a,a' ~ p_theta(.|H, inspection prefix), b ~ q_EMA(.|H,F,prefix):

```
L_packet = MMD_k²(S#p, S#q)
c_a  = k(S(a), S(a')) - k(S(a), S(b))
c_a' = k(S(a), S(a')) - k(S(a'), S(b))
g_packet = stop_gradient(c_a)  * grad_theta log p_theta(a  | H, prefix)
         + stop_gradient(c_a') * grad_theta log p_theta(a' | H, prefix)
```

`objective.py` implements these coefficients; `policy.py` scores the exact sampled
token IDs using the same positive temperature and no top-k/top-p truncation.
The score is a **sequence sum**, not a token average. The coefficients/surrogate
can be negative and is not a logged nonnegative MMD estimate. Conditional on
fixed H/F, this is the MMD score-function gradient. It is not the gradient of
the distribution that generated H/F. The factual action that produced F cannot
serve as an independent MMD draw and is never reused as one. This symmetric
estimator averages the two possible orderings of the student pair. It has the
same expectation as `2*(kpp-kpq)*score(a)` and cannot increase covariance when
second moments exist. It requires one extra sequence scoring/backward, with
the same two student / one teacher generations and unchanged tool budget.

All candidate draws precede the single optimizer update. There are no replay
epochs or stale-policy samples. Invalid or truncated argument draws map to one
explicit invalid symbol; they are not rejection-resampled. Turning
`--packet-weight 0` leaves the same natural rollout, teacher information, and
hindsight token objective (the current implementation still records the three
candidate draws for a matched compute comparison).

## What the packet identity does and does not establish

The original sampler normalizes windows to at least 15 seconds, takes an
initial 2-fps prefix (at most 64 per span), then allocates the **shared** 64-frame
budget round-robin, globally sorts by timestamp, and dynamically resizes. The
module reproduces these rules, including repeated entries, timestamp rounding,
and shared JPEG filename overwrites. Its Gaussian kernel acts on the ordered
whole packet, count, and final image dimensions, rather than frame marginals.

**The default is a planned-frame-packet approximation.** Original timestamps
are requested labels, not decoded PTS. OpenCV seeking, preheat, read-after-index
semantics, decode failures, rotation, and JPEG bytes cannot be reconstructed
exactly from FPS and duration alone. Calling target indices actual frame IDs
would be incorrect. `packets.py` keeps those identities separate.

The trainer requires the factual planned labels/count and total post-resize
pixels to match the original Inspector metadata, otherwise it rejects that
state. This does not certify unexecuted candidate pixels. A strict identity
mode is available only with complete backend-bound per-span decoder traces;
it refuses missing/uncovered candidate traces and never silently falls back.
No extra candidate video decoding is performed to obtain such traces.

The kernel does not represent optional context/span-label language, subtitle
rendering, JPEG bytes, or the server's vision processor. Hence this is not an
exact full Inspector-observation kernel or a guarantee of equal answers for
equal metadata packets. Metadata cannot tell which unseen event a frame
contains, distinguish causal localization error from missed evidence, or prove
that an accepted Inspector answer is correct.

## Run

Use the baseline's existing runtime and licensed model/media/index assets.
Observed dependencies: Torch 2.8.0+cu128, Transformers 4.57.1, PEFT 0.18.1,
Accelerate 1.13.0, PyArrow 23.0.1, NumPy, OpenCV, Pillow, requests,
rank-bm25, and ordinary VideoSEAL dependencies. Use the original vLLM 0.11.0
servers for the frozen Inspector and compactor; this is not a complete lock.

The task manifest is private answer-free JSONL with these keys:

```
uid, video_id, question, video_path, visual_index
```

Optional: `benchmark`, `video_duration_sec`, `subtitle_path`, `summary_dir`,
`summary_file`. Unexpected fields (including gold and reference time) are
rejected. Paths must point to available licensed assets. The default rejects
fixed300 UID aliases, exact question hashes, and overlapping video identities.
The explicit `--allow-shared-videos-canary` switch relaxes only video overlap,
never held-out question/UID overlap; do not use it for a generalization claim.

```
CUDA_VISIBLE_DEVICES=0 /path/to/runtime/python -m methods.packet_opsd.train \
  --model /path/to/Qwen3-8B \
  --tasks /path/to/answer-free-video-disjoint-train.jsonl \
  --output /path/to/new-run \
  --inspector-endpoint http://127.0.0.1:19741/v1 \
  --compactor-endpoint http://127.0.0.1:19742/v1 \
  --steps 10 --max-attempts 30 --max-steps 16
```

Use a new output directory. Defaults: LoRA rank8 on Q/V projections, alpha16,
dropout0, AdamW lr1e-5, clip1, EMA .99, training temperature .7,
packet weight .05, bandwidth1, one eligible inspection per update, 4096 factual
tokens / 384 tokens per argument candidate. Histories are never truncated;
context overflow is recorded and skipped. Evaluation remains greedy with the
baseline budget; stochastic training is an explicit change, not a claim that
baseline inference used .7.

`metrics.jsonl` records fresh attempts, factual-parity rejection, three candidate
validities/frame counts, kernel coefficients, score probabilities, gradient norm,
and EMA updates. It separately measures the **actual accumulated packet gradient
increment before clipping**, its norm and its ratio to the gradient present
before the packet terms. A nonzero scalar coefficient alone does not certify
a contribution that survives floating-point accumulation. The two packet terms
can also cancel. Private candidate text/completion lengths are retained in
`candidate_audit.private.jsonl`; never commit it or the trajectories.
`step-N/` contains student/teacher adapters, tokenizer and
optimizer/RNG state. There is no automatic resume interface; do not imply that
loading an adapter alone resumes an optimizer. Private trajectories/questions
and weights must not be committed to this public repository.

For GPU allocation, use the host's existing reservation handoff. This module
does not kill processes, start model servers, release reservations, or write
keeper terminal status. The run supervisor must clean its own CUDA children
before returning the allocation.

For a trained adapter, prepare the same answer-free manifest for the fixed300
identities and run greedy search with the original tool servers:

```
CUDA_VISIBLE_DEVICES=0 /path/to/runtime/python -m methods.packet_opsd.evaluate \
  --model /path/to/Qwen3-8B --adapter /path/to/training-run/step-10 \
  --tasks /path/to/answer-free-fixed300.jsonl --output /path/to/new-evaluation \
  --inspector-endpoint http://127.0.0.1:19741/v1 \
  --compactor-endpoint http://127.0.0.1:19742/v1
```

This validates all 300 public identities/question hashes, normalizes UID/video
aliases, and retains failed rows in `results.jsonl`. It does not load gold or
calculate accuracy. The baseline's unchanged `run_conditional_observer.py` can
then consume that search directory. `--smoke-limit N` produces a separate
`SMOKE_RESULTS.jsonl`, which deliberately cannot be mistaken for the complete
search input expected by the fixed300 Observer. Neither a partial check nor
the engineering training canary is a 300-question evaluation.

## Validation and comparisons

```
python baselines/videoseal_6000d/scripts/verify_source.py
python -m unittest discover -s tests/packet_opsd -p 'test_*.py' -v
```

Tests compare packing against the captured functions, preserve duplicate/order
and filename overwrite behavior, distinguish same-marginal packet laws, check
the MMD gradient by exact enumeration and finite differences, and exercise the
actual original agent loop including terminal commit / forced exclusion. Tiny
random Qwen3/LoRA tests (no downloads) verify scoring against full logits and
generation probabilities, student-only gradients/updates, and the EMA. These
five model tests skip when Torch/Transformers/PEFT are unavailable; run them in
the actual training environment before a GPU job.

This implements a testable execution-aware OPSD prototype. Its mathematical
ingredients are standard. Novelty relative to HERO, OCSD, VEI and ActionOPD,
the relevance of shared-budget coupling, and any accuracy/cost benefit still
require matched experiments. The token objective, teacher information, tool
budget, and extra LM sampling should be held fixed across those comparisons.
