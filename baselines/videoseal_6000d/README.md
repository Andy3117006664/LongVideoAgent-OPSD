# 6000D BM25 + conditional DirectObserver baseline

This is the inference baseline completed on 17–18 October 2026 and the starting
point for subsequent VideoSEAL-based work in this repository.

| Fixed LVBench subset | Correct / total | Accuracy |
|---|---:|---:|
| Original search, single Planner-query BM25 | 154 / 300 | 51.33% |
| After the original conditional DirectObserver | 159 / 300 | 53.00% |

The Observer ran on 91 questions and its original parser supplied a valid A–D
prediction for all 91. Thirteen answers improved and eight worsened: net +5
questions, +1.67 percentage points. All 300 questions remain in the denominator.
These are archived inference results, not an OPSD or training gain.

## What is included

- `source/`: 90 usable files from the actual terminal source capture, six
  additional import dependencies, and the actual BM25 launcher. The source
  manifest records the original and published SHA256 for all 97 files.
- `scripts/run_search.sh`: supplies portable paths to the captured launcher.
- `scripts/run_conditional_observer.py`: invokes the captured packet selection,
  original Transformers Observer, and original adoption rule. It does not manage
  model servers or stop other processes.
- `selection/`: the fixed UID order and public identity/question hashes, without
  questions, answers, or machine paths.
- `results/`: aggregate results and the 300 machine predictions, without gold
  answers or model prompts.
- `configs/baseline.json`: method settings and observed runtime versions.

Ninety-two source files are byte-identical to their captured originals. Five have
deployment-path strings replaced by placeholders; model logic, prompts, numeric
settings, and parsing are unchanged. Two AppleDouble files and unrelated methods
from the supplemental archive are omitted. VideoSEAL's MIT license is retained
in `source/baselines/VideoSEAL/LICENSE`.

The portable wrappers were added for publication and checked locally. **They have
not been run through a new GPU evaluation.** The reported 53% belongs to the
captured experiment, whose source identities are recorded in `provenance/`.

## Method contract

This run disables the five-signal Fixed30 fusion tool, not the 30-candidate limit.
It uses stock `VisualRetrieveAliasTool`: one Planner query, BM25 positive scores,
TOPK30, and the original 15-second diversity rule. The original Qwen3-8B
lossless caption compactor remains enabled. No Embedding-4B retrieval is used.

Planner and compactor use Qwen3-8B; Inspector and DirectObserver use
Qwen2.5-VL-7B-Instruct. There is no training or task LoRA in this run.

- Planner: 4096 output tokens, temperature 0, context 40960, 600-second timeout.
- Compactor: 2048 tokens, temperature 0.1, original prompt unchanged.
- Search: 16 steps, 2400-second task timeout, no benchmark `time_reference`.
- Inspector: original prefix/round-robin sampling, global time order, 2 fps,
  at most 64 frames per call, long edge 1280 and original dynamic pixel budget.
  Output limit 4096, temperature 0.1.
- Original uncertainty normalization and MCQ/full-video fallback are retained.
  This is **not** the strict refusal-preserving, cache-first paper controller.
- Observer: after search, only questions that actually entered forced full-video
  fallback and have a usable non-forced inspection packet are selected.
  The original selection prefers the last non-forced decisive inspection,
  otherwise the last non-forced inspection.
- Observer decoding: original Transformers implementation, timestamped images,
  2 fps / 64 frames, greedy generation with at most two new tokens. The original
  `raw.upper()[:1]` parser is retained. A valid A–D prediction is adopted only
  for the forced group; otherwise the original prediction remains.

This baseline differs from the earlier local-4B/paper-controller proposal and
from the five-way retrieval run on H20. Hardware and serving engines also differ
across hosts; their score difference cannot isolate a retrieval effect.

## External assets

Supply licensed LVBench media, the original per-video `semantic_captions.json`
BM25 indexes, a public answer-free parquet with the same 300 rows in the same
order, and the two original model assets. These files, model weights, installed
environments, and raw HTTP/model logs are not uploaded here.

The captured public parquet SHA256 is
`3828504a9e9630c4bc2d998ad4dc33e96ee77f438f6c70aba8fa340cdb7fb4c1`.
Path rewriting may change parquet bytes; use `selection/identity_mapping.json`
to preserve UID, video identity, order, and question hashes. Absolute media/index
paths must resolve on the evaluation host. Gold stays in a separate offline
scoring process and must not be included in model requests.

The observed 6000D environment used Torch 2.8.0 / CUDA 12.8, vLLM 0.11.0 V1,
Transformers 4.57.1, and PyArrow 23.0.1. This is an environment record, not a
complete environment lock or a claim that installing four packages is sufficient.
The source also imports NumPy, Pillow, OpenAI's client, rank-bm25, video decoders,
and their ordinary dependencies. The historical installed environment was not
packaged. Original model revision identities have not been reverified in this
publication; reuse the captured assets when comparing exact results.

## Run the two stages

First verify source integrity without loading models:

```bash
python baselines/videoseal_6000d/scripts/verify_source.py
```

Start your own compatible Planner, compactor, and Inspector API servers. The
wrappers do not reserve GPUs, start servers, or interfere with running jobs.
Then launch search from the repository root:

```bash
PARQUET=/path/to/public_fixed300.parquet \
RUN_ROOT=/path/to/new-search-run \
PYTHON=/path/to/runtime/python \
PLANNER_ENDPOINT=http://127.0.0.1:19207/v1 \
SUMMARIZER_ENDPOINT=http://127.0.0.1:19507/v1 \
INSPECTOR_ENDPOINT=http://127.0.0.1:19405/v1 \
CONCURRENCY=2 \
bash baselines/videoseal_6000d/scripts/run_search.sh
```

The actual run began at two-question concurrency and later used eight with
3 Planner / 3 compactor / 2 Inspector replicas. This changed throughput only;
replication controllers and live process-injection scripts are not part of the
method or this release. API server processor limits must also match the recorded
configuration, including the original Inspector `max_pixels=200704`.

After the complete search stage, use an available GPU for the original local
Observer. Manage your own API processes before assigning the GPU:

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/runtime/python \
  baselines/videoseal_6000d/scripts/run_conditional_observer.py \
  --parquet /path/to/public_fixed300.parquet \
  --search-root /path/to/new-search-run \
  --output-root /path/to/new-observer-run \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct
```

The output directory must be new. The wrapper preserves original search results,
records Observer failures, and writes `FINAL_PREDICTIONS.jsonl` using the original
conditional adoption rule. It does not score or read gold answers.

For a different dataset, model, index, parser, or generation configuration,
record a new experiment identity. Do not attribute such changes to OPSD or claim
the archived score as a fresh reproduction.
