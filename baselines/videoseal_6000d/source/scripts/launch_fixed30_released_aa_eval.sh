#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/path/to/python}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/fixed30_released_aa_holdout48_20261001}"
PARQUET="${PARQUET:-${PROJECT_ROOT}/datasets/LVBench/videoseal/lvbench_full_home_indexed.parquet}"
UIDS_FILE="${UIDS_FILE:-${PROJECT_ROOT}/runs/fixed30_predicate_holdout48_20261001/uids.txt}"
PLANNER_ENDPOINT="${PLANNER_ENDPOINT:-http://127.0.0.1:18107/v1}"
INSPECTOR_ENDPOINT="${INSPECTOR_ENDPOINT:-http://127.0.0.1:18405/v1}"
SUMMARIZER_ENDPOINT="${SUMMARIZER_ENDPOINT:-http://127.0.0.1:18507/v1}"
BGE_ENDPOINT="${BGE_ENDPOINT:-http://127.0.0.1:18109/embed}"
SUMMARY_PROMPT_FILE="${SUMMARY_PROMPT_FILE:-${PROJECT_ROOT}/research/prompts/fixed30_lossless_compactor_v1.txt}"
PROTOCOL_FILE="${PROTOCOL_FILE:-${PROJECT_ROOT}/research/protocols/fixed30_released_aa_holdout48_20261001.json}"
CONCURRENCY="${CONCURRENCY:-2}"

cd "${PROJECT_ROOT}"
if [[ -e "${RUN_ROOT}" ]]; then
  printf 'refusing to overwrite existing run: %s\n' "${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

for url in \
  "${PLANNER_ENDPOINT%/v1}/v1/models" \
  "${INSPECTOR_ENDPOINT%/v1}/v1/models" \
  "${SUMMARIZER_ENDPOINT%/v1}/v1/models" \
  "${BGE_ENDPOINT%/embed}/health"; do
  curl -fsS --max-time 10 "${url}" >/dev/null
done

{
  printf 'method=fixed30_five_signal_rrf_released_sampler_aa\n'
  printf 'only_intervention=none_aa_replicate\n'
  printf 'visual_inspect_frame_selection=released_prefix_round_robin\n'
  printf 'answer_postprocessing=false\ngold_used_online=false\ntime_reference_used_online=false\n'
  printf 'weights=0.5,0.5,0.5,1.0,0.5\nrrf_constant=60\nsignal_depth=all\n'
  printf 'planner_endpoint=%s\ninspector_endpoint=%s\nsummarizer_endpoint=%s\nbge_endpoint=%s\n' \
    "${PLANNER_ENDPOINT}" "${INSPECTOR_ENDPOINT}" "${SUMMARIZER_ENDPOINT}" "${BGE_ENDPOINT}"
  printf 'summarizer_model=Qwen3-8B\nsummary_prompt_file=%s\n' "${SUMMARY_PROMPT_FILE}"
  printf 'max_steps=16\nframe_budget=64\nstrict_final_gate=1\ntask_timeout_sec=2400\n'
  printf 'planner_max_tokens=4096\nplanner_context_limit=40960\nconcurrency=%s\n' "${CONCURRENCY}"
  printf 'uids_file=%s\nparquet=%s\n' "${UIDS_FILE}" "${PARQUET}"
  printf 'protocol_file=%s\nprotocol_sha256=%s\n' \
    "${PROTOCOL_FILE}" "$(sha256sum "${PROTOCOL_FILE}" | awk '{print $1}')"
} >"${RUN_ROOT}/command.txt"

export PYTHONPATH="${PROJECT_ROOT}/baselines/VideoSEAL:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VIDEOSEAL_VISUAL_RETRIEVE_CLASS="methods.FIXED30_RETRIEVAL.tool:Fixed30VisualRetrieveTool"
export FIXED30_BGE_ENDPOINT="${BGE_ENDPOINT}"
export FIXED30_SIGNAL_DEPTH="0"
export AGENT_LLM_BACKEND="api"
export AGENT_API_USE_MESSAGES="0"
export AGENT_LLM_API_BASE="${PLANNER_ENDPOINT}"
export AGENT_LLM_API_KEY="local-vllm"
export AGENT_LLM_MODEL="Qwen3-8B"
export AGENT_LLM_MAX_TOKENS="4096"
export AGENT_LLM_CONTEXT_LIMIT="40960"
export AGENT_LLM_TEMPERATURE="0.0"
export AGENT_LLM_TIMEOUT="600"
export AGENT_LEGACY_HISTORY_MAX_BLOCKS="0"
export AGENT_LEGACY_HISTORY_MAX_CHARS="0"
export VIDEOSEAL_DISABLE_TIME_REFERENCE="1"
export AGENT_MLLM_BACKEND="openai"
export MLLM_BACKEND="openai"
export MLLM_MAX_TOKENS="4096"
export MLLM_TIMEOUT="300"
export MLLM_RETRY_TIMES="1"
export MLLM_RETRY_DELAY="0"
export VISUAL_INSPECT_BACKEND="openai"
export VISUAL_INSPECT_API_BASE="${INSPECTOR_ENDPOINT}"
export VISUAL_INSPECT_API_KEY="local-vllm"
export VISUAL_INSPECT_MODEL="Qwen2.5-VL-7B-Instruct"
export VISUAL_INSPECT_EXCLUSIVE_OUTPUT="1"
export VISUAL_INSPECT_PRESERVE_UNCERTAINTY="0"
unset INSPECT_VLM_SYSTEM_PROMPT INSPECT_VLM_SYSTEM_PROMPT_FILE
unset VISUAL_INSPECT_PROMPT VISUAL_INSPECT_PROMPT_FILE VISUAL_INSPECT_SYSTEM_PROMPT_FILE
export VISUAL_RETRIEVE_SUM_BACKEND="openai"
export VISUAL_RETRIEVE_SUM_API_BASE="${SUMMARIZER_ENDPOINT}"
export VISUAL_RETRIEVE_SUM_API_KEY="local-vllm"
export VISUAL_RETRIEVE_SUM_MODEL="Qwen3-8B"
export VISUAL_RETRIEVE_SUM_MAX_TOKENS="2048"
export VISUAL_RETRIEVE_SUM_TEMPERATURE="0.1"
export VISUAL_RETRIEVE_SUMMARY_PROMPT_FILE="${SUMMARY_PROMPT_FILE}"
export SEMANTIC_RETRIEVE_MIX="bm25"
export SEMANTIC_RETRIEVE_TOPK="40"
export VISUAL_RETRIEVE_TOPK="30"
export RETRIEVE_SUMMARY_ENABLED="1"
export RETRIEVE_SUMMARY_MAX_SPANS="100"
export VISUAL_RETRIEVE_RETURN_SPANS="1"
export VISUAL_RETRIEVE_LOG_SPANS="1"
export RETRIEVAL_PRIORITY_CHANNEL="0"
export INSPECT_PENDING_SLOT="0"
export INSPECT_GLOBAL_ORDER="1"
export INSPECT_MAX_TOTAL_IMAGES="64"
export VISUAL_INSPECT_MAX_TOTAL_IMAGES="64"
export INSPECT_FPS="2"
export INSPECT_MAX_LONG_EDGE="1280"
export VISUAL_INSPECT_MAX_LONG_EDGE="1280"
export VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE="1"
export VISUAL_INSPECT_TOTAL_PIXELS="20971520"
export VISUAL_INSPECT_MIN_PIXELS="12544"
export VISUAL_INSPECT_EDGE_MULTIPLE="32"
export INSPECT_VLM_MAX_TOKENS="4096"
export INSPECT_VLM_TEMPERATURE="0.1"
unset VISUAL_INSPECT_FRAME_SELECTION INSPECT_FRAME_SELECTION
export VISUAL_INSPECT_RETURN_FRAME_METADATA="1"
export MAX_STEPS="16"
export AGENT_FORCE_LAST_STEP_VISUAL_INSPECT="1"
export AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK="1"
export AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK="1"
export AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE="mcq"
export AGENT_PARSE_FAILURE_FALLBACK_TO_C="0"
export VIDEOSEAL_STRICT_FINAL_GATE="1"
export CLEAN_FRAMES="0"
export CLEAN_FRAMES_ASYNC="1"
export TASK_TIMEOUT_SEC="2400"
export LVQA_PATH_REMAP_FROM="/path/to/lvqa-experiments"
export LVQA_PATH_REMAP_TO="${PROJECT_ROOT}"
export EVAL_RUN_ROOT="${RUN_ROOT}"
export EVAL_MODE="decoupled"

setsid nohup "${PYTHON}" "${PROJECT_ROOT}/scripts/run_videoseal_decoupled_eval.py" \
  --parquet "${PARQUET}" --run-root "${RUN_ROOT}" --uids-file "${UIDS_FILE}" \
  --concurrency "${CONCURRENCY}" --max-steps 16 --task-timeout-sec 2400 \
  --retrieval-mode bm25 --strict-final-gate >"${RUN_ROOT}/eval.log" 2>&1 < /dev/null &
worker_pid="$!"
printf '%s\n' "${worker_pid}" >"${RUN_ROOT}/worker.pid"
printf '[%s] worker=%s run=%s\n' "$(date -Is)" "${worker_pid}" "${RUN_ROOT}"
