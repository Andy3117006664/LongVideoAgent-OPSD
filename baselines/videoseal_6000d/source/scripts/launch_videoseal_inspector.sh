#!/usr/bin/env bash
set -Eeuo pipefail

# Project-local visual evidence reader for the VideoSEAL Decoupled evaluator.
# Keeping the JSON arguments in a script avoids losing their quotes when the
# command is submitted through an authenticated SSH control socket.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/path/to/python}"
MODEL="${MODEL:-${PROJECT_ROOT}/models/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-18101}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MM_IMAGE_LIMIT="${MM_IMAGE_LIMIT:-128}"
TAG="${REPLICA_TAG:-gpu${GPU}}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-${PROJECT_ROOT}/.tmp/vllm-inspector-rpc-${TAG}}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.tmp/vllm-inspector-tmp-${TAG}}"
export TMP="${TMP:-${PROJECT_ROOT}/.tmp/vllm-inspector-tmp-${TAG}}"
export TEMP="${TEMP:-${PROJECT_ROOT}/.tmp/vllm-inspector-tmp-${TAG}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_ROOT}/.tmp/vllm-inspector-triton-${TAG}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PROJECT_ROOT}/.tmp/vllm-inspector-torch-${TAG}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PROJECT_ROOT}/.tmp/vllm-inspector-cache-${TAG}}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.tmp/hf-cache-inspector-${TAG}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.tmp/xdg-cache-inspector-${TAG}}"

mkdir -p "${VLLM_RPC_BASE_PATH}" "${TMPDIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${HF_HOME}" "${XDG_CACHE_HOME}"

exec "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.45}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --dtype bfloat16 \
  --trust-remote-code \
  --enforce-eager \
  --disable-frontend-multiprocessing \
  --limit-mm-per-prompt "{\"image\":${MM_IMAGE_LIMIT}}" \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":200704}'
