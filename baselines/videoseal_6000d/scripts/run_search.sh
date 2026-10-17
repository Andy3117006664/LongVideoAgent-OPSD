#!/usr/bin/env bash
set -Eeuo pipefail
BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-python3}"
export PARQUET="${PARQUET:?Set PARQUET to your public, answer-free evaluation parquet}"
export RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to a new search output directory}"
export UIDS_FILE="${UIDS_FILE:-${BASELINE_ROOT}/selection/uids.txt}"
export PROTOCOL_FILE="${PROTOCOL_FILE:-${BASELINE_ROOT}/selection/identity_mapping.json}"
export PLANNER_ENDPOINT="${PLANNER_ENDPOINT:-http://127.0.0.1:19207/v1}"
export INSPECTOR_ENDPOINT="${INSPECTOR_ENDPOINT:-http://127.0.0.1:19405/v1}"
export SUMMARIZER_ENDPOINT="${SUMMARIZER_ENDPOINT:-http://127.0.0.1:19507/v1}"
export CONCURRENCY="${CONCURRENCY:-2}"
baseline_absolute_path() {
  "${PYTHON}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"
}
if [[ "${PYTHON}" == */* ]]; then
  export PYTHON="$(baseline_absolute_path "${PYTHON}")"
fi
export PARQUET="$(baseline_absolute_path "${PARQUET}")"
export RUN_ROOT="$(baseline_absolute_path "${RUN_ROOT}")"
export UIDS_FILE="$(baseline_absolute_path "${UIDS_FILE}")"
export PROTOCOL_FILE="$(baseline_absolute_path "${PROTOCOL_FILE}")"
if [[ -n "${SUMMARY_PROMPT_FILE:-}" ]]; then
  export SUMMARY_PROMPT_FILE="$(baseline_absolute_path "${SUMMARY_PROMPT_FILE}")"
fi
# The captured launcher owns all method settings. This wrapper only supplies paths.
source "${BASELINE_ROOT}/source/scripts/launch_original_bm25_no_fixed30_eval.sh"
wait "${worker_pid}"
