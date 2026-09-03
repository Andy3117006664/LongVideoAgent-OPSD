#!/usr/bin/env bash
set -euo pipefail

# Minimal paired LongVideoAgent launcher for a patched local-cache checkout.
#
# This launcher never calls a remote grounding/vision endpoint: the custom
# agent loop is put in ``offline_cache=true`` mode and receives local JSON
# cache paths.  It is a structural canary by default (10 steps, 2 rollouts,
# two agent turns), not a paper-scale reproduction.  Run the same command
# twice with MODE=baseline and MODE=ot and keep all other paths/settings fixed.
#
# Expected remote layout (override every path with an environment variable if
# yours differs):
#   ${LVA_ROOT}/repo                 patched LongVideoAgent checkout
#   ${LVA_ROOT}/env/bin/python       environment used by the checkout
#   ${LVA_ROOT}/models/Qwen2.5-3B-Instruct
#   ${LVA_ROOT}/data/{parquet_500,parquet_val_100}/*.parquet
#   ${LVA_ROOT}/data/cache/{grounding,observations,ot_records}.json
#
# Examples:
#   LVA_ROOT=/root/autodl-tmp/lva-pilot MODE=baseline TOTAL_STEPS=10 \
#     bash /tmp/run_lva_localcache_otopd.sh
#   LVA_ROOT=/root/autodl-tmp/lva-pilot MODE=ot OT_OPD_ALPHA=0.1 TOTAL_STEPS=10 \
#     bash /tmp/run_lva_localcache_otopd.sh
#
# The OT run is meaningful only if ot_records.json was generated for the
# *same sampled trajectories* (the patched loop checks response/observation/
# target hashes by default).  Set ALLOW_EMPTY_OT_CACHE=1 only for a plumbing
# canary; an empty/mismatched cache makes alpha>0 a no-op and is not a result.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LVA_ROOT:-${SCRIPT_DIR}}"
REPO="${LVA_REPO:-${ROOT}/repo}"
PYTHON="${PYTHON:-${ROOT}/env/bin/python}"

MODE="${MODE:-baseline}"
case "${MODE}" in
  baseline)
    ALPHA="0"
    ;;
  ot)
    ALPHA="${OT_OPD_ALPHA:-${ALPHA:-0.1}}"
    if [[ "${ALPHA}" == "0" || "${ALPHA}" == "0.0" ]]; then
      echo "[error] MODE=ot requires OT_OPD_ALPHA/ALPHA != 0" >&2
      exit 2
    fi
    ;;
  *)
    echo "[error] MODE must be baseline or ot (got ${MODE})" >&2
    exit 2
    ;;
esac
export OT_OPD_ALPHA="${ALPHA}"
export OT_OPD_SIGNAL_KEY="${OT_OPD_SIGNAL_KEY:-e_obs}"
export OT_OPD_NORMALIZE="${OT_OPD_NORMALIZE:-true}"
export OT_OPD_CLIP="${OT_OPD_CLIP:-2.0}"

# Keep the reference-policy regularizer explicit. The released GRPO
# launchers use the actor-side low-variance KL loss (coefficient 1e-3), while
# ``algorithm.use_kl_in_reward`` is disabled. Override these for ablations.
USE_KL_LOSS="${USE_KL_LOSS:-true}"
KL_COEF="${KL_COEF:-0.001}"
KL_TYPE="${KL_TYPE:-low_var_kl}"
ALGORITHM_KL_CTRL_COEF="${ALGORITHM_KL_CTRL_COEF:-0.0001}"

MODEL="${MODEL:-${ROOT}/models/Qwen2.5-3B-Instruct}"
TRAIN="${TRAIN:-${ROOT}/data/parquet_500/train.parquet}"
# The official converter writes ``val.parquet``.  A few pilot builders put a
# held-out pilot under ``train.parquet`` in a separate directory, so resolve
# that legacy layout only when the canonical file is absent (and make the
# fallback visible in the preflight log).
VAL="${VAL:-${ROOT}/data/parquet_val_100/val.parquet}"
if [[ ! -f "${VAL}" && -f "${ROOT}/data/parquet_val_100/train.parquet" ]]; then
  VAL="${ROOT}/data/parquet_val_100/train.parquet"
  echo "[warn] canonical validation path missing; using legacy pilot path ${VAL}" >&2
fi
SUBS="${SUBS:-${ROOT}/data/LongTVQA_plus/LongTVQA_plus_subtitle_clip_level.json}"
FRAME_DIR="${FRAME_DIR:-${ROOT}/data/frames/none}"
BBOX_JSON="${BBOX_JSON:-${ROOT}/data/frames/none.json}"
GROUNDING_CACHE="${GROUNDING_CACHE:-${ROOT}/data/cache/grounding.json}"
OBSERVATION_CACHE="${OBSERVATION_CACHE:-${ROOT}/data/cache/observations.json}"
OT_CACHE="${OT_CACHE:-${ROOT}/data/cache/ot_records.json}"
# Optional exact-prefix scorer.  It is deliberately off by default because it
# loads a frozen student/teacher pair inside each rollout worker.  Enable it
# only for a small smoke or when the worker GPU has been reserved for scoring.
OT_ONLINE_SCORE="${OT_ONLINE_SCORE:-false}"
OT_ONLINE_STUDENT_MODEL="${OT_ONLINE_STUDENT_MODEL:-${MODEL}}"
OT_ONLINE_TEACHER_MODEL="${OT_ONLINE_TEACHER_MODEL:-}"
OT_ONLINE_DEVICE="${OT_ONLINE_DEVICE:-cuda:0}"
OT_ONLINE_DTYPE="${OT_ONLINE_DTYPE:-auto}"
OT_ONLINE_BATCH_SIZE="${OT_ONLINE_BATCH_SIZE:-1}"
OT_ONLINE_MAX_RECORDS="${OT_ONLINE_MAX_RECORDS:-0}"
OT_ONLINE_REDUCTION="${OT_ONLINE_REDUCTION:-mean}"
OT_ONLINE_URL="${OT_ONLINE_URL:-}"
OT_ONLINE_TIMEOUT="${OT_ONLINE_TIMEOUT:-120}"
OFFLINE_CACHE="${OFFLINE_CACHE:-1}"
if [[ "${OFFLINE_CACHE}" == "0" ]]; then
  OFFLINE_CACHE_BOOL=false
else
  OFFLINE_CACHE_BOOL=true
fi

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NGPU="${NGPU:-${#GPU_IDS[@]}}"

TOTAL_STEPS="${TOTAL_STEPS:-10}"
TRAIN_BSZ="${TRAIN_BSZ:-2}"
ROLLOUT_N="${ROLLOUT_N:-2}"
# In the current FSDP2 trainer this is the per-rollout actor mini-batch
# setting; ray_trainer multiplies it by rollout.n before checking divisibility
# by the data-parallel world size.  Default to TRAIN_BSZ so B=2,N=2 works on
# four GPUs (global mini = 4).  Override explicitly for other layouts.
PPO_MINI="${PPO_MINI:-${TRAIN_BSZ}}"
VAL_BSZ="${VAL_BSZ:-1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-500}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-100}"
MAX_TURNS="${MAX_TURNS:-2}"
MAX_OBS_LEN="${MAX_OBS_LEN:-256}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESP_LEN="${MAX_RESP_LEN:-512}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
TEST_FREQ="${TEST_FREQ:-${TOTAL_STEPS}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-lva_localcache_${MODE}_a${ALPHA//./p}_${STAMP}}"
OUT="${OUT:-${ROOT}/outputs/${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DATA_DIR:-${OUT}/rollouts}"
VAL_DIR="${VALIDATION_DATA_DIR:-${OUT}/val}"

if ! [[ "${NGPU}" =~ ^[0-9]+$ && "${NGPU}" -gt 0 ]]; then
  echo "[error] NGPU must be a positive integer (got ${NGPU})" >&2
  exit 2
fi
if ! [[ "${TOTAL_STEPS}" =~ ^[0-9]+$ && "${TOTAL_STEPS}" -ge 1 && "${TOTAL_STEPS}" -le 50 ]]; then
  echo "[error] TOTAL_STEPS must be an integer in [1,50] for this canary" >&2
  exit 2
fi
if (( (TRAIN_BSZ * ROLLOUT_N) % NGPU != 0 )); then
  echo "[error] train_batch_size * rollout.n must be divisible by NGPU: ${TRAIN_BSZ}*${ROLLOUT_N} vs ${NGPU}" >&2
  echo "        e.g. on 4 GPUs use TRAIN_BSZ=2,ROLLOUT_N=2 or TRAIN_BSZ=4,ROLLOUT_N=4" >&2
  exit 2
fi
if (( (PPO_MINI * ROLLOUT_N) % NGPU != 0 )); then
  echo "[error] actor ppo_mini_batch_size * rollout.n must be divisible by NGPU: ${PPO_MINI}*${ROLLOUT_N} vs ${NGPU}" >&2
  echo "        set PPO_MINI so PPO_MINI*ROLLOUT_N is a multiple of NGPU (e.g. 2*2 on 4 GPUs)" >&2
  exit 2
fi
if (( PPO_MINI > TRAIN_BSZ )); then
  echo "[error] PPO_MINI cannot exceed TRAIN_BSZ: ${PPO_MINI} > ${TRAIN_BSZ}" >&2
  exit 2
fi

require_file() {
  local path="$1"
  local label="$2"
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    echo "[error] missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_executable() {
  local path="$1"
  local label="$2"
  if [[ -z "${path}" || ! -x "${path}" ]]; then
    echo "[error] ${label} is not executable: ${path}" >&2
    exit 2
  fi
}

require_executable "${PYTHON}" "python executable"
require_file "${MODEL}/config.json" "model config"
require_file "${TRAIN}" "train parquet"
require_file "${VAL}" "validation parquet"
require_file "${SUBS}" "clip subtitle JSON"
require_file "${REPO}/videoagent/verl_ext/agent_loop.py" "patched agent_loop.py"
require_file "${REPO}/videoagent/verl_ext/ot_opd.py" "OT-OPD helper"
require_file "${REPO}/verl/trainer/ppo/ray_trainer.py" "trainer ray_trainer.py"
require_file "${REPO}/verl/experimental/agent_loop/agent_loop.py" "agent-loop base"

search_text() {
  local pattern="$1"
  local path="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q "${pattern}" "${path}"
  else
    grep -Eq "${pattern}" "${path}"
  fi
}

if ! search_text "offline_cache" "${REPO}/videoagent/verl_ext/agent_loop.py"; then
  echo "[error] checkout does not contain the offline_cache agent-loop patch: ${REPO}" >&2
  exit 2
fi
if ! search_text "apply_ot_opd_advantage" "${REPO}/verl/trainer/ppo/ray_trainer.py"; then
  echo "[error] checkout does not contain the OT-OPD trainer hook: ${REPO}" >&2
  exit 2
fi
if ! search_text "rollout_n.*trajectory|trajectory.*rollout_n" "${REPO}/verl/experimental/agent_loop/agent_loop.py"; then
  echo "[warn] base agent loop does not appear to forward rollout_n; strict OT cache keys may all miss" >&2
fi

if [[ "${OFFLINE_CACHE}" != "0" ]]; then
  require_file "${GROUNDING_CACHE}" "grounding cache"
  if [[ "${REQUIRE_OBS_CACHE:-0}" != "0" ]]; then
    require_file "${OBSERVATION_CACHE}" "observation cache"
  elif [[ ! -f "${OBSERVATION_CACHE}" ]]; then
    echo "[warn] observation cache missing; visual-query transitions will be deterministic cache misses" >&2
  fi
fi

if [[ "${MODE}" == "ot" ]]; then
  require_file "${OT_CACHE}" "OT replay cache"
  if [[ "${ALLOW_EMPTY_OT_CACHE:-0}" == "0" ]]; then
    "${PYTHON}" - "${OT_CACHE}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    obj = json.load(f)
entries = obj.get("records", obj) if isinstance(obj, dict) else None
if not isinstance(entries, dict) or not entries:
    raise SystemExit(f"[error] OT cache has no records: {path}; use ALLOW_EMPTY_OT_CACHE=1 only for plumbing")
print(f"[check] OT cache records: {len(entries)}")
PY
  fi
fi

# An HTTP scorer is self-contained on its own host.  The in-process scorer,
# however, is imported lazily by the rollout worker; fail during preflight
# instead of discovering a missing module after Ray has allocated GPUs.
if [[ "${OT_ONLINE_SCORE}" != "false" && -z "${OT_ONLINE_URL}" ]]; then
  require_file "${REPO}/videoagent/verl_ext/online_ot_scorer.py" "online paired scorer module"
fi

mkdir -p "${OUT}" "${ROLLOUT_DIR}" "${VAL_DIR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
cd "${REPO}"

echo "[config] root=${ROOT} repo=${REPO} mode=${MODE} alpha=${ALPHA} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[config] train=${TRAIN} val=${VAL} steps=${TOTAL_STEPS} batch=${TRAIN_BSZ} rollout_n=${ROLLOUT_N}"
echo "[config] grounding_cache=${GROUNDING_CACHE} observation_cache=${OBSERVATION_CACHE} ot_cache=${OT_CACHE}"
echo "[output] rollout_data_dir=${ROLLOUT_DIR} validation_data_dir=${VAL_DIR}"

if [[ "${DRY_RUN:-0}" != "0" ]]; then
  echo "[dry-run] preflight passed; set DRY_RUN=0 to launch"
  exit 0
fi

# OT settings are exported above.  Keep the command line free of an
# ``algorithm.ot_opd`` subtree: stock verl's ``algorithm`` is instantiated as
# the structured ``AlgoConfig`` dataclass, which rejects unknown keys.  The
# patched hook intentionally reads OT_OPD_* from the environment so this
# launcher remains compatible without changing that dataclass/schema.
set +e
"${PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  algorithm.kl_ctrl.kl_coef="${ALGORITHM_KL_CTRL_COEF}" \
  data.train_files="${TRAIN}" \
  data.val_files="${VAL}" \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.val_batch_size="${VAL_BSZ}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.seed="${SEED}" \
  data.dataloader_num_workers=0 \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  +data.max_obs_length="${MAX_OBS_LEN}" \
  data.truncation=error \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.use_remove_padding=false \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=false \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.data_loader_seed="${SEED}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_TYPE}" \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.fsdp_config.strategy=fsdp2 \
  actor_rollout_ref.ref.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.n_gpus_per_node="${NGPU}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enable_prefix_caching=false \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_OBS_LEN}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=longvideoagent_multiturn \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${REPO}/videoagent/verl_ext/config/agent_loop.yaml" \
  +actor_rollout_ref.rollout.custom={} \
  +actor_rollout_ref.rollout.custom.videoagent.offline_cache="${OFFLINE_CACHE_BOOL}" \
  +actor_rollout_ref.rollout.custom.videoagent.offline_mock=false \
  +actor_rollout_ref.rollout.custom.videoagent.max_turns="${MAX_TURNS}" \
  +actor_rollout_ref.rollout.custom.videoagent.max_obs_length="${MAX_OBS_LEN}" \
  +actor_rollout_ref.rollout.custom.videoagent.subs_path="${SUBS}" \
  +actor_rollout_ref.rollout.custom.videoagent.base_frame_dir="${FRAME_DIR}" \
  +actor_rollout_ref.rollout.custom.videoagent.bbox_json_path="${BBOX_JSON}" \
  +actor_rollout_ref.rollout.custom.videoagent.grounding_cache_path="${GROUNDING_CACHE}" \
  +actor_rollout_ref.rollout.custom.videoagent.observation_cache_path="${OBSERVATION_CACHE}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_cache_path="${OT_CACHE}" \
  +actor_rollout_ref.rollout.custom.videoagent.grounding_cache_strict=true \
  +actor_rollout_ref.rollout.custom.videoagent.observation_cache_strict=true \
  +actor_rollout_ref.rollout.custom.videoagent.observation_cache_allow_generic="${ALLOW_GENERIC_OBS_CACHE:-false}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_cache_strict=true \
  +actor_rollout_ref.rollout.custom.videoagent.ot_cache_hash_lookup="${OT_CACHE_HASH_LOOKUP:-false}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_score="${OT_ONLINE_SCORE}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_student_model="${OT_ONLINE_STUDENT_MODEL}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_teacher_model="${OT_ONLINE_TEACHER_MODEL}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_device="${OT_ONLINE_DEVICE}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_dtype="${OT_ONLINE_DTYPE}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_batch_size="${OT_ONLINE_BATCH_SIZE}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_max_records="${OT_ONLINE_MAX_RECORDS}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_reduction="${OT_ONLINE_REDUCTION}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_privileged="${OT_ONLINE_PRIVILEGED:-false}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_standard_gap="${OT_ONLINE_STANDARD_GAP:-true}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_url="${OT_ONLINE_URL}" \
  +actor_rollout_ref.rollout.custom.videoagent.ot_online_timeout="${OT_ONLINE_TIMEOUT}" \
  +actor_rollout_ref.rollout.custom.videoagent.vision_model=local-cache \
  +actor_rollout_ref.rollout.custom.videoagent.grounding_model=local-cache \
  +actor_rollout_ref.rollout.custom.videoagent.api_key=offline \
  +actor_rollout_ref.rollout.custom.videoagent.vision_api=offline \
  +actor_rollout_ref.rollout.custom.videoagent.grounding_api=offline \
  +actor_rollout_ref.rollout.custom.videoagent.vision_base_url=http://127.0.0.1:9/v1 \
  +actor_rollout_ref.rollout.custom.videoagent.grounding_base_url=http://127.0.0.1:9/v1 \
  reward.custom_reward_function.path="${REPO}/videoagent/verl_ext/reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.use_legacy_worker_impl=disable \
  trainer.n_gpus_per_node="${NGPU}" \
  trainer.nnodes=1 \
  trainer.resume_mode=disable \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.val_before_train=true \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq=-1 \
  trainer.rollout_data_dir="${ROLLOUT_DIR}" \
  trainer.validation_data_dir="${VAL_DIR}" \
  trainer.log_val_generations=0 \
  trainer.logger='["console"]' \
  trainer.project_name=longvideoagent_otopd \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${OUT}/checkpoints" \
  2>&1 | tee "${OUT}/run.log"
STATUS=${PIPESTATUS[0]}
set -e

echo "[done] status=${STATUS} out=${OUT}"
echo "[done] validation JSONL: ${VAL_DIR}/0.jsonl and (if final validation ran) ${VAL_DIR}/${TOTAL_STEPS}.jsonl"
echo "[done] rollout JSONL files: ${ROLLOUT_DIR}/*.jsonl"
exit "${STATUS}"
