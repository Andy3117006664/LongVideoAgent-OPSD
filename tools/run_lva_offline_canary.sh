#!/usr/bin/env bash
set -euo pipefail

# Run from the isolated pilot directory.  This intentionally exercises the
# official async agent-loop path with subtitle-backed mock tools; it is a
# transport/serialization canary, not an OT-OPD result.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${ROOT}/env/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Qwen2.5-3B-Instruct}"
TRAIN="${TRAIN:-${ROOT}/data/parquet_500/train.parquet}"
VAL="${VAL:-${ROOT}/data/parquet_val_100/train.parquet}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
NGPU="${NGPU:-$(awk -F, '{print NF}' <<<"${GPUS}")}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${ROOT}/outputs/offline_canary_${STAMP}}"
TRAIN_BSZ="${TRAIN_BSZ:-1}"
PPO_MINI="${PPO_MINI:-1}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
STRATEGY="${STRATEGY:-fsdp2}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"

test -x "${PYTHON}"
test -f "${MODEL}/config.json"
test -f "${TRAIN}"
test -f "${VAL}"
export PYTHONPATH="${ROOT}/repo:${PYTHONPATH:-}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT}"
exec "${PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="${TRAIN}" \
  data.val_files="${VAL}" \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.val_batch_size=1 \
  data.dataloader_num_workers="${DATALOADER_WORKERS}" \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  +data.max_obs_length=256 \
  data.truncation=error \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.strategy="${STRATEGY}" \
  actor_rollout_ref.actor.fsdp_config.strategy="${STRATEGY}" \
  actor_rollout_ref.ref.strategy="${STRATEGY}" \
  actor_rollout_ref.ref.fsdp_config.strategy="${STRATEGY}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=false \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node="${NGPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enable_prefix_caching=false \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256 \
  actor_rollout_ref.rollout.agent.default_agent_loop=longvideoagent_multiturn \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${ROOT}/repo/videoagent/verl_ext/config/agent_loop.yaml" \
  +actor_rollout_ref.rollout.custom.videoagent.offline_mock=true \
  +actor_rollout_ref.rollout.custom.videoagent.max_turns=2 \
  +actor_rollout_ref.rollout.custom.videoagent.max_obs_length=256 \
  +actor_rollout_ref.rollout.custom.videoagent.subs_path="${ROOT}/data/LongTVQA_plus/LongTVQA_plus_subtitle_clip_level.json" \
  +actor_rollout_ref.rollout.custom.videoagent.base_frame_dir="${ROOT}/data/frames/none" \
  +actor_rollout_ref.rollout.custom.videoagent.bbox_json_path="${ROOT}/data/frames/none.json" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  reward.custom_reward_function.path="${ROOT}/repo/videoagent/verl_ext/reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.use_legacy_worker_impl=disable \
  trainer.n_gpus_per_node="${NGPU}" \
  trainer.nnodes=1 \
  trainer.resume_mode=disable \
  trainer.ray_wait_register_center_timeout=120 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.logger='["console"]' \
  trainer.project_name=longvideoagent_otopd \
  trainer.experiment_name="offline_canary_${STAMP}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${OUT}" \
  2>&1 | tee "${OUT}/run.log"
