#!/bin/bash
set -ex
# All VERL_* env vars are set by the launch script from config/verl/*.yaml
: "${VERL_MODEL:=Qwen/Qwen2.5-1.5B}"
: "${VERL_EXPERIMENT:=default}"
: "${VERL_TRAIN_BATCH_SIZE:=256}"
: "${VERL_MAX_PROMPT_LENGTH:=512}"
: "${VERL_MAX_RESPONSE_LENGTH:=1024}"
: "${VERL_MINI_BATCH_SIZE:=128}"
: "${VERL_MICRO_BATCH_SIZE:=16}"
: "${VERL_LR:=5e-6}"
: "${VERL_ACTOR_PARAM_OFFLOAD:=False}"
: "${VERL_ACTOR_OPTIMIZER_OFFLOAD:=False}"
: "${VERL_GRADIENT_CHECKPOINTING:=True}"
: "${VERL_GPU_MEM_UTIL:=0.5}"
: "${VERL_ROLLOUT_N:=8}"
: "${VERL_ROLLOUT_LOG_PROB_MICRO_BATCH:=32}"
: "${VERL_ROLLOUT_TP:=1}"
: "${VERL_REF_LOG_PROB_MICRO_BATCH:=32}"
: "${VERL_REF_PARAM_OFFLOAD:=True}"
: "${VERL_N_GPUS:=8}"
: "${VERL_NNODES:=2}"
: "${VERL_TOTAL_EPOCHS:=1}"
: "${VERL_SAVE_FREQ:=20}"
: "${VERL_TEST_FREQ:=10}"

# Prep data
python3 /scripts/prep_data.py
# Download model
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('${VERL_MODEL}'); AutoModelForCausalLM.from_pretrained('${VERL_MODEL}')"
# Run GRPO
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/data/gsm8k/train.parquet \
  data.val_files=/data/gsm8k/test.parquet \
  data.train_batch_size=${VERL_TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${VERL_MAX_PROMPT_LENGTH} \
  data.max_response_length=${VERL_MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=${VERL_MODEL} \
  actor_rollout_ref.actor.optim.lr=${VERL_LR} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${VERL_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${VERL_MICRO_BATCH_SIZE} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.1 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=${VERL_GRADIENT_CHECKPOINTING} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${VERL_ACTOR_PARAM_OFFLOAD} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${VERL_ACTOR_OPTIMIZER_OFFLOAD} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${VERL_ROLLOUT_LOG_PROB_MICRO_BATCH} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${VERL_ROLLOUT_TP} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=${VERL_GPU_MEM_UTIL} \
  actor_rollout_ref.rollout.n=${VERL_ROLLOUT_N} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${VERL_REF_LOG_PROB_MICRO_BATCH} \
  actor_rollout_ref.ref.fsdp_config.param_offload=${VERL_REF_PARAM_OFFLOAD} \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path=/scripts/reward.py \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=grpo_gsm8k \
  trainer.experiment_name=${VERL_EXPERIMENT} \
  trainer.n_gpus_per_node=${VERL_N_GPUS} \
  trainer.nnodes=${VERL_NNODES} \
  trainer.save_freq=${VERL_SAVE_FREQ} \
  trainer.test_freq=${VERL_TEST_FREQ} \
  trainer.total_epochs=${VERL_TOTAL_EPOCHS} \
  trainer.default_hdfs_dir=null

