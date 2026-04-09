"""Monarch GRPO Controller -- spawns actors and runs the training loop.

This is the entry point that runs inside the cluster on GPU Node 0.
It spawns all actors on the right GPU meshes and orchestrates the
synchronous GRPO training loop:

  DatasetActor -> Generator -> RewardActor -> advantages -> Learner -> weight sync

Usage:
    python -m src.monarch.main config/monarch/qwen-1.5b.yaml
"""
import asyncio
import os
import sys
import time

import torch
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env

from src.monarch.config import load_config
from src.monarch.metrics import MetricsLogger
from src.shared.grpo import compute_grpo_advantages

import urllib.request
import json

from src.monarch.actors.generator import GeneratorActor
from src.monarch.actors.learner import LearnerActor
from src.monarch.actors.dataset_actor import DatasetActor
from src.monarch.actors.replay_buffer import ReplayBufferActor
from src.monarch.actors.param_server import ParameterServerActor


async def main(config_path: str):
    """Run the distributed GRPO training loop.

    Args:
        config_path: Path to YAML config file
    """
    config = load_config(config_path)

    # --- Metrics ---
    metrics_logger = MetricsLogger(experiment_name=config.experiment_name)
    metrics_logger.start()

    # --- Provision meshes ---
    # Option 1: KubernetesJob (attach to pre-provisioned MonarchMesh pods)
    # Option 2: Local (for development with this_host())
    try:
        from monarch._src.job.kubernetes import KubernetesJob
        job = KubernetesJob(namespace="default")
        job.add_mesh("grpomonarch", num_replicas=2, label_selector="monarch.pytorch.org/mesh-name=grpo-monarch")
        state = job.state(cached_path=None)
        hosts = state.grpomonarch
    except Exception as e:
        print(f"KubernetesJob failed ({e}), falling back to local host", flush=True)
        # Fallback to local for development
        print("KubernetesJob not available, falling back to local host", flush=True)
        hosts = this_host()

    # Split hosts: host 0 = Learner (FSDP), host 1 = Generator (vLLM)
    learner_host = hosts.slice(hosts=slice(0, 1))
    generator_host = hosts.slice(hosts=slice(1, 2))

    # Generator: single process (vLLM handles GPU parallelism internally)
    generator_procs = generator_host.spawn_procs(per_host={"gpus": 1})

    # Learner: one process per GPU for FSDP (each is an FSDP rank)
    learner_procs = learner_host.spawn_procs(per_host={"gpus": config.trainer.n_gpus_per_node})

    # Set RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT on all learner processes
    # so dist.init_process_group("nccl") works inside the actor
    print("Setting up torch elastic env on learner mesh...", flush=True)
    setup_torch_elastic_env(learner_procs)

    # CPU mesh (on the same host as the controller)
    cpu_procs = this_host().spawn_procs(per_host={"cpus": 1})

    # --- Spawn actors ---
    print("Spawning actors...", flush=True)

    dataset = cpu_procs.spawn(
        "dataset",
        DatasetActor,
        dataset_name=config.data.dataset_name,
        max_samples=config.data.max_samples,
    )

    # Reward service runs as a separate FastAPI deployment (lightweight CPU image)
    reward_url = os.environ.get("REWARD_SERVICE_URL", "http://monarch-reward:8080")

    replay_buffer = cpu_procs.spawn("replay_buffer", ReplayBufferActor)

    generator = generator_procs.spawn(
        "generator",
        GeneratorActor,
        model_name=config.model,
        tensor_parallel_size=config.generator.tensor_parallel_size,
        gpu_memory_utilization=config.generator.gpu_memory_utilization,
        max_tokens=config.data.max_response_length,
    )

    learner = learner_procs.spawn(
        "learner",
        LearnerActor,
        model_name=config.model,
        learning_rate=config.learner.lr,
        kl_coef=config.learner.kl_coef,
        clip_range=config.learner.clip_range,
        max_grad_norm=config.learner.max_grad_norm,
        gradient_checkpointing=config.learner.gradient_checkpointing,
    )

    print("All actors spawned, initializing...", flush=True)
    init_results = await learner.initialize.call()
    for r in init_results:
        print(f"  Learner: {r}", flush=True)

    gen_result = await generator.initialize.call_one()
    print(f"  Generator: {gen_result}", flush=True)

    # --- Training loop ---
    prompts_per_step = config.data.train_batch_size // config.generator.group_size
    dataset_stats = await dataset.get_stats.call_one()
    total_problems = dataset_stats["total_problems"]
    steps_per_epoch = total_problems // prompts_per_step
    max_steps = steps_per_epoch * config.trainer.total_epochs

    print(
        f"Training: {total_problems} problems, {prompts_per_step} prompts/step, "
        f"{steps_per_epoch} steps/epoch, {max_steps} total steps",
        flush=True,
    )

    for step in range(max_steps):
        step_start = time.time()
        t_phase = time.time()

        # 1. Get batch of prompts
        batch = await dataset.next_batch.call_one(prompts_per_step)
        prompts = batch["prompts"]
        ground_truths = batch["ground_truths"]

        prompt_strings = [
            p[0]["content"] if isinstance(p, list) and len(p) > 0 else str(p)
            for p in prompts
        ]

        # 2. Generate completions (G per prompt)
        gen_result = await generator.generate.call_one(
            prompt_strings,
            group_size=config.generator.group_size,
            temperature=config.generator.temperature,
            top_p=config.generator.top_p,
        )
        t_gen = time.time() - t_phase
        t_phase = time.time()

        # 3. Score rewards via HTTP reward service
        gt_expanded = []
        for gt in ground_truths:
            gt_expanded.extend([gt] * config.generator.group_size)

        score_payload = json.dumps({
            "completions": gen_result["completions"],
            "ground_truths": gt_expanded,
        }).encode()
        req = urllib.request.Request(
            f"{reward_url}/score",
            data=score_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            rewards = json.loads(resp.read())

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # 4. Compute GRPO advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        advantages = compute_grpo_advantages(
            rewards_tensor, config.generator.group_size
        )

        # 5. Store in replay buffer
        episodes = {
            "prompts": gen_result["prompts_expanded"],
            "completions": gen_result["completions"],
            "log_probs": gen_result["log_probs"],
            "rewards": rewards,
            "advantages": advantages.tolist(),
        }
        await replay_buffer.add.call_one(episodes, step)
        t_score = time.time() - t_phase
        t_phase = time.time()

        # 6. Train
        train_batch = await replay_buffer.sample.call_one(
            batch_size=1,
            current_version=step,
        )

        if train_batch is not None:
            train_results = await learner.train_step.call(train_batch)
            train_metrics = train_results.values()[0]
        else:
            train_metrics = {"loss": 0.0, "kl_divergence": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0}
        t_train = time.time() - t_phase
        t_phase = time.time()

        # 7. Sync weights to generator (every N steps)
        t_sync = 0.0
        if step > 0 and step % config.trainer.weight_sync_interval == 0:
            weights_list = await learner.get_weights.call()
            weights_bytes = weights_list.values()[0]
            await generator.update_weights.call_one(weights_bytes, step)
            t_sync = time.time() - t_phase

        # 8. Checkpoint
        if step > 0 and step % config.trainer.save_freq == 0:
            save_path = f"{config.trainer.checkpoint_dir}/step_{step}"
            await learner.save_checkpoint.call(save_path)  # all ranks participate in summon_full_params
            print(f"Checkpoint saved: {save_path}", flush=True)

        # 9. Log metrics with timing breakdown
        step_time = time.time() - step_start
        metrics_logger.log(step, {
            "policy_loss": train_metrics.get("loss", 0.0),
            "kl_divergence": train_metrics.get("kl_divergence", 0.0),
            "mean_reward": mean_reward,
            "clip_fraction": train_metrics.get("clip_fraction", 0.0),
            "grad_norm": train_metrics.get("grad_norm", 0.0),
            "step_time_seconds": step_time,
            "t_generation": t_gen,
            "t_scoring": t_score,
            "t_training": t_train,
            "t_weight_sync": t_sync,
            "throughput_samples_per_sec": len(rewards) / step_time if step_time > 0 else 0,
            "epoch": batch["epoch"],
        })

    # --- Final checkpoint ---
    final_path = f"{config.trainer.checkpoint_dir}/final"
    await learner.save_checkpoint.call_one(final_path)
    print(f"Training complete. Final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.monarch.main <config.yaml>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
