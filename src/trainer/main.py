"""Training entrypoint script for GRPO training with distributed support."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from src.trainer.config import TrainingConfig, FSDPConfig
from src.trainer.trainer import GRPOTrainer
from src.trainer.dataset import MathProblemDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ:
        # Running with torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # Running with SLURM
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config_from_file(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_config_from_env() -> dict:
    """Load configuration from environment variables."""
    config = {}
    
    env_mappings = {
        "MODEL_NAME": "model_name",
        "GROUP_SIZE": ("group_size", int),
        "LEARNING_RATE": ("learning_rate", float),
        "KL_COEF": ("kl_coef", float),
        "CLIP_RANGE": ("clip_range", float),
        "MAX_GRAD_NORM": ("max_grad_norm", float),
        "BATCH_SIZE": ("batch_size", int),
        "NUM_EPOCHS": ("num_epochs", int),
        "GRADIENT_ACCUMULATION_STEPS": ("gradient_accumulation_steps", int),
        "MAX_NEW_TOKENS": ("max_new_tokens", int),
        "TEMPERATURE": ("temperature", float),
        "TOP_P": ("top_p", float),
        "ENVIRONMENT_SERVICE_URL": "environment_service_url",
        "CHECKPOINT_DIR": "checkpoint_dir",
        "CHECKPOINT_INTERVAL": ("checkpoint_interval", int),
        "MAX_SAMPLES": ("max_samples", int),
    }
    
    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            if isinstance(mapping, tuple):
                key, converter = mapping
                config[key] = converter(value)
            else:
                config[mapping] = value
    
    return config


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Build TrainingConfig from args, env vars, and config file."""
    config_dict = {}
    
    # Load from config file if provided
    if args.config:
        config_dict.update(load_config_from_file(args.config))
    
    # Override with environment variables
    config_dict.update(load_config_from_env())
    
    # Override with CLI arguments
    if args.model_name:
        config_dict["model_name"] = args.model_name
    if args.batch_size:
        config_dict["batch_size"] = args.batch_size
    if args.num_epochs:
        config_dict["num_epochs"] = args.num_epochs
    if args.checkpoint_dir:
        config_dict["checkpoint_dir"] = args.checkpoint_dir
    if args.environment_url:
        config_dict["environment_service_url"] = args.environment_url
    
    # Build FSDP config
    fsdp_dict = config_dict.pop("fsdp", {})
    fsdp_config = FSDPConfig(**fsdp_dict) if fsdp_dict else FSDPConfig()
    
    return TrainingConfig(**config_dict, fsdp=fsdp_config)


async def train(config: TrainingConfig, dataset_path: str, max_samples: int = None):
    """Run the training loop with distributed support."""
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    if is_main:
        logger.info("Starting GRPO training")
        logger.info(f"Config: {config}")
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    
    # Use max_samples from config if not provided as argument
    if max_samples is None:
        max_samples = getattr(config, 'max_samples', None)
    
    # Load dataset
    if dataset_path:
        if is_main:
            logger.info(f"Loading dataset from {dataset_path}")
        dataset = MathProblemDataset.from_json(dataset_path)
    else:
        if is_main:
            logger.info(f"Loading GSM8K dataset from HuggingFace (max_samples={max_samples})")
        dataset = MathProblemDataset.from_huggingface(
            max_samples=max_samples
        )
    
    if is_main:
        logger.info(f"Dataset size: {len(dataset)}")
    
    # Initialize trainer with distributed config
    trainer = GRPOTrainer(config)
    trainer.setup(rank=rank, world_size=world_size, local_rank=local_rank)
    
    # Synchronize all ranks before starting training
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        if is_main:
            logger.info("All ranks synchronized, starting training loop")
    
    # Training loop
    total_steps = 0
    for epoch in range(config.num_epochs):
        if is_main:
            logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
        
        offset = 0
        while offset < len(dataset):
            batch = dataset.get_batch(config.batch_size, offset)
            if not batch:
                break
            
            metrics = await trainer.train_step(batch)
            total_steps += 1
            
            if is_main:
                logger.info(
                    f"Step {total_steps}: "
                    f"loss={metrics.policy_loss:.4f}, "
                    f"reward={metrics.mean_reward:.4f}, "
                    f"kl={metrics.kl_divergence:.4f}"
                )
            
            offset += config.batch_size
    
    # Final checkpoint (only main process)
    if is_main:
        final_path = f"{config.checkpoint_dir}/final"
        trainer.save_checkpoint(final_path)
        logger.info(f"Training complete. Final checkpoint saved to {final_path}")
    
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Math LLM")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to local dataset JSON file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to load from dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--environment-url",
        type=str,
        help="URL of the environment service",
    )
    
    args = parser.parse_args()
    
    config = build_config(args)
    
    asyncio.run(train(config, args.dataset, args.max_samples))


if __name__ == "__main__":
    main()
