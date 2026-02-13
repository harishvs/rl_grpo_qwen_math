"""Training entrypoint script for GRPO training."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from src.trainer.config import TrainingConfig, FSDPConfig
from src.trainer.trainer import GRPOTrainer
from src.trainer.dataset import MathProblemDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    """Run the training loop."""
    logger.info("Starting GRPO training")
    logger.info(f"Config: {config}")
    
    # Load dataset
    if dataset_path:
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = MathProblemDataset.from_json(dataset_path)
    else:
        logger.info("Loading GSM8K dataset from HuggingFace")
        dataset = MathProblemDataset.from_huggingface(
            max_samples=max_samples
        )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Initialize trainer
    trainer = GRPOTrainer(config)
    trainer.setup()
    
    # Training loop
    total_steps = 0
    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
        
        offset = 0
        while offset < len(dataset):
            batch = dataset.get_batch(config.batch_size, offset)
            if not batch:
                break
            
            metrics = await trainer.train_step(batch)
            total_steps += 1
            
            logger.info(
                f"Step {total_steps}: "
                f"loss={metrics.policy_loss:.4f}, "
                f"reward={metrics.mean_reward:.4f}, "
                f"kl={metrics.kl_divergence:.4f}"
            )
            
            offset += config.batch_size
    
    # Final checkpoint
    final_path = f"{config.checkpoint_dir}/final"
    trainer.save_checkpoint(final_path)
    logger.info(f"Training complete. Final checkpoint saved to {final_path}")


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
