"""Training configuration for Monarch GRPO.

Loads from YAML config files. Monarch-specific fields (GPU mesh layout,
weight sync interval, actor placement) on top of shared GRPO hyperparameters.
"""
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class LearnerConfig:
    """Learner (training) actor configuration."""
    mini_batch_size: int = 128
    micro_batch_size_per_gpu: int = 16
    lr: float = 5e-6
    kl_coef: float = 0.1
    clip_range: float = 0.2
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    fsdp_sharding: str = "FULL_SHARD"
    mixed_precision: str = "bf16"


@dataclass
class GeneratorConfig:
    """Generator (inference) actor configuration."""
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1
    group_size: int = 8
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class RefConfig:
    """Reference model configuration."""
    enabled: bool = True
    param_offload: bool = True


@dataclass
class TrainerConfig:
    """Cluster and training loop configuration."""
    n_gpus_per_node: int = 8
    nnodes: int = 2
    total_epochs: int = 1
    save_freq: int = 20
    test_freq: int = 10
    weight_sync_interval: int = 3
    checkpoint_dir: str = "/checkpoints"


@dataclass
class DataConfig:
    """Dataset configuration."""
    train_batch_size: int = 256
    max_prompt_length: int = 512
    max_response_length: int = 1024
    dataset_name: str = "openai/gsm8k"
    max_samples: Optional[int] = None


@dataclass
class MonarchTrainingConfig:
    """Top-level configuration for Monarch GRPO training."""
    model: str = "Qwen/Qwen2.5-1.5B"
    experiment_name: str = "monarch-qwen1.5b"

    data: DataConfig = field(default_factory=DataConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def load_config(yaml_path: str) -> MonarchTrainingConfig:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Populated MonarchTrainingConfig
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    config = MonarchTrainingConfig()

    # Top-level fields
    if "model" in raw:
        config.model = raw["model"]
    if "experiment_name" in raw:
        config.experiment_name = raw["experiment_name"]

    # Nested sections
    section_map = {
        "data": (config.data, DataConfig),
        "learner": (config.learner, LearnerConfig),
        "generator": (config.generator, GeneratorConfig),
        "ref": (config.ref, RefConfig),
        "trainer": (config.trainer, TrainerConfig),
    }

    for section_name, (section_obj, section_cls) in section_map.items():
        if section_name in raw and isinstance(raw[section_name], dict):
            for key, value in raw[section_name].items():
                if hasattr(section_obj, key):
                    # Coerce to match the dataclass field type
                    expected_type = section_cls.__dataclass_fields__[key].type
                    if expected_type == "float" or expected_type is float:
                        value = float(value)
                    elif expected_type == "int" or expected_type is int:
                        value = int(value)
                    setattr(section_obj, key, value)

    return config
