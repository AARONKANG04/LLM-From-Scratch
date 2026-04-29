"""Typed config loader for pretrain runs.

Mirrors the sections in configs/pretrain.yaml. A typo in a key fails fast at
load time instead of deep inside training.
"""

from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    dropout: float


@dataclass
class DataConfig:
    train_bin: str
    val_bin: str
    block_size: int


@dataclass
class OptimConfig:
    lr: float
    min_lr: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float


@dataclass
class ScheduleConfig:
    warmup_steps: int
    max_steps: int
    anchor_step: int = 0
    re_peak_lr: Optional[float] = None


@dataclass
class TrainConfig:
    batch_size: int
    grad_accum_steps: int
    dtype: str
    compile: bool
    seed: int


@dataclass
class EvalConfig:
    eval_interval: int
    eval_iters: int


@dataclass
class IOConfig:
    data_dir: str
    out_dir: str
    log_interval: int
    save_interval: int


@dataclass
class WandbConfig:
    project: str
    run_name: Optional[str]
    enabled: bool


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    schedule: ScheduleConfig
    train: TrainConfig
    eval: EvalConfig
    io: IOConfig
    wandb: WandbConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**raw["model"]),
            data=DataConfig(**raw["data"]),
            optim=OptimConfig(**raw["optim"]),
            schedule=ScheduleConfig(**raw["schedule"]),
            train=TrainConfig(**raw["train"]),
            eval=EvalConfig(**raw["eval"]),
            io=IOConfig(**raw["io"]),
            wandb=WandbConfig(**raw["wandb"]),
        )
