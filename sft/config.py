"""Typed config loader for SFT runs.

Mirrors the sections in configs/sft.yaml. Model architecture is NOT in this
config — it comes from the pretrain checkpoint passed via --init-from, so the
shapes always match the loaded weights.
"""

from dataclasses import dataclass
from typing import Optional

import yaml


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
    epochs: int


@dataclass
class TrainConfig:
    batch_size: int
    grad_accum_steps: int
    max_seq_len: int
    dtype: str
    compile: bool
    seed: int
    num_workers: int


@dataclass
class IOConfig:
    out_dir: str
    log_interval: int
    save_at_epoch_end: bool


@dataclass
class WandbConfig:
    project: str
    run_name: Optional[str]
    enabled: bool


@dataclass
class Config:
    optim: OptimConfig
    schedule: ScheduleConfig
    train: TrainConfig
    io: IOConfig
    wandb: WandbConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            optim=OptimConfig(**raw["optim"]),
            schedule=ScheduleConfig(**raw["schedule"]),
            train=TrainConfig(**raw["train"]),
            io=IOConfig(**raw["io"]),
            wandb=WandbConfig(**raw["wandb"]),
        )
