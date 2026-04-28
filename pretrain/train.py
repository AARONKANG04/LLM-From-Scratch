"""Pretrain a Transformer LM. All knobs come from configs/pretrain.yaml.

Run from the repo root:

    python -m pretrain.train --config configs/pretrain.yaml
    python -m pretrain.train --resume checkpoints/ckpt_5000.pt
"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# Allow `python pretrain/train.py` from repo root in addition to `-m pretrain.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import Transformer
from pretrain.config import Config
from pretrain.data import PackedTokenDataset


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def resolve_dtype(desired: str, device_type: str) -> torch.dtype:
    if desired == "bfloat16":
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device_type == "cpu":
            return torch.bfloat16
        return torch.float32
    if desired in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.schedule.warmup_steps:
        return cfg.optim.lr * (step + 1) / cfg.schedule.warmup_steps
    if step >= cfg.schedule.max_steps:
        return cfg.optim.min_lr
    progress = (step - cfg.schedule.warmup_steps) / max(
        1, cfg.schedule.max_steps - cfg.schedule.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.optim.min_lr + coeff * (cfg.optim.lr - cfg.optim.min_lr)


def make_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def get_batch(ds: PackedTokenDataset, batch_size: int, device, device_type: str):
    ix = torch.randint(0, len(ds), (batch_size,)).tolist()
    xs, ys = zip(*(ds[i] for i in ix))
    x = torch.stack(xs)
    y = torch.stack(ys)
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate(model, val_ds, cfg, device, device_type, autocast_ctx):
    model.eval()
    losses = []
    for _ in range(cfg.eval.eval_iters):
        x, y = get_batch(val_ds, cfg.train.batch_size, device, device_type)
        with autocast_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    device, device_type = pick_device()
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    dtype = resolve_dtype(cfg.train.dtype, device_type)
    scaler = torch.amp.GradScaler(device_type) if dtype == torch.float16 else None
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if dtype != torch.float32
        else nullcontext()
    )
    print(f"device={device_type}  dtype={dtype}  scaler={'on' if scaler else 'off'}")

    train_ds = PackedTokenDataset(cfg.data.train_bin, cfg.data.block_size)
    val_ds = PackedTokenDataset(cfg.data.val_bin, cfg.data.block_size)

    raw_model = Transformer(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        make_param_groups(raw_model, cfg.optim.weight_decay),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        fused=(device_type == "cuda"),
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    model = raw_model
    if cfg.train.compile and device_type != "mps":
        model = torch.compile(raw_model)

    os.makedirs(cfg.io.out_dir, exist_ok=True)

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=asdict(cfg),
        mode=("online" if cfg.wandb.enabled else "disabled"),
    )

    tokens_per_step = (
        cfg.train.batch_size * cfg.train.grad_accum_steps * cfg.data.block_size
    )
    pbar = tqdm(
        range(start_step, cfg.schedule.max_steps),
        desc="train",
        initial=start_step,
        total=cfg.schedule.max_steps,
    )
    t_log = time.time()
    model.train()

    # Prefetch the first batch so the very first microbatch overlaps with no setup cost.
    next_x, next_y = get_batch(train_ds, cfg.train.batch_size, device, device_type)

    for step in pbar:
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Accumulate loss as a tensor on-device — no per-microbatch sync.
        loss_accum = torch.zeros((), device=device)
        for _ in range(cfg.train.grad_accum_steps):
            x, y = next_x, next_y
            # Kick off the next batch's host->device copy while the GPU is busy.
            next_x, next_y = get_batch(
                train_ds, cfg.train.batch_size, device, device_type
            )
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                ) / cfg.train.grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum = loss_accum + loss.detach()

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.optim.grad_clip
        )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.io.log_interval == 0:
            dt = time.time() - t_log
            tok_s = (
                tokens_per_step * cfg.io.log_interval / dt if dt > 0 else 0.0
            )
            loss_value = loss_accum.item()  # only sync at log time
            pbar.set_postfix(
                loss=f"{loss_value:.4f}", lr=f"{lr:.2e}", tok_s=int(tok_s)
            )
            wandb.log(
                {
                    "train/loss": loss_value,
                    "train/lr": lr,
                    "train/tok_per_sec": tok_s,
                    "train/grad_norm": float(grad_norm),
                },
                step=step,
            )
            t_log = time.time()

        if step > 0 and step % cfg.eval.eval_interval == 0:
            val_loss = evaluate(
                model, val_ds, cfg, device, device_type, autocast_ctx
            )
            tqdm.write(f"step {step} | val_loss {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=step)

        if step > 0 and step % cfg.io.save_interval == 0:
            ckpt_path = os.path.join(cfg.io.out_dir, f"ckpt_{step}.pt")
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
            tqdm.write(f"saved {ckpt_path}")

    final_path = os.path.join(cfg.io.out_dir, "ckpt_final.pt")
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": cfg.schedule.max_steps - 1,
            "config": asdict(cfg),
        },
        final_path,
    )
    tqdm.write(f"saved {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
