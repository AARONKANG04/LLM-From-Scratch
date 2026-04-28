"""SFT a Transformer LM on Alpaca. All knobs come from configs/sft.yaml.

Run from the repo root:

    python -m sft.train --config configs/sft.yaml \\
        --init-from checkpoints/pretrain/ckpt_final.pt

    python -m sft.train --config configs/sft.yaml \\
        --init-from checkpoints/pretrain/ckpt_final.pt \\
        --resume checkpoints/sft/ckpt_epoch_1.pt
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python sft/train.py` from repo root in addition to `-m sft.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import Transformer
from sft.config import Config
from sft.data import AlpacaDataset, collate_fn


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


def get_lr(step: int, cfg: Config, max_steps: int) -> float:
    if step < cfg.schedule.warmup_steps:
        return cfg.optim.lr * (step + 1) / cfg.schedule.warmup_steps
    if step >= max_steps:
        return cfg.optim.min_lr
    progress = (step - cfg.schedule.warmup_steps) / max(
        1, max_steps - cfg.schedule.warmup_steps
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


def compute_loss_sum(model, batch, device, device_type, autocast_ctx):
    """Returns the sum-reduced cross-entropy over supervised positions.

    Caller divides by the total non-pad token count across the entire
    accumulation window — see the gradient-accumulation loss correction at
    https://huggingface.co/blog/gradient_accumulation.
    """
    nonblocking = device_type == "cuda"
    input_ids = batch["input_ids"].to(device, non_blocking=nonblocking)
    labels = batch["labels"].to(device, non_blocking=nonblocking)
    with autocast_ctx:
        logits = model(input_ids)
        # Shift so logit at position t predicts token at t+1.
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(
            shift_logits, shift_labels,
            ignore_index=-100, reduction="sum",
        )
    return loss


def save_ckpt(path, raw_model, optimizer, step, epoch, cfg, model_cfg, init_from):
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": asdict(cfg),
            "model_config": model_cfg,  # arch from --init-from, embedded so the SFT ckpt is self-contained
            "init_from": init_from,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--init-from", required=True,
                        help="Pretrain checkpoint to initialize weights from")
    parser.add_argument("--resume", default=None,
                        help="Optional SFT checkpoint to resume training from")
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

    init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
    model_cfg = init_ckpt["config"]["model"]
    raw_model = Transformer(**model_cfg).to(device)
    raw_model.load_state_dict(init_ckpt["model"])
    print(f"initialized from {args.init_from}")
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    ds = AlpacaDataset(max_seq_len=cfg.train.max_seq_len)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device_type == "cuda"),
    )
    steps_per_epoch = len(loader) // cfg.train.grad_accum_steps
    max_steps = steps_per_epoch * cfg.schedule.epochs
    print(
        f"examples={len(ds)}  micro_batches/epoch={len(loader)}  "
        f"steps/epoch={steps_per_epoch}  max_steps={max_steps}"
    )

    optimizer = torch.optim.AdamW(
        make_param_groups(raw_model, cfg.optim.weight_decay),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        fused=(device_type == "cuda"),
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"]  # next epoch to run (0-indexed)
        print(f"resumed from {args.resume} starting at epoch {start_epoch}")

    model = raw_model
    if cfg.train.compile and device_type != "mps":
        # dynamic=True compiles with a symbolic T axis so variable-length SFT
        # batches don't trigger per-shape recompiles.
        model = torch.compile(raw_model, dynamic=True)

    os.makedirs(cfg.io.out_dir, exist_ok=True)

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=asdict(cfg),
        mode=("online" if cfg.wandb.enabled else "disabled"),
    )

    step = start_epoch * steps_per_epoch
    pbar = tqdm(total=max_steps, initial=step, desc="sft")
    t_log = time.time()
    tokens_since_log = 0
    model.train()

    for epoch in range(start_epoch, cfg.schedule.epochs):
        loader_iter = iter(loader)
        for _ in range(steps_per_epoch):
            lr = get_lr(step, cfg, max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Gather the whole accumulation window so we can normalize the
            # loss by total non-pad tokens (see HF gradient_accumulation fix).
            micro = [next(loader_iter) for _ in range(cfg.train.grad_accum_steps)]
            n_items = sum(
                (b["labels"][:, 1:] != -100).sum().item() for b in micro
            )
            n_items = max(n_items, 1)
            tokens_since_log += sum(b["input_ids"].numel() for b in micro)

            loss_sum = torch.zeros((), device=device)
            for batch in micro:
                loss_b = compute_loss_sum(
                    model, batch, device, device_type, autocast_ctx
                )
                scaled = loss_b / n_items
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                loss_sum = loss_sum + loss_b.detach()

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
                tok_s = tokens_since_log / dt if dt > 0 else 0.0
                loss_value = (loss_sum / n_items).item()
                pbar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{lr:.2e}",
                    tok_s=int(tok_s),
                    epoch=epoch,
                )
                wandb.log(
                    {
                        "train/loss": loss_value,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "train/tok_per_sec": tok_s,
                        "train/epoch": epoch,
                    },
                    step=step,
                )
                t_log = time.time()
                tokens_since_log = 0

            step += 1
            pbar.update(1)

        if cfg.io.save_at_epoch_end:
            ckpt_path = os.path.join(
                cfg.io.out_dir, f"ckpt_epoch_{epoch + 1}.pt"
            )
            save_ckpt(
                ckpt_path, raw_model, optimizer, step - 1, epoch + 1, cfg,
                model_cfg, args.init_from,
            )
            tqdm.write(f"saved {ckpt_path}")

    final_path = os.path.join(cfg.io.out_dir, "ckpt_sft_final.pt")
    save_ckpt(
        final_path, raw_model, optimizer, max_steps - 1, cfg.schedule.epochs,
        cfg, model_cfg, args.init_from,
    )
    tqdm.write(f"saved {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
