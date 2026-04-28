"""Generate text from a pretrain checkpoint.

Run from the repo root:

    python -m pretrain.inference \\
        --checkpoint checkpoints/pretrain/ckpt_final.pt \\
        --prompt "The quick brown fox" \\
        --max-new-tokens 64 \\
        --temperature 0.8 \\
        --top-k 50

Also exports `pick_device`, `load_model`, `generate`, `EOS_ID` for reuse by
sibling stages (e.g., sft/inference.py).
"""

import argparse
import sys
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

# Allow `python pretrain/inference.py` from repo root in addition to `-m pretrain.inference`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import Transformer


EOS_ID = 50256  # gpt2 BPE end-of-text, matches pretrain/data.py


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" in ckpt:
        # New format (SFT ckpts and pretrain ckpts going forward): explicit model_config.
        model_cfg = ckpt["model_config"]
    elif "model" in ckpt.get("config", {}):
        # Pretrain ckpts: model arch nested under config.
        model_cfg = ckpt["config"]["model"]
    elif "init_from" in ckpt:
        # Older SFT ckpts saved before model_config was embedded: pull arch from
        # the original pretrain checkpoint.
        init_ckpt = torch.load(
            ckpt["init_from"], map_location=device, weights_only=False
        )
        model_cfg = init_ckpt["config"]["model"]
    else:
        raise ValueError(
            f"Cannot infer model architecture from checkpoint {checkpoint_path}: "
            "no 'model_config', 'config.model', or 'init_from' field found."
        )
    model = Transformer(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    eos_id: int,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]
        logits = model(idx_cond)[:, -1, :]

        if repetition_penalty != 1.0:
            # Keskar et al. 2019: scale logits of tokens already in context.
            # Positive logits get divided, negative logits get multiplied — both
            # push the value toward zero / away from being sampled.
            seen = idx_cond
            seen_logits = logits.gather(-1, seen)
            seen_logits = torch.where(
                seen_logits > 0,
                seen_logits / repetition_penalty,
                seen_logits * repetition_penalty,
            )
            logits.scatter_(-1, seen, seen_logits)

        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = torch.where(
                    logits < kth, torch.full_like(logits, float("-inf")), logits
                )

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                # Drop tokens where cumulative prob exceeds top_p, but always keep
                # the top-1 by shifting the mask right.
                drop = cum_probs > top_p
                drop[..., 1:] = drop[..., :-1].clone()
                drop[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.empty_like(logits).scatter_(
                    -1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == eos_id:
            break

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device()
    model, model_cfg = load_model(args.checkpoint, device)
    max_seq_len = model_cfg["max_seq_len"]

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode_ordinary(args.prompt)
    if len(prompt_ids) >= max_seq_len:
        raise ValueError(
            f"prompt has {len(prompt_ids)} tokens, must be < max_seq_len={max_seq_len}"
        )

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=max_seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_id=EOS_ID,
    )

    new_ids = out[0, len(prompt_ids):].tolist()
    completion = enc.decode(new_ids)

    print("=== prompt ===")
    print(args.prompt)
    print("=== completion ===")
    print(completion)


if __name__ == "__main__":
    main()
