"""Generate from an SFT checkpoint using the Alpaca instruction template.

Run from the repo root:

    python -m sft.inference \\
        --checkpoint checkpoints/sft/ckpt_sft_final.pt \\
        --instruction "What is a black hole?"

    python -m sft.inference \\
        --checkpoint checkpoints/sft/ckpt_sft_final.pt \\
        --instruction "Translate to French" \\
        --input "I love programming."
"""

import argparse
import sys
from pathlib import Path

import tiktoken
import torch

# Allow `python sft/inference.py` from repo root in addition to `-m sft.inference`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrain.inference import EOS_ID, generate, load_model, pick_device
from sft.data import PROMPT_NO_INPUT, PROMPT_WITH_INPUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instruction", required=True,
                        help="The instruction the model should follow.")
    parser.add_argument("--input", default="",
                        help="Optional input/context the instruction operates on.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device()
    model, model_cfg = load_model(args.checkpoint, device)
    max_seq_len = model_cfg["max_seq_len"]

    template = PROMPT_WITH_INPUT if args.input else PROMPT_NO_INPUT
    prompt = template.format(instruction=args.instruction, input=args.input)

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode_ordinary(prompt)
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
    if new_ids and new_ids[-1] == EOS_ID:
        new_ids = new_ids[:-1]
    response = enc.decode(new_ids)

    print("=== instruction ===")
    print(args.instruction)
    if args.input:
        print("=== input ===")
        print(args.input)
    print("=== response ===")
    print(response)


if __name__ == "__main__":
    main()
