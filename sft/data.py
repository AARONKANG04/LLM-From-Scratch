"""Alpaca SFT dataset.

Library mode: import AlpacaDataset and collate_fn for the training loop.
Script mode: `python -m sft.data` runs sanity checks on the pipeline.
"""

import tiktoken
import torch
from torch.utils.data import Dataset


EOS_TOKEN_ID = 50256
IGNORE_INDEX = -100


PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


class AlpacaDataset(Dataset):
    def __init__(self, max_seq_len: int = 1024):
        from datasets import load_dataset

        enc = tiktoken.get_encoding("gpt2")
        ds = load_dataset("tatsu-lab/alpaca", split="train")

        self.examples: list[tuple[list[int], int]] = []
        dropped = 0
        for row in ds:
            template = PROMPT_WITH_INPUT if row["input"] else PROMPT_NO_INPUT
            prompt = template.format(
                instruction=row["instruction"],
                input=row["input"],
            )
            prompt_ids = enc.encode_ordinary(prompt)
            response_ids = enc.encode_ordinary(row["output"]) + [EOS_TOKEN_ID]
            full = prompt_ids + response_ids
            if len(full) > max_seq_len:
                dropped += 1
                continue
            self.examples.append((full, len(prompt_ids)))

        print(
            f"AlpacaDataset: kept {len(self.examples)}, dropped {dropped} "
            f"(>{max_seq_len} tokens)"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, prompt_len = self.examples[i]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(
    batch,
    pad_token_id: int = EOS_TOKEN_ID,
    ignore_index: int = IGNORE_INDEX,
):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), ignore_index, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
    return {"input_ids": input_ids, "labels": labels}


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    ds = AlpacaDataset()

    print("\n--- per-example sanity check ---")
    for i in [0, 1, 2]:
        ids, prompt_len = ds.examples[i]
        item = ds[i]
        masked = (item["labels"] == IGNORE_INDEX).sum().item()
        assert masked == prompt_len, (masked, prompt_len)
        print(f"\n[example {i}] total={len(ids)}  prompt_len={prompt_len}")
        print("  PROMPT:", enc.decode(ids[:prompt_len])[:200].replace("\n", "\\n"))
        print("  RESPONSE:", enc.decode(ids[prompt_len:])[:200].replace("\n", "\\n"))

    print("\n--- collate sanity check ---")
    indices = [0, 1, 2, 3]
    items = [ds[i] for i in indices]
    batch = collate_fn(items)
    real_lens = [len(ds.examples[i][0]) for i in indices]
    B, T = batch["input_ids"].shape
    assert batch["labels"].shape == (B, T)
    assert B == 4
    assert T == max(real_lens)
    for r, n in enumerate(real_lens):
        # past real content: input is pad-EOS, labels are -100
        assert (batch["input_ids"][r, n:] == EOS_TOKEN_ID).all()
        assert (batch["labels"][r, n:] == IGNORE_INDEX).all()
        # within real content: at least one supervised position
        assert (batch["labels"][r, :n] != IGNORE_INDEX).any()
    print(f"batch input_ids: {tuple(batch['input_ids'].shape)}")
    print(f"batch labels:    {tuple(batch['labels'].shape)}")
    supervised = sum(
        (batch["labels"][r, :n] != IGNORE_INDEX).sum().item()
        for r, n in enumerate(real_lens)
    )
    print(f"supervised tokens in batch: {supervised}")
