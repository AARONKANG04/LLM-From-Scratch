"""Pretrain dataset.

Library mode: import PackedTokenDataset for the training loop.
Script mode: `python -m pretrain.data` runs the FineWeb-Edu prep pipeline.
"""

import os
from multiprocessing import cpu_count

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset


EOS_TOKEN_ID = 50256


class PackedTokenDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = torch.from_numpy(
            self.data[idx : idx + self.block_size + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def _tokenize(example):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(example["text"])
    ids.append(EOS_TOKEN_ID)
    return {"ids": ids, "len": len(ids)}


def prepare(
    out_dir: str = "data",
    val_tokens: int = 50_000_000,
    num_proc: int = max(1, cpu_count() - 2),
    n_shards: int = 1024,
):
    from datasets import load_dataset
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        num_proc=num_proc,
    )

    tokenized = ds.map(
        _tokenize,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="tokenizing",
    )

    total = int(np.sum(tokenized["len"], dtype=np.int64))
    val_n = min(val_tokens, total // 200)
    train_n = total - val_n
    print(f"Total tokens: {total:,}  (train: {train_n:,}, val: {val_n:,})")

    train_arr = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(train_n,))
    val_arr = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(val_n,))

    val_idx = 0
    train_idx = 0
    val_full = False
    for s in tqdm(range(n_shards), desc="writing"):
        shard = tokenized.shard(num_shards=n_shards, index=s, contiguous=True).with_format("numpy")
        chunk = np.concatenate(shard["ids"]).astype(np.uint16)
        if not val_full:
            take = min(len(chunk), val_n - val_idx)
            val_arr[val_idx : val_idx + take] = chunk[:take]
            val_idx += take
            chunk = chunk[take:]
            val_full = val_idx >= val_n
        if len(chunk):
            train_arr[train_idx : train_idx + len(chunk)] = chunk
            train_idx += len(chunk)

    train_arr.flush()
    val_arr.flush()
    print(f"Wrote {val_idx:,} val tokens to {val_path}")
    print(f"Wrote {train_idx:,} train tokens to {train_path}")


if __name__ == "__main__":
    prepare()
