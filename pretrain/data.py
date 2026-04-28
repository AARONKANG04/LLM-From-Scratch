"""Pretrain dataset.

Library mode: import PackedTokenDataset for the training loop.
Script mode: `python -m pretrain.data` runs the FineWeb-Edu prep pipeline.

Prep streams documents through a multiprocessing pool of tiktoken encoders
and writes uint16 tokens directly to data/{train,val}.bin. We avoid
`Dataset.map()` because it caches its output as int64 (8 bytes/token) — for
the 10BT sample that would balloon to ~80GB of intermediate Arrow on top of
the 28GB raw-text Arrow and 28GB of parquets, blowing past most pod volumes.
"""

import os
from multiprocessing import Pool, cpu_count

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


# Per-worker encoder, populated by Pool initializer.
_enc = None


def _worker_init():
    global _enc
    _enc = tiktoken.get_encoding("gpt2")


def _tokenize_text(text: str):
    ids = _enc.encode_ordinary(text)
    ids.append(EOS_TOKEN_ID)
    return ids


def prepare(
    out_dir: str = "data",
    val_tokens: int = 50_000_000,
    num_proc: int = None,
    batch_size: int = 2048,
):
    from datasets import load_dataset
    from tqdm import tqdm

    if num_proc is None:
        num_proc = max(1, cpu_count() - 2)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        num_proc=num_proc,
    )
    print(f"Loaded {len(ds):,} documents; tokenizing with num_proc={num_proc}")

    val_count = 0
    train_count = 0
    val_full = False

    val_f = open(val_path, "wb")
    train_f = open(train_path, "wb")
    pool = Pool(num_proc, initializer=_worker_init)
    try:
        pbar = tqdm(total=len(ds), desc="tokenize", unit="doc")
        batch = []

        def flush(batch):
            nonlocal val_count, train_count, val_full
            if not batch:
                return
            results = pool.map(_tokenize_text, batch)
            ids = np.concatenate(
                [np.asarray(r, dtype=np.uint16) for r in results]
            )
            if not val_full:
                take = min(len(ids), val_tokens - val_count)
                val_f.write(ids[:take].tobytes())
                val_count += take
                ids = ids[take:]
                val_full = val_count >= val_tokens
            if len(ids):
                train_f.write(ids.tobytes())
                train_count += len(ids)

        for ex in ds:
            batch.append(ex["text"])
            if len(batch) >= batch_size:
                flush(batch)
                pbar.update(len(batch))
                batch = []
        if batch:
            flush(batch)
            pbar.update(len(batch))
        pbar.close()
    finally:
        pool.close()
        pool.join()
        val_f.close()
        train_f.close()

    print(f"Wrote {val_count:,} val tokens to {val_path}")
    print(f"Wrote {train_count:,} train tokens to {train_path}")


if __name__ == "__main__":
    prepare()
