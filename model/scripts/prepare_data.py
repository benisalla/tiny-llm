"""
Turn a folder of .txt files (e.g. OpenWebText) into train.bin / val.bin / test.bin token streams.
Each document is GPT-2 tokenized (r50k_base) and followed by <|endoftext|>; ids are stored as uint16.

    python scripts/prepare_data.py --src ./raw_txt --dst ./data/openwebtext --val_ratio 0.1 --test_ratio 0.1
"""
import argparse
import os
import random

import numpy as np
import tiktoken


def write_split(docs, enc, path):
    eot = enc.eot_token
    # first pass: count tokens, second pass: write them into a memmap (keeps memory flat)
    lengths = [len(ids) + 1 for ids in enc.encode_ordinary_batch(docs)]
    total = int(np.sum(lengths, dtype=np.uint64))
    arr = np.memmap(path, dtype=np.uint16, mode='w+', shape=(total,))
    pos = 0
    for ids in enc.encode_ordinary_batch(docs):
        ids.append(eot)
        arr[pos:pos + len(ids)] = ids
        pos += len(ids)
    arr.flush()
    print(f"✅ {path}: {len(docs)} documents, {total:,} tokens")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="folder with .txt files (searched recursively)")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)

    files = sorted(os.path.join(root, f) for root, _, names in os.walk(args.src) for f in names if f.endswith(".txt"))
    assert files, f"no .txt files found in {args.src}"
    random.Random(args.seed).shuffle(files)

    n_val = max(1, int(len(files) * args.val_ratio))
    n_test = int(len(files) * args.test_ratio)
    splits = {"val": files[:n_val], "test": files[n_val:n_val + n_test], "train": files[n_val + n_test:]}
    assert splits["train"], "not enough files for a train split"

    enc = tiktoken.get_encoding("r50k_base")
    os.makedirs(args.dst, exist_ok=True)
    for split, paths in splits.items():
        if not paths:
            continue
        docs = []
        for p in paths:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                docs.append(f.read())
        write_split(docs, enc, os.path.join(args.dst, f"{split}.bin"))


if __name__ == "__main__":
    main()
