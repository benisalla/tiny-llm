import json
import os
from random import Random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from tiny_llm.tokenizer import ANS, format_qa_sample

IGNORE_INDEX = -1  # targets with this value don't contribute to the loss


def load_bin(data_dir, split):
    """Token ids stored as a flat uint16 binary file (see scripts/prepare_data.py)."""
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode='r')


def get_batch(data, batch_size, block_size, device):
    """Sample `batch_size` random windows from a flat token array. y is x shifted by one."""
    assert len(data) > block_size, f"dataset has {len(data)} tokens, need more than block_size={block_size}"
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if torch.device(device).type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def _field(sample, name):
    # the original dataset stores its keys reversed ("txetnoc", "noitseuq", "rewsna"), accept both
    return sample.get(name, sample.get(name[::-1], ""))


def load_qa_samples(path):
    """JSON list of {"context": ..., "question": ..., "answer": ...} -> list of formatted strings."""
    with open(path, 'r', encoding='utf-8') as json_file:
        raw_data = json.load(json_file)
    return [format_qa_sample(_field(s, "question"), _field(s, "answer"), _field(s, "context")) for s in raw_data]


class BatchLoader:
    """
    Infinite iterator of padded (x, y) batches for question/answer fine-tuning.

    Samples are sorted by length and grouped so that each batch holds about `toks_in_batch` tokens
    (little padding waste). Training batches are shuffled every epoch.

    Args:
        data_folder (str): folder containing `{split}.json`.
        split (str): "train", "val" or "test".
        tokenizer: tokenizer with our special tokens (see tiny_llm.tokenizer).
        toks_in_batch (int): token budget per batch.
        max_len (int): samples longer than this (in tokens) are dropped.
        answer_only (bool): if True, targets before `<ans>` are ignored (learn to answer, not to ask).
    """

    def __init__(self, data_folder, split, tokenizer, toks_in_batch, device="cpu", max_len=400,
                 answer_only=True, seed=0):
        self.toks_in_batch = toks_in_batch
        self.training = (split == "train")
        self.tokenizer = tokenizer
        self.device = device
        self.answer_only = answer_only
        self.rng = Random(seed)
        self.ans_id = tokenizer.convert_tokens_to_ids(ANS)

        texts = load_qa_samples(os.path.join(data_folder, f"{split}.json"))
        encoded = [tokenizer.encode(t) for t in texts]
        # filter our data and pre-sort by lengths
        self.data = sorted((ids for ids in encoded if 2 < len(ids) <= max_len), key=len)
        assert self.data, f"no usable samples in {split}.json"
        self.create_batches()

    def create_batches(self):
        self.batches = []
        batch = []
        for ids in self.data:  # data is sorted, so the last sample of a batch is the longest
            if batch and len(ids) * (len(batch) + 1) > self.toks_in_batch:
                self.batches.append(batch)
                batch = []
            batch.append(ids)
        if batch:
            self.batches.append(batch)
        if self.training:
            self.rng.shuffle(self.batches)
        self.n_batches = len(self.batches)
        self.current = -1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batches)

    def __next__(self):
        self.current += 1
        if self.current >= self.n_batches:  # new epoch
            self.create_batches()
            self.current = 0
        seqs = [torch.LongTensor(ids) for ids in self.batches[self.current]]

        input_ids = pad_sequence(seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        targets = pad_sequence(seqs, batch_first=True, padding_value=IGNORE_INDEX)
        if self.answer_only:
            for row, ids in enumerate(seqs):
                ans_pos = (ids == self.ans_id).nonzero()
                if len(ans_pos):
                    targets[row, :ans_pos[0, 0] + 1] = IGNORE_INDEX  # keep only tokens after <ans>

        x_batch = input_ids[:, :-1].to(self.device)
        y_batch = targets[:, 1:].to(self.device)
        return x_batch, y_batch
