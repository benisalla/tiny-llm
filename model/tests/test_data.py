import json

import numpy as np
import torch

from tiny_llm import format_qa_prompt, format_qa_sample
from tiny_llm.data import IGNORE_INDEX, BatchLoader, get_batch, load_bin, load_qa_samples


def test_get_batch_targets_are_shifted_inputs():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, batch_size=4, block_size=16, device="cpu")
    assert x.shape == y.shape == (4, 16) and x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y - x, torch.ones_like(x))


def test_load_bin(bin_dir):
    data = load_bin(str(bin_dir), "train")
    assert data.dtype == np.uint16 and len(data) == 5000


def test_qa_format():
    assert format_qa_prompt(" q ", " c ") == "<sost>c<ques>q<ans>"
    assert format_qa_sample("q", "a", "c") == "<sost>c<ques>q<ans>a<eost>"


def test_reversed_keys_are_supported(tmp_path):
    """The original dataset stored its keys reversed ("txetnoc", "noitseuq", "rewsna")."""
    path = tmp_path / "train.json"
    path.write_text(json.dumps([{"txetnoc": "c", "noitseuq": "q", "rewsna": "a"}]))
    assert load_qa_samples(path) == ["<sost>c<ques>q<ans>a<eost>"]


def test_batch_loader(qa_dir, tokenizer):
    loader = BatchLoader(str(qa_dir), "train", tokenizer, toks_in_batch=64, answer_only=True)
    pad_id = tokenizer.pad_token_id
    ans_id = tokenizer.convert_tokens_to_ids("<ans>")
    eos_id = tokenizer.convert_tokens_to_ids("<eost>")

    seen = 0
    for _ in range(len(loader) * 2 + 1):  # more than one epoch: the iterator must never stop
        x, y = next(loader)
        assert x.shape == y.shape
        assert x.shape[0] * (x.shape[1] + 1) <= 64 or x.shape[0] == 1  # token budget
        for row in range(x.shape[0]):
            valid = y[row] != IGNORE_INDEX
            assert valid.any()
            # padding is never a target, the last target is <eost>
            assert not (y[row][valid] == pad_id).any()
            assert y[row][valid][-1].item() == eos_id
            # answer_only: the first learned token comes right after <ans>
            first = valid.nonzero()[0, 0]
            assert x[row, first].item() == ans_id
        seen += x.shape[0]
    assert seen > len(loader.data)


def test_batch_loader_full_sequence_loss(qa_dir, tokenizer):
    loader = BatchLoader(str(qa_dir), "val", tokenizer, toks_in_batch=64, answer_only=False)
    x, y = next(loader)
    # without answer masking, every non padded position is a target
    assert (y[:, 0] != IGNORE_INDEX).all()


def test_special_token_ids_are_unchanged_for_gpt2(tokenizer):
    """Checkpoints fine-tuned with the first version of the repo rely on these ids."""
    ids = tokenizer.convert_tokens_to_ids(["<pad>", "<sost>", "<eost>", "<ques>", "<ans>"])
    assert ids == [50257, 50258, 50259, 50260, 50261]
    assert tokenizer.encode("<sost>c<ques>q<ans>a<eost>")[0] == 50258  # one id per special token
    assert tokenizer.eos_token == "<|endoftext|>"  # the native end of text is kept
