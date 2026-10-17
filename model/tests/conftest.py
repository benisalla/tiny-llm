import json

import numpy as np
import pytest
import torch

from tiny_llm import ModelConfig, TinyLLM


def tiny_config(**overrides):
    args = dict(vocab_size=50304, block_size=64, n_layer=2, n_head=4, n_embd=32, drop_rate=0.0)
    args.update(overrides)
    return ModelConfig(**args)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return TinyLLM(tiny_config(vocab_size=97)).eval()


@pytest.fixture(scope="session")
def tokenizer():
    try:
        from tiny_llm import create_tokenizer
        return create_tokenizer("gpt2")
    except Exception as e:  # no network and no cache
        pytest.skip(f"GPT-2 tokenizer unavailable: {e}")


QA_SAMPLES = [
    {"context": "Paris is the capital of France.", "question": "What is the capital of France?", "answer": "Paris"},
    {"context": "The sky is blue.", "question": "What color is the sky?", "answer": "Blue"},
    {"context": "Ismail loves neural networks.", "question": "What does Ismail love?", "answer": "Neural networks"},
    {"context": "", "question": "How many legs does a cat have?", "answer": "Four"},
]


@pytest.fixture
def qa_dir(tmp_path):
    for split in ("train", "val"):
        (tmp_path / f"{split}.json").write_text(json.dumps(QA_SAMPLES * 3))
    return tmp_path


@pytest.fixture
def bin_dir(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 50257, size=5000, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    return tmp_path
