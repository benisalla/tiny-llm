import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import CHAT, COMPLETE, QA, ChatEngine, GenerationSettings  # noqa: E402
from tiny_llm import ModelConfig, TinyLLM, save_checkpoint  # noqa: E402


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    model = TinyLLM(ModelConfig(vocab_size=50304, block_size=64, n_layer=2, n_head=2, n_embd=32, lora_rank=2))
    path = tmp_path / "finetuned.pt"
    save_checkpoint(str(path), model, extra={"stage": "finetune"})
    return str(path)


@pytest.fixture
def engine(ckpt):
    try:
        return ChatEngine.from_checkpoint(ckpt, device="cpu")
    except OSError as e:
        pytest.skip(f"GPT-2 tokenizer unavailable: {e}")


def test_finetuned_checkpoint_defaults_to_qa(engine):
    assert engine.default_mode == QA
    assert engine.name == "finetuned.pt"
    assert all(m.lora_merged for m in engine.model.modules() if getattr(m, "rank", 0) > 0)


@pytest.mark.parametrize("mode", [QA, COMPLETE])
def test_stream_yields_text(engine, mode):
    torch.manual_seed(0)
    settings = GenerationSettings(max_new_tokens=8, temperature=1.0)
    pieces = list(engine.stream("What is the capital of France?", "Paris is in France.", mode, settings))
    assert all(isinstance(p, str) for p in pieces)
    assert len(pieces) <= 8


def test_greedy_reply_is_deterministic(engine):
    settings = GenerationSettings(max_new_tokens=6, temperature=0)
    assert engine.reply("hello", settings=settings) == engine.reply("hello", settings=settings)


def test_long_prompt_is_truncated_to_the_context_window(engine):
    settings = GenerationSettings(max_new_tokens=4, temperature=0)
    engine.reply("word " * 500, mode=COMPLETE, settings=settings)  # must not raise


def test_bad_mode_and_missing_checkpoint(engine, tmp_path):
    with pytest.raises(ValueError):
        engine.stream("hi", mode="poetry")
    with pytest.raises(FileNotFoundError):
        ChatEngine.from_checkpoint(str(tmp_path / "nope.pt"))


def test_chat_mode_needs_a_chat_template(engine):
    # the GPT-2 tokenizer has no chat template: chat is not offered and fails clearly if forced
    assert engine.modes == [QA, COMPLETE]
    with pytest.raises(ValueError, match="chat template"):
        engine.stream("hi", mode=CHAT)
