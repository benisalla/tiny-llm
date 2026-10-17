"""Drive the real Streamlit UI headlessly: load a checkpoint, send a message, get a reply."""
import os

import pytest
import torch
from streamlit.testing.v1 import AppTest

from tiny_llm import ModelConfig, TinyLLM, save_checkpoint

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = TinyLLM(ModelConfig(vocab_size=50304, block_size=64, n_layer=1, n_head=2, n_embd=32))
    path = tmp_path / "finetuned.pt"
    save_checkpoint(str(path), model, extra={"stage": "finetune"})
    monkeypatch.setenv("TINY_LLM_CKPT", str(path))
    monkeypatch.chdir(os.path.dirname(APP))  # app.py imports `engine` from its own folder
    at = AppTest.from_file(APP, default_timeout=120).run()
    if at.exception:
        pytest.fail(str(at.exception))
    return at


def test_app_loads_checkpoint_in_qa_mode(app):
    assert not app.error
    assert app.sidebar.radio[1].value == "qa"
    assert "finetuned.pt" in app.sidebar.caption[0].value


def test_chat_roundtrip(app):
    app.chat_input[0].set_value("What is the capital of France?").run()
    assert not app.exception
    roles = [m.name for m in app.chat_message]
    assert roles == ["user", "assistant"]
    assert len(app.session_state.messages) == 2
    app.sidebar.button[0].click().run()  # clear chat
    assert app.session_state.messages == []
