"""
Chatbot backend: loads an TinyLLM model once and streams answers. No UI code here, so it can be reused
from Streamlit (app.py), a REST API, or a notebook.
"""
import os
import sys
from dataclasses import dataclass

# make `tiny_llm` importable even if the model package was not pip-installed
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_MODEL_DIR))

from tiny_llm import TinyLLM, create_tokenizer, load_checkpoint, merge_lora  # noqa: E402
from tiny_llm.generation import has_chat_template, stream_answer, stream_chat, stream_tokens  # noqa: E402
from tiny_llm.runtime import resolve_device  # noqa: E402

QA, COMPLETE, CHAT = "qa", "complete", "chat"


@dataclass
class GenerationSettings:
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


class ChatEngine:
    def __init__(self, model, tokenizer, device, name, default_mode):
        self.model = merge_lora(model).eval()  # LoRA merged once: no extra cost per token
        self.tokenizer = tokenizer
        self.device = device
        self.name = name
        self.default_mode = default_mode

    @classmethod
    def from_checkpoint(cls, path, device="auto"):
        """A checkpoint written by model/scripts/train.py or finetune.py."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        device = resolve_device(device)
        model, ckpt = load_checkpoint(path, device)
        # fine-tuned checkpoints answer questions, pre-trained ones complete text
        mode = QA if ckpt.get("stage") == "finetune" else COMPLETE
        return cls(model, create_tokenizer(ckpt.get("tokenizer", "gpt2")), device, os.path.basename(path), mode)

    @classmethod
    def from_pretrained(cls, name="gpt2", device="auto"):
        """
        Any supported Hugging Face model (GPT-2, SmolLM2, Qwen2.5, Pythia...) loaded into our from-scratch
        architecture (downloaded on first use). Instruct models get the chat mode.
        """
        device = resolve_device(device)
        tokenizer = create_tokenizer(name)
        model = TinyLLM.from_pretrained(name, tokenizer).to(device)
        return cls(model, tokenizer, device, name, CHAT if has_chat_template(tokenizer) else COMPLETE)

    @property
    def modes(self):
        """Modes this model can be used in."""
        return [QA, COMPLETE] + ([CHAT] if has_chat_template(self.tokenizer) else [])

    @property
    def num_params(self):
        return self.model.get_num_params()

    def stream(self, message, context="", mode=None, settings=None, history=None):
        """
        Yield the reply piece by piece.
        history: previous turns [{"role": "user"|"assistant", "content": ...}], only used in chat mode
        (the Q&A and completion modes answer each message independently).
        """
        settings = settings or GenerationSettings()
        mode = mode or self.default_mode
        kwargs = dict(max_new_tokens=settings.max_new_tokens, temperature=settings.temperature,
                      top_k=settings.top_k, top_p=settings.top_p, device=self.device)
        if mode == QA:
            return stream_answer(self.model, self.tokenizer, message, context, **kwargs)
        if mode == COMPLETE:
            return stream_tokens(self.model, self.tokenizer, message, **kwargs)
        if mode == CHAT:
            if not has_chat_template(self.tokenizer):
                raise ValueError(f"{self.name} has no chat template, use an instruct model for chat mode")
            messages = [{"role": "system", "content": f"Answer using this context:\n{context.strip()}"}] \
                if context.strip() else []
            messages += list(history or []) + [{"role": "user", "content": message}]
            return stream_chat(self.model, self.tokenizer, messages, **kwargs)
        raise ValueError(f"unknown mode {mode!r}, expected one of {QA!r}, {COMPLETE!r}, {CHAT!r}")

    def reply(self, message, context="", mode=None, settings=None, history=None):
        return "".join(self.stream(message, context, mode, settings, history)).strip()
