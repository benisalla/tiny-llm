# Inference

## Command line (from `model/`)

```bash
python scripts/generate.py --ckpt ./checkpoints/ckpt.pt --prompt "Once upon a time"           # our checkpoint
python scripts/generate.py --pretrained HuggingFaceTB/SmolLM2-135M --prompt "Once upon a time"  # any supported LLM
python scripts/ask.py --ckpt ./checkpoints/finetuned.pt --context "..." --question "..."        # fine-tuned Q&A
```

## Python

```python
from tiny_llm import TinyLLM, create_tokenizer, generate_text, answer_me, stream_chat, load_checkpoint

tokenizer = create_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
model = TinyLLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", tokenizer).eval()
for piece in stream_chat(model, tokenizer, [{"role": "user", "content": "Hi!"}], max_new_tokens=50):
    print(piece, end="")
```

| function | use |
| --- | --- |
| `generate_text` / `stream_tokens` | continue a prompt (base models) |
| `answer_me` / `stream_answer` | `<sost>context<ques>question<ans>` format (models fine-tuned with `finetune.py`) |
| `stream_chat` | instruct models, with their own chat template |

Sampling: `temperature` (0 = greedy), `top_k`, `top_p`. Generation stops at `<eost>` or the model's own
end-of-text token. Padding and unused vocabulary rows are never sampled.

**KV cache:** each new token only computes its own keys/values and attends to the cached ones
(`TinyLLM.stream_generate(use_cache=True)`, the default). Once the context window is full, generation
falls back to re-reading the last `block_size` tokens, exactly like the no-cache version.

## Chatbot

```bash
streamlit run chatbot/app.py
TINY_LLM_CKPT=model/checkpoints/finetuned.pt streamlit run chatbot/app.py
```

`chatbot/engine.py` (backend, no UI code) loads a checkpoint or a Hugging Face model once and streams
replies; `chatbot/app.py` (Streamlit) is the interface: model picker, chat / Q&A / completion modes,
optional context, sampling settings.
