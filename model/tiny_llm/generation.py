"""High level inference helpers: text completion, question answering and chat, with streaming."""
import torch

from tiny_llm.tokenizer import PAD, format_qa_prompt, stop_token_ids


def _banned_ids(tokenizer, model):
    """Padding and the unused rows added to round the vocabulary up must never be sampled."""
    banned = [tokenizer.convert_tokens_to_ids(PAD)]
    banned += list(range(len(tokenizer), model.config.vocab_size))
    return banned


def stream_tokens(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95,
                  device=None, stop_at_eos=True, add_special_tokens=True):
    """
    Yield the generated text piece by piece (decoded incrementally so multi-byte characters
    never show up half written). The prompt itself is not yielded.
    """
    device = device or next(model.parameters()).device
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    # keep room for the answer inside the context window
    max_prompt = max(1, model.config.block_size - max_new_tokens)
    ids = ids[-max_prompt:]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    stop_ids = stop_token_ids(tokenizer)
    generated, emitted = [], ""
    for tok in model.stream_generate(x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                                     eos_token_id=stop_ids if stop_at_eos else None,
                                     banned_token_ids=_banned_ids(tokenizer, model)):
        tok = tok.item()
        if stop_at_eos and tok in stop_ids:
            break
        generated.append(tok)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if text.endswith("�"):  # incomplete utf-8 character, wait for the next token
            continue
        if len(text) > len(emitted):
            yield text[len(emitted):]
            emitted = text


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=100, top_p=None,
                  device=None):
    """Complete `prompt` and return prompt + completion."""
    return prompt + "".join(stream_tokens(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p,
                                          device))


def stream_answer(model, tokenizer, question, context="", **kwargs):
    """Question answering with a fine-tuned model (format: <sost>context<ques>question<ans>answer<eost>)."""
    return stream_tokens(model, tokenizer, format_qa_prompt(question, context), **kwargs)


def answer_me(model, tokenizer, question, context="", **kwargs):
    return "".join(stream_answer(model, tokenizer, question, context, **kwargs)).strip()


def has_chat_template(tokenizer):
    return bool(getattr(tokenizer, "chat_template", None))


def stream_chat(model, tokenizer, messages, **kwargs):
    """
    Chat with an instruct model (SmolLM2-Instruct, Qwen2.5-Instruct, TinyLlama-Chat...) using its own
    chat template. messages = [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # the template already contains the special tokens (e.g. <s>), don't add them twice
    return stream_tokens(model, tokenizer, prompt, add_special_tokens=False, **kwargs)
