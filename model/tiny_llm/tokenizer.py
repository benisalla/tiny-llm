"""
Any Hugging Face tokenizer + the special tokens used to format question/answer samples.

The tokenizer's own bos/eos tokens are kept (Llama tokenizers prepend <s>, generation still stops at the
model's native end of text), our tokens are added on top.
"""
from transformers import AutoTokenizer

PAD, BOS, EOS, QUES, ANS = "<pad>", "<sost>", "<eost>", "<ques>", "<ans>"
SPECIAL_TOKENS = [PAD, BOS, EOS, QUES, ANS]

# end-of-text tokens used by the supported families (base and chat models)
NATIVE_STOP_TOKENS = ["<|endoftext|>", "</s>", "<|im_end|>", "<|end_of_text|>", "<|eot_id|>"]


def create_tokenizer(name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # for GPT-2 this keeps the ids of the first version: <pad>=50257 <sost>=50258 <eost>=50259 <ques>=50260 <ans>=50261
    tokenizer.add_special_tokens({'pad_token': PAD, 'additional_special_tokens': [BOS, EOS, QUES, ANS]})
    # we handle truncation ourselves, silence the "sequence too long" warning
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def stop_token_ids(tokenizer):
    """<eost> for our fine-tuned models + the native end-of-text tokens of the base model."""
    vocab = tokenizer.get_vocab()
    ids = {vocab[t] for t in [EOS] + NATIVE_STOP_TOKENS if t in vocab}
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def format_qa_prompt(question, context=""):
    """The exact text the model sees before it starts writing the answer."""
    return f"{BOS}{context.strip()}{QUES}{question.strip()}{ANS}"


def format_qa_sample(question, answer, context=""):
    """A full training sample: prompt + answer + end of text."""
    return f"{format_qa_prompt(question, context)}{answer.strip()}{EOS}"
