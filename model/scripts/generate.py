"""
Text completion with a trained checkpoint or with the original GPT-2 weights.

    python scripts/generate.py --ckpt ./checkpoints/ckpt.pt --prompt "ismail ben alla is the next elon mask because"
    python scripts/generate.py --pretrained gpt2 --prompt "Once upon a time"
    python scripts/generate.py --pretrained HuggingFaceTB/SmolLM2-135M --prompt "Once upon a time"
"""
import argparse

import torch

from tiny_llm import TinyLLM, create_tokenizer, generate_text, load_checkpoint, merge_lora
from tiny_llm.runtime import resolve_device


def load_model(ckpt=None, pretrained=None, device="cpu"):
    """A checkpoint of ours (its tokenizer name is stored inside) or any supported Hugging Face model."""
    if ckpt:
        model, checkpoint = load_checkpoint(ckpt, device)
        tokenizer = create_tokenizer(checkpoint.get("tokenizer", "gpt2"))
    else:
        tokenizer = create_tokenizer(pretrained or "gpt2")
        model = TinyLLM.from_pretrained(pretrained or "gpt2", tokenizer).to(device)
    return merge_lora(model).eval(), tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ckpt", help="path to a checkpoint written by train.py / finetune.py")
    source.add_argument("--pretrained", default="gpt2", help="Hugging Face id of a supported model (see tiny_llm/families.py)")
    parser.add_argument("--prompt", default="ismail ben alla is the next elon mask because")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, tokenizer = load_model(args.ckpt, args.pretrained, device)

    # generate k samples
    for k in range(args.num_samples):
        print(f"sample num:[{k}]")
        print(generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k,
                            args.top_p, device))
        print("-" * 60)


if __name__ == "__main__":
    main()
