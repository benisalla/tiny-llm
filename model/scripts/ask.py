"""
Ask a question to a fine-tuned checkpoint (see scripts/finetune.py).

    python scripts/ask.py --ckpt ./checkpoints/finetuned.pt --question "Who wrote the paper?" \
        --context "The paper was written by Ismail Ben Alla in 2023."
"""
import argparse

import torch

from tiny_llm import answer_me
from tiny_llm.runtime import resolve_device
from generate import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, tokenizer = load_model(args.ckpt, device=device)
    print(answer_me(model, tokenizer, args.question, args.context, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p))


if __name__ == "__main__":
    main()
