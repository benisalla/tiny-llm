"""
Exact-match accuracy on the toy datasets (see make_toy_data.py), with greedy decoding.

    python scripts/eval_toy.py arithmetic --ckpt ./checkpoints/arithmetic.pt --data ./data/toy/arithmetic/test.json
    python scripts/eval_toy.py qa --ckpt ./checkpoints/finetuned.pt --data ./data/toy/qa/val.json
    python scripts/eval_toy.py qa --pretrained gpt2 --data ./data/toy/qa/val.json      # baseline, no fine-tuning
"""
import argparse
import json

import torch

from tiny_llm import answer_me, stream_tokens
from tiny_llm.runtime import resolve_device
from generate import load_model


def evaluate(task, model, tokenizer, samples, verbose=3):
    correct = 0
    for i, s in enumerate(samples):
        if task == "arithmetic":
            out = "".join(stream_tokens(model, tokenizer, s["prompt"], max_new_tokens=6, temperature=0))
            pred = out.split("\n")[0].strip()
        else:
            pred = answer_me(model, tokenizer, s["question"], s["context"], max_new_tokens=12, temperature=0)
        ok = pred == s["answer"]
        correct += ok
        if i < verbose:
            question = s.get("prompt") or s["question"]
            print(f"  {question!r} -> {pred!r} (expected {s['answer']!r}) {'✅' if ok else '❌'}")
    return correct / len(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["arithmetic", "qa"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt")
    source.add_argument("--pretrained")
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N samples")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    model, tokenizer = load_model(args.ckpt, args.pretrained, resolve_device(args.device))
    with open(args.data) as f:
        samples = json.load(f)
    samples = samples[:args.limit] if args.limit else samples
    acc = evaluate(args.task, model, tokenizer, samples)
    print(f"{args.task} exact-match accuracy: {acc:.1%} on {len(samples)} samples")
    return acc


if __name__ == "__main__":
    main()
