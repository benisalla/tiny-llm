"""
Small synthetic datasets to check, in minutes on a CPU, that the model really learns.

    arithmetic/  train.bin, val.bin      "a+b=c" lines (pre-training from scratch), 10% of the (a, b) pairs
                 test.json               are never seen during training and are used to measure accuracy
    qa/          train.json, val.json    context / question / answer about made-up people; val uses
                                         names, cities and jobs never seen in train (needs real reading, not memory)

    python scripts/make_toy_data.py --out ./data/toy
"""
import argparse
import json
import os
import random

import numpy as np
import tiktoken

FIRST = ["Amina", "Youssef", "Sara", "Omar", "Lina", "Karim", "Nora", "Adam", "Hiba", "Ilyas", "Maya", "Rayan",
         "Salma", "Hamza", "Ines", "Zakaria", "Yasmine", "Anas", "Kenza", "Mehdi", "Aya", "Bilal", "Rania", "Taha",
         "Leila", "Samir", "Dounia", "Imane", "Walid", "Farah", "Nabil", "Siham"]
CITIES = ["Rabat", "Paris", "Tokyo", "Cairo", "Lima", "Oslo", "Madrid", "Dakar", "Lagos", "Rome", "Berlin", "Quito",
          "Seoul", "Lisbon", "Nairobi", "Vienna", "Dublin", "Havana", "Prague", "Tunis", "Athens", "Bogota"]
JOBS = ["doctor", "teacher", "pilot", "baker", "farmer", "nurse", "painter", "lawyer", "chef", "engineer",
        "dentist", "writer", "singer", "plumber", "architect", "librarian", "carpenter", "journalist"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown"]

TEMPLATE = "{name} is a {job} who lives in {city}. Their favourite color is {color}."
QUESTIONS = [
    ("Where does {name} live?", "{city}"),
    ("What is the job of {name}?", "{job}"),
    ("What is the favourite color of {name}?", "{color}"),
    ("Who lives in {city}?", "{name}"),
]


def make_arithmetic(out, rng, enc, max_n=100, test_ratio=0.1):
    pairs = [(a, b) for a in range(max_n) for b in range(max_n)]
    rng.shuffle(pairs)
    n_test = int(len(pairs) * test_ratio)
    test, train = pairs[:n_test], pairs[n_test:]

    def write(name, items, repeats):
        lines = [f"{a}+{b}={a + b}\n" for _ in range(repeats) for a, b in rng.sample(items, len(items))]
        ids = enc.encode_ordinary("".join(lines))
        np.array(ids, dtype=np.uint16).tofile(os.path.join(out, f"{name}.bin"))
        return len(ids)

    os.makedirs(out, exist_ok=True)
    n_train = write("train", train, repeats=3)
    n_val = write("val", test, repeats=1)  # loss on unseen pairs
    with open(os.path.join(out, "test.json"), "w") as f:
        json.dump([{"prompt": f"{a}+{b}=", "answer": str(a + b)} for a, b in test], f)
    print(f"✅ arithmetic: {n_train:,} train tokens, {n_val:,} val tokens, {len(test)} unseen test problems")


def make_qa(out, rng, n_train=600, n_val=100):
    def split(items, k):
        items = items[:]
        rng.shuffle(items)
        return items[k:], items[:k]

    # val people / cities / jobs never appear in train
    names_tr, names_va = split(FIRST, 8)
    cities_tr, cities_va = split(CITIES, 6)
    jobs_tr, jobs_va = split(JOBS, 5)

    def sample(names, cities, jobs):
        name, city, job, color = rng.choice(names), rng.choice(cities), rng.choice(jobs), rng.choice(COLORS)
        fields = dict(name=name, city=city, job=job, color=color)
        # a distractor sentence about someone else makes the model actually read the question
        other = dict(name=rng.choice([n for n in names if n != name]), city=rng.choice([c for c in cities if c != city]),
                     job=rng.choice(jobs), color=rng.choice(COLORS))
        sentences = [TEMPLATE.format(**fields), TEMPLATE.format(**other)]
        rng.shuffle(sentences)
        q, a = rng.choice(QUESTIONS)
        return {"context": " ".join(sentences), "question": q.format(**fields), "answer": a.format(**fields)}

    os.makedirs(out, exist_ok=True)
    for name, n, pools in (("train", n_train, (names_tr, cities_tr, jobs_tr)),
                           ("val", n_val, (names_va, cities_va, jobs_va))):
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump([sample(*pools) for _ in range(n)], f, indent=1)
    print(f"✅ qa: {n_train} train / {n_val} val samples (val names, cities and jobs are unseen)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="./data/toy")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    make_arithmetic(os.path.join(args.out, "arithmetic"), rng, tiktoken.get_encoding("r50k_base"))
    make_qa(os.path.join(args.out, "qa"), rng)


if __name__ == "__main__":
    main()
