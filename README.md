# Tiny LLM FROM SCRATCH
This is our Tiny LLM (😁 we are still learning), built from scratch and inspired by the innovative approaches of Hugging Face Transformers and OpenAI architectures. Developed during our internship at [3D Smart Factory](https://3dsmartfactory.csit.ma/), this model showcases our committed efforts to create an advanced AI solution.


<div align="center">
  <h5>Tiny-LLM</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/264447768-39a2a1a9-ca80-44cd-9d83-fd1358cb379b.png" width="200" height="200" />
</div>


<div align="center">
  <h5>A little demo</h5>
  <a href="https://drive.google.com/file/d/1QO3Is-Il5koQb8WpgAtzE-KWBa_E2Xsr/view?usp=sharing">
    <img src="https://github.com/benisalla/micro-tiny-gpt-from-scratch/assets/89405673/ac9b010d-1612-4f28-8856-421062fe0175.png" width="600" height="250" alt="Explanatory Video"/>
  </a>
</div>


## NOTE

**Important:** This project involves fine-tuning various GPT family models (small, medium, large, etc.) to develop two distinct chatbots: one for question and answer interactions [here](https://github.com/benisalla/fine-tune-GPT-chatbot) and another for context-based question and answer interactions [here](https://github.com/benisalla/fine-tune-gpt2_on_context_Q-A). Additionally, we have implemented a question and answer system based on the core transformer architecture which was initially used for translation tasks [here](https://github.com/benisalla/chatbot-based-translation-transformer).


**Note:** This is a learning project and does not represent a production-grade solution. The focus is on educational growth and experimentation.


## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data](#data)
- [Background on GPT-2](#background-on-gpt-2)
- [Model Architecture](#model-architecture)
- [Supported Model Families](#supported-model-families)
- [Features of this project](#features-of-this-project)
- [Explanatory Videos](#explanatory-videos)
- [Training](#training)
- [Inference](#inference)
- [Fine Tuning](#fine-tuning-tiny-llm)
- [Chatbot Interface](#chatbot-interface)
- [Sanity Check on Simple Data](#sanity-check-on-simple-data)
- [Tests](#tests)
- [Contributors](#contributors)
- [Contact Us](#contact-us)
- [About Me](#about-me)


# Introduction
Welcome to the Tiny LLM repository! This project is an exploration into building a compact language model from scratch, taking inspiration from the Hugging Face Transformers and OpenAI architectures. Developed during an internship at 3D Smart Factory, this model represents our dedication to creating advanced AI solutions despite our learning phase.

This README walks through the whole process of building the foundational GPT model from scratch: data collection, model architecture, training, fine-tuning (with LoRA), and finally a small chatbot interface to talk with the model.

The same from-scratch code is not limited to GPT-2: a handful of architecture switches let it load other small LLMs (Llama / SmolLM2, Qwen2.5, Pythia), see [Supported Model Families](#supported-model-families).


## Quick Start

```bash
git clone https://github.com/benisalla/tiny-llm.git
cd tiny-llm

python -m venv .venv && source .venv/bin/activate
pip install -e model/              # the `tiny_llm` package (torch, transformers, numpy, tiktoken)
pip install -r chatbot/requirements.txt

# talk to the model in your browser
streamlit run chatbot/app.py
```

**Free GPU:** upload [`notebooks/train_kaggle_colab.ipynb`](notebooks/train_kaggle_colab.ipynb) to Kaggle or Colab and press *Run all*: it trains, fine-tunes and chats, with no setup.

**Docs:** [architecture](docs/architecture.md) · [training](docs/training.md) · [inference](docs/inference.md) · [testing](docs/testing.md)

No checkpoint yet? The chatbot can load pre-trained small LLMs (GPT-2, SmolLM2, Qwen2.5, Pythia...) straight into our from-scratch architecture (downloaded on first use). Instruct models get a real multi-turn chat.


## Project Structure

```
tiny-llm/
├── model/                      # everything about the LLM
│   ├── tiny_llm/               # the python package
│   │   ├── config.py           #   ModelConfig (architecture switches) + TrainConfig (optimisation)
│   │   ├── norm.py             #   LayerNorm (GPT-2, Pythia) and RMSNorm (Llama, Qwen2)
│   │   ├── rope.py             #   rotary position embeddings (Llama, Qwen2, Pythia)
│   │   ├── attention.py        #   causal self-attention, grouped-query attention, KV cache
│   │   ├── mlp.py              #   MLP: classic (GELU) or gated (SwiGLU)
│   │   ├── block.py            #   DBlock = attention + MLP, sequential or parallel residual
│   │   ├── model.py            #   TinyLLM: embeddings, N blocks, lm_head, generation
│   │   ├── families.py         #   what changes for each model family + weight name mapping
│   │   ├── pretrained.py       #   load any supported Hugging Face checkpoint into TinyLLM
│   │   ├── lora.py             #   LoraLinear, LoraEmbedding, freeze / merge helpers
│   │   ├── tokenizer.py        #   any HF tokenizer + our special tokens + Q&A prompt format
│   │   ├── data.py             #   pre-training batches (.bin) and Q&A BatchLoader (.json)
│   │   ├── training.py         #   lr schedule, loss estimation, the training loop
│   │   ├── checkpoint.py       #   save / load checkpoints
│   │   ├── generation.py       #   completion, question answering and chat (streaming)
│   │   └── runtime.py          #   device, mixed precision, model init, CLI parsing
│   ├── scripts/
│   │   ├── prepare_data.py     #   .txt files -> train.bin / val.bin / test.bin
│   │   ├── train.py            #   pre-training
│   │   ├── finetune.py         #   (LoRA) fine-tuning on question/answer data
│   │   ├── generate.py         #   text completion from the command line
│   │   ├── ask.py              #   ask a question to a fine-tuned checkpoint
│   │   ├── make_toy_data.py    #   small synthetic datasets (arithmetic, reading comprehension)
│   │   ├── eval_toy.py         #   exact-match accuracy on them
│   │   └── list_families.py    #   print the supported families table
│   ├── tests/                  #   pytest suite
│   └── pyproject.toml
├── chatbot/                    # a simple interface to use the model
│   ├── engine.py               #   backend: loads the model, streams answers
│   ├── app.py                  #   frontend: Streamlit chat UI
│   └── tests/
├── notebooks/                  # self-contained Kaggle / Colab notebook (free GPU)
│   ├── build_notebook.py       #   generates it from the package, so it never drifts from the code
│   └── train_kaggle_colab.ipynb
└── docs/                       # architecture, training, inference, testing
```


## Data

### WebText (OpenWebTextCorpus)

To train GPT-2, OpenAI needed a substantial corpus of 40 GB of high-quality text. While Common Crawl provided the necessary scale for modern language models, its quality was often inconsistent. Manually curating data from Common Crawl was an option but a costly one. Fortunately, Reddit provided a decentralized curation approach by design, proving to be a crucial innovation for creating the WebText dataset.

The WebText generation process can be summarized as follows:

1. Retrieve URLs of all Reddit submissions until December 2017 with a score of 3 or higher.
2. Deduplicate retrieved content based on URLs.
3. Exclude Wikipedia content, as OpenAI already had a separate Wikipedia dataset.
4. Further deduplicate the remaining content using an undisclosed "heuristic" cleaning method, including the removal of non-English web pages.

Neither the resulting corpus nor the source code for its generation was made public, which later inspired Aaron Gokaslan and Vanya Cohen to create the OpenWebText corpus.

### OpenWebText

OpenWebText2 is an enhanced version of the original OpenWebText corpus, covering all Reddit submissions from 2005 to April 2020, with additional months becoming available after the publication of corresponding PushShift backup files.


<div align="center">
  <a href="https://skylion007.github.io/OpenWebTextCorpus/">
    <h5>OpenWebText</h5>
  </a>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406287-b908e1a2-8650-48fd-a54e-d7ed79aed037.png" width="300" height="150" />
</div>


Due to resource constraints, we trained our model on only a quarter of the OpenWebText dataset. This limitation in training data was necessary to optimize computational resources while still achieving significant language model performance.


### Data Preprocessing

We combined approximately 5 million files from the OpenWebText dataset, roughly equivalent to a quarter of the entire OpenWebText corpus. Subsequently, we performed the following steps:

1. We used the GPT-2 tokenizer, also known as "r50k_base", to tokenize the dataset.
```python
import tiktoken
tokenizer = tiktoken.get_encoding("r50k_base")
```

2. Following established conventions for dataset splitting, we divided the data into training, validation and test sets (80% / 10% / 10%).
3. To optimize data management and efficiency, we stored the token ids as a binary stream (`uint16`) in the `train.bin`, `val.bin` and `test.bin` files.

All of this is now one command:

```bash
python model/scripts/prepare_data.py --src ./raw_txt --dst ./data/openwebtext --val_ratio 0.1 --test_ratio 0.1
```


Our dataset is currently accessible on Kaggle:

<div align="center">
  <a href="https://www.kaggle.com/datasets/benallaismail/gpt-data">
    <h5>Quarter of OpenWebText by Ben Alla Ismail</h5>
  </a>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406286-22c3d2ab-df41-4c49-981f-ae153445ce5b.jpeg" width="600" height="300" />
</div>


## Background on GPT-2

The Generative Pre-trained Transformer 2 (GPT-2), developed by OpenAI, is the second installment in their foundational series of GPT models. Its predecessor, GPT-1, was pre-trained on the BookCorpus dataset (over 7,000 unpublished fiction books); GPT-2 was trained on WebText, a dataset of about 8 million web pages. It was partially unveiled in February 2019, followed by the full release of the 1.5 billion parameter model on November 5, 2019.


<div align="center">
  <h5>GPT Model</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406289-a66928b2-b651-4fb5-81b2-f9429685d9d0.jpg" width="600" height="300" />
</div>


GPT-2 represents a "direct scale-up" from its predecessor, GPT-1, with a tenfold increase in both the number of parameters and the size of the training dataset. This versatile model owes its ability to perform various tasks to its intrinsic capacity to accurately predict the next element in a sequence. This predictive capability enables GPT-2 to accomplish tasks such as text translation, answering questions based on textual content, summarizing text passages, and generating text that can sometimes closely resemble human style. However, it may exhibit repetitive or nonsensical behavior when generating long passages.

There is a family of GPT-2 models; below, we can see the pre-trained GPT-2 model family:


<div align="center">
  <h5>GPT Model Family</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406288-e0dc48f8-a5e1-4807-8184-39ca3e2a88c5.jpg" width="700" height="300" />
</div>


## Model Architecture

The architecture of GPT, a groundbreaking language model, represents a notable evolution of deep learning-based transformer models. The original transformer had both an encoder and a decoder; GPT keeps only the decoder stack (masked self-attention + feed-forward), trained to predict the next token.

<div align="center">
  <h5>GPT Relationship with Transformers</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406295-de6880cf-7d77-4b16-bcb5-d58b950e3404.jpg" width="700" height="300" />
</div>


This led to models with exceptionally high stacks of transformer blocks and massive volumes of training data, often requiring significant computational resources and costs.


<div align="center">
  <h5>Full GPT Architecture</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406294-83845cfb-f442-478f-9993-3412225dabb7.png" width="350" height="600" />
</div>


In code (`model/tiny_llm/`):

| GPT-2 component | Our module |
| --- | --- |
| token + position embeddings | `TinyLLM.transformer.wte` / `wpe` |
| masked multi-head self-attention | `CausalSelfAttention` |
| feed-forward (4x, GELU) | `MLP` |
| pre-LayerNorm residual block (x N) | `DBlock` |
| final LayerNorm + language model head (tied with `wte`) | `ln_f`, `lm_head` |

Module names match Hugging Face GPT-2, so the official weights load 1:1. The test suite checks that our implementation produces **the same logits and the same greedy generations as Hugging Face GPT-2**.


## Supported Model Families

Most small open LLMs are the same decoder-only transformer as GPT-2. They only differ by a few design choices, and each of those choices is one field of `ModelConfig`. Here is what should be changed for each type of model (defaults = GPT-2):

| `ModelConfig` field | GPT-2 / DistilGPT2 | Llama (SmolLM2, TinyLlama) | Qwen2 (Qwen2.5) | GPT-NeoX (Pythia) |
| --- | --- | --- | --- | --- |
| `norm_type` | `layernorm` | `rmsnorm` | `rmsnorm` | `layernorm` |
| `pos_encoding` | `learned` (position matrix) | `rope` | `rope` | `rope` |
| `rotary_pct` | - | `1.0` | `1.0` | `0.25` (only a quarter of each head is rotated) |
| `activation` | `gelu_tanh` | `silu` | `silu` | `gelu` |
| `gated_mlp` | `False` | `True` (SwiGLU) | `True` (SwiGLU) | `False` |
| `n_kv_head` | = `n_head` | fewer (grouped-query attention) | fewer (grouped-query attention) | = `n_head` |
| `bias` / `qkv_bias` | `True` / `True` | `False` / `False` | `False` / `True` | `True` / `True` |
| `parallel_residual` | `False` | `False` | `False` | `True` |
| `tie_embeddings` | `True` | depends on the checkpoint | depends on the checkpoint | `False` |

The weights also need to be renamed into our layout (GPT-2 names): for example Llama's separate `q_proj`, `k_proj`, `v_proj` are concatenated into our fused `c_attn`, and Pythia's per-head interleaved `query_key_value` is reordered. All of this lives in [`model/tiny_llm/families.py`](model/tiny_llm/families.py), one readable block per family (`python model/scripts/list_families.py` prints its summary).

```python
from tiny_llm import TinyLLM, create_tokenizer, generate_text

name = "HuggingFaceTB/SmolLM2-135M"          # or "gpt2", "Qwen/Qwen2.5-0.5B", "EleutherAI/pythia-160m", ...
tokenizer = create_tokenizer(name)            # its own tokenizer + our special tokens
model = TinyLLM.from_pretrained(name, tokenizer)
print(generate_text(model, tokenizer, "Once upon a time", max_new_tokens=40))
```

Every script accepts these names too (`--init_from HuggingFaceTB/SmolLM2-135M`, `--pretrained Qwen/Qwen2.5-0.5B`), so you can continue pre-training or LoRA fine-tune any of them.

**Verified:** for each family the tests build a model with Hugging Face, load it into `TinyLLM`, and check that the logits and the greedy generations (with and without KV cache) are identical.

**Adding a new family:** write its `to_config` (read the Hugging Face config) and `convert` (rename the weights) in `families.py` and call `register(...)`. If it needs a design choice that doesn't exist yet (e.g. Gemma multiplies the embeddings by sqrt(n_embd)), add a field to `ModelConfig` and use it in the layer. Not supported yet: RoPE scaling (Llama 3.x), sliding-window attention, q/k norms (Qwen3), mixture of experts.


## Features of this project

- **Tiny Scale:** This GPT model is intentionally designed to be on a tiny scale, showcasing our learning journey and commitment to innovation.

- **Inspired by Industry Leaders:** Influenced by the Hugging Face Transformers and OpenAI architectures, we've incorporated industry best practices into our model's design.

- **From scratch, many models:** every layer is written by hand, yet GPT-2, Llama/SmolLM2, Qwen2.5 and Pythia checkpoints load into it and give the same outputs as Hugging Face.

- **Fast generation:** KV cache, grouped-query attention, top-k / top-p sampling, streaming.

- **LoRA fine-tuning:** fine-tune HUGE LLMs by training only small low-rank matrices, then merge them back for free inference.

- **A chatbot interface:** a Streamlit app to chat with your model.

- **Internship Project:** Developed during our internship at 3D Smart Factory, this project reflects real-world experience and hands-on learning.


## Explanatory Videos

I've created explanatory videos to help you understand the intricate link between the architecture and the code, as it might not be immediately obvious.

<div align="center">
  <a href="https://drive.google.com/file/d/14V5PbIBW3fgVWAvB1yux_lGHL6ULtdcD/view?usp=drive_link">
    <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288115000-0db1ec7c-06a9-4801-8ae0-900b2c0382e2.png" width="700" height="300" alt="Explanatory Video"/>
  </a>
</div>


## Training

```bash
cd model

# pre-train from scratch (GPT-2 small size by default)
python scripts/train.py --data_dir ./data/openwebtext --init_from scratch

# a small run that fits on a laptop
python scripts/train.py --data_dir ./data/openwebtext --n_layer 4 --n_head 4 --n_embd 256 \
    --block_size 256 --batch_size 8 --grad_accum_steps 4

# continue from the last checkpoint
python scripts/train.py --data_dir ./data/openwebtext --init_from resume
```

Every field of `ModelConfig` and `TrainConfig` is a command-line flag (`python scripts/train.py --help`). Checkpoints go to `./checkpoints/ckpt.pt` whenever the validation loss improves. Mixed precision (bfloat16 / float16) is used automatically on GPU.

<div align="center">
  <h4>Loss function</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406292-81226d87-8166-4f6d-b913-baa419f794ff.png" width="600" height="300"/>
</div>


## Inference

```bash
python scripts/generate.py --ckpt ./checkpoints/ckpt.pt --prompt "ismail ben alla is the next elon mask because"
python scripts/generate.py --pretrained gpt2 --prompt "Once upon a time"   # original GPT-2 weights
```

<div align="center">
  <h4>Example 1</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406291-a362a3cf-3886-40d8-a1cc-71f448616a22.jpeg" width="700" height="300"/>
</div>
<div align="center">
  <h4>Example 2</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288919714-b1151a7c-f15e-46d1-b39b-7035cb864406.png" width="700" height="300"/>
</div>
<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288919814-f2983df6-a849-49ef-9bba-e47b88266de2.png" width="700" height="300"/>
</div>
<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288919922-62558357-7578-45e1-b762-18a9327f6890.png" width="700" height="300"/>
</div>


## Fine tuning Tiny LLM

Fine-tuning data is a JSON list of `{"context": ..., "question": ..., "answer": ...}` in `train.json` and `val.json`. Each sample becomes:

```
<sost>{context}<ques>{question}<ans>{answer}<eost>
```

By default the loss is only computed on the answer (`--loss_on_answer_only true`) and padding is never part of the loss.

```bash
cd model

# LoRA fine-tuning of GPT-2 (only the low-rank matrices + token embeddings are trained)
python scripts/finetune.py --init_from gpt2 --data_dir ./data/qa --lora_rank 8 \
    --learning_rate 3e-4 --drop_rate 0.1 --max_iters 2000

# LoRA fine-tuning of our own pre-trained model
python scripts/finetune.py --init_from resume --resume_from ./checkpoints/ckpt.pt --data_dir ./data/qa --lora_rank 8

# ask it something
python scripts/ask.py --ckpt ./checkpoints/finetuned.pt \
    --context "Ismail Ben Alla built Tiny LLM at 3D Smart Factory." --question "Who built Tiny LLM?"
```

For more details, see the project [Fine-Tuning Pretrained GPT-2](https://github.com/benisalla/fine-tune-gpt2_on_context_Q-A), where GPT-2 has been fine-tuned specifically for context-question-answer scenarios.

<div align="center">
  <h4>Example 1</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288918563-bd3fc6ca-d24d-4358-b5f8-22ce6afc09b8.jpeg" width="700" height="300"/>
</div>
<div align="center">
  <h4>Example 2</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288918548-b812b6e0-d3bf-4aea-8c89-370a4e79f7e2.jpeg" width="700" height="300"/>
</div>
<div align="center">
  <h4>Example 3</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288918495-a1bca43c-82f1-4322-ae98-18f76836aaab.jpeg" width="700" height="300"/>
</div>


## Chatbot Interface

```bash
streamlit run chatbot/app.py
# or point it to a specific checkpoint
TINY_LLM_CKPT=model/checkpoints/finetuned.pt streamlit run chatbot/app.py
```

- **Weights:** one of your checkpoints (`.pt`) or any supported pre-trained small LLM (SmolLM2, Qwen2.5, GPT-2, Pythia, or any Hugging Face id of a supported family).
- **Modes:** *chat* for instruct models (multi-turn, uses the model's own chat template), *question answering* (with an optional context paragraph) for checkpoints fine-tuned with `finetune.py`, *text completion* for base models. The right mode is picked automatically.
- **Generation settings:** max new tokens, temperature, top-k, top-p. Answers are streamed token by token.

The backend (`chatbot/engine.py`) has no UI code, so the same `ChatEngine` can be used from a notebook or behind an API:

```python
from engine import ChatEngine
bot = ChatEngine.from_checkpoint("model/checkpoints/finetuned.pt")
print(bot.reply("Who built Tiny LLM?", context="Ismail Ben Alla built Tiny LLM."))

bot = ChatEngine.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
print(bot.reply("What is the capital of France?"))   # -> "The capital of France is Paris."
```


## Sanity Check on Simple Data

Unit tests prove the code is correct; these two small synthetic tasks prove the model actually **learns**, in minutes on a CPU.

```bash
cd model
python scripts/make_toy_data.py --out ./data/toy

# 1. pre-training from scratch: learn addition, scored on 1,000 problems never seen in training
python scripts/train.py --data_dir ./data/toy/arithmetic --ckpt_name arithmetic.pt \
    --n_layer 4 --n_head 4 --n_embd 128 --block_size 32 --batch_size 16 --grad_accum_steps 1 \
    --learning_rate 3e-3 --min_lr 3e-4 --warmup_iters 50 --lr_decay_iters 1500 --max_iters 1500 --eval_interval 250
python scripts/eval_toy.py arithmetic --ckpt ./checkpoints/arithmetic.pt --data ./data/toy/arithmetic/test.json

# 2. LoRA fine-tuning: reading comprehension, scored on people / cities / jobs never seen in training
python scripts/eval_toy.py qa --pretrained distilgpt2 --data ./data/toy/qa/val.json        # before fine-tuning
python scripts/finetune.py --init_from distilgpt2 --data_dir ./data/toy/qa --lora_rank 8 --learning_rate 1e-3 \
    --min_lr 1e-4 --max_iters 300 --warmup_iters 20 --lr_decay_iters 300 --grad_accum_steps 1 --toks_per_batch 2000 \
    --eval_interval 100 --ckpt_name qa_finetuned.pt
python scripts/eval_toy.py qa --ckpt ./checkpoints/qa_finetuned.pt --data ./data/toy/qa/val.json
```

**Results** (CPU only, on a shared and heavily loaded machine; exact match, greedy decoding):

| task | setup | before | after | evaluated on |
| --- | --- | --- | --- | --- |
| addition `a+b=` | 7M-param model **from scratch**, 1,500 steps (~45 min) | 0% (random) | **30.1%** | 1,000 problems never seen in training |
| reading comprehension | **LoRA** (rank 8) on DistilGPT2, 300 steps | 0% | **40.0%** | 100 questions about unseen people, cities and jobs |

| | train loss | val loss |
| --- | --- | --- |
| addition (step 0 → 1,500) | 10.83 → 1.86 | 10.83 → 1.87 |
| reading comprehension (step 0 → 300) | 55.99 → 0.00 | 55.28 → 1.82 (best 1.71 at step 200) |

Reading the numbers honestly:
- **Addition:** train and val loss stay together, so the model learns the rule, not the pairs. The loss
  can't go much below ~1.2 because `a` and `b` are random. Wrong answers are usually close
  (`95+96=` → `188`, `7+83=` → `89`), and the loss was still falling: longer training (e.g. the Kaggle
  notebook on a GPU) raises the accuracy.
- **Reading comprehension:** the base model has never seen the `<sost>…<ques>…<ans>` format and just
  repeats the question; after LoRA it answers in the right format and reads the right fact for 40% of
  unseen entities. The gap between train (0.00) and val (1.82) loss shows it overfits the 600 training
  samples: more data, early stopping (best val loss was at step 200) or a larger model would help.


## Tests

```bash
pip install pytest
cd model && pytest                 # layers, LoRA, model, data, training, scripts end-to-end,
                                   # every family vs Hugging Face (tiny random models, no download), GPT-2 weights
cd ../chatbot && pytest            # engine + the Streamlit UI (headless)
pytest -m "not network"            # skip tests that download GPT-2 from the Hugging Face Hub
TINY_LLM_TEST_HUB=1 pytest tests/test_hub_models.py   # real DistilGPT2, SmolLM2, Pythia, Qwen2.5 (~1.7 GB)
```


## Contributors

Thanks to the team who built this project with me:

- [Ben Alla Ismail (me 😃)](https://github.com/benisalla)
- [Agrat Mohammed](https://github.com/agrat)
- [Mhaoui Siham](https://github.com/siham)
- [Souhayle ou-aabi](https://github.com/Souhayle-ou-aabi)
- [Daoudi Ayoub](https://github.com/Daoudi-Ayoub)


## Contact Us
For inquiries or suggestions, please contact:
- Project Lead: Ben alla ismail ([ismailbenalla52@gmail.com](mailto:ismailbenalla52@gmail.com))
- Co-lead: mhaoui Siham ([mahouisiham@gmail.com](mailto:mahouisiham@gmail.com))


## About Me

🎓 I'm Ismail Ben Alla, and I have a deep passion for neural networks 😍. My mission is to assist neural networks in unraveling the mysteries of our universe.</br>
⛵ I'm an enthusiast when it comes to AI, Deep Learning, and Machine Learning algorithms.</br>
✅ I'm an optimist and a dedicated hard worker, constantly striving to push the boundaries of what's possible.</br>
🌱 I'm committed to continuously learning and staying updated with advanced computer science technologies.</br>
😄 I absolutely love what I do, and I'm excited about the endless possibilities in the world of AI and machine learning!</br>

Let's connect and explore the fascinating world of artificial intelligence together! 🤖🌟


<div align="center">
  <a href="https://twitter.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
  <a href="https://linkedin.com/in/ismail-ben-alla-7144b5221/" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="ismail-ben-alla-7144b5221/" height="30" width="40" />
  </a>
  <a href="https://instagram.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
</div>


<div align="center">
  <h4>You are about to witness some pure magic ✨🎩 !! Ta-da!</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/270190847-0b3ee23b-c082-483e-9e12-8b15a1b8f0a3.gif" width="500" height="300"/>
</div>
