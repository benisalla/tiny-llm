# Architecture

Tiny LLM is one decoder-only transformer whose design choices are switches in `ModelConfig`
(`model/tiny_llm/config.py`). With the defaults it is GPT-2; flipping a few switches turns it into
Llama, Qwen2 or Pythia.

```
token ids ─► wte (+ wpe if pos_encoding="learned") ─► dropout
          ─► N x DBlock ─► final norm (ln_f) ─► lm_head (tied with wte if tie_embeddings) ─► logits

DBlock (sequential):  x = x + attn(ln_1(x));  x = x + mlp(ln_2(x))
DBlock (parallel):    x = x + attn(ln_1(x)) + mlp(ln_2(x))              # parallel_residual=True
```

| module | file | switches it reads |
| --- | --- | --- |
| `LayerNorm`, `RMSNorm` | `norm.py` | `norm_type`, `norm_eps`, `bias` |
| `RotaryEmbedding` (one table shared by all layers) | `rope.py` | `pos_encoding`, `rope_theta`, `rotary_pct` |
| `CausalSelfAttention` + `KVCache` | `attention.py` | `n_head`, `n_kv_head`, `head_dim`, `qkv_bias`, `bias` |
| `MLP` | `mlp.py` | `activation`, `gated_mlp`, `intermediate_size`, `bias` |
| `DBlock` | `block.py` | `parallel_residual` |
| `TinyLLM` | `model.py` | `tie_embeddings`, `block_size`, `vocab_size` |
| `LoraLinear`, `LoraEmbedding` | `lora.py` | `lora_rank`, `lora_alpha`, `lora_dropout` |

## What changes for each model family

| `ModelConfig` field | GPT-2 / DistilGPT2 | Llama (SmolLM2, TinyLlama) | Qwen2 (Qwen2.5) | GPT-NeoX (Pythia) |
| --- | --- | --- | --- | --- |
| `norm_type` | `layernorm` | `rmsnorm` | `rmsnorm` | `layernorm` |
| `pos_encoding` | `learned` | `rope` | `rope` | `rope` |
| `rotary_pct` | - | `1.0` | `1.0` | `0.25` |
| `activation` | `gelu_tanh` | `silu` | `silu` | `gelu` |
| `gated_mlp` | `False` | `True` (SwiGLU) | `True` (SwiGLU) | `False` |
| `n_kv_head` | = `n_head` | fewer (GQA) | fewer (GQA) | = `n_head` |
| `bias` / `qkv_bias` | `True` / `True` | `False` / `False` | `False` / `True` | `True` / `True` |
| `parallel_residual` | `False` | `False` | `False` | `True` |
| `tie_embeddings` | `True` | per checkpoint | per checkpoint | `False` |

`python model/scripts/list_families.py` prints the same information from the code.

## Loading Hugging Face checkpoints

`TinyLLM.from_pretrained(name, tokenizer)` (`pretrained.py`):

1. reads `config.model_type` and picks the family in `families.py`;
2. `to_config` turns the Hugging Face config into a `ModelConfig`;
3. `convert` renames the weights into our layout (GPT-2 names). Examples: GPT-2 `Conv1D` weights are
   transposed, Llama's `q_proj`/`k_proj`/`v_proj` are concatenated into `c_attn`, Pythia's per-head
   interleaved `query_key_value` is reordered. Any weight left unconverted raises an error;
4. our special tokens get embeddings (new rows, or unused spare rows), sampled around the trained ones;
5. the model is built on the `meta` device and the tensors are assigned directly, so loading needs one
   copy of the weights in memory, not three.

## Adding a family

Write `to_config` and `convert` in `families.py`, call `register(Family(...))`, and add a tiny random
model of that family to `model/tests/test_families.py`: the test checks logits and greedy generation
against Hugging Face. If the family needs a new design choice (e.g. Gemma scales embeddings by
sqrt(n_embd)), add a field to `ModelConfig` and use it in the layer.

Not supported yet: RoPE scaling (Llama 3.x), sliding-window attention, q/k norms (Qwen3), mixture of experts.
