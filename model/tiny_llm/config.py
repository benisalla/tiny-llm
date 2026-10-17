from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class ModelConfig:
    """
    Architecture of the model (everything needed to rebuild it from a checkpoint).

    The defaults describe GPT-2. Other small LLMs are the same decoder-only transformer with a few
    design choices changed. Those choices are the fields of the second block below, and
    `tiny_llm/families.py` lists, for each supported family, which of them it changes.
    """

    # ---- size ----
    block_size: int = 1024  # context length
    vocab_size: int = 50304  # 50257 GPT-2 tokens + our special tokens, padded to a multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    drop_rate: float = 0.0  # 0.0 for pre-training, ~0.1 for fine-tuning (prevent it from learning too much)
    bias: bool = True  # biases in Linear layers and LayerNorms

    # ---- what changes between model families (defaults = GPT-2) ----
    n_kv_head: int = 0  # 0 -> same as n_head. Fewer KV heads = grouped-query attention (Llama, Qwen2)
    head_dim: int = 0  # 0 -> n_embd // n_head
    qkv_bias: Optional[bool] = None  # None -> same as `bias`. Qwen2: biases on q/k/v only
    norm_type: str = "layernorm"  # "layernorm" (GPT-2, Pythia) or "rmsnorm" (Llama, Qwen2)
    norm_eps: float = 1e-5
    pos_encoding: str = "learned"  # "learned" absolute positions (GPT-2) or "rope" rotary (Llama, Qwen2, Pythia)
    rope_theta: float = 10000.0  # RoPE base frequency
    rotary_pct: float = 1.0  # fraction of each head that is rotated (Pythia: 0.25)
    activation: str = "gelu_tanh"  # "gelu_tanh" (GPT-2), "gelu" (Pythia), "silu" (Llama, Qwen2), "relu"
    gated_mlp: bool = False  # SwiGLU MLP: down(act(gate(x)) * up(x)) (Llama, Qwen2)
    intermediate_size: int = 0  # MLP hidden size, 0 -> 4 * n_embd
    parallel_residual: bool = False  # x + attn(ln_1(x)) + mlp(ln_2(x)) instead of sequential (Pythia)
    tie_embeddings: bool = True  # lm_head shares the token embedding matrix (GPT-2, SmolLM2, Qwen2.5-0.5B)

    # for Mrs. LoRA (lora_rank = 0 means LoRA is disabled)
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    def __post_init__(self):
        assert self.head_dim or self.n_embd % self.n_head == 0, "n_embd % n_head should be 0."
        assert self.n_head % self.kv_heads == 0, "n_head must be a multiple of n_kv_head"
        assert self.norm_type in ("layernorm", "rmsnorm"), self.norm_type
        assert self.pos_encoding in ("learned", "rope"), self.pos_encoding
        assert self.activation in ("gelu_tanh", "gelu", "silu", "relu"), self.activation

    # derived sizes
    @property
    def kv_heads(self):
        return self.n_kv_head or self.n_head

    @property
    def head_size(self):
        return self.head_dim or self.n_embd // self.n_head

    @property
    def hidden_size(self):
        return self.intermediate_size or 4 * self.n_embd

    @property
    def use_qkv_bias(self):
        return self.bias if self.qkv_bias is None else self.qkv_bias

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Build a config from a dict, ignoring unknown keys (e.g. old checkpoints)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class TrainConfig:
    """Everything about the optimisation loop (pre-training and fine-tuning)."""

    # "scratch", "resume", or a Hugging Face model id / local folder of a supported family
    # ("gpt2", "distilgpt2", "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B", "EleutherAI/pythia-70m", ...)
    init_from: str = "scratch"
    tokenizer: str = ""  # Hugging Face tokenizer, "" -> the pre-trained model's own (or "gpt2" from scratch)

    # data
    data_dir: str = "./data"
    batch_size: int = 12  # sequences per micro-batch (pre-training)
    toks_per_batch: int = 6000  # tokens per micro-batch (fine-tuning, batches are built by length)
    grad_accum_steps: int = 40  # micro-batches per optimizer step

    # optimizer
    learning_rate: float = 6e-4  # change this to 3e-5 in fine-tuning stage
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # lr schedule (linear warmup + cosine decay)
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    max_iters: int = 600000

    # evaluation / logging / checkpointing
    eval_interval: int = 250
    eval_iters: int = 200
    log_interval: int = 10
    always_save_checkpoint: bool = False  # if False, only save when val loss improves
    out_dir: str = "./checkpoints"
    ckpt_name: str = "ckpt.pt"  # checkpoints are written to out_dir/ckpt_name
    resume_from: str = ""  # checkpoint to load with init_from="resume" (default: out_dir/ckpt_name)

    # fine-tuning
    loss_on_answer_only: bool = True  # only learn to predict the answer, not the context/question
    train_embeddings: bool = True  # with LoRA: keep the (tied) token embeddings trainable so new special tokens get learned
    max_seq_len: int = 400  # drop fine-tuning samples longer than this (in tokens)

    # system
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    dtype: str = "auto"  # "auto", "float32", "bfloat16", "float16"
    compile: bool = False
    seed: int = 2000
