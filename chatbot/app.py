"""
Chatbot frontend (Streamlit).

    streamlit run chatbot/app.py
    TINY_LLM_CKPT=model/checkpoints/finetuned.pt streamlit run chatbot/app.py
"""
import os

import streamlit as st

from engine import CHAT, COMPLETE, QA, ChatEngine, GenerationSettings

DEFAULT_CKPT = os.environ.get(
    "TINY_LLM_CKPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "checkpoints",
                               "finetuned.pt"))
# small models of every supported family (tiny_llm/families.py), any other Hub id works too
SUGGESTED_MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",  # llama family, chat
    "Qwen/Qwen2.5-0.5B-Instruct",  # qwen2 family, chat
    "gpt2", "distilgpt2", "gpt2-medium",  # gpt2 family
    "HuggingFaceTB/SmolLM2-135M",  # llama family, base
    "EleutherAI/pythia-160m",  # gpt_neox family
]
CUSTOM = "Other (type a Hugging Face id)..."
MODE_LABELS = {CHAT: "Chat", QA: "Question answering", COMPLETE: "Text completion"}

st.set_page_config(page_title="Tiny LLM", page_icon="🤖", layout="centered")


@st.cache_resource(show_spinner="Loading the model...")
def get_engine(source, value):
    if source == "checkpoint":
        return ChatEngine.from_checkpoint(value)
    return ChatEngine.from_pretrained(value)


# ---------------------------------------------------------------- sidebar: model + settings
with st.sidebar:
    st.header("🧠 Model")
    has_ckpt = os.path.isfile(DEFAULT_CKPT)
    source = st.radio("Weights", ["checkpoint", "pretrained"], index=0 if has_ckpt else 1,
                      format_func=lambda s: "My checkpoint (.pt)" if s == "checkpoint" else "Pre-trained small LLM")
    if source == "checkpoint":
        value = st.text_input("Checkpoint path", value=os.path.normpath(DEFAULT_CKPT))
    else:
        value = st.selectbox("Model", SUGGESTED_MODELS + [CUSTOM],
                             help="Loaded into our from-scratch architecture. Downloaded on first use.")
        if value == CUSTOM:
            value = st.text_input("Hugging Face id", placeholder="EleutherAI/pythia-70m").strip()
            if not value:
                st.stop()

    try:
        engine = get_engine(source, value)
    except Exception as e:  # bad path, no network, corrupted file...
        st.error(f"Could not load the model: {e}")
        st.stop()
    st.caption(f"**{engine.name}** · {engine.num_params / 1e6:.0f}M params · {engine.device}")

    st.header("💬 Mode")
    modes = engine.modes
    mode = st.radio("Mode", modes, index=modes.index(engine.default_mode), format_func=MODE_LABELS.get,
                    label_visibility="collapsed")
    context = ""
    if mode in (QA, CHAT):
        context = st.text_area("Context (optional)", height=150,
                               placeholder="Paste a paragraph, the model answers questions about it.")
        if mode == QA and engine.default_mode != QA:
            st.info("This model was not fine-tuned for Q&A, answers may be poor. "
                    "Fine-tune it with `model/scripts/finetune.py`.")

    st.header("🎛️ Generation")
    settings = GenerationSettings(
        max_new_tokens=st.slider("Max new tokens", 10, 500, 100, 10),
        temperature=st.slider("Temperature", 0.0, 2.0, 0.7, 0.05, help="0 = always pick the most likely token"),
        top_k=st.slider("Top-k", 0, 200, 50, 5, help="0 = disabled"),
        top_p=st.slider("Top-p", 0.05, 1.0, 0.95, 0.05),
    )
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []

# ---------------------------------------------------------------- chat
st.title("🤖 Tiny LLM")
st.caption("A small LLM built from scratch. " + ("The conversation is remembered." if mode == CHAT
                                                  else "Each message is answered independently."))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

placeholder = "Start a sentence, the model continues it..." if mode == COMPLETE else "Ask a question..."
if prompt := st.chat_input(placeholder):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reply = st.write_stream(engine.stream(prompt, context, mode, settings, history))
        if not reply or not str(reply).strip():
            reply = "_(the model stopped without answering, try a higher temperature)_"
            st.markdown(reply)
        elif mode == COMPLETE:
            reply = f"{prompt}{reply}"
    st.session_state.messages.append({"role": "assistant", "content": reply})
