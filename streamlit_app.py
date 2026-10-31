"""Streamlit chat UI for the personal RAG. Runs locally, like rag.py.

Run:  streamlit run streamlit_app.py
"""

import random
import time
from pathlib import Path
from urllib.parse import quote

import ollama
import streamlit as st

from app import config
from app.generate import generate_stream
from app.index import get_collection, load_index, rebuild_index
from app.retrieval import Retriever

# Internal mode names (used in .env and the eval) -> names shown in the UI
MODES = {
    "hybrid": "Hybrid",
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid_rerank": "Hybrid + Rerank",
}
MODE_HELP = (
    "**Hybrid**: meaning search + keyword search, merged (best in my eval).  \n"
    "**Dense**: meaning search with embeddings.  \n"
    "**BM25**: keyword search, good for exact names and codes.  \n"
    "**Hybrid + Rerank**: Hybrid, then a cross-encoder re-scores the results (slower)."
)

# Good general-purpose models on Ollama. Any other model you've pulled is added to the list too.
SUGGESTED_LLMS = ["llama3.1:8b", "qwen2.5:7b", "gemma3:4b", "mistral:7b", "llama3.2:3b", "qwen2.5:3b"]

AVATARS = {"user": ":material/person:", "assistant": ":material/smart_toy:"}
VAULT_NAME = Path(config.VAULT_PATH).name

st.set_page_config(page_title="Personal Knowledge RAG")


# --- Ollama helpers ---

def with_tag(model):
    """Ollama names models 'name:tag', and a missing tag means ':latest'."""
    return model if ":" in model else f"{model}:latest"


def installed_models():
    return [m.model for m in ollama.list().models]


def loaded_models():
    """Models currently loaded in memory (what `ollama ps` shows)."""
    return [m.model for m in ollama.ps().models]


def download_model(model):
    """Pull a model with a live progress bar."""
    bar = st.progress(0.0, text=f"Starting download of {model}...")
    fraction = 0.0
    for update in ollama.pull(model, stream=True):
        text = update.status
        if update.total and update.total > 1e6:  # the model weights; skip the tiny config files
            completed = update.completed or 0
            fraction = completed / update.total
            text = f"Downloading {model}: {completed / 1e9:.1f} / {update.total / 1e9:.1f} GB"
        bar.progress(min(fraction, 1.0), text=text)
    bar.empty()


# --- Index helpers ---

@st.cache_resource(show_spinner="Loading your notes...")
def get_retriever():
    # Cached so the index and BM25 are loaded once, not on every question
    return Retriever(load_index())


def index_progress_bar():
    """Returns an on_progress callback for build_index() that draws a progress bar."""
    bar = st.progress(0.0, text="Indexing your vault...")

    def on_progress(done, total, note):
        bar.progress(done / total, text=f"Indexing note {done} of {total}: {note}")

    return on_progress


# --- Display helpers ---

def unique_notes(hits):
    return list(dict.fromkeys(hit.source for hit in hits))  # unique, in ranked order


def obsidian_link(source):
    """A link that opens the note in the Obsidian app."""
    note = source.replace("\\", "/").removesuffix(".md")
    return f"obsidian://open?vault={quote(VAULT_NAME)}&file={quote(note)}"


def show_sources(hits):
    with st.expander(f"Sources: {len(unique_notes(hits))} notes, {len(hits)} chunks"):
        for i, hit in enumerate(hits, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. [{Path(hit.source).stem}]({obsidian_link(hit.source)})**")
                st.caption(f"{hit.source} · score {hit.score:.3f}")
                st.caption(hit.text[:400] + ("..." if len(hit.text) > 400 else ""))


def answer(question, llm, mode, top_k):
    """Search, then stream the answer, with live progress in a status box. Returns the chat message."""
    status = st.status(f"Searching your notes ({MODES[mode]})...")
    answer_area = st.container()  # created now so the answer appears below the status box

    try:
        start = time.perf_counter()
        hits = retriever.search(question, k=top_k, mode=mode)
        search_time = time.perf_counter() - start
        status.write(f"Found {len(hits)} chunks from {len(unique_notes(hits))} notes in {search_time:.1f}s")

        if llm not in loaded_models():
            status.write(f"Loading {llm} into memory (the first answer takes longer)")
        status.update(label=f"Writing the answer with {llm}...")

        start = time.perf_counter()
        text = answer_area.write_stream(generate_stream(question, hits, llm))  # shows the answer as it's written
        answer_time = time.perf_counter() - start
    except Exception:
        status.update(label="Something went wrong", state="error")
        raise
    status.update(label=f"Done in {search_time + answer_time:.1f}s", state="complete")

    details = f"{MODES[mode]} · {top_k} chunks · search {search_time:.1f}s · answer {answer_time:.1f}s · {llm}"
    st.caption(details)
    show_sources(hits)
    return {"role": "assistant", "content": text, "hits": hits, "details": details}


# --- Startup checks ---

try:
    installed = installed_models()
except ConnectionError:
    st.error("Can't reach Ollama. Make sure it's running (open the Ollama app or run `ollama serve`), then refresh.")
    st.stop()

if with_tag(config.EMBED_MODEL) not in installed:
    st.warning(f"The embedding model **{config.EMBED_MODEL}** isn't downloaded yet. It's needed to index and search your notes.")
    if st.button(f"Download {config.EMBED_MODEL}", type="primary"):
        download_model(config.EMBED_MODEL)
        st.rerun()
    st.stop()

if get_collection().count() == 0:
    st.info("First run: indexing your vault. This takes a few minutes and only happens once.")
    load_index(on_progress=index_progress_bar())

retriever = get_retriever()


# --- Sidebar: settings ---

with st.sidebar:
    st.header("Settings")

    st.subheader("Model")
    chat_models = [m for m in installed if "embed" not in m]  # embedding models can't chat
    llm_options = list(dict.fromkeys([with_tag(config.LLM_MODEL), *SUGGESTED_LLMS, *chat_models]))
    llm = st.selectbox(
        "LLM", llm_options,
        format_func=lambda m: m if m in installed else f"{m} (not downloaded)",
        help="The model that writes the answers. Your default comes from LLM_MODEL in .env.",
    )
    llm_ready = llm in installed
    if not llm_ready:
        st.warning(f"{llm} isn't downloaded yet.")
        if st.button(f"Download {llm}", type="primary"):
            try:
                download_model(llm)
                st.rerun()
            except ollama.ResponseError as error:
                st.error(f"Download failed: {error}")

    st.divider()
    st.subheader("Retrieval")
    mode = st.selectbox(
        "Search method", list(MODES), format_func=MODES.get,
        index=list(MODES).index(config.RETRIEVAL_MODE), help=MODE_HELP,
    )
    top_k = st.slider("Chunks given to the LLM", min_value=1, max_value=10, value=config.TOP_K)

    st.divider()
    st.subheader("Vault")
    notes_col, chunks_col = st.columns(2)
    notes_col.metric("Notes", len(set(retriever.sources)))
    chunks_col.metric("Chunks", f"{len(retriever.ids):,}")
    st.caption(f"Vault: {VAULT_NAME}  \nEmbedding model: {config.EMBED_MODEL}")
    if st.button("Re-index vault", help="Run this after adding or editing notes"):
        rebuild_index(on_progress=index_progress_bar())
        get_retriever.clear()
        st.rerun()

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []


# --- Chat ---

st.title("Personal Knowledge RAG")
st.caption("Ask questions about your Obsidian notes. Search and answers run locally with Ollama.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "examples" not in st.session_state:
    notes = sorted(set(retriever.sources))
    st.session_state.examples = [f"What do my notes say about {Path(n).stem}?" for n in random.sample(notes, k=min(3, len(notes)))]

# Welcome panel, shown until the first question
example_question = None
welcome = st.empty()
if not st.session_state.messages:
    with welcome.container():
        steps = [
            ("1. Search", f"Finds the most relevant chunks out of {len(retriever.ids):,} in your vault."),
            ("2. Answer", "A local LLM answers using only those chunks."),
            ("3. Cite", "Shows the notes it used, with links that open them in Obsidian."),
        ]
        for column, (title, text) in zip(st.columns(3), steps):
            with column.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(text)

        st.markdown("**Try asking**")
        for example in st.session_state.examples:
            if st.button(example, disabled=not llm_ready):
                example_question = example

# Streamlit re-runs this script on every interaction, so redraw the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(message["details"])
            show_sources(message["hits"])

question = st.chat_input("Ask about your notes...", disabled=not llm_ready) or example_question
if question:
    welcome.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            st.session_state.messages.append(answer(question, llm, mode, top_k))
        except ConnectionError:
            st.error("Can't reach Ollama. Make sure it's running (open the Ollama app or run `ollama serve`).")
        except ollama.ResponseError as error:
            st.error(f"Ollama error: {error}")
