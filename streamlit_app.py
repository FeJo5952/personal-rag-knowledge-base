"""Streamlit chat UI for the personal RAG. Runs locally, like rag.py.

Run:  streamlit run streamlit_app.py
"""

from pathlib import Path
from urllib.parse import quote

import streamlit as st

from app import config
from app.generate import generate_stream
from app.index import load_index, rebuild_index
from app.retrieval import Retriever

MODES = ["hybrid", "dense", "bm25", "hybrid_rerank"]
VAULT_NAME = Path(config.VAULT_PATH).name

st.set_page_config(page_title="Personal Knowledge RAG", page_icon="📚")


@st.cache_resource(show_spinner="Loading your notes (the first run builds the index, which can take a few minutes)...")
def get_retriever():
    # Cached so the index and BM25 are loaded once, not on every question
    return Retriever(load_index())


def obsidian_link(source):
    """A link that opens the note in the Obsidian app."""
    note = source.replace("\\", "/").removesuffix(".md")
    return f"obsidian://open?vault={quote(VAULT_NAME)}&file={quote(note)}"


def show_sources(hits):
    notes = list(dict.fromkeys(hit.source for hit in hits))  # unique, in ranked order
    with st.expander(f"Sources: {len(notes)} notes, {len(hits)} chunks"):
        for i, hit in enumerate(hits, start=1):
            st.markdown(f"**{i}. [{hit.source}]({obsidian_link(hit.source)})** · score {hit.score:.3f}")
            st.caption(hit.text[:400] + ("..." if len(hit.text) > 400 else ""))


# --- Sidebar: settings ---
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index(config.RETRIEVAL_MODE))
    top_k = st.slider("Chunks given to the LLM", min_value=1, max_value=10, value=config.TOP_K)

    st.caption(f"Vault: {VAULT_NAME}  \nLLM: {config.LLM_MODEL}  \nEmbeddings: {config.EMBED_MODEL}")

    if st.button("Re-index vault", help="Run this after adding or editing notes"):
        with st.spinner("Re-indexing your vault..."):
            rebuild_index()
            get_retriever.clear()
        st.success("Index rebuilt.")

    if st.button("Clear chat"):
        st.session_state.messages = []


# --- Chat ---
st.title("📚 Personal Knowledge RAG")
st.caption("Ask questions about your Obsidian notes. Everything runs locally on your machine.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit re-runs this script on every interaction, so redraw the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message["hits"])

if question := st.chat_input("Ask about your notes..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes..."):
                hits = get_retriever().search(question, k=top_k, mode=mode)
            answer = st.write_stream(generate_stream(question, hits))  # shows the answer as it's written
            show_sources(hits)
            st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
        except ConnectionError:
            st.error("Can't reach Ollama. Make sure it's running (open the Ollama app or run `ollama serve`).")
