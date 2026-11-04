"""Streamlit chat UI for the personal RAG. Runs locally, like rag.py.

Run:  streamlit run streamlit_app.py
"""

import time
from pathlib import Path
from urllib.parse import quote

import ollama
import streamlit as st

from app import config
from app.generate import chat_stream
from app.index import get_collection, load_index, rebuild_index
from app.pipeline import prepare
from app.retrieval import CANDIDATES, RRF_K, Retriever
from app.sql_safety import db_available

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
NOT_DOWNLOADED = "↓"
DEFAULT_CHUNKS = 4  # UI default; the eval keeps using TOP_K from .env

# Where answers come from (the router's choices) -> names shown in the UI
ROUTES = {"auto": "Auto", "notes": "Notes", "sql": "Database", "both": "Both"}
ROUTE_HELP = (
    "**Auto**: an LLM decides for each question.  \n"
    "**Notes**: search the text of your notes.  \n"
    "**Database**: the LLM writes a SQL query over the catalog of your vault (counts, lists, tasks, deadlines).  \n"
    "**Both**: a SQL result plus note search."
)

STEPS = [
    ("1. Route", "Decides where to look: your notes, the database of your vault, or both."),
    ("2. Retrieve", "Searches your notes and/or runs a read-only SQL query."),
    ("3. Answer & cite", "A local LLM answers from them and lists every source."),
]
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
    with st.expander(f"Notes: {len(unique_notes(hits))} notes, {len(hits)} chunks"):
        for i, hit in enumerate(hits, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. [{Path(hit.source).stem}]({obsidian_link(hit.source)})**")
                st.caption(f"{hit.source} · score {hit.score:.3f}")
                st.caption(hit.text[:400] + ("..." if len(hit.text) > 400 else ""))


def show_database(sql):
    """The SQL the LLM wrote and the rows it returned (or why it failed)."""
    label = f"Database: {len(sql.rows)} rows" if not sql.error else "Database: query failed"
    with st.expander(label, expanded=not sql.error):
        if sql.sql:
            st.code(sql.sql, language="sql")
        if sql.error:
            st.caption(f"Error: {sql.error}")
        elif sql.rows:
            st.dataframe([dict(zip(sql.columns, row)) for row in sql.rows], hide_index=True)
        attempts = f"{sql.attempts} attempt{'s' if sql.attempts > 1 else ''}"
        st.caption(f"Run as the read-only user · {attempts}" + (" · more rows exist" if sql.truncated else ""))


def show_answer_extras(message):
    """Route, notices, timings and sources under an answer."""
    st.caption(message["details"])
    for notice in message["notices"]:
        st.info(notice, icon=":material/info:")
    if message["sql"] and message["sql"].sql:
        show_database(message["sql"])
    if message["hits"]:
        show_sources(message["hits"])


def answer(question, llm, mode, top_k, route_choice):
    """Route, query and/or search, then stream the answer, with live progress in a status box."""
    status = st.status("Choosing where to look...")
    answer_area = st.container()  # created now so the answer appears below the status box

    try:
        start = time.perf_counter()
        result = prepare(question, retriever, route_choice=route_choice, k=top_k, mode=mode,
                         on_step=lambda text: status.update(label=text))
        prepare_time = time.perf_counter() - start
        status.write(f"Route: **{ROUTES[result.route.route]}** ({result.route.reason})")
        if result.sql and not result.sql.error:
            status.write(f"Database query returned {len(result.sql.rows)} rows")
        if result.hits:
            status.write(f"Found {len(result.hits)} chunks from {len(unique_notes(result.hits))} notes")

        if llm not in loaded_models():
            status.write(f"Loading {llm} into memory (the first answer takes longer)")
        status.update(label=f"Writing the answer with {llm}...")

        start = time.perf_counter()
        text = answer_area.write_stream(chat_stream(result.prompt, llm))  # shows the answer as it's written
        answer_time = time.perf_counter() - start
    except Exception:
        status.update(label="Something went wrong", state="error")
        raise
    status.update(label=f"Done in {prepare_time + answer_time:.1f}s", state="complete")

    steps = " · ".join(f"{step} {seconds:.1f}s" for step, seconds in result.timings.items())
    details = f"Route: {ROUTES[result.route.route]} · {steps} · answer {answer_time:.1f}s · {llm}"
    message = {"role": "assistant", "content": text, "hits": result.hits, "sql": result.sql,
               "notices": result.notices, "details": details}
    show_answer_extras(message)
    return message


@st.cache_data(ttl=30, show_spinner=False)
def database_status():
    """Checked at most every 30 s, so the sidebar doesn't connect on every click."""
    return db_available()


def show_about(llm, mode, top_k):
    """The current setup: every part of the pipeline, its settings, and where to change them."""
    st.title("About this setup")
    st.caption("A local RAG (retrieval-augmented generation) system over an Obsidian vault. "
               "This page shows how it's configured right now and where to change each part.")

    st.markdown(f"""
**Pipeline for each question**
1. The router (an LLM call) decides: notes, database, or both
2. Database: the LLM writes a SQL query, which runs as a read-only user
3. Notes: dense search (embeddings) and BM25 search (keywords) each return their top {CANDIDATES} chunks,
   Reciprocal Rank Fusion merges the two lists, and the cross-encoder re-scores them (Hybrid + Rerank only)
4. The query result and/or the top {top_k} chunks go to the LLM with the question
5. The LLM answers, and the database rows and notes it used are listed as sources
""")

    rerank_note = "in use" if mode == "hybrid_rerank" else "only used by Hybrid + Rerank"
    in_memory = ", ".join(loaded_models()) or "none right now"
    st.markdown(f"""
| Part | Current setup | Change it with |
|---|---|---|
| **Vault** | "{VAULT_NAME}": {len(set(retriever.sources))} Markdown notes (hidden folders skipped) | `VAULT_PATH` in `.env` |
| **Chunking** | {config.CHUNK_SIZE}-character chunks with {config.CHUNK_OVERLAP} overlap, {len(retriever.ids):,} chunks in total | `CHUNK_SIZE`, `CHUNK_OVERLAP`, then Re-index |
| **Embedding model** | `{config.EMBED_MODEL}` via Ollama | `EMBED_MODEL`, then Re-index |
| **Vector store** | ChromaDB on disk in `{config.CHROMA_DIR}/`, cosine similarity | `CHROMA_DIR` |
| **Keyword search** | BM25 (rank-bm25), common words ignored | |
| **Merging** | Reciprocal Rank Fusion of the top {CANDIDATES} from each search (k = {RRF_K}) | |
| **Reranker** | Cross-encoder `{config.RERANK_MODEL}` via fastembed on the CPU, {rerank_note} | `RERANK_MODEL` |
| **Search method** | {MODES[mode]} (default: {MODES[config.RETRIEVAL_MODE]}) | Sidebar, or `RETRIEVAL_MODE` |
| **Chunks given to the LLM** | {top_k} | Sidebar |
| **LLM** | `{llm}` via Ollama (default: `{config.LLM_MODEL}`) | Sidebar, or `LLM_MODEL` |
| **Router** | `{config.LLM_MODEL}` picks notes / database / both (keyword rules if it fails) | Sidebar "Answer from", `app/router.py` |
| **Database** | PostgreSQL catalog of the vault: courses, notes, tasks, frontmatter; {"connected" if database_status() else "not available"} | `DATABASE_URL`, `DATABASE_LOADER_URL` |
| **Text-to-SQL** | `{config.SQL_MODEL}` writes one SELECT over 3 views; one retry on error; {config.SQL_TIMEOUT_MS / 1000:g} s timeout, {config.SQL_MAX_ROWS} rows max | `SQL_MODEL`, `SQL_TIMEOUT_MS`, `SQL_MAX_ROWS` |
| **Answer rules** | Answer only from the retrieved notes and database rows; say so when nothing relevant is found | `app/generate.py` |
| **Eval judge** | `{config.JUDGE_MODEL}`, only used by `python -m eval.run_eval` | `JUDGE_MODEL` |
""")

    st.markdown(f"**Models in memory right now:** {in_memory}")
    st.caption("Everything runs on this machine. The internet is only used to download models "
               "(Ollama models, and the reranker from Hugging Face the first time it's used).")


def toggle_about():
    st.session_state.show_about = not st.session_state.get("show_about", False)


# --- Startup checks ---

try:
    installed = installed_models()
except ConnectionError:
    st.error("Can't reach Ollama. Make sure it's running (open the Ollama app or run `ollama serve`), then refresh.")
    st.stop()

if with_tag(config.EMBED_MODEL) not in installed:
    st.warning(f"The embedding model **{config.EMBED_MODEL}** isn't downloaded yet. It's needed to index and search your notes.")
    if st.button(f"Download {config.EMBED_MODEL}", icon=":material/download:", type="primary"):
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

    chat_models = [m for m in installed if "embed" not in m]  # embedding models can't chat
    llm_options = list(dict.fromkeys([with_tag(config.LLM_MODEL), *SUGGESTED_LLMS, *chat_models]))
    llm = st.selectbox(
        "LLM", llm_options,
        format_func=lambda m: m if m in installed else f"{m}  {NOT_DOWNLOADED}",
        help=f"The model that writes the answers. {NOT_DOWNLOADED} means not downloaded yet: select it to download it.",
    )
    llm_ready = llm in installed
    if not llm_ready:
        if st.button(f"Download {llm}", icon=":material/download:", type="primary"):
            try:
                download_model(llm)
                st.rerun()
            except ollama.ResponseError as error:
                st.error(f"Download failed: {error}")

    mode = st.selectbox(
        "Search method", list(MODES), format_func=MODES.get,
        index=list(MODES).index(config.RETRIEVAL_MODE), help=MODE_HELP,
    )
    top_k = st.slider("Chunks given to the LLM", min_value=1, max_value=10, value=DEFAULT_CHUNKS)

    database_up = database_status()
    route_choice = st.segmented_control(
        "Answer from", list(ROUTES), format_func=ROUTES.get, default="auto",
        disabled=not database_up, help=ROUTE_HELP,
    ) or "auto"
    if database_up:
        st.caption(":material/database: Database connected")
    else:
        st.caption(":material/database: Database not available: answering from notes only")

    reindex_col, clear_col = st.columns(2)
    reindex = reindex_col.button("Re-index", icon=":material/refresh:", help="Run this after adding or editing notes")
    if clear_col.button("Clear chat", icon=":material/delete:"):
        st.session_state.messages = []
    if reindex:
        rebuild_index(on_progress=index_progress_bar())
        get_retriever.clear()
        st.rerun()

    st.button(":material/help:", key="about_button", type="tertiary", help="About this setup", on_click=toggle_about)
    # Streamlit has no option to pin a button to the bottom, so one small CSS rule does it
    st.html("<style>.st-key-about_button { position: fixed; bottom: 1rem; }</style>")


# --- About page (opened with the ? button) ---

if st.session_state.get("show_about"):
    st.button("Back to chat", icon=":material/arrow_back:", type="tertiary", on_click=toggle_about)
    show_about(llm, mode, top_k)
    st.stop()


# --- Chat ---

st.title("Personal Knowledge RAG")
st.caption(f"Ask questions about your Obsidian vault **{VAULT_NAME}**. Search and answers run locally with Ollama.")

notes_col, chunks_col, mode_col = st.columns(3)
notes_col.metric("Notes", len(set(retriever.sources)))
chunks_col.metric("Chunks", f"{len(retriever.ids):,}")
mode_col.metric("Search method", MODES[mode])

if "messages" not in st.session_state:
    st.session_state.messages = []

# "What you can do" panel, shown until the first question
welcome = st.empty()
if not st.session_state.messages:
    with welcome.container():
        st.markdown("**What you can do**")
        for column, (title, text) in zip(st.columns(3), STEPS):
            with column.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(text)

# Streamlit re-runs this script on every interaction, so redraw the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_answer_extras(message)

question = st.chat_input("Ask about your notes...", disabled=not llm_ready)
if question:
    welcome.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            st.session_state.messages.append(answer(question, llm, mode, top_k, route_choice))
        except ConnectionError:
            st.error("Can't reach Ollama. Make sure it's running (open the Ollama app or run `ollama serve`).")
        except ollama.ResponseError as error:
            st.error(f"Ollama error: {error}")
