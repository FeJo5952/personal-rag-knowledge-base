# Personal Knowledge RAG — Project Documentation

> New to RAG? Read [learning.md](learning.md) first. It explains every concept used here.

## 1. Overview

A **local** question-answering system over my personal **Obsidian** notes. I ask a question in plain English. The system finds the most relevant parts of my notes and has a local LLM answer using only those parts, then shows which notes the answer came from.

Everything runs on my own computer: no cloud, no API keys, and my notes stay private.

**Resume line (target):**

> Built a local RAG system that ingests and embeds my personal Obsidian knowledge base, uses hybrid retrieval to provide relevant context to Llama 3.1 8B via Ollama, and generates responses with source attribution.

v1 builds the core pipeline. Hybrid retrieval is planned for v2 (see the Roadmap).

## 2. Goals (v1: simple working demo)

- Ingest all `.md` notes from my Obsidian vault into a vector database
- Ask questions from the terminal
- Get answers grounded in my notes
- See the source notes for every answer

Out of scope for v1: UI, hybrid search, incremental re-indexing, evaluation.

## 3. Tech stack

| Component       | Choice                                | Role                                                        |
| --------------- | ------------------------------------- | ----------------------------------------------------------- |
| Language        | **Python 3.11+**                      | All the glue code                                           |
| Model runner    | **Ollama**                            | Runs both AI models locally (`localhost:11434`)             |
| Generator LLM   | **Llama 3.1 8B** (`llama3.1:8b`)      | Writes the answer from the retrieved notes                  |
| Embedding model | **EmbeddingGemma** (`embeddinggemma`) | Turns chunks and questions into 768-number vectors          |
| Vector database | **ChromaDB**                          | Stores the vectors and finds the closest ones to a question |
| Python packages | `ollama`, `chromadb`                  | Clients for Ollama and Chroma                               |

No frameworks like LangChain. We write the pipeline ourselves to understand each step.

## 4. Architecture

### Ingestion (first run)

```text
Vault .md files → read → split into chunks → EmbeddingGemma → ChromaDB (./chroma_db)
```

### Query (each question)

```text
Question → EmbeddingGemma → ChromaDB top-5 chunks → prompt → Llama 3.1 8B → answer + sources
```

## 5. Code

`rag.py` is the terminal chat. The logic lives in the `app/` package:

- `config.py`: settings from `.env` (vault path, models, chunking, retrieval mode)
- `index.py`: `chunk_text()`, `embed_documents()` / `embed_query()`, `build_index()` and `load_index()` (ChromaDB)
- `retrieval.py`: the `Retriever` class with `dense_search`, `bm25_search`, `hybrid_search` (RRF) and `rerank`; `search()` picks the mode
- `generate.py`: `build_prompt()` and `generate()` (Llama via Ollama)

`eval/run_eval.py` scores any retrieval mode on the test set, and `eval/judge.py` grades answers with an LLM.

## 6. Setup: what to download and install

Do these in order. Each step has a check so you know it worked.

### Step 1: Python 3.11+

- Download: <https://www.python.org/downloads/>
- During install, tick **"Add Python to PATH"**.
- Check: `python --version`

### Step 2: Ollama for Windows

- Download: <https://ollama.com/download>
- Install it. It runs in the background (look for the llama icon in the system tray).
- Check: `ollama --version`

### Step 3: Download the two models

```bash
ollama pull llama3.1:8b       # ~4.9 GB, the generator
ollama pull embeddinggemma    # ~600 MB, the embedder
```

- Check: `ollama list` shows both.
- Quick test: `ollama run llama3.1:8b "Say hi in one sentence"`
- Optional: `nvidia-smi` shows your GPU's VRAM (4 GB or 8 GB affects speed, not whether it works).

### Step 4: Python virtual environment and packages

Run in the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell / cmd)
pip install -r requirements.txt
```

- Check: `python -c "import ollama, chromadb; print('ok')"`

### Step 5: Know your vault path

Find the folder of your Obsidian vault (the folder that contains the `.obsidian` subfolder). Put it in the `.env` file as `VAULT_PATH=...`.

## 7. Usage

```bash
python rag.py
```

The first run builds the index, then it asks for questions. To re-index after editing notes, delete `chroma_db/` and run again.

## 8. Key settings (v1 defaults)

| Setting       | Value            | Why                                             |
| ------------- | ---------------- | ----------------------------------------------- |
| Chunk size    | ~1000 characters | Focused but keeps context                       |
| Chunk overlap | ~150 characters  | Ideas aren't lost at chunk boundaries           |
| Top-k         | 5                | Enough context without overloading the 8B model |
| Distance      | cosine           | Standard for text embeddings                    |

## 9. Roadmap

- **Phase 1 (done)**: package refactor, personal evaluation set, baseline metrics (Hit@1, Hit@5, MRR, All@5, LLM-judge answer score)
- **Phase 2 (done)**: hybrid retrieval (BM25 + embeddings, Reciprocal Rank Fusion), optional cross-encoder reranker
- **Phase 3**: Streamlit UI
- **Phase 4**: LangGraph agent (note search, web search, calculator) and per-query latency/token logging

## 10. Progress log

| Date       | Step                                                              | Notes                                                            |
| ---------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| 2026-10-13 | Project planned; learning.md and documentation.md written         | Stack decided: Ollama + Llama 3.1 8B + EmbeddingGemma + ChromaDB |
| 2026-10-20 | Simplified to a single `rag.py` with an interactive terminal loop | Semantic retrieval only; hybrid search moved to the roadmap      |
| 2026-10-27 | Phase 1: `app/` package, 20-question personal eval set, baseline | Dense baseline: Hit@1 0.69, answer score 0.68 |
| 2026-10-30 | Phase 2: BM25, hybrid (RRF), rerankers via fastembed | Hybrid Hit@1 0.81; rerankers didn't help, so hybrid is the default |
