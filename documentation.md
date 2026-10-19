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

Everything is in one file, [rag.py](rag.py), with settings as constants at the top:

- `chunk_text()` splits text into overlapping fixed-size chunks
- `embed()` turns text into vectors with EmbeddingGemma
- `build_index()` reads the vault, then chunks, embeds and stores everything in ChromaDB (first run only)
- `answer()` retrieves the top 5 chunks, builds the prompt, asks Llama, and prints the answer and sources
- `main()` loops: asks for a question in the terminal and answers it

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

- **v1: simple demo** (current): dense retrieval, CLI, source list
- **v2: hybrid retrieval**: BM25 keyword search + vector search, merged with Reciprocal Rank Fusion
- **v3: better answers**: heading-aware chunking, inline `[1]` citations, Obsidian links to sources
- **v4: usability**: Streamlit UI, incremental ingest (only changed notes)
- **v5: quality**: evaluation set with recall@k and MRR, reranker

## 10. Progress log

| Date       | Step                                                              | Notes                                                            |
| ---------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| 2026-10-13 | Project planned; learning.md and documentation.md written         | Stack decided: Ollama + Llama 3.1 8B + EmbeddingGemma + ChromaDB |
| 2026-10-20 | Simplified to a single `rag.py` with an interactive terminal loop | Semantic retrieval only; hybrid search moved to the roadmap      |
