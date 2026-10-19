# Personal Knowledge RAG

A simple, fully local RAG (Retrieval-Augmented Generation) system for an Obsidian vault. Ask a question in the terminal and it answers from your own notes, listing the notes it used.

Everything runs on your machine through Ollama: no cloud, no API keys.

## How it works

1. **Chunk**: read every `.md` file in the vault and split it into overlapping ~1000-character chunks.
2. **Embed**: turn each chunk into a vector with **EmbeddingGemma**.
3. **Store**: save the vectors, text and source file name in **ChromaDB**.
4. **Ask**: embed your question, find the 5 most similar chunks, and send them with the question to **Llama 3.1 8B**, which answers using only those notes.

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com/download).

```bash
ollama pull llama3.1:8b
ollama pull embeddinggemma

python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the path to your Obsidian vault:

```text
VAULT_PATH=C:\path\to\your\vault
```

## Run

```bash
python rag.py
```

The first run builds the index. After that, type questions:

```text
Ask a question (or 'exit'): Why did I stop running?
```

If you edit your notes, delete the `chroma_db` folder and run again to re-index.

## Stack

Python · Ollama · Llama 3.1 8B · EmbeddingGemma · ChromaDB

## Ideas for later

Heading-aware chunking, hybrid (keyword + semantic) search, re-indexing only changed notes, a web UI.
