# Personal Knowledge RAG

A fully local RAG (Retrieval-Augmented Generation) system for an Obsidian vault. Ask a question in the terminal and it answers from your own notes, listing the notes it used.

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

Copy `.env.example` to `.env` and set the path to your Obsidian vault:

```text
VAULT_PATH=C:\path\to\your\vault
```

Other settings (models, chunk size, top-k) are optional and listed in `.env.example`.

## Run

```bash
python rag.py
```

The first run builds the index. After that, type questions:

```text
Ask a question (or 'exit'): Why did I stop running?
```

If you edit your notes, delete the `chroma_db` folder and run again to re-index.

## Evaluation

The `eval/` folder measures how well the system works on a set of questions with known answers.

1. Create `eval/questions.jsonl` about your own notes, in the format of `eval/questions.example.jsonl`. It's gitignored because it describes personal notes.
2. Run:

```bash
python -m eval.run_eval --retrieval-only   # Hit@5 and MRR only (fast)
python -m eval.run_eval                    # also generates answers and grades them with an LLM judge
```

| Metric | Meaning |
|---|---|
| **Hit@5** | Share of questions where a correct source note is in the top 5 retrieved chunks |
| **MRR** | Mean reciprocal rank of the first correct source (1st = 1.0, 2nd = 0.5, ...) |
| **Answer score** | LLM-as-judge grade against the expected answer (correct 1, partial 0.5, wrong 0) |

The score summary goes to `eval/results/` and is safe to commit. Per-question details go to `eval/runs/`, which is gitignored.

## Project structure

```text
rag.py              terminal chat
app/
  config.py         settings from .env
  index.py          load vault, chunk, embed, store in ChromaDB
  retrieval.py      find relevant chunks
  generate.py       build the prompt and ask the LLM
eval/
  run_eval.py       evaluation runner
  judge.py          LLM-as-judge grading
  results/          score summaries
```

## Stack

Python · Ollama · Llama 3.1 8B · EmbeddingGemma · ChromaDB

## Roadmap

- [x] Evaluation set and baseline metrics
- [ ] Hybrid search (BM25 + embeddings) with a cross-encoder reranker
- [ ] Agent (LangGraph) that chooses between note search, web search and a calculator
- [ ] Streamlit UI and per-query latency/token logging
