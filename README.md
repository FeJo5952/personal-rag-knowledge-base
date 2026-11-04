# Personal Knowledge RAG

A fully local RAG (Retrieval-Augmented Generation) system for an Obsidian vault. Ask a question in a chat UI (or the terminal) and it answers from your own notes, listing the notes it used.

Everything runs on your machine through Ollama: no cloud, no API keys, and your notes are never uploaded.

## How it works

1. **Chunk**: read every `.md` file in the vault and split it into overlapping ~1000-character chunks.
2. **Embed**: turn each chunk into a vector with **EmbeddingGemma**.
3. **Store**: save the vectors, text and source file name in **ChromaDB**.
4. **Retrieve**: find the 5 most relevant chunks with **hybrid search**: semantic search (embeddings) and keyword search (BM25) run side by side, and their rankings are merged with Reciprocal Rank Fusion.
5. **Answer**: send those chunks with the question to **Llama 3.1 8B**, which answers using only those notes and lists its sources.

### Retrieval modes

Set `RETRIEVAL_MODE` in `.env`:

| Mode               | How it finds chunks                               | Good at                                         |
| ------------------ | ------------------------------------------------- | ----------------------------------------------- |
| `dense`            | Embedding similarity (ChromaDB)                   | Questions phrased differently from the note     |
| `bm25`             | Keyword matching, ignoring common words           | Exact names, codes, acronyms                    |
| `hybrid` (default) | Both, merged with Reciprocal Rank Fusion          | Both kinds of question                          |
| `hybrid_rerank`    | Hybrid, then a cross-encoder re-scores the top 20 | Optional; didn't help on my notes (see Results) |

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

### Choosing models

You can use **any Ollama model** that runs on your machine. Set them in `.env`:

| Setting       | Job                                                                   | Default          |
| ------------- | --------------------------------------------------------------------- | ---------------- |
| `EMBED_MODEL` | Turns notes and questions into vectors                                | `embeddinggemma` |
| `LLM_MODEL`   | Writes the answers                                                    | `llama3.1:8b`    |
| `JUDGE_MODEL` | Grades answers in the evaluation (best if different from `LLM_MODEL`) | `qwen2.5:7b`     |

Pull whatever you choose with `ollama pull <model>`. With a small GPU (4 GB), a 3B model such as `llama3.2:3b` or `qwen2.5:3b` is much faster than an 8B one. If you change `EMBED_MODEL`, delete the `chroma_db` folder so the notes are re-embedded with the new model.

Other settings (chunk size, top-k) are optional and listed in `.env.example`.

## Run

**Chat UI (recommended):**

```bash
streamlit run streamlit_app.py
```

This opens `http://localhost:8501` in your browser. The UI is served from your own machine, so it reads your local vault and Ollama; nothing is hosted online. In the app you can:

- chat with your notes, with live progress (searching, loading the model, writing) and the answer streamed as it's written
- see how long search and generation took under each answer
- expand **Sources** to see the exact chunks used, and click a note name to open it in Obsidian
- choose the LLM in the sidebar; models marked ↓ aren't pulled yet and can be downloaded from the UI with a progress bar
- pick the search method (Hybrid, Dense, BM25, Hybrid + Rerank) and number of chunks (default 4)
- see vault stats and click **Re-index** (with a progress bar) after adding or editing notes
- click **?** at the bottom of the sidebar for an About page showing the current setup (models, chunking, retrieval settings) and which `.env` setting changes each part

The embedding model is set only in `.env`, not in the UI: the index is built with it, so changing it means re-embedding the whole vault.

**Terminal:**

```bash
python rag.py
```

The first run of either one builds the index, which takes a few minutes. After editing notes, use **Re-index vault** in the UI (or delete the `chroma_db` folder).

## SQL catalog (PostgreSQL, optional)

Search finds the 5 most relevant chunks, so it can't **count or list everything**. "How many open tasks do I have per course?" needs every checkbox in the vault, not 5 chunks. So the vault's structure is also loaded into PostgreSQL. The schema is modelled on my own vault: IIT Madras course folders, university application notes with frontmatter, and task checkboxes.

| Table             | One row per     | Key columns                                                                                                      |
| ----------------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| `courses`         | course folder   | `code` (UNIQUE), `name`, `folder`                                                                                |
| `notes`           | Markdown note   | `path` (UNIQUE, same as the ChromaDB source), `kind`, `course_id` → courses, `week`, `modified_at`, `word_count` |
| `tasks`           | checkbox        | `note_id` → notes (ON DELETE CASCADE), `text`, `is_done`                                                         |
| `note_properties` | frontmatter key | (`note_id`, `key`) primary key, `value`, `value_date`                                                            |

- **Constraints:** primary and foreign keys, `UNIQUE`, `CHECK` (e.g. `kind IN (...)`, `week BETWEEN 0 AND 20`, forward-slash paths).
- **Views:** three views (`v_notes`, `v_tasks`, `v_universities`) join and pivot the tables. `v_universities` turns frontmatter rows into columns with `max(...) FILTER (WHERE key = ...)`.
- **Two database users:**
  - `rag_owner` loads the data.
  - `rag_reader` can only `SELECT` from the views, in read-only transactions with a 5 s timeout.

  So a bad or malicious generated query can't change anything: `INSERT`, `DELETE`, `CREATE` and reading the base tables are refused by the database itself.

**Setup:**

1. Install PostgreSQL.
2. Create the users and the database, with your own passwords:

   ```bash
   psql -U postgres -v owner_password=... -v reader_password=... -f sql/roles.sql
   ```

3. Put both URLs in `.env` (see `.env.example`).
4. Load the catalog. The tables are created on the first run:

   ```bash
   python -m app.sql_ingest            # --dry-run to preview without a database, --init to re-create the tables
   ```

After that the catalog is refreshed automatically on every re-index. It's rebuilt in one transaction with an integrity check, so a failed load changes nothing. With the URLs left empty, the app works exactly as before, notes search only.

## Evaluation

The `eval/` folder measures how well the system works on a set of questions with known answers.

1. Create `eval/questions.jsonl` about your own notes, in the format of `eval/questions.example.jsonl`. It's gitignored because it describes personal notes.
2. Run:

```bash
python -m eval.run_eval --retrieval-only   # retrieval metrics only (fast, no LLM)
python -m eval.run_eval                    # also generates answers and grades them with an LLM judge
```

All metrics range from 0 to 1, and higher is better.

| Metric           | Meaning                                                                                                 | How to read a score                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hit@1** (main) | Share of questions where the top retrieved chunk comes from a correct source note                       | **0.70** = the very first result is from the right note for 7 out of 10 questions. Below 0.5 is weak; 0.8+ is strong.                                                     |
| **Hit@5**        | Share of questions where a correct source note is in the top 5 retrieved chunks                         | **0.40** = for 6 out of 10 questions the right note isn't among the 5 chunks the LLM sees, so it can't answer them. Aim for 0.9+.                                         |
| **MRR**          | Mean reciprocal rank of the first correct source (1st = 1.0, 2nd = 0.5, 3rd = 0.33, ..., not found = 0) | **0.80** = the right note is usually ranked 1st, occasionally 2nd. **0.50** ≈ on average it's 2nd. MRR close to Hit@1 means "when found, it's at the top".                |
| **All@5**        | Share of questions where every needed note is in the top 5 (matters for multi-note questions)           | **0.75** = for 1 in 4 questions at least one needed note is missing, so the answer will be incomplete.                                                                    |
| **Answer score** | LLM-as-judge grade against the expected answer (correct 1, partial 0.5, wrong 0)                        | **0.68** ≈ about two-thirds of the answers are fully right, or many are partly right. A rough signal, since a small LLM judge is noisy; trust the retrieval metrics more. |

The score summary goes to `eval/results/` and is safe to commit. Per-question details go to `eval/runs/`, which is gitignored.

### Results

On my vault (119 notes, 1,129 chunks), with 20 personal questions: 16 answerable, 4 whose answer isn't in the notes.

| Retrieval mode                           | Hit@1    | Hit@5    | MRR      | All@5    |
| ---------------------------------------- | -------- | -------- | -------- | -------- |
| dense (baseline)                         | 0.69     | 0.94     | 0.79     | 0.88     |
| bm25                                     | 0.50     | 0.69     | 0.56     | 0.50     |
| **hybrid**                               | **0.81** | **1.00** | **0.91** | **0.94** |
| hybrid + rerank (ms-marco-MiniLM-L-6)    | 0.75     | 0.94     | 0.82     | 0.88     |
| hybrid + rerank (jina-reranker-v1-turbo) | 0.75     | 1.00     | 0.84     | 0.88     |
| hybrid + rerank (bge-reranker-base)      | 0.69     | 0.94     | 0.77     | 0.81     |

End-to-end answer quality (Llama 3.1 8B answers, Qwen 2.5 7B judges):

| Retrieval mode   | Answer score | Paraphrase | Exact-term | Multi-note | Not in notes (correctly refused) |
| ---------------- | ------------ | ---------- | ---------- | ---------- | -------------------------------- |
| dense (baseline) | 0.68         | 0.50       | 0.67       | 0.62       | 1.00                             |
| **hybrid**       | **0.80**     | **0.83**   | **0.83**   | 0.50       | 1.00                             |

- **Hybrid search is the clear winner:** the right note is ranked first for 81% of questions (up from 69%) and is always in the top 5. Better retrieval carried through to the answers: the answer score rose from 0.68 to 0.80.
- **Multi-note questions are the weak spot** (answer score 0.50): the LLM gets the right notes but doesn't always combine facts from several of them. That's a target for the agent phase.
- **Removing stopwords mattered for BM25.** Without it, long code-heavy notes matched casual questions on words like "what" and "how", and hybrid only tied with dense (Hit@1 0.69).
- **The vault grew to 138 notes (1,737 chunks).** Hybrid retrieval re-run on it: Hit@1 0.81, Hit@5 1.00, MRR 0.89, All@5 0.94, so quality held as the vault grew.
- **None of the three rerankers improved on hybrid** and each added noticeable latency, so `hybrid` is the default. They're general web-search models, and the eval set is small (one question = 0.06), so treat small differences with caution.

## Project structure

```text
streamlit_app.py    chat UI (runs locally)
rag.py              terminal chat
app/
  config.py         settings from .env
  index.py          load vault, chunk, embed, store in ChromaDB
  retrieval.py      dense, BM25, hybrid (RRF) and reranked search
  generate.py       build the prompt and ask the LLM
  sql_ingest.py     load courses, notes, tasks and frontmatter into PostgreSQL
sql/
  roles.sql         database users (owner + read-only reader) and the database
  schema.sql        tables, constraints, views and grants
eval/
  run_eval.py       evaluation runner
  judge.py          LLM-as-judge grading
  results/          score summaries
```

## Stack

Python · Ollama · Llama 3.1 8B · EmbeddingGemma · ChromaDB · BM25 (rank-bm25) · fastembed (reranker) · PostgreSQL (psycopg) · Streamlit

## Roadmap

- [x] Evaluation set and baseline metrics
- [x] Hybrid search (BM25 + embeddings) with an optional cross-encoder reranker
- [ ] Agent (LangGraph) that chooses between note search, web search and a calculator
- [x] Streamlit chat UI
- [ ] Per-query latency/token logging
