# Learning RAG — From Zero to How This Project Works

This file explains every idea behind the project, step by step. Read it top to bottom once, then use it as a reference.

---

## 1. The problem we are solving

Large Language Models (LLMs) like ChatGPT or Llama are trained on huge amounts of public text. That gives them three big weaknesses:

1. **They don't know your private data.** Llama has never seen your Obsidian notes, so it can't answer "What did I write about project X last month?"
2. **Their knowledge is frozen.** A model only knows what was in its training data, up to a cutoff date. Anything newer is invisible to it.
3. **They hallucinate.** When a model doesn't know something, it often makes up a confident-sounding answer instead of saying "I don't know."

We want a system that can answer questions **about our own notes**, accurately, and **show us which notes the answer came from** so we can check it.

---

## 2. What is RAG?

**RAG = Retrieval-Augmented Generation.**

Break the name down:

- **Retrieval**: search your own documents and find the pieces relevant to the question.
- **Augmented**: add (augment) those pieces into the prompt you send to the LLM.
- **Generation**: the LLM generates an answer using those pieces.

### Analogy: the open-book exam

- A normal LLM is a student taking a **closed-book exam**. It answers from memory, and when memory fails it guesses.
- A RAG system is a student taking an **open-book exam**. Before answering, it flips to the right pages of the textbook (your notes), reads them, and writes an answer based on what's on the page. It can also say "the answer is on page 42."

The LLM itself doesn't change at all. We just give it the right pages to read every time we ask a question.

---

## 3. Why RAG instead of other options?

There are three ways to make an LLM "know" your notes:

| Option | How it works | Problems |
|---|---|---|
| **Paste everything into the prompt** | Copy all notes into every question | 200+ notes is far too much text. Models have a limited *context window* (max input size). It's also slow, and the model gets confused by irrelevant text. |
| **Fine-tuning** | Re-train the model on your notes | Expensive, needs a strong GPU, and has to be redone every time you edit a note. The model still hallucinates and can't tell you *which* note a fact came from. |
| **RAG** ✅ | Search for the few relevant pieces, paste only those | Cheap, fast, always up to date (just re-index changed notes), and you can show sources. |

RAG is the standard approach for "chat with my documents" systems.

---

## 4. Embeddings — how a computer understands "meaning"

This is the most important idea in RAG.

### The problem with keyword search

If your note says *"I adopted a golden retriever"* and you ask *"Do I have a dog?"*, a keyword search for "dog" finds nothing, because the word "dog" isn't in the note. We need search that understands **meaning**, not just exact words.

### What an embedding is

An **embedding** is a list of numbers (a **vector**) that represents the *meaning* of a piece of text.

```text
"I adopted a golden retriever"  →  [0.12, -0.48, 0.91, ..., 0.05]   (768 numbers)
"Do I have a dog?"              →  [0.10, -0.45, 0.88, ..., 0.07]   (768 numbers)
"The stock market fell today"   →  [-0.71, 0.33, -0.02, ..., 0.64]  (768 numbers)
```

An **embedding model** is a neural network trained so that:

- texts with **similar meaning** get vectors that are **close together**
- texts with **different meaning** get vectors that are **far apart**

Think of each vector as a point on a map of meaning. "Golden retriever" and "dog" are neighbours, and "stock market" is in a different region. The real map has 768 dimensions instead of 2, but the idea is the same.

### Measuring "closeness": cosine similarity

To compare two vectors we usually use **cosine similarity**. It measures the angle between them:

- **1.0** → pointing the same direction → same meaning
- **0.0** → unrelated
- **−1.0** → opposite

A tiny example with 2-number vectors:

```text
A = [1, 0]     B = [0.9, 0.1]     C = [0, 1]

cosine(A, B) ≈ 0.99   → very similar
cosine(A, C) = 0.0    → unrelated
```

Some systems talk about **distance** instead of similarity. It's the same idea flipped around: *smaller distance = more similar*.

### How RAG uses embeddings

1. Embed every chunk of every note **once** and store the vectors.
2. When you ask a question, embed the question.
3. Find the stored vectors **closest** to the question vector. Those chunks are the most relevant pieces of your notes.

This is called **semantic search** or **dense retrieval**.

---

## 5. Chunking — why we split notes into pieces

We don't embed whole notes. We split them into **chunks** (e.g. ~1000 characters each). Why?

1. **Precision.** A long note about "2024 goals" might cover fitness, money and career. One vector for the whole note blurs all of that together. Separate chunks keep each topic sharp.
2. **Prompt size.** We paste the retrieved text into the LLM prompt. Five focused chunks fit easily, while five whole notes may not.
3. **Model limits.** Embedding models have a maximum input length (EmbeddingGemma: 2048 tokens). Text beyond that is cut off.

### Chunk size trade-off

| Too small (e.g. 100 chars) | Too large (e.g. 10,000 chars) |
|---|---|
| Loses context ("He agreed." Who agreed to what?) | Meaning gets diluted, and it wastes prompt space |

A few hundred to ~1000 characters is a common starting point. We'll start simple and tune it later.

### Overlap

If we cut strictly every 1000 characters, a sentence might be split in half across two chunks. **Overlap** (e.g. each chunk repeats the last 100–200 characters of the previous one) makes sure no idea is lost at a boundary.

### Smarter chunking (later)

Obsidian notes have headings (`#`, `##`). Splitting at headings gives chunks that match how you organised your thoughts. We'll start with simple fixed-size chunks and improve this later.

---

## 6. Vector databases and ChromaDB

### Why we need a special database

After ingesting 200+ notes we may have thousands of chunks, each with a 768-number vector. For every question we need to find the closest vectors **fast**. A normal database (like a spreadsheet or SQL) isn't built for "find the nearest vectors." A **vector database** is.

### What ChromaDB stores

Chroma organises data into **collections** (like a table). Each record in a collection has:

| Field | Example | Purpose |
|---|---|---|
| **id** | `"notes/dogs.md::chunk-0"` | Unique name for the chunk |
| **embedding** | `[0.12, -0.48, ...]` | The vector, used for search |
| **document** | `"I adopted a golden retriever..."` | The original text, which we give to the LLM |
| **metadata** | `{"source": "notes/dogs.md"}` | Extra info, which we use for source attribution |

### How the search works

You give Chroma a query vector and ask for the top *k* (e.g. 5). It returns the 5 closest chunks with their text, metadata and distance. To do this fast it uses an index called **HNSW**, a clever graph structure that finds *approximately* nearest neighbours without comparing against every single vector.

### Why Chroma for this project

- Runs **locally** inside your Python program. There's no server to set up and no account.
- **Persistent**: `chromadb.PersistentClient(path="./chroma_db")` saves everything to a folder, so you ingest once and query many times.
- Simple Python API. It's ideal for learning and for small-to-medium projects.

---

## 7. Ollama — running AI models on your own computer

### What it is

**Ollama** is a free program that downloads and runs open-source AI models **locally** on your machine. Think of it as an "app store + engine" for LLMs.

- `ollama pull llama3.1:8b` downloads a model
- `ollama run llama3.1:8b` starts a chat in the terminal
- `ollama list` shows the models you've downloaded

### How our Python code talks to it

When Ollama is running, it starts a small **local server** at `http://localhost:11434`. Our Python code uses the `ollama` package to send requests to it:

- `ollama.embed(model="embeddinggemma", input=...)` returns vectors
- `ollama.chat(model="llama3.1:8b", messages=[...])` returns a generated answer

"localhost" means *this computer*. Nothing goes to the internet.

### Why local?

- **Privacy**: your personal notes never leave your machine.
- **Free**: no API bills.
- **Offline**: works without internet once the models are downloaded.

### Quantization (briefly)

A model's "weights" are billions of numbers. Stored at full precision, Llama 3.1 8B would need ~16 GB of memory. **Quantization** stores each number with fewer bits (e.g. 4 bits instead of 16), which shrinks it to ~5 GB with only a small quality loss. Ollama's default download of `llama3.1:8b` is already quantized (4-bit), which is why it fits on a consumer GPU.

---

## 8. Llama 3.1 8B — the "generator"

- Made by **Meta**, open-weights (free to download and run).
- **8B = 8 billion parameters.** Parameters are the learned numbers inside the network. More parameters usually means smarter but slower and more memory-hungry. 8B is a good balance for a laptop or desktop GPU.
- It's a **chat/instruction model**: give it instructions plus text and it writes a response.
- Its job in our system is to **read the retrieved chunks and write a clear answer**. It does not search. Retrieval is done before it ever sees the question.

### Your GPU (RTX 3050)

The quantized model is ~4.9 GB.

- **8 GB RTX 3050** (desktop): the model fits entirely in GPU memory, so it's fast.
- **4 GB RTX 3050** (laptop): Ollama automatically puts part of the model on the GPU and the rest on the CPU and system RAM. It still works, just slower (a few words per second).

Either way it runs. Check your VRAM with `nvidia-smi` in a terminal.

---

## 9. EmbeddingGemma — the "embedder"

- Made by **Google**, open-weights, available in Ollama as `embeddinggemma`.
- Small: ~300M parameters (~600 MB). It runs fast even next to Llama.
- Outputs a **768-number vector** per piece of text.
- Max input: **2048 tokens** (roughly 1500 words), which is plenty for our chunks.

### Embedding model vs chat model

| | Embedding model (EmbeddingGemma) | Chat model (Llama 3.1) |
|---|---|---|
| Input | Text | Text |
| Output | **A list of numbers** (vector) | **New text** (an answer) |
| Used for | Searching / comparing meaning | Writing responses |

They're different tools for different jobs. RAG needs both.

### Task prefixes (important for EmbeddingGemma)

EmbeddingGemma was trained to expect a short **prefix** telling it what kind of text it is embedding. Using them noticeably improves search quality:

```text
When storing a note chunk:   "title: none | text: <chunk text>"
When embedding a question:   "task: search result | query: <your question>"
```

Questions and documents look different (a short question vs a long paragraph). The prefixes help the model place them correctly on the meaning map.

> **Rule:** the same embedding model must be used for both chunks and questions. Vectors from different models live on different "maps" and can't be compared.

---

## 10. The prompt — combining question and context

After retrieval we have, say, 5 relevant chunks. We build a prompt like this:

```text
SYSTEM:
You are an assistant that answers questions using ONLY the provided notes.
If the notes do not contain the answer, say "I couldn't find this in your notes."

USER:
Notes:
---
[Source: health/fitness.md]
I started running 5km three times a week in March...
---
[Source: journal/2024-04.md]
Knee started hurting after long runs, switched to cycling...
---

Question: What exercise am I doing now?
```

Why each part matters:

- **"ONLY the provided notes"** stops the model from mixing in made-up or general knowledge (this reduces hallucination).
- **"If not found, say so"** gives the model permission to say "I don't know" instead of guessing.
- **Source labels** let the model (and us) know where each piece came from.

This step is called **grounding**: we tie the answer to real retrieved text.

---

## 11. How everything connects

RAG has two phases.

### Phase A: Ingestion (run once, and again when notes change)

```text
 Obsidian vault (.md files)
          │
          ▼
   [1] Load files            ← Python reads every .md file
          │
          ▼
   [2] Split into chunks     ← ~1000 chars, with overlap
          │
          ▼
   [3] Embed each chunk      ← Ollama + EmbeddingGemma → vectors
          │
          ▼
   [4] Store in ChromaDB     ← id + vector + text + metadata(source)
          │
          ▼
     ./chroma_db folder (saved on disk)
```

### Phase B: Querying (every time you ask a question)

```text
   Your question: "What exercise am I doing now?"
          │
          ▼
   [1] Embed the question    ← Ollama + EmbeddingGemma → query vector
          │
          ▼
   [2] Search ChromaDB       ← find top 5 closest chunk vectors
          │
          ▼
   [3] Build the prompt      ← instructions + 5 chunks + question
          │
          ▼
   [4] Generate answer       ← Ollama + Llama 3.1 8B
          │
          ▼
   [5] Show answer + sources ← sources come from chunk metadata
```

### Who does what

| Component | Role | Phase |
|---|---|---|
| **Python (our code)** | The glue: reads files, chunks, calls the models and DB, builds the prompt | Both |
| **Ollama** | Runs both AI models locally | Both |
| **EmbeddingGemma** | Turns text into vectors | Both (chunks in A, the question in B) |
| **ChromaDB** | Stores vectors and finds the nearest ones | Stores in A, searches in B |
| **Llama 3.1 8B** | Reads the context and writes the answer | B only |

---

## 12. Source attribution

When we stored each chunk, we saved metadata like `{"source": "health/fitness.md"}`. When Chroma returns the top chunks, that metadata comes back with them. So after Llama answers, we simply print:

```text
Answer: You switched from running to cycling because of knee pain.

Sources:
  - health/fitness.md
  - journal/2024-04.md
```

This lets you **verify** the answer by opening the original note. That's the key trust feature of RAG.

---

## 13. Limitations and future improvements

Our first version is deliberately simple. Known weaknesses and how we'll fix them later:

| Limitation | Improvement |
|---|---|
| Semantic search can miss **exact terms** (names, codes, rare words) | **Hybrid retrieval**: combine vector search with keyword search (BM25), merged with *Reciprocal Rank Fusion* |
| Top results aren't always perfectly ordered | **Reranking**: a second model re-scores the top ~20 chunks more carefully |
| Fixed-size chunks may cut through topics | **Heading-aware chunking** for Markdown |
| Re-ingesting everything is slow | **Incremental ingest**: only re-embed changed notes |
| "Is it actually good?" is hard to answer | **Evaluation**: a set of test questions plus metrics like recall@k |
| Sources are listed but not tied to sentences | **Inline citations** like `[1]`, `[2]` in the answer |
| Terminal only | **Streamlit** web UI |

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **LLM** | Large Language Model: a neural network that generates text |
| **RAG** | Retrieval-Augmented Generation: retrieve relevant text, add it to the prompt, then generate |
| **Embedding** | A vector (list of numbers) representing the meaning of text |
| **Vector** | An ordered list of numbers, e.g. `[0.1, -0.3, 0.8]` |
| **Embedding model** | A model that converts text to embeddings (here, EmbeddingGemma) |
| **Cosine similarity** | A measure of how similar two vectors are (1 = same direction) |
| **Chunk** | A small piece of a document that gets embedded and retrieved |
| **Overlap** | Text repeated between neighbouring chunks so nothing is lost at the boundaries |
| **Vector database** | A database optimised for storing vectors and finding the nearest ones (here, ChromaDB) |
| **Collection** | A group of records in Chroma, like a table |
| **Metadata** | Extra info stored with a chunk (e.g. its source file) |
| **Top-k** | The *k* most similar results returned by a search |
| **HNSW** | The fast approximate nearest-neighbour index Chroma uses |
| **Ollama** | A tool to download and run LLMs locally, served at `localhost:11434` |
| **Parameters** | The learned numbers inside a model (8B = 8 billion) |
| **Quantization** | Storing model weights with fewer bits to save memory |
| **VRAM** | GPU memory; models run fastest when they fit in it |
| **Token** | A piece of a word that models read and write (~¾ of an English word) |
| **Context window** | The maximum number of tokens a model can take as input at once |
| **Prompt** | The full text/instructions sent to the LLM |
| **Grounding** | Making the LLM answer only from the provided context |
| **Hallucination** | When an LLM states something false with confidence |
| **Ingestion** | The pipeline that loads, chunks, embeds and stores documents |
| **Dense retrieval** | Search using embeddings (meaning-based) |
| **Sparse / keyword retrieval (BM25)** | Search using word matches (exact-term-based) |
| **Hybrid retrieval** | Combining dense and sparse search |
| **Source attribution** | Showing which documents an answer came from |
| **Hit@k** | Share of questions where a correct source note appears in the top *k* retrieved chunks |
| **MRR** | Mean Reciprocal Rank: average of 1/rank of the first correct result |
| **LLM-as-judge** | Using an LLM to grade another LLM's answer against an expected answer |

---

## 15. Evaluating a RAG system

"It seems to work" isn't a measurement. To know whether a change (like hybrid search) actually helps, we need **numbers**, measured the same way **before and after** the change.

### The test set (a "golden dataset")

A list of questions where **we already know the right answer and which note contains it**:

```json
{"question": "Why did I switch to cycling?", "answer": "Knee injury", "sources": ["Health/Fitness.md"], "type": "paraphrase"}
```

The `type` lets us see *where* the system is weak:

| Type | What it tests |
|---|---|
| `paraphrase` | Question uses different words than the note. Tests **meaning**-based search. |
| `exact-term` | Question hinges on a specific name, acronym or code. Tests **keyword** matching. Semantic search often struggles here. |
| `multi-note` | The answer is spread across several notes. |
| `not-in-notes` | The answer isn't in the vault. The system should say so instead of making something up. |

Ours lives in `eval/questions.jsonl`. It's **not committed to git** because it's about personal notes.

### Two separate things to measure

A RAG answer can go wrong in two places, so we measure them separately:

```text
question → [RETRIEVAL] → chunks → [GENERATION] → answer
              ▲                        ▲
         Hit@k, MRR              Answer score (LLM judge)
```

If retrieval doesn't find the right note, even a perfect LLM can't answer. If retrieval finds it but the answer is still wrong, the problem is the prompt or the model.

### Retrieval metrics

**Hit@k:** did the correct note appear anywhere in the top *k* (we use k = 5)?

```text
Q1: correct note at rank 1  → hit
Q2: correct note at rank 4  → hit
Q3: correct note not in top 5 → miss
Hit@5 = 2/3 = 0.67
```

**MRR (Mean Reciprocal Rank):** *how high* was the first correct result? Each question scores 1/rank:

```text
Q1: rank 1 → 1/1 = 1.00
Q2: rank 4 → 1/4 = 0.25
Q3: missing → 0
MRR = (1.00 + 0.25 + 0) / 3 = 0.42
```

Hit@k says "did we find it?", while MRR also rewards putting it **at the top**. Both are cheap to compute and objective (no LLM needed), which is why they are our headline numbers.

### Answer metric: LLM-as-judge

Checking whether an answer is correct is hard to automate with code: "runner's knee" and "a knee injury from running" mean the same thing. So we ask an LLM to **grade** it:

- it gets the question, the **expected** answer and the **generated** answer
- it replies `correct` (1), `partial` (0.5) or `wrong` (0)
- for `not-in-notes` questions, it checks whether the system admitted it couldn't find the answer

We set `temperature: 0` (no randomness) and force JSON output so grades are consistent and machine-readable.

**Caveat:** a small local judge (8B) is noisy and sometimes too generous or too strict. Treat the answer score as a rough signal, and the retrieval metrics as the reliable ones. Spot-check a few grades by hand in `eval/runs/`.

### Baseline first

We measure the current system **before** improving it. That number is the *baseline*. Every later change (hybrid search, reranking, the agent) is run on the **same questions** so the comparison is fair:

| Version | Hit@5 | MRR | Answer |
|---|---|---|---|
| Dense (baseline) | ? | ? | ? |
| + Hybrid | ? | ? | ? |
| + Reranker | ? | ? | ? |

Filling in this table is the goal of the next phases.
