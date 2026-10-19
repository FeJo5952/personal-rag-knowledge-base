"""A simple local RAG over an Obsidian vault.

Run:  python rag.py
First run: reads the vault, chunks it, embeds the chunks and stores them in ChromaDB.
Then:      asks for questions in the terminal and answers them from your notes.
To re-index after editing notes, delete the chroma_db folder and run again.
"""

import os
from pathlib import Path

import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()
VAULT_PATH = os.environ["VAULT_PATH"]  # set in the .env file
EMBED_MODEL = "embeddinggemma"
LLM_MODEL = "llama3.1:8b"
CHUNK_SIZE = 1000                  # characters per chunk
CHUNK_OVERLAP = 150                # characters shared between neighbouring chunks
TOP_K = 5                          # how many chunks to give the LLM


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def build_index(collection):
    print("Indexing vault...")
    for file in Path(VAULT_PATH).rglob("*.md"):
        # skip hidden folders like .obsidian (settings) or tool folders, they aren't notes
        if any(part.startswith(".") for part in file.relative_to(VAULT_PATH).parts):
            continue
        source = str(file.relative_to(VAULT_PATH))
        chunks = chunk_text(file.read_text(encoding="utf-8"))
        if not chunks:
            continue
        collection.add(
            ids=[f"{source}-{i}" for i in range(len(chunks))],
            documents=chunks,
            # EmbeddingGemma expects this prefix for documents being stored
            embeddings=embed([f"title: none | text: {c}" for c in chunks]),
            metadatas=[{"source": source}] * len(chunks),
        )
        print(f"  {source}: {len(chunks)} chunks")
    print(f"Done. {collection.count()} chunks stored.\n")


def answer(collection, question):
    # 1. Retrieve: find the chunks closest in meaning to the question
    query_vector = embed([f"task: search result | query: {question}"])[0]
    results = collection.query(query_embeddings=[query_vector], n_results=TOP_K)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # 2. Augment: put those chunks into the prompt
    context = "\n\n".join(f"(from {s})\n{c}" for s, c in zip(sources, chunks))
    prompt = (
        "You answer questions about the user's personal notes. Use only the notes below, "
        "but you may draw reasonable conclusions from them. "
        "Only if nothing in the notes is relevant, say you couldn't find it.\n\n"
        f"Notes:\n{context}\n\nQuestion: {question}"
    )

    # 3. Generate: let the LLM write the answer
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    print("\n" + response["message"]["content"])
    print("\nSources:")
    for source in dict.fromkeys(sources):  # unique, in order
        print(f"  - {source}")


def main():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        build_index(collection)

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        answer(collection, question)


if __name__ == "__main__":
    main()
