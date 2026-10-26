"""Indexing: read the vault, chunk the notes, embed the chunks, store them in ChromaDB."""

from pathlib import Path

import chromadb
import ollama

from app import config


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + config.CHUNK_SIZE])
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return chunks


def embed(texts):
    return ollama.embed(model=config.EMBED_MODEL, input=texts)["embeddings"]


def embed_documents(chunks):
    # EmbeddingGemma expects this prefix for documents being stored
    return embed([f"title: none | text: {c}" for c in chunks])


def embed_query(question):
    # ...and this prefix for search queries
    return embed([f"task: search result | query: {question}"])[0]


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})


def note_files():
    vault = Path(config.VAULT_PATH)
    for file in vault.rglob("*.md"):
        # skip hidden folders like .obsidian (settings) or tool folders, they aren't notes
        if not any(part.startswith(".") for part in file.relative_to(vault).parts):
            yield file


def build_index(collection):
    print("Indexing vault...")
    for file in note_files():
        source = str(file.relative_to(config.VAULT_PATH))
        chunks = chunk_text(file.read_text(encoding="utf-8"))
        if not chunks:
            continue
        collection.add(
            ids=[f"{source}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embed_documents(chunks),
            metadatas=[{"source": source}] * len(chunks),
        )
        print(f"  {source}: {len(chunks)} chunks")
    print(f"Done. {collection.count()} chunks stored.\n")


def load_index():
    """Open the index, building it first if it's empty."""
    collection = get_collection()
    if collection.count() == 0:
        build_index(collection)
    return collection
