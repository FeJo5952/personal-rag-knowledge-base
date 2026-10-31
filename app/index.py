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


def uses_gemma_prefixes():
    # EmbeddingGemma was trained with task prefixes; other embedding models don't need them
    return "embeddinggemma" in config.EMBED_MODEL


def embed_documents(chunks):
    if uses_gemma_prefixes():
        chunks = [f"title: none | text: {c}" for c in chunks]
    return embed(chunks)


def embed_query(question):
    if uses_gemma_prefixes():
        question = f"task: search result | query: {question}"
    return embed([question])[0]


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})


def note_files():
    vault = Path(config.VAULT_PATH)
    for file in vault.rglob("*.md"):
        # skip hidden folders like .obsidian (settings) or tool folders, they aren't notes
        if not any(part.startswith(".") for part in file.relative_to(vault).parts):
            yield file


def build_index(collection, on_progress=None):
    """Chunk and embed every note. on_progress(done, total, note) lets a UI show a progress bar."""
    print("Indexing vault...")
    files = list(note_files())
    for done, file in enumerate(files, start=1):
        source = str(file.relative_to(config.VAULT_PATH))
        if on_progress:
            on_progress(done, len(files), source)
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


def load_index(on_progress=None):
    """Open the index, building it first if it's empty."""
    collection = get_collection()
    if collection.count() == 0:
        build_index(collection, on_progress)
    return collection


def rebuild_index(on_progress=None):
    """Delete the stored index and build it again (use after editing notes)."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    client.delete_collection("notes")
    return load_index(on_progress)
