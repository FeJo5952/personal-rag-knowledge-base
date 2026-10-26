"""Retrieval: find the chunks most relevant to a question."""

from dataclasses import dataclass

from app import config
from app.index import embed_query


@dataclass
class Hit:
    text: str     # the chunk
    source: str   # note path inside the vault
    score: float  # higher = more relevant


def dense_search(collection, question, k):
    """Semantic search: the chunks whose embeddings are closest to the question's."""
    results = collection.query(query_embeddings=[embed_query(question)], n_results=k)
    return [
        Hit(text=doc, source=meta["source"], score=1 - dist)  # cosine distance -> similarity
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def retrieve(collection, question, k=config.TOP_K, mode="dense"):
    if mode == "dense":
        return dense_search(collection, question, k)
    raise ValueError(f"Unknown retrieval mode: {mode}")
