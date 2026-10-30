"""Retrieval: find the chunks most relevant to a question.

Four modes:
    dense          semantic search with embeddings (ChromaDB)
    bm25           keyword search
    hybrid         dense + bm25, merged with Reciprocal Rank Fusion (RRF)
    hybrid_rerank  hybrid, then a cross-encoder re-scores the candidates
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app import config
from app.index import embed_query

CANDIDATES = 20  # how many results each method returns before merging / reranking
RRF_K = 60       # standard constant in the RRF formula


@dataclass
class Hit:
    id: str       # chunk id in ChromaDB
    text: str     # the chunk
    source: str   # note path inside the vault
    score: float  # higher = more relevant (scale depends on the method)


def tokenize(text):
    """Lowercase words, e.g. "Fixed __reduce_ex__ in v2" -> ["fixed", "__reduce_ex__", "in", "v2"]."""
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, collection):
        self.collection = collection

        # BM25 needs every chunk's text, so load them once from ChromaDB
        data = collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.texts = data["documents"]
        self.sources = [meta["source"] for meta in data["metadatas"]]
        self.bm25 = BM25Okapi([tokenize(text) for text in self.texts])

        self.reranker = None  # loaded the first time it's needed (takes a few seconds)

    def dense_search(self, question, k):
        """Semantic search: chunks whose embeddings are closest to the question's."""
        results = self.collection.query(query_embeddings=[embed_query(question)], n_results=k)
        return [
            Hit(id=chunk_id, text=text, source=meta["source"], score=1 - distance)
            for chunk_id, text, meta, distance in zip(
                results["ids"][0], results["documents"][0],
                results["metadatas"][0], results["distances"][0],
            )
        ]

    def bm25_search(self, question, k):
        """Keyword search: chunks that share the most (rare) words with the question."""
        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Hit(id=self.ids[i], text=self.texts[i], source=self.sources[i], score=scores[i])
            for i in best
            if scores[i] > 0  # no shared words = not a match
        ]

    def hybrid_search(self, question, k):
        """Run both searches and merge their rankings with Reciprocal Rank Fusion.

        Each chunk gets 1 / (RRF_K + rank) from every list it appears in, so a chunk
        ranked well by both methods beats one ranked well by only one.
        """
        rankings = [
            self.dense_search(question, CANDIDATES),
            self.bm25_search(question, CANDIDATES),
        ]
        rrf_scores = {}
        hits_by_id = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                rrf_scores[hit.id] = rrf_scores.get(hit.id, 0) + 1 / (RRF_K + rank)
                hits_by_id[hit.id] = hit

        best_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        hits = [hits_by_id[chunk_id] for chunk_id in best_ids]
        for hit in hits:
            hit.score = rrf_scores[hit.id]
        return hits

    def rerank(self, question, hits, k):
        """Re-score each (question, chunk) pair with a cross-encoder and keep the best k.

        A cross-encoder reads the question and the chunk together, which is more
        accurate than comparing two separate embeddings, but too slow to run on
        every chunk. So it only re-orders the candidates the other methods found.
        """
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(config.RERANK_MODEL)

        scores = self.reranker.predict([(question, hit.text) for hit in hits])
        for hit, score in zip(hits, scores):
            hit.score = float(score)
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:k]

    def search(self, question, k=config.TOP_K, mode=config.RETRIEVAL_MODE):
        if mode == "dense":
            return self.dense_search(question, k)
        if mode == "bm25":
            return self.bm25_search(question, k)
        if mode == "hybrid":
            return self.hybrid_search(question, k)
        if mode == "hybrid_rerank":
            candidates = self.hybrid_search(question, CANDIDATES)
            return self.rerank(question, candidates, k)
        raise ValueError(f"Unknown retrieval mode: {mode}")
