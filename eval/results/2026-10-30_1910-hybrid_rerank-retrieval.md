# Eval: mode=hybrid_rerank, k=5

- Date: 2026-10-30 19:10
- Run type: retrieval only (no answers generated, so no LLM calls and no answer score)
- Retrieval mode: hybrid_rerank | Reranker: Xenova/ms-marco-MiniLM-L-6-v2
- LLM: llama3.1:8b | Judge: qwen2.5:7b | Embeddings: embeddinggemma
- Questions: 20 | Runtime: 42s

| Questions | n | Hit@1 | Hit@5 | MRR | All@5 | Answer score |
|---|---|---|---|---|---|---|
| all | 20 | 0.75 | 0.94 | 0.82 | 0.88 | - |
| paraphrase | 6 | 0.67 | 1.00 | 0.79 | 1.00 | - |
| exact-term | 6 | 0.83 | 0.83 | 0.83 | 0.83 | - |
| multi-note | 4 | 0.75 | 1.00 | 0.83 | 0.75 | - |
| not-in-notes | 4 | - | - | - | - | - |
