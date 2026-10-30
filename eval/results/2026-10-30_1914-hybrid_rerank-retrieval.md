# Eval: mode=hybrid_rerank, k=5

- Date: 2026-10-30 19:14
- Run type: retrieval only (no answers generated, so no LLM calls and no answer score)
- Retrieval mode: hybrid_rerank | Reranker: BAAI/bge-reranker-base
- LLM: llama3.1:8b | Judge: qwen2.5:7b | Embeddings: embeddinggemma
- Questions: 20 | Runtime: 252s

| Questions | n | Hit@1 | Hit@5 | MRR | All@5 | Answer score |
|---|---|---|---|---|---|---|
| all | 20 | 0.69 | 0.94 | 0.77 | 0.81 | - |
| paraphrase | 6 | 0.67 | 0.83 | 0.71 | 0.67 | - |
| exact-term | 6 | 0.67 | 1.00 | 0.76 | 1.00 | - |
| multi-note | 4 | 0.75 | 1.00 | 0.88 | 0.75 | - |
| not-in-notes | 4 | - | - | - | - | - |
