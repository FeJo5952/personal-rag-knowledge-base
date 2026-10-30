# Eval: mode=hybrid, k=5

- Date: 2026-10-30 19:07
- Run type: retrieval only (no answers generated, so no LLM calls and no answer score)
- Retrieval mode: hybrid
- Note: BM25 without stopword removal (superseded by the later run)
- LLM: llama3.1:8b | Judge: qwen2.5:7b | Embeddings: embeddinggemma
- Questions: 20 | Runtime: 3s

| Questions | n | Hit@1 | Hit@5 | MRR | All@5 | Answer score |
|---|---|---|---|---|---|---|
| all | 20 | 0.69 | 0.88 | 0.77 | 0.75 | - |
| paraphrase | 6 | 0.50 | 0.83 | 0.64 | 0.83 | - |
| exact-term | 6 | 0.67 | 0.83 | 0.75 | 0.83 | - |
| multi-note | 4 | 1.00 | 1.00 | 1.00 | 0.50 | - |
| not-in-notes | 4 | - | - | - | - | - |
