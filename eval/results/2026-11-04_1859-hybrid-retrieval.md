# Eval: mode=hybrid, k=5

- Date: 2026-11-04 18:59
- Run type: retrieval only (no answers generated, so no LLM calls and no answer score)
- Retrieval mode: hybrid
- LLM: llama3.1:8b | Judge: qwen2.5:7b | Embeddings: embeddinggemma
- Questions: 20 | Runtime: 2s

| Questions | n | Hit@1 | Hit@5 | MRR | All@5 | Answer score |
|---|---|---|---|---|---|---|
| all | 20 | 0.81 | 1.00 | 0.89 | 0.94 | - |
| paraphrase | 6 | 0.50 | 1.00 | 0.71 | 1.00 | - |
| exact-term | 6 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| multi-note | 4 | 1.00 | 1.00 | 1.00 | 0.75 | - |
| not-in-notes | 4 | - | - | - | - | - |
