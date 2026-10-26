# Eval: mode=dense, k=5

- Date: 2026-10-27 00:48
- LLM: llama3.1:8b | Judge: qwen2.5:7b | Embeddings: embeddinggemma
- Questions: 20 | Runtime: 7s

| Questions | n | Hit@1 | Hit@5 | MRR | All@5 | Answer score |
|---|---|---|---|---|---|---|
| all | 20 | 0.69 | 0.94 | 0.79 | 0.88 | - |
| paraphrase | 6 | 0.67 | 0.83 | 0.75 | 0.83 | - |
| exact-term | 6 | 0.67 | 1.00 | 0.76 | 1.00 | - |
| multi-note | 4 | 0.75 | 1.00 | 0.88 | 0.75 | - |
| not-in-notes | 4 | - | - | - | - | - |
