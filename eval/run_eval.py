"""Score the RAG pipeline on a test set of questions.

Run from the project root:
    python -m eval.run_eval                     # retrieval + answer quality
    python -m eval.run_eval --retrieval-only    # just Hit@k and MRR (fast, no LLM)

Metrics:
    Hit@k  - share of questions where a correct source note is in the top k chunks
    MRR    - mean reciprocal rank: 1/rank of the first correct source (1st = 1.0, 2nd = 0.5, ...)
    Answer - LLM-as-judge grade: correct = 1, partial = 0.5, wrong = 0
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app import config
from app.generate import generate
from app.index import load_index
from app.retrieval import retrieve
from eval.judge import grade

EVAL_DIR = Path(__file__).parent


def normalize(path):
    return path.replace("\\", "/").lower()


def reciprocal_rank(hits, expected_sources):
    """1/rank of the first hit from an expected source, or 0 if none was retrieved."""
    expected = {normalize(s) for s in expected_sources}
    for rank, hit in enumerate(hits, start=1):
        if normalize(hit.source) in expected:
            return 1 / rank
    return 0.0


def average(values):
    return sum(values) / len(values) if values else None


def fmt(value):
    return "-" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dense", help="retrieval mode to evaluate")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--questions", default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    collection = load_index()
    rows = []
    started = time.time()

    for i, q in enumerate(questions, start=1):
        not_in_notes = q["type"] == "not-in-notes"
        hits = retrieve(collection, q["question"], k=args.k, mode=args.mode)
        row = {"question": q["question"], "type": q["type"], "retrieved": [h.source for h in hits]}

        if not not_in_notes:
            rr = reciprocal_rank(hits, q["sources"])
            row["hit"] = 1.0 if rr > 0 else 0.0
            row["rr"] = rr

        if not args.retrieval_only:
            answer = generate(q["question"], hits)
            row["answer"] = answer
            row["grade"] = grade(q["question"], q.get("answer", ""), answer, not_in_notes)

        rows.append(row)
        print(f"[{i}/{len(questions)}] hit={row.get('hit', '-')} grade={row.get('grade', '-')}  {q['question'][:60]}")

    # Summary, overall and per question type
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["type"]].append(row)

    lines = [
        f"# Eval: mode={args.mode}, k={args.k}",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- LLM: {config.LLM_MODEL} | Judge: {config.JUDGE_MODEL} | Embeddings: {config.EMBED_MODEL}",
        f"- Questions: {len(rows)} | Runtime: {time.time() - started:.0f}s",
        "",
        f"| Questions | n | Hit@{args.k} | MRR | Answer score |",
        "|---|---|---|---|---|",
    ]
    for name, group in groups.items():
        hit = average([r["hit"] for r in group if "hit" in r])
        mrr = average([r["rr"] for r in group if "rr" in r])
        score = average([r["grade"] for r in group if "grade" in r])
        lines.append(f"| {name} | {len(group)} | {fmt(hit)} | {fmt(mrr)} | {fmt(score)} |")
    summary = "\n".join(lines)
    print("\n" + summary)

    # The summary has no note content, so it's safe to commit. Per-question details stay private.
    stamp = f"{datetime.now():%Y-%m-%d_%H%M}-{args.mode}{'-retrieval' if args.retrieval_only else ''}"
    (EVAL_DIR / "results").mkdir(exist_ok=True)
    (EVAL_DIR / "runs").mkdir(exist_ok=True)
    (EVAL_DIR / "results" / f"{stamp}.md").write_text(summary + "\n", encoding="utf-8")
    with open(EVAL_DIR / "runs" / f"{stamp}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved eval/results/{stamp}.md (summary) and eval/runs/{stamp}.jsonl (details, private)")


if __name__ == "__main__":
    main()
