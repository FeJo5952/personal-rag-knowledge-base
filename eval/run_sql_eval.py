"""Score the router and Text-to-SQL on a test set of questions.

Run from the project root:
    python -m eval.run_sql_eval                  # routing accuracy + SQL execution accuracy
    python -m eval.run_sql_eval --router-only    # just routing accuracy (faster)

Questions come from two files:
    eval/questions_sql.jsonl   {"question", "route": "sql" | "both" | "notes", "gold_sql", optional "ordered": true}
    eval/questions.jsonl       the notes questions from run_eval.py; all of them should be routed to "notes"

Metrics:
    Route acc   - share of questions sent to the expected source (notes / sql / both)
    Valid SQL   - share of sql/both questions where the generated query ran without an error
    Exec acc    - share where the generated query returned the same rows as the hand-written gold query
                  (column names, extra columns and row order are ignored, unless "ordered" is set)
    Attempts    - average number of tries per query (2 = the first query failed and was fixed)
"""

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from app import config
from app.router import ROUTES, route
from app.sql_safety import run_query
from app.text_to_sql import text_to_sql
from eval.run_eval import average, fmt

EVAL_DIR = Path(__file__).parent


def norm(value):
    """Make values comparable: 75 == Decimal('75') == 75.0, 'India ' == 'india'."""
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return round(float(value), 2)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip().lower()
    return value


def column_values(rows, i):
    return sorted(repr(norm(row[i])) for row in rows)


def same_result(gen_columns, gen_rows, gold_columns, gold_rows, ordered=False):
    """True if every gold column appears among the generated columns, with the same rows.

    Column names don't matter (the LLM picks its own aliases) and extra generated columns are fine.
    """
    if len(gen_rows) != len(gold_rows):
        return False
    picked = []  # for each gold column, the generated column holding the same values
    for j in range(len(gold_columns)):
        gold_values = column_values(gold_rows, j)
        match = next((i for i in range(len(gen_columns))
                      if i not in picked and column_values(gen_rows, i) == gold_values), None)
        if match is None:
            return False
        picked.append(match)
    generated = [tuple(norm(row[i]) for i in picked) for row in gen_rows]
    gold = [tuple(norm(value) for value in row) for row in gold_rows]
    return generated == gold if ordered else Counter(generated) == Counter(gold)


def load_questions():
    sql_questions = [json.loads(line) for line in open(EVAL_DIR / "questions_sql.jsonl", encoding="utf-8") if line.strip()]
    notes_file = EVAL_DIR / "questions.jsonl"
    notes_questions = []
    if notes_file.exists():
        notes_questions = [{"question": json.loads(line)["question"], "route": "notes"}
                           for line in open(notes_file, encoding="utf-8") if line.strip()]
    return sql_questions + notes_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-only", action="store_true")
    args = parser.parse_args()

    questions = load_questions()
    rows = []
    started = time.time()

    for i, q in enumerate(questions, start=1):
        row = {"question": q["question"], "expected_route": q["route"]}

        start = time.perf_counter()
        chosen = route(q["question"])
        row.update(route=chosen.route, route_method=chosen.method, route_ok=float(chosen.route == q["route"]),
                   route_s=time.perf_counter() - start)

        # SQL is scored on every question with a gold query, whatever the router chose,
        # so a routing mistake and a SQL mistake are counted separately
        if q.get("gold_sql") and not args.router_only:
            gold = run_query(q["gold_sql"])
            start = time.perf_counter()
            result = text_to_sql(q["question"], part_only=q["route"] == "both")  # as the app does for "both"
            row.update(sql=result.sql, sql_error=result.error, attempts=result.attempts,
                       sql_s=time.perf_counter() - start,
                       valid=0.0 if result.error else 1.0,
                       exec_ok=0.0 if result.error else float(same_result(
                           result.columns, result.rows, gold.columns, gold.rows, q.get("ordered", False))))
        rows.append(row)
        print(f"[{i}/{len(questions)}] route {row['route']:5} (expected {q['route']:5}) "
              f"exec={row.get('exec_ok', '-')}  {q['question'][:60]}")

    # Summary per expected route
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["expected_route"]].append(row)

    def median(values):
        return statistics.median(values) if values else None

    lines = [
        "# Eval: router + Text-to-SQL",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Run type: {'router only' if args.router_only else 'router + Text-to-SQL execution accuracy'}",
        f"- Router LLM: {config.LLM_MODEL} | SQL LLM: {config.SQL_MODEL}",
        f"- Questions: {len(rows)} | Runtime: {time.time() - started:.0f}s",
        "",
        "| Expected route | n | Route acc | Valid SQL | Exec acc | Avg attempts | Median SQL s |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, group in groups.items():
        values = [average([r[m] for r in group if m in r]) for m in ("route_ok", "valid", "exec_ok", "attempts")]
        lines.append(f"| {name} | {len(group)} | " + " | ".join(fmt(v) for v in values)
                     + f" | {fmt(median([r['sql_s'] for r in group if 'sql_s' in r]))} |")

    # Confusion matrix: rows = expected route, columns = route the router chose
    lines += ["", "Routing (rows: expected, columns: chosen):", "",
              "| expected \\ chosen | " + " | ".join(ROUTES) + " |", "|---|" + "---|" * len(ROUTES)]
    for expected in ROUTES:
        counts = Counter(r["route"] for r in rows if r["expected_route"] == expected)
        lines.append(f"| {expected} | " + " | ".join(str(counts[chosen]) for chosen in ROUTES) + " |")

    summary = "\n".join(lines)
    print("\n" + summary)

    # The summary has no note content, so it's safe to commit. Per-question details stay private.
    stamp = f"{datetime.now():%Y-%m-%d_%H%M}-sql{'-router' if args.router_only else ''}"
    (EVAL_DIR / "results").mkdir(exist_ok=True)
    (EVAL_DIR / "runs").mkdir(exist_ok=True)
    (EVAL_DIR / "results" / f"{stamp}.md").write_text(summary + "\n", encoding="utf-8")
    with open(EVAL_DIR / "runs" / f"{stamp}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"\nSaved eval/results/{stamp}.md (summary) and eval/runs/{stamp}.jsonl (details, private)")


if __name__ == "__main__":
    main()
