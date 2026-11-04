"""A simple local RAG over an Obsidian vault, with an optional SQL catalog of the vault.

Run:  python rag.py                  # the router picks notes, database or both for each question
      python rag.py --route notes    # always use one source: notes | sql | both
First run: reads the vault, chunks it, embeds the chunks and stores them in ChromaDB.
Then:      asks for questions in the terminal and answers them from your notes (and the database).
To re-index after editing notes, delete the chroma_db folder and run again.
"""

import argparse

from app.generate import format_table
from app.index import load_index
from app.pipeline import ask
from app.retrieval import Retriever


def show(answer):
    route = answer.route
    print(f"\nRoute: {route.route}  ({route.method}: {route.reason})")
    if answer.sql and answer.sql.sql:
        print(f"SQL:   {answer.sql.sql}")
        if not answer.sql.error:
            print(format_table(answer.sql.columns, answer.sql.rows))
    for notice in answer.notices:
        print(f"Note:  {notice}")

    print("\n" + answer.text)

    print("\nSources:")
    if answer.sql and not answer.sql.error:
        print(f"  [Database] {len(answer.sql.rows)} rows")
    for source in dict.fromkeys(h.source for h in answer.hits):  # unique, in order
        print(f"  - {source}")
    print("\n" + " · ".join(f"{step} {seconds:.1f}s" for step, seconds in answer.timings.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default="auto", choices=["auto", "notes", "sql", "both"])
    args = parser.parse_args()

    retriever = Retriever(load_index())
    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        show(ask(question, retriever, route_choice=args.route))  # route, retrieve, augment, generate


if __name__ == "__main__":
    main()
