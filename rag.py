"""A simple local RAG over an Obsidian vault.

Run:  python rag.py
First run: reads the vault, chunks it, embeds the chunks and stores them in ChromaDB.
Then:      asks for questions in the terminal and answers them from your notes.
To re-index after editing notes, delete the chroma_db folder and run again.
"""

from app.generate import generate
from app.index import load_index
from app.retrieval import Retriever


def main():
    retriever = Retriever(load_index())
    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        hits = retriever.search(question)          # 1. Retrieve
        print("\n" + generate(question, hits))     # 2. Augment + 3. Generate
        print("\nSources:")
        for source in dict.fromkeys(h.source for h in hits):  # unique, in order
            print(f"  - {source}")


if __name__ == "__main__":
    main()
