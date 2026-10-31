"""Generation: put the retrieved chunks into a prompt and let the LLM answer."""

import ollama

from app import config


def build_prompt(question, hits):
    context = "\n\n".join(f"(from {h.source})\n{h.text}" for h in hits)
    return (
        "You answer questions about the user's personal notes. Use only the notes below, "
        "but you may draw reasonable conclusions from them. "
        "Only if nothing in the notes is relevant, say you couldn't find it.\n\n"
        f"Notes:\n{context}\n\nQuestion: {question}"
    )


def generate(question, hits, model=config.LLM_MODEL):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    return response["message"]["content"]


def generate_stream(question, hits, model=config.LLM_MODEL):
    """Same as generate(), but yields the answer piece by piece as the LLM writes it."""
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]
