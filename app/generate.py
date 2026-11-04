"""Generation: put the retrieved chunks (and database results) into a prompt and let the LLM answer."""

from datetime import date
from decimal import Decimal

import ollama

from app import config

MAX_TABLE_ROWS = 15  # rows of a database result shown to the LLM


def format_cell(value):
    if value is None:
        return ""
    if isinstance(value, Decimal):  # SUM() and COUNT() results
        return str(int(value)) if value == value.to_integral_value() else str(round(value, 2))
    if isinstance(value, date):
        return value.isoformat()
    return str(value).replace("|", "/").replace("\n", " ")[:120]


def format_table(columns, rows, max_rows=MAX_TABLE_ROWS):
    """A database result as a small Markdown table."""
    if not rows:
        return "(no rows)"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(format_cell(value) for value in row) + " |" for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"... and {len(rows) - max_rows} more rows (not shown)")
    return "\n".join(lines)


def build_combined_prompt(question, sql_result, hits):
    """Prompt with a database result and (for "both" questions) note chunks, each labelled with its source."""
    rows = f"{len(sql_result.rows)} rows" + (", more exist" if sql_result.truncated else "")
    database = f"SQL: {sql_result.sql}\nResult ({rows}):\n{format_table(sql_result.columns, sql_result.rows)}"
    notes = "\n\n".join(f"(from {h.source})\n{h.text}" for h in hits) or "(none)"
    return (
        "You answer questions about the user's personal notes. You get two kinds of context:\n"
        "- DATABASE: exact facts computed from the whole vault (counts, lists, tasks, deadlines). "
        "Copy its numbers and names exactly; never recount the rows yourself.\n"
        "- NOTES: excerpts from the notes, for explanations and details.\n"
        "Use only this context, but you may draw reasonable conclusions. "
        "Say where each fact comes from: [Database] or (from <note>). "
        "If the database returned no rows, say that nothing matched. "
        "If nothing relevant is in the context, say you couldn't find it.\n\n"
        f"DATABASE:\n{database}\n\nNOTES:\n{notes}\n\nQuestion: {question}"
    )


def chat(prompt, model=config.LLM_MODEL):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def chat_stream(prompt, model=config.LLM_MODEL):
    """Yields the answer piece by piece as the LLM writes it."""
    for part in ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=True):
        yield part["message"]["content"]


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
