"""Query routing: decide where the answer to a question comes from.

    notes   the text inside the notes (the hybrid search pipeline)
    sql     the PostgreSQL catalog of the vault (counting, listing, filtering)
    both    the database picks something first, then the notes explain it

The LLM decides; if it fails, simple keyword rules decide instead.
"""

import json
import re
from dataclasses import dataclass

import ollama

from app import config

ROUTES = ["notes", "sql", "both"]

PROMPT = """You decide where the answer to a question about my Obsidian notes comes from.

"sql": a database of the vault's structure: my IIT Madras courses (MLP, MLF, MLT, BDM, Maths 2, Statistics 2, MLP Project),
  how many notes each course has and which weeks they cover, task checkboxes (open or done) in every note, when notes were
  edited, and the master's programmes I'm considering (university, country, entrance exam, application deadline, status).
  Use it to count, list, filter, rank or compare, and for tasks, deadlines and dates.
"notes": the text written inside the notes: explanations, concepts, how and why, summaries, and any detail
  written in a note's body (fees, syllabus, my reasons and opinions).
"both": the database must first find something (a course, a university, tasks), AND the notes' text is needed
  to answer about it.
If unsure, answer "notes".

Examples:
How many open tasks do I have in MLT? -> sql
Which universities on my list are in India? -> sql
Which notes did I edit last week? -> sql
Which course has the most weekly notes? -> sql
Explain the bias-variance trade-off from my notes. -> notes
How did I fix the icalendar pickling bug? -> notes
What is the fee at Chennai Mathematical Institute? -> notes
Summarize my notes for the course with the most open tasks. -> both
For the programmes with deadlines in March, what are the eligibility requirements? -> both

Reply with JSON: {"route": "notes" | "sql" | "both", "reason": "<at most 10 words>"}"""

# Ollama forces the reply into this JSON shape, so the route is always one of the three
ROUTE_FORMAT = {
    "type": "object",
    "properties": {"route": {"enum": ROUTES}, "reason": {"type": "string"}},
    "required": ["route", "reason"],
}

SQL_WORDS = re.compile(
    r"\b(how many|number of|count|list (all|my|the)|which (courses?|universit\w*|programmes?|notes|tasks)|"
    r"most|least|latest|open tasks|deadlines?|per (course|week))\b", re.IGNORECASE)
NOTES_WORDS = re.compile(r"\b(explain|why|what is|what are|how (do|does|did|to)|summari[sz]e|describe|difference)\b",
                         re.IGNORECASE)


@dataclass
class Route:
    route: str   # notes | sql | both
    reason: str  # shown to the user
    method: str  # llm | keywords | manual


def keyword_route(question):
    """Fallback: SQL-type words -> sql, explanation words -> notes, both kinds -> both."""
    sql, notes = bool(SQL_WORDS.search(question)), bool(NOTES_WORDS.search(question))
    return "both" if sql and notes else "sql" if sql else "notes"


def route(question, model=config.LLM_MODEL):
    try:
        reply = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": question}],
            format=ROUTE_FORMAT,
            options={"temperature": 0, "num_predict": 60},  # same question -> same route; short reply
        )
        data = json.loads(reply["message"]["content"])
        if data["route"] in ROUTES:
            return Route(data["route"], data.get("reason", ""), "llm")
    except (ollama.ResponseError, ConnectionError, ValueError, KeyError):
        pass
    return Route(keyword_route(question), "keyword rules (the LLM router failed)", "keywords")
