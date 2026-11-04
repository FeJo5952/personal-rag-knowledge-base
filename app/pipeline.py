"""The full question flow: route -> (SQL and/or note search) -> prompt -> answer.

prepare() does everything up to the prompt, so the terminal can print the answer at once
and the UI can stream it.
"""

import time
from dataclasses import dataclass, field

from app import config
from app.generate import build_combined_prompt, build_prompt, chat
from app.router import Route, route
from app.sql_safety import db_available
from app.text_to_sql import SqlResult, text_to_sql


@dataclass
class Answer:
    question: str
    route: Route                                  # where the answer comes from (after any fallback)
    hits: list = field(default_factory=list)      # note chunks (Hit)
    sql: SqlResult | None = None                  # database query and result
    notices: list = field(default_factory=list)   # things the user should know, e.g. a fallback
    prompt: str = ""
    timings: dict = field(default_factory=dict)   # seconds per step
    text: str = ""                                # the LLM's answer


def choose_route(question, route_choice, database_up):
    if not database_up:
        reason = "the database isn't available" if config.DATABASE_URL else "no database is set up"
        return Route("notes", reason, "fallback")
    if route_choice == "auto":
        return route(question)
    return Route(route_choice, "chosen by you", "manual")


def prepare(question, retriever, route_choice="auto", k=config.TOP_K, mode=config.RETRIEVAL_MODE, on_step=None):
    """Route the question, query the database and/or search the notes, and build the prompt.

    route_choice: auto | notes | sql | both. on_step(text) reports progress to a UI.
    """
    step = on_step or (lambda text: None)
    answer = Answer(question, route=None)

    start = time.perf_counter()
    database_up = db_available()
    if route_choice == "auto" and database_up:
        step("Choosing where to look...")
    answer.route = choose_route(question, route_choice, database_up)
    if route_choice in ("sql", "both") and not database_up:
        answer.notices.append("The database isn't available, so this was answered from your notes only.")
    answer.timings["route"] = time.perf_counter() - start

    if answer.route.route in ("sql", "both"):
        step("Writing and running a database query...")
        start = time.perf_counter()
        answer.sql = text_to_sql(question)
        answer.timings["sql"] = time.perf_counter() - start
        if answer.sql.error:  # no working query: fall back to the notes
            answer.notices.append(f"Database query failed ({answer.sql.error}), so this was answered from your notes.")
            answer.route = Route("notes", "database query failed", "fallback")

    if answer.route.route in ("notes", "both"):
        step("Searching your notes...")
        start = time.perf_counter()
        answer.hits = retriever.search(question, k=k, mode=mode)
        answer.timings["search"] = time.perf_counter() - start

    if answer.route.route == "notes":
        answer.prompt = build_prompt(question, answer.hits)  # the same prompt as before the database existed
    else:
        answer.prompt = build_combined_prompt(question, answer.sql, answer.hits)
    return answer


def ask(question, retriever, model=config.LLM_MODEL, **options):
    """prepare() + let the LLM write the answer (for the terminal and the eval)."""
    answer = prepare(question, retriever, **options)
    start = time.perf_counter()
    answer.text = chat(answer.prompt, model)
    answer.timings["answer"] = time.perf_counter() - start
    return answer
