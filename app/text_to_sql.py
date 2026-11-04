"""Text-to-SQL: the LLM turns a question into a PostgreSQL query over the vault catalog, and we run it safely.

Try it from the project root:
    python -m app.text_to_sql "Which course has the most open tasks?"
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import date
from functools import cache

import ollama
import psycopg

from app import config
from app.sql_safety import UnsafeSQL, run_query

SCHEMA = """You can query these PostgreSQL views (and nothing else):

v_notes(note_id, title, path, area, kind, course_code, course_name, week, modified_on date, word_count, open_tasks, done_tasks)
  One row per note. open_tasks and done_tasks are already counted per note: to count tasks per course use SUM(open_tasks).
  path is the file's path in the vault, e.g. 'IIT Madras Diploma/MLP - Machine Learning Practice/My Notes/week2 notes.md'.
  area is the top-level folder. kind is one of: {kinds}.
v_tasks(task_id, task, is_done boolean, note_title, note_path, kind, course_code, course_name)
  One row per checkbox ("- [ ] ...") in a note.
v_universities(note_title, note_path, university, programme, country, eligibility, entrance_exam, application_deadline, exam_timeline, status, official_url, open_tasks)
  One row per master's programme I'm considering. application_deadline and exam_timeline are free text like 'March - April'.

Courses (course_code = course_name): {courses}. Notes outside a course have course_code NULL."""

RULES = """You write one PostgreSQL SELECT query that answers the user's question about their Obsidian vault.

{schema}

Rules:
- Use only the views above. Write one statement, no semicolons, no comments.
- Match names with ILIKE '%...%' (people type "mlp" or "birmingham"), or use course_code.
- Notes of kind 'plugin', 'template' and 'other' weren't written by the user: leave them out unless asked.
- Give columns readable aliases. For "most", "top", "latest" use ORDER BY ... LIMIT.
- Today is {today}.
- If the question is about what the notes say (explanations, concepts, details in the text), the database can't answer it: set "answerable" to false and "sql" to ""."""

# Example questions and the query we want, shown to the model before the real question
EXAMPLES = [
    ("Which course has the most open tasks?",
     "SELECT course_name, SUM(open_tasks) AS open_tasks FROM v_notes WHERE course_code IS NOT NULL "
     "GROUP BY course_name ORDER BY open_tasks DESC LIMIT 1"),
    ("Which universities on my list are in the UK?",
     "SELECT university, programme, country FROM v_universities WHERE country ILIKE '%united kingdom%'"),
    ("Which programmes have their application deadline in March?",
     "SELECT university, programme, application_deadline FROM v_universities WHERE application_deadline ILIKE '%march%'"),
    ("List my open tasks for the MLP project",
     "SELECT task, note_title FROM v_tasks WHERE NOT is_done AND course_code = 'MLPPROJ'"),
    ("What is the longest note in each course?",
     "WITH ranked AS (SELECT course_name, title, word_count, "
     "RANK() OVER (PARTITION BY course_name ORDER BY word_count DESC) AS rnk "
     "FROM v_notes WHERE course_code IS NOT NULL) "
     "SELECT course_name, title, word_count FROM ranked WHERE rnk = 1"),
    ("How many notes did I edit each month?",
     "SELECT date_trunc('month', modified_on)::date AS month, COUNT(*) AS notes FROM v_notes "
     "WHERE kind NOT IN ('plugin', 'template', 'other') GROUP BY 1 ORDER BY 1"),
    ("Explain PCA from my notes", ""),
]

# Ollama forces the reply into this JSON shape
SQL_FORMAT = {
    "type": "object",
    "properties": {"answerable": {"type": "boolean"}, "sql": {"type": "string"}},
    "required": ["answerable", "sql"],
}


@dataclass
class SqlResult:
    sql: str | None                              # the last query the model wrote
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    truncated: bool = False
    error: str | None = None                     # set when no working query was found
    attempts: int = 0


@cache
def live_values():
    """Kinds and courses that are actually in the database, so the prompt matches the vault."""
    kinds = run_query("SELECT DISTINCT kind FROM v_notes ORDER BY kind").rows
    courses = run_query("SELECT DISTINCT course_code, course_name FROM v_notes "
                        "WHERE course_code IS NOT NULL ORDER BY course_code").rows
    return ", ".join(k for (k,) in kinds), ", ".join(f"{code} = {name}" for code, name in courses)


def schema_description():
    kinds, courses = live_values()
    return SCHEMA.format(kinds=kinds, courses=courses)


def build_messages(question):
    messages = [{"role": "system", "content": RULES.format(schema=schema_description(), today=date.today())}]
    for example_question, example_sql in EXAMPLES:
        messages.append({"role": "user", "content": example_question})
        messages.append({"role": "assistant", "content": json.dumps({"answerable": bool(example_sql), "sql": example_sql})})
    messages.append({"role": "user", "content": question})
    return messages


def error_text(error):
    """A short error for the model to fix its query, with Postgres's hint when it has one."""
    if isinstance(error, psycopg.Error) and error.diag.message_primary:
        hint = error.diag.message_hint
        return error.diag.message_primary + (f" (hint: {hint})" if hint else "")
    return str(error)


PART_HINT = ("\n\n(Write the query only for the part the database can answer, e.g. which course, university, "
             "notes or tasks the question is about. The notes' text is searched separately, so this counts as answerable.)")


def text_to_sql(question, model=config.SQL_MODEL, max_attempts=2, part_only=False):
    """Ask the LLM for a query and run it. If it fails, show the model the error and let it try once more.

    part_only: for "both" questions, only the database part is needed (the notes are searched separately).
    """
    messages = build_messages(question + (PART_HINT if part_only else ""))
    sql, error = None, None
    for attempt in range(1, max_attempts + 1):
        reply = ollama.chat(model=model, messages=messages, format=SQL_FORMAT, options={"temperature": 0})
        content = reply["message"]["content"]
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            error = "the model didn't reply with valid JSON"
            continue
        if not data.get("answerable") or not data.get("sql", "").strip():
            return SqlResult(sql=None, error="the database doesn't hold this information", attempts=attempt)

        sql = data["sql"].strip()
        try:
            result = run_query(sql)
            return SqlResult(sql, result.columns, result.rows, result.truncated, attempts=attempt)
        except (UnsafeSQL, psycopg.Error) as exc:
            error = error_text(exc)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"That query failed: {error}\nWrite a corrected query."})
    return SqlResult(sql, error=error, attempts=max_attempts)


if __name__ == "__main__":
    from app.generate import format_table

    result = text_to_sql(" ".join(sys.argv[1:]) or "Which course has the most open tasks?")
    print(f"SQL ({result.attempts} attempt{'s' if result.attempts > 1 else ''}): {result.sql}")
    print(f"Error: {result.error}" if result.error else format_table(result.columns, result.rows))
