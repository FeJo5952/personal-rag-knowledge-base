"""Safe execution of SQL written by the LLM.

Two layers:
    1. The database (the real protection): queries run as rag_reader, which can only SELECT from
       three views, in a read-only transaction, with a timeout (see sql/roles.sql and sql/schema.sql).
    2. check_sql() below: catches the model's mistakes early and explains them in plain words,
       so the model can fix its query on the retry.
"""

import re
from dataclasses import dataclass

import psycopg

from app import config

STRING = re.compile(r"'(?:[^']|'')*'")  # a quoted text value like 'March', where any word is allowed
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|copy|call|do|into|set|reset|"
    r"lock|vacuum|analyze|listen|notify|prepare|execute|refresh|pg_sleep\w*|set_config|pg_read_\w+|lo_\w+|dblink\w*)\b",
    re.IGNORECASE,
)


class UnsafeSQL(ValueError):
    """The query was rejected before reaching the database."""


@dataclass
class QueryResult:
    columns: list   # column names
    rows: list      # tuples
    truncated: bool  # True if there were more than max_rows rows


def check_sql(sql):
    """Return the query without a trailing semicolon, or raise UnsafeSQL saying what's wrong."""
    sql = sql.strip().rstrip(";").strip()
    code = STRING.sub("''", sql)  # ignore the words inside quoted values, e.g. ILIKE '%update%'
    if not code:
        raise UnsafeSQL("The query is empty.")
    if any(token in code for token in (";", "--", "/*", "$$")):
        raise UnsafeSQL("Write exactly one statement, without semicolons or comments.")
    if not re.match(r"(select|with)\b", code, re.IGNORECASE):
        raise UnsafeSQL("The query must start with SELECT or WITH.")
    word = FORBIDDEN.search(code)
    if word:
        raise UnsafeSQL(f"'{word.group(0)}' isn't allowed: only read data with SELECT.")
    return sql


def run_query(sql, max_rows=config.SQL_MAX_ROWS):
    """Check the query, then run it as the read-only user and return at most max_rows rows."""
    sql = check_sql(sql)
    options = f"-c statement_timeout={config.SQL_TIMEOUT_MS}"
    with psycopg.connect(config.DATABASE_URL, connect_timeout=3, options=options) as conn:
        conn.read_only = True  # BEGIN READ ONLY, on top of the role's own read-only setting
        # Fetch one row more than needed, to know if the result was cut off
        cursor = conn.execute(f"SELECT * FROM ({sql}) AS q LIMIT {max_rows + 1}")
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]
    return QueryResult(columns, rows[:max_rows], truncated=len(rows) > max_rows)


def db_available():
    """True if DATABASE_URL is set and the database answers."""
    if not config.DATABASE_URL:
        return False
    try:
        with psycopg.connect(config.DATABASE_URL, connect_timeout=2):
            return True
    except psycopg.Error:
        return False
