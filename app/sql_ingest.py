"""SQL catalog: read the vault's structure (courses, notes, tasks, frontmatter) into PostgreSQL.

Run from the project root:
    python -m app.sql_ingest --dry-run    # parse the vault and print what would be loaded (no database needed)
    python -m app.sql_ingest              # load it (creates the tables the first time)
    python -m app.sql_ingest --init       # drop and re-create the tables and views first (after editing sql/schema.sql)

It also runs at the end of every (re-)index, so the catalog and ChromaDB describe the same notes.
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
import yaml

from app import config
from app.index import note_files, note_source

SCHEMA_FILE = Path(__file__).parent.parent / "sql" / "schema.sql"

# Course folders inside COURSES_ROOT -> (code, name). The folder name's typo is in the vault itself.
COURSES_ROOT = "IIT Madras Diploma"
COURSES = {
    "Business Data Mangement": ("BDM", "Business Data Management"),
    "MLF - Machine Learning Foundations": ("MLF", "Machine Learning Foundations"),
    "MLP - Machine Learning Practice": ("MLP", "Machine Learning Practice"),
    "MLP Project": ("MLPPROJ", "Machine Learning Practice Project"),
    "MLT - Machine Learning Techniques": ("MLT", "Machine Learning Techniques"),
    "Maths 2": ("MATHS2", "Maths 2"),
    "Statistics 2": ("STATS2", "Statistics 2"),
}
NOT_COURSES = {"Workshops", "Certificates"}  # folders in COURSES_ROOT that aren't courses

THIRD_PARTY = {"README", "CONTRIBUTORS", "CODE_OF_CONDUCT", "LICENSE"}  # files copied in with downloaded repos

TASK = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\[(.)\]\s+(.+?)\s*$")    # "- [ ] text" or "1. [x] text"
WEEK = re.compile(r"\bweek\s*[-_]?\s*(\d{1,2})", re.IGNORECASE)       # "Week 3", "Week - 3", "week3"
FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


@dataclass
class Note:
    path: str
    title: str
    area: str
    kind: str
    course: str | None      # course folder name, a key of COURSES
    week: int | None
    modified_at: datetime
    word_count: int
    tasks: list = field(default_factory=list)       # (line_no, text, is_done)
    properties: list = field(default_factory=list)  # (key, value, value_date)


# --- Parsing (no database) ---

def course_folder(folders, problems, path):
    """The course a note belongs to, or None. folders = the note's folder names, top first."""
    if len(folders) < 2 or folders[0] != COURSES_ROOT or folders[1] in NOT_COURSES:
        return None
    if folders[1] not in COURSES:
        problems.append(f"{path}: unknown course folder '{folders[1]}' (add it to COURSES in app/sql_ingest.py)")
        return None
    return folders[1]


def note_kind(folders, title, course):
    """What sort of note this is, decided by where it lives in the vault."""
    top = folders[0] if folders else ""
    if folders[:2] == ["Plugins", "Templates"]:
        return "template"
    if top in ("Plugins", "copilot"):
        return "plugin"
    if "01 - Universities" in folders:
        return "university"
    if "02 - Entrance Exams" in folders:
        return "exam"
    if folders[:2] == ["Personal", "Projects"] or course == "MLP Project":
        return "project"
    if "Workshops" in folders:
        return "workshop"
    if top == "Parentof":
        return "work"
    if title.upper() in THIRD_PARTY:
        return "other"
    if course:
        return "course_note"
    if top == "Personal":
        return "personal"
    return "other"


def week_number(folders, title, problems, path):
    """The last "Week N" in the note's folders or file name, e.g. ".../Week 3/notes.md" -> 3."""
    matches = WEEK.findall(" / ".join([*folders, title]))
    if not matches:
        return None
    week = int(matches[-1])
    if week > 20:
        problems.append(f"{path}: week {week} is out of range, stored as no week")
        return None
    return week


def parse_frontmatter(text, problems, path):
    """The "key: value" block at the top of a note, as a dict ({} if there is none or it's broken)."""
    match = FRONTMATTER.match(text)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        problems.append(f"{path}: frontmatter isn't valid YAML, skipped")
        return {}
    return data if isinstance(data, dict) else {}


def property_rows(frontmatter):
    """Frontmatter -> (key, value, value_date) rows. Lists are joined; nested dicts and empty values are skipped."""
    rows = []
    for key, value in frontmatter.items():
        if value is None or isinstance(value, dict):
            continue
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value if item is not None)
        value_date = None
        if isinstance(value, date):  # YAML turns 2026-10-19 into a date (a datetime is a date too)
            value_date = value.date() if isinstance(value, datetime) else value
            value = value.isoformat()
        value = str(value).strip()
        if value:
            rows.append((str(key), value, value_date))
    return rows


def find_tasks(text):
    """Checkboxes outside code blocks, as (line_no, text, is_done)."""
    tasks, in_code = [], False
    for line_no, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
            continue
        match = None if in_code else TASK.match(line)
        if match:
            tasks.append((line_no, match.group(2), match.group(1) in "xX"))
    return tasks


def parse_note(file, problems):
    path = note_source(file)
    folders = path.split("/")[:-1]
    title = file.stem
    text = file.read_text(encoding="utf-8")
    course = course_folder(folders, problems, path)
    return Note(
        path=path,
        title=title,
        area=folders[0] if folders else "",
        kind=note_kind(folders, title, course),
        course=course,
        week=week_number(folders, title, problems, path) if course else None,
        modified_at=datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc),
        word_count=len(text.split()),
        tasks=find_tasks(text),
        properties=property_rows(parse_frontmatter(text, problems, path)),
    )


def parse_vault():
    """Every note in the vault, plus a list of problems found while reading them."""
    problems = []
    notes = [parse_note(file, problems) for file in note_files()]
    return notes, problems


def print_summary(notes, problems):
    tasks = [task for note in notes for task in note.tasks]
    done = sum(1 for task in tasks if task[2])
    print(f"Notes: {len(notes)} | Tasks: {len(tasks)} ({len(tasks) - done} open, {done} done) | "
          f"Frontmatter properties: {sum(len(note.properties) for note in notes)}")
    print("By kind:   " + ", ".join(f"{k} {n}" for k, n in Counter(n.kind for n in notes).most_common()))
    print("By course: " + ", ".join(f"{COURSES[c][0]} {n}" for c, n in Counter(n.course for n in notes if n.course).most_common()))
    for problem in problems:
        print(f"  ! {problem}")


# --- Loading into PostgreSQL ---

def tables_exist(conn):
    return conn.execute("SELECT to_regclass('notes') IS NOT NULL").fetchone()[0]


def sync_vault(init=False):
    """Replace the catalog with the vault's current notes, in one transaction (all or nothing)."""
    notes, problems = parse_vault()
    if not notes:
        raise RuntimeError(f"No notes found in {config.VAULT_PATH}: check VAULT_PATH, the catalog was left unchanged")

    # `with` commits if everything succeeds and rolls back on any error, so a failed load changes nothing
    with psycopg.connect(config.DATABASE_LOADER_URL) as conn:
        if init or not tables_exist(conn):
            conn.execute(SCHEMA_FILE.read_text(encoding="utf-8"))

        # Deleting the notes also deletes their tasks and properties (ON DELETE CASCADE)
        conn.execute("DELETE FROM notes")

        # Upsert: insert each course, or update it if its code is already there
        course_ids = {}
        for folder, (code, name) in COURSES.items():
            course_ids[folder] = conn.execute(
                """INSERT INTO courses (code, name, folder) VALUES (%s, %s, %s)
                   ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name, folder = EXCLUDED.folder
                   RETURNING id""",
                (code, name, folder),
            ).fetchone()[0]

        with conn.cursor() as cur:
            for note in notes:
                note_id = cur.execute(
                    """INSERT INTO notes (path, title, area, kind, course_id, week, modified_at, word_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (note.path, note.title, note.area, note.kind, course_ids.get(note.course),
                     note.week, note.modified_at, note.word_count),
                ).fetchone()[0]
                cur.executemany(
                    "INSERT INTO tasks (note_id, line_no, text, is_done) VALUES (%s, %s, %s, %s)",
                    [(note_id, *task) for task in note.tasks],
                )
                cur.executemany(
                    "INSERT INTO note_properties (note_id, key, value, value_date) VALUES (%s, %s, %s, %s)",
                    [(note_id, *prop) for prop in note.properties],
                )

        # Integrity check: the database must hold exactly what was parsed, or the whole load is undone
        loaded = conn.execute("SELECT (SELECT count(*) FROM notes), (SELECT count(*) FROM tasks)").fetchone()
        expected = (len(notes), sum(len(note.tasks) for note in notes))
        if loaded != expected:
            raise RuntimeError(f"Loaded (notes, tasks) {loaded} but parsed {expected}; rolled back")

    print("SQL catalog updated.")
    print_summary(notes, problems)


def main():
    parser = argparse.ArgumentParser(description="Load the vault's structure into PostgreSQL.")
    parser.add_argument("--dry-run", action="store_true", help="parse the vault and print counts, no database needed")
    parser.add_argument("--init", action="store_true", help="drop and re-create the tables and views first")
    args = parser.parse_args()

    if args.dry_run:
        print_summary(*parse_vault())
    elif not config.DATABASE_LOADER_URL:
        raise SystemExit("Set DATABASE_LOADER_URL in .env first (see .env.example).")
    else:
        sync_vault(init=args.init)


if __name__ == "__main__":
    main()
