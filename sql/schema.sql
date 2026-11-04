-- Tables and views of the vault catalog, modelled on my Obsidian vault.
-- Created automatically by `python -m app.sql_ingest` (or re-created with --init).
-- Every row is rebuilt from the vault, so dropping and re-creating loses nothing.

DROP VIEW IF EXISTS v_universities, v_tasks, v_notes;
DROP TABLE IF EXISTS note_properties, tasks, notes, courses;


-- One row per course folder in "IIT Madras Diploma"
CREATE TABLE courses (
    id      integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code    text NOT NULL UNIQUE CHECK (code ~ '^[A-Z0-9]{2,10}$'),   -- e.g. MLP
    name    text NOT NULL,                                            -- e.g. Machine Learning Practice
    folder  text NOT NULL UNIQUE                                      -- folder name in the vault
);

-- One row per Markdown note (the same notes that are chunked into ChromaDB)
CREATE TABLE notes (
    id           integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    path         text NOT NULL UNIQUE                                 -- = "source" in ChromaDB
                 CHECK (path LIKE '%.md' AND strpos(path, '\') = 0),  -- forward slashes only
    title        text NOT NULL,                                       -- file name without .md
    area         text NOT NULL,                                       -- top-level folder ('' = vault root)
    kind         text NOT NULL CHECK (kind IN ('course_note', 'workshop', 'project', 'university', 'exam',
                                               'work', 'personal', 'template', 'plugin', 'other')),
    course_id    integer REFERENCES courses (id) ON DELETE RESTRICT,  -- NULL = not a course note
    week         smallint CHECK (week BETWEEN 0 AND 20),              -- from "Week N" in the path
    modified_at  timestamptz NOT NULL,                                -- file's last modified time
    word_count   integer NOT NULL CHECK (word_count >= 0),
    CHECK (week IS NULL OR course_id IS NOT NULL)                     -- weeks only exist inside a course
);
CREATE INDEX notes_course_id_idx ON notes (course_id);  -- Postgres doesn't index foreign keys by itself

-- One row per checkbox ("- [ ] ..." or "- [x] ...") in a note
CREATE TABLE tasks (
    id       integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    note_id  integer NOT NULL REFERENCES notes (id) ON DELETE CASCADE,  -- delete a note -> its tasks go too
    line_no  integer NOT NULL CHECK (line_no > 0),
    text     text NOT NULL CHECK (text <> ''),
    is_done  boolean NOT NULL,
    UNIQUE (note_id, line_no)                                          -- one checkbox per line
);

-- One row per frontmatter property (the "key: value" lines at the top of a note)
CREATE TABLE note_properties (
    note_id     integer NOT NULL REFERENCES notes (id) ON DELETE CASCADE,
    key         text NOT NULL,
    value       text NOT NULL CHECK (value <> ''),
    value_date  date,                                                  -- set when the value is a real date
    PRIMARY KEY (note_id, key)                                         -- each key once per note
);


-- Views: the only things the LLM's SQL can read. They join the tables so its queries stay simple.

-- Every note with its course and task counts
CREATE VIEW v_notes AS
SELECT n.id AS note_id, n.title, n.path, n.area, n.kind,
       c.code AS course_code, c.name AS course_name, n.week,
       n.modified_at::date AS modified_on, n.word_count,
       count(t.id) FILTER (WHERE NOT t.is_done) AS open_tasks,
       count(t.id) FILTER (WHERE t.is_done)     AS done_tasks
FROM notes n
LEFT JOIN courses c ON c.id = n.course_id
LEFT JOIN tasks t   ON t.note_id = n.id
GROUP BY n.id, c.id;

-- Every task with the note and course it belongs to
CREATE VIEW v_tasks AS
SELECT t.id AS task_id, t.text AS task, t.is_done,
       n.title AS note_title, n.path AS note_path, n.kind,
       c.code AS course_code, c.name AS course_name
FROM tasks t
JOIN notes n        ON n.id = t.note_id
LEFT JOIN courses c ON c.id = n.course_id;

-- One row per university note: frontmatter rows pivoted into columns.
-- Merges the two frontmatter versions in the vault (application_deadline vs deadline)
-- and fixes the misspelled exam_timline key.
CREATE VIEW v_universities AS
SELECT n.title AS note_title, n.path AS note_path,
       coalesce(max(p.value) FILTER (WHERE p.key = 'university'), n.title) AS university,
       max(p.value) FILTER (WHERE p.key = 'course')        AS programme,
       max(p.value) FILTER (WHERE p.key = 'country')       AS country,
       max(p.value) FILTER (WHERE p.key = 'eligibility')   AS eligibility,
       max(p.value) FILTER (WHERE p.key = 'entrance_exam') AS entrance_exam,
       coalesce(max(p.value) FILTER (WHERE p.key = 'application_deadline'),
                max(p.value) FILTER (WHERE p.key = 'deadline'))     AS application_deadline,  -- free text
       max(p.value) FILTER (WHERE p.key = 'exam_timline')  AS exam_timeline,
       max(p.value) FILTER (WHERE p.key = 'status')        AS status,
       max(p.value) FILTER (WHERE p.key = 'official_url')  AS official_url,
       (SELECT count(*) FROM tasks t WHERE t.note_id = n.id AND NOT t.is_done) AS open_tasks
FROM notes n
LEFT JOIN note_properties p ON p.note_id = n.id
WHERE n.kind = 'university'
GROUP BY n.id;


-- The read-only role sees the views, not the tables (a view runs with its owner's rights)
GRANT SELECT ON v_notes, v_tasks, v_universities TO rag_reader;
