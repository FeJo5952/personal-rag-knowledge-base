-- One-time setup: the two database users (roles) and the database.
-- Run as the postgres superuser, choosing your own passwords (they go in .env, not in this file):
--
--   psql -U postgres -v owner_password=CHOOSE_ONE -v reader_password=CHOOSE_ANOTHER -f sql/roles.sql
--
-- rag_owner  owns the tables and fills them from the vault (DATABASE_LOADER_URL)
-- rag_reader runs the SQL the LLM writes (DATABASE_URL). It can only SELECT from three views,
--            every transaction is read-only, and slow queries are cancelled. Even if the LLM
--            writes DELETE or DROP, the database refuses it.

CREATE ROLE rag_owner LOGIN PASSWORD :'owner_password';
CREATE ROLE rag_reader LOGIN PASSWORD :'reader_password' CONNECTION LIMIT 3;

ALTER ROLE rag_reader SET default_transaction_read_only = on;
ALTER ROLE rag_reader SET statement_timeout = '5s';
ALTER ROLE rag_reader SET idle_in_transaction_session_timeout = '10s';

CREATE DATABASE rag_kb OWNER rag_owner ENCODING 'UTF8' TEMPLATE template0;

-- Only our two roles may connect to it
REVOKE ALL ON DATABASE rag_kb FROM PUBLIC;
GRANT CONNECT ON DATABASE rag_kb TO rag_owner, rag_reader;
