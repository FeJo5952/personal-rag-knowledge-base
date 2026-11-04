"""All settings, read from the .env file (with sensible defaults)."""

import os

from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = os.environ["VAULT_PATH"]                         # your Obsidian vault folder
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")      # Ollama model that makes vectors
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")             # Ollama model that writes answers
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)             # Ollama model that grades answers in eval
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")             # where the index is saved
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))             # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))        # characters shared by neighbouring chunks
TOP_K = int(os.getenv("TOP_K", "5"))                          # chunks given to the LLM
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")        # dense | bm25 | hybrid | hybrid_rerank
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")  # cross-encoder used by fastembed

# PostgreSQL catalog of the vault (optional: leave the URLs empty to use notes search only)
DATABASE_URL = os.getenv("DATABASE_URL", "")                  # read-only role, runs the LLM's SQL
DATABASE_LOADER_URL = os.getenv("DATABASE_LOADER_URL", "")    # owner role, fills the tables when indexing
SQL_MODEL = os.getenv("SQL_MODEL", LLM_MODEL)                 # Ollama model that writes SQL
SQL_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "5000"))     # a query running longer is cancelled
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "50"))           # rows kept from a result
