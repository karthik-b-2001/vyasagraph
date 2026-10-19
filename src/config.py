"""Runtime configuration, read from environment variables (and from a
git-ignored .env file in the project root, if present).

Defaults match the local docker-compose / Ollama setup, so nothing needs
to be set for local development.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """Read KEY=VALUE lines from .env (git-ignored) into the environment.
    Variables already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
SOURCES_DIR = DATA_DIR / "sources"
GRAPH_DIR = DATA_DIR / "graph"
INDEX_DIR = Path(os.getenv("INDEX_DIR", DATA_DIR / "index"))  # built by `python -m src.ingest text`

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "vyasagraph")

# Any OpenAI-compatible chat endpoint. The default is local Ollama; a hosted
# provider only needs these three variables changed.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
# Optional, for reasoning models (e.g. gpt-oss on Groq): low | medium | high.
# Lower effort spends fewer tokens, which matters under a tokens-per-minute cap.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")

# bge-small-en-v1.5 beat all-MiniLM-L6-v2 (same size) on scripts/eval_retrieval.py:
# 80% vs 60% hit@5 within the Ganguli translation. Changing the model
# changes every chunk ID, so re-run `python -m src.ingest text --rebuild`.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# BGE expects this instruction on queries (not on documents).
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")

# Passages given to the LLM per question: the full translation always gets
# its share, so the shorter, more "similar-sounding" retellings cannot crowd
# it out (they took 42% of the top-5 while being 10% of the corpus).
PASSAGES_TRANSLATION = int(os.getenv("PASSAGES_TRANSLATION", "5"))
PASSAGES_OTHER = int(os.getenv("PASSAGES_OTHER", "2"))

# Guardrail: a question with no known character in it must have at least
# one passage this similar, or it is refused without calling the LLM.
# Set low on purpose: it only drops obvious junk and must never block a
# real question; the system prompt handles off-topic questions that pass.
# Measured with scripts/eval_guardrail.py; depends on the embedding model
# (bge-small: lowest on-topic score 0.595 of 40 questions).
GUARD_MIN_SCORE = float(os.getenv("GUARD_MIN_SCORE", "0.55"))

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
