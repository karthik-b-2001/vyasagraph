# VyasaGraph API (FastAPI + embedding model + vector index), for Cloud Run.
# The index must be built first: python -m src.ingest text
# Secrets (Neo4j, LLM key) come from environment variables, never the image.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false \
    PORT=8080

WORKDIR /app

# CPU-only PyTorch first, so sentence-transformers does not pull the CUDA build.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so cold starts do not download it.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"
ENV HF_HUB_OFFLINE=1

COPY src/ src/
COPY data/index/ data/index/

CMD exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT}
