"""VyasaGraph FastAPI backend.

Exposes the RAG pipeline, graph queries, and semantic search as API endpoints.
Streams chat responses via Server-Sent Events.

Usage:
    python -m uvicorn src.api:app --reload --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from src import config, rag

log = logging.getLogger("vyasagraph")


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.load()
    print(f"  VyasaGraph API ready: {rag.stats()}")
    yield
    rag.close()


app = FastAPI(
    title="VyasaGraph",
    description="Mahabharata Knowledge Graph & RAG Chatbot API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class SearchRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    top_k: int = Field(default=5, ge=1, le=20)


def _meta(ctx):
    return {
        "type": "meta",
        "on_topic": ctx["on_topic"],
        "entities": ctx["entities"],
        "graph_edges": ctx["facts"],
        "sources": [{**p, "text": p["text"][:300]} for p in ctx["passages"]],
    }


def _sse(data):
    return f"data: {json.dumps(data)}\n\n"


# ── Chat ──────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """RAG chat. SSE events: one 'meta', then 'token's, then 'done' (or 'error')."""
    # Embedding and Neo4j calls block; keep them off the event loop so other
    # streams keep flowing.
    ctx = await run_in_threadpool(rag.retrieve, req.message)

    async def stream():
        if not ctx["on_topic"]:
            yield _sse({"type": "meta", "on_topic": False, "entities": [], "graph_edges": [], "sources": []})
            yield _sse({"type": "token", "content": rag.REFUSAL})
            yield _sse({"type": "done"})
            return
        yield _sse(_meta(ctx))
        try:
            async for token in rag.stream_answer(rag.build_messages(req.message, ctx)):
                yield _sse({"type": "token", "content": token})
        except (httpx.HTTPError, ValueError) as e:
            log.error("LLM request failed: %r", e)
            yield _sse({"type": "error", "message": "The language model is unavailable. Please try again shortly."})
            return
        yield _sse({"type": "done"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming chat, for testing."""
    ctx = await run_in_threadpool(rag.retrieve, req.message)
    meta = _meta(ctx)
    if not ctx["on_topic"]:
        return {"answer": rag.REFUSAL, **meta, "entities": [], "graph_edges": [], "sources": []}
    try:
        answer = await rag.answer(rag.build_messages(req.message, ctx))
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        log.error("LLM request failed: %r", e)
        raise HTTPException(status_code=503, detail="The language model is unavailable.")
    return {"answer": answer, **meta}


# ── Search and graph ──────────────────────────────────────────────────────

@app.post("/api/search")
def search(req: SearchRequest):
    """Semantic search over the text sources."""
    return {"query": req.query, "results": rag.search(req.query, req.top_k)}


def _session():
    return rag._state["driver"].session()


@app.get("/api/graph/character/{name}")
def get_character(name: str):
    """A character and all their relationships."""
    with _session() as s:
        if not s.run("MATCH (n {name: $name}) RETURN n LIMIT 1", name=name).single():
            raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
        result = s.run(
            """
            MATCH (a {name: $name})-[r]-(b)
            RETURN startNode(r).name AS source, type(r) AS type, endNode(r).name AS target,
                   CASE WHEN startNode(r) = a THEN 'outgoing' ELSE 'incoming' END AS direction,
                   r.origins AS origins
            """,
            name=name,
        )
        return {"name": name, "relationships": [dict(r) for r in result]}


@app.get("/api/graph/path/{name1}/{name2}")
def get_path(name1: str, name2: str):
    """Shortest path between two characters (up to 6 hops)."""
    if name1 == name2:  # Neo4j's shortestPath rejects identical endpoints
        with _session() as s:
            found = s.run("MATCH (n {name: $n}) RETURN n LIMIT 1", n=name1).single() is not None
        return {"found": found, "nodes": [name1] if found else [], "relationships": []}
    with _session() as s:
        record = s.run(
            """
            MATCH path = shortestPath((a {name: $n1})-[*..6]-(b {name: $n2}))
            RETURN [n IN nodes(path) | n.name] AS nodes, [r IN relationships(path) | type(r)] AS rels
            """,
            n1=name1, n2=name2,
        ).single()
    if not record:
        return {"found": False, "nodes": [], "relationships": []}
    return {"found": True, "nodes": record["nodes"], "relationships": record["rels"]}


@app.get("/api/graph/subgraph/{name}")
def get_subgraph(name: str, depth: int = Query(default=1, ge=1, le=3)):
    """Nodes and edges within `depth` hops of a character, for visualization."""
    with _session() as s:
        # depth is validated to 1-3 above; Cypher cannot parameterize path length.
        record = s.run(
            f"""
            MATCH path = (a {{name: $name}})-[*1..{depth}]-()
            UNWIND relationships(path) AS r
            WITH collect(DISTINCT r) AS rels
            UNWIND rels AS r
            WITH rels, [startNode(r), endNode(r)] AS ends
            UNWIND ends AS n
            WITH rels, collect(DISTINCT n) AS ns
            RETURN [n IN ns | {{name: n.name, label: labels(n)[0]}}] AS nodes,
                   [r IN rels | {{source: startNode(r).name, target: endNode(r).name, type: type(r)}}] AS edges
            """,
            name=name,
        ).single()
    if not record:
        return {"nodes": [], "edges": []}
    return {"nodes": record["nodes"], "edges": record["edges"]}


@app.get("/api/graph/stats")
def graph_stats():
    """Node, relationship and most-connected-character counts."""
    with _session() as s:
        nodes = {r["label"]: r["cnt"] for r in s.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")}
        rels = {r["type"]: r["cnt"] for r in s.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt ORDER BY cnt DESC")}
        top = [dict(r) for r in s.run(
            "MATCH (c:Character)-[r]-() RETURN c.name AS name, count(r) AS connections "
            "ORDER BY connections DESC LIMIT 10")]
    return {"nodes": nodes, "relationships": rels, "top_characters": top}


@app.get("/api/entities")
def list_entities():
    """All character names in the graph."""
    return {"entities": rag.entity_names()}


@app.get("/api/health")
def health():
    return {"status": "ok", "neo4j": rag.neo4j_ok(), **rag.stats()}
