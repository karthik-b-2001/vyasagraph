"""VyasaGraph FastAPI backend.

Exposes the RAG pipeline, graph queries, and semantic search as API endpoints.
Streams chat responses via Server-Sent Events.

Usage:
    python -m uvicorn src.api:app --reload --port 8000
"""

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chromadb"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "vyasagraph"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mahabharata_chunks"

# ── Alias map ─────────────────────────────────────────────────────────────

ALIAS_MAP = {
    "partha": "Arjuna", "dhananjaya": "Arjuna", "gudakesha": "Arjuna",
    "savyasachi": "Arjuna", "phalguna": "Arjuna",
    "vasudeva": "Krishna", "govinda": "Krishna", "keshava": "Krishna",
    "madhava": "Krishna", "janardana": "Krishna", "hari": "Krishna",
    "dharmaraja": "Yudhishthira", "ajatashatru": "Yudhishthira",
    "yudhisthira": "Yudhishthira",
    "bhimasena": "Bhima", "vrikodara": "Bhima",
    "panchali": "Draupadi", "krishnaa": "Draupadi", "yajnaseni": "Draupadi",
    "suyodhana": "Duryodhana",
    "radheya": "Karna", "vasusena": "Karna", "sutaputra": "Karna",
    "devavrata": "Bhishma", "gangeya": "Bhishma", "pitamaha": "Bhishma",
    "dronacharya": "Drona",
    "saubala": "Shakuni",
    "saindhava": "Jayadratha",
    "pritha": "Kunti",
    "salya": "Shalya",
    "yuyudhana": "Satyaki",
    "drauni": "Ashwatthama",
    "shikhandi": "Shikhandhi",
}

# ── Globals (set in lifespan) ─────────────────────────────────────────────

embedder = None
collection = None
neo4j_driver = None
known_names = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, collection, neo4j_driver, known_names

    print("  Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("  Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"  Collection: {collection.count()} chunks")

    print("  Connecting to Neo4j...")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()

    with neo4j_driver.session() as s:
        result = s.run("MATCH (n:Character) RETURN n.name as name")
        known_names = [r["name"] for r in result]
    print(f"  Known entities: {len(known_names)}")

    print("  VyasaGraph API ready")
    yield

    neo4j_driver.close()

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VyasaGraph",
    description="Mahabharata Knowledge Graph & RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

class GraphNode(BaseModel):
    name: str
    label: str = "Character"

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    origins: str = ""

# ── Entity detection ──────────────────────────────────────────────────────

def detect_entities(query):
    found = set()
    query_lower = query.lower()
    for alias, canonical in ALIAS_MAP.items():
        if alias in query_lower:
            found.add(canonical)
    for name in known_names:
        if name.lower() in query_lower:
            found.add(name)
    return list(found)

# ── Graph retrieval ───────────────────────────────────────────────────────

def get_graph_context(entities, max_rels=20):
    if not entities:
        return "", []
    lines = []
    edges = []
    with neo4j_driver.session() as s:
        for entity in entities:
            result = s.run(
                "MATCH (a {name: $name})-[r]->(b) RETURN a.name as src, type(r) as rel, b.name as tgt, r.origins as origins LIMIT $limit",
                {"name": entity, "limit": max_rels}
            )
            for record in result:
                lines.append(f"  {record['src']} --[{record['rel']}]--> {record['tgt']}")
                edges.append({"source": record["src"], "target": record["tgt"], "type": record["rel"], "origins": record.get("origins", "")})

            result = s.run(
                "MATCH (b)-[r]->(a {name: $name}) RETURN b.name as src, type(r) as rel, a.name as tgt, r.origins as origins LIMIT $limit",
                {"name": entity, "limit": max_rels}
            )
            for record in result:
                lines.append(f"  {record['src']} --[{record['rel']}]--> {record['tgt']}")
                edges.append({"source": record["src"], "target": record["tgt"], "type": record["rel"], "origins": record.get("origins", "")})

    unique_lines = list(dict.fromkeys(lines))
    unique_edges = {(e["source"], e["target"], e["type"]): e for e in edges}
    context = ""
    if unique_lines:
        context = "KNOWN FACTS FROM KNOWLEDGE GRAPH:\n" + "\n".join(unique_lines[:max_rels])
    return context, list(unique_edges.values())

# ── Vector retrieval ──────────────────────────────────────────────────────

def get_vector_context(query, top_k=5):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    passages = []
    sources = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1 - dist
        if score < 0.2:
            continue
        source = f"{meta['parva']}, Chapter {meta['chapter']}"
        passages.append(f"[{source}] (relevance: {score:.2f})\n{doc}")
        sources.append({
            "parva": meta["parva"],
            "chapter": meta["chapter"],
            "chapter_title": meta.get("chapter_title", ""),
            "score": round(score, 3),
            "text": doc[:300],
        })

    context = ""
    if passages:
        context = "RELEVANT PASSAGES FROM THE MAHABHARATA TEXT:\n\n" + "\n\n".join(passages)
    return context, sources

# ── Prompt building ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable scholar of the Mahabharata. Answer questions based on the provided context.

RULES:
- Answer ONLY from the provided context (knowledge graph facts and text passages).
- PRIORITIZE knowledge graph facts for factual questions (parents, spouses, kills, siblings). The graph is curated and accurate.
- Use text passages for narrative details, descriptions, and context.
- If graph says "X SON_OF Y" that means Y is a parent of X.
- If the context doesn't contain enough information, say so honestly.
- Cite which parva/chapter your answer comes from when possible.
- Be concise but thorough. Aim for 3-5 sentences.
- Use canonical names (Arjuna not Partha, Krishna not Vasudeva).

ALIAS REFERENCE:
Partha/Dhananjaya = Arjuna, Vasudeva/Govinda/Keshava = Krishna,
Dharmaraja = Yudhishthira, Bhimasena/Vrikodara = Bhima,
Panchali = Draupadi, Suyodhana = Duryodhana, Radheya = Karna,
Devavrata/Pitamaha = Bhishma, Dronacharya = Drona, Pritha = Kunti"""

def build_prompt(query, graph_context, vector_context):
    parts = [SYSTEM_PROMPT, ""]
    if graph_context:
        parts.append(graph_context)
        parts.append("")
    if vector_context:
        parts.append(vector_context)
        parts.append("")
    if not graph_context and not vector_context:
        parts.append("No relevant context found.")
        parts.append("")
    parts.append(f"QUESTION: {query}")
    parts.append("\nANSWER:")
    return "\n".join(parts)

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    neo4j_ok = False
    try:
        neo4j_driver.verify_connectivity()
        neo4j_ok = True
    except Exception:
        pass
    return {
        "status": "ok",
        "neo4j": neo4j_ok,
        "chromadb_chunks": collection.count() if collection else 0,
        "known_entities": len(known_names),
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """RAG chat endpoint. Returns streamed response via SSE."""
    entities = detect_entities(req.message)
    graph_context, graph_edges = get_graph_context(entities)
    vector_context, sources = get_vector_context(req.message)
    prompt = build_prompt(req.message, graph_context, vector_context)

    async def stream():
        # Send metadata first.
        meta = {
            "type": "meta",
            "entities": entities,
            "graph_edges": graph_edges,
            "sources": sources,
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # Stream LLM response.
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.3, "num_predict": 500},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                        if data.get("done"):
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            break

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming chat endpoint for testing."""
    entities = detect_entities(req.message)
    graph_context, graph_edges = get_graph_context(entities)
    vector_context, sources = get_vector_context(req.message)
    prompt = build_prompt(req.message, graph_context, vector_context)

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 500},
        })
        data = response.json()

    return {
        "answer": data.get("response", ""),
        "entities": entities,
        "graph_edges": graph_edges,
        "sources": sources,
    }

@app.post("/api/search")
async def search(req: SearchRequest):
    """Semantic search over Mahabharata text."""
    query_embedding = embedder.encode([req.query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=req.top_k)

    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({
            "text": doc,
            "parva": meta["parva"],
            "chapter": meta["chapter"],
            "chapter_title": meta.get("chapter_title", ""),
            "score": round(1 - dist, 3),
        })

    return {"query": req.query, "results": items}

@app.get("/api/graph/character/{name}")
async def get_character(name: str):
    """Get a character and all their relationships."""
    with neo4j_driver.session() as s:
        # Check if character exists.
        result = s.run("MATCH (n {name: $name}) RETURN n.name as name, labels(n)[0] as label", {"name": name})
        record = result.single()
        if not record:
            raise HTTPException(status_code=404, detail=f"Character '{name}' not found")

        # Get all relationships.
        result = s.run(
            """
            MATCH (a {name: $name})-[r]->(b)
            RETURN a.name as src, type(r) as rel, b.name as tgt, 'outgoing' as dir, r.origins as origins
            UNION ALL
            MATCH (b)-[r]->(a {name: $name})
            RETURN b.name as src, type(r) as rel, a.name as tgt, 'incoming' as dir, r.origins as origins
            """,
            {"name": name}
        )
        relationships = []
        for r in result:
            relationships.append({
                "source": r["src"],
                "type": r["rel"],
                "target": r["tgt"],
                "direction": r["dir"],
                "origins": r.get("origins", ""),
            })

    return {"name": name, "relationships": relationships}

@app.get("/api/graph/path/{name1}/{name2}")
async def get_path(name1: str, name2: str):
    """Find shortest path between two characters."""
    with neo4j_driver.session() as s:
        result = s.run(
            """
            MATCH path = shortestPath((a {name: $n1})-[*..6]-(b {name: $n2}))
            RETURN [n IN nodes(path) | n.name] as nodes,
                   [r IN relationships(path) | type(r)] as rels
            """,
            {"n1": name1, "n2": name2}
        )
        record = result.single()
        if not record:
            return {"found": False, "nodes": [], "relationships": []}
        return {
            "found": True,
            "nodes": record["nodes"],
            "relationships": record["rels"],
        }

@app.get("/api/graph/subgraph/{name}")
async def get_subgraph(name: str, depth: int = Query(default=1, ge=1, le=3)):
    """Get a subgraph around a character for visualization."""
    with neo4j_driver.session() as s:
        result = s.run(
            f"""
            MATCH path = (a {{name: $name}})-[*1..{depth}]-(b)
            WITH nodes(path) as ns, relationships(path) as rs
            UNWIND ns as n
            WITH DISTINCT n, rs
            UNWIND rs as r
            WITH DISTINCT n, r
            RETURN collect(DISTINCT {{name: n.name, label: labels(n)[0]}}) as nodes,
                   collect(DISTINCT {{source: startNode(r).name, target: endNode(r).name, type: type(r)}}) as edges
            """,
            {"name": name}
        )
        record = result.single()
        if not record:
            return {"nodes": [], "edges": []}
        return {
            "nodes": record["nodes"],
            "edges": record["edges"],
        }

@app.get("/api/graph/stats")
async def graph_stats():
    """Get graph statistics."""
    with neo4j_driver.session() as s:
        node_result = s.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC")
        nodes = {r["label"]: r["cnt"] for r in node_result}

        rel_result = s.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as cnt ORDER BY cnt DESC")
        rels = {r["type"]: r["cnt"] for r in rel_result}

        top_result = s.run("MATCH (c:Character)-[r]-() RETURN c.name as name, count(r) as connections ORDER BY connections DESC LIMIT 10")
        top = [{"name": r["name"], "connections": r["connections"]} for r in top_result]

    return {"nodes": nodes, "relationships": rels, "top_characters": top}

@app.get("/api/entities")
async def list_entities():
    """List all known character names."""
    return {"entities": sorted(known_names)}
