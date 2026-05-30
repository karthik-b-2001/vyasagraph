"""Phase 5: RAG pipeline — Graph + Vector + Ollama LLM.

Combines:
  - Neo4j knowledge graph for structured facts (relationships, family trees)
  - ChromaDB vector store for relevant text passages
  - Ollama (llama3.2:3b) for answer generation

Usage:
    python scripts/test_rag.py
    python scripts/test_rag.py --interactive
"""

import argparse
import json
import re
from pathlib import Path

import chromadb
import httpx
from neo4j import GraphDatabase
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

# ── Alias map for query entity detection ──────────────────────────────────

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

# ── Entity detection from query ───────────────────────────────────────────

def detect_entities(query, known_names):
    """Find Mahabharata character names in the query."""
    found = set()
    query_lower = query.lower()
    # Check aliases first.
    for alias, canonical in ALIAS_MAP.items():
        if alias in query_lower:
            found.add(canonical)
    # Check known entity names.
    for name in known_names:
        if name.lower() in query_lower:
            found.add(name)
    return list(found)

def get_known_names(neo4j_session):
    """Get all character names from Neo4j."""
    result = neo4j_session.run("MATCH (n:Character) RETURN n.name as name")
    return [r["name"] for r in result]

# ── Neo4j graph retrieval ─────────────────────────────────────────────────

def get_graph_context(session, entities, max_rels=20):
    """Fetch relationships for detected entities from Neo4j."""
    if not entities:
        return ""

    lines = []
    for entity in entities:
        # Outgoing relationships.
        result = session.run(
            """
            MATCH (a {name: $name})-[r]->(b)
            RETURN a.name as src, type(r) as rel, b.name as tgt, r.context as ctx
            LIMIT $limit
            """,
            {"name": entity, "limit": max_rels}
        )
        for record in result:
            ctx = f" ({record['ctx']})" if record.get("ctx") else ""
            lines.append(f"  {record['src']} --[{record['rel']}]--> {record['tgt']}{ctx}")

        # Incoming relationships.
        result = session.run(
            """
            MATCH (b)-[r]->(a {name: $name})
            RETURN b.name as src, type(r) as rel, a.name as tgt, r.context as ctx
            LIMIT $limit
            """,
            {"name": entity, "limit": max_rels}
        )
        for record in result:
            ctx = f" ({record['ctx']})" if record.get("ctx") else ""
            lines.append(f"  {record['src']} --[{record['rel']}]--> {record['tgt']}{ctx}")

    # Deduplicate.
    unique = list(dict.fromkeys(lines))
    if not unique:
        return ""
    return "KNOWN FACTS FROM KNOWLEDGE GRAPH:\n" + "\n".join(unique[:max_rels])

# ── ChromaDB vector retrieval ─────────────────────────────────────────────

def get_vector_context(collection, embedder, query, top_k=5):
    """Retrieve relevant text passages from ChromaDB."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1 - dist
        if score < 0.2:  # skip very low relevance
            continue
        source = f"{meta['parva']}, Chapter {meta['chapter']}"
        passages.append(f"[{source}] (relevance: {score:.2f})\n{doc}")

    if not passages:
        return ""
    return "RELEVANT PASSAGES FROM THE MAHABHARATA TEXT:\n\n" + "\n\n".join(passages)

# ── Ollama LLM generation ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable scholar of the Mahabharata. Answer questions based on the provided context.

RULES:
- Answer ONLY from the provided context (knowledge graph facts and text passages).
- If the context doesn't contain enough information, say so honestly.
- Cite which parva/chapter your answer comes from when possible.
- Be concise but thorough. Aim for 3-5 sentences unless more detail is needed.
- Use the character's canonical names (Arjuna not Partha, Krishna not Vasudeva).
- If the knowledge graph provides a direct fact (like "X killed Y"), state it confidently.
- If the text passages give narrative detail, weave it into your answer."""

def build_prompt(query, graph_context, vector_context):
    """Construct the full prompt for the LLM."""
    parts = [SYSTEM_PROMPT, ""]

    if graph_context:
        parts.append(graph_context)
        parts.append("")

    if vector_context:
        parts.append(vector_context)
        parts.append("")

    if not graph_context and not vector_context:
        parts.append("No relevant context was found in the knowledge graph or text.")
        parts.append("")

    parts.append(f"QUESTION: {query}")
    parts.append("\nANSWER:")

    return "\n".join(parts)

def generate_answer(prompt, stream=True):
    """Send prompt to Ollama and return the response."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "num_predict": 500,
        },
    }

    if stream:
        answer = ""
        with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=120) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    print(token, end="", flush=True)
                    answer += token
                    if data.get("done"):
                        break
        print()
        return answer
    else:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=120)
        data = response.json()
        return data.get("response", "")

# ── Full RAG pipeline ─────────────────────────────────────────────────────

def rag_query(query, neo4j_session, collection, embedder, known_names):
    """Run the full RAG pipeline for a single query."""
    # Step 1: Detect entities in the query.
    entities = detect_entities(query, known_names)

    # Step 2: Get graph context.
    graph_context = get_graph_context(neo4j_session, entities)

    # Step 3: Get vector context.
    vector_context = get_vector_context(collection, embedder, query, top_k=5)

    # Step 4: Build prompt.
    prompt = build_prompt(query, graph_context, vector_context)

    # Step 5: Generate answer.
    answer = generate_answer(prompt)

    return {
        "query": query,
        "entities": entities,
        "graph_context": graph_context,
        "vector_context": vector_context,
        "answer": answer,
    }

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5: RAG PIPELINE")
    print("=" * 70)

    # Connect to everything.
    print("\n  Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("  Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"  Collection: {collection.count()} chunks")

    print("  Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    session = driver.session()

    known_names = get_known_names(session)
    print(f"  Known entities: {len(known_names)}")

    if args.interactive:
        # Interactive mode.
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE (type 'quit' to exit)")
        print("=" * 70)

        while True:
            print()
            query = input("  Ask about the Mahabharata: ").strip()
            if not query or query.lower() in ("quit", "exit", "q"):
                break

            entities = detect_entities(query, known_names)
            if entities:
                print(f"  Entities detected: {', '.join(entities)}")

            print()
            result = rag_query(query, session, collection, embedder, known_names)
            print()

    else:
        # Test mode.
        test_queries = [
            "Who were Arjuna's parents?",
            "How did Bhishma die?",
            "What happened to Abhimanyu in the Chakravyuha?",
            "Who killed Duryodhana and how?",
            "What was Karna's true identity?",
            "Why did the Kurukshetra war happen?",
            "What role did Krishna play in the war?",
        ]

        for query in test_queries:
            print(f"\n{'─' * 70}")
            print(f"  Q: {query}")
            entities = detect_entities(query, known_names)
            if entities:
                print(f"  Entities: {', '.join(entities)}")
            print(f"{'─' * 70}")

            result = rag_query(query, session, collection, embedder, known_names)

            # Show context summary.
            g_lines = len(result["graph_context"].split("\n")) - 1 if result["graph_context"] else 0
            v_chunks = result["vector_context"].count("[") if result["vector_context"] else 0
            print(f"\n  [Context: {g_lines} graph facts, {v_chunks} text passages]")

    session.close()
    driver.close()
    print("\n  Done.")

if __name__ == "__main__":
    main()
