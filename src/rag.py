"""Retrieval-augmented answering: entities -> graph facts + passages -> LLM.

Used by the API and by scripts. Call load() once before anything else.
"""

import json
import re

import httpx
import numpy as np

from src import config

REFUSAL = "I can only answer questions about the Mahabharata."

# Epithets a user may type instead of the canonical name. Names that are
# also graph nodes (e.g. Vasudeva is Krishna's epithet and his father's
# name) match both, and the model sorts it out from the facts.
ALIASES = {
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
    "dhritarastra": "Dhritarashtra",
    "baladeva": "Balarama",
}

# Names and aliases with a common modern meaning (a dance, a Linux
# distribution, a given name). They still fetch graph facts, but typing one
# does not by itself make a question on-topic for the guardrail.
GENERIC_TERMS = {"anu", "chandra", "hari", "ila", "jayanti", "kali", "sala", "samba",
                 "satya", "usha", "vajra"}

REL_PHRASES = {
    "CHILD_OF": "is a child of",
    "MARRIED_TO": "was married to",
    "SIBLING_OF": "is a sibling of",
    "KILLED": "killed",
    "MENTOR_OF": "was a teacher of",
    "PARTICIPATED_IN": "took part in",
}

SYSTEM_PROMPT = f"""You are a scholar of the Mahabharata. You explain the Mahabharata only: its story, characters, places, events, and teachings (including the Bhagavad Gita).

Reply with exactly this sentence and nothing else:
"{REFUSAL}"
- when the question is not about the Mahabharata, or
- when the user asks you to do a task instead of explaining the epic (write a poem, story, code, essay, translation, advice), even if it mentions a character.

Otherwise answer from the reference material in the user's message:
- Family relations and killings: state only those listed under "Relationships"; never infer new ones.
- Story detail: use the passages. A passage marked COMMENTARY is a modern author's interpretation; say so if you rely on it.
- Cite passages by the label shown before them, e.g. (Karna Parva, Section 91). Cite nothing that is not shown.
- If the material does not contain the answer, say so plainly. Do not guess or use outside knowledge.
- Do not mention "reference material", "relationships" or "passages"; just answer.
- Use canonical names (Arjuna, not Partha). Be concise: 3-6 sentences."""

_state = {}


def load():
    """Load the embedder, vector index and graph. Safe to call more than once;
    state is only published once everything has loaded."""
    if _state:
        return
    from neo4j import GraphDatabase
    from sentence_transformers import SentenceTransformer

    from src.ingest.text import load_index

    _, texts, metas, embeddings = load_index()
    if not texts:
        raise RuntimeError(f"no vector index in {config.INDEX_DIR}; run `python -m src.ingest text`")
    is_translation = np.array([m["source_kind"] == "translation" for m in metas])
    state = {
        "texts": texts, "metas": metas, "embeddings": embeddings,
        "masks": {"translation": is_translation, "other": ~is_translation},
        "embedder": SentenceTransformer(config.EMBEDDING_MODEL),
    }
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        with driver.session() as s:
            names = [r["name"] for r in s.run("MATCH (n:Character) RETURN n.name AS name")]
            relationships = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    except Exception:
        driver.close()
        raise

    # One whole-word, case-insensitive pattern for every name and alias,
    # longest first so "Lakshmana Kumara" wins over "Lakshmana".
    lookup = {n.lower(): [n] for n in names}
    for alias, name in ALIASES.items():
        lookup.setdefault(alias, [])
        if name not in lookup[alias]:
            lookup[alias].append(name)
    terms = sorted(lookup, key=len, reverse=True)
    state.update(
        driver=driver, names=names, relationships=relationships, lookup=lookup,
        entity_re=re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE),
    )
    _state.update(state)


def close():
    if "driver" in _state:
        _state["driver"].close()
    _state.clear()


def stats():
    return {
        "chunks": len(_state["texts"]),
        "characters": len(_state["names"]),
        "relationships": _state["relationships"],
    }


def entity_names():
    return sorted(_state["names"])


def neo4j_ok():
    try:
        _state["driver"].verify_connectivity()
        return True
    except Exception:
        return False


# ── Retrieval ─────────────────────────────────────────────────────────────

def _matched_terms(question):
    return [m.group(1).lower() for m in _state["entity_re"].finditer(question)]


def detect_entities(question):
    found = []
    for term in _matched_terms(question):
        for name in _state["lookup"][term]:
            if name not in found:
                found.append(name)
    return found


def graph_facts(entities, budget=40):
    """Facts touching each entity (both directions), deduplicated. The budget
    is split evenly across entities, most important relation types first."""
    if not entities:
        return []
    per_entity = max(5, budget // len(entities))
    with _state["driver"].session() as s:
        result = s.run(
            """
            UNWIND $names AS name
            CALL (name) {
                MATCH (a {name: name})-[r]-(b)
                RETURN r ORDER BY
                    CASE type(r) WHEN 'CHILD_OF' THEN 0 WHEN 'MARRIED_TO' THEN 1
                                 WHEN 'KILLED' THEN 2 WHEN 'SIBLING_OF' THEN 3 ELSE 4 END,
                    b.name
                LIMIT $per_entity
            }
            RETURN DISTINCT startNode(r).name AS source, type(r) AS type,
                   endNode(r).name AS target, r.origins AS origins
            """,
            names=entities, per_entity=per_entity,
        )
        return [dict(r) for r in result]


def citation(meta):
    where = f"{meta['parva']}, {meta['unit']} {meta['chapter']}"
    if meta.get("chapter_title"):
        where += f" ({meta['chapter_title']})"
    return where


def _embed_query(question):
    return _state["embedder"].encode([config.EMBEDDING_QUERY_PREFIX + question], normalize_embeddings=True)[0]


def search(question, top_k=5, subset=None, embedding=None):
    """Exact cosine search. `subset` is "translation", "other" or None (all)."""
    query = _embed_query(question) if embedding is None else embedding
    scores = _state["embeddings"] @ query
    if subset:
        scores = np.where(_state["masks"][subset], scores, -np.inf)
    top_k = min(top_k, int(np.isfinite(scores).sum()))
    best = np.argpartition(-scores, top_k - 1)[:top_k] if top_k else []
    hits = []
    for i in sorted(best, key=lambda i: -scores[i]):
        meta = _state["metas"][i]
        hits.append({
            "text": _state["texts"][i],
            "score": round(float(scores[i]), 3),
            "source": meta["source"],
            "source_title": meta["source_title"],
            "section": meta["section"],
            "parva": meta["parva"],
            "chapter": meta["chapter"],
            "chapter_title": meta["chapter_title"],
            "citation": citation(meta),
        })
    return hits


def retrieve(question):
    """Everything needed to answer, plus the guardrail verdict."""
    entities = detect_entities(question)
    embedding = _embed_query(question)
    # Separate quotas so the full translation is never crowded out.
    hits = (search(question, config.PASSAGES_TRANSLATION, "translation", embedding)
            + search(question, config.PASSAGES_OTHER, "other", embedding))
    hits.sort(key=lambda h: -h["score"])
    best = hits[0]["score"] if hits else 0.0
    return {
        "entities": entities,
        "facts": graph_facts(entities),
        "passages": hits,
        "best_score": best,
        # Guardrail, step 1: with no (unambiguous) character named, the question
        # must still be close to some passage. The prompt rule is step 2.
        "on_topic": (any(t not in GENERIC_TERMS for t in _matched_terms(question))
                     or best >= config.GUARD_MIN_SCORE),
    }


# ── Generation ────────────────────────────────────────────────────────────

def build_messages(question, ctx):
    parts = []
    if ctx["facts"]:
        lines = [f"- {f['source']} {REL_PHRASES.get(f['type'], f['type'])} {f['target']}" for f in ctx["facts"]]
        parts.append("Relationships (curated):\n" + "\n".join(lines))
    if ctx["passages"]:
        blocks = []
        for p in ctx["passages"]:
            label = "COMMENTARY on " if p["section"] == "commentary" else ""
            blocks.append(f"[{label}{p['citation']}] from {p['source_title']}\n{p['text']}")
        parts.append("Passages:\n\n" + "\n\n".join(blocks))
    if not parts:
        parts.append("No reference material was found for this question.")
    parts.append(f"Question: {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _llm_request(messages, stream):
    return {
        "url": f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        "headers": {"Authorization": f"Bearer {config.LLM_API_KEY}"},
        "json": {"model": config.LLM_MODEL, "messages": messages, "stream": stream,
                 "temperature": 0.2, "max_tokens": 600},
    }


async def stream_answer(messages):
    """Yield answer tokens. Raises httpx.HTTPError if the LLM is unreachable or errors."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5)) as client:
        async with client.stream("POST", **_llm_request(messages, stream=True)) as response:
            if response.status_code != 200:
                await response.aread()
                response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or [{}]
                token = (choices[0].get("delta") or {}).get("content")
                if token:
                    yield token


async def answer(messages):
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5)) as client:
        response = await client.post(**_llm_request(messages, stream=False))
        response.raise_for_status()
        choices = response.json().get("choices") or []
        if not choices:
            raise ValueError("LLM response has no choices")
        return choices[0]["message"]["content"]
