# VyasaGraph

A knowledge graph and RAG chatbot over the Mahabharata. Ask questions about the epic and get answers grounded in the original text and a curated relationship graph.

Built from the ground up: text parsing, NLP extraction, knowledge graph construction, semantic search, and a streaming chat interface.

## What It Does

- **Knowledge Graph**: 228 characters, 315 relationships (family trees, kills, alliances, marriages) in Neo4j, sourced from three independent pipelines and deduplicated with human review
- **Semantic Search**: 960 text chunks embedded locally with sentence-transformers (all-MiniLM-L6-v2) and stored in ChromaDB
- **RAG Chat**: Questions are answered by combining structured graph facts with relevant text passages, streamed through a local LLM (llama3.1:8b via Ollama)
- **React Frontend**: Chat interface with entity detection, expandable graph facts, source citations, and suggested questions

## Architecture

```
Mahabharata Text (1.8M words, 18 parvas)
        │
        ▼
┌─────────────────────┐
│  Text Parsing        │  Split into 17 parvas, 100 chapters, 960 chunks
│  (regex, Python)     │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│ Neo4j  │ │ ChromaDB   │
│ Graph  │ │ Vectors    │
│        │ │            │
│ 228    │ │ 960 chunks │
│ nodes  │ │ embedded   │
│ 315    │ │ locally    │
│ rels   │ │            │
└───┬────┘ └─────┬──────┘
    │            │
    └─────┬──────┘
          ▼
┌─────────────────────┐
│  RAG Pipeline        │  Entity detection → Graph context + Vector context → LLM
│  (FastAPI + Ollama)  │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  React Frontend      │  Streaming chat, citations, graph facts
│  (Vite + Tailwind)   │
└─────────────────────┘
```

## Data Pipeline

The knowledge graph is built from three independent sources, merged and deduplicated:

**1. Attestation-based extraction from text** (91 relationships)
- spaCy dependency parsing on every sentence in the corpus
- Epithet mining: "X, son of Y" extracted from grammatical structure, not regex
- SVO kill extraction: subject-verb-object triples where the verb is kill/slay
- Attestation scoring: relationships ranked by how many independent sentences support them
- Human review of every extracted relationship

**2. Wikidata SPARQL queries** (245 relationships)
- Queried Wikidata for all Mahabharata characters tagged with P1441 (present in work) = Q8276
- Parents (P22, P25), spouses (P26), siblings (P3373), children (P40), killed by (P157), conflicts (P607)
- Sanskrit diacritics normalized to English transliterations

**3. Structured CSV dataset** (67 relationships)
- Pre-built family relationships: Son/Father, Son/Mother, Husband/Wife, Brothers

66 relationships are confirmed by multiple independent sources.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Text Processing | Python, regex, spaCy |
| Knowledge Graph | Neo4j |
| Vector Store | ChromaDB, sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (llama3.1:8b) |
| Backend | FastAPI, Pydantic v2, httpx |
| Frontend | React 19, TypeScript, Tailwind CSS, Vite |
| Data Sources | KM Ganguli translation, Wikidata SPARQL, CSV dataset |

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- Docker
- Ollama

### Setup

```bash
# Clone
git clone https://github.com/karthik-b-2001/vyasagraph.git
cd vyasagraph

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn neo4j chromadb sentence-transformers spacy httpx pydantic pydantic-settings
python -m spacy download en_core_web_sm

# Frontend
cd frontend
npm install
cd ..

# Ollama
ollama pull llama3.1:8b

# Start services
./run.sh start
```

### Or step by step

```bash
# Terminal 1: Neo4j
docker compose up neo4j -d

# Terminal 2: Ollama
ollama serve

# Terminal 3: Build graph + vectorstore
python scripts/build_enriched_graph.py
python scripts/build_vectorstore.py

# Terminal 4: Backend
source venv/bin/activate
python -m uvicorn src.api:app --reload --port 8000

# Terminal 5: Frontend
cd frontend && npm run dev
```

Open http://localhost:5173

## Project Structure

```
vyasagraph/
├── src/
│   └── api.py                      # FastAPI backend (chat, graph, search endpoints)
├── scripts/
│   ├── verify_parse.py             # Phase 1: text parsing verification
│   ├── extract_attested.py         # Phase 2-3: attestation-based NLP extraction
│   ├── extract_ground_truth.py     # Gemini-based extraction (optional)
│   ├── build_enriched_graph.py     # Merge Wikidata + attested + CSV → Neo4j
│   ├── build_vectorstore.py        # Embed chunks into ChromaDB
│   └── test_rag.py                 # Test RAG pipeline from terminal
├── frontend/
│   └── src/
│       ├── App.tsx                 # Chat UI
│       └── lib/api.ts              # API client
├── data/
│   ├── 1-18_books_combined.txt     # Mahabharata source text
│   ├── test_relations.csv          # CSV family relationships
│   ├── reviewed_relationships.csv  # Human-reviewed attested extractions
│   ├── extracted_relationships.csv # Raw NLP extractions with attestation scores
│   ├── wikidata_*.csv              # Wikidata SPARQL query results
│   └── chromadb/                   # Vector store (generated)
├── docker-compose.yml
├── run.sh                          # Start/stop all services
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service status |
| `/api/chat` | POST | Streamed RAG chat (SSE) |
| `/api/chat/sync` | POST | Non-streamed chat |
| `/api/search` | POST | Semantic search over text |
| `/api/graph/character/{name}` | GET | Character relationships |
| `/api/graph/path/{name1}/{name2}` | GET | Shortest path between characters |
| `/api/graph/subgraph/{name}` | GET | Subgraph for visualization |
| `/api/graph/stats` | GET | Graph statistics |
| `/api/entities` | GET | List all character names |

## Sample Queries

- "Who were Arjuna's parents?"
- "How did Karna die?"
- "Who killed Duryodhana and how?"
- "What happened at the dice game?"
- "What was Karna's true identity?"

## What Makes This Different

Most Mahabharata knowledge graphs either use a pre-built dataset or run an LLM over the text and trust the output. This project does neither.

The attestation-based extraction pipeline uses spaCy dependency parsing to find relationships from grammatical structure at the sentence level, then scores each relationship by how many independent sentences support it. A relationship mentioned in 7 separate sentences across 3 parvas is near-certain. One mentioned once in a battle scene is suspect.

The final graph merges three independent sources (text NLP, Wikidata, CSV) with deduplication. 66 relationships are confirmed by multiple sources. Every relationship is traceable to its origin.

## License

MIT