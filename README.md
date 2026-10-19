# VyasaGraph

A question-answering app for the Mahabharata. Answers are grounded in two
things: KM Ganguli's complete English translation of the epic, and a curated
knowledge graph of who is whose child, spouse, sibling, killer or teacher.
Every answer shows the graph facts and the passages it was built from, with
citations such as "Karna Parva, Section 91". Questions that are not about the
Mahabharata are refused.

The app has two tabs: **Chat**, and **Architecture**, an interactive flow
chart that walks through every stage below.

## How it works

```text
                 BUILDING THE DATA  (python -m src.ingest)

 data/sources/*/ ─► parser ─► chunker ─► content-hash IDs ─► bge-small ─► data/index/
 (Ganguli translation,         (≤250                          embedder      embeddings.npz
  Summary Study, your notes)    word-pieces)                                chunks.jsonl

 data/graph/*.csv ─► normalize names, merge sources, apply exclusions ─► Neo4j
 (Wikidata, reviewed spaCy extractions, family table)


                 ANSWERING A QUESTION  (FastAPI, streamed over SSE)

 question ─┬─► character names + epithets ─► graph facts (Neo4j) ─┐
           │                                                       ├─► guardrail ─┬─► prompt ─► LLM ─► answer
           └─► query embedding ─► exact search: 5 translation  ────┘   gate       │
                                  passages + 2 from other sources                 └─► fixed refusal (no LLM call)
```

## Data

### Text sources

| Source | What it is | Size |
|---|---|---|
| `data/sources/ganguli/` | KM Ganguli's complete English prose translation (1883–1896), Project Gutenberg eBooks #15474–#15477, unmodified. Public domain in the US. | All 18 parvas, 2,107 sections, ~2.4M words → 16,948 chunks |
| `data/sources/summary_study/` | *Summary Study of the Mahabharata*, a modern prose retelling with a commentary after many chapters (written in the New Vrindavana community, per its closing colophon). Not a translation. | 17 parvas, 157 sections (101 narrative, 56 commentary) → 1,913 chunks |

Each source folder has a `source.yaml` naming its parser. Three parsers exist:

- **ganguli**: handles the Gutenberg layout. Section headers are roman
  numerals in most books but bare numbers in books 8–11 and 16–18. It drops
  prefaces, footnotes and the Gutenberg editor's notes, and uses the
  sub-parva names (e.g. "Jayadratha-Vadha Parva") in citations.
- **summary_study**: separates narrative from commentary, so commentary is
  never presented as the epic itself. It also handles the Bhagavad-gita
  nested inside Bhishma Parva and the file's header typos.
- **markdown**: the format for adding your own material (see below).

### Knowledge graph

The graph is built only from curated files in `data/graph/`. It is not
generated from the text at query time.

| Origin | File(s) | Facts after merging |
|---|---|---|
| Wikidata | `wikidata_*.csv` (parents, spouses, siblings, children, killers, conflicts) | 231 |
| Text extraction, reviewed by hand | `reviewed_relationships.csv` | 82 |
| Family-relations table | `csv_family.csv` | 65 |

The loader:

- maps spellings to one name (diacritics, titles, epithets: Śāntanu and
  Santanu → Shantanu);
- stores parent links as one type, `CHILD_OF`;
- merges duplicates while keeping every origin that supports a fact;
- drops the 11 facts listed with reasons in `exclusions.csv` (e.g.
  Krishna–Radha, which comes from later devotional tradition).

The result is **210 characters, 1 event and 289 relationships**: CHILD_OF
157, SIBLING_OF 52, MARRIED_TO 49, KILLED 26, PARTICIPATED_IN 3, MENTOR_OF
2. **75 relationships are supported by two or more origins.**

The text extraction is done by `scripts/extract_attested.py`, an offline
tool:

- It parses every narrative sentence with spaCy's dependency parser.
- It extracts epithets ("X, son of Y") and kill statements
  (subject–verb–object).
- It counts how many separate sentences support each relationship.
- The output is reviewed by hand. Only approved rows go into
  `reviewed_relationships.csv`.

## Retrieval

- **Chunking.** Text is split into sentence-aligned chunks of at most 250
  word-pieces, measured with the embedding model's own tokenizer. None of
  the 18,861 chunks exceeds the model's input limit.
- **Embeddings.** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
  (33M parameters, 384 dimensions), run locally.
- **Search.** Exact cosine similarity over all vectors, with numpy. At this
  size one matrix product takes milliseconds, so no vector database is
  needed. The index is two files totalling 55 MB.
- **Per-source quotas.** Each question gets the 5 best passages from the
  translation and the 2 best from other sources. Without the quota, the
  summary's shorter modern-English chunks filled 42% of the top results
  while being 10% of the corpus.

`scripts/eval_retrieval.py` measures this on 25 questions with known answer
locations in the Ganguli text, such as Karna's death in Karna Parva §90–91.
It ranks within the translation:

| Embedding model | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 56% | 60% | 0.47 |
| **bge-small-en-v1.5** (used) | 68% | 80% | 0.55 |

## Guardrail

Off-topic questions are stopped in two steps.

1. **Gate (no LLM call).** A question passes if it names a known character,
   or if its best passage scores at least `GUARD_MIN_SCORE` (0.55).
   Everything else gets a fixed reply (`REFUSAL` in `src/rag.py`) in about
   0.1 s. Names with everyday meanings (Samba,
   Vajra, Kali, …) do not count on their own. The threshold is deliberately
   loose: on `scripts/eval_guardrail.py` it passes all 40 on-topic questions
   and stops 18 of 40 off-topic ones.
2. **Prompt rule.** The model must refuse off-topic questions and task
   requests (poems, code) with the same sentence.

In a run with llama3.1:8b, both steps together refused the off-topic tests
that were tried:

- the capital of France, Napoleon, Elon Musk, Harry Potter, the Ramayana
- "Write a poem about the ocean"
- "What would Arjuna think of the iPhone 15?"
- a prompt injection asking for Python code
- "What is Vajra in Buddhism?" and "How do I dance the samba?"

"How to meditate?" was answered from the Gita's chapter on meditation.

## Tech stack

| Layer | Technology |
|---|---|
| Text processing | Python, regex parsers, spaCy (offline extraction only) |
| Knowledge graph | Neo4j 5 |
| Embeddings and search | sentence-transformers (bge-small-en-v1.5), numpy |
| LLM | Any OpenAI-compatible chat API; default llama3.1:8b via Ollama |
| Backend | FastAPI, Pydantic v2, httpx, Server-Sent Events |
| Frontend | React 19, TypeScript, Tailwind CSS 4, Vite |
| Data | KM Ganguli translation (Project Gutenberg), *Summary Study of the Mahabharata*, Wikidata, a family-relations CSV |

## Quick start

Prerequisites: Python 3.12+ (tested on 3.13), Node.js 18+, Docker, and
[Ollama](https://ollama.com).

```bash
git clone https://github.com/karthik-b-2001/vyasagraph.git
cd vyasagraph

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

(cd frontend && npm install)
ollama pull llama3.1:8b

docker compose up neo4j -d --wait   # Neo4j on :7474 / :7687
python -m src.ingest                # builds data/index/ and the graph

./run.sh start                      # Neo4j, Ollama, API on :8000, UI on :5173
```

Open http://localhost:5173. `./run.sh status` shows what is running;
`./run.sh stop` stops it.

To use a hosted LLM instead of Ollama, set three environment variables:
`LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`.

Embedding the corpus takes about 2 minutes on an Apple Silicon Mac. A re-run
with nothing changed takes about 7 seconds.

## Deploying (free tiers)

| Piece | Host | Notes |
|---|---|---|
| Graph | Neo4j AuraDB Free | Load it with `python -m src.ingest graph` and the Aura credentials. It pauses after 3 days without activity; `.github/workflows/keepalive.yml` queries it daily. |
| LLM | Groq (free tier), `openai/gpt-oss-120b` | Free tier: 8,000 tokens/minute, and a question uses ~2,500, so about 3 questions per minute in total. The UI says so when the limit is hit. |
| Backend | Google Cloud Run | The `Dockerfile` bakes in the embedding model and `data/index/`. ~500 MB memory. Measured on Cloud Run: a cold start after idle takes 37–52 s, mostly PyTorch loading from the image (5 s locally); warm requests ~0.3 s. The UI calls `/api/health` on page load to start it early and shows a notice while it wakes. |
| Frontend | GitHub Pages | `.github/workflows/pages.yml` builds it with `VITE_API_URL` pointing at the backend and publishes it on every push to `main` that touches `frontend/`. |

1. Put the hosted settings in `.env` (git-ignored):
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
   - `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_REASONING_EFFORT`

   Then load the graph with `python -m src.ingest graph`.
2. Build the index if it is not there: `python -m src.ingest text`.
3. Deploy the backend. `.gcloudignore` uploads only `Dockerfile`,
   `requirements.txt`, `src/` and `data/index/`:
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   sed -E 's/^([A-Z0-9_]+)=(.*)$/\1: "\2"/' .env > env.yaml   # git-ignored
   gcloud run deploy vyasagraph-api --source . --region us-central1 \
     --memory 1Gi --cpu 1 --cpu-boost --execution-environment gen1 \
     --min-instances 0 --max-instances 1 \
     --allow-unauthenticated --env-vars-file env.yaml
   ```
4. In the GitHub repository:
   - Under Settings → Pages, set Source to **GitHub Actions**.
   - Under Settings → Secrets and variables → Actions → Variables, add `API_URL`
     with the Cloud Run URL. It is used by both the Pages build and the Aura
     keep-alive.
   - Push to `main`, or run the "Deploy frontend to GitHub Pages" workflow.
     The site appears at `https://<user>.github.io/<repo>/`.
5. Restrict CORS to the frontend's origin:
   ```bash
   gcloud run services update vyasagraph-api --region us-central1 \
     --update-env-vars CORS_ORIGINS=https://<user>.github.io
   ```

`--max-instances 1` caps cost; `--min-instances 0` lets the service scale to
zero, so the first request after idle waits for a cold start.

## Adding more material

Ingestion is incremental. Each chunk's ID is a hash of the embedding model,
the chunk's metadata and its text, so a re-run:

- embeds only new chunks;
- reuses the vectors of unchanged ones;
- drops chunks that no longer exist, including every chunk of a deleted
  source folder.

To add notes, create a folder with a `source.yaml` and a Markdown file:

```yaml
# data/sources/my_notes/source.yaml
id: my_notes
title: My notes on the Mahabharata
kind: commentary        # translation | summary | commentary
parser: markdown
file: notes.md
```

```markdown
# Karna Parva
## The chariot wheel
Text that will be cited as "Karna Parva, Chapter 1 (The chariot wheel)".
### Commentary
Text that will be tagged as commentary.
```

Then run `python -m src.ingest text`.

A source in a new layout needs one parser function in
`src/ingest/parsers.py`.

Graph facts are changed by editing the CSVs in `data/graph/` and running
`python -m src.ingest graph`. Relationship types are checked against a fixed
list.

| Command | What it does |
|---|---|
| `python -m src.ingest` | Text index and graph |
| `python -m src.ingest text [--rebuild]` | Vector index only (incremental unless `--rebuild`) |
| `python -m src.ingest graph` | Rebuild Neo4j from `data/graph/` |
| `python -m src.ingest stats` | Chunk counts and token lengths; writes nothing |

## Configuration

All settings are environment variables with local defaults (`src/config.py`).
They can also go in a `.env` file in the project root (git-ignored); variables
set in the environment take priority.

| Variable | Default |
|---|---|
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | `bolt://localhost:7687`, `neo4j`, `vyasagraph` |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | `http://localhost:11434/v1`, `ollama`, `llama3.1:8b` |
| `LLM_REASONING_EFFORT` | empty; `low` / `medium` / `high` for reasoning models such as gpt-oss |
| `EMBEDDING_MODEL`, `EMBEDDING_QUERY_PREFIX` | `BAAI/bge-small-en-v1.5` and its query instruction |
| `PASSAGES_TRANSLATION`, `PASSAGES_OTHER` | `5`, `2` |
| `GUARD_MIN_SCORE` | `0.55` |
| `INDEX_DIR` | `data/index` |
| `CORS_ORIGINS` | `*` (comma-separated list) |
| `VITE_API_URL` (frontend build) | empty, so the UI calls `/api` on its own host |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Status and live counts (chunks, characters, relationships) |
| `/api/chat` | POST | Streamed answer (SSE events: `meta`, `token`, `done` or `error`) |
| `/api/chat/sync` | POST | Same, as a single JSON response |
| `/api/search` | POST | Semantic search over all passages |
| `/api/graph/character/{name}` | GET | A character's relationships |
| `/api/graph/path/{name1}/{name2}` | GET | Shortest path between two characters (up to 6 hops) |
| `/api/graph/subgraph/{name}?depth=1-3` | GET | Nodes and edges around a character |
| `/api/graph/stats` | GET | Counts by type, most connected characters |
| `/api/entities` | GET | All character names |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/ask.py "question" [--context]` | Ask from the terminal; `--context` prints what was retrieved |
| `scripts/eval_retrieval.py [model] [prefix]` | Retrieval benchmark (any embedding model) |
| `scripts/eval_guardrail.py` | Guardrail gate on 40 on-topic and 40 off-topic questions |
| `scripts/extract_attested.py [source_id]` | spaCy relationship extraction for review (`pip install -r requirements-extract.txt`) |

## Project structure

```
vyasagraph/
├── src/
│   ├── config.py             # settings from environment variables
│   ├── rag.py                # entities, graph facts, search, guardrail, prompt, LLM
│   ├── api.py                # FastAPI endpoints, SSE
│   └── ingest/
│       ├── parsers.py        # ganguli, summary_study, markdown
│       ├── chunker.py        # token-budgeted, sentence-aligned chunks
│       ├── text.py           # sources → vector index (incremental)
│       ├── graph.py          # curated CSVs → Neo4j
│       └── __main__.py       # python -m src.ingest
├── scripts/                  # ask, eval_retrieval, eval_guardrail, extract_attested
├── data/
│   ├── sources/              # ganguli/ (4 Gutenberg volumes), summary_study/
│   ├── graph/                # Wikidata CSVs, reviewed extractions, family table, exclusions
│   └── index/                # built by ingestion (git-ignored)
├── frontend/src/
│   ├── App.tsx               # chat tab
│   ├── Architecture.tsx      # interactive architecture tab
│   └── lib/api.ts            # API client and SSE parsing
├── docker-compose.yml        # Neo4j
├── run.sh                    # start / stop / status
├── requirements.txt
└── requirements-extract.txt  # + spaCy, for the extractor only
```


## Known limitations

- **Invented citations.** The default 8B model sometimes cites a section it
  was not given, or blends in a detail from an unrelated passage. For
  example, it answered "Who killed Jayadratha?" correctly but added a detail
  taken from Abhimanyu's story. With `openai/gpt-oss-120b` on Groq, spot
  checks (Karna's death, Krishna's wives) were correct and correctly cited,
  but it has not been benchmarked systematically.
- **Retrieval misses.** 4 of the 25 benchmark questions still miss the right
  Ganguli section in the top 5: Duryodhana's mace duel, Karna giving away
  his armour, Shishupala's death, and Gandhari's curse.
- **Missing word in the summary text.** The *Summary Study* file is missing
  every occurrence of the word "day" ("One , the King…"). This happened in
  preprocessing before this repository. Replacing the file with a clean copy
  and re-running `python -m src.ingest text` fixes it.
- **Section numbering in the Gutenberg text.** A few section numbers are
  absent from the source itself: Adi 177, Sabha 67, Drona 54/55/189, Shanti
  35/364. Their text stays under the preceding section. Two Drona Parva
  sub-parvas are not marked in the source, so their sections carry the
  preceding sub-parva's name.
- **Cold starts when hosted.** The free Cloud Run setup scales to zero, so
  the first visit after idle waits 37–52 s for the backend (measured).
- **Judgement calls in `exclusions.csv`.** Some exclusions are judgement
  calls about later tradition, e.g. Barbarika and Ashokasundari.

## License

MIT
