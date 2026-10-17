# VyasaGraph development log

A factual record of what was changed and why, kept so the README can
describe the project truthfully. Numbers are measured on this machine
(Apple Silicon Mac), not estimated, unless marked otherwise.

## State before this rework (commit 795374c)

- Source text: `data/1-18_books_combined.txt`, 1.7 MB, ~291k words, 17 parvas
  (no Svargarohana). The README called it the KM Ganguli translation of
  1.8M words. It is actually the *Summary Study of the Mahabharata*, a prose
  retelling with per-chapter commentary (written in the New Vrindavana
  community, per its colophon). The same text is on wisdomlib.org.
- Pipeline: parsing logic copied into four scripts; `src/ingestion/` could
  not be imported (it referenced `src.config` and `src.nlp`, which did not
  exist); graph built by `scripts/build_enriched_graph.py`; vectors by
  `scripts/build_vectorstore.py` into ChromaDB; 960 chunks;
  all-MiniLM-L6-v2 embeddings; Ollama llama3.1:8b.
- `requirements.txt` belonged to another project (streamlit, mediapipe,
  yt-dlp) and lacked fastapi, neo4j, chromadb, sentence-transformers.
- `docker-compose.yml` defined `api` and `frontend` services whose
  Dockerfiles did not exist, and an `.env` file that did not exist.

## Problems found

### Text and retrieval
1. **Chunks were mostly not embedded.** all-MiniLM-L6-v2 reads at most 256
   word-pieces. 903 of 960 chunks (94%) were longer; on average ~40% of
   each chunk was never seen by the embedder.
2. **The word "day" is missing from the summary text** (0 occurrences, case
   insensitive): "One , the King…", "ten s", "few s". Preprocessing upstream
   of this repo stripped the substring. Not fixable from the file itself.
3. **The Bhagavad-gita is nested inside Bhishma Parva Chapter 1** of the
   summary and reuses "Chapter One…Eighteen" headers, which scrambled
   chapter numbers in citations.
4. **Header typos** in the summary: "Chpater Six" (Bhishma Parva; the old
   regex merged Chapter 6 into 5) and a second "Chapter Ten" in Drona Parva
   (really Chapter Eleven).
5. **Commentary (9% of the summary) was indexed and cited as epic text.**
   It is a modern author's interpretation (Gita quotes, theology).
6. **End matter** ("Concluding Words") was filed under the last parva.
7. **Chunk IDs were positional** (`chunk-{i}`), so adding text renumbered
   everything; only full rebuilds were possible.
8. **Entity detection used substring matching**, e.g. "hari" matched inside
   "chariot", adding Krishna's facts to unrelated questions.
9. The relationship extractor (`extract_attested.py`) also read commentary
   text. Re-running it on narrative only: 2 approved facts are no longer
   extracted (Bhima son of Vayu, Satya daughter of Nagnajit); both are true
   and were kept.

### Graph
10. **All 94 Wikidata spouse rows were never loaded**: the reader expected
    `childLabel/parentLabel` columns in a file that has
    `personLabel,spouseLabel`.
11. **Three people merged into one "Lakshmana" node**: Duryodhana's son,
    his daughter Lakshmanaa (who married Samba), and Krishna's wife. The
    graph claimed Krishna and Samba married Duryodhana's son.
12. **Karna's foster mother Radha merged with the Puranic Radha**, so the
    graph had Krishna married to Karna's mother.
13. **Duplicate nodes for one person**: Santanu/Shantanu (diacritics of
    Śāntanu stripped to a different spelling), Devavrata/Bhishma,
    Shurasena/Surasena, Prativindya/Prativindhya, Srutakarman/Shrutakarma,
    God Brahma/Brahma, Rohini Devi/Rohini.
14. **Daughters stored as SON_OF** (e.g. "Kunti SON_OF Shurasena"), while the
    text extractor used DAUGHTER_OF, so the same fact from two sources never
    deduplicated.
15. **Wikidata noise**: "Askia Musa" (a 15th-century Songhai emperor) listed
    as Kamsa's son; group nodes "Pandava", "Wives of Karna"; Jagannath as a
    sibling of Balarama; Puranic/folk figures (Barbarika, Ashokasundari,
    Alakshmi, Lalita, Krishna–Rohini, Tilottama–Aniruddha).
16. A later key in the name map (`"Lakshman": "Lakshmana"`) silently
    overrode an earlier one; found while fixing #11.

### App and tooling
17. The UI listed more graph edges than the LLM was shown (the API
    truncated the prompt to 20 lines but returned all edges).
18. The frontend hung forever if the LLM was down (no error handling), and
    the header counts (228 / 315 / 960) were hardcoded.
19. CORS allowed `*` together with credentials.
20. `run.sh stop` killed any running Ollama, even one it did not start;
    `run.sh status` always reported Neo4j stopped (it grepped for "running",
    Compose prints "Up"); `run.sh start` waited a fixed 3 s for a backend
    that needs ~10 s to load models.

### Found while building the new pipeline
21. **Ganguli layout differs between books**: books 8–11 and 16–18 number
    sections with bare "1", "2" lines instead of "SECTION I"; the first
    parser version silently dropped those 7 books.
22. **Missing sub-parva markers in the Gutenberg text**: no "(Bhishma-vadha
    Parva)" marker, so Bhishma Parva Sections 43–124 were labelled
    "Bhagavat-Gita Parva". The editor's "[End of the Bhagavad Gita]" note in
    Section 42 proves the boundary; fixed with a one-entry override. Drona
    Parva's Samsaptaka-vadha and Pratijna sub-parvas are also unmarked;
    their exact boundaries could not be proven from the text, so those
    sections keep the coarser preceding label (known limitation).
23. **Editor notes in the Gutenberg text** ("[...--JBH]", "([This where is
    the Bhagavad Gita proper starts ...])"); removed. The editor's Gita
    chapter labels are used in citations instead ("Gita Chapter 2").
24. **The retellings crowded out the translation**: the summary is 10% of
    the chunks but filled 42% of the top-5 results (shorter, modern-English
    chunks score higher). Fixed with per-source quotas.
25. **ChromaDB's approximate search lost results under a metadata filter**:
    with `where={"source_kind": "translation"}`, Chroma returned the exact
    top-5 for only 66% of 105 test questions (mean recall@5 91%, worst 40%);
    raising `n_results` did not help. Example: the first Gita section (score
    0.724, true rank 3) never came back for "What did Krishna teach Arjuna on
    the battlefield?". Replaced with exact search (below).
26. **ChromaDB does not reclaim disk space**: replacing all chunks grew the
    store from 192 MB to 368 MB, and deleting the collection left the old
    segment folder behind (400 MB).

## What was built

### Sources
- `data/sources/ganguli/`: KM Ganguli's complete English translation
  (1883–1896), Project Gutenberg eBooks #15474–#15477, unmodified, public
  domain in the US. All 18 parvas, 2,107 sections, ~2.40M words of text.
- `data/sources/summary_study/`: the original summary file, unchanged
  apart from the move. Kept as a second source (modern-English retellings
  help with plain-language questions); "day" is still missing (#2).
- Each source folder has a `source.yaml`: `id`, `title`, `kind`
  (translation | summary | commentary), `parser`, `file` (one or a list).

### Ingestion package `src/ingest/` (replaces 5 scripts and `src/ingestion/`)
- `parsers.py`, one parser per text layout:
  - `summary_study`: nested Gita chapters (#3), header typos (#4),
    commentary tagging (#5), stops at the book's final colophon (#6),
    strips stray `<T>`/`<N>` tags. 157 sections (101 narrative, 56
    commentary); every parva's chapters numbered 1..N with no gaps.
  - `ganguli`: extracts each Gutenberg body; `BOOK N`; `SECTION <roman>` or
    bare-number headers (accepted only in sequence, only in books without
    roman headers) (#21); sub-parva names become section titles, with one
    override (#22); drops front matter, FOOTNOTES blocks, `[n]` markers,
    editor notes (#23) and "The end of … Parva" closers. Section numbers
    missing from the Gutenberg text itself (Adi 177, Sabha 67, Drona
    54/55/189, Shanti 35/364; one header reads "SECTION XXXIV-XXXV") are
    left as is; their text stays under the preceding section.
  - `markdown`: `# Parva` / `## Chapter` / `### Commentary`, the format for
    adding new material without writing code.
- `chunker.py`: sentence-aligned chunks of ≤250 word-pieces with 40-token
  overlap, measured with the embedding model's own tokenizer. Per-word
  counts are summed, which matches the tokenizer exactly because it splits
  on whitespace first (verified). 0 of 18,861 chunks exceed the model's
  input limit (#1).
- `text.py`: chunk ID = hash of (embedding model, metadata, text) (#7). The
  index is two files in `data/index/`: `embeddings.npz` (float32,
  normalized vectors plus their chunk IDs) and `chunks.jsonl`; each is
  written to a temporary name and renamed, and loading verifies the IDs in
  both files match row for row. A run reuses vectors
  of unchanged IDs, embeds new ones, and drops removed ones, including all
  chunks of a deleted source folder. Tested: adding a 2-chunk Markdown
  source embedded exactly 2 chunks; deleting its folder removed them; an
  unchanged re-run embeds nothing (7 s).
- `graph.py`: reads the curated CSVs in `data/graph/`, normalizes names,
  maps SON_OF/DAUGHTER_OF to CHILD_OF (#14), applies
  `data/graph/exclusions.csv` (11 documented facts) (#12, #15), fixes #10,
  #11, #13, #16, and rebuilds Neo4j in one transaction with batched
  `UNWIND` writes. Result: 210 characters, 1 event, 289 relationships
  (CHILD_OF 157, SIBLING_OF 52, MARRIED_TO 49, KILLED 26, PARTICIPATED_IN 3,
  MENTOR_OF 2); by origin wikidata 231, attested 82, csv 65; 75 supported by
  2+ origins; 0 isolated nodes. Spot check: Krishna's wives are exactly the
  eight (Ashtabharya).
- `python -m src.ingest [all|text|graph|stats] [--rebuild]`.
- One data edit: `reviewed_relationships.csv` row 124 source "Lakshmana" →
  "Lakshman" (it duplicates row 30, Duryodhana's son), so it no longer lands
  on Krishna's wife (#11).

### Retrieval
- **Embedding model: BAAI/bge-small-en-v1.5** (33M parameters, 384-d, same
  class as MiniLM), with its query instruction prefix. Chosen by
  `scripts/eval_retrieval.py`: 25 questions with gold Ganguli section ranges
  located by searching the text (e.g. Karna's death: Karna Parva 90–91).
  Ranking within the translation:

  | model | hit@3 | hit@5 | hit@10 | MRR@10 |
  |---|---|---|---|---|
  | all-MiniLM-L6-v2 | 56% | 60% | 72%¹ | 0.47 |
  | bge-small-en-v1.5 | 68% | 80% | 84% | 0.55 |

  ¹ hit@8. Remaining misses: Duryodhana's death, Karna's armour, Shishupala,
  Gandhari's curse.
- **Exact search** with numpy over the file index replaces ChromaDB (#25,
  #26): one matrix product over 18,861 × 384 vectors per query. Index size
  55 MB (was 192 MB in Chroma). ChromaDB removed from dependencies.
- **Per-source quotas** (#24): 5 passages from the translation + 2 from the
  other sources per question.

### RAG and API
- `src/config.py`: all settings from environment variables (credentials
  were hardcoded in 4 files).
- `src/rag.py`, shared by the API and scripts:
  - Entity detection: one whole-word, case-insensitive regex over graph
    names and aliases, longest first (#8).
  - Graph facts: up to 25 per entity, 40 total; the API returns exactly the
    facts the LLM sees (#17); phrased as sentences ("X is a child of Y").
  - Commentary passages are labelled "COMMENTARY on …" in the prompt and
    tagged in the UI.
  - LLM: any OpenAI-compatible `/chat/completions` endpoint (default local
    Ollama at `/v1`), so a hosted provider is a config change.
- Guardrail, two steps:
  1. Retrieval gate (no LLM call): refuse when no known character is named
     and the best passage similarity is below `GUARD_MIN_SCORE` = 0.55. Set
     so it never blocks a real question: on 40 on-topic questions the
     lowest score is 0.595; it refuses 18 of 40 off-topic questions.
     A first threshold (0.40, MiniLM) wrongly refused 2 of 20 held-out
     on-topic questions, which is why the gate is deliberately loose.
  2. System prompt: off-topic questions and task requests (poems, code)
     get the fixed refusal sentence.
  Measured with llama3.1:8b on 9 off-topic questions that reach the LLM or
  gate: 8 refused (Napoleon, Elon Musk, Harry Potter, the Ramayana, a poem,
  "What would Arjuna think of the iPhone 15?", a prompt-injection asking
  for Python code, the capital of France); "How to meditate?" was answered
  from the Gita's Dhyana Yoga chapter.
- `src/api.py`: same endpoints; SSE `error` event when the LLM fails (tested
  with the LLM unreachable: stream ends with an error event, `/chat/sync`
  returns 503); refusals stream the fixed sentence; CORS from
  `CORS_ORIGINS` without credentials (#19); `/api/health` returns live
  counts. Streaming verified through the Vite dev proxy (first bytes after
  0.08 s).

### Answer quality (llama3.1:8b, local)
- Correct and cited: Arjuna's parents; Karna's death (Arjuna, Anjalika
  weapon, Karna Parva §91); Krishna's eight wives; the house of lac (Vidura's
  warning, Adi Parva §148); Abhimanyu's wife (Uttara).
- Before the retrieval and prompt fixes the model claimed Krishna married
  Draupadi and labelled sources with prompt headings; both fixed by
  restricting relationships to the curated list and rewording the prompt.
- Still weak: the 8B model sometimes cites a section that was not among
  its passages (e.g. "Adi Parva, Section 111" for Karna's birth, a correct
  fact with an invented citation); "What did Krishna teach Arjuna on the
  battlefield?" answers from the pre-Gita Durga hymn (Bhishma §23);
  "Who killed Duryodhana and how?" omits the mace duel (retrieval miss);
  "Who killed Jayadratha?" is right (Arjuna, Drona Parva §145) but adds a
  detail ("fighting on foot") taken from an unrelated passage.

### Frontend
- Shows server/LLM errors instead of hanging; live counts from
  `/api/health` (#18); citations like "Karna Parva, Section 91"; commentary
  tag; origin of each graph fact; page title "VyasaGraph". Type-checks,
  builds and lints clean.
- `VITE_API_URL` lets the built frontend call an API on another host.
- Removed unused Vite boilerplate (App.css, assets, icons.svg, template
  README) and unused API client functions.

### Scripts and ops
- `scripts/extract_attested.py`: shared parser, narrative only (#9); writes
  `data/graph/extracted_<source>.csv` (the old output was renamed
  `extracted_summary_study.csv`); dropped the unused `ground_truth.json`
  writer. Re-run on the summary: 15,047 sentences, 171 unique
  relationships, 45 s.
- `scripts/ask.py` (terminal Q&A, `--context` shows retrieval),
  `scripts/eval_guardrail.py`, `scripts/eval_retrieval.py`.
- Removed: `build_and_load.py`, `build_enriched_graph.py`,
  `build_vectorstore.py`, `verify_parse.py`, `test_rag.py`,
  `src/ingestion/`, `data/ground_truth.json`,
  `data/extraction_checkpoint.json`.
- `requirements.txt` (runtime) and `requirements-extract.txt` (+ spaCy).
- `docker-compose.yml`: Neo4j only. `run.sh`: fixes in #20; refuses to start
  before the index exists.

## Code review fixes
An independent review of the rewritten code found 11 issues; 10 were real
and fixed, each re-tested:
- Relationship types from the hand-edited CSVs went into Cypher unchecked;
  now whitelisted (a typo like "SON OF" fails with a clear message).
- Blocking work (embedding, Neo4j) ran on the async event loop, so one
  request could stall every open stream. Retrieval now runs in a thread
  pool; with 8 search + depth-3 subgraph requests during a stream, the
  longest gap between streamed tokens was 0.06 s.
- `/api/graph/path/X/X` returned 500 (Neo4j's shortestPath rejects equal
  endpoints); now returns the single node.
- The index's two files could disagree after a crash between writes with no
  detection; the vector file now carries chunk IDs that are checked on load.
- Graph facts for multi-character questions were cut in no fixed order, so
  the first character used up the budget; now split evenly (Bhima /
  Arjuna / Karna: 14 / 14 / 13) and ordered parents, spouses, killings,
  siblings.
- A failed `rag.load()` left partial state that made later calls crash; state
  is now published only after everything loads.
- `CORS_ORIGINS` with spaces after commas blocked origins; now stripped.
- Names with modern meanings (Samba, Vajra, Kali, Satya, Usha, alias
  "hari", ...) no longer bypass the guardrail gate by themselves. With
  bge-small these questions still score above 0.55, so the prompt rule
  refuses them ("What is Vajra in Buddhism?", "How do I dance the samba?":
  both refused).
- An LLM reply without `choices` returned 500 instead of 503; whitespace-only
  messages reached the LLM (now 422).
- `run.sh`: honours `INDEX_DIR`, shows Docker errors, and refuses to start
  when port 8000 is taken instead of reporting a false success.
Not a bug: the review flagged `FATHER_OF`/`MOTHER_OF`/`DISCIPLE_OF` types,
but the extractor already rewrites them before writing any CSV row.

## Measurements
- Full embedding of 18,861 chunks with bge-small: ~2 min (MPS). No-op
  re-run: 7 s. Parsing + chunking all sources: ~10 s.
- Index on disk: 55 MB (vectors + IDs 31 MB, chunks 24 MB).
- LLM latency (llama3.1:8b, local): 7–23 s per answer; gate refusals
  ~0.1 s.

## Open issues
- "day" missing from the summary source (#2). Fix: replace the file with a
  clean copy (e.g. from wisdomlib) and re-run `python -m src.ingest text`.
- `exclusions.csv` contains judgement calls (Barbarika, Ashokasundari,
  Puranic spouses); review them.
- Drona Parva sub-parva labels are coarse (#22).
- Invented citations from the 8B model; a larger hosted model is expected
  to help (not yet measured).
- 4 of 25 benchmark questions still miss the gold passages in the top 5.
