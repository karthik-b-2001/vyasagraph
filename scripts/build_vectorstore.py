"""Phase 4: Embed all text chunks into ChromaDB using a local model.

Uses all-MiniLM-L6-v2 (80MB, runs fast on Apple Silicon).
No API key needed.

Usage:
    python scripts/build_vectorstore.py
"""

import re
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_PATH = DATA_DIR / "1-18_books_combined.txt"
CHROMA_DIR = DATA_DIR / "chromadb"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mahabharata_chunks"
BATCH_SIZE = 64

# ── Parsing (reused) ─────────────────────────────────────────────────────

PARVA_NAMES = {
    "Adi Parva":"Adi Parva","Sabha Parva":"Sabha Parva","Vana Parva":"Vana Parva",
    "Virata Parva":"Virata Parva","Udyoga Parva":"Udyoga Parva",
    "Bhisma Parva":"Bhishma Parva","Drona Parva":"Drona Parva",
    "Karna Parva":"Karna Parva","Salya Parva":"Shalya Parva",
    "Sauptika Parva":"Sauptika Parva","Stree Parva":"Stri Parva",
    "Shanti Parva":"Shanti Parva","Anushasana Parva":"Anushasana Parva",
    "Ashvamedha Parva":"Ashvamedhika Parva","Ashramvasika Parva":"Ashramavasika Parva",
    "Mausala Parva":"Mausala Parva","Mahaprasthanika Parva":"Mahaprasthanika Parva",
}
_NW = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
       r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
       r"twenty|twenty[\s-]?one|twenty[\s-]?two|twenty[\s-]?three|"
       r"twenty[\s-]?four|twenty[\s-]?five|twenty[\s-]?six|twenty[\s-]?seven|"
       r"twenty[\s-]?eight|twenty[\s-]?nine|thirty|thirty[\s-]?one|"
       r"thirty[\s-]?two|thirty[\s-]?three|thirty[\s-]?four|thirty[\s-]?five|\d+")
CHAPTER_RE = re.compile(rf"^Chapter\s+(?:{_NW})\s*$", re.IGNORECASE | re.MULTILINE)
THUS_ENDS_RE = re.compile(r"^Thus Ends.*$", re.IGNORECASE | re.MULTILINE)
COMMENTARY_RE = re.compile(r"^Chapter\s+Commentary\s*$", re.IGNORECASE | re.MULTILINE)

WORD_TO_INT = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
    "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
    "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
    "nineteen":19,"twenty":20,"twenty-one":21,"twenty-two":22,
    "twenty-three":23,"twenty-four":24,"twenty-five":25,"twenty-six":26,
    "twenty-seven":27,"twenty-eight":28,"twenty-nine":29,"thirty":30,
    "thirty-one":31,"thirty-two":32,"thirty-three":33,"thirty-four":34,
    "thirty-five":35,
}

def clean_text(raw):
    text = raw.replace("\r\n","\n").replace("\r","\n")
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

def parse_chapter_number(header):
    match = re.search(r"Chapter\s+(.+)", header, re.IGNORECASE)
    if not match:
        return 0
    word = match.group(1).strip().lower().replace(" ", "-")
    if word.isdigit():
        return int(word)
    return WORD_TO_INT.get(word, 0)

def approx_tokens(text):
    return len(text) // 4

def chunk_text(text, chunk_size=512, overlap=64):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_tokens = [], [], 0
    for sent in sentences:
        st = approx_tokens(sent)
        if current_tokens + st > chunk_size and current:
            chunks.append(" ".join(current))
            ov_s, ov_t = [], 0
            for s in reversed(current):
                t = approx_tokens(s)
                if ov_t + t > overlap: break
                ov_s.insert(0, s); ov_t += t
            current, current_tokens = ov_s, ov_t
        current.append(sent); current_tokens += st
    if current:
        chunks.append(" ".join(current))
    return chunks

def parse_and_chunk():
    """Parse text and return chunks with metadata."""
    raw = TEXT_PATH.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    parva_pattern = re.compile(
        r"^(" + "|".join(re.escape(n) for n in PARVA_NAMES) + r")\s*$",
        re.MULTILINE,
    )
    parts = parva_pattern.split(cleaned)
    all_chunks = []
    chunk_idx = 0
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        canonical = PARVA_NAMES.get(header, header)
        parva_idx = (i // 2) + 1
        ch_headers = CHAPTER_RE.findall(body)
        ch_splits = CHAPTER_RE.split(body)
        if not ch_headers:
            lines = body.strip().split("\n")
            title = ""
            text_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if text_lines: text_lines.append("")
                    continue
                if THUS_ENDS_RE.match(stripped): continue
                if not title and not text_lines:
                    title = stripped
                else:
                    text_lines.append(stripped)
            full = "\n".join(text_lines).strip()
            if full:
                for ci, ct in enumerate(chunk_text(full)):
                    all_chunks.append({
                        "id": f"chunk-{chunk_idx}",
                        "text": ct,
                        "parva": canonical,
                        "parva_index": parva_idx,
                        "chapter": 1,
                        "chapter_title": title,
                        "chunk_index": ci,
                    })
                    chunk_idx += 1
        else:
            for chi, ch_header in enumerate(ch_headers):
                ch_body = ch_splits[chi+1] if chi+1 < len(ch_splits) else ""
                ch_num = parse_chapter_number(ch_header) or chi + 1
                lines = ch_body.strip().split("\n")
                title = ""
                text_lines = []
                in_commentary = False
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        if text_lines: text_lines.append("")
                        continue
                    if COMMENTARY_RE.match(stripped):
                        in_commentary = True
                        continue
                    if THUS_ENDS_RE.match(stripped): continue
                    if in_commentary:
                        text_lines.append(stripped)
                    elif not title and not text_lines:
                        title = stripped
                    else:
                        text_lines.append(stripped)
                full = "\n".join(text_lines).strip()
                if full:
                    for ci, ct in enumerate(chunk_text(full)):
                        all_chunks.append({
                            "id": f"chunk-{chunk_idx}",
                            "text": ct,
                            "parva": canonical,
                            "parva_index": parva_idx,
                            "chapter": ch_num,
                            "chapter_title": title,
                            "chunk_index": ci,
                        })
                        chunk_idx += 1
    return all_chunks

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 4: BUILD VECTOR STORE")
    print("=" * 70)

    # Step 1: Parse and chunk.
    print("\n  Parsing and chunking text...")
    chunks = parse_and_chunk()
    print(f"  {len(chunks)} chunks ready")

    # Step 2: Load embedding model.
    print(f"\n  Loading embedding model ({MODEL_NAME})...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Step 3: Generate embeddings.
    print(f"\n  Generating embeddings (batch_size={BATCH_SIZE})...")
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    elapsed = time.time() - t0
    print(f"  {len(embeddings)} embeddings in {elapsed:.1f}s ({len(embeddings)/elapsed:.0f} chunks/sec)")

    # Step 4: Store in ChromaDB.
    print(f"\n  Storing in ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if present.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches.
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_embeddings = embeddings[i:i+BATCH_SIZE].tolist()
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{
                "parva": c["parva"],
                "parva_index": c["parva_index"],
                "chapter": c["chapter"],
                "chapter_title": c["chapter_title"],
                "chunk_index": c["chunk_index"],
            } for c in batch],
        )

    print(f"  Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'")

    # Step 5: Test queries.
    print("\n" + "=" * 70)
    print("TEST QUERIES")
    print("=" * 70)

    test_queries = [
        "Who killed Bhishma?",
        "How did Arjuna get the Gandiva bow?",
        "What happened at the dice game?",
        "How did Abhimanyu die in the Chakravyuha?",
        "Why did Krishna show the Vishwarupa?",
        "What was Karna's real identity?",
        "How did the Kurukshetra war end?",
    ]

    for query in test_queries:
        results = collection.query(
            query_embeddings=model.encode([query]).tolist(),
            n_results=3,
        )
        print(f"\n  Q: {query}")
        for j, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            score = 1 - dist  # cosine similarity
            preview = doc[:120].replace("\n", " ")
            print(f"    [{j+1}] ({score:.3f}) {meta['parva']}, Ch.{meta['chapter']}")
            print(f"        {preview}...")

    # Summary.
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"  Chunks embedded:  {collection.count()}")
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Store:            {CHROMA_DIR}")
    print(f"  Collection:       {COLLECTION_NAME}")
    print()

if __name__ == "__main__":
    main()
