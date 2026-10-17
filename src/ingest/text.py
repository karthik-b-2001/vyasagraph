"""Text sources -> chunks -> vector index.

A source is a folder in data/sources/ holding a source.yaml and its text
file(s). Chunk IDs are a hash of the embedding model, metadata and text, so
re-running is incremental: unchanged chunks keep their stored vectors, new
ones are embedded, and chunks that no longer exist (including those of
deleted sources) are dropped.

The index is two files in config.INDEX_DIR: embeddings.npz (float32,
L2-normalized vectors plus their chunk IDs) and chunks.jsonl (id, text,
metadata, same order). Loading checks the IDs match row for row. Search is exact: at ~19k chunks one matrix product takes a few ms,
which is simpler and more accurate than an approximate vector database.
"""

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import yaml

from src import config
from src.ingest.chunker import chunk_text
from src.ingest.parsers import PARSERS

KINDS = {"translation", "summary", "commentary"}
EMBEDDINGS_FILE = "embeddings.npz"
CHUNKS_FILE = "chunks.jsonl"


@dataclass
class Source:
    id: str
    title: str
    kind: str
    parser: str
    paths: list  # read in order and concatenated

    def read(self):
        return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in self.paths)


def load_sources():
    sources = []
    for meta_path in sorted(config.SOURCES_DIR.glob("*/source.yaml")):
        meta = yaml.safe_load(meta_path.read_text())
        missing = {"id", "title", "kind", "parser", "file"} - meta.keys()
        if missing:
            raise ValueError(f"{meta_path}: missing {sorted(missing)}")
        if meta["parser"] not in PARSERS:
            raise ValueError(f"{meta_path}: unknown parser '{meta['parser']}', expected one of {sorted(PARSERS)}")
        if meta["kind"] not in KINDS:
            raise ValueError(f"{meta_path}: kind must be one of {sorted(KINDS)}")
        files = meta["file"] if isinstance(meta["file"], list) else [meta["file"]]
        paths = [meta_path.parent / f for f in files]
        for p in paths:
            if not p.exists():
                raise ValueError(f"{meta_path}: file not found: {p.name}")
        sources.append(Source(meta["id"], meta["title"], meta["kind"], meta["parser"], paths))
    ids = [s.id for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate source ids: {ids}")
    return sources


def build_chunks(source):
    """Return {chunk_id: (text, metadata)} for one source."""
    chunks = {}
    for sec in PARSERS[source.parser](source.read()):
        for i, text in enumerate(chunk_text(sec.text)):
            meta = {
                "source": source.id,
                "source_title": source.title,
                "source_kind": source.kind,
                "parva": sec.parva,
                "parva_index": sec.parva_index,
                "chapter": sec.chapter,
                "chapter_title": sec.chapter_title,
                "unit": sec.unit,
                "section": sec.kind,
                "chunk_index": i,
            }
            # The ID covers model, metadata and text, so a change to any of
            # them replaces the chunk instead of leaving a stale copy.
            key = json.dumps([config.EMBEDDING_MODEL, meta, text], sort_keys=True)
            chunk_id = f"{source.id}-{hashlib.sha1(key.encode()).hexdigest()[:16]}"
            chunks[chunk_id] = (text, meta)
    return chunks


def _source_of(chunk_id):
    return chunk_id.rsplit("-", 1)[0]  # IDs are "<source id>-<hex hash>"


# ── Index files ───────────────────────────────────────────────────────────

def load_index(index_dir=None):
    """Return (ids, texts, metas, embeddings); empty if no index exists."""
    index_dir = index_dir or config.INDEX_DIR
    if not ((index_dir / CHUNKS_FILE).exists() and (index_dir / EMBEDDINGS_FILE).exists()):
        return [], [], [], np.zeros((0, 0), dtype=np.float32)
    ids, texts, metas = [], [], []
    with open(index_dir / CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["id"])
            texts.append(row["text"])
            metas.append(row["meta"])
    with np.load(index_dir / EMBEDDINGS_FILE) as data:
        embeddings, vector_ids = data["embeddings"], data["ids"].tolist()
    if vector_ids != ids:
        raise ValueError("index is inconsistent: vectors and chunks.jsonl do not match; "
                         "rebuild with `python -m src.ingest text --rebuild`")
    return ids, texts, metas, embeddings


def _save_index(ids, texts, metas, embeddings):
    """Write each file under a temporary name, then rename it into place, so
    neither is ever half-written. If a crash falls between the two renames,
    the ID check in load_index() reports it instead of mismatching vectors."""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks_tmp = config.INDEX_DIR / (CHUNKS_FILE + ".tmp")
    emb_tmp = config.INDEX_DIR / "embeddings.tmp.npz"
    with open(chunks_tmp, "w", encoding="utf-8") as f:
        for i, t, m in zip(ids, texts, metas):
            f.write(json.dumps({"id": i, "text": t, "meta": m}, ensure_ascii=False) + "\n")
    np.savez(emb_tmp, embeddings=embeddings.astype(np.float32), ids=np.array(ids, dtype=str))
    os.replace(emb_tmp, config.INDEX_DIR / EMBEDDINGS_FILE)
    os.replace(chunks_tmp, config.INDEX_DIR / CHUNKS_FILE)


def ingest_text(rebuild=False):
    from sentence_transformers import SentenceTransformer

    sources = load_sources()
    old_ids, _, _, old_emb = ([], [], [], None) if rebuild else load_index()
    old_row = {cid: n for n, cid in enumerate(old_ids)}
    embedder = None

    ids, texts, metas, vectors = [], [], [], []
    for source in sources:
        chunks = build_chunks(source)
        new = [cid for cid in chunks if cid not in old_row]
        if new:
            embedder = embedder or SentenceTransformer(config.EMBEDDING_MODEL)
            fresh = embedder.encode([chunks[c][0] for c in new], batch_size=64,
                                    normalize_embeddings=True, show_progress_bar=False)
            fresh = dict(zip(new, fresh))
        for cid, (text, meta) in chunks.items():
            ids.append(cid)
            texts.append(text)
            metas.append(meta)
            vectors.append(fresh[cid] if cid not in old_row else old_emb[old_row[cid]])
        kept_before = sum(1 for cid in old_ids if _source_of(cid) == source.id)
        removed = kept_before - (len(chunks) - len(new))
        print(f"  {source.id}: {len(chunks)} chunks ({len(new)} embedded, "
              f"{len(chunks) - len(new)} unchanged, {removed} removed)")

    gone = {_source_of(cid) for cid in old_ids} - {s.id for s in sources}
    for source_id in sorted(gone):
        print(f"  {source_id}: source deleted, its chunks removed")

    embeddings = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 0), np.float32)
    _save_index(ids, texts, metas, embeddings)
    size = sum(f.stat().st_size for f in config.INDEX_DIR.iterdir()) / 1e6
    print(f"  index total: {len(ids)} chunks, {size:.0f} MB in {config.INDEX_DIR}")


def print_stats():
    """Report chunk counts and how many chunks exceed the embedder's input limit."""
    from statistics import median

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    limit = embedder.max_seq_length
    for source in load_sources():
        chunks = build_chunks(source)
        texts = [t for t, _ in chunks.values()]
        lengths = [len(embedder.tokenizer(t)["input_ids"]) for t in texts]
        over = sum(n > limit for n in lengths)
        sections = {}
        for _, m in chunks.values():
            sections[m["section"]] = sections.get(m["section"], 0) + 1
        print(f"  {source.id}: {len(texts)} chunks {sections}")
        print(f"    word-pieces: median {median(lengths):.0f}, max {max(lengths)}; "
              f"{over} chunks ({over / len(texts):.1%}) exceed the {limit}-token limit")
