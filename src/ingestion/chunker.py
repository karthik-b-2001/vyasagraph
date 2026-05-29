"""Token-aware text chunker with configurable overlap.

Produces TextChunk objects ready for embedding and storage. Uses tiktoken
for accurate token counting aligned with OpenAI embedding models.
"""

from __future__ import annotations

import logging

import tiktoken

from src.config import settings
from src.ingestion.models import Parva, TextChunk

logger = logging.getLogger(__name__)

_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def chunk_chapter(
    text: str,
    parva_index: int,
    parva_name: str,
    chapter_index: int,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[TextChunk]:
    """Split a single chapter's text into overlapping token-bounded chunks.

    Strategy: sentence-aware splitting. We accumulate sentences until we
    hit the token budget, then rewind by `chunk_overlap` tokens for the
    next chunk. This avoids cutting mid-sentence.
    """
    sentences = _split_sentences(text)
    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    char_offset = 0
    chunk_idx = 0

    for sentence in sentences:
        sent_tokens = _count_tokens(sentence)

        if current_tokens + sent_tokens > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            char_end = char_offset + len(chunk_text)

            chunks.append(
                TextChunk(
                    chunk_id=f"{parva_index}-{chapter_index}-{chunk_idx}",
                    text=chunk_text,
                    parva_index=parva_index,
                    parva_name=parva_name,
                    chapter_index=chapter_index,
                    chunk_index=chunk_idx,
                    char_start=char_offset,
                    char_end=char_end,
                    token_count=current_tokens,
                )
            )
            chunk_idx += 1

            # Rewind: keep enough trailing sentences to cover overlap tokens.
            overlap_sentences, overlap_tokens = _compute_overlap(
                current_sentences, chunk_overlap
            )
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens
            char_offset = char_end - sum(len(s) + 1 for s in overlap_sentences)

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Flush remaining.
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(
            TextChunk(
                chunk_id=f"{parva_index}-{chapter_index}-{chunk_idx}",
                text=chunk_text,
                parva_index=parva_index,
                parva_name=parva_name,
                chapter_index=chapter_index,
                chunk_index=chunk_idx,
                char_start=char_offset,
                char_end=char_offset + len(chunk_text),
                token_count=current_tokens,
            )
        )

    return chunks


def chunk_parvas(parvas: list[Parva]) -> list[TextChunk]:
    """Chunk all parvas and return a flat list of TextChunks."""
    all_chunks: list[TextChunk] = []
    for parva in parvas:
        for chapter in parva.chapters:
            chapter_chunks = chunk_chapter(
                text=chapter.text,
                parva_index=parva.index,
                parva_name=parva.name,
                chapter_index=chapter.chapter_index,
            )
            all_chunks.extend(chapter_chunks)

    logger.info("Generated %d chunks across %d parvas", len(all_chunks), len(parvas))
    return all_chunks


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter. Good enough for translated prose."""
    import re

    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def _compute_overlap(
    sentences: list[str], target_tokens: int
) -> tuple[list[str], int]:
    """Return trailing sentences that fit within the overlap token budget."""
    overlap: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        tokens = _count_tokens(sentence)
        if total + tokens > target_tokens:
            break
        overlap.insert(0, sentence)
        total += tokens
    return overlap, total