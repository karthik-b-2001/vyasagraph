"""Split section text into overlapping, sentence-aligned chunks.

Sizes are measured in the embedding model's own word-pieces. An embedding
model silently ignores input past its limit (256 for all-MiniLM-L6-v2, 512
for bge-small-en-v1.5), so every chunk must fit or part of it is invisible
to retrieval; `python -m src.ingest stats` checks this. Word-count
heuristics are not enough: lists of Sanskrit names can cost 4+ word-pieces
per word. 250 also keeps passages short enough to cite precisely.

The tokenizer splits on whitespace before word-pieces, so the length of a
text is exactly the sum of its words' lengths; each distinct word is
tokenized once and cached.
"""

import re
from functools import lru_cache

from transformers import AutoTokenizer

from src import config

CHUNK_TOKENS = 250
OVERLAP_TOKENS = 40

_SENTENCE_RE = re.compile(r"(?<=[.!?\"'”’])\s+")


@lru_cache(maxsize=1)
def _tokenizer():
    name = config.EMBEDDING_MODEL
    return AutoTokenizer.from_pretrained(name if "/" in name else f"sentence-transformers/{name}")


@lru_cache(maxsize=200_000)
def word_tokens(word):
    return len(_tokenizer().tokenize(word))


def text_tokens(text):
    return sum(word_tokens(w) for w in text.split())


def _sentences(text):
    """Sentences as word lists; any sentence over the budget is cut into pieces."""
    out = []
    for sent in _SENTENCE_RE.split(text):
        piece, size = [], 0
        for w in sent.split():
            n = word_tokens(w)
            if piece and size + n > CHUNK_TOKENS:
                out.append((piece, size))
                piece, size = [], 0
            piece.append(w)
            size += n
        if piece:
            out.append((piece, size))
    return out


def chunk_text(text):
    chunks = []
    current, size = [], 0  # list of (words, tokens) sentences
    for sent, n in _sentences(text):
        if current and size + n > CHUNK_TOKENS:
            chunks.append(" ".join(w for s, _ in current for w in s))
            # Carry whole trailing sentences into the next chunk, up to OVERLAP_TOKENS.
            carry, carry_size = [], 0
            for prev, pn in reversed(current):
                if carry_size + pn > OVERLAP_TOKENS or carry_size + pn + n > CHUNK_TOKENS:
                    break
                carry.insert(0, (prev, pn))
                carry_size += pn
            current, size = carry, carry_size
        current.append((sent, n))
        size += n
    if current:
        chunks.append(" ".join(w for s, _ in current for w in s))
    return chunks
