"""Domain models for the text ingestion pipeline.

These Pydantic models define the canonical shapes for parsed and chunked
Mahabharata text. Every downstream consumer (NER, embedding, graph loader)
works with these types.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Parva(BaseModel):
    """One of the 18 books of the Mahabharata."""

    index: int = Field(ge=1, le=18, description="1-indexed parva number")
    name: str = Field(description="English name, e.g. 'Adi Parva'")
    chapters: list[Chapter] = Field(default_factory=list)

    @property
    def total_text(self) -> str:
        return "\n\n".join(ch.text for ch in self.chapters)


class Chapter(BaseModel):
    """A chapter (adhyaya) within a parva."""

    parva_index: int
    chapter_index: int
    title: str = ""
    text: str = Field(description="Full chapter text, cleaned")


class TextChunk(BaseModel):
    """A chunk of text ready for embedding and retrieval.

    Chunks are the atomic unit for the vector store and RAG context.
    """

    chunk_id: str = Field(description="Deterministic ID: parva-chapter-chunk_index")
    text: str
    parva_index: int
    parva_name: str
    chapter_index: int
    chunk_index: int = Field(description="0-indexed position within chapter")
    char_start: int = Field(description="Character offset in chapter text")
    char_end: int
    token_count: int = Field(ge=0, description="Approximate token count")

    @property
    def metadata(self) -> dict:
        """Flat metadata dict for ChromaDB / search filters."""
        return {
            "parva_index": self.parva_index,
            "parva_name": self.parva_name,
            "chapter_index": self.chapter_index,
            "chunk_index": self.chunk_index,
        }