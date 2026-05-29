"""Parse the combined Mahabharata text into structured Parva/Chapter objects.

Tuned for the KM Ganguli summary study format where:
- Parva headers appear on standalone lines: "Adi Parva", "Sabha Parva", etc.
- Chapter headers use English number words: "Chapter One", "Chapter Two"
- Commentary sections follow each chapter
- "Thus Ends..." markers separate chapters
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.ingestion.models import Chapter, Parva

logger = logging.getLogger(__name__)

# Canonical parva names mapped to the variants found in the text file.
PARVA_NAMES: dict[str, str] = {
    "Adi Parva": "Adi Parva",
    "Sabha Parva": "Sabha Parva",
    "Vana Parva": "Vana Parva",
    "Virata Parva": "Virata Parva",
    "Udyoga Parva": "Udyoga Parva",
    "Bhisma Parva": "Bhishma Parva",
    "Drona Parva": "Drona Parva",
    "Karna Parva": "Karna Parva",
    "Salya Parva": "Shalya Parva",
    "Sauptika Parva": "Sauptika Parva",
    "Stree Parva": "Stri Parva",
    "Shanti Parva": "Shanti Parva",
    "Anushasana Parva": "Anushasana Parva",
    "Ashvamedha Parva": "Ashvamedhika Parva",
    "Ashramvasika Parva": "Ashramavasika Parva",
    "Mausala Parva": "Mausala Parva",
    "Mahaprasthanika Parva": "Mahaprasthanika Parva",
}

# English number words used in chapter headers.
_NUMBER_WORDS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|twenty[\s-]?one|twenty[\s-]?two|twenty[\s-]?three|"
    r"twenty[\s-]?four|twenty[\s-]?five|twenty[\s-]?six|twenty[\s-]?seven|"
    r"twenty[\s-]?eight|twenty[\s-]?nine|thirty|thirty[\s-]?one|"
    r"thirty[\s-]?two|thirty[\s-]?three|thirty[\s-]?four|thirty[\s-]?five|"
    r"\d+"
)

_CHAPTER_HEADER = re.compile(
    rf"^Chapter\s+(?:{_NUMBER_WORDS})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_COMMENTARY_HEADER = re.compile(
    r"^Chapter\s+Commentary\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_THUS_ENDS = re.compile(
    r"^Thus Ends.*$",
    re.IGNORECASE | re.MULTILINE,
)

_WORD_TO_INT: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
    "twenty-three": 23, "twenty-four": 24, "twenty-five": 25,
    "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28,
    "twenty-nine": 29, "thirty": 30, "thirty-one": 31,
    "thirty-two": 32, "thirty-three": 33, "thirty-four": 34,
    "thirty-five": 35,
}


def _clean_text(raw: str) -> str:
    """Normalize line endings and whitespace."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_chapter_number(header: str) -> int:
    """Extract chapter number from a header like 'Chapter Four'."""
    match = re.search(r"Chapter\s+(.+)", header, re.IGNORECASE)
    if not match:
        return 0
    word = match.group(1).strip().lower().replace(" ", "-")
    if word.isdigit():
        return int(word)
    return _WORD_TO_INT.get(word, 0)


def _split_chapter_body(raw_body: str) -> tuple[str, str, str]:
    """Split a chapter's raw text into (title, narrative, commentary).

    The first non-empty line after the chapter header is the title.
    Everything between title and 'Chapter Commentary' or 'Thus Ends' is narrative.
    Everything after 'Chapter Commentary' is commentary.
    """
    lines = raw_body.strip().split("\n")
    title = ""
    narrative_lines: list[str] = []
    commentary_lines: list[str] = []
    in_commentary = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_commentary:
                commentary_lines.append("")
            else:
                narrative_lines.append("")
            continue

        if _COMMENTARY_HEADER.match(stripped):
            in_commentary = True
            continue

        if _THUS_ENDS.match(stripped):
            continue

        if in_commentary:
            commentary_lines.append(stripped)
        elif not title and not narrative_lines:
            title = stripped
        else:
            narrative_lines.append(stripped)

    narrative = "\n".join(narrative_lines).strip()
    commentary = "\n".join(commentary_lines).strip()
    return title, narrative, commentary


def parse_combined_file(path: Path) -> list[Parva]:
    """Parse the combined Mahabharata text file.

    Returns a list of Parva objects, one per book, each containing
    parsed chapters with titles, narrative text, and commentary separated.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = _clean_text(raw)

    # Build parva header pattern from the actual names in the file.
    parva_names_in_file = list(PARVA_NAMES.keys())
    parva_pattern = re.compile(
        r"^(" + "|".join(re.escape(name) for name in parva_names_in_file) + r")\s*$",
        re.MULTILINE,
    )

    # Split on parva headers.
    parva_splits = parva_pattern.split(cleaned)
    # parva_splits alternates: [preamble, header1, body1, header2, body2, ...]

    parvas: list[Parva] = []
    parva_index = 0

    for i in range(1, len(parva_splits), 2):
        header = parva_splits[i].strip()
        body = parva_splits[i + 1] if i + 1 < len(parva_splits) else ""
        parva_index += 1

        canonical_name = PARVA_NAMES.get(header, header)

        # Split body into chapters.
        chapter_splits = _CHAPTER_HEADER.split(body)
        chapter_headers = _CHAPTER_HEADER.findall(body)

        chapters: list[Chapter] = []
        for ci, ch_header in enumerate(chapter_headers):
            ch_body = chapter_splits[ci + 1] if ci + 1 < len(chapter_splits) else ""
            ch_number = _parse_chapter_number(ch_header)
            title, narrative, commentary = _split_chapter_body(ch_body)

            # Store narrative as the main text; commentary as a separate field
            # could be added later. For now, include both for completeness.
            full_text = narrative
            if commentary:
                full_text += "\n\n[Commentary]\n" + commentary

            if full_text.strip():
                chapters.append(
                    Chapter(
                        parva_index=parva_index,
                        chapter_index=ch_number if ch_number else ci + 1,
                        title=title,
                        text=full_text,
                    )
                )

        parvas.append(Parva(index=parva_index, name=canonical_name, chapters=chapters))
        logger.info(
            "Parsed %s (%s): %d chapters",
            canonical_name, header, len(chapters),
        )

    logger.info(
        "Total: %d parvas, %d chapters",
        len(parvas),
        sum(len(p.chapters) for p in parvas),
    )
    return parvas