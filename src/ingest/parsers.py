"""Turn a source text file into Sections.

A Section is one contiguous run of text with a single citation:
(parva, chapter, kind). `kind` is "narrative" for the story itself and
"commentary" for a translator's or author's interpretation, so the two
are never confused at answer time.

Each source declares its parser in source.yaml. Adding a source in a new
layout means adding one function here and registering it in PARSERS.
"""

import re
from dataclasses import dataclass

# The 18 parvas in canonical order, with spellings seen in sources.
PARVAS = [
    ("Adi Parva", []),
    ("Sabha Parva", []),
    ("Vana Parva", ["Aranyaka Parva"]),
    ("Virata Parva", []),
    ("Udyoga Parva", []),
    ("Bhishma Parva", ["Bhisma Parva"]),
    ("Drona Parva", []),
    ("Karna Parva", []),
    ("Shalya Parva", ["Salya Parva"]),
    ("Sauptika Parva", []),
    ("Stri Parva", ["Stree Parva"]),
    ("Shanti Parva", ["Santi Parva"]),
    ("Anushasana Parva", ["Anusasana Parva"]),
    ("Ashvamedhika Parva", ["Ashvamedha Parva", "Aswamedha Parva"]),
    ("Ashramavasika Parva", ["Ashramvasika Parva", "Asramavasika Parva"]),
    ("Mausala Parva", []),
    ("Mahaprasthanika Parva", []),
    ("Svargarohana Parva", ["Svargarohanika Parva"]),
]

_PARVA_LOOKUP = {}
for _i, (_name, _variants) in enumerate(PARVAS, start=1):
    for _spelling in [_name, *_variants]:
        _PARVA_LOOKUP[_spelling.lower()] = (_i, _name)


def canonical_parva(name):
    """Return (index, canonical name) for a parva heading, or None."""
    return _PARVA_LOOKUP.get(name.strip().lower())


@dataclass
class Section:
    parva_index: int  # 1-18, or 0 when the text is not tied to a parva
    parva: str
    chapter: int
    chapter_title: str
    kind: str  # "narrative" | "commentary"
    text: str
    unit: str = "Chapter"  # what `chapter` counts, for citations


# ── Number words ("Chapter Twenty-Three") ──────────────────────────────────

_UNITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def parse_number(text):
    """'Twenty-Three' -> 23, '7' -> 7, anything else -> None."""
    t = text.strip().lower().replace(" ", "-")
    if t.isdigit():
        return int(t)
    if t in _UNITS:
        return _UNITS.index(t) + 1
    if t in _TEENS:
        return _TEENS.index(t) + 10
    tens, _, unit = t.partition("-")
    if tens in _TENS and (not unit or unit in _UNITS):
        return (_TENS.index(tens) + 2) * 10 + (_UNITS.index(unit) + 1 if unit else 0)
    return None


# ── Parser: "Summary Study of the Mahabharata" layout ──────────────────────

# "Ch\w*ter" also accepts the source's one typo, "Chpater Six".
_CHAPTER_RE = re.compile(r"^Ch\w*ter\s+(.+?)\s*$", re.IGNORECASE)
_COMMENTARY_RE = re.compile(r"^Chapter\s+Commentary\s*$", re.IGNORECASE)
_TAG_RE = re.compile(r"<[A-Z]>")  # stray markup, e.g. "<T>"
_THUS_ENDS_RE = re.compile(r"^Thus Ends\b", re.IGNORECASE)
_BOOK_END_RE = re.compile(r"^Thus Ends the Summary Study of the Mahabharata", re.IGNORECASE)


def parse_summary_study(raw):
    """Layout: a bare parva-name line, then 'Chapter <N>' + a title line,
    narrative paragraphs, a 'Thus Ends ...' colophon, and an optional
    'Chapter Commentary' block. Some short parvas have no chapter headers;
    their first line is the title of an implicit Chapter 1.
    Everything after the book's final colophon (end matter) is dropped.

    Nested chapters: Bhishma Parva Chapter One contains the Bhagavad-gita,
    whose 18 chapters reuse "Chapter One..Eighteen" headers. Rule: a
    "Chapter One" that does not move the parva forward starts a nested
    run (1, 2, 3, ...) that stays under the current chapter; the first
    header that breaks the run ends it. Any other number that does not move
    forward is a numbering typo (Drona Parva has "Chapter Ten" twice) and
    becomes the next chapter.
    """
    sections = []
    parva = None  # (index, name)
    chapter, title, kind = 1, "", "narrative"
    parent_title, nested = "", 0
    want_title, seen_chapter_header = False, False
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        if parva and text:
            full_title = f"{parent_title} {nested}: {title}" if nested else title
            sections.append(Section(parva[0], parva[1], chapter, full_title, kind, text))
        buf.clear()

    for line in raw.replace("\r\n", "\n").split("\n"):
        line = _TAG_RE.sub("", re.sub(r"[ \t]+", " ", line)).strip()
        if not line:
            continue
        if _BOOK_END_RE.match(line):
            break
        if canonical_parva(line):
            flush()
            parva = canonical_parva(line)
            chapter, title, kind, want_title = 1, "", "narrative", True
            nested, seen_chapter_header = 0, False
            continue
        if _COMMENTARY_RE.match(line):
            flush()
            kind = "commentary"
            continue
        m = _CHAPTER_RE.match(line)
        if m and parse_number(m.group(1)):
            flush()
            n = parse_number(m.group(1))
            if nested and n == nested + 1:
                nested = n
            elif seen_chapter_header and not nested and n == 1:
                parent_title, nested = title, n
            elif seen_chapter_header and not nested and n <= chapter:
                chapter = chapter + 1
            else:
                chapter, nested = n, 0
            seen_chapter_header = True
            title, kind, want_title = "", "narrative", True
            continue
        if _THUS_ENDS_RE.match(line):
            continue
        if want_title:
            title, want_title = line, False
            continue
        buf.append(line)
    flush()
    return sections


# ── Parser: Markdown (the recommended format for new material) ─────────────

def parse_markdown(raw):
    """Layout:
        # Adi Parva                 -> parva (canonical names recognised)
        ## The Birth of Bhishma     -> new chapter, numbered in order
        ### Commentary              -> following text is commentary
    Text before any '#' heading is filed under parva_index 0.
    """
    sections = []
    parva = (0, "General")
    chapter, title, kind = 0, "", "narrative"
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(parva[0], parva[1], max(chapter, 1), title, kind, text))
        buf.clear()

    for line in raw.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("### "):
            flush()
            kind = "commentary" if "commentary" in stripped.lower() else "narrative"
        elif stripped.startswith("## "):
            flush()
            chapter, title, kind = chapter + 1, stripped[3:].strip(), "narrative"
        elif stripped.startswith("# "):
            flush()
            heading = stripped[2:].strip()
            parva = canonical_parva(heading) or (0, heading)
            chapter, title, kind = 0, "", "narrative"
        else:
            buf.append(stripped)
    flush()
    return sections


# ── Parser: KM Ganguli translation, Project Gutenberg plain text ───────────

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def parse_roman(text):
    total = 0
    for ch, nxt in zip(text, text[1:] + " "):
        value = _ROMAN[ch]
        total += -value if _ROMAN.get(nxt, 0) > value else value
    return total


_GUTENBERG_START = re.compile(r"^\*\*\* START OF THE PROJECT GUTENBERG", re.M)
_GUTENBERG_END = re.compile(r"^\*\*\* END OF THE PROJECT GUTENBERG", re.M)
_BOOK_RE = re.compile(r"^BOOK (\d+)$")
_SECTION_RE = re.compile(r"^SECTION ([IVXLCDM]+)\b")  # one header is "SECTION XXXIV-XXXV"
# Books 8-11 and 16-18 (a different proofreader) number sections with a
# bare "1", "2", ... line instead.
_NUMBER_LINE_RE = re.compile(r"^(\d+)$")
_SUBPARVA_RE = re.compile(r"^\((.+? Parva)(?: [Cc]ontinued)?\)$")
_FOOTNOTE_MARK_RE = re.compile(r"\[\d+\]")
# "END OF ADI PARVA", "The end of Mausala-parva.", "The end of the Santi Parva [, Part two of three]."
_BOOK_CLOSE_RE = re.compile(r"^(the )?end of .*parv", re.IGNORECASE)
# The Gutenberg editor (J. B. Hare) labelled the Gita's chapters, e.g. "[(Bhagavad Gita, Chapter XIII)]".
_GITA_CHAPTER_RE = re.compile(r"^\[\(Bhagavad Gita,? Chapter ([IVXL]+)\)\]$")
# Other editor notes: "[...--JBH]", "([This where is the Bhagavad Gita proper starts ...])",
# "[End of the Bhagavad Gita]". Ganguli's own bracketed glosses are kept.
_EDITOR_NOTE_RE = re.compile(r"\(\[[^\]]*\]\)|\[[^\[\]]*JBH\)?\]|\[End of the Bhagavad Gita\]")
# Sub-parvas whose opening marker is missing from the Gutenberg text:
# (parva index, first section) -> name. Bhishma Parva's Gita ends at
# Section XLII ("[End of the Bhagavad Gita]"); the rest is Bhishma-vadha Parva.
_MISSING_SUBPARVAS = {(6, 43): "Bhishma-vadha Parva"}


def parse_ganguli(raw):
    """Layout, per book: 'BOOK N', front matter (title, preface), then
    'SECTION <roman>' headers, each optionally followed by a sub-parva line
    like '(Bhagavat-Gita Parva)'. A book ends with 'END OF ... PARVA' and a
    FOOTNOTES block. Some books use bare-number section lines instead; a
    number only counts as a header when it is the next section in sequence,
    so a stray number in the prose is never mistaken for one.
    Text outside sections (front matter, footnotes) and
    the Gutenberg licence is dropped; inline '[1]' markers are removed.
    Several volumes can be concatenated: each Gutenberg body is extracted.
    """
    bodies, pos = [], 0
    while (start := _GUTENBERG_START.search(raw, pos)):
        end = _GUTENBERG_END.search(raw, start.end())
        bodies.append(raw[start.end():end.start() if end else len(raw)])
        pos = end.end() if end else len(raw)
    text = "\n".join(bodies) if bodies else raw

    sections = []
    parva, section, subparva, gita_chapter = None, 0, "", 0
    in_section, roman_headers = False, False
    paragraphs, para = [], []

    def flush():
        if para:
            paragraphs.append(" ".join(para))
            para.clear()
        body = "\n".join(paragraphs)
        body = _EDITOR_NOTE_RE.sub("", _FOOTNOTE_MARK_RE.sub("", body)).strip()
        if parva and in_section and body:
            title = f"{subparva}, Gita Chapter {gita_chapter}" if gita_chapter else subparva
            sections.append(Section(parva[0], parva[1], section, title, "narrative", body, "Section"))
        paragraphs.clear()

    def start_section(n):
        nonlocal section, in_section, subparva, gita_chapter
        section, in_section, gita_chapter = n, True, 0
        subparva = _MISSING_SUBPARVAS.get((parva[0], n), subparva)

    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if m := _BOOK_RE.match(line):
            flush()
            index = int(m.group(1))
            parva, section, in_section, subparva = (index, PARVAS[index - 1][0]), 0, False, ""
            roman_headers = False
            continue
        if parva and (m := _SECTION_RE.match(line)):
            flush()
            roman_headers = True
            start_section(parse_roman(m.group(1)))
            continue
        if (parva and not roman_headers and (m := _NUMBER_LINE_RE.match(line))
                and int(m.group(1)) == section + 1):
            flush()
            start_section(section + 1)
            continue
        if in_section and (m := _GITA_CHAPTER_RE.match(line)) and not paragraphs and not para:
            gita_chapter = parse_roman(m.group(1))
            continue
        if line == "FOOTNOTES" or _BOOK_CLOSE_RE.match(line):
            flush()
            in_section = False
            continue
        if in_section and (m := _SUBPARVA_RE.match(line)) and not paragraphs and not para:
            subparva = m.group(1)
            continue
        if not line:
            if para:
                paragraphs.append(" ".join(para))
                para.clear()
            continue
        para.append(line)
    flush()
    return sections


PARSERS = {
    "summary_study": parse_summary_study,
    "markdown": parse_markdown,
    "ganguli": parse_ganguli,
}
