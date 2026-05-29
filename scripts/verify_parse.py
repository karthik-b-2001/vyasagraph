"""Standalone Phase 1 test — no external dependencies.

Usage:
    python scripts/verify_parse.py
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_PATH = DATA_DIR / "1-18_books_combined.txt"
CSV_PATH = DATA_DIR / "test_relations.csv"

PARVA_NAMES = {
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

_NUMBER_WORDS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|twenty[\s-]?one|twenty[\s-]?two|twenty[\s-]?three|"
    r"twenty[\s-]?four|twenty[\s-]?five|twenty[\s-]?six|twenty[\s-]?seven|"
    r"twenty[\s-]?eight|twenty[\s-]?nine|thirty|thirty[\s-]?one|"
    r"thirty[\s-]?two|thirty[\s-]?three|thirty[\s-]?four|thirty[\s-]?five|"
    r"\d+"
)

CHAPTER_RE = re.compile(
    rf"^Chapter\s+(?:{_NUMBER_WORDS})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

COMMENTARY_RE = re.compile(r"^Chapter\s+Commentary\s*$", re.IGNORECASE | re.MULTILINE)
THUS_ENDS_RE = re.compile(r"^Thus Ends.*$", re.IGNORECASE | re.MULTILINE)

WORD_TO_INT = {
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


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_chapter_number(header: str) -> int:
    match = re.search(r"Chapter\s+(.+)", header, re.IGNORECASE)
    if not match:
        return 0
    word = match.group(1).strip().lower().replace(" ", "-")
    if word.isdigit():
        return int(word)
    return WORD_TO_INT.get(word, 0)


def split_chapter_body(raw_body: str) -> tuple[str, str, str]:
    """Returns (title, narrative, commentary)."""
    lines = raw_body.strip().split("\n")
    title = ""
    narrative: list[str] = []
    commentary: list[str] = []
    in_commentary = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            (commentary if in_commentary else narrative).append("")
            continue
        if COMMENTARY_RE.match(stripped):
            in_commentary = True
            continue
        if THUS_ENDS_RE.match(stripped):
            continue
        if in_commentary:
            commentary.append(stripped)
        elif not title and not narrative:
            title = stripped
        else:
            narrative.append(stripped)

    return title, "\n".join(narrative).strip(), "\n".join(commentary).strip()


def approx_tokens(text: str) -> int:
    return len(text) // 4


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = approx_tokens(sent)
        if current_tokens + sent_tokens > chunk_size and current:
            chunks.append({
                "text": " ".join(current),
                "tokens": current_tokens,
            })
            overlap_sents: list[str] = []
            overlap_tok = 0
            for s in reversed(current):
                t = approx_tokens(s)
                if overlap_tok + t > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += t
            current = overlap_sents
            current_tokens = overlap_tok

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append({"text": " ".join(current), "tokens": current_tokens})

    return chunks


def main() -> None:
    assert TEXT_PATH.exists(), f"Text file not found: {TEXT_PATH}"

    raw = TEXT_PATH.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)

    parva_names = list(PARVA_NAMES.keys())
    parva_pattern = re.compile(
        r"^(" + "|".join(re.escape(n) for n in parva_names) + r")\s*$",
        re.MULTILINE,
    )
    parts = parva_pattern.split(cleaned)

    print("=" * 70)
    print("PHASE 1: TEXT PARSING")
    print("=" * 70)

    parvas = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        canonical = PARVA_NAMES.get(header, header)

        ch_splits = CHAPTER_RE.split(body)
        ch_headers = CHAPTER_RE.findall(body)

        chapters = []

        if not ch_headers:
            # No "Chapter One/Two" markers — treat entire parva as one chapter.
            lines = body.strip().split("\n")
            title = ""
            text_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if text_lines:
                        text_lines.append("")
                    continue
                if THUS_ENDS_RE.match(stripped):
                    continue
                if not title and not text_lines:
                    title = stripped
                else:
                    text_lines.append(stripped)
            full_text = "\n".join(text_lines).strip()
            if full_text:
                chapters.append({
                    "index": 1,
                    "title": title,
                    "text": full_text,
                    "narrative_len": len(full_text),
                    "commentary_len": 0,
                })
        else:
            # Normal parva with chapter headers.
            for ci, ch_header in enumerate(ch_headers):
                ch_body = ch_splits[ci + 1] if ci + 1 < len(ch_splits) else ""
                ch_num = parse_chapter_number(ch_header)
                title, narrative, commentary = split_chapter_body(ch_body)
                full_text = narrative
                if commentary:
                    full_text += "\n\n[Commentary]\n" + commentary
                if full_text.strip():
                    chapters.append({
                        "index": ch_num or ci + 1,
                        "title": title,
                        "text": full_text,
                        "narrative_len": len(narrative),
                        "commentary_len": len(commentary),
                    })

        parvas.append({
            "index": len(parvas) + 1,
            "name": canonical,
            "file_name": header,
            "chapters": chapters,
        })

    # --- Print results ---
    total_ch = 0
    total_chars = 0
    print(f"\nFound {len(parvas)} parvas:\n")
    print(f"  {'#':>3}  {'Canonical Name':<30}  {'In File':<25}  {'Ch':>4}  {'Chars':>10}")
    print(f"  {'─'*3}  {'─'*30}  {'─'*25}  {'─'*4}  {'─'*10}")
    for p in parvas:
        ch_count = len(p["chapters"])
        char_count = sum(ch["narrative_len"] + ch["commentary_len"] for ch in p["chapters"])
        total_ch += ch_count
        total_chars += char_count
        print(f"  {p['index']:3d}  {p['name']:<30}  {p['file_name']:<25}  {ch_count:4d}  {char_count:>10,}")

    print(f"\n  Total: {total_ch} chapters, {total_chars:,} characters\n")

    # --- Sample chapters ---
    print("=" * 70)
    print("SAMPLE CHAPTERS (Adi Parva, first 3)")
    print("=" * 70)
    if parvas:
        for ch in parvas[0]["chapters"][:3]:
            print(f"\n  Chapter {ch['index']}: {ch['title']}")
            print(f"  Narrative: {ch['narrative_len']:,} chars | Commentary: {ch['commentary_len']:,} chars")
            preview = ch["text"][:150].replace("\n", " ")
            print(f"  Preview: {preview}...")

    # --- Chunking ---
    print("\n" + "=" * 70)
    print("CHUNKING")
    print("=" * 70)

    all_chunks = []
    for p in parvas:
        for ch in p["chapters"]:
            ch_chunks = chunk_text(ch["text"])
            for ci, chunk in enumerate(ch_chunks):
                chunk["parva"] = p["name"]
                chunk["chapter"] = ch["index"]
                chunk["chunk_id"] = f"{p['index']}-{ch['index']}-{ci}"
            all_chunks.extend(ch_chunks)

    print(f"\n  Total chunks: {len(all_chunks):,}")
    if all_chunks:
        tokens = [c["tokens"] for c in all_chunks]
        print(f"  Avg tokens/chunk: {sum(tokens)/len(tokens):.0f}")
        print(f"  Min: {min(tokens)}, Max: {max(tokens)}")

        print(f"\n  Chunks per parva:")
        parva_counts = Counter(c["parva"] for c in all_chunks)
        for name, count in parva_counts.most_common():
            print(f"    {name:<30}  {count:4d}")

        print(f"\n  Sample chunk ({all_chunks[0]['chunk_id']}):")
        print(f"    {all_chunks[0]['text'][:200]}...")

    # --- Relations CSV ---
    if CSV_PATH.exists():
        print("\n" + "=" * 70)
        print("RELATIONS CSV")
        print("=" * 70)

        entities: set[str] = set()
        relationships: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        with open(CSV_PATH, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k: v.strip() if v else "" for k, v in row.items()}

                for col in ["Son", "Father", "Son2", "Mother", "Husband", "Wife",
                            "Brothers1_1", "Brothers1_2", "Brothers2_1", "Brothers2_2"]:
                    name = row.get(col, "")
                    if name:
                        entities.add(name)

                pairs = [
                    ("Son", "Father", "SON_OF"),
                    ("Son2", "Mother", "SON_OF"),
                    ("Husband", "Wife", "MARRIED_TO"),
                    ("Brothers1_1", "Brothers1_2", "SIBLING_OF"),
                    ("Brothers2_1", "Brothers2_2", "SIBLING_OF"),
                ]
                for src_col, tgt_col, rel in pairs:
                    src = row.get(src_col, "")
                    tgt = row.get(tgt_col, "")
                    if src and tgt:
                        key = (src, tgt, rel)
                        rev = (tgt, src, rel)
                        if key not in seen and rev not in seen:
                            seen.add(key)
                            relationships.append((src, tgt, rel))

        print(f"\n  Entities: {len(entities)}")
        print(f"  Relationships: {len(relationships)}")

        rel_counts = Counter(r[2] for r in relationships)
        print(f"\n  By type:")
        for rel_type, count in rel_counts.most_common():
            print(f"    {rel_type:<20}  {count:3d}")

        print(f"\n  Sample relationships:")
        for src, tgt, rel in relationships[:12]:
            print(f"    {src} --[{rel}]--> {tgt}")

        print(f"\n  All entities ({len(entities)}):")
        for name in sorted(entities):
            print(f"    {name}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PHASE 1 VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"  Parvas:        {len(parvas)}")
    print(f"  Chapters:      {total_ch}")
    print(f"  Characters:    {total_chars:,}")
    print(f"  Chunks:        {len(all_chunks):,}")
    if CSV_PATH.exists():
        print(f"  CSV entities:  {len(entities)}")
        print(f"  CSV relations: {len(relationships)}")
    print()


if __name__ == "__main__":
    main()
