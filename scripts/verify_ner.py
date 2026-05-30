"""Phase 2 verification: NER & Relationship Extraction.

Usage:
    python scripts/verify_ner.py
"""

import re
import spacy
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_PATH = DATA_DIR / "1-18_books_combined.txt"

# ── Reuse parsing logic from Phase 1 ──────────────────────────────────────

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

def clean_text(raw):
    text = raw.replace("\r\n","\n").replace("\r","\n")
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

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
    raw = TEXT_PATH.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    parva_pattern = re.compile(
        r"^(" + "|".join(re.escape(n) for n in PARVA_NAMES) + r")\s*$",
        re.MULTILINE,
    )
    parts = parva_pattern.split(cleaned)
    all_chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        canonical = PARVA_NAMES.get(header, header)
        ch_headers = CHAPTER_RE.findall(body)
        ch_splits = CHAPTER_RE.split(body)
        texts = []
        if not ch_headers:
            lines = body.strip().split("\n")
            text_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if text_lines: text_lines.append("")
                    continue
                if THUS_ENDS_RE.match(stripped): continue
                text_lines.append(stripped)
            texts.append("\n".join(text_lines).strip())
        else:
            for ci, _ in enumerate(ch_headers):
                ch_body = ch_splits[ci+1] if ci+1 < len(ch_splits) else ""
                lines = ch_body.strip().split("\n")
                text_lines = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        if text_lines: text_lines.append("")
                        continue
                    if THUS_ENDS_RE.match(stripped): continue
                    if COMMENTARY_RE.match(stripped): continue
                    text_lines.append(stripped)
                texts.append("\n".join(text_lines).strip())
        for t in texts:
            if t:
                for chunk in chunk_text(t):
                    all_chunks.append({"text": chunk, "parva": canonical})
    return all_chunks

# ── Seed Entity Catalog ───────────────────────────────────────────────────

SEED_ENTITIES = {
    "CHARACTER": {
        "Krishna": ["Vasudeva","Govinda","Keshava","Madhava","Janardana","Hari","Hrishikesha"],
        "Arjuna": ["Partha","Dhananjaya","Gudakesha","Savyasachi","Phalguna","Vijaya"],
        "Yudhishthira": ["Dharmaraja","Ajatashatru","Yudhisthira"],
        "Bhima": ["Bhimasena","Vrikodara"],
        "Nakula": [],
        "Sahadeva": [],
        "Draupadi": ["Panchali","Krishnaa","Yajnaseni"],
        "Duryodhana": ["Suyodhana"],
        "Karna": ["Radheya","Vasusena","Angaraja","Sutaputra"],
        "Drona": ["Dronacharya","Bharadvaja"],
        "Bhishma": ["Devavrata","Gangeya","Pitamaha"],
        "Dhritarashtra": ["Dhritarastra"],
        "Vidura": [],
        "Kunti": ["Pritha"],
        "Gandhari": [],
        "Shakuni": ["Saubala"],
        "Ashwatthama": ["Drauni"],
        "Abhimanyu": [],
        "Ghatotkacha": [],
        "Vyasa": ["Veda Vyasa","Krishna Dvaipayana"],
        "Sanjaya": [],
        "Pandu": [],
        "Shikhandhi": ["Shikhandi"],
        "Drupada": [],
        "Dhrishtadyumna": [],
        "Jayadratha": ["Saindhava"],
        "Shalya": ["Salya"],
        "Kripa": ["Kripacharya"],
        "Satyaki": ["Yuyudhana"],
        "Ekalavya": [],
        "Shantanu": [],
        "Satyavati": [],
        "Ganga": [],
        "Vichitravirya": [],
        "Chitrangada": [],
        "Parashurama": ["Rama"],
        "Balarama": ["Baladeva"],
        "Subhadra": [],
        "Pradyumna": [],
        "Kritavarman": ["Hridika"],
        "Jarasandha": [],
        "Duhshasana": [],
    },
    "LOCATION": {
        "Hastinapura": ["Hastinapur"],
        "Indraprastha": [],
        "Kurukshetra": ["Dharmakshetra","Samantapanchaka"],
        "Dwarka": ["Dwaraka","Dvaravati"],
        "Panchala": [],
        "Khandava": ["Khandavaprastha"],
        "Kamyaka": [],
        "Virata": [],
        "Prabhasa": [],
        "Mathura": [],
    },
    "WEAPON": {
        "Gandiva": ["Gaandiva"],
        "Sudarshana Chakra": ["Sudarshana"],
        "Brahmastra": [],
        "Pashupatastra": ["Pashupata"],
        "Narayanastra": [],
        "Shakti": ["Vasavi Shakti"],
    },
    "CLAN": {
        "Pandava": ["Pandavas"],
        "Kaurava": ["Kauravas","Dhartarashtras"],
        "Yadava": ["Yadavas","Vrishni","Vrishnis"],
        "Panchala": ["Panchalas"],
    },
    "EVENT": {
        "Dice Game": ["Game of Dice","Dyuta"],
        "Kurukshetra War": ["Great War"],
        "Burning of Khandava": ["Khandava Dahana"],
        "Bhagavad Gita": ["Gita","Song of God"],
        "Chakravyuha": ["Padmavyuha"],
    },
}

# ── Build spaCy pipeline ──────────────────────────────────────────────────

def build_nlp():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    patterns = []
    for label, entities in SEED_ENTITIES.items():
        for canonical, aliases in entities.items():
            all_names = [canonical] + aliases
            for name in all_names:
                patterns.append({"label": label, "pattern": name, "id": canonical})
                tokens = name.split()
                if len(tokens) > 1:
                    patterns.append({
                        "label": label,
                        "pattern": [{"LOWER": t.lower()} for t in tokens],
                        "id": canonical,
                    })
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)
    print(f"EntityRuler loaded with {len(patterns)} patterns")
    return nlp

# ── NER extraction ────────────────────────────────────────────────────────

SPACY_LABEL_MAP = {
    "PERSON": "CHARACTER",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "EVENT": "EVENT",
    "ORG": "CLAN",
}

VALID_LABELS = {"CHARACTER","LOCATION","WEAPON","CLAN","EVENT"}

def extract_entities_from_doc(doc):
    seen = {}
    for ent in doc.ents:
        canonical = ent.ent_id_ if ent.ent_id_ else ent.text.strip()
        label = ent.label_
        if label not in VALID_LABELS:
            label = SPACY_LABEL_MAP.get(label)
        if label and label in VALID_LABELS:
            if canonical not in seen:
                seen[canonical] = label
    return seen

def extract_cooccurrences(entities_per_chunk):
    cooccur = Counter()
    for chunk_entities in entities_per_chunk:
        characters = sorted([name for name, label in chunk_entities.items() if label == "CHARACTER"])
        for i in range(len(characters)):
            for j in range(i + 1, len(characters)):
                cooccur[(characters[i], characters[j])] += 1
    return cooccur

# ── Relationship pattern extraction ───────────────────────────────────────

FAMILY_PATTERNS = [
    (r"(\w+),?\s+(?:the\s+)?son\s+of\s+(\w+)", "SON_OF"),
    (r"(\w+),?\s+(?:the\s+)?daughter\s+of\s+(\w+)", "DAUGHTER_OF"),
    (r"(\w+)\s+married\s+(\w+)", "MARRIED_TO"),
    (r"(\w+),?\s+(?:the\s+)?wife\s+of\s+(\w+)", "MARRIED_TO"),
    (r"(\w+)\s+(?:killed|slew|slain)\s+(\w+)", "KILLED"),
    (r"(\w+),?\s+(?:the\s+)?brother\s+of\s+(\w+)", "SIBLING_OF"),
    (r"(\w+),?\s+(?:the\s+)?teacher\s+of\s+(\w+)", "MENTOR_OF"),
    (r"(\w+),?\s+(?:the\s+)?preceptor\s+of\s+(\w+)", "MENTOR_OF"),
    (r"(\w+),?\s+(?:the\s+)?disciple\s+of\s+(\w+)", "MENTOR_OF"),
]

def extract_pattern_relationships(text, known_entities):
    rels = []
    for pattern, rel_type in FAMILY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            source = match.group(1).strip()
            target = match.group(2).strip()
            if source in known_entities and target in known_entities:
                if rel_type == "MENTOR_OF" and "disciple" in pattern:
                    source, target = target, source
                rels.append((source, target, rel_type))
    return rels

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 2: NER & RELATIONSHIP EXTRACTION")
    print("=" * 70)

    print("\nParsing and chunking text...")
    chunks = parse_and_chunk()
    print(f"  {len(chunks)} chunks ready\n")

    nlp = build_nlp()

    print(f"\nRunning NER on {len(chunks)} chunks...")
    all_entity_counts = Counter()
    all_entity_types = {}
    entities_per_chunk = []
    all_texts = [c["text"] for c in chunks]

    for i, doc in enumerate(nlp.pipe(all_texts, batch_size=64)):
        chunk_ents = extract_entities_from_doc(doc)
        entities_per_chunk.append(chunk_ents)
        for name, label in chunk_ents.items():
            all_entity_counts[name] += 1
            all_entity_types[name] = label
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(chunks)} chunks...")

    print(f"  Done. Found {len(all_entity_counts)} unique entities.\n")

    print("=" * 70)
    print("ENTITIES BY TYPE")
    print("=" * 70)

    type_counts = Counter()
    for name, label in all_entity_types.items():
        type_counts[label] += 1

    for label, count in type_counts.most_common():
        print(f"\n  {label} ({count}):")
        entities_of_type = [(name, all_entity_counts[name]) for name, l in all_entity_types.items() if l == label]
        entities_of_type.sort(key=lambda x: -x[1])
        for name, freq in entities_of_type[:15]:
            bar = "#" * min(freq, 50)
            print(f"    {name:<25} {freq:4d} {bar}")
        if len(entities_of_type) > 15:
            print(f"    ... and {len(entities_of_type) - 15} more")

    print("\n" + "=" * 70)
    print("CO-OCCURRENCE RELATIONSHIPS (top 20)")
    print("=" * 70)

    cooccur = extract_cooccurrences(entities_per_chunk)
    print(f"\n  Total co-occurrence pairs: {len(cooccur)}")
    print(f"\n  Strongest connections:")
    for (a, b), count in cooccur.most_common(20):
        bar = "#" * min(count, 40)
        print(f"    {a:<15} <-> {b:<15} {count:4d} {bar}")

    print("\n" + "=" * 70)
    print("PATTERN-BASED RELATIONSHIPS")
    print("=" * 70)

    known = set(all_entity_types.keys())
    full_text = "\n".join(all_texts)
    all_pattern_rels = extract_pattern_relationships(full_text, known)

    unique_rels = {}
    for src, tgt, rel in all_pattern_rels:
        key = (src, tgt, rel)
        unique_rels[key] = unique_rels.get(key, 0) + 1

    print(f"\n  Unique pattern relationships: {len(unique_rels)}")
    rel_type_counts = Counter(r[2] for r in unique_rels.keys())
    print(f"\n  By type:")
    for rtype, cnt in rel_type_counts.most_common():
        print(f"    {rtype:<20} {cnt:3d}")

    print(f"\n  All extracted relationships:")
    for (src, tgt, rel), freq in sorted(unique_rels.items(), key=lambda x: -x[1]):
        print(f"    {src} --[{rel}]--> {tgt}  (x{freq})")

    print("\n" + "=" * 70)
    print("PHASE 2 VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"  Unique entities:         {len(all_entity_counts)}")
    print(f"  Entity types:            {dict(type_counts.most_common())}")
    print(f"  Co-occurrence pairs:     {len(cooccur)}")
    print(f"  Pattern relationships:   {len(unique_rels)}")
    print()

if __name__ == "__main__":
    main()
