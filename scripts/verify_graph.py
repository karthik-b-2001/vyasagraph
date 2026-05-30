"""Phase 3: Load entities and relationships into Neo4j.

Filters noisy NER output, loads clean entities, CSV relationships,
pattern-based relationships, and top co-occurrence edges into the graph.

Usage:
    docker compose up neo4j -d
    python scripts/verify_graph.py
"""

import csv
import re
import spacy
from collections import Counter
from pathlib import Path
from neo4j import GraphDatabase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_PATH = DATA_DIR / "1-18_books_combined.txt"
CSV_PATH = DATA_DIR / "test_relations.csv"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "vyasagraph"

# ── Parsing (reused from Phase 1/2) ──────────────────────────────────────

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

# ── Seed Entities ─────────────────────────────────────────────────────────

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

# Known noise to exclude from graph.
NOISE_ENTITIES = {
    "Maharaja", "Supreme", "Yamaraja", "Madhu", "Prtha", "Narayana",
    "Vyasadeva", "Madri", "Earth", "Vaikuntha", "lotus",
    "the Supreme Personality of Godhead", "Rajasuya", "Bhagavad",
    "Ganges", "Rakshasa", "Bharata", "cursed", "Godhead", "King",
    "Sunaman", "the Great Battle", "the World Before",
}

# ── NER pipeline ──────────────────────────────────────────────────────────

SPACY_LABEL_MAP = {"PERSON":"CHARACTER","GPE":"LOCATION","LOC":"LOCATION","FAC":"LOCATION","EVENT":"EVENT","ORG":"CLAN"}
VALID_LABELS = {"CHARACTER","LOCATION","WEAPON","CLAN","EVENT"}

def build_nlp():
    nlp = spacy.load("en_core_web_sm")
    patterns = []
    for label, entities in SEED_ENTITIES.items():
        for canonical, aliases in entities.items():
            for name in [canonical] + aliases:
                patterns.append({"label": label, "pattern": name, "id": canonical})
                tokens = name.split()
                if len(tokens) > 1:
                    patterns.append({"label": label, "pattern": [{"LOWER": t.lower()} for t in tokens], "id": canonical})
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)
    return nlp

def run_ner(chunks, nlp):
    entity_counts = Counter()
    entity_types = {}
    entities_per_chunk = []
    for doc in nlp.pipe([c["text"] for c in chunks], batch_size=64):
        chunk_ents = {}
        for ent in doc.ents:
            canonical = ent.ent_id_ if ent.ent_id_ else ent.text.strip()
            label = ent.label_ if ent.label_ in VALID_LABELS else SPACY_LABEL_MAP.get(ent.label_)
            if label and label in VALID_LABELS and canonical not in NOISE_ENTITIES:
                chunk_ents[canonical] = label
        entities_per_chunk.append(chunk_ents)
        for name, label in chunk_ents.items():
            entity_counts[name] += 1
            entity_types[name] = label
    return entity_counts, entity_types, entities_per_chunk

# ── Relationship extraction ───────────────────────────────────────────────

FAMILY_PATTERNS = [
    (r"(\w+),?\s+(?:the\s+)?son\s+of\s+(\w+)", "SON_OF"),
    (r"(\w+),?\s+(?:the\s+)?daughter\s+of\s+(\w+)", "DAUGHTER_OF"),
    (r"(\w+)\s+married\s+(\w+)", "MARRIED_TO"),
    (r"(\w+),?\s+(?:the\s+)?wife\s+of\s+(\w+)", "MARRIED_TO"),
    (r"(\w+)\s+(?:killed|slew|slain)\s+(\w+)", "KILLED"),
    (r"(\w+),?\s+(?:the\s+)?brother\s+of\s+(\w+)", "SIBLING_OF"),
    (r"(\w+),?\s+(?:the\s+)?teacher\s+of\s+(\w+)", "MENTOR_OF"),
    (r"(\w+),?\s+(?:the\s+)?preceptor\s+of\s+(\w+)", "MENTOR_OF"),
]

def extract_pattern_rels(texts, known):
    full = "\n".join(texts)
    rels = {}
    for pattern, rel_type in FAMILY_PATTERNS:
        for match in re.finditer(pattern, full, re.IGNORECASE):
            src, tgt = match.group(1).strip(), match.group(2).strip()
            if src in known and tgt in known and src not in NOISE_ENTITIES and tgt not in NOISE_ENTITIES:
                key = (src, tgt, rel_type)
                rels[key] = rels.get(key, 0) + 1
    return rels

def extract_cooccurrences(entities_per_chunk, min_count=3):
    cooccur = Counter()
    for chunk_ents in entities_per_chunk:
        chars = sorted([n for n, l in chunk_ents.items() if l == "CHARACTER"])
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                cooccur[(chars[i], chars[j])] += 1
    return {pair: count for pair, count in cooccur.items() if count >= min_count}

def parse_csv_rels():
    if not CSV_PATH.exists():
        return [], []
    entities = set()
    rels = []
    seen = set()
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: v.strip() if v else "" for k, v in row.items()}
            for col in ["Son","Father","Son2","Mother","Husband","Wife","Brothers1_1","Brothers1_2","Brothers2_1","Brothers2_2"]:
                name = row.get(col, "")
                if name:
                    entities.add(name)
            pairs = [("Son","Father","SON_OF"),("Son2","Mother","SON_OF"),("Husband","Wife","MARRIED_TO"),("Brothers1_1","Brothers1_2","SIBLING_OF"),("Brothers2_1","Brothers2_2","SIBLING_OF")]
            for sc, tc, rel in pairs:
                s, t = row.get(sc,""), row.get(tc,"")
                if s and t:
                    key = (s, t, rel)
                    rev = (t, s, rel)
                    if key not in seen and rev not in seen:
                        seen.add(key)
                        rels.append((s, t, rel))
    return list(entities), rels

# ── Neo4j loading ─────────────────────────────────────────────────────────

def create_schema(session):
    constraints = [
        "CREATE CONSTRAINT character_name IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT weapon_name IF NOT EXISTS FOR (w:Weapon) REQUIRE w.name IS UNIQUE",
        "CREATE CONSTRAINT clan_name IF NOT EXISTS FOR (cl:Clan) REQUIRE cl.name IS UNIQUE",
        "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
    ]
    for c in constraints:
        session.run(c)

LABEL_MAP = {"CHARACTER":"Character","LOCATION":"Location","WEAPON":"Weapon","CLAN":"Clan","EVENT":"Event"}

def load_entities(session, entity_types, entity_counts, min_freq=3):
    """Load entities that are either in the seed catalog or appear 3+ times."""
    seed_names = set()
    for entities in SEED_ENTITIES.values():
        seed_names.update(entities.keys())

    loaded = 0
    for name, label in entity_types.items():
        if name in NOISE_ENTITIES:
            continue
        freq = entity_counts.get(name, 0)
        if name not in seed_names and freq < min_freq:
            continue
        neo_label = LABEL_MAP.get(label)
        if not neo_label:
            continue
        # Get aliases from seed catalog.
        aliases = []
        for etype_entities in SEED_ENTITIES.values():
            if name in etype_entities:
                aliases = etype_entities[name]
                break
        session.run(
            f"MERGE (n:{neo_label} {{name: $name}}) SET n.aliases = $aliases, n.frequency = $freq",
            {"name": name, "aliases": aliases, "freq": freq}
        )
        loaded += 1
    return loaded

def load_csv_entities(session, csv_entities):
    """Load entities from CSV that might not be in NER results."""
    loaded = 0
    for name in csv_entities:
        session.run(
            "MERGE (n:Character {name: $name})",
            {"name": name}
        )
        loaded += 1
    return loaded

def load_relationships(session, rels, rel_source):
    """Load relationships. Tries to match nodes regardless of label."""
    loaded = 0
    for src, tgt, rel_type in rels:
        result = session.run(
            f"""
            MATCH (a {{name: $src}})
            MATCH (b {{name: $tgt}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.source = $source
            RETURN count(r) as cnt
            """,
            {"src": src, "tgt": tgt, "source": rel_source}
        )
        record = result.single()
        if record and record["cnt"] > 0:
            loaded += 1
    return loaded

def load_cooccurrences(session, cooccur, top_n=100):
    """Load top co-occurrence edges."""
    sorted_pairs = sorted(cooccur.items(), key=lambda x: -x[1])[:top_n]
    loaded = 0
    for (a, b), count in sorted_pairs:
        result = session.run(
            """
            MATCH (x:Character {name: $a})
            MATCH (y:Character {name: $b})
            MERGE (x)-[r:CO_OCCURS]->(y)
            SET r.count = $count
            RETURN count(r) as cnt
            """,
            {"a": a, "b": b, "count": count}
        )
        record = result.single()
        if record and record["cnt"] > 0:
            loaded += 1
    return loaded

# ── Test queries ──────────────────────────────────────────────────────────

def run_test_queries(session):
    print("\n" + "=" * 70)
    print("TEST QUERIES")
    print("=" * 70)

    # Node counts by label.
    print("\n  Node counts:")
    for label in ["Character","Location","Weapon","Clan","Event"]:
        result = session.run(f"MATCH (n:{label}) RETURN count(n) as cnt")
        cnt = result.single()["cnt"]
        print(f"    {label:<15} {cnt:4d}")

    # Relationship counts by type.
    print("\n  Relationship counts:")
    result = session.run("MATCH ()-[r]->() RETURN type(r) as t, count(r) as cnt ORDER BY cnt DESC")
    for record in result:
        print(f"    {record['t']:<20} {record['cnt']:4d}")

    # Arjuna's relationships.
    print("\n  Arjuna's relationships:")
    result = session.run("""
        MATCH (a:Character {name: 'Arjuna'})-[r]->(b)
        RETURN type(r) as rel, b.name as name, 'outgoing' as dir
        UNION
        MATCH (b)-[r]->(a:Character {name: 'Arjuna'})
        RETURN type(r) as rel, b.name as name, 'incoming' as dir
    """)
    for record in result:
       print(f"    {record['rel']:<20} {record['name']} ({record['dir']})")

    # Shortest path: Arjuna to Duryodhana.
    print("\n  Shortest path Arjuna -> Duryodhana:")
    result = session.run("""
        MATCH path = shortestPath(
            (a:Character {name: 'Arjuna'})-[*..6]-(b:Character {name: 'Duryodhana'})
        )
        RETURN [n IN nodes(path) | n.name] as nodes,
               [r IN relationships(path) | type(r)] as rels
    """)
    record = result.single()
    if record:
        print(f"    Nodes: {' -> '.join(record['nodes'])}")
        print(f"    Via:   {' -> '.join(record['rels'])}")
    else:
        print("    No path found")

    # Top connected characters.
    print("\n  Most connected characters (by relationship count):")
    result = session.run("""
        MATCH (c:Character)-[r]-()
        RETURN c.name as name, count(r) as rels
        ORDER BY rels DESC LIMIT 10
    """)
    for record in result:
        bar = "#" * min(record["rels"], 40)
        print(f"    {record['name']:<20} {record['rels']:4d} {bar}")

    # Family tree for Arjuna.
    print("\n  Arjuna's family (SON_OF, MARRIED_TO, SIBLING_OF):")
    result = session.run("""
        MATCH (a:Character {name: 'Arjuna'})-[r:SON_OF|MARRIED_TO|SIBLING_OF]-(b)
        RETURN type(r) as rel, b.name as name
    """)
    for record in result:
        print(f"    {record['rel']:<15} {record['name']}")

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 3: NEO4J GRAPH LOADING")
    print("=" * 70)

    # Step 1: Parse, chunk, and run NER.
    print("\n  Parsing and chunking...")
    chunks = parse_and_chunk()
    print(f"  {len(chunks)} chunks")

    print("  Loading spaCy and running NER...")
    nlp = build_nlp()
    entity_counts, entity_types, entities_per_chunk = run_ner(chunks, nlp)
    print(f"  {len(entity_counts)} unique entities from NER")

    # Step 2: Extract relationships.
    print("  Extracting pattern relationships...")
    pattern_rels = extract_pattern_rels([c["text"] for c in chunks], set(entity_types.keys()))
    pattern_rel_list = [(s, t, r) for (s, t, r) in pattern_rels.keys()]
    print(f"  {len(pattern_rel_list)} pattern relationships")

    print("  Extracting co-occurrences...")
    cooccur = extract_cooccurrences(entities_per_chunk, min_count=3)
    print(f"  {len(cooccur)} co-occurrence pairs (3+ shared chunks)")

    print("  Parsing CSV relationships...")
    csv_entities, csv_rels = parse_csv_rels()
    print(f"  {len(csv_entities)} CSV entities, {len(csv_rels)} CSV relationships")

    # Step 3: Connect to Neo4j and load.
    print("\n  Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("  Connected.")

    with driver.session() as session:
        # Clear existing data.
        print("\n  Clearing existing data...")
        session.run("MATCH (n) DETACH DELETE n")

        # Create schema.
        print("  Creating schema...")
        create_schema(session)

        # Load entities.
        print("  Loading NER entities...")
        ner_loaded = load_entities(session, entity_types, entity_counts, min_freq=3)
        print(f"  Loaded {ner_loaded} entities from NER")

        print("  Loading CSV entities...")
        csv_e_loaded = load_csv_entities(session, csv_entities)
        print(f"  Loaded {csv_e_loaded} entities from CSV")

        # Load relationships.
        print("  Loading CSV relationships...")
        csv_r_loaded = load_relationships(session, csv_rels, "csv")
        print(f"  Loaded {csv_r_loaded} CSV relationships")

        print("  Loading pattern relationships...")
        pat_loaded = load_relationships(session, pattern_rel_list, "pattern")
        print(f"  Loaded {pat_loaded} pattern relationships")

        print("  Loading co-occurrence edges (top 100)...")
        cooc_loaded = load_cooccurrences(session, cooccur, top_n=100)
        print(f"  Loaded {cooc_loaded} co-occurrence edges")

        # Run test queries.
        run_test_queries(session)

    driver.close()

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"  Browse your graph at http://localhost:7474")
    print(f"  Login: neo4j / vyasagraph")
    print(f"  Try: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    print()

if __name__ == "__main__":
    main()
