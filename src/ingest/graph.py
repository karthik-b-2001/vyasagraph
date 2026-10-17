"""Curated relationship CSVs -> Neo4j.

Graph facts come from three origins, all reviewed files in data/graph/:
  wikidata  - Wikidata SPARQL exports (wikidata_*.csv)
  attested  - spaCy extractions approved by hand (reviewed_relationships.csv)
  csv       - a pre-built family-relations table (csv_family.csv)

Names are normalized, facts are deduplicated (each keeps the set of origins
that support it), facts listed in exclusions.csv are dropped, and the graph
is rebuilt from scratch. At a few hundred facts a full rebuild takes seconds
and guarantees deleted facts really disappear.
"""

import csv
import re
import unicodedata
from collections import defaultdict

from neo4j import GraphDatabase

from src import config

SYMMETRIC = {"SIBLING_OF", "MARRIED_TO"}

# ── Name normalization ────────────────────────────────────────────────────

TITLE_PREFIXES = ["King ", "Lord ", "Maharaja ", "Queen ", "Sri ", "Prince ", "Princess "]

# Wikidata labels whose diacritic-stripped form is not the common spelling.
DIACRITICS_MAP = {
    "Karṇa": "Karna", "Pāṇḍu": "Pandu", "Mādrī": "Madri", "Kuntī": "Kunti",
    "Dhṛṣṭadyumna": "Dhrishtadyumna", "Śakuni": "Shakuni",
    "Śikhaṇḍī": "Shikhandhi", "Aśvatthāmā": "Ashwatthama",
    "Ghaṭotkaca": "Ghatotkacha", "Virāṭa": "Virata",
    "Rukmiṇī": "Rukmini", "Satyabhāmā": "Satyabhama",
    "Jāmbavatī": "Jambavati", "Devakī": "Devaki",
    "Yudhiṣṭhira": "Yudhishthira", "Śiśupāla": "Shishupala",
    "Bhīṣma": "Bhishma", "Bhīma": "Bhima", "Droṇa": "Drona",
    "Subhadrā": "Subhadra", "Draupadī": "Draupadi",
    "Gāndhārī": "Gandhari", "Sātyaki": "Satyaki",
    "Revatī": "Revati", "Rādhā": "Radha", "Uttarā": "Uttara",
    "Kālindī": "Kalindi", "Mitravindā": "Mitravinda",
    "Nāgnajitī": "Nagnajiti",
    "Bhānumatī": "Bhanumati", "Śalya": "Shalya",
    "Kṛpā": "Kripa", "Kṛpa": "Kripa", "Vidurā": "Vidura",
    "Satyavatī": "Satyavati", "Ambikā": "Ambika",
    "Ambālikā": "Ambalika", "Vicitravīrya": "Vichitravirya",
    "Śantanu": "Shantanu", "Śāntanu": "Shantanu", "Gaṅgā": "Ganga",
    "Gangā": "Ganga", "Vyāsa": "Vyasa",
    "Paraśurāma": "Parashurama", "Dhṛtarāṣṭra": "Dhritarashtra",
    # Three different people: Duryodhana's son (Lakshmana Kumara), his
    # daughter (Lakshmanaa, married Samba), Krishna's wife (Lakshmana).
    "Laxman Kumara": "Lakshmana Kumara", "Lakṣmaṇā": "Lakshmana",
    "Wives of Duryodhana": "Bhanumati",
    "Irāvān": "Iravan", "Balarāma": "Balarama",
    "Śvetā": "Shveta", "Citrāṅgadā": "Chitrangada",
    "Ulūpī": "Ulupi", "Sūrya": "Surya", "Hiḍimbī": "Hidimbi",
    "Duḥśāsana": "Duhshasana", "Duṣyanta": "Dushyanta",
    "Śakuntalā": "Shakuntala", "Damayantī": "Damayanti",
    "Devayānī": "Devayani", "Yayāti": "Yayati", "Nahuṣa": "Nahusha",
    "Manasā": "Manasa",
}

# Spellings in the text extractions and CSV dataset. "" = not a person, drop.
NAME_MAP = {
    "Gandhara King": "", "Madras King": "", "King": "", "Demigod": "",
    "Personality": "", "Rakshasa": "", "Pandavas": "", "Kauravas": "",
    "Dasarna King": "", "Nivatakavachas": "", "Trigartas": "",
    "Pandava": "", "Wives of Karna": "",
    "Askia Musa": "",  # Wikidata error: a 15th-century Songhai emperor listed as Kamsa's son
    "Jagannath": "",  # later Odia tradition, listed as a sibling of Balarama and Subhadra
    "Wind God": "Vayu",
    "God Brahma": "Brahma", "Rohini Devi": "Rohini",
    "Prativindya": "Prativindhya", "Srutakarman": "Shrutakarma",
    "Surasena": "Shurasena",
    "Lakshman": "Lakshmana Kumara",
    "Rakshasa Kirmira": "Kirmira",
    "Aunt Kunti": "Kunti", "Prtha": "Kunti",
    "Vyasadeva": "Vyasa", "Krishna Dvaipayana Vyasa": "Vyasa", "Krishna Dvaipayana": "Vyasa",
    "Lord Ramachandra": "Rama", "Lord Baladeva": "Balarama", "Baladeva": "Balarama",
    "Shisupala": "Shishupala", "Hidimva": "Hidimba",
    "Dhristadyumna": "Dhrishtadyumna", "Dhristaketu": "Dhrishtaketu",
    "Ashvatthama": "Ashwatthama", "Salya": "Shalya", "Bhurisravas": "Bhurishravas",
    "Parasurama": "Parashurama",
    "Dhritarastra": "Dhritarashtra", "Yudhisthira": "Yudhishthira",
    "Shikhandi": "Shikhandhi", "Devavrata": "Bhishma",
}


def normalize_name(name):
    name = name.strip()
    if not name or re.fullmatch(r"Q\d+", name):  # unlabeled Wikidata item
        return ""
    if name in NAME_MAP:
        return NAME_MAP[name]
    if name in DIACRITICS_MAP:
        return DIACRITICS_MAP[name]
    for prefix in TITLE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    name = name[:1].upper() + name[1:]
    return NAME_MAP.get(name, name)


# ── Readers: each yields (source, target, type, origin) ───────────────────

def _rows(filename):
    path = config.GRAPH_DIR / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


# (file, source column, target column, relationship type)
WIKIDATA_FILES = [
    ("wikidata_characters.csv", "charLabel", "fatherLabel", "SON_OF"),
    ("wikidata_characters.csv", "charLabel", "motherLabel", "SON_OF"),
    ("wikidata_characters.csv", "charLabel", "spouseLabel", "MARRIED_TO"),
    ("wikidata_children.csv", "childLabel", "parentLabel", "SON_OF"),
    ("wikidata_spouses.csv", "personLabel", "spouseLabel", "MARRIED_TO"),
    ("wikidata_siblings.csv", "personLabel", "siblingLabel", "SIBLING_OF"),
    ("wikidata_kills.csv", "killerLabel", "victimLabel", "KILLED"),
    ("wikidata_conflicts.csv", "charLabel", "conflictLabel", "PARTICIPATED_IN"),
]

CSV_FAMILY_PAIRS = [
    ("Son", "Father", "SON_OF"),
    ("Son2", "Mother", "SON_OF"),
    ("Husband", "Wife", "MARRIED_TO"),
    ("Brothers1_1", "Brothers1_2", "SIBLING_OF"),
    ("Brothers2_1", "Brothers2_2", "SIBLING_OF"),
]


def read_all():
    facts = []
    for filename, src_col, tgt_col, rel in WIKIDATA_FILES:
        for row in _rows(filename):
            facts.append((row.get(src_col, ""), row.get(tgt_col, ""), rel, "wikidata"))
    for row in _rows("reviewed_relationships.csv"):
        if row["approved"].lower() == "yes":
            facts.append((row["source"], row["target"], row["type"], "attested"))
    for row in _rows("csv_family.csv"):
        for src_col, tgt_col, rel in CSV_FAMILY_PAIRS:
            facts.append((row.get(src_col, ""), row.get(tgt_col, ""), rel, "csv"))
    return facts


# Sources say SON_OF / DAUGHTER_OF, and Wikidata's father/mother columns
# cover daughters too, so parent edges are stored as one gender-neutral type.
REL_ALIASES = {"SON_OF": "CHILD_OF", "DAUGHTER_OF": "CHILD_OF"}
# Relationship types are written into Cypher, so only these are accepted.
REL_TYPES = {"CHILD_OF", "MARRIED_TO", "SIBLING_OF", "KILLED", "MENTOR_OF", "PARTICIPATED_IN"}


def _key(src, tgt, rel):
    return (min(src, tgt), max(src, tgt), rel) if rel in SYMMETRIC else (src, tgt, rel)


def merge(raw_facts):
    """Normalize, drop excluded, dedupe. Returns (facts, number excluded)."""
    excluded = {_key(normalize_name(r["source"]), normalize_name(r["target"]),
                     REL_ALIASES.get(r["type"], r["type"]))
                for r in _rows("exclusions.csv")}
    merged, dropped = {}, 0
    for raw_src, raw_tgt, raw_rel, origin in raw_facts:
        src, tgt = normalize_name(raw_src), normalize_name(raw_tgt)
        rel = REL_ALIASES.get(raw_rel, raw_rel)
        if rel not in REL_TYPES:
            raise ValueError(f"unknown relationship type {raw_rel!r} ({origin}: {raw_src} -> {raw_tgt}); "
                             f"expected one of {sorted(REL_TYPES | REL_ALIASES.keys())}")
        if not src or not tgt or src == tgt:
            continue
        key = _key(src, tgt, rel)
        if key in excluded:
            dropped += 1
            continue
        if key not in merged:
            merged[key] = {"source": src, "target": tgt, "type": rel, "origins": set()}
        merged[key]["origins"].add(origin)
    return list(merged.values()), dropped


# ── Load ──────────────────────────────────────────────────────────────────

def ingest_graph():
    facts, dropped = merge(read_all())
    events = {f["target"] for f in facts if f["type"] == "PARTICIPATED_IN"}
    characters = ({f["source"] for f in facts} | {f["target"] for f in facts}) - events

    by_type = defaultdict(list)
    for f in facts:
        by_type[f["type"]].append({"src": f["source"], "tgt": f["target"],
                                   "origins": ", ".join(sorted(f["origins"]))})

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session() as s:
        for label in ("Character", "Event"):
            s.run(f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
                  f"FOR (n:{label}) REQUIRE n.name IS UNIQUE")

        def write(tx):
            tx.run("MATCH (n) DETACH DELETE n")
            tx.run("UNWIND $names AS name CREATE (:Character {name: name})", names=sorted(characters))
            tx.run("UNWIND $names AS name CREATE (:Event {name: name})", names=sorted(events))
            for rel, rows in by_type.items():
                # rel is checked against REL_TYPES in merge(); Cypher cannot parameterize types.
                tx.run(f"UNWIND $rows AS r MATCH (a {{name: r.src}}) MATCH (b {{name: r.tgt}}) "
                       f"CREATE (a)-[:{rel} {{origins: r.origins}}]->(b)", rows=rows)

        s.execute_write(write)
    driver.close()

    origins = defaultdict(int)
    for f in facts:
        for o in f["origins"]:
            origins[o] += 1
    multi = sum(len(f["origins"]) > 1 for f in facts)
    print(f"  {len(characters)} characters, {len(events)} events, {len(facts)} relationships "
          f"({dropped} raw facts excluded)")
    print(f"  by type: {dict(sorted(((t, len(r)) for t, r in by_type.items()), key=lambda x: -x[1]))}")
    print(f"  by origin: {dict(origins)}; supported by 2+ origins: {multi}")
