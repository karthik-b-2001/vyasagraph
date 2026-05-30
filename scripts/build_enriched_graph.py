"""Build enriched Neo4j graph from Wikidata CSVs + text-attested + CSV dataset.

Usage:
    python scripts/build_enriched_graph.py
"""

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REVIEWED_CSV = DATA_DIR / "reviewed_relationships.csv"
RELATIONS_CSV = DATA_DIR / "test_relations.csv"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "vyasagraph"

# ── Name normalization ────────────────────────────────────────────────────

TITLE_PREFIXES = ["King ", "Lord ", "Maharaja ", "Queen ", "Sri ", "Prince ", "Princess "]

# Wikidata diacritics → English
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
    "Nāgnajitī": "Nagnajiti", "Lakṣmaṇā": "Lakshmana",
    "Bhānumatī": "Bhanumati", "Śalya": "Shalya",
    "Kṛpā": "Kripa", "Kṛpa": "Kripa", "Vidurā": "Vidura",
    "Satyavatī": "Satyavati", "Ambikā": "Ambika",
    "Ambālikā": "Ambalika", "Vicitravīrya": "Vichitravirya",
    "Śantanu": "Shantanu", "Gaṅgā": "Ganga", "Vyāsa": "Vyasa",
    "Paraśurāma": "Parashurama", "Dhṛtarāṣṭra": "Dhritarashtra",
    "Laxman Kumara": "Lakshmana", "Lakshmanaa": "Lakshmana",
    "Wives of Duryodhana": "Bhanumati",
    "Irāvān": "Iravan", "Balarāma": "Balarama",
    "Śvetā": "Shveta", "Citrāṅgadā": "Chitrangada",
    "Ulūpī": "Ulupi", "Sūrya": "Surya",
}

# Attested name cleanup — titles, groups, parse errors
ATTESTED_NAME_MAP = {
    "Gandhara King": "",
    "Madras King": "",
    "King": "",
    "Demigod": "",
    "Wind God": "Vayu",
    "Personality": "",
    "Rakshasa": "",
    "Rakshasa Kirmira": "Kirmira",
    "King Surasena": "Surasena",
    "King Subala": "Subala",
    "King Nagnajit": "Nagnajit",
    "King Jarasandha": "Jarasandha",
    "King Drupada": "Drupada",
    "King Dasaratha": "Dasaratha",
    "Maharaja Shantanu": "Shantanu",
    "Maharaja Yudhisthira": "Yudhishthira",
    "King Yudhisthira": "Yudhishthira",
    "King Dhritarastra": "Dhritarashtra",
    "Queen Kunti": "Kunti",
    "Aunt Kunti": "Kunti",
    "Lord Krishna": "Krishna",
    "Lord Balarama": "Balarama",
    "Lord Baladeva": "Balarama",
    "Lord Shiva": "Shiva",
    "Lord Ramachandra": "Rama",
    "Vyasadeva": "Vyasa",
    "Krishna Dvaipayana Vyasa": "Vyasa",
    "Krishna Dvaipayana": "Vyasa",
    "Pandavas": "",
    "Kauravas": "",
    "Dasarna King": "",
    "Nivatakavachas": "",
    "Trigartas": "",
    "Prtha": "Kunti",
    "Shisupala": "Shishupala",
    "Hidimva": "Hidimba",
    "Dhristadyumna": "Dhrishtadyumna",
    "Dhristaketu": "Dhrishtaketu",
    "Ashvatthama": "Ashwatthama",
    "Salya": "Shalya",
    "Bhurisravas": "Bhurishravas",
    "Lakshman": "Lakshmana",
    "Parasurama": "Parashurama",
    "Dhritarastra": "Dhritarashtra",
    "Yudhisthira": "Yudhishthira",
    "Shikhandi": "Shikhandhi",
    "Baladeva": "Balarama",
}

def strip_diacritics(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalize_name(name):
    name = name.strip()
    if not name:
        return ""
    if re.match(r"^Q\d+$", name):
        return ""
    # Check attested name map first.
    if name in ATTESTED_NAME_MAP:
        return ATTESTED_NAME_MAP[name]
    # Check diacritics map.
    if name in DIACRITICS_MAP:
        return DIACRITICS_MAP[name]
    # Strip title prefixes.
    for prefix in TITLE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Strip diacritics.
    cleaned = strip_diacritics(name)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    # One more pass through attested map after cleaning.
    if cleaned in ATTESTED_NAME_MAP:
        return ATTESTED_NAME_MAP[cleaned]
    return cleaned

# ── CSV readers ───────────────────────────────────────────────────────────

def read_wikidata_characters():
    path = DATA_DIR / "wikidata_characters.csv"
    if not path.exists():
        return []
    rels = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            char = normalize_name(row.get("charLabel", ""))
            father = normalize_name(row.get("fatherLabel", ""))
            mother = normalize_name(row.get("motherLabel", ""))
            spouse = normalize_name(row.get("spouseLabel", ""))
            if char and father:
                rels.append((char, father, "SON_OF", "wikidata"))
            if char and mother:
                rels.append((char, mother, "SON_OF", "wikidata"))
            if char and spouse:
                rels.append((char, spouse, "MARRIED_TO", "wikidata"))
    return rels

def read_wikidata_parents():
    path = DATA_DIR / "wikidata_spouses.csv"
    if not path.exists():
        return []
    rels = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            child = normalize_name(row.get("childLabel", ""))
            parent = normalize_name(row.get("parentLabel", ""))
            if child and parent:
                rels.append((child, parent, "SON_OF", "wikidata"))
    return rels

def read_wikidata_children():
    path = DATA_DIR / "wikidata_children.csv"
    if not path.exists():
        return []
    rels = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parent = normalize_name(row.get("parentLabel", ""))
            child = normalize_name(row.get("childLabel", ""))
            if parent and child:
                rels.append((child, parent, "SON_OF", "wikidata"))
    return rels

def read_wikidata_siblings():
    path = DATA_DIR / "wikidata_siblings.csv"
    if not path.exists():
        return []
    rels = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person = normalize_name(row.get("personLabel", ""))
            sibling = normalize_name(row.get("siblingLabel", ""))
            if person and sibling:
                rels.append((person, sibling, "SIBLING_OF", "wikidata"))
    return rels

def read_wikidata_kills():
    path = DATA_DIR / "wikidata_kills.csv"
    if not path.exists():
        return []
    rels = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            victim = normalize_name(row.get("victimLabel", ""))
            killer = normalize_name(row.get("killerLabel", ""))
            if victim and killer:
                rels.append((killer, victim, "KILLED", "wikidata"))
    return rels

def read_wikidata_conflicts():
    path = DATA_DIR / "wikidata_conflicts.csv"
    if not path.exists():
        return []
    rels = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            char = normalize_name(row.get("charLabel", ""))
            conflict = normalize_name(row.get("conflictLabel", ""))
            if char and conflict:
                rels.append((char, conflict, "PARTICIPATED_IN", "wikidata"))
    return rels

def read_reviewed():
    if not REVIEWED_CSV.exists():
        return []
    rels = []
    with open(REVIEWED_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("approved", "").strip().lower() != "yes":
                continue
            src = normalize_name(row["source"].strip())
            tgt = normalize_name(row["target"].strip())
            if not src or not tgt:
                continue
            rels.append((src, tgt, row["type"].strip(), "attested"))
    return rels

def read_csv_relations():
    if not RELATIONS_CSV.exists():
        return []
    rels = []
    seen = set()
    with open(RELATIONS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: v.strip() if v else "" for k, v in row.items()}
            pairs = [
                ("Son", "Father", "SON_OF"),
                ("Son2", "Mother", "SON_OF"),
                ("Husband", "Wife", "MARRIED_TO"),
                ("Brothers1_1", "Brothers1_2", "SIBLING_OF"),
                ("Brothers2_1", "Brothers2_2", "SIBLING_OF"),
            ]
            for sc, tc, rel in pairs:
                s, t = row.get(sc, ""), row.get(tc, "")
                if s and t:
                    s = normalize_name(s)
                    t = normalize_name(t)
                    if not s or not t:
                        continue
                    key = (s, t, rel)
                    rev = (t, s, rel)
                    if key not in seen and rev not in seen:
                        seen.add(key)
                        rels.append((s, t, rel, "csv"))
    return rels

# ── Deduplication ─────────────────────────────────────────────────────────

def deduplicate(all_rels):
    merged = {}
    for src, tgt, rel_type, origin in all_rels:
        if not src or not tgt or src == tgt:
            continue
        if rel_type in ("SIBLING_OF", "MARRIED_TO"):
            key = (min(src, tgt), max(src, tgt), rel_type)
        else:
            key = (src, tgt, rel_type)
        if key not in merged:
            merged[key] = {"source": src, "target": tgt, "type": rel_type, "origins": {origin}}
        else:
            merged[key]["origins"].add(origin)
    return list(merged.values())

# ── Neo4j loading ─────────────────────────────────────────────────────────

def load_neo4j(rels):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("  Connected to Neo4j")

    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        print("  Cleared existing data")
        for label in ["Character", "Event"]:
            s.run(f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")
        print("  Schema ready")

        events = set()
        for r in rels:
            if r["type"] == "PARTICIPATED_IN":
                events.add(r["target"])

        loaded = 0
        type_counts = defaultdict(int)
        for r in rels:
            src, tgt, rel_type = r["source"], r["target"], r["type"]
            origins = ", ".join(sorted(r["origins"]))
            src_label = "Event" if src in events else "Character"
            tgt_label = "Event" if tgt in events else "Character"
            s.run(f"MERGE (n:{src_label} {{name: $name}})", {"name": src})
            s.run(f"MERGE (n:{tgt_label} {{name: $name}})", {"name": tgt})
            result = s.run(
                f"MATCH (a {{name: $src}}) MATCH (b {{name: $tgt}}) MERGE (a)-[r:{rel_type}]->(b) SET r.origins = $origins RETURN count(r) as cnt",
                {"src": src, "tgt": tgt, "origins": origins}
            )
            record = result.single()
            if record and record["cnt"] > 0:
                loaded += 1
                type_counts[rel_type] += 1

        print(f"\n  Loaded {loaded} relationships")
        print(f"\n  By type:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t:<20} {c:4d}")

        # Summary.
        print("\n" + "=" * 70)
        print("GRAPH SUMMARY")
        print("=" * 70)
        print("\n  Node counts:")
        result = s.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC")
        for record in result:
            print(f"    {record['label']:<15} {record['cnt']:4d}")

        print("\n  Relationship counts:")
        result = s.run("MATCH ()-[r]->() RETURN type(r) as t, count(r) as cnt ORDER BY cnt DESC")
        for record in result:
            print(f"    {record['t']:<20} {record['cnt']:4d}")

        # Sanity checks.
        print("\n  Sanity checks:")
        checks = [
            ("Arjuna killed:", "MATCH (a {name:'Arjuna'})-[:KILLED]->(b) RETURN b.name as n ORDER BY n"),
            ("Bhima killed:", "MATCH (a {name:'Bhima'})-[:KILLED]->(b) RETURN b.name as n ORDER BY n"),
            ("Krishna killed:", "MATCH (a {name:'Krishna'})-[:KILLED]->(b) RETURN b.name as n ORDER BY n"),
            ("Arjuna's parents:", "MATCH (a {name:'Arjuna'})-[:SON_OF]->(b) RETURN b.name as n ORDER BY n"),
            ("Krishna's parents:", "MATCH (a {name:'Krishna'})-[:SON_OF]->(b) RETURN b.name as n ORDER BY n"),
            ("Krishna's wives:", "MATCH (a {name:'Krishna'})-[:MARRIED_TO]-(b) RETURN DISTINCT b.name as n ORDER BY n"),
            ("Draupadi married:", "MATCH (a)-[:MARRIED_TO]-(b {name:'Draupadi'}) RETURN DISTINCT a.name as n ORDER BY n"),
            ("Bhishma's parents:", "MATCH (a {name:'Bhishma'})-[:SON_OF]->(b) RETURN b.name as n ORDER BY n"),
            ("Arjuna's siblings:", "MATCH (a {name:'Arjuna'})-[:SIBLING_OF]-(b) RETURN DISTINCT b.name as n ORDER BY n"),
            ("War participants:", "MATCH (a)-[:PARTICIPATED_IN]->(b {name:'Kurukshetra War'}) RETURN a.name as n ORDER BY n"),
            ("Duryodhana's parents:", "MATCH (a {name:'Duryodhana'})-[:SON_OF]->(b) RETURN b.name as n ORDER BY n"),
            ("Karna's parents:", "MATCH (a {name:'Karna'})-[:SON_OF]->(b) RETURN b.name as n ORDER BY n"),
            ("Abhimanyu's parents:", "MATCH (a {name:'Abhimanyu'})-[:SON_OF]->(b) RETURN b.name as n ORDER BY n"),
        ]
        for label, query in checks:
            result = s.run(query)
            names = [r["n"] for r in result]
            print(f"    {label:<25} {', '.join(names) if names else '(none)'}")

        # Shortest paths.
        print("\n  Shortest paths:")
        paths = [("Arjuna", "Duryodhana"), ("Krishna", "Bhishma"), ("Draupadi", "Gandhari")]
        for src, tgt in paths:
            result = s.run(
                "MATCH path = shortestPath((a {name: $src})-[*..6]-(b {name: $tgt})) RETURN [n IN nodes(path) | n.name] as nodes, [r IN relationships(path) | type(r)] as rels",
                {"src": src, "tgt": tgt}
            )
            record = result.single()
            if record:
                print(f"    {src} -> {tgt}: {' -> '.join(record['nodes'])} (via {', '.join(record['rels'])})")
            else:
                print(f"    {src} -> {tgt}: no path found")

        # Most connected.
        print("\n  Most connected characters:")
        result = s.run("MATCH (c:Character)-[r]-() RETURN c.name as name, count(r) as rels ORDER BY rels DESC LIMIT 15")
        for record in result:
            bar = "#" * min(record["rels"], 40)
            print(f"    {record['name']:<25} {record['rels']:4d} {bar}")

    driver.close()

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("BUILD ENRICHED GRAPH (Wikidata + Text + CSV)")
    print("=" * 70)

    all_rels = []

    print("\n  Reading Wikidata CSVs...")
    sources = [
        ("characters.csv", read_wikidata_characters),
        ("parents (spouses.csv)", read_wikidata_parents),
        ("children.csv", read_wikidata_children),
        ("siblings.csv", read_wikidata_siblings),
        ("kills.csv", read_wikidata_kills),
        ("conflicts.csv", read_wikidata_conflicts),
    ]
    for name, reader in sources:
        rels = reader()
        print(f"    {name:<30} {len(rels):4d}")
        all_rels.extend(rels)

    wikidata_total = len(all_rels)
    print(f"\n  Wikidata total:              {wikidata_total}")

    print("\n  Reading local sources...")
    attested = read_reviewed()
    print(f"    Attested (text):           {len(attested)}")
    all_rels.extend(attested)

    csv_rels = read_csv_relations()
    print(f"    CSV dataset:               {len(csv_rels)}")
    all_rels.extend(csv_rels)

    print(f"\n  Combined raw:                {len(all_rels)}")

    deduped = deduplicate(all_rels)
    print(f"  After dedup:                 {len(deduped)}")

    # Origin breakdown.
    origin_counts = defaultdict(int)
    for r in deduped:
        for o in r["origins"]:
            origin_counts[o] += 1
    print(f"\n  By origin:")
    for o, c in sorted(origin_counts.items(), key=lambda x: -x[1]):
        print(f"    {o:<20} {c:4d}")

    type_counts = defaultdict(int)
    for r in deduped:
        type_counts[r["type"]] += 1
    print(f"\n  By type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<20} {c:4d}")

    # Multi-source.
    multi = [r for r in deduped if len(r["origins"]) > 1]
    print(f"\n  Multi-source confirmed:      {len(multi)}")
    for r in multi[:15]:
        print(f"    {r['source']:<20} --[{r['type']}]--> {r['target']:<20} ({', '.join(r['origins'])})")

    print("\n  Loading into Neo4j...")
    load_neo4j(deduped)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Sources: Wikidata ({wikidata_total}) + Attested ({len(attested)}) + CSV ({len(csv_rels)})")
    print(f"  Total unique: {len(deduped)}")
    print(f"  Browse: http://localhost:7474 (neo4j / vyasagraph)")
    print()

if __name__ == "__main__":
    main()
