"""Build ground_truth.json from reviewed extractions + CSV dataset, then load into Neo4j.

Reads:
  - data/reviewed_relationships.csv (your approved attestation-extracted relationships)
  - data/test_relations.csv (pre-built family relationships)

Writes:
  - data/ground_truth.json (merged, deduplicated)

Then loads everything into Neo4j.

Usage:
    python scripts/build_and_load.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from neo4j import GraphDatabase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REVIEWED_CSV = DATA_DIR / "reviewed_relationships.csv"
RELATIONS_CSV = DATA_DIR / "test_relations.csv"
OUTPUT_JSON = DATA_DIR / "ground_truth.json"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "vyasagraph"

# ── Step 1: Read approved relationships from reviewed CSV ─────────────────

def read_reviewed():
    rels = []
    with open(REVIEWED_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("approved", "").strip().lower() != "yes":
                continue
            rels.append({
                "source": row["source"].strip(),
                "target": row["target"].strip(),
                "type": row["type"].strip(),
                "origin": "attested",
                "attestations": int(row.get("attestations", 1)),
            })
    return rels

# ── Step 2: Read test_relations.csv ───────────────────────────────────────

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
                    key = (s, t, rel)
                    rev = (t, s, rel)
                    if key not in seen and rev not in seen:
                        seen.add(key)
                        rels.append({
                            "source": s,
                            "target": t,
                            "type": rel,
                            "origin": "csv",
                            "attestations": 1,
                        })
    return rels

# ── Step 3: Merge and deduplicate ─────────────────────────────────────────

def merge(attested, csv_rels):
    merged = {}
    # CSV first (baseline).
    for r in csv_rels:
        key = (r["source"], r["target"], r["type"])
        merged[key] = r
    # Attested on top (higher attestation wins).
    for r in attested:
        key = (r["source"], r["target"], r["type"])
        if key in merged:
            existing = merged[key]
            if r["attestations"] > existing.get("attestations", 0):
                r["origin"] = "attested+csv"
                merged[key] = r
            else:
                existing["origin"] = "attested+csv"
        else:
            merged[key] = r
    return list(merged.values())

# ── Step 4: Build ground_truth.json ───────────────────────────────────────

def build_json(rels):
    characters = set()
    locations = set()
    weapons = set()
    clans = set()
    events = set()

    # Categorize entities by relationship context.
    for r in rels:
        t = r["type"]
        if t in ("SON_OF", "DAUGHTER_OF", "MARRIED_TO", "SIBLING_OF", "KILLED", "MENTOR_OF"):
            characters.add(r["source"])
            characters.add(r["target"])
        elif t == "WIELDED":
            characters.add(r["source"])
            weapons.add(r["target"])
        elif t == "BELONGS_TO":
            characters.add(r["source"])
            clans.add(r["target"])
        elif t == "PARTICIPATED_IN":
            characters.add(r["source"])
            events.add(r["target"])
        elif t == "RULES":
            characters.add(r["source"])
            locations.add(r["target"])
        else:
            characters.add(r["source"])
            characters.add(r["target"])

    mk = lambda names: [{"name": n, "aliases": [], "description": ""} for n in sorted(names)]

    rels_by_type = defaultdict(list)
    for r in rels:
        rels_by_type[r["type"]].append({
            "source": r["source"],
            "target": r["target"],
            "verified": True,
            "origin": r["origin"],
            "attestations": r["attestations"],
        })

    gt = {
        "_meta": {
            "description": "Ground truth built from attestation-based extraction (reviewed) + CSV dataset",
            "method": "spaCy dependency parsing + human review + test_relations.csv",
            "total_relationships": len(rels),
        },
        "entities": {
            "characters": mk(characters),
            "locations": mk(locations),
            "weapons": mk(weapons),
            "clans": mk(clans),
            "events": mk(events),
        },
        "relationships": dict(rels_by_type),
    }
    return gt

# ── Step 5: Load into Neo4j ───────────────────────────────────────────────

LABEL_MAP = {
    "SON_OF": ("Character", "Character"),
    "DAUGHTER_OF": ("Character", "Character"),
    "MARRIED_TO": ("Character", "Character"),
    "SIBLING_OF": ("Character", "Character"),
    "KILLED": ("Character", "Character"),
    "MENTOR_OF": ("Character", "Character"),
    "WIELDED": ("Character", "Weapon"),
    "BELONGS_TO": ("Character", "Clan"),
    "PARTICIPATED_IN": ("Character", "Event"),
    "RULES": ("Character", "Location"),
    "LOCATED_IN": ("Event", "Location"),
    "ALLIED_WITH": ("Character", "Character"),
}

def load_neo4j(gt):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("  Connected to Neo4j")

    with driver.session() as s:
        # Clear.
        s.run("MATCH (n) DETACH DELETE n")
        print("  Cleared existing data")

        # Constraints.
        for label in ["Character", "Location", "Weapon", "Clan", "Event"]:
            s.run(f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")
        print("  Schema ready")

        # Load entities.
        total_entities = 0
        type_map = {
            "characters": "Character",
            "locations": "Location",
            "weapons": "Weapon",
            "clans": "Clan",
            "events": "Event",
        }
        for etype, label in type_map.items():
            entities = gt.get("entities", {}).get(etype, [])
            for e in entities:
                s.run(
                    f"MERGE (n:{label} {{name: $name}}) SET n.aliases = $aliases",
                    {"name": e["name"], "aliases": e.get("aliases", [])}
                )
                total_entities += 1
            print(f"  {len(entities)} {etype}")

        # Load relationships.
        total_rels = 0
        for rel_type, rels in gt.get("relationships", {}).items():
            loaded = 0
            failed = 0
            for r in rels:
                if not r.get("verified", False):
                    continue
                # Ensure both nodes exist.
                src_label = LABEL_MAP.get(rel_type, ("Character", "Character"))[0]
                tgt_label = LABEL_MAP.get(rel_type, ("Character", "Character"))[1]
                s.run(f"MERGE (n:{src_label} {{name: $name}})", {"name": r["source"]})
                s.run(f"MERGE (n:{tgt_label} {{name: $name}})", {"name": r["target"]})

                result = s.run(
                    f"""
                    MATCH (a {{name: $src}})
                    MATCH (b {{name: $tgt}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.origin = $origin, r.attestations = $att
                    RETURN count(r) as cnt
                    """,
                    {
                        "src": r["source"],
                        "tgt": r["target"],
                        "origin": r.get("origin", ""),
                        "att": r.get("attestations", 1),
                    }
                )
                record = result.single()
                if record and record["cnt"] > 0:
                    loaded += 1
                else:
                    failed += 1
            total_rels += loaded
            print(f"  {rel_type:<20} {loaded:3d} loaded" + (f" ({failed} failed)" if failed else ""))

        # Summary.
        print("\n" + "=" * 70)
        print("GRAPH SUMMARY")
        print("=" * 70)

        print("\n  Node counts:")
        for label in ["Character", "Location", "Weapon", "Clan", "Event"]:
            result = s.run(f"MATCH (n:{label}) RETURN count(n) as cnt")
            print(f"    {label:<15} {result.single()['cnt']:4d}")

        print("\n  Relationship counts:")
        result = s.run("MATCH ()-[r]->() RETURN type(r) as t, count(r) as cnt ORDER BY cnt DESC")
        for record in result:
            print(f"    {record['t']:<20} {record['cnt']:4d}")

        print(f"\n  Total: {total_entities} entities, {total_rels} relationships")

        # Sanity checks.
        print("\n  Sanity checks:")

        checks = [
            ("Arjuna killed:", "MATCH (a:Character {name:'Arjuna'})-[:KILLED]->(b) RETURN b.name as n ORDER BY n"),
            ("Bhima killed:", "MATCH (a:Character {name:'Bhima'})-[:KILLED]->(b) RETURN b.name as n ORDER BY n"),
            ("Arjuna's parents:", "MATCH (a:Character {name:'Arjuna'})-[:SON_OF]->(b) RETURN b.name as n ORDER BY n"),
            ("Arjuna's siblings:", "MATCH (a:Character {name:'Arjuna'})-[:SIBLING_OF]-(b) RETURN DISTINCT b.name as n ORDER BY n"),
            ("Arjuna's teachers:", "MATCH (t)-[:MENTOR_OF]->(a:Character {name:'Arjuna'}) RETURN t.name as n ORDER BY n"),
            ("Draupadi's husbands:", "MATCH (h)-[:MARRIED_TO]->(d:Character {name:'Draupadi'}) RETURN h.name as n ORDER BY n"),
            ("Bhishma's parents:", "MATCH (a:Character {name:'Bhishma'})-[:SON_OF]->(b) RETURN b.name as n ORDER BY n"),
        ]

        for label, query in checks:
            result = s.run(query)
            names = [r["n"] for r in result]
            print(f"    {label:<25} {', '.join(names) if names else '(none)'}")

        # Shortest path.
        print("\n  Shortest path Arjuna -> Duryodhana:")
        result = s.run("""
            MATCH path = shortestPath(
                (a:Character {name:'Arjuna'})-[*..6]-(b:Character {name:'Duryodhana'})
            )
            RETURN [n IN nodes(path) | n.name] as nodes,
                   [r IN relationships(path) | type(r)] as rels
        """)
        record = result.single()
        if record:
            print(f"    {' -> '.join(record['nodes'])}")
            print(f"    via: {' -> '.join(record['rels'])}")
        else:
            print("    No path found")

    driver.close()

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("BUILD GROUND TRUTH + LOAD NEO4J")
    print("=" * 70)

    # Read sources.
    print("\n  Reading reviewed attestations...")
    attested = read_reviewed()
    print(f"  {len(attested)} approved relationships")

    print("  Reading CSV relations...")
    csv_rels = read_csv_relations()
    print(f"  {len(csv_rels)} CSV relationships")

    # Merge.
    print("\n  Merging and deduplicating...")
    merged = merge(attested, csv_rels)
    print(f"  {len(merged)} total relationships after merge")

    # Show breakdown.
    type_counts = defaultdict(int)
    origin_counts = defaultdict(int)
    for r in merged:
        type_counts[r["type"]] += 1
        origin_counts[r["origin"]] += 1

    print(f"\n  By type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<20} {c:4d}")

    print(f"\n  By origin:")
    for o, c in sorted(origin_counts.items(), key=lambda x: -x[1]):
        print(f"    {o:<20} {c:4d}")

    # Build JSON.
    print("\n  Building ground_truth.json...")
    gt = build_json(merged)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    print(f"  Written to {OUTPUT_JSON}")

    # Load Neo4j.
    print("\n  Loading into Neo4j...")
    load_neo4j(gt)
    
    print() 
    print(f"  Browse: http://localhost:7474 (neo4j / vyasagraph)")
    print()

if __name__ == "__main__":
    main()
