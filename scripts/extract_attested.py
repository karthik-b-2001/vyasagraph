"""Attestation-based relationship extraction from Mahabharata text.

Uses spaCy dependency parsing (not regex) to extract relationships
from grammatical structure. Each relationship is scored by how many
independent sentences attest to it across the entire text.

Three extraction layers:
1. Epithet mining: "X, son/daughter/wife of Y" via dependency tree
2. SVO kill extraction: subject-verb-object where verb is kill/slay
3. Attestation scoring: relationships ranked by independent evidence

Output: data/extracted_relationships.csv for human review

Usage:
    python scripts/extract_attested.py
"""

import csv
import re
import spacy
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_PATH = DATA_DIR / "1-18_books_combined.txt"
OUTPUT_CSV = DATA_DIR / "extracted_relationships.csv"
OUTPUT_JSON = DATA_DIR / "ground_truth.json"

# ── Alias resolution ──────────────────────────────────────────────────────
# Map known aliases/epithets to canonical names.
# This is the ONE place where we inject domain knowledge.

ALIAS_MAP = {
    # Arjuna
    "partha": "Arjuna", "dhananjaya": "Arjuna", "gudakesha": "Arjuna",
    "savyasachi": "Arjuna", "phalguna": "Arjuna", "vijaya": "Arjuna",
    "kiriti": "Arjuna", "jishnu": "Arjuna",
    # Krishna
    "vasudeva": "Krishna", "govinda": "Krishna", "keshava": "Krishna",
    "madhava": "Krishna", "janardana": "Krishna", "hari": "Krishna",
    "hrishikesha": "Krishna", "achyuta": "Krishna", "madhusudana": "Krishna",
    # Yudhishthira
    "dharmaraja": "Yudhishthira", "ajatashatru": "Yudhishthira",
    "yudhisthira": "Yudhishthira",
    # Bhima
    "bhimasena": "Bhima", "vrikodara": "Bhima",
    # Draupadi
    "panchali": "Draupadi", "krishnaa": "Draupadi", "yajnaseni": "Draupadi",
    # Duryodhana
    "suyodhana": "Duryodhana",
    # Karna
    "radheya": "Karna", "vasusena": "Karna", "sutaputra": "Karna",
    "angaraja": "Karna",
    # Bhishma
    "devavrata": "Bhishma", "gangeya": "Bhishma", "pitamaha": "Bhishma",
    # Drona
    "dronacharya": "Drona", "bharadvaja": "Drona",
    # Shakuni
    "saubala": "Shakuni",
    # Jayadratha
    "saindhava": "Jayadratha",
    # Kunti
    "pritha": "Kunti",
    # Shalya
    "salya": "Shalya",
    # Satyaki
    "yuyudhana": "Satyaki",
    # Ashwatthama
    "drauni": "Ashwatthama",
    # Dhritarashtra
    "dhritarastra": "Dhritarashtra",
    # Shikhandhi
    "shikhandi": "Shikhandhi",
    # Balarama
    "baladeva": "Balarama",
    # Kritavarman
    "hridika": "Kritavarman",
}

def canonicalize(name):
    """Resolve aliases to canonical name."""
    clean = name.strip()
    lookup = clean.lower()
    if lookup in ALIAS_MAP:
        return ALIAS_MAP[lookup]
    # Capitalize first letter if not already mapped.
    return clean if clean[0].isupper() else clean.capitalize()

# ── Parsing (reused) ─────────────────────────────────────────────────────

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

def parse_sentences_by_parva():
    """Parse text and return list of (sentence, parva) tuples."""
    raw = TEXT_PATH.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    parva_pattern = re.compile(
        r"^(" + "|".join(re.escape(n) for n in PARVA_NAMES) + r")\s*$",
        re.MULTILINE,
    )
    parts = parva_pattern.split(cleaned)
    all_sentences = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        canonical = PARVA_NAMES.get(header, header)
        # Remove chapter headers, commentary markers, "Thus Ends" lines.
        body = CHAPTER_RE.sub("", body)
        body = COMMENTARY_RE.sub("", body)
        body = THUS_ENDS_RE.sub("", body)
        # Split into sentences.
        for sent in re.split(r"(?<=[.!?])\s+", body):
            sent = sent.strip()
            if len(sent) > 20:  # skip very short fragments
                all_sentences.append((sent, canonical))
    return all_sentences

# ── Layer 1: Epithet mining via dependency parsing ────────────────────────

KINSHIP_NOUNS = {
    "son": "SON_OF",
    "sons": "SON_OF",
    "daughter": "DAUGHTER_OF",
    "daughters": "DAUGHTER_OF",
    "wife": "MARRIED_TO",
    "husband": "MARRIED_TO",
    "brother": "SIBLING_OF",
    "brothers": "SIBLING_OF",
    "sister": "SIBLING_OF",
    "sisters": "SIBLING_OF",
    "mother": "MOTHER_OF",
    "father": "FATHER_OF",
    "teacher": "MENTOR_OF",
    "guru": "MENTOR_OF",
    "preceptor": "MENTOR_OF",
    "disciple": "DISCIPLE_OF",
    "student": "DISCIPLE_OF",
    "pupil": "DISCIPLE_OF",
}

def extract_epithets(doc, sentence_text, parva):
    """Extract kinship relationships from dependency tree.
    
    Looks for patterns like:
    - "X, son of Y" -> appositive (X) + prep_of (Y)
    - "son of Y" where Y is clearly named
    - "Y's son X" -> possessive
    """
    extractions = []
    
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in KINSHIP_NOUNS:
            continue
        
        rel_type = KINSHIP_NOUNS[lemma]
        target = None  # the "of Y" part
        source = None  # the "X" part (the child/wife/brother)
        
        # Pattern 1: "son of Y" — look for prep "of" child with a proper noun.
        for child in token.children:
            if child.dep_ == "prep" and child.text.lower() == "of":
                for pobj in child.children:
                    if pobj.dep_ == "pobj" and pobj.pos_ == "PROPN":
                        target = get_full_name(pobj)
                        break
            # Pattern 2: "Y's son" — possessive modifier.
            if child.dep_ == "poss" and child.pos_ == "PROPN":
                target = get_full_name(child)
        
        if not target:
            continue
        
        # Find the source (the person being described).
        # Pattern A: "X, [the] son of Y" — X is the head of the appositive.
        if token.dep_ == "appos" and token.head.pos_ == "PROPN":
            source = get_full_name(token.head)
        
        # Pattern B: "son of Y" as subject/object — check for nsubj or attr.
        if not source:
            for child in token.children:
                if child.dep_ in ("nsubj", "attr", "nmod") and child.pos_ == "PROPN":
                    source = get_full_name(child)
                    break
        
        # Pattern C: "X was the son of Y" — X is nsubj of a copula.
        if not source and token.dep_ == "attr":
            head = token.head
            for child in head.children:
                if child.dep_ == "nsubj" and child.pos_ == "PROPN":
                    source = get_full_name(child)
                    break
        
        # Pattern D: token is head and subject is named.
        if not source:
            if token.head.pos_ == "PROPN" and token.dep_ in ("appos", "conj"):
                source = get_full_name(token.head)
        
        if source and target and source != target:
            # Normalize direction for some relationship types.
            src = canonicalize(source)
            tgt = canonicalize(target)
            
            if rel_type == "MOTHER_OF":
                # "X, mother of Y" -> Y SON_OF X
                extractions.append((tgt, src, "SON_OF", sentence_text, parva))
            elif rel_type == "FATHER_OF":
                # "X, father of Y" -> Y SON_OF X
                extractions.append((tgt, src, "SON_OF", sentence_text, parva))
            elif rel_type == "DISCIPLE_OF":
                # "X, disciple of Y" -> Y MENTOR_OF X
                extractions.append((tgt, src, "MENTOR_OF", sentence_text, parva))
            elif rel_type == "MARRIED_TO":
                # "X, wife of Y" -> X MARRIED_TO Y
                extractions.append((src, tgt, "MARRIED_TO", sentence_text, parva))
            else:
                extractions.append((src, tgt, rel_type, sentence_text, parva))
    
    return extractions

def get_full_name(token):
    """Get the full proper noun span including compounds."""
    # Walk left for compound modifiers.
    start = token.i
    for left in token.lefts:
        if left.dep_ in ("compound", "flat", "flat:name") and left.pos_ == "PROPN":
            start = min(start, left.i)
    # Walk right for names.
    end = token.i + 1
    for right in token.rights:
        if right.dep_ in ("flat", "flat:name", "compound") and right.pos_ == "PROPN":
            end = max(end, right.i + 1)
    return token.doc[start:end].text

# ── Layer 2: SVO kill extraction via dependency parsing ───────────────────

KILL_LEMMAS = {"kill", "slay", "behead", "defeat", "vanquish", "destroy", "smite"}

# Words that negate or weaken a kill claim.
NEGATION_DEPS = {"neg"}
WEAK_ADVMODS = {"almost", "nearly", "barely", "hardly"}

def extract_kills(doc, sentence_text, parva):
    """Extract KILLED relationships from subject-verb-object structure.
    
    Only extracts when:
    - The verb lemma is a kill word
    - The grammatical subject (nsubj) is a proper noun
    - The grammatical object (dobj) is a proper noun
    - There is no negation on the verb
    - The verb is past tense or passive (actually happened, not hypothetical)
    """
    extractions = []
    
    for token in doc:
        if token.lemma_.lower() not in KILL_LEMMAS:
            continue
        if token.pos_ != "VERB":
            continue
        
        # Check for negation.
        has_negation = False
        has_weak_adv = False
        for child in token.children:
            if child.dep_ in NEGATION_DEPS:
                has_negation = True
            if child.dep_ == "advmod" and child.text.lower() in WEAK_ADVMODS:
                has_weak_adv = True
        
        if has_negation or has_weak_adv:
            continue
        
        # Check tense: should be past or passive to indicate completed action.
        # VBD = past tense, VBN = past participle (passive).
        if token.tag_ not in ("VBD", "VBN", "VB"):
            continue
        
        # Find subject and object.
        subject = None
        obj = None
        
        for child in token.children:
            if child.dep_ == "nsubj" and child.pos_ == "PROPN":
                subject = get_full_name(child)
            if child.dep_ == "nsubjpass" and child.pos_ == "PROPN":
                # Passive: "Y was killed by X"
                obj = get_full_name(child)  # Y is the victim
            if child.dep_ in ("dobj", "nsubjpass") and child.pos_ == "PROPN":
                if child.dep_ == "dobj":
                    obj = get_full_name(child)
            # "killed by X" in passive.
            if child.dep_ == "agent":
                for pobj in child.children:
                    if pobj.dep_ == "pobj" and pobj.pos_ == "PROPN":
                        subject = get_full_name(pobj)
        
        # Handle passive voice: "Y was killed by X"
        if not subject and obj:
            # Look for agent "by X"
            for child in token.children:
                if child.dep_ == "agent":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj" and pobj.pos_ == "PROPN":
                            subject = get_full_name(pobj)
            if subject:
                # In passive, obj is the victim, subject is the killer.
                killer = canonicalize(subject)
                victim = canonicalize(obj)
                if killer != victim:
                    extractions.append((killer, victim, "KILLED", sentence_text, parva))
                continue
        
        # Active voice: "X killed Y"
        if subject and obj:
            killer = canonicalize(subject)
            victim = canonicalize(obj)
            if killer != victim:
                extractions.append((killer, victim, "KILLED", sentence_text, parva))
    
    return extractions

# ── Layer 3: Attestation scoring ──────────────────────────────────────────

def score_attestations(all_extractions):
    """Group extractions by (source, target, type) and count attestations.
    
    Returns a list of dicts sorted by attestation count descending.
    Each dict has: source, target, type, count, parvas, sample_sentences.
    """
    groups = defaultdict(lambda: {
        "sentences": [],
        "parvas": set(),
    })
    
    for src, tgt, rel_type, sentence, parva in all_extractions:
        key = (src, tgt, rel_type)
        groups[key]["sentences"].append(sentence[:150])
        groups[key]["parvas"].add(parva)
    
    scored = []
    for (src, tgt, rel_type), data in groups.items():
        # Deduplicate very similar sentences (within 80% overlap).
        unique_sentences = deduplicate_sentences(data["sentences"])
        scored.append({
            "source": src,
            "target": tgt,
            "type": rel_type,
            "count": len(unique_sentences),
            "parvas": sorted(data["parvas"]),
            "sample_sentences": unique_sentences[:3],
        })
    
    scored.sort(key=lambda x: -x["count"])
    return scored

def deduplicate_sentences(sentences):
    """Remove near-duplicate sentences."""
    unique = []
    for sent in sentences:
        is_dup = False
        s_words = set(sent.lower().split())
        for existing in unique:
            e_words = set(existing.lower().split())
            overlap = len(s_words & e_words) / max(len(s_words | e_words), 1)
            if overlap > 0.8:
                is_dup = True
                break
        if not is_dup:
            unique.append(sent)
    return unique

# ── Output ────────────────────────────────────────────────────────────────

def write_csv(scored, path):
    """Write scored relationships to CSV for human review."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source", "target", "type", "attestations",
            "parvas", "sample_1", "sample_2", "sample_3", "approved"
        ])
        for r in scored:
            samples = r["sample_sentences"]
            writer.writerow([
                r["source"],
                r["target"],
                r["type"],
                r["count"],
                "; ".join(r["parvas"]),
                samples[0] if len(samples) > 0 else "",
                samples[1] if len(samples) > 1 else "",
                samples[2] if len(samples) > 2 else "",
                "",  # empty "approved" column for human review
            ])

def build_ground_truth_json(scored, min_attestations=2):
    """Build ground_truth.json from scored relationships above threshold."""
    import json
    
    characters = set()
    locations = set()
    weapons = set()
    clans = set()
    events = set()
    
    rels_by_type = defaultdict(list)
    
    for r in scored:
        if r["count"] < min_attestations:
            continue
        
        src, tgt, rel_type = r["source"], r["target"], r["type"]
        characters.add(src)
        characters.add(tgt)
        
        rels_by_type[rel_type].append({
            "source": src,
            "target": tgt,
            "verified": False,
            "attestations": r["count"],
            "parvas": r["parvas"],
            "evidence": r["sample_sentences"][:2],
        })
    
    mk = lambda names: [{"name": n, "aliases": [], "description": ""} for n in sorted(names)]
    
    gt = {
        "_meta": {
            "description": "Attestation-extracted relationships from Mahabharata text",
            "method": "spaCy dependency parsing with attestation scoring",
            "min_attestations": min_attestations,
            "total_relationships": sum(len(v) for v in rels_by_type.values()),
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
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ATTESTATION-BASED RELATIONSHIP EXTRACTION")
    print("=" * 70)
    
    # Step 1: Parse text into sentences.
    print("\n  Parsing text into sentences...")
    sentence_parva_pairs = parse_sentences_by_parva()
    print(f"  {len(sentence_parva_pairs)} sentences across {len(PARVA_NAMES)} parvas")
    
    # Step 2: Load spaCy.
    print("\n  Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    # Increase max length for long sentences.
    nlp.max_length = 2_000_000
    print("  Model loaded")
    
    # Step 3: Process sentences in batches.
    print("\n  Extracting relationships via dependency parsing...")
    
    all_extractions = []
    epithet_count = 0
    kill_count = 0
    batch_size = 200
    
    sentences = [s for s, _ in sentence_parva_pairs]
    parvas = [p for _, p in sentence_parva_pairs]
    
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i+batch_size]
        batch_parvas = parvas[i:i+batch_size]
        
        # Process batch through spaCy.
        docs = list(nlp.pipe(batch_sents, batch_size=batch_size))
        
        for doc, sent_text, parva in zip(docs, batch_sents, batch_parvas):
            # Layer 1: Epithets.
            epithets = extract_epithets(doc, sent_text, parva)
            all_extractions.extend(epithets)
            epithet_count += len(epithets)
            
            # Layer 2: Kills.
            kills = extract_kills(doc, sent_text, parva)
            all_extractions.extend(kills)
            kill_count += len(kills)
        
        processed = min(i + batch_size, len(sentences))
        if processed % 2000 == 0 or processed == len(sentences):
            print(f"    {processed}/{len(sentences)} sentences | {epithet_count} epithets | {kill_count} kills")
    
    print(f"\n  Total raw extractions: {len(all_extractions)}")
    print(f"    Epithets: {epithet_count}")
    print(f"    Kills: {kill_count}")
    
    # Step 4: Score attestations.
    print("\n  Scoring attestations...")
    scored = score_attestations(all_extractions)
    print(f"  {len(scored)} unique relationships")
    
    # Step 5: Print results by attestation tier.
    print("\n" + "=" * 70)
    print("RESULTS BY ATTESTATION TIER")
    print("=" * 70)
    
    tiers = [
        ("HIGH CONFIDENCE (5+ attestations)", lambda r: r["count"] >= 5),
        ("MEDIUM CONFIDENCE (3-4 attestations)", lambda r: 3 <= r["count"] <= 4),
        ("LOW CONFIDENCE (2 attestations)", lambda r: r["count"] == 2),
        ("SINGLE ATTESTATION (1 only)", lambda r: r["count"] == 1),
    ]
    
    for tier_name, predicate in tiers:
        tier_rels = [r for r in scored if predicate(r)]
        print(f"\n  {tier_name}: {len(tier_rels)} relationships")
        
        # Group by type.
        type_counts = defaultdict(int)
        for r in tier_rels:
            type_counts[r["type"]] += 1
        for rtype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {rtype:<20} {cnt:4d}")
        
        # Show top examples.
        if tier_rels:
            print(f"\n    Top examples:")
            for r in tier_rels[:10]:
                parvas_str = ", ".join(r["parvas"][:2])
                print(f"      {r['source']:<18} --[{r['type']}]--> {r['target']:<18} (x{r['count']}, {parvas_str})")
                if r["sample_sentences"]:
                    sent = r["sample_sentences"][0][:100]
                    print(f"        \"{sent}...\"")
    
    # Step 6: Write outputs.
    print("\n" + "=" * 70)
    print("WRITING OUTPUTS")
    print("=" * 70)
    
    write_csv(scored, OUTPUT_CSV)
    print(f"\n  CSV (all relationships): {OUTPUT_CSV}")
    
    # Build ground truth with 2+ attestation threshold.
    min_att = 2
    build_ground_truth_json(scored, min_attestations=min_att)
    above_threshold = len([r for r in scored if r["count"] >= min_att])
    print(f"  JSON (>={min_att} attestations): {OUTPUT_JSON} ({above_threshold} relationships)")
    
    # Summary.
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Sentences processed:   {len(sentence_parva_pairs)}")
    print(f"  Raw extractions:       {len(all_extractions)}")
    print(f"  Unique relationships:  {len(scored)}")
    print(f"  5+ attestations:       {len([r for r in scored if r['count'] >= 5])}")
    print(f"  3-4 attestations:      {len([r for r in scored if 3 <= r['count'] <= 4])}")
    print(f"  2 attestations:        {len([r for r in scored if r['count'] == 2])}")
    print(f"  1 attestation:         {len([r for r in scored if r['count'] == 1])}")
    print()
    print("  NEXT STEPS:")
    print(f"  1. Open {OUTPUT_CSV} in a spreadsheet")
    print(f"  2. Review relationships, mark 'approved' column as 'yes' or 'no'")
    print(f"  3. Run: python scripts/load_graph.py")
    print()

if __name__ == "__main__":
    main()
