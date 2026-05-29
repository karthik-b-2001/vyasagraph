"""Parse the pre-built relationships CSV into Entity and Relationship objects.

The CSV has columns: Son, Father, Son2, Mother, Husband, Wife,
Brothers1_1, Brothers1_2, Brothers2_1, Brothers2_2.

Each row encodes multiple relationship facts. This parser extracts all
unique entities and relationships, deduplicates them, and returns canonical
objects ready for Neo4j insertion.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.nlp.models import Entity, EntityType, Relationship, RelationshipType

logger = logging.getLogger(__name__)


def parse_relations_csv(path: Path) -> tuple[list[Entity], list[Relationship]]:
    """Parse the relations CSV and return (entities, relationships)."""
    entities: dict[str, Entity] = {}
    relationships: list[Relationship] = []
    seen_rels: set[tuple[str, str, str]] = set()

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean whitespace from all values.
            row = {k: v.strip() if v else "" for k, v in row.items()}

            # --- Extract entities ---
            for col in ["Son", "Father", "Son2", "Mother", "Husband", "Wife",
                        "Brothers1_1", "Brothers1_2", "Brothers2_1", "Brothers2_2"]:
                name = row.get(col, "")
                if name and name not in entities:
                    entities[name] = Entity(
                        name=name,
                        entity_type=EntityType.CHARACTER,
                    )

            # --- Extract relationships ---

            # Son -> Father (SON_OF)
            _add_rel(row, "Son", "Father", RelationshipType.SON_OF,
                     relationships, seen_rels)

            # Son2 -> Mother (SON_OF)
            _add_rel(row, "Son2", "Mother", RelationshipType.SON_OF,
                     relationships, seen_rels)

            # Husband -> Wife (MARRIED_TO)
            _add_rel(row, "Husband", "Wife", RelationshipType.MARRIED_TO,
                     relationships, seen_rels)

            # Brothers pairs
            _add_rel(row, "Brothers1_1", "Brothers1_2", RelationshipType.SIBLING_OF,
                     relationships, seen_rels)
            _add_rel(row, "Brothers2_1", "Brothers2_2", RelationshipType.SIBLING_OF,
                     relationships, seen_rels)

    logger.info(
        "Parsed relations CSV: %d entities, %d relationships",
        len(entities), len(relationships),
    )
    return list(entities.values()), relationships


def _add_rel(
    row: dict[str, str],
    source_col: str,
    target_col: str,
    rel_type: RelationshipType,
    relationships: list[Relationship],
    seen: set[tuple[str, str, str]],
) -> None:
    """Add a relationship if both source and target are non-empty and not a duplicate."""
    source = row.get(source_col, "")
    target = row.get(target_col, "")
    if not source or not target:
        return

    key = (source, target, rel_type.value)
    reverse_key = (target, source, rel_type.value)

    # For symmetric relations (SIBLING_OF, MARRIED_TO), check both directions.
    if rel_type in (RelationshipType.SIBLING_OF, RelationshipType.MARRIED_TO):
        if key in seen or reverse_key in seen:
            return
    else:
        if key in seen:
            return

    seen.add(key)
    relationships.append(
        Relationship(
            source=source,
            target=target,
            relation=rel_type,
            context=f"From relations dataset: {source} {rel_type.value} {target}",
            confidence=1.0,
        )
    )