"""
Safe incremental writer for REVIEWED Dynamic KG candidates (full V9 concrete types).

Target: production Neo4j database from NEO4J_DATABASE (default: neo4j).
Does NOT wipe, DROP, or DETACH DELETE. Does NOT call setup_database.py.

Extraction (/pdf-receive) must never call insert_reviewed_candidates automatically.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from neo4j import GraphDatabase

from backend.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from dynamic_kg.extract_facts import (
    EVENT_LABELS,
    PERSON_LABELS,
    POSITION_LABELS,
    PROPERTIES_BY_PRIMARY,
    REL_ENDPOINT_LABELS,
    SINGLE_LABEL_TYPES,
    STAGE_LABELS,
    CandidateGraph,
    CandidateNode,
    CandidateRelationship,
    primary_label,
)

# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

BASE_LABEL = "InertialSensors"
ALLOWED_POSITION_LABELS: frozenset[str] = frozenset(POSITION_LABELS)
ALLOWED_PRIMARY_LABELS: frozenset[str] = frozenset(
    SINGLE_LABEL_TYPES | {BASE_LABEL}
)
ALLOWED_SENSOR_PROPERTIES: frozenset[str] = PROPERTIES_BY_PRIMARY["InertialSensors"]
ALLOWED_PROVENANCE_PROPERTIES: frozenset[str] = frozenset(
    {
        "uri",
        "id",
        "sourceDocument",
        "rphdFileId",
        "ingestedAt",
    }
)
ALLOWED_RELATIONSHIP_TYPES: frozenset[str] = frozenset(REL_ENDPOINT_LABELS.keys())

V9_NS = "http://www.semanticweb.org/noamaadra/ontologies/2024/2/Horses#"

_POSITION_SET_CYPHER: dict[str, str] = {
    "Withers": "MATCH (s:InertialSensors) WHERE elementId(s) = $eid SET s:Withers",
    "Sternum": "MATCH (s:InertialSensors) WHERE elementId(s) = $eid SET s:Sternum",
    "CanonOfForelimb": (
        "MATCH (s:InertialSensors) WHERE elementId(s) = $eid SET s:CanonOfForelimb"
    ),
    "CanonOfHindlimb": (
        "MATCH (s:InertialSensors) WHERE elementId(s) = $eid SET s:CanonOfHindlimb"
    ),
}

# Static CREATE templates — label never interpolated from LLM.
_CREATE_NODE_CYPHER: dict[str, str] = {
    "Horse": "CREATE (n:Horse) SET n.uri = $uri, n.id = $id RETURN elementId(n) AS eid",
    "Rider": "CREATE (n:Rider) SET n.uri = $uri, n.id = $id RETURN elementId(n) AS eid",
    "Veterinarian": (
        "CREATE (n:Veterinarian) SET n.uri = $uri, n.id = $id RETURN elementId(n) AS eid"
    ),
    "Caretaker": (
        "CREATE (n:Caretaker) SET n.uri = $uri, n.id = $id RETURN elementId(n) AS eid"
    ),
    "ShowJumping": (
        "CREATE (n:ShowJumping) SET n.uri = $uri, n.id = $id RETURN elementId(n) AS eid"
    ),
    "Dressage": (
        "CREATE (n:Dressage) SET n.uri = $uri, n.id = $id RETURN elementId(n) AS eid"
    ),
    "Cross": "CREATE (n:Cross) SET n.uri = $uri, n.id = $id RETURN elementId(n) AS eid",
    "EventParticipation": (
        "CREATE (n:EventParticipation) SET n.uri = $uri, n.id = $id "
        "RETURN elementId(n) AS eid"
    ),
    "PreparationStage": (
        "CREATE (n:PreparationStage) SET n.uri = $uri, n.id = $id "
        "RETURN elementId(n) AS eid"
    ),
    "PreCompetitionStage": (
        "CREATE (n:PreCompetitionStage) SET n.uri = $uri, n.id = $id "
        "RETURN elementId(n) AS eid"
    ),
    "CompetitionStage": (
        "CREATE (n:CompetitionStage) SET n.uri = $uri, n.id = $id "
        "RETURN elementId(n) AS eid"
    ),
    "TransitionStage": (
        "CREATE (n:TransitionStage) SET n.uri = $uri, n.id = $id "
        "RETURN elementId(n) AS eid"
    ),
    "ExperimentalObjective": (
        "CREATE (n:ExperimentalObjective) SET n.uri = $uri, n.id = $id "
        "RETURN elementId(n) AS eid"
    ),
    "CompetitiveSeason": (
        "CREATE (n:CompetitiveSeason) SET n.uri = $uri, n.id = $id "
        "RETURN elementId(n) AS eid"
    ),
    "InertialSensors": (
        "CREATE (s:InertialSensors) SET s.hasSensorID = $hasSensorID, "
        "s.uri = $uri, s.id = $id RETURN elementId(s) AS eid"
    ),
}

_MERGE_REL_CYPHER: dict[str, str] = {
    rel: (
        f"MATCH (a), (b) WHERE elementId(a) = $sid AND elementId(b) = $tid "
        f"MERGE (a)-[r:{rel}]->(b) RETURN elementId(r) AS rid"
    )
    for rel in ALLOWED_RELATIONSHIP_TYPES
}

_EXISTING_REL_CYPHER: dict[str, str] = {
    rel: (
        f"MATCH (a)-[r:{rel}]->(b) "
        f"WHERE elementId(a) = $sid AND elementId(b) = $tid "
        f"RETURN elementId(r) AS rid LIMIT 1"
    )
    for rel in ALLOWED_RELATIONSHIP_TYPES
}

_FORBIDDEN_WRITE_TOKENS = re.compile(
    r"\b(DELETE|DETACH\s+DELETE|DROP|CREATE\s+DATABASE|DROP\s+DATABASE|REMOVE)\b",
    re.IGNORECASE,
)

ActionKind = Literal["create", "update", "noop", "conflict", "rejected"]

_OPTIONAL_SET_KEYS = (
    "hasName",
    "hasRace",
    "hasFormat",
    "hasSensorOffset",
    "hasFileSize",
    "hasSensorTime",
    "category",
    "eventDate",
    "eventLocation",
    "rank",
    "status",
    "Volume",
    "Intensity",
    "Frequency",
    "description",
    "seasonName",
    "seasonStart",
    "seasonEnd",
    "sourceDocument",
    "rphdFileId",
    "ingestedAt",
)


@dataclass
class Provenance:
    source_filename: str | None = None
    rphd_file_id: str | None = None
    source_hash: str | None = None
    ingested_at: str | None = None


@dataclass
class NodePlanItem:
    action: ActionKind
    local_id: str
    primary_label: str
    detail: str = ""
    hasSensorID: str | None = None
    position_label: str | None = None
    existing_element_id: str | None = None
    stable_id: str | None = None
    properties_to_set: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    ontology_id: str | None = None


@dataclass
class RelPlanItem:
    action: ActionKind
    type: str
    source_local_id: str
    target_name: str
    detail: str = ""
    target_local_id: str | None = None
    source_hasSensorID: str | None = None
    source_element_id: str | None = None
    source_stable_id: str | None = None
    target_element_id: str | None = None
    target_stable_id: str | None = None
    stable_id: str | None = None
    existing_rel_element_id: str | None = None


@dataclass
class PreflightResult:
    valid: bool
    database: str
    creates: list[dict[str, Any]] = field(default_factory=list)
    updates: list[dict[str, Any]] = field(default_factory=list)
    noops: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    relationships_to_create: list[dict[str, Any]] = field(default_factory=list)
    relationships_noop: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    affected_nodes: list[dict[str, Any]] = field(default_factory=list)
    affected_relationships: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InsertResult:
    success: bool
    database: str
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    noop: list[str] = field(default_factory=list)
    error: str | None = None
    preflight: dict[str, Any] | None = None
    affected_nodes: list[dict[str, Any]] = field(default_factory=list)
    affected_relationships: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DynamicIngestionValidationError(ValueError):
    """Candidate graph failed allowlist / structural validation."""


class DynamicIngestionConflictError(RuntimeError):
    """Batch cannot be applied due to conflicts or ambiguous matches."""


def assert_cypher_safe(cypher: str) -> None:
    if _FORBIDDEN_WRITE_TOKENS.search(cypher):
        raise DynamicIngestionValidationError(
            "Forbidden destructive Cypher token detected"
        )


def dynamic_sensor_uri(has_sensor_id: str) -> str:
    return f"urn:dynamic:sensor:{has_sensor_id}"


def dynamic_sensor_id(has_sensor_id: str) -> str:
    return f"dynamic_sensor_{has_sensor_id}"


def stable_node_id(
    props: dict[str, Any] | None,
    has_sensor_id: str | None = None,
) -> str:
    """Prefer uri → id → hasSensorID → hasName for KnowledgeGraph membership."""
    props = props or {}
    for key in ("uri", "id", "hasSensorID", "hasName"):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value)
    if has_sensor_id and str(has_sensor_id).strip():
        return dynamic_sensor_uri(str(has_sensor_id).strip())
    raise DynamicIngestionValidationError("Cannot derive stable node id")


def _normalize_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _position_label_from_node(node: CandidateNode) -> str | None:
    if BASE_LABEL not in node.labels:
        return None
    positions = [lab for lab in node.labels if lab in ALLOWED_POSITION_LABELS]
    if len(positions) != 1:
        raise DynamicIngestionValidationError(
            f"Node {node.local_id}: expected exactly one position label, got {node.labels}"
        )
    return positions[0]


def _node_property_dict(node: CandidateNode) -> dict[str, Any]:
    return node.properties.model_dump(mode="python")


def validate_candidate_graph(graph: CandidateGraph) -> None:
    """Strict allowlist validation. Raises on unknown labels/properties/rels."""
    for node in graph.nodes:
        primary = primary_label(list(node.labels))
        if primary not in ALLOWED_PRIMARY_LABELS:
            raise DynamicIngestionValidationError(
                f"Node {node.local_id}: unknown primary label {primary!r}"
            )
        if primary == BASE_LABEL:
            _position_label_from_node(node)
        elif len(node.labels) != 1 or node.labels[0] not in SINGLE_LABEL_TYPES:
            raise DynamicIngestionValidationError(
                f"Node {node.local_id}: invalid labels {node.labels}"
            )

        allowed_props = PROPERTIES_BY_PRIMARY[primary]
        raw = _node_property_dict(node)
        for key, value in raw.items():
            if value is None:
                continue
            if key not in allowed_props:
                raise DynamicIngestionValidationError(
                    f"Node {node.local_id}: unknown property {key!r}"
                )
        if primary == BASE_LABEL and not raw.get("hasSensorID"):
            raise DynamicIngestionValidationError(
                f"Node {node.local_id}: hasSensorID is required"
            )

    for rel in graph.relationships:
        if rel.type not in ALLOWED_RELATIONSHIP_TYPES:
            raise DynamicIngestionValidationError(
                f"Relationship {rel.source_local_id}->…: unknown type {rel.type!r}"
            )
        if not rel.target_local_id and not (rel.target_name or "").strip():
            raise DynamicIngestionValidationError(
                f"Relationship {rel.source_local_id}: target_local_id or target_name required"
            )


def _candidate_property_values(node: CandidateNode) -> dict[str, Any]:
    primary = primary_label(list(node.labels))
    allowed = PROPERTIES_BY_PRIMARY[primary]
    raw = _node_property_dict(node)
    return {
        k: v
        for k, v in raw.items()
        if v is not None and k in allowed and k != "id"
    }


def _provenance_props(provenance: Provenance | None) -> dict[str, Any]:
    if provenance is None:
        return {}
    out: dict[str, Any] = {}
    if provenance.source_filename:
        out["sourceDocument"] = provenance.source_filename
    if provenance.rphd_file_id:
        out["rphdFileId"] = provenance.rphd_file_id
    if provenance.ingested_at:
        out["ingestedAt"] = provenance.ingested_at
    return out


def _values_equivalent(existing_val: Any, cand: Any) -> bool:
    """Deterministic equivalence for Neo4j vs PDF literal representations."""
    if existing_val == cand:
        return True
    if existing_val is None or cand is None:
        return False
    # Date / datetime objects from Neo4j vs ISO strings from PDF
    for attr in ("iso_format", "isoformat"):
        fn = getattr(existing_val, attr, None)
        if callable(fn):
            try:
                return str(fn())[:10] == str(cand).strip()[:10]
            except Exception:  # noqa: BLE001
                break
    return str(existing_val).strip() == str(cand).strip()


def _diff_properties(
    existing: dict[str, Any],
    candidate_props: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    to_set: dict[str, Any] = {}
    conflicts: list[dict[str, Any]] = []
    comparable = False

    for key, cand in candidate_props.items():
        existing_val = existing.get(key)
        if existing_val is None:
            to_set[key] = cand
            comparable = True
            continue
        if _values_equivalent(existing_val, cand):
            comparable = True
            continue
        conflicts.append(
            {"property": key, "existing": existing_val, "candidate": cand}
        )
        comparable = True

    all_noop = comparable and not to_set and not conflicts
    if not candidate_props:
        all_noop = True
    elif not to_set and not conflicts:
        all_noop = True
    return to_set, conflicts, all_noop


def _get_driver():
    if not NEO4J_URI or not NEO4J_USER or NEO4J_PASSWORD is None:
        raise RuntimeError("Neo4j connection env vars are not configured")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _database_name() -> str:
    return NEO4J_DATABASE or "neo4j"


def _derive_ontology_id(node: CandidateNode) -> str:
    props = node.properties
    if props.id and str(props.id).strip():
        return str(props.id).strip()
    primary = primary_label(list(node.labels))
    if primary == BASE_LABEL:
        return dynamic_sensor_id(str(props.hasSensorID).strip())
    if primary == "Horse" and props.hasName:
        return f"Horse_{str(props.hasName).strip().replace(' ', '_')}"
    if primary == "Rider" and props.hasName:
        return f"Rider_{str(props.hasName).strip().replace(' ', '_')}"
    if primary == "Veterinarian" and props.hasName:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", str(props.hasName).strip())
        return cleaned if cleaned.startswith("Vet") else f"Vet_{cleaned}"
    if primary == "Caretaker" and props.hasName:
        return f"Caretaker_{str(props.hasName).strip().replace(' ', '_')}"
    if primary == "ExperimentalObjective":
        if props.hasName:
            return re.sub(r"[^A-Za-z0-9_]", "", str(props.hasName).strip().replace(" ", ""))
    if primary == "CompetitiveSeason" and props.seasonName:
        return f"Season_{str(props.seasonName).strip().replace(' ', '_')}"
    raise DynamicIngestionValidationError(
        f"Node {node.local_id}: cannot derive deterministic id"
    )


def _derive_uri(ontology_id: str, *, sensor: bool = False, has_sensor_id: str | None = None) -> str:
    if sensor and has_sensor_id:
        # Prefer matching existing V9 sensor uri when id looks like IMU_*
        if ontology_id.startswith("IMU_") or ontology_id.startswith("http"):
            if ontology_id.startswith("http"):
                return ontology_id
            return f"{V9_NS}{ontology_id}"
        return dynamic_sensor_uri(has_sensor_id)
    if ontology_id.startswith("http"):
        return ontology_id
    return f"{V9_NS}{ontology_id}"


def _fetch_sensors_by_id(session, has_sensor_id: str) -> list[dict[str, Any]]:
    cypher = (
        "MATCH (s:InertialSensors {hasSensorID: $hasSensorID}) "
        "RETURN elementId(s) AS eid, labels(s) AS labels, properties(s) AS props"
    )
    assert_cypher_safe(cypher)
    rows = []
    for record in session.run(cypher, hasSensorID=has_sensor_id):
        rows.append(
            {
                "eid": record["eid"],
                "labels": list(record["labels"] or []),
                "props": dict(record["props"] or {}),
            }
        )
    return rows


def _fetch_entities_by_identity(
    session,
    *,
    label: str,
    ontology_id: str | None,
    has_name: str | None = None,
    season_name: str | None = None,
    has_sensor_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read-only identity lookup. Label is static allowlisted token only."""
    if label not in ALLOWED_PRIMARY_LABELS:
        raise DynamicIngestionValidationError(f"Unsupported label: {label}")

    if label == BASE_LABEL:
        if not has_sensor_id:
            return []
        return _fetch_sensors_by_id(session, has_sensor_id)

    # Static label in Cypher string from allowlist — never from raw LLM text.
    cypher = (
        f"MATCH (t:{label}) "
        "WHERE ($oid <> '' AND toLower(coalesce(t.id, '')) = toLower($oid)) "
        "   OR ($oid <> '' AND toLower(coalesce(t.uri, '')) = toLower($uri)) "
        "   OR ($oid <> '' AND toLower(coalesce(t.uri, '')) ENDSWITH toLower($oid_suffix)) "
        "   OR ($name <> '' AND toLower(coalesce(t.hasName, '')) = toLower($name)) "
        "   OR ($season <> '' AND toLower(coalesce(t.seasonName, '')) = toLower($season)) "
        "RETURN elementId(t) AS eid, labels(t) AS labels, properties(t) AS props"
    )
    # ENDSWITH may not exist on all Neo4j versions — use alternative
    cypher = (
        f"MATCH (t:{label}) "
        "WHERE ($oid <> '' AND toLower(coalesce(t.id, '')) = toLower($oid)) "
        "   OR ($oid <> '' AND toLower(coalesce(t.uri, '')) = toLower($uri)) "
        "   OR ($oid <> '' AND toLower(coalesce(t.uri, '')) CONTAINS toLower($oid)) "
        "   OR ($name <> '' AND toLower(coalesce(t.hasName, '')) = toLower($name)) "
        "   OR ($season <> '' AND toLower(coalesce(t.seasonName, '')) = toLower($season)) "
        "RETURN elementId(t) AS eid, labels(t) AS labels, properties(t) AS props"
    )
    assert_cypher_safe(cypher)
    oid = (ontology_id or "").strip()
    name = (has_name or "").strip()
    season = (season_name or "").strip()
    uri = f"{V9_NS}{oid}" if oid else ""
    rows = []
    for record in session.run(
        cypher, oid=oid, uri=uri, name=name, season=season
    ):
        rows.append(
            {
                "eid": record["eid"],
                "labels": list(record["labels"] or []),
                "props": dict(record["props"] or {}),
            }
        )
    if len(rows) <= 1:
        return rows
    # Prefer exact id / hasName
    exact = []
    needle_id = oid.lower()
    needle_name = name.lower()
    for row in rows:
        props = row["props"]
        pid = str(props.get("id") or "").strip().lower()
        pname = str(props.get("hasName") or "").strip().lower()
        if needle_id and pid == needle_id:
            exact.append(row)
        elif needle_name and pname == needle_name:
            exact.append(row)
    return exact or rows


def _fetch_target_entities(
    session,
    *,
    label: str,
    target_name: str,
) -> list[dict[str, Any]]:
    """Resolve Horse / ExperimentalObjective / other by name (read-only)."""
    if label not in ALLOWED_PRIMARY_LABELS:
        raise DynamicIngestionValidationError(f"Unsupported target label: {label}")
    return _fetch_entities_by_identity(
        session,
        label=label,
        ontology_id=target_name.strip(),
        has_name=target_name.strip(),
        season_name=target_name.strip() if label == "CompetitiveSeason" else None,
    )


def _stable_from_entity_props(props: dict[str, Any]) -> str:
    return stable_node_id(
        props, props.get("hasSensorID") or props.get("hasName") or props.get("id")
    )


def _existing_relationship(
    session,
    *,
    rel_type: str,
    source_eid: str,
    target_eid: str,
) -> str | None:
    cypher = _EXISTING_REL_CYPHER.get(rel_type)
    if not cypher:
        return None
    assert_cypher_safe(cypher)
    record = session.run(cypher, sid=source_eid, tid=target_eid).single()
    return record["rid"] if record else None


def _plan_node(
    node: CandidateNode,
    session,
    provenance: Provenance | None,
) -> NodePlanItem:
    primary = primary_label(list(node.labels))
    position = _position_label_from_node(node) if primary == BASE_LABEL else None
    prov = _provenance_props(provenance)
    cand_props = _candidate_property_values(node)
    ontology_id = _derive_ontology_id(node)
    sensor_id = (
        str(node.properties.hasSensorID).strip()
        if primary == BASE_LABEL and node.properties.hasSensorID
        else None
    )

    matches = _fetch_entities_by_identity(
        session,
        label=primary,
        ontology_id=ontology_id,
        has_name=node.properties.hasName,
        season_name=node.properties.seasonName,
        has_sensor_id=sensor_id,
    )

    if len(matches) > 1:
        return NodePlanItem(
            action="rejected",
            local_id=node.local_id,
            primary_label=primary,
            hasSensorID=sensor_id,
            position_label=position,
            ontology_id=ontology_id,
            detail=f"Ambiguous: {len(matches)} existing {primary} matches",
        )

    if len(matches) == 0:
        uri = _derive_uri(
            ontology_id, sensor=(primary == BASE_LABEL), has_sensor_id=sensor_id
        )
        props_to_set: dict[str, Any] = {
            "uri": uri,
            "id": ontology_id,
            **cand_props,
            **prov,
        }
        if sensor_id:
            props_to_set["hasSensorID"] = sensor_id
        for key in list(props_to_set):
            allowed = (
                PROPERTIES_BY_PRIMARY[primary]
                | ALLOWED_PROVENANCE_PROPERTIES
            )
            if key not in allowed and key not in ("uri", "id", "hasSensorID"):
                del props_to_set[key]
        return NodePlanItem(
            action="create",
            local_id=node.local_id,
            primary_label=primary,
            hasSensorID=sensor_id,
            position_label=position,
            ontology_id=ontology_id,
            detail=f"No existing {primary}; would create",
            stable_id=stable_node_id(props_to_set, sensor_id),
            properties_to_set=props_to_set,
        )

    existing = matches[0]
    to_set, conflicts, _all_noop = _diff_properties(existing["props"], cand_props)
    for pk, pv in prov.items():
        if existing["props"].get(pk) is None:
            to_set[pk] = pv
    stable = stable_node_id(existing["props"], sensor_id)

    if conflicts:
        return NodePlanItem(
            action="conflict",
            local_id=node.local_id,
            primary_label=primary,
            hasSensorID=sensor_id,
            position_label=position,
            ontology_id=ontology_id,
            detail="Property conflict with existing production values",
            existing_element_id=existing["eid"],
            stable_id=stable,
            properties_to_set=to_set,
            conflicts=conflicts,
        )
    if to_set:
        return NodePlanItem(
            action="update",
            local_id=node.local_id,
            primary_label=primary,
            hasSensorID=sensor_id,
            position_label=position,
            ontology_id=ontology_id,
            detail="Would populate null properties on existing node",
            existing_element_id=existing["eid"],
            stable_id=stable,
            properties_to_set=to_set,
        )
    return NodePlanItem(
        action="noop",
        local_id=node.local_id,
        primary_label=primary,
        hasSensorID=sensor_id,
        position_label=position,
        ontology_id=ontology_id,
        detail="Existing node already matches candidate values",
        existing_element_id=existing["eid"],
        stable_id=stable,
    )


def _expected_target_labels(rel_type: str) -> frozenset[str]:
    return REL_ENDPOINT_LABELS[rel_type][1]


def _resolve_target_from_name(
    session,
    *,
    rel_type: str,
    target_name: str,
) -> list[dict[str, Any]]:
    labels = list(_expected_target_labels(rel_type))
    # Prefer Horse / ExperimentalObjective single-label lookups used historically.
    if len(labels) == 1:
        return _fetch_target_entities(
            session, label=labels[0], target_name=target_name
        )
    # Multi-label targets (events, stages, people): try each label.
    all_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lab in labels:
        for row in _fetch_target_entities(
            session, label=lab, target_name=target_name
        ):
            if row["eid"] not in seen:
                seen.add(row["eid"])
                all_rows.append(row)
    return all_rows


def _plan_relationship(
    rel: CandidateRelationship,
    nodes_by_local: dict[str, CandidateNode],
    session,
    node_plans_by_local: dict[str, NodePlanItem],
) -> RelPlanItem:
    target_display = rel.target_local_id or rel.target_name or ""
    if rel.type not in ALLOWED_RELATIONSHIP_TYPES:
        return RelPlanItem(
            action="rejected",
            type=rel.type,
            source_local_id=rel.source_local_id,
            target_name=target_display,
            target_local_id=rel.target_local_id,
            detail=f"Relationship type not allowlisted: {rel.type}",
        )

    source_node = nodes_by_local.get(rel.source_local_id)
    if source_node is None:
        return RelPlanItem(
            action="rejected",
            type=rel.type,
            source_local_id=rel.source_local_id,
            target_name=target_display,
            target_local_id=rel.target_local_id,
            detail="Source candidate node missing",
        )

    src_allowed, _tgt_allowed = REL_ENDPOINT_LABELS[rel.type]
    source_ok = (
        BASE_LABEL in source_node.labels and BASE_LABEL in src_allowed
    ) or any(lab in src_allowed for lab in source_node.labels)
    if not source_ok:
        return RelPlanItem(
            action="rejected",
            type=rel.type,
            source_local_id=rel.source_local_id,
            target_name=target_display,
            detail=f"Source labels {source_node.labels} invalid for {rel.type}",
        )

    source_plan = node_plans_by_local.get(rel.source_local_id)
    if source_plan and source_plan.action == "rejected":
        return RelPlanItem(
            action="rejected",
            type=rel.type,
            source_local_id=rel.source_local_id,
            target_name=target_display,
            target_local_id=rel.target_local_id,
            source_hasSensorID=source_plan.hasSensorID,
            detail="Source node plan was rejected",
        )

    source_eid = source_plan.existing_element_id if source_plan else None
    source_stable = source_plan.stable_id if source_plan else None
    sensor_id = source_plan.hasSensorID if source_plan else None

    if source_eid is None and source_plan and source_plan.action == "create":
        source_stable = source_plan.stable_id
    elif source_eid is None and BASE_LABEL in source_node.labels and sensor_id:
        matches = _fetch_sensors_by_id(session, sensor_id)
        if len(matches) == 1:
            source_eid = matches[0]["eid"]
            source_stable = stable_node_id(matches[0]["props"], sensor_id)
        elif len(matches) > 1:
            return RelPlanItem(
                action="rejected",
                type=rel.type,
                source_local_id=rel.source_local_id,
                target_name=target_display,
                source_hasSensorID=sensor_id,
                detail=f"Ambiguous source sensor hasSensorID={sensor_id!r}",
            )
        elif not (source_plan and source_plan.action == "create"):
            return RelPlanItem(
                action="rejected",
                type=rel.type,
                source_local_id=rel.source_local_id,
                target_name=target_display,
                source_hasSensorID=sensor_id,
                detail=f"Source sensor hasSensorID={sensor_id!r} not found in Neo4j",
            )

    # Target resolution
    target_eid = None
    target_stable = None
    if rel.target_local_id:
        tplan = node_plans_by_local.get(rel.target_local_id)
        if tplan is None or tplan.action == "rejected":
            return RelPlanItem(
                action="rejected",
                type=rel.type,
                source_local_id=rel.source_local_id,
                target_name=target_display,
                target_local_id=rel.target_local_id,
                source_hasSensorID=sensor_id,
                source_element_id=source_eid,
                source_stable_id=source_stable,
                detail="Target candidate missing or rejected",
            )
        target_eid = tplan.existing_element_id
        target_stable = tplan.stable_id
        if target_eid is None and tplan.action != "create":
            return RelPlanItem(
                action="rejected",
                type=rel.type,
                source_local_id=rel.source_local_id,
                target_name=target_display,
                target_local_id=rel.target_local_id,
                detail="Target node not found in Neo4j",
            )
    else:
        targets = _resolve_target_from_name(
            session, rel_type=rel.type, target_name=rel.target_name or ""
        )
        if len(targets) == 0:
            return RelPlanItem(
                action="rejected",
                type=rel.type,
                source_local_id=rel.source_local_id,
                target_name=target_display,
                source_hasSensorID=sensor_id,
                source_element_id=source_eid,
                source_stable_id=source_stable,
                detail=f"Zero matches for target_name={rel.target_name!r}",
            )
        if len(targets) > 1:
            return RelPlanItem(
                action="rejected",
                type=rel.type,
                source_local_id=rel.source_local_id,
                target_name=target_display,
                source_hasSensorID=sensor_id,
                source_element_id=source_eid,
                source_stable_id=source_stable,
                detail=f"Ambiguous: {len(targets)} matches for target_name={rel.target_name!r}",
            )
        target_eid = targets[0]["eid"]
        target_stable = _stable_from_entity_props(targets[0]["props"])

    stable = f"{rel.type}:{source_stable}:{target_stable}"

    if source_eid is not None and target_eid is not None:
        existing_rid = _existing_relationship(
            session,
            rel_type=rel.type,
            source_eid=source_eid,
            target_eid=target_eid,
        )
        if existing_rid:
            return RelPlanItem(
                action="noop",
                type=rel.type,
                source_local_id=rel.source_local_id,
                target_name=target_display,
                target_local_id=rel.target_local_id,
                detail="Relationship already exists",
                source_hasSensorID=sensor_id,
                source_element_id=source_eid,
                source_stable_id=source_stable,
                target_element_id=target_eid,
                target_stable_id=target_stable,
                stable_id=stable,
                existing_rel_element_id=existing_rid,
            )

    return RelPlanItem(
        action="create",
        type=rel.type,
        source_local_id=rel.source_local_id,
        target_name=target_display,
        target_local_id=rel.target_local_id,
        detail="Would create allowlisted relationship",
        source_hasSensorID=sensor_id,
        source_element_id=source_eid,
        source_stable_id=source_stable,
        target_element_id=target_eid,
        target_stable_id=target_stable,
        stable_id=stable,
    )


def plan_ingestion(
    graph: CandidateGraph,
    session,
    provenance: Provenance | None = None,
) -> tuple[list[NodePlanItem], list[RelPlanItem]]:
    validate_candidate_graph(graph)
    node_plans = [_plan_node(node, session, provenance) for node in graph.nodes]
    nodes_by_local = {n.local_id: n for n in graph.nodes}
    node_plans_by_local = {p.local_id: p for p in node_plans}
    rel_plans = [
        _plan_relationship(rel, nodes_by_local, session, node_plans_by_local)
        for rel in graph.relationships
    ]
    return node_plans, rel_plans


def _node_plan_to_dict(item: NodePlanItem) -> dict[str, Any]:
    return {
        "action": item.action,
        "hasSensorID": item.hasSensorID,
        "local_id": item.local_id,
        "primary_label": item.primary_label,
        "position_label": item.position_label,
        "detail": item.detail,
        "stable_id": item.stable_id,
        "ontology_id": item.ontology_id,
        "properties_to_set": item.properties_to_set,
        "conflicts": item.conflicts,
    }


def _membership_action(action: ActionKind) -> str | None:
    if action == "create":
        return "created"
    if action == "noop":
        return "reused"
    if action == "update":
        return "updated"
    return None


def _affected_nodes_from_plans(
    node_plans: list[NodePlanItem],
    rel_plans: list[RelPlanItem] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for plan in node_plans:
        membership = _membership_action(plan.action)
        if membership is None or not plan.stable_id:
            continue
        if plan.stable_id in seen:
            continue
        seen.add(plan.stable_id)
        out.append(
            {
                "stable_id": plan.stable_id,
                "action": membership,
                "hasSensorID": plan.hasSensorID,
                "local_id": plan.local_id,
                "primary_label": plan.primary_label,
            }
        )
    for rel in rel_plans or []:
        if rel.action not in ("create", "noop"):
            continue
        for sid, role in (
            (rel.source_stable_id, "relationship_source"),
            (rel.target_stable_id, "relationship_target"),
        ):
            if sid and sid not in seen:
                seen.add(sid)
                out.append(
                    {
                        "stable_id": sid,
                        "action": "reused",
                        "hasSensorID": None,
                        "local_id": None,
                        "role": role,
                        "target_name": rel.target_name,
                    }
                )
    return out


def _affected_relationships_from_plans(
    rel_plans: list[RelPlanItem],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in rel_plans:
        membership = _membership_action(rel.action)
        if membership is None or not rel.stable_id:
            continue
        out.append(
            {
                "stable_id": rel.stable_id,
                "action": membership,
                "type": rel.type,
                "source_stable_id": rel.source_stable_id,
                "target_stable_id": rel.target_stable_id,
                "target_name": rel.target_name,
                "source_hasSensorID": rel.source_hasSensorID,
            }
        )
    return out


def preflight_reviewed_candidates(
    graph: CandidateGraph,
    provenance: Provenance | None = None,
    *,
    driver=None,
    database: str | None = None,
) -> PreflightResult:
    db = database or _database_name()
    own_driver = driver is None
    if own_driver:
        driver = _get_driver()

    try:
        try:
            validate_candidate_graph(graph)
        except DynamicIngestionValidationError as exc:
            return PreflightResult(
                valid=False,
                database=db,
                rejected=[{"reason": str(exc)}],
            )

        with driver.session(database=db) as session:
            node_plans, rel_plans = plan_ingestion(graph, session, provenance)

        creates = [_node_plan_to_dict(p) for p in node_plans if p.action == "create"]
        updates = [_node_plan_to_dict(p) for p in node_plans if p.action == "update"]
        noops = [_node_plan_to_dict(p) for p in node_plans if p.action == "noop"]
        conflicts = [
            _node_plan_to_dict(p) for p in node_plans if p.action == "conflict"
        ]
        rejected = [
            _node_plan_to_dict(p) for p in node_plans if p.action == "rejected"
        ]
        rejected.extend(
            {
                "action": r.action,
                "type": r.type,
                "source_local_id": r.source_local_id,
                "target_name": r.target_name,
                "target_local_id": r.target_local_id,
                "detail": r.detail,
            }
            for r in rel_plans
            if r.action == "rejected"
        )
        relationships_to_create = [
            {
                "type": r.type,
                "source_local_id": r.source_local_id,
                "target_name": r.target_name,
                "target_local_id": r.target_local_id,
                "stable_id": r.stable_id,
                "source_stable_id": r.source_stable_id,
                "target_stable_id": r.target_stable_id,
            }
            for r in rel_plans
            if r.action == "create"
        ]
        relationships_noop = [
            {
                "type": r.type,
                "source_local_id": r.source_local_id,
                "target_name": r.target_name,
                "target_local_id": r.target_local_id,
                "stable_id": r.stable_id,
                "source_stable_id": r.source_stable_id,
                "target_stable_id": r.target_stable_id,
            }
            for r in rel_plans
            if r.action == "noop"
        ]

        valid = not conflicts and not any(p.action == "rejected" for p in node_plans)

        return PreflightResult(
            valid=valid,
            database=db,
            creates=creates,
            updates=updates,
            noops=noops,
            conflicts=conflicts,
            relationships_to_create=relationships_to_create,
            relationships_noop=relationships_noop,
            rejected=rejected,
            affected_nodes=_affected_nodes_from_plans(node_plans, rel_plans),
            affected_relationships=_affected_relationships_from_plans(rel_plans),
        )
    finally:
        if own_driver and driver is not None:
            driver.close()


def _apply_optional_props(tx, eid: str, props: dict[str, Any], match_label: str) -> None:
    optional_sets = []
    params: dict[str, Any] = {"eid": eid}
    for key in _OPTIONAL_SET_KEYS:
        if key in props and props[key] is not None:
            optional_sets.append(f"n.{key} = ${key}")
            params[key] = props[key]
    if not optional_sets:
        return
    # Static label from allowlist
    if match_label not in ALLOWED_PRIMARY_LABELS:
        raise DynamicIngestionValidationError(f"Invalid label: {match_label}")
    set_cypher = (
        f"MATCH (n:{match_label}) WHERE elementId(n) = $eid SET "
        + ", ".join(optional_sets)
    )
    assert_cypher_safe(set_cypher)
    tx.run(set_cypher, **params)


def _apply_create(tx, plan: NodePlanItem) -> str:
    props = dict(plan.properties_to_set)
    primary = plan.primary_label
    if primary == BASE_LABEL:
        position = plan.position_label
        if position not in _POSITION_SET_CYPHER:
            raise DynamicIngestionValidationError(f"Invalid position label: {position}")
        create_cypher = _CREATE_NODE_CYPHER[BASE_LABEL]
        assert_cypher_safe(create_cypher)
        record = tx.run(
            create_cypher,
            hasSensorID=plan.hasSensorID,
            uri=props["uri"],
            id=props["id"],
        ).single()
        eid = record["eid"]
        _apply_optional_props(tx, eid, props, BASE_LABEL)
        pos_cypher = _POSITION_SET_CYPHER[position]
        assert_cypher_safe(pos_cypher)
        tx.run(pos_cypher, eid=eid)
        return eid

    create_cypher = _CREATE_NODE_CYPHER.get(primary)
    if not create_cypher:
        raise DynamicIngestionValidationError(f"No create template for {primary}")
    assert_cypher_safe(create_cypher)
    record = tx.run(create_cypher, uri=props["uri"], id=props["id"]).single()
    eid = record["eid"]
    _apply_optional_props(tx, eid, props, primary)
    return eid


def _apply_update(tx, plan: NodePlanItem) -> str:
    if not plan.existing_element_id:
        raise DynamicIngestionValidationError("Update plan missing element id")
    _apply_optional_props(
        tx, plan.existing_element_id, plan.properties_to_set, plan.primary_label
    )
    if plan.primary_label == BASE_LABEL and plan.position_label:
        pos_cypher = _POSITION_SET_CYPHER.get(plan.position_label)
        if pos_cypher:
            assert_cypher_safe(pos_cypher)
            tx.run(pos_cypher, eid=plan.existing_element_id)
    return plan.existing_element_id


def _apply_relationship(tx, plan: RelPlanItem, source_eid: str) -> str:
    if not plan.target_element_id:
        raise DynamicIngestionValidationError("Relationship plan missing target element id")
    cypher = _MERGE_REL_CYPHER.get(plan.type)
    if not cypher:
        raise DynamicIngestionValidationError(
            f"Unsupported relationship type: {plan.type}"
        )
    assert_cypher_safe(cypher)
    record = tx.run(cypher, sid=source_eid, tid=plan.target_element_id).single()
    if not record:
        raise DynamicIngestionConflictError("Relationship MERGE returned no row")
    return record["rid"]


def insert_reviewed_candidates(
    graph: CandidateGraph,
    provenance: Provenance | None = None,
    *,
    confirm_write_to_production: bool = False,
    driver=None,
    database: str | None = None,
) -> InsertResult:
    db = database or _database_name()
    if not confirm_write_to_production:
        raise PermissionError(
            "Refusing write: pass confirm_write_to_production=True after explicit approval"
        )

    if provenance is None:
        provenance = Provenance(
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
    elif provenance.ingested_at is None:
        provenance = Provenance(
            source_filename=provenance.source_filename,
            rphd_file_id=provenance.rphd_file_id,
            source_hash=provenance.source_hash,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )

    own_driver = driver is None
    if own_driver:
        driver = _get_driver()

    try:
        preflight = preflight_reviewed_candidates(
            graph, provenance, driver=driver, database=db
        )
        if not preflight.valid:
            return InsertResult(
                success=False,
                database=db,
                error="Preflight invalid: conflicts or rejected items",
                preflight=preflight.to_dict(),
            )

        with driver.session(database=db) as session:
            node_plans, rel_plans = plan_ingestion(graph, session, provenance)
            if any(p.action in ("conflict", "rejected") for p in node_plans):
                return InsertResult(
                    success=False,
                    database=db,
                    error="Plan contains conflicts or rejected nodes",
                    preflight=preflight.to_dict(),
                )

            created: list[str] = []
            updated: list[str] = []
            noop: list[str] = []
            written_rels: list[str] = []

            def _work(tx):
                nonlocal created, updated, noop, written_rels
                created, updated, noop, written_rels = [], [], [], []
                eid_by_local: dict[str, str] = {}
                for plan in node_plans:
                    if plan.action == "create":
                        eid = _apply_create(tx, plan)
                        eid_by_local[plan.local_id] = eid
                        created.append(plan.ontology_id or plan.local_id)
                    elif plan.action == "update":
                        eid = _apply_update(tx, plan)
                        eid_by_local[plan.local_id] = eid
                        updated.append(plan.ontology_id or plan.local_id)
                    elif plan.action == "noop":
                        if plan.existing_element_id:
                            eid_by_local[plan.local_id] = plan.existing_element_id
                        noop.append(plan.ontology_id or plan.local_id)
                    else:
                        raise DynamicIngestionConflictError(
                            f"Unexpected action during write: {plan.action}"
                        )

                for rel in rel_plans:
                    if rel.action == "rejected":
                        continue
                    source_eid = rel.source_element_id or eid_by_local.get(
                        rel.source_local_id
                    )
                    if not source_eid:
                        raise DynamicIngestionConflictError(
                            f"Missing source element id for {rel.source_local_id}"
                        )
                    target_eid = rel.target_element_id or (
                        eid_by_local.get(rel.target_local_id)
                        if rel.target_local_id
                        else None
                    )
                    if not target_eid:
                        raise DynamicIngestionConflictError(
                            f"Missing target element id for {rel.target_name}"
                        )
                    # Refresh target eid on plan for MERGE
                    rel.target_element_id = target_eid
                    if rel.action == "create":
                        rid = _apply_relationship(tx, rel, source_eid)
                        written_rels.append(rel.stable_id or rid)
                    elif rel.action == "noop":
                        written_rels.append(
                            rel.stable_id or rel.existing_rel_element_id or ""
                        )

            try:
                session.execute_write(_work)
            except Exception as exc:  # noqa: BLE001
                return InsertResult(
                    success=False,
                    database=db,
                    error=f"Write failed and was rolled back: {type(exc).__name__}",
                    preflight=preflight.to_dict(),
                )

        affected_nodes = _affected_nodes_from_plans(node_plans, rel_plans)
        affected_relationships = _affected_relationships_from_plans(rel_plans)

        return InsertResult(
            success=True,
            database=db,
            created=created,
            updated=updated,
            noop=noop,
            preflight=preflight.to_dict(),
            affected_nodes=affected_nodes,
            affected_relationships=affected_relationships,
        )
    finally:
        if own_driver and driver is not None:
            driver.close()


def lookup_sensors_readonly(
    has_sensor_ids: list[str],
    *,
    driver=None,
    database: str | None = None,
) -> dict[str, dict[str, Any]]:
    """READ-ONLY existence check for hasSensorID values."""
    db = database or _database_name()
    own_driver = driver is None
    if own_driver:
        driver = _get_driver()
    out: dict[str, dict[str, Any]] = {}
    try:
        with driver.session(database=db) as session:
            for sid in has_sensor_ids:
                matches = _fetch_sensors_by_id(session, sid)
                out[sid] = {
                    "count": len(matches),
                    "status": (
                        "absent"
                        if len(matches) == 0
                        else "exists"
                        if len(matches) == 1
                        else "ambiguous"
                    ),
                    "labels": [m["labels"] for m in matches],
                }
        return out
    finally:
        if own_driver and driver is not None:
            driver.close()
