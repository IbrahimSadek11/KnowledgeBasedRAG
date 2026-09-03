"""
Dynamic KG â€” ontology-constrained candidate extraction (full V9 concrete types).

GPT-4o-mini Structured Outputs â†’ CandidateGraph. No Neo4j writes.
"""

from __future__ import annotations

import json
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from backend.config import OPENAI_API_KEY
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

PDF_PATH = Path(__file__).parent / "data" / "VISIR_Dressage_5_Sensor_Summary.pdf"

# ---------------------------------------------------------------------------
# Concrete V9 runtime labels only (no abstract Event / Human / Sensor / â€¦)
# ---------------------------------------------------------------------------

ConcreteLabel = Literal[
    "Horse",
    "Rider",
    "Veterinarian",
    "Caretaker",
    "ShowJumping",
    "Dressage",
    "Cross",
    "EventParticipation",
    "PreparationStage",
    "PreCompetitionStage",
    "CompetitionStage",
    "TransitionStage",
    "InertialSensors",
    "Withers",
    "Sternum",
    "CanonOfForelimb",
    "CanonOfHindlimb",
    "ExperimentalObjective",
    "CompetitiveSeason",
]

RelationshipType = Literal[
    "ASSOCIATEDWITH",
    "COMPETESIN",
    "TRAINSIN",
    "DEPENDSON",
    "INVOLVESACTOR",
    "HASPARTICIPATION",
    "HASHORSE",
    "HASRIDER",
    "INSEASON",
    "ISATTACHEDTO",
    "ISUSEDFOR",
]

OntologyLimitation = Literal[
    "unsupported by current ontology",
    "entity identity not stated in source text",
    "relationship target identity not stated in source text",
    "laterality not represented in current ontology",
    "placeholder or inferred identity is forbidden",
]

POSITION_LABELS: frozenset[str] = frozenset(
    ("Withers", "Sternum", "CanonOfForelimb", "CanonOfHindlimb")
)
SENSOR_FAMILY_LABELS: frozenset[str] = frozenset(
    ("InertialSensors", *POSITION_LABELS)
)
PERSON_LABELS: frozenset[str] = frozenset(("Rider", "Veterinarian", "Caretaker"))
EVENT_LABELS: frozenset[str] = frozenset(("ShowJumping", "Dressage", "Cross"))
STAGE_LABELS: frozenset[str] = frozenset(
    (
        "PreparationStage",
        "PreCompetitionStage",
        "CompetitionStage",
        "TransitionStage",
    )
)
SINGLE_LABEL_TYPES: frozenset[str] = frozenset(
    {
        "Horse",
        "Rider",
        "Veterinarian",
        "Caretaker",
        "ShowJumping",
        "Dressage",
        "Cross",
        "EventParticipation",
        "PreparationStage",
        "PreCompetitionStage",
        "CompetitionStage",
        "TransitionStage",
        "ExperimentalObjective",
        "CompetitiveSeason",
    }
)

# Relationship type â†’ allowed source / target primary labels
REL_ENDPOINT_LABELS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "ASSOCIATEDWITH": (frozenset({"Rider"}), frozenset({"Horse"})),
    "COMPETESIN": (frozenset({"Horse"}), EVENT_LABELS),
    "TRAINSIN": (frozenset({"Horse"}), STAGE_LABELS),
    "DEPENDSON": (STAGE_LABELS, EVENT_LABELS),
    "INVOLVESACTOR": (STAGE_LABELS, PERSON_LABELS),
    "HASPARTICIPATION": (EVENT_LABELS, frozenset({"EventParticipation"})),
    "HASHORSE": (frozenset({"EventParticipation"}), frozenset({"Horse"})),
    "HASRIDER": (frozenset({"EventParticipation"}), frozenset({"Rider"})),
    "INSEASON": (EVENT_LABELS, frozenset({"CompetitiveSeason"})),
    "ISATTACHEDTO": (frozenset({"InertialSensors"}), frozenset({"Horse"})),
    "ISUSEDFOR": (
        frozenset({"InertialSensors"}),
        frozenset({"ExperimentalObjective"}),
    ),
}

PROPERTIES_BY_PRIMARY: dict[str, frozenset[str]] = {
    "Horse": frozenset({"id", "hasName", "hasRace"}),
    "Rider": frozenset({"id", "hasName"}),
    "Veterinarian": frozenset({"id", "hasName"}),
    "Caretaker": frozenset({"id", "hasName"}),
    "ShowJumping": frozenset({"id", "category", "eventDate", "eventLocation"}),
    "Dressage": frozenset({"id", "category", "eventDate", "eventLocation"}),
    "Cross": frozenset({"id", "category", "eventDate", "eventLocation"}),
    "EventParticipation": frozenset({"id", "rank", "status"}),
    "PreparationStage": frozenset({"id", "Volume", "Intensity", "Frequency"}),
    "PreCompetitionStage": frozenset({"id", "Volume", "Intensity", "Frequency"}),
    "CompetitionStage": frozenset({"id", "Volume", "Intensity", "Frequency"}),
    "TransitionStage": frozenset({"id", "Volume", "Intensity", "Frequency"}),
    "InertialSensors": frozenset(
        {
            "id",
            "hasSensorID",
            "hasFormat",
            "hasSensorOffset",
            "hasFileSize",
            "hasSensorTime",
        }
    ),
    "ExperimentalObjective": frozenset({"id", "hasName", "description"}),
    "CompetitiveSeason": frozenset({"id", "seasonName", "seasonStart", "seasonEnd"}),
}

SYSTEM_MESSAGE = """You extract ontology-constrained knowledge-graph candidates from one PDF.

Return only facts that fit the fixed Horse V9 ontology already used at runtime. Do not extend it.
Do not invent abstract labels: never use Event, Sensor, TrainingStage, SportingEvent, or Human.
Use only the concrete labels listed below.

SOURCE FAITHFULNESS (critical):
- Emit a node or relationship ONLY when the PDF text explicitly supports it.
- Do not invent unnamed horses, riders, events, vets, caretakers, ranks, or objectives.
- Do not invent relationships that are not stated.
- RDF-style dump PDFs often contain explicit NODE / REL lines: preserve those facts using the stated ids and names.
- When the text contains lines like `REL | source=... | type=... | target=...`, emit ONE
  relationship for EVERY such line. Do not skip TRAINSIN, DEPENDSON, INVOLVESACTOR,
  ISUSEDFOR, COMPETESIN, INSEASON, HASPARTICIPATION, HASHORSE, HASRIDER, ASSOCIATEDWITH,
  ISATTACHEDTO, or any other allowlisted type present in the text.
- Use the exact `source=` token as source_local_id and the exact `target=` token as
  target_local_id (ontology ids are valid local ids). Endpoints do NOT need to appear
  as nodes in the same response â€” a later merge step resolves them.
- An intentionally sparse graph is valid when the document only covers sensors.

Allowed concrete node labels:
People: Horse, Rider, Veterinarian, Caretaker
Events: ShowJumping, Dressage, Cross
Participation: EventParticipation
Stages: PreparationStage, PreCompetitionStage, CompetitionStage, TransitionStage
Sensors: InertialSensors + exactly one of Withers, Sternum, CanonOfForelimb, CanonOfHindlimb
Other: ExperimentalObjective, CompetitiveSeason

Sensor nodes MUST be dual-labeled: ["InertialSensors", "<position>"].
All other nodes MUST have exactly one concrete label (not a position label alone).

Allowed properties (null when not stated; never fabricate):
- Horse: id, hasName, hasRace
- Rider / Veterinarian / Caretaker: id, hasName
- ShowJumping / Dressage / Cross: id, category, eventDate, eventLocation
- EventParticipation: id, rank, status
- Stages: id, Volume, Intensity, Frequency
- InertialSensors: id, hasSensorID (required), hasFormat, hasSensorOffset, hasFileSize, hasSensorTime
- ExperimentalObjective: id, hasName, description
- CompetitiveSeason: id, seasonName, seasonStart, seasonEnd

Sensor property conventions:
- hasSensorID is the MERGE key from the text (e.g. IMU-CH-005 or 6845).
- hasFormat is a short format code such as CSV when stated â€” not a filename.
- hasSensorTime must already match integerHz (e.g. 200Hz) in the PDF; do not rewrite numbers.

Allowed relationships and directions (no others):
- (Rider)-[:ASSOCIATEDWITH]->(Horse)
- (Horse)-[:COMPETESIN]->(ShowJumping|Dressage|Cross)
- (Horse)-[:TRAINSIN]->(PreparationStage|PreCompetitionStage|CompetitionStage|TransitionStage)
- (stage)-[:DEPENDSON]->(ShowJumping|Dressage|Cross)
- (stage)-[:INVOLVESACTOR]->(Rider|Veterinarian|Caretaker)
- (event)-[:HASPARTICIPATION]->(EventParticipation)
- (EventParticipation)-[:HASHORSE]->(Horse)
- (EventParticipation)-[:HASRIDER]->(Rider)
- (event)-[:INSEASON]->(CompetitiveSeason)
- (InertialSensors)-[:ISATTACHEDTO]->(Horse)
- (InertialSensors)-[:ISUSEDFOR]->(ExperimentalObjective)

Relationship endpoints:
- Prefer target_local_id when the target is also a candidate node in this extraction.
- Use target_name (exact name/id from text) when the target is named but not emitted as a node,
  especially for sensorâ†’Horse / sensorâ†’ExperimentalObjective links.
- At least one of target_local_id or target_name must be set.

VISIR limb laterality (left/right) is not in the ontology: map limbs to CanonOfForelimb /
CanonOfHindlimb and put left/right into rejected_facts.

local_id is extraction-local only (not Neo4j uri). Prefer stable keys like horse_Dakota, rider_Antoine, sensor_IMU-CH-005.
When the PDF states an ontology id (Rider_Antoine, Horse_Dakota, Vet_DrMartin), put it in props as name=id.
Emit ONLY stated properties in props[] â€” never emit null placeholders.
Keep source_evidence short (one line, <=160 characters).
"""


PropertyName = Literal[
    "id",
    "hasName",
    "hasRace",
    "hasSensorID",
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
]

_INT_PROP_NAMES = frozenset({"hasFileSize", "rank"})


class NodeProperties(BaseModel):
    """Union of allowlisted V9 datatype properties. Null = not stated."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, description="Ontology id when stated (e.g. Rider_Antoine).")
    hasName: str | None = None
    hasRace: str | None = None
    hasSensorID: str | None = None
    hasFormat: str | None = None
    hasSensorOffset: str | None = None
    hasFileSize: int | None = None
    hasSensorTime: str | None = None
    category: str | None = None
    eventDate: str | None = None
    eventLocation: str | None = None
    rank: int | None = None
    status: str | None = None
    Volume: str | None = None
    Intensity: str | None = None
    Frequency: str | None = None
    description: str | None = None
    seasonName: str | None = None
    seasonStart: str | None = None
    seasonEnd: str | None = None


# Backward-compatible name used by older tests / imports.
SensorProperties = NodeProperties


class StatedProperty(BaseModel):
    """Compact property bag for LLM structured output (stated values only)."""

    model_config = ConfigDict(extra="forbid")

    name: PropertyName
    value: str = Field(description="Property value as written in the PDF (numbers as digits).")


class CandidateNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_id: str = Field(description="Extraction-local node id, not a Neo4j id.")
    labels: list[ConcreteLabel] = Field(
        min_length=1,
        max_length=2,
        description="One concrete label, or InertialSensors + one position label.",
    )
    properties: NodeProperties
    source_evidence: str = Field(
        description="Verbatim or near-verbatim span from the PDF that supports this node."
    )

    @field_validator("labels")
    @classmethod
    def validate_label_combo(cls, labels: list[str]) -> list[str]:
        if not labels:
            raise ValueError("labels must not be empty")
        unknown = [lab for lab in labels if lab not in SINGLE_LABEL_TYPES | SENSOR_FAMILY_LABELS]
        if unknown:
            raise ValueError(f"unsupported labels: {unknown}")
        if "InertialSensors" in labels:
            if len(labels) != 2:
                raise ValueError(
                    "sensor nodes must have exactly two labels: "
                    "InertialSensors + one position"
                )
            if labels.count("InertialSensors") != 1:
                raise ValueError("InertialSensors must appear exactly once")
            positions = [lab for lab in labels if lab in POSITION_LABELS]
            if len(positions) != 1:
                raise ValueError("sensor nodes need exactly one position label")
            return labels
        if len(labels) != 1:
            raise ValueError(
                "non-sensor nodes must have exactly one concrete label"
            )
        if labels[0] not in SINGLE_LABEL_TYPES:
            raise ValueError(f"unsupported primary label: {labels[0]}")
        return labels

    @model_validator(mode="after")
    def validate_required_identity_props(self) -> CandidateNode:
        props = self.properties
        if "InertialSensors" in self.labels:
            sid = (props.hasSensorID or "").strip()
            oid = (props.id or "").strip()
            if not sid and not oid:
                raise ValueError(
                    f"Node {self.local_id}: sensor needs hasSensorID or id from the source"
                )
            if sid:
                props.hasSensorID = sid
            return self
        primary = self.labels[0]
        identity_keys = {
            "Horse": ("id", "hasName"),
            "Rider": ("id", "hasName"),
            "Veterinarian": ("id", "hasName"),
            "Caretaker": ("id", "hasName"),
            "ShowJumping": ("id",),
            "Dressage": ("id",),
            "Cross": ("id",),
            "EventParticipation": ("id",),
            "PreparationStage": ("id",),
            "PreCompetitionStage": ("id",),
            "CompetitionStage": ("id",),
            "TransitionStage": ("id",),
            "ExperimentalObjective": ("id", "hasName"),
            "CompetitiveSeason": ("id", "seasonName"),
        }
        keys = identity_keys.get(primary, ("id", "hasName"))
        if not any(getattr(props, k, None) for k in keys):
            raise ValueError(
                f"Node {self.local_id}: {primary} needs one of {keys} from the source"
            )
        return self


class CandidateRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RelationshipType
    source_local_id: str = Field(
        description="local_id of the source candidate node."
    )
    target_local_id: str | None = Field(
        default=None,
        description="local_id of the target candidate when both are in nodes[].",
    )
    target_name: str | None = Field(
        default=None,
        description=(
            "Explicit target name/id from the PDF when target is not a candidate "
            "node (e.g. Dakota, FatigueDetection)."
        ),
    )
    source_evidence: str = Field(
        description="Verbatim or near-verbatim span from the PDF that supports this relationship."
    )

    @model_validator(mode="after")
    def require_target_ref(self) -> CandidateRelationship:
        local = (self.target_local_id or "").strip() or None
        name = (self.target_name or "").strip() or None
        self.target_local_id = local
        self.target_name = name
        if not local and not name:
            raise ValueError(
                "Relationship requires target_local_id and/or target_name"
            )
        return self


class RejectedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_fact: str = Field(description="The source value or claim that could not be stored.")
    reason: str = Field(description="Why this fact was not turned into a node, property, or relationship.")
    ontology_limitation: OntologyLimitation
    source_evidence: str = Field(
        description="Verbatim or near-verbatim span from the PDF for this rejected fact."
    )


def primary_label(labels: list[str]) -> str:
    if "InertialSensors" in labels:
        return "InertialSensors"
    return labels[0]


def _labels_match_endpoint(node_labels: list[str], allowed: frozenset[str]) -> bool:
    if "InertialSensors" in node_labels and "InertialSensors" in allowed:
        return True
    return any(lab in allowed for lab in node_labels)


class CandidateGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[CandidateNode]
    relationships: list[CandidateRelationship]
    rejected_facts: list[RejectedFact]

    @model_validator(mode="after")
    def enforce_relationship_endpoints_and_directions(self) -> CandidateGraph:
        """
        Soft validation: keep relationships whose endpoints are missing from
        this partial graph so document-level merge can resolve them later.
        Only drop structurally illegal types / wrong-direction when BOTH ends
        are present in nodes[].
        """
        nodes_by_id = {node.local_id: node for node in self.nodes}
        kept: list[CandidateRelationship] = []
        rejected = list(self.rejected_facts)

        for rel in self.relationships:
            target_ref = rel.target_local_id or rel.target_name or "?"
            fact = f"{rel.type} {rel.source_local_id}->{target_ref}"

            if rel.type not in REL_ENDPOINT_LABELS:
                rejected.append(
                    RejectedFact(
                        source_fact=fact,
                        reason=f"Relationship type {rel.type} is not allowlisted.",
                        ontology_limitation="unsupported by current ontology",
                        source_evidence=rel.source_evidence,
                    )
                )
                continue

            source = nodes_by_id.get(rel.source_local_id)
            target = (
                nodes_by_id.get(rel.target_local_id)
                if rel.target_local_id
                else None
            )
            src_allowed, tgt_allowed = REL_ENDPOINT_LABELS[rel.type]

            # Defer if endpoint not in this partial node set.
            if source is None or (rel.target_local_id and target is None):
                kept.append(rel)
                continue

            if not _labels_match_endpoint(source.labels, src_allowed):
                rejected.append(
                    RejectedFact(
                        source_fact=fact,
                        reason=(
                            f"{rel.type} source must be one of {sorted(src_allowed)}; "
                            f"got {source.labels}."
                        ),
                        ontology_limitation="unsupported by current ontology",
                        source_evidence=rel.source_evidence,
                    )
                )
                continue
            if target is not None and not _labels_match_endpoint(
                target.labels, tgt_allowed
            ):
                rejected.append(
                    RejectedFact(
                        source_fact=fact,
                        reason=(
                            f"{rel.type} target must be one of {sorted(tgt_allowed)}; "
                            f"got {target.labels}."
                        ),
                        ontology_limitation="unsupported by current ontology",
                        source_evidence=rel.source_evidence,
                    )
                )
                continue
            kept.append(rel)

        self.relationships = kept
        self.rejected_facts = rejected
        return self


class LlmCandidateNode(BaseModel):
    """LLM-facing compact node (props list avoids null-property token bloat)."""

    model_config = ConfigDict(extra="forbid")

    local_id: str
    labels: list[ConcreteLabel] = Field(min_length=1, max_length=2)
    props: list[StatedProperty] = Field(
        default_factory=list,
        description="Only properties explicitly stated in the PDF.",
    )
    source_evidence: str


class LlmCandidateGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[LlmCandidateNode]
    relationships: list[CandidateRelationship]
    rejected_facts: list[RejectedFact]


def stated_props_to_node_properties(
    props: list[StatedProperty],
    *,
    local_id: str | None = None,
    labels: list[str] | None = None,
) -> NodeProperties:
    data: dict[str, object] = {}
    for item in props:
        raw = (item.value or "").strip()
        if not raw:
            continue
        if item.name in _INT_PROP_NAMES:
            try:
                data[item.name] = int(raw)
            except ValueError:
                continue
        else:
            data[item.name] = raw

    # RDF-dump extractors often set local_id to the ontology id and omit props.id.
    # Promote local_id → id when it is a stable ontology-style token.
    if data.get("id") is None and local_id and "_" in local_id and " " not in local_id:
        data["id"] = local_id.strip()

    labels = labels or []
    if "InertialSensors" in labels:
        if data.get("hasSensorID") is None and local_id:
            # Accept user-facing sensor ids used as local_id (e.g. IMU-ST-004).
            token = local_id.strip()
            if "-" in token and " " not in token:
                data["hasSensorID"] = token

    return NodeProperties(**data)


def llm_graph_to_candidate_graph(llm_graph: LlmCandidateGraph) -> CandidateGraph:
    nodes: list[CandidateNode] = []
    conversion_rejects: list[RejectedFact] = []
    for node in llm_graph.nodes:
        try:
            nodes.append(
                CandidateNode(
                    local_id=node.local_id,
                    labels=node.labels,
                    properties=stated_props_to_node_properties(
                        node.props,
                        local_id=node.local_id,
                        labels=list(node.labels),
                    ),
                    source_evidence=node.source_evidence,
                )
            )
        except Exception as exc:  # noqa: BLE001
            conversion_rejects.append(
                RejectedFact(
                    source_fact=node.local_id,
                    reason=f"Node conversion failed: {exc}",
                    ontology_limitation="entity identity not stated in source text",
                    source_evidence=node.source_evidence,
                )
            )
    relationships: list[CandidateRelationship] = []
    for rel in llm_graph.relationships:
        try:
            relationships.append(
                CandidateRelationship(
                    type=rel.type,
                    source_local_id=rel.source_local_id,
                    target_local_id=rel.target_local_id,
                    target_name=rel.target_name,
                    source_evidence=rel.source_evidence,
                )
            )
        except Exception as exc:  # noqa: BLE001
            conversion_rejects.append(
                RejectedFact(
                    source_fact=f"{rel.type} {rel.source_local_id}",
                    reason=f"Relationship conversion failed: {exc}",
                    ontology_limitation="unsupported by current ontology",
                    source_evidence=rel.source_evidence,
                )
            )
    return CandidateGraph.model_construct(
        nodes=nodes,
        relationships=relationships,
        rejected_facts=list(llm_graph.rejected_facts) + conversion_rejects,
    )


def _normalize_token(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _all_identity_keys(node: CandidateNode) -> list[str]:
    """Deterministic multi-alias keys so overlapping chunks collapse correctly."""
    primary = primary_label(list(node.labels))
    props = node.properties
    keys: list[str] = []
    if primary == "InertialSensors":
        if props.hasSensorID:
            keys.append(f"sensor:{_normalize_token(props.hasSensorID)}")
        if props.id:
            keys.append(f"sensor_ont:{_normalize_token(props.id)}")
        keys.append(f"local:{_normalize_token(node.local_id)}")
        return list(dict.fromkeys(keys))
    if props.id:
        keys.append(f"{primary}:id:{_normalize_token(props.id)}")
    if props.hasName:
        keys.append(f"{primary}:name:{_normalize_token(props.hasName)}")
    if props.seasonName:
        keys.append(f"{primary}:season:{_normalize_token(props.seasonName)}")
    keys.append(f"local:{_normalize_token(node.local_id)}")
    if props.id:
        keys.append(f"any_id:{_normalize_token(props.id)}")
    return list(dict.fromkeys(k for k in keys if not k.endswith(":")))


def _node_identity_key(node: CandidateNode) -> str:
    keys = _all_identity_keys(node)
    return keys[0] if keys else f"local:{node.local_id}"


def _merge_properties(canonical: CandidateNode, incoming: CandidateNode) -> None:
    """Fill nulls on canonical from incoming; never overwrite non-null values."""
    cprops = canonical.properties
    iprops = incoming.properties
    for field_name in NodeProperties.model_fields:
        cv = getattr(cprops, field_name)
        iv = getattr(iprops, field_name)
        if cv is None and iv is not None:
            setattr(cprops, field_name, iv)


def _page_kind(text: str) -> str:
    has_node = bool(re.search(r"(?i)\bNODE\s*\|", text))
    has_rel = bool(re.search(r"(?i)\bREL\s*\|", text))
    if has_rel and not has_node:
        return "rel"
    if has_node:
        return "node"
    return "other"


class NodeRegistry:
    """Document-level node index for cross-chunk relationship resolution."""

    def __init__(self) -> None:
        self.nodes: list[CandidateNode] = []
        self._by_key: dict[str, CandidateNode] = {}

    def _index(self, canonical: CandidateNode) -> None:
        for key in _all_identity_keys(canonical):
            self._by_key[key] = canonical
        self._by_key[f"local:{_normalize_token(canonical.local_id)}"] = canonical
        if canonical.properties.id:
            oid = _normalize_token(canonical.properties.id)
            self._by_key[f"any_id:{oid}"] = canonical
            self._by_key[f"local:{oid}"] = canonical
        if canonical.properties.hasSensorID:
            self._by_key[
                f"sensor:{_normalize_token(canonical.properties.hasSensorID)}"
            ] = canonical
        if canonical.properties.hasName:
            self._by_key[
                f"{primary_label(list(canonical.labels))}:"
                f"name:{_normalize_token(canonical.properties.hasName)}"
            ] = canonical

    def add(self, node: CandidateNode) -> CandidateNode:
        keys = _all_identity_keys(node)
        existing: CandidateNode | None = None
        for key in keys:
            if key in self._by_key:
                existing = self._by_key[key]
                break
        if existing is None:
            self.nodes.append(node)
            canonical = node
        else:
            _merge_properties(existing, node)
            if existing.properties.id and node.local_id == existing.properties.id:
                existing.local_id = node.local_id
            canonical = existing
        self._index(canonical)
        return canonical

    def resolve(self, token: str | None) -> CandidateNode | None:
        if not token or not str(token).strip():
            return None
        raw = str(token).strip()
        candidates = [
            f"local:{_normalize_token(raw)}",
            f"any_id:{_normalize_token(raw)}",
            f"sensor:{_normalize_token(raw)}",
            f"sensor_ont:{_normalize_token(raw)}",
        ]
        for primary in SINGLE_LABEL_TYPES | {"InertialSensors"}:
            candidates.append(f"{primary}:id:{_normalize_token(raw)}")
            candidates.append(f"{primary}:name:{_normalize_token(raw)}")
        for key in candidates:
            hit = self._by_key.get(key)
            if hit is not None:
                return hit
        return None


def merge_candidate_nodes(graphs: list[CandidateGraph]) -> NodeRegistry:
    registry = NodeRegistry()
    for graph in graphs:
        for node in graph.nodes:
            registry.add(node)
    return registry


def resolve_relationships_against_registry(
    relationships: list[CandidateRelationship],
    registry: NodeRegistry,
) -> tuple[list[CandidateRelationship], list[RejectedFact]]:
    """PASS 2: bind relationship endpoints to the document-level node registry."""
    kept: list[CandidateRelationship] = []
    rejected: list[RejectedFact] = []
    seen: set[tuple[str, str, str]] = set()

    for rel in relationships:
        target_ref = rel.target_local_id or rel.target_name or "?"
        fact = f"{rel.type} {rel.source_local_id}->{target_ref}"

        if rel.type not in REL_ENDPOINT_LABELS:
            rejected.append(
                RejectedFact(
                    source_fact=fact,
                    reason=f"Relationship type {rel.type} is not allowlisted.",
                    ontology_limitation="unsupported by current ontology",
                    source_evidence=rel.source_evidence,
                )
            )
            continue

        source = registry.resolve(rel.source_local_id)
        target: CandidateNode | None = None
        if rel.target_local_id:
            target = registry.resolve(rel.target_local_id)
        if target is None and rel.target_name:
            target = registry.resolve(rel.target_name)

        if source is None or target is None:
            rejected.append(
                RejectedFact(
                    source_fact=fact,
                    reason=(
                        "Unresolved cross-chunk relationship endpoint(s): "
                        f"source={'ok' if source else 'missing'}, "
                        f"target={'ok' if target else 'missing'}."
                    ),
                    ontology_limitation="relationship target identity not stated in source text",
                    source_evidence=rel.source_evidence,
                )
            )
            continue

        src_allowed, tgt_allowed = REL_ENDPOINT_LABELS[rel.type]
        if not _labels_match_endpoint(source.labels, src_allowed):
            rejected.append(
                RejectedFact(
                    source_fact=fact,
                    reason=(
                        f"{rel.type} source must be one of {sorted(src_allowed)}; "
                        f"got {source.labels}."
                    ),
                    ontology_limitation="unsupported by current ontology",
                    source_evidence=rel.source_evidence,
                )
            )
            continue
        if not _labels_match_endpoint(target.labels, tgt_allowed):
            rejected.append(
                RejectedFact(
                    source_fact=fact,
                    reason=(
                        f"{rel.type} target must be one of {sorted(tgt_allowed)}; "
                        f"got {target.labels}."
                    ),
                    ontology_limitation="unsupported by current ontology",
                    source_evidence=rel.source_evidence,
                )
            )
            continue

        sig = (rel.type, source.local_id, target.local_id)
        if sig in seen:
            continue
        seen.add(sig)
        kept.append(
            CandidateRelationship(
                type=rel.type,
                source_local_id=source.local_id,
                target_local_id=target.local_id,
                target_name=None,
                source_evidence=rel.source_evidence,
            )
        )
    return kept, rejected


def merge_candidate_graphs(graphs: list[CandidateGraph]) -> CandidateGraph:
    """
    Two-pass document merge:
    1) union nodes into a registry (dedupe by deterministic identity aliases)
    2) resolve all relationships against that registry
    """
    rejected: list[RejectedFact] = []
    for graph in graphs:
        rejected.extend(graph.rejected_facts)

    registry = merge_candidate_nodes(graphs)
    all_rels: list[CandidateRelationship] = []
    for graph in graphs:
        all_rels.extend(graph.relationships)

    kept, rel_rejects = resolve_relationships_against_registry(all_rels, registry)
    rejected.extend(rel_rejects)

    final_nodes: list[CandidateNode] = []
    for node in registry.nodes:
        if "InertialSensors" in node.labels and not node.properties.hasSensorID:
            rejected.append(
                RejectedFact(
                    source_fact=node.local_id,
                    reason=(
                        "Sensor candidate lacked hasSensorID after document merge."
                    ),
                    ontology_limitation="entity identity not stated in source text",
                    source_evidence=node.source_evidence,
                )
            )
            continue
        final_nodes.append(node)

    return CandidateGraph.model_construct(
        nodes=final_nodes,
        relationships=kept,
        rejected_facts=rejected,
    )


REL_PASS_INSTRUCTIONS = """This chunk is relationship-focused.
Emit a relationship for EVERY explicit REL record in the text
(REL | source=... | type=... | target=...).
Cover all allowlisted types present, including TRAINSIN, DEPENDSON,
INVOLVESACTOR, ISUSEDFOR, COMPETESIN, INSEASON, HASPARTICIPATION, HASHORSE,
HASRIDER, ASSOCIATEDWITH, ISATTACHEDTO.
Set source_local_id and target_local_id to the exact source=/target= tokens.
nodes[] should be empty unless this chunk also contains NODE lines.
Do not invent nodes or relationships.
"""


NODE_PASS_INSTRUCTIONS = """This chunk is node-focused.
Emit EVERY NODE entity stated in the text (Horse, Rider, Veterinarian, Caretaker,
ShowJumping, Dressage, Cross, EventParticipation, all training-stage labels,
InertialSensors, ExperimentalObjective, CompetitiveSeason).
Prefer ontology id as local_id when present, AND always also emit props name=id
with that same ontology id value.
Emit relationships only when REL lines appear in this same chunk.
Do not invent entities.
"""


GENERIC_CHUNK_INSTRUCTIONS = """This PDF text may be prose or diagram text without
literal NODE | / REL | lines. Extract every ontology-supported fact that is
explicitly stated.

Emit nodes for horses, riders, vets, caretakers, events, EventParticipation,
training stages, sensors, objectives, and seasons when present.

For every EventParticipation that names a horse and/or rider, emit:
- HASHORSE from the participation to the horse
- HASRIDER from the participation to the rider
when those links are stated (including rank/status on the participation).

Also emit stated relationships of types:
TRAINSIN, COMPETESIN, HASPARTICIPATION, DEPENDSON, INVOLVESACTOR,
ISATTACHEDTO, ISUSEDFOR, ASSOCIATEDWITH, INSEASON.
Do not invent identities or relationships that are not in the text.
"""


def extract_pdf_text_from_reader(reader: PdfReader) -> tuple[str, int]:
    """Return (full_text, page_count) from an open PdfReader."""
    text = "".join((page.extract_text() or "") for page in reader.pages)
    return text, len(reader.pages)


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text, _pages = extract_pdf_text_from_reader(reader)
    return text


def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> tuple[str, int]:
    """Return (full_text, page_count) from in-memory PDF bytes. No disk write."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return extract_pdf_text_from_reader(reader)


def _pdf_page_texts(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    return [(page.extract_text() or "") for page in reader.pages]


def build_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Load it via .env / backend.config."
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=16384,
        openai_api_key=OPENAI_API_KEY,
    )


def _invoke_llm_candidate_graph(
    pdf_text: str, *, extra_instructions: str = ""
) -> CandidateGraph:
    llm = build_llm().with_structured_output(
        LlmCandidateGraph,
        method="json_schema",
        strict=True,
    )
    user_message = (
        "Extract ontology-constrained graph candidates from this PDF text.\n"
        "Emit ONLY stated props in each node's props list (no null placeholders).\n"
        f"{extra_instructions}\n"
        "--- PDF TEXT ---\n"
        f"{pdf_text}\n"
        "--- END PDF TEXT ---"
    )
    result = llm.invoke(
        [
            SystemMessage(content=SYSTEM_MESSAGE),
            HumanMessage(content=user_message),
        ]
    )
    if not isinstance(result, LlmCandidateGraph):
        raise TypeError(f"Expected LlmCandidateGraph, got {type(result)!r}")
    return llm_graph_to_candidate_graph(result)


def extract_candidates(pdf_text: str) -> CandidateGraph:
    graph = _invoke_llm_candidate_graph(pdf_text)
    return apply_source_grounded_property_guards(graph, pdf_text)


_LAST_EXTRACTION_STATS: dict[str, int | str] = {
    "llm_calls": 0,
    "mode": "none",
}


def get_last_extraction_stats() -> dict[str, int | str]:
    """Diagnostics from the most recent extract_candidates_* call."""
    return dict(_LAST_EXTRACTION_STATS)


def extract_candidates_chunked(
    page_texts: list[str], *, pages_per_chunk: int = 2
) -> CandidateGraph:
    """
    Two-pass extraction for large PDFs:
    PASS 1 — merge all nodes from node-bearing page chunks
    PASS 2 — extract relationships from REL pages (one page at a time) and
             resolve endpoints against the document-level node registry

    When the PDF has no literal NODE | / REL | markers (diagram / prose PDFs),
    fall back to generic multi-page LLM chunking instead of returning empty
    without any model calls.
    """
    global _LAST_EXTRACTION_STATS
    llm_calls = 0

    node_page_indexes: list[int] = []
    rel_page_indexes: list[int] = []
    for i, text in enumerate(page_texts):
        kind = _page_kind(text)
        if kind == "rel":
            rel_page_indexes.append(i)
        elif kind == "node":
            node_page_indexes.append(i)
        # Skip intro/outro "other" pages as standalone NODE/REL chunks (they
        # cause hallucinated nodes). They may still appear as adjacent context.
        # If *no* NODE/REL pages exist, the generic fallback below uses all pages.

    node_indexes = sorted(node_page_indexes)
    node_graphs: list[CandidateGraph] = []
    for i in range(0, len(node_indexes), pages_per_chunk):
        idxs = node_indexes[i : i + pages_per_chunk]
        # Optionally include immediate neighboring non-rel pages as context only
        context_idxs = set(idxs)
        for j in list(idxs):
            for neigh in (j - 1, j + 1):
                if 0 <= neigh < len(page_texts) and _page_kind(page_texts[neigh]) == "other":
                    context_idxs.add(neigh)
        ordered = sorted(context_idxs)
        chunk = "\n".join(page_texts[j] for j in ordered).strip()
        if not chunk:
            continue
        node_graphs.append(
            _invoke_llm_candidate_graph(
                chunk, extra_instructions=NODE_PASS_INSTRUCTIONS
            )
        )
        llm_calls += 1

    if not node_graphs and not rel_page_indexes:
        # Unmarked multi-page PDF: still run LLM extraction on page chunks.
        generic_graphs: list[CandidateGraph] = []
        for i in range(0, len(page_texts), pages_per_chunk):
            chunk = "\n".join(page_texts[i : i + pages_per_chunk]).strip()
            if not chunk:
                continue
            generic_graphs.append(
                _invoke_llm_candidate_graph(
                    chunk, extra_instructions=GENERIC_CHUNK_INSTRUCTIONS
                )
            )
            llm_calls += 1
        _LAST_EXTRACTION_STATS = {
            "llm_calls": llm_calls,
            "mode": "generic-chunked",
        }
        if not generic_graphs:
            return CandidateGraph(nodes=[], relationships=[], rejected_facts=[])
        merged = merge_candidate_graphs(generic_graphs)
        return apply_source_grounded_property_guards(merged, "\n".join(page_texts))

    rel_graphs: list[CandidateGraph] = []
    for idx in rel_page_indexes:
        chunk = page_texts[idx].strip()
        if not chunk:
            continue
        rel_graphs.append(
            _invoke_llm_candidate_graph(
                chunk, extra_instructions=REL_PASS_INSTRUCTIONS
            )
        )
        llm_calls += 1

    _LAST_EXTRACTION_STATS = {
        "llm_calls": llm_calls,
        "mode": "node-rel-chunked",
    }
    merged = merge_candidate_graphs(node_graphs + rel_graphs)
    return apply_source_grounded_property_guards(merged, "\n".join(page_texts))


def apply_source_grounded_property_guards(
    graph: CandidateGraph, pdf_text: str
) -> CandidateGraph:
    """
    Deterministic sensor property grounding (format / offset / sampling literal).
    Does not invent labels or relationships. Identity fields (id, hasName,
    hasSensorID) are left to the LLM + CandidateNode validators.
    """
    rejected = list(graph.rejected_facts)

    for node in graph.nodes:
        if "InertialSensors" not in node.labels:
            continue
        props = node.properties

        if props.hasFormat is not None and props.hasFormat not in pdf_text:
            rejected.append(
                RejectedFact(
                    source_fact=props.hasFormat,
                    reason=(
                        "hasFormat dropped because the exact format code does not "
                        "appear in the PDF text. Filenames/extensions are not format codes."
                    ),
                    ontology_limitation="unsupported by current ontology",
                    source_evidence=node.source_evidence,
                )
            )
            props.hasFormat = None

        if props.hasSensorOffset is not None and props.hasSensorOffset not in pdf_text:
            rejected.append(
                RejectedFact(
                    source_fact=props.hasSensorOffset,
                    reason=(
                        "hasSensorOffset dropped because the exact offset literal "
                        "does not appear in the PDF text."
                    ),
                    ontology_limitation="unsupported by current ontology",
                    source_evidence=node.source_evidence,
                )
            )
            props.hasSensorOffset = None

        if props.hasSensorTime is not None and props.hasSensorTime not in pdf_text:
            rejected.append(
                RejectedFact(
                    source_fact=props.hasSensorTime,
                    reason=(
                        "hasSensorTime dropped because the exact ontology literal "
                        f"{props.hasSensorTime!r} does not appear in the PDF text. "
                        "Do not rewrite nominal-Hz numbers into integerHz form."
                    ),
                    ontology_limitation="unsupported by current ontology",
                    source_evidence=node.source_evidence,
                )
            )
            props.hasSensorTime = None

    graph.rejected_facts = rejected
    return graph


def extract_candidates_from_pdf_bytes(
    pdf_bytes: bytes,
) -> tuple[CandidateGraph, int, int]:
    """
    Reusable API entry: PDF bytes → pypdf text → gpt-4o-mini CandidateGraph.

    Returns (candidate_graph, page_count, extracted_character_count).
    Does not write to Neo4j. Does not persist the PDF.
    Large / multi-section PDFs use two-pass page-chunked extraction.
    Call get_last_extraction_stats() for llm_calls / mode diagnostics.
    """
    global _LAST_EXTRACTION_STATS
    page_texts = _pdf_page_texts(pdf_bytes)
    page_count = len(page_texts)
    pdf_text = "".join(page_texts)
    has_rel_section = any(_page_kind(t) == "rel" for t in page_texts)
    if page_count > 3 or len(pdf_text) > 14000 or has_rel_section:
        graph = extract_candidates_chunked(page_texts, pages_per_chunk=2)
    else:
        try:
            graph = extract_candidates(pdf_text)
            _LAST_EXTRACTION_STATS = {"llm_calls": 1, "mode": "single-shot"}
        except Exception as exc:  # noqa: BLE001
            if page_count <= 1:
                raise
            name = type(exc).__name__
            if "Length" not in name and "length" not in str(exc).lower():
                raise
            graph = extract_candidates_chunked(page_texts, pages_per_chunk=2)
    return graph, page_count, len(pdf_text)


def extract_candidates_from_pdf_path(
    pdf_path: Path,
) -> tuple[CandidateGraph, int, int]:
    """Same pipeline as extract_candidates_from_pdf_bytes, from a filesystem path."""
    return extract_candidates_from_pdf_bytes(Path(pdf_path).read_bytes())


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF not found at: {PDF_PATH}")
        return

    graph, _pages, _chars = extract_candidates_from_pdf_path(PDF_PATH)
    print(json.dumps(graph.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

