"""
Unit tests for document-level chunk merge / cross-chunk relationship resolution.
No LLM calls. No Neo4j writes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_kg.extract_facts import (
    CandidateGraph,
    CandidateNode,
    CandidateRelationship,
    NodeProperties,
    NodeRegistry,
    merge_candidate_graphs,
    resolve_relationships_against_registry,
)


def _node(local_id: str, label: str, **props) -> CandidateNode:
    return CandidateNode(
        local_id=local_id,
        labels=[label],  # type: ignore[arg-type]
        properties=NodeProperties(**props),
        source_evidence=local_id,
    )


def _sensor(local_id: str, **props) -> CandidateNode:
    return CandidateNode(
        local_id=local_id,
        labels=["InertialSensors", "Sternum"],  # type: ignore[arg-type]
        properties=NodeProperties(**props),
        source_evidence=local_id,
    )


def _rel(
    rel_type: str,
    source: str,
    target: str,
) -> CandidateRelationship:
    return CandidateRelationship(
        type=rel_type,  # type: ignore[arg-type]
        source_local_id=source,
        target_local_id=target,
        source_evidence=f"{rel_type} {source}->{target}",
    )


class TestChunkMerge(unittest.TestCase):
    def test_relationship_endpoint_in_another_chunk(self):
        chunk_a = CandidateGraph.model_construct(
            nodes=[
                _node("Rider_Antoine", "Rider", id="Rider_Antoine", hasName="Antoine"),
                _node("Horse_Thunder", "Horse", id="Horse_Thunder", hasName="Thunder"),
            ],
            relationships=[],
            rejected_facts=[],
        )
        chunk_b = CandidateGraph.model_construct(
            nodes=[],
            relationships=[
                _rel("ASSOCIATEDWITH", "Rider_Antoine", "Horse_Thunder"),
            ],
            rejected_facts=[],
        )
        merged = merge_candidate_graphs([chunk_a, chunk_b])
        self.assertEqual(len(merged.nodes), 2)
        self.assertEqual(len(merged.relationships), 1)
        self.assertEqual(merged.relationships[0].type, "ASSOCIATEDWITH")

    def test_duplicate_sensor_across_overlapping_chunks(self):
        chunk_a = CandidateGraph.model_construct(
            nodes=[
                _sensor(
                    "sensor_a",
                    hasSensorID="IMU-ST-004",
                    id="IMU_Sternum_Dakota_01",
                    hasFormat="CSV",
                )
            ],
            relationships=[],
            rejected_facts=[],
        )
        # Overlapping chunk uses ontology id as local_id and omits hasSensorID initially
        chunk_b = CandidateGraph.model_construct(
            nodes=[
                _sensor(
                    "IMU_Sternum_Dakota_01",
                    id="IMU_Sternum_Dakota_01",
                    hasSensorID="IMU-ST-004",
                )
            ],
            relationships=[],
            rejected_facts=[],
        )
        merged = merge_candidate_graphs([chunk_a, chunk_b])
        sensors = [n for n in merged.nodes if "InertialSensors" in n.labels]
        self.assertEqual(len(sensors), 1)
        self.assertEqual(sensors[0].properties.hasSensorID, "IMU-ST-004")

    def test_duplicate_nonsensor_across_chunks(self):
        a = CandidateGraph.model_construct(
            nodes=[_node("h1", "Horse", id="Horse_Dakota", hasName="Dakota")],
            relationships=[],
            rejected_facts=[],
        )
        b = CandidateGraph.model_construct(
            nodes=[_node("Horse_Dakota", "Horse", id="Horse_Dakota", hasName="Dakota")],
            relationships=[],
            rejected_facts=[],
        )
        merged = merge_candidate_graphs([a, b])
        horses = [n for n in merged.nodes if "Horse" in n.labels]
        self.assertEqual(len(horses), 1)

    def test_relationship_duplicated_across_chunks(self):
        nodes = [
            _node("Rider_Antoine", "Rider", id="Rider_Antoine", hasName="Antoine"),
            _node("Horse_Thunder", "Horse", id="Horse_Thunder", hasName="Thunder"),
        ]
        rel = _rel("ASSOCIATEDWITH", "Rider_Antoine", "Horse_Thunder")
        a = CandidateGraph.model_construct(nodes=nodes, relationships=[rel], rejected_facts=[])
        b = CandidateGraph.model_construct(nodes=[], relationships=[rel], rejected_facts=[])
        merged = merge_candidate_graphs([a, b])
        self.assertEqual(len(merged.relationships), 1)

    def test_unresolved_cross_chunk_relationship(self):
        graph = CandidateGraph.model_construct(
            nodes=[_node("Horse_Thunder", "Horse", id="Horse_Thunder", hasName="Thunder")],
            relationships=[_rel("ASSOCIATEDWITH", "Rider_Antoine", "Horse_Thunder")],
            rejected_facts=[],
        )
        merged = merge_candidate_graphs([graph])
        self.assertEqual(len(merged.relationships), 0)
        self.assertTrue(
            any("Unresolved" in (rf.reason or "") for rf in merged.rejected_facts)
        )

    def test_full_relationship_type_preservation(self):
        registry_nodes = [
            _node("Rider_A", "Rider", id="Rider_A", hasName="A"),
            _node("Horse_H", "Horse", id="Horse_H", hasName="H"),
            _node("Event_E", "ShowJumping", id="Event_E"),
            _node("Part_P", "EventParticipation", id="Part_P"),
            _node("Stage_S", "PreparationStage", id="Stage_S"),
            _node("Vet_V", "Veterinarian", id="Vet_V", hasName="V"),
            _node("Season_S", "CompetitiveSeason", id="Season_S", seasonName="2026"),
            _node("Obj_O", "ExperimentalObjective", id="Obj_O", hasName="O"),
            _sensor("Sens_1", hasSensorID="S1", id="Sens_1"),
        ]
        registry = NodeRegistry()
        for n in registry_nodes:
            registry.add(n)
        rels = [
            _rel("ASSOCIATEDWITH", "Rider_A", "Horse_H"),
            _rel("COMPETESIN", "Horse_H", "Event_E"),
            _rel("TRAINSIN", "Horse_H", "Stage_S"),
            _rel("DEPENDSON", "Stage_S", "Event_E"),
            _rel("INVOLVESACTOR", "Stage_S", "Vet_V"),
            _rel("HASPARTICIPATION", "Event_E", "Part_P"),
            _rel("HASHORSE", "Part_P", "Horse_H"),
            _rel("HASRIDER", "Part_P", "Rider_A"),
            _rel("INSEASON", "Event_E", "Season_S"),
            _rel("ISATTACHEDTO", "Sens_1", "Horse_H"),
            _rel("ISUSEDFOR", "Sens_1", "Obj_O"),
        ]
        kept, rejected = resolve_relationships_against_registry(rels, registry)
        self.assertEqual(len(rejected), 0)
        types = {r.type for r in kept}
        self.assertEqual(len(types), 11)
        self.assertEqual(len(kept), 11)

    def test_local_id_promoted_to_ontology_id(self):
        from dynamic_kg.extract_facts import (
            LlmCandidateGraph,
            LlmCandidateNode,
            llm_graph_to_candidate_graph,
        )

        llm = LlmCandidateGraph(
            nodes=[
                LlmCandidateNode(
                    local_id="Event_SJ_2026_01",
                    labels=["ShowJumping"],
                    props=[],
                    source_evidence="NODE | type=ShowJumping | id=Event_SJ_2026_01",
                )
            ],
            relationships=[],
            rejected_facts=[],
        )
        graph = llm_graph_to_candidate_graph(llm)
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.nodes[0].properties.id, "Event_SJ_2026_01")


if __name__ == "__main__":
    unittest.main()
