"""
Unit tests for full-V9 dynamic ingestion (stdlib unittest).

Covers extraction-model validation + writer planning A–L.
No production Neo4j writes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.graph_rag.dynamic_ingestion_writer import (
    DynamicIngestionValidationError,
    Provenance,
    assert_cypher_safe,
    insert_reviewed_candidates,
    plan_ingestion,
    preflight_reviewed_candidates,
    validate_candidate_graph,
    _POSITION_SET_CYPHER,
)
from dynamic_kg.extract_facts import (
    CandidateGraph,
    CandidateNode,
    CandidateRelationship,
    NodeProperties,
    SensorProperties,
)


def _props(**kwargs) -> NodeProperties:
    return NodeProperties(**kwargs)


def _sensor(local_id: str, sensor_id: str, position: str, **prop_overrides) -> CandidateNode:
    return CandidateNode(
        local_id=local_id,
        labels=["InertialSensors", position],  # type: ignore[arg-type]
        properties=_props(hasSensorID=sensor_id, **prop_overrides),
        source_evidence=f"test {sensor_id}",
    )


def _node(local_id: str, label: str, **prop_kwargs) -> CandidateNode:
    return CandidateNode(
        local_id=local_id,
        labels=[label],  # type: ignore[arg-type]
        properties=_props(**prop_kwargs),
        source_evidence=f"test {local_id}",
    )


def _rel(
    rel_type: str,
    source: str,
    *,
    target_local_id: str | None = None,
    target_name: str | None = None,
) -> CandidateRelationship:
    return CandidateRelationship(
        type=rel_type,  # type: ignore[arg-type]
        source_local_id=source,
        target_local_id=target_local_id,
        target_name=target_name,
        source_evidence=f"{rel_type} {source}",
    )


def visir_graph() -> CandidateGraph:
    return CandidateGraph(
        nodes=[
            _sensor("sensor_6845", "6845", "Sternum"),
            _sensor("sensor_0C31", "0C31", "CanonOfForelimb"),
            _sensor("sensor_5032", "5032", "CanonOfForelimb"),
            _sensor("sensor_4D3C", "4D3C", "CanonOfHindlimb"),
            _sensor("sensor_5CBF", "5CBF", "CanonOfHindlimb"),
        ],
        relationships=[],
        rejected_facts=[],
    )


def _session_from_entities(entities: dict[str, list[dict]]) -> MagicMock:
    """
    entities keys are coarse match tags inspected in cypher:
    'InertialSensors', 'Horse', 'Rider', 'Veterinarian', 'ASSOCIATEDWITH', …
    """
    session = MagicMock()

    def run_side_effect(cypher, **params):
        result = MagicMock()
        text = cypher
        matched: list[dict] = []
        for key, rows in entities.items():
            if key in text:
                matched = rows
                break
        result.__iter__ = lambda self: iter(matched)
        result.single.return_value = matched[0] if matched else None
        return result

    session.run.side_effect = run_side_effect
    return session


class TestFullV9Ingestion(unittest.TestCase):
    def test_A_rider_horse_associatedwith(self):
        graph = CandidateGraph(
            nodes=[
                _node("rider_antoine", "Rider", id="Rider_Antoine", hasName="Antoine"),
                _node("horse_thunder", "Horse", id="Horse_Thunder", hasName="Thunder"),
            ],
            relationships=[
                _rel(
                    "ASSOCIATEDWITH",
                    "rider_antoine",
                    target_local_id="horse_thunder",
                )
            ],
            rejected_facts=[],
        )
        validate_candidate_graph(graph)
        session = _session_from_entities(
            {
                "Rider": [
                    {
                        "eid": "e-rider",
                        "labels": ["Rider"],
                        "props": {
                            "id": "Rider_Antoine",
                            "hasName": "Antoine",
                            "uri": "http://ex#Rider_Antoine",
                        },
                    }
                ],
                "Horse": [
                    {
                        "eid": "e-horse",
                        "labels": ["Horse"],
                        "props": {
                            "id": "Horse_Thunder",
                            "hasName": "Thunder",
                            "uri": "http://ex#Horse_Thunder",
                        },
                    }
                ],
                "ASSOCIATEDWITH": [],
            }
        )
        nodes, rels = plan_ingestion(graph, session)
        self.assertTrue(all(p.action == "noop" for p in nodes))
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].action, "create")
        self.assertEqual(rels[0].type, "ASSOCIATEDWITH")

    def test_B_veterinarian_involvesactor(self):
        graph = CandidateGraph(
            nodes=[
                _node(
                    "vet",
                    "Veterinarian",
                    id="Vet_DrMartin",
                    hasName="Dr Martin",
                ),
                _node(
                    "stage",
                    "PreparationStage",
                    id="Training_Prepa_Thunder_01",
                    Volume="45min",
                ),
            ],
            relationships=[
                _rel("INVOLVESACTOR", "stage", target_local_id="vet"),
            ],
            rejected_facts=[],
        )
        validate_candidate_graph(graph)
        session = _session_from_entities(
            {
                "Veterinarian": [
                    {
                        "eid": "e-vet",
                        "labels": ["Veterinarian"],
                        "props": {
                            "id": "Vet_DrMartin",
                            "hasName": "Dr Martin",
                            "uri": "http://ex#Vet_DrMartin",
                        },
                    }
                ],
                "PreparationStage": [
                    {
                        "eid": "e-stage",
                        "labels": ["PreparationStage"],
                        "props": {
                            "id": "Training_Prepa_Thunder_01",
                            "uri": "http://ex#Training_Prepa_Thunder_01",
                        },
                    }
                ],
                "INVOLVESACTOR": [],
            }
        )
        _, rels = plan_ingestion(graph, session)
        self.assertEqual(rels[0].action, "create")
        self.assertEqual(rels[0].type, "INVOLVESACTOR")

    def test_C_event_participation_horse_rider(self):
        graph = CandidateGraph(
            nodes=[
                _node("event", "ShowJumping", id="Event_SJ_2026_01"),
                _node("part", "EventParticipation", id="Participation_Test_01", rank=1),
                _node("horse", "Horse", id="Horse_Thunder", hasName="Thunder"),
                _node("rider", "Rider", id="Rider_Antoine", hasName="Antoine"),
            ],
            relationships=[
                _rel("HASPARTICIPATION", "event", target_local_id="part"),
                _rel("HASHORSE", "part", target_local_id="horse"),
                _rel("HASRIDER", "part", target_local_id="rider"),
            ],
            rejected_facts=[],
        )
        validate_candidate_graph(graph)
        self.assertEqual(len(graph.relationships), 3)

    def test_D_horse_trainsin_stage_dependson_event(self):
        graph = CandidateGraph(
            nodes=[
                _node("horse", "Horse", hasName="Thunder", id="Horse_Thunder"),
                _node("stage", "PreparationStage", id="Training_Prepa_Thunder_01"),
                _node("event", "ShowJumping", id="Event_SJ_2026_01"),
            ],
            relationships=[
                _rel("TRAINSIN", "horse", target_local_id="stage"),
                _rel("DEPENDSON", "stage", target_local_id="event"),
            ],
            rejected_facts=[],
        )
        validate_candidate_graph(graph)
        self.assertEqual({r.type for r in graph.relationships}, {"TRAINSIN", "DEPENDSON"})

    def test_E_sensor_to_horse(self):
        session = _session_from_entities(
            {
                "InertialSensors": [
                    {
                        "eid": "eid-s1",
                        "labels": ["InertialSensors", "Sternum"],
                        "props": {
                            "hasSensorID": "A1",
                            "uri": "urn:dynamic:sensor:A1",
                            "id": "dynamic_sensor_A1",
                        },
                    }
                ],
                "Horse": [
                    {
                        "eid": "eid-horse",
                        "labels": ["Horse"],
                        "props": {
                            "hasName": "Dakota",
                            "id": "Horse_Dakota",
                            "uri": "http://ex#Horse_Dakota",
                        },
                    }
                ],
                "ISATTACHEDTO": [],
            }
        )
        graph = CandidateGraph(
            nodes=[_sensor("s1", "A1", "Sternum")],
            relationships=[
                _rel("ISATTACHEDTO", "s1", target_name="Dakota"),
            ],
            rejected_facts=[],
        )
        _, rels = plan_ingestion(graph, session)
        self.assertEqual(rels[0].action, "create")

    def test_F_sensor_to_experimental_objective(self):
        session = _session_from_entities(
            {
                "InertialSensors": [
                    {
                        "eid": "eid-s1",
                        "labels": ["InertialSensors", "Withers"],
                        "props": {
                            "hasSensorID": "B1",
                            "uri": "urn:dynamic:sensor:B1",
                            "id": "dynamic_sensor_B1",
                        },
                    }
                ],
                "ExperimentalObjective": [
                    {
                        "eid": "eid-obj",
                        "labels": ["ExperimentalObjective"],
                        "props": {
                            "id": "FatigueDetection",
                            "hasName": "Fatigue Detection",
                            "uri": "http://ex#FatigueDetection",
                        },
                    }
                ],
                "ISUSEDFOR": [],
            }
        )
        graph = CandidateGraph(
            nodes=[_sensor("s1", "B1", "Withers")],
            relationships=[
                _rel("ISUSEDFOR", "s1", target_name="FatigueDetection"),
            ],
            rejected_facts=[],
        )
        _, rels = plan_ingestion(graph, session)
        self.assertEqual(rels[0].action, "create")
        self.assertEqual(rels[0].type, "ISUSEDFOR")

    def test_G_competitive_season(self):
        graph = CandidateGraph(
            nodes=[
                _node(
                    "season",
                    "CompetitiveSeason",
                    id="Season_2026",
                    seasonName="2026",
                )
            ],
            relationships=[],
            rejected_facts=[],
        )
        validate_candidate_graph(graph)
        session = _session_from_entities(
            {
                "CompetitiveSeason": [
                    {
                        "eid": "e-season",
                        "labels": ["CompetitiveSeason"],
                        "props": {
                            "id": "Season_2026",
                            "seasonName": "2026",
                            "uri": "http://ex#Season_2026",
                        },
                    }
                ]
            }
        )
        plans, _ = plan_ingestion(graph, session)
        self.assertEqual(plans[0].action, "noop")
        self.assertIn("Season_2026", plans[0].stable_id or "")

    def test_H_dakota_reuse_not_duplicate(self):
        session = _session_from_entities(
            {
                "Horse": [
                    {
                        "eid": "e-dakota",
                        "labels": ["Horse"],
                        "props": {
                            "id": "Horse_Dakota",
                            "hasName": "Dakota",
                            "uri": "http://ex#Horse_Dakota",
                        },
                    }
                ]
            }
        )
        graph = CandidateGraph(
            nodes=[_node("horse_dakota", "Horse", hasName="Dakota", id="Horse_Dakota")],
            relationships=[],
            rejected_facts=[],
        )
        plans, _ = plan_ingestion(graph, session)
        self.assertEqual(plans[0].action, "noop")
        self.assertEqual(plans[0].existing_element_id, "e-dakota")
        self.assertNotEqual(plans[0].action, "create")

    def test_I_unsupported_label_rejected(self):
        bad = CandidateNode.model_construct(
            local_id="bad",
            labels=["InertialSensors", "FrontLeft"],
            properties=SensorProperties(hasSensorID="X1"),
            source_evidence="x",
        )
        graph = CandidateGraph.model_construct(
            nodes=[bad], relationships=[], rejected_facts=[]
        )
        with self.assertRaises(DynamicIngestionValidationError):
            validate_candidate_graph(graph)

    def test_J_unsupported_property_rejected(self):
        node = CandidateNode(
            local_id="n1",
            labels=["InertialSensors", "Sternum"],  # type: ignore[arg-type]
            properties=SensorProperties(hasSensorID="X1"),
            source_evidence="x",
        )
        graph = CandidateGraph(nodes=[node], relationships=[], rejected_facts=[])
        original = NodeProperties.model_dump

        def dump_with_extra(self, *args, **kwargs):
            data = original(self, *args, **kwargs)
            data["evilProp"] = "nope"
            return data

        with patch.object(NodeProperties, "model_dump", dump_with_extra):
            with self.assertRaisesRegex(DynamicIngestionValidationError, "unknown property"):
                validate_candidate_graph(graph)

    def test_K_unsupported_relationship_rejected(self):
        graph = CandidateGraph.model_construct(
            nodes=[_sensor("s1", "A1", "Sternum")],
            relationships=[
                CandidateRelationship.model_construct(
                    type="MADE_UP_REL",
                    source_local_id="s1",
                    target_name="Dakota",
                    target_local_id=None,
                    source_evidence="x",
                )
            ],
            rejected_facts=[],
        )
        with self.assertRaisesRegex(DynamicIngestionValidationError, "unknown type"):
            validate_candidate_graph(graph)

    def test_L_missing_relationship_target_rejected(self):
        session = _session_from_entities(
            {
                "InertialSensors": [
                    {
                        "eid": "eid-s1",
                        "labels": ["InertialSensors", "Sternum"],
                        "props": {
                            "hasSensorID": "A1",
                            "uri": "urn:dynamic:sensor:A1",
                            "id": "dynamic_sensor_A1",
                        },
                    }
                ],
                "Horse": [],
                "ISATTACHEDTO": [],
            }
        )
        graph = CandidateGraph(
            nodes=[_sensor("s1", "A1", "Sternum")],
            relationships=[_rel("ISATTACHEDTO", "s1", target_name="NoSuchHorse")],
            rejected_facts=[],
        )
        _, rels = plan_ingestion(graph, session)
        self.assertEqual(rels[0].action, "rejected")
        self.assertIn("Zero matches", rels[0].detail)


class TestLegacySensorWriter(unittest.TestCase):
    def test_null_does_not_overwrite(self):
        session = MagicMock()
        session.run.return_value = [
            {
                "eid": "e1",
                "labels": ["InertialSensors", "Sternum"],
                "props": {"hasSensorID": "6845", "hasFormat": "CSV"},
            }
        ]
        graph = CandidateGraph(
            nodes=[_sensor("sensor_6845", "6845", "Sternum", hasFormat=None)],
            relationships=[],
            rejected_facts=[],
        )
        plans, _ = plan_ingestion(graph, session)
        self.assertEqual(plans[0].action, "noop")
        self.assertNotIn("hasFormat", plans[0].properties_to_set)

    def test_identical_value_noop(self):
        session = MagicMock()
        session.run.return_value = [
            {
                "eid": "e1",
                "labels": ["InertialSensors", "Sternum"],
                "props": {"hasSensorID": "6845", "hasFormat": "CSV"},
            }
        ]
        graph = CandidateGraph(
            nodes=[_sensor("sensor_6845", "6845", "Sternum", hasFormat="CSV")],
            relationships=[],
            rejected_facts=[],
        )
        plans, _ = plan_ingestion(graph, session)
        self.assertEqual(plans[0].action, "noop")

    def test_conflicting_existing_value_reported(self):
        session = MagicMock()
        session.run.return_value = [
            {
                "eid": "e1",
                "labels": ["InertialSensors", "Sternum"],
                "props": {"hasSensorID": "6845", "hasFormat": "CSV"},
            }
        ]
        graph = CandidateGraph(
            nodes=[_sensor("sensor_6845", "6845", "Sternum", hasFormat="MAT")],
            relationships=[],
            rejected_facts=[],
        )
        plans, _ = plan_ingestion(graph, session)
        self.assertEqual(plans[0].action, "conflict")
        self.assertEqual(plans[0].conflicts[0]["property"], "hasFormat")

    def test_duplicate_has_sensor_id_rejected(self):
        session = MagicMock()
        session.run.return_value = [
            {"eid": "e1", "labels": ["InertialSensors"], "props": {"hasSensorID": "6845"}},
            {"eid": "e2", "labels": ["InertialSensors"], "props": {"hasSensorID": "6845"}},
        ]
        graph = CandidateGraph(
            nodes=[_sensor("sensor_6845", "6845", "Sternum")],
            relationships=[],
            rejected_facts=[],
        )
        plans, _ = plan_ingestion(graph, session)
        self.assertEqual(plans[0].action, "rejected")
        self.assertIn("Ambiguous", plans[0].detail)

    def test_controlled_position_label_mapping(self):
        for label, cypher in _POSITION_SET_CYPHER.items():
            self.assertIn(label, cypher)
            self.assertIn("$eid", cypher)
            assert_cypher_safe(cypher)

    def test_no_relationships_when_empty(self):
        session = MagicMock()
        session.run.return_value = []
        plans, rels = plan_ingestion(visir_graph(), session)
        self.assertEqual(len(plans), 5)
        self.assertTrue(all(p.action == "create" for p in plans))
        self.assertEqual(rels, [])

    def test_dry_run_performs_no_writes(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        mock_session.run.return_value = []

        result = preflight_reviewed_candidates(
            visir_graph(),
            Provenance(source_filename="VISIR_Dressage_5_Sensor_Summary_v2.pdf"),
            driver=mock_driver,
            database="neo4j",
        )
        self.assertTrue(result.valid)
        self.assertEqual(len(result.creates), 5)
        self.assertEqual(result.relationships_to_create, [])
        for c in mock_session.run.call_args_list:
            cypher = c.args[0] if c.args else ""
            self.assertIn("MATCH", cypher)
            self.assertNotIn("DELETE", cypher.upper())
            self.assertNotIn("MERGE", cypher.upper())
            self.assertFalse(cypher.strip().upper().startswith("CREATE"))

    def test_insert_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(PermissionError, "confirm_write_to_production"):
            insert_reviewed_candidates(visir_graph(), confirm_write_to_production=False)

    def test_transaction_rollback_behavior(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        mock_session.run.return_value = []

        def boom(_fn):
            raise RuntimeError("simulated write failure")

        mock_session.execute_write.side_effect = boom
        result = insert_reviewed_candidates(
            visir_graph(),
            confirm_write_to_production=True,
            driver=mock_driver,
            database="neo4j",
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_no_destructive_cypher(self):
        with self.assertRaises(DynamicIngestionValidationError):
            assert_cypher_safe("MATCH (n) DETACH DELETE n")
        with self.assertRaises(DynamicIngestionValidationError):
            assert_cypher_safe("DROP DATABASE neo4j")
        assert_cypher_safe(
            "MATCH (s:InertialSensors {hasSensorID: $hasSensorID}) RETURN s"
        )

    def test_allowlisted_relationship_resolves_existing_horse(self):
        session = _session_from_entities(
            {
                "InertialSensors": [
                    {
                        "eid": "eid-s1",
                        "labels": ["InertialSensors", "Sternum"],
                        "props": {
                            "hasSensorID": "A1",
                            "uri": "urn:dynamic:sensor:A1",
                            "id": "dynamic_sensor_A1",
                        },
                    }
                ],
                "Horse": [
                    {
                        "eid": "eid-horse",
                        "labels": ["Horse"],
                        "props": {
                            "hasName": "Dakota",
                            "id": "Horse_Dakota",
                            "uri": "http://example.org#Horse_Dakota",
                        },
                    }
                ],
                "ISATTACHEDTO": [],
            }
        )
        graph = CandidateGraph(
            nodes=[_sensor("s1", "A1", "Sternum")],
            relationships=[
                CandidateRelationship(
                    type="ISATTACHEDTO",
                    source_local_id="s1",
                    target_name="Dakota",
                    source_evidence="DYNTEST-S1 ISATTACHEDTO Dakota",
                )
            ],
            rejected_facts=[],
        )
        validate_candidate_graph(graph)
        _, rels = plan_ingestion(graph, session)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].action, "create")
        self.assertEqual(rels[0].target_name, "Dakota")

    def test_associatedwith_membership_includes_both_ends(self):
        graph = CandidateGraph(
            nodes=[
                _node("rider_antoine", "Rider", id="Rider_Antoine", hasName="Antoine"),
                _node("horse_thunder", "Horse", id="Horse_Thunder", hasName="Thunder"),
            ],
            relationships=[
                _rel(
                    "ASSOCIATEDWITH",
                    "rider_antoine",
                    target_local_id="horse_thunder",
                )
            ],
            rejected_facts=[],
        )
        mock_driver = MagicMock()
        mock_session = _session_from_entities(
            {
                "Rider": [
                    {
                        "eid": "e-rider",
                        "labels": ["Rider"],
                        "props": {
                            "id": "Rider_Antoine",
                            "hasName": "Antoine",
                            "uri": "http://ex#Rider_Antoine",
                        },
                    }
                ],
                "Horse": [
                    {
                        "eid": "e-horse",
                        "labels": ["Horse"],
                        "props": {
                            "id": "Horse_Thunder",
                            "hasName": "Thunder",
                            "uri": "http://ex#Horse_Thunder",
                        },
                    }
                ],
                "ASSOCIATEDWITH": [],
            }
        )
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        pf = preflight_reviewed_candidates(graph, driver=mock_driver, database="neo4j")
        ids = {a["stable_id"] for a in pf.affected_nodes}
        self.assertTrue(any("Antoine" in i or "Rider_Antoine" in i for i in ids))
        self.assertTrue(any("Thunder" in i or "Horse_Thunder" in i for i in ids))
        self.assertEqual(len(pf.affected_relationships), 1)


if __name__ == "__main__":
    unittest.main()
