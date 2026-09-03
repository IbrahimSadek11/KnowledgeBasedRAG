"""Unit tests for QA context sensor-identity enrichment."""
from __future__ import annotations

from backend.graph_rag.cypher_context_enrichment import (
    _companion_has_sensor_id_key,
    enrich_neo4j_rows_for_qa,
)


def test_companion_keys():
    assert _companion_has_sensor_id_key("sensor_id") == "sensor_hasSensorID"
    assert _companion_has_sensor_id_key("id") == "hasSensorID"
    assert _companion_has_sensor_id_key("hasSensorID") is None


def test_enrich_scalar_sensor_id():
    def run_query(cypher, params=None):
        assert "InertialSensors" in cypher
        ids = set(params["ids"])
        out = []
        if "IMU_Sternum_Dakota_01" in ids:
            out.append(
                {"id": "IMU_Sternum_Dakota_01", "hasSensorID": "IMU-ST-004"}
            )
        return out

    rows = [{"horse": "Dakota", "sensor_id": "IMU_Sternum_Dakota_01"}]
    enriched = enrich_neo4j_rows_for_qa(rows, run_query)
    assert enriched[0]["sensor_id"] == "IMU_Sternum_Dakota_01"
    assert enriched[0]["sensor_hasSensorID"] == "IMU-ST-004"
    assert enriched[0]["horse"] == "Dakota"


def test_enrich_collect_list():
    def run_query(cypher, params=None):
        return [
            {"id": "IMU_A", "hasSensorID": "A-1"},
            {"id": "IMU_B", "hasSensorID": "B-2"},
        ]

    rows = [{"sensor_ids": ["IMU_A", "IMU_B", "not_a_sensor"]}]
    enriched = enrich_neo4j_rows_for_qa(rows, run_query)
    assert enriched[0]["sensor_ids"] == ["IMU_A", "IMU_B", "not_a_sensor"]
    assert enriched[0]["sensor_ids_hasSensorID"] == ["A-1", "B-2", None]


def test_no_op_when_no_sensor_ids():
    def run_query(cypher, params=None):
        raise AssertionError("lookup should not run with empty candidates")

    rows = [{"horse": "Dakota", "n": 4}]
    # candidates non-empty strings still trigger lookup — use empty-ish
    def run_query2(cypher, params=None):
        return []

    assert enrich_neo4j_rows_for_qa(rows, run_query2) == rows


def test_human_enrichment_appends_id_and_has_name():
    def run_query(cypher, params=None):
        if "InertialSensors" in cypher:
            return []
        assert "Veterinarian" in cypher
        assert params["literal"] == "Martin"
        return [{"id": "Vet_DrMartin", "hasName": "Dr Martin"}]

    cypher = (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Veterinarian) "
        'WHERE a.id CONTAINS "Martin" RETURN t.id AS stage'
    )
    rows = [{"stage": "Training_PreComp_Zephyr_01"}]
    enriched = enrich_neo4j_rows_for_qa(rows, run_query, cypher=cypher)
    assert enriched[0]["stage"] == "Training_PreComp_Zephyr_01"
    assert enriched[0]["queried_human_id"] == "Vet_DrMartin"
    assert enriched[0]["queried_human_hasName"] == "Dr Martin"
    assert enriched[0]["queried_human_label"] == "Veterinarian"


def test_human_enrichment_caretaker_and_rider():
    def run_query(cypher, params=None):
        if "InertialSensors" in cypher:
            return []
        lit = params["literal"]
        if "Caretaker" in cypher and lit == "Sophie":
            return [{"id": "Caretaker_Sophie", "hasName": "Sophie"}]
        if "Rider" in cypher and lit == "Alex":
            return [{"id": "Rider_Alex", "hasName": "Alex"}]
        raise AssertionError(cypher)

    caretaker_cypher = (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Caretaker) "
        'WHERE a.id CONTAINS "Sophie" RETURN t.id AS stage'
    )
    rider_cypher = (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Rider) "
        'WHERE a.hasName CONTAINS "Alex" RETURN t.id AS stage'
    )
    c_rows = enrich_neo4j_rows_for_qa(
        [{"stage": "s1"}], run_query, cypher=caretaker_cypher
    )
    assert c_rows[0]["queried_human_hasName"] == "Sophie"
    assert c_rows[0]["queried_human_label"] == "Caretaker"
    r_rows = enrich_neo4j_rows_for_qa(
        [{"stage": "s2"}], run_query, cypher=rider_cypher
    )
    assert r_rows[0]["queried_human_id"] == "Rider_Alex"
    assert r_rows[0]["queried_human_label"] == "Rider"


def test_human_enrichment_skips_ambiguous_and_zero_matches():
    def run_query(cypher, params=None):
        if "InertialSensors" in cypher:
            return []
        if params and params.get("literal") == "Marie":
            return [
                {"id": "Caretaker_MarieA", "hasName": "Marie A"},
                {"id": "Caretaker_MarieB", "hasName": "Marie B"},
            ]
        return []

    ambiguous = (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Caretaker) "
        'WHERE a.id CONTAINS "Marie" RETURN t'
    )
    none = (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Veterinarian) "
        'WHERE a.id CONTAINS "Nobody" RETURN t'
    )
    rows = [{"stage": "s1"}]
    assert "queried_human_hasName" not in enrich_neo4j_rows_for_qa(
        rows, run_query, cypher=ambiguous
    )[0]
    assert "queried_human_id" not in enrich_neo4j_rows_for_qa(
        rows, run_query, cypher=none
    )[0]


def test_human_enrichment_does_not_apply_to_horse_or_sensor_only_cypher():
    def run_query(cypher, params=None):
        if "InertialSensors" in cypher:
            return []
        raise AssertionError("human lookup should not run")

    horse = 'MATCH (h:Horse) WHERE h.id CONTAINS "Dakota" RETURN h'
    sensor = 'MATCH (s:InertialSensors) WHERE s.id = "CODE-1" RETURN s'
    rows = [{"x": 1}]
    assert enrich_neo4j_rows_for_qa(rows, run_query, cypher=horse) == rows
    assert enrich_neo4j_rows_for_qa(rows, run_query, cypher=sensor) == rows
