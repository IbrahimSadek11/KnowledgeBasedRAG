"""Unit tests for generic InertialSensors id → hasSensorID Cypher rewrite."""
from __future__ import annotations

from backend.graph_rag.cypher_sensor_identity import (
    RETRY_REASON_HAS_SENSOR_ID,
    extract_inertial_id_literals,
    is_read_only_cypher,
    plan_sensor_identity_rewrite,
    rewrite_inertial_id_literal_to_has_sensor_id,
)


class _FakeGraph:
    def __init__(self, id_hits: dict[str, int], has_hits: dict[str, int]):
        self.id_hits = id_hits
        self.has_hits = has_hits

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        lit = params.get("literal")
        if "s.hasSensorID" in cypher:
            return [{"n": self.has_hits.get(lit, 0)}]
        if "s.id" in cypher:
            return [{"n": self.id_hits.get(lit, 0)}]
        raise AssertionError(f"unexpected probe: {cypher}")


def test_extract_map_and_where_literals():
    q = (
        'MATCH (s:InertialSensors {id: "CODE-1"})-[:ISATTACHEDTO]->(h:Horse) '
        "RETURN h.hasName"
    )
    assert extract_inertial_id_literals(q) == ["CODE-1"]

    q2 = (
        "MATCH (s:InertialSensors) WHERE s.id = 'CODE-2' "
        "RETURN s.id AS sensor"
    )
    assert extract_inertial_id_literals(q2) == ["CODE-2"]


def test_rewrite_only_filter_not_return_projection_of_other_literal():
    q = (
        'MATCH (s:InertialSensors {id: "CODE-1"}) '
        "RETURN s.id AS sensor, s.hasSensorID AS sid"
    )
    out = rewrite_inertial_id_literal_to_has_sensor_id(q, "CODE-1")
    assert '{hasSensorID: "CODE-1"}' in out or "{hasSensorID:\"CODE-1\"}" in out.replace(
        " ", ""
    )
    assert "RETURN s.id AS sensor" in out


def test_does_not_touch_unrelated_labels():
    q = 'MATCH (e:Event {id: "Event_01"}) RETURN e.id'
    assert extract_inertial_id_literals(q) == []
    assert rewrite_inertial_id_literal_to_has_sensor_id(q, "Event_01") == q


def test_plan_rewrites_when_only_has_sensor_id_matches():
    graph = _FakeGraph(id_hits={}, has_hits={"CODE-1": 1})
    q = (
        'MATCH (s:InertialSensors {id: "CODE-1"})-[:ISATTACHEDTO]->(h:Horse) '
        "RETURN h.hasName"
    )
    corrected, reason = plan_sensor_identity_rewrite(q, graph)
    assert reason == RETRY_REASON_HAS_SENSOR_ID
    assert corrected is not None
    assert "hasSensorID" in corrected
    assert '{id: "CODE-1"}' not in corrected


def test_plan_keeps_real_internal_id():
    graph = _FakeGraph(id_hits={"IMU_Sternum_X_01": 1}, has_hits={})
    q = (
        'MATCH (s:InertialSensors {id: "IMU_Sternum_X_01"})-[:ISATTACHEDTO]->(h) '
        "RETURN h.hasName"
    )
    corrected, reason = plan_sensor_identity_rewrite(q, graph)
    assert corrected is None
    assert reason is None


def test_plan_ambiguous_has_sensor_id():
    graph = _FakeGraph(id_hits={}, has_hits={"DUP": 2})
    q = 'MATCH (s:InertialSensors {id: "DUP"}) RETURN s'
    corrected, reason = plan_sensor_identity_rewrite(q, graph)
    assert corrected is None
    assert reason is not None
    assert "multiple" in reason.lower()


def test_write_guard():
    assert is_read_only_cypher("MATCH (s) RETURN s")
    assert not is_read_only_cypher("MATCH (s) SET s.x = 1 RETURN s")
    assert not is_read_only_cypher("CREATE (s:InertialSensors)")
