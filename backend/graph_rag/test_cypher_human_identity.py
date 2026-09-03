"""Unit tests for generic Rider/Veterinarian/Caretaker id → hasName Cypher rewrite."""
from __future__ import annotations

from backend.graph_rag.cypher_human_identity import (
    RETRY_REASON_AMBIGUOUS,
    RETRY_REASON_HAS_NAME,
    RETRY_REASON_NO_MATCH,
    apply_human_identity_retry_if_needed,
    extract_human_id_predicates,
    plan_human_identity_rewrite,
    rewrite_human_id_to_has_name,
)
from backend.graph_rag.cypher_membership_scope import membership_predicate
from backend.graph_rag.cypher_sensor_identity import (
    RETRY_REASON_HAS_SENSOR_ID,
    apply_sensor_identity_retry_if_needed,
    plan_sensor_identity_rewrite,
)


class _FakeHumanGraph:
    """Counts keyed by (label, prop, op, literal). Optional retry-query rows."""

    def __init__(
        self,
        counts: dict[tuple[str, str, str, str], int],
        retry_rows: list | None = None,
    ):
        self.counts = counts
        self.retry_rows = retry_rows or [{"stage": "PreparationStage"}]
        self.executed: list[str] = []

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        self.executed.append(cypher)
        if "RETURN count(h) AS n" in cypher:
            lit = params.get("literal")
            label = "Veterinarian"
            if ":Rider)" in cypher or ":Rider " in cypher:
                label = "Rider"
            elif ":Caretaker)" in cypher or ":Caretaker " in cypher:
                label = "Caretaker"
            elif ":Veterinarian)" in cypher or ":Veterinarian " in cypher:
                label = "Veterinarian"
            prop = "hasName" if "h.hasName" in cypher else "id"
            op = "="
            if " CONTAINS " in cypher:
                op = "CONTAINS"
            elif " STARTS WITH " in cypher:
                op = "STARTS WITH"
            elif " ENDS WITH " in cypher:
                op = "ENDS WITH"
            return [{"n": self.counts.get((label, prop, op, lit), 0)}]
        return list(self.retry_rows)


class _FakeQA:
    output_key = "text"

    def invoke(self, inputs, callbacks=None):
        ctx = inputs.get("context") or []
        return {"text": f"qa rows={len(ctx)}"}


class _FakeChain:
    def __init__(self, graph):
        self.graph = graph
        self.top_k = 50
        self.return_intermediate_steps = True
        self.output_key = "result"
        self.qa_chain = _FakeQA()


def _empty_result(cypher: str) -> dict:
    return {
        "result": "I don't know",
        "intermediate_steps": [
            {"query": cypher},
            {"context": []},
        ],
    }


def _vet_contains(name: str = "Martin") -> str:
    return (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Veterinarian) "
        f'WHERE a.id CONTAINS "{name}" '
        "RETURN t"
    )


def _caretaker_contains(name: str = "Sophie") -> str:
    return (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Caretaker) "
        f'WHERE a.id CONTAINS "{name}" '
        "RETURN t"
    )


def _rider_contains(name: str = "Alex") -> str:
    return (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Rider) "
        f'WHERE a.id CONTAINS "{name}" '
        "RETURN t"
    )


def test_veterinarian_contains_id_retries_has_name():
    graph = _FakeHumanGraph(
        {("Veterinarian", "hasName", "CONTAINS", "Martin"): 1}
    )
    q = _vet_contains("Martin")
    preds = extract_human_id_predicates(q)
    assert len(preds) == 1
    assert preds[0].label == "Veterinarian"
    assert preds[0].var == "a"
    assert preds[0].op_norm == "CONTAINS"

    rewritten = rewrite_human_id_to_has_name(q, preds[0])
    assert 'a.hasName CONTAINS "Martin"' in rewritten
    assert 'a.id CONTAINS "Martin"' not in rewritten
    assert "[:INVOLVESACTOR]" in rewritten
    assert "(a:Veterinarian)" in rewritten

    corrected, reason = plan_human_identity_rewrite(q, graph)
    assert reason == RETRY_REASON_HAS_NAME
    assert corrected is not None
    assert 'a.hasName CONTAINS "Martin"' in corrected

    chain = _FakeChain(graph)
    out = apply_human_identity_retry_if_needed(
        chain, {"query": "Which training stages involve Dr Martin?"}, _empty_result(q)
    )
    assert out["human_identity_retry_used"] is True
    assert out["cypher_retry_used"] is True
    assert 'a.hasName CONTAINS "Martin"' in out["final_cypher"]
    assert out["original_cypher"] == q


def test_caretaker_contains_id_retries_has_name():
    graph = _FakeHumanGraph(
        {("Caretaker", "hasName", "CONTAINS", "Sophie"): 1}
    )
    q = _caretaker_contains("Sophie")
    corrected, reason = plan_human_identity_rewrite(q, graph)
    assert reason == RETRY_REASON_HAS_NAME
    assert corrected is not None
    assert 'a.hasName CONTAINS "Sophie"' in corrected
    assert 'a.id CONTAINS "Sophie"' not in corrected


def test_rider_contains_id_retries_has_name():
    graph = _FakeHumanGraph({("Rider", "hasName", "CONTAINS", "Alex"): 1})
    q = _rider_contains("Alex")
    corrected, reason = plan_human_identity_rewrite(q, graph)
    assert reason == RETRY_REASON_HAS_NAME
    assert corrected is not None
    assert 'a.hasName CONTAINS "Alex"' in corrected
    assert "(a:Rider)" in corrected


def test_genuine_internal_human_id_with_rows_does_not_retry():
    graph = _FakeHumanGraph(
        {("Veterinarian", "hasName", "CONTAINS", "Martin"): 1}
    )
    q = _vet_contains("Martin")
    result = {
        "result": "PreparationStage",
        "intermediate_steps": [
            {"query": q},
            {"context": [{"stage": "PreparationStage"}]},
        ],
    }
    out = apply_human_identity_retry_if_needed(
        _FakeChain(graph), {"query": "q"}, result
    )
    assert out["human_identity_retry_used"] is False
    assert out.get("final_cypher") == q
    assert not any("hasName CONTAINS" in c for c in graph.executed)


def test_genuine_internal_id_probe_skips_rewrite():
    graph = _FakeHumanGraph(
        {
            ("Veterinarian", "id", "CONTAINS", "Vet_DrMartin"): 1,
            ("Veterinarian", "hasName", "CONTAINS", "Vet_DrMartin"): 0,
        }
    )
    q = (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Veterinarian) "
        'WHERE a.id CONTAINS "Vet_DrMartin" RETURN t'
    )
    corrected, reason = plan_human_identity_rewrite(q, graph)
    assert corrected is None
    assert reason is None


def test_horse_id_does_not_trigger_human_retry():
    graph = _FakeHumanGraph({("Horse", "hasName", "CONTAINS", "Dakota"): 1})
    q = (
        "MATCH (h:Horse)-[:TRAINSIN]->(t) "
        'WHERE h.id CONTAINS "Dakota" RETURN t'
    )
    assert extract_human_id_predicates(q) == []
    corrected, reason = plan_human_identity_rewrite(q, graph)
    assert corrected is None
    assert reason is None


def test_inertial_sensors_id_is_not_human_retry():
    q = (
        'MATCH (s:InertialSensors) WHERE s.id = "CODE-1" '
        "RETURN s"
    )
    assert extract_human_id_predicates(q) == []

    class _SensorGraph:
        def query(self, cypher: str, params: dict | None = None):
            params = params or {}
            lit = params.get("literal")
            if "s.hasSensorID" in cypher:
                return [{"n": 1 if lit == "CODE-1" else 0}]
            if "s.id" in cypher:
                return [{"n": 0}]
            raise AssertionError(f"unexpected probe: {cypher}")

    sensor_graph = _SensorGraph()
    human_corrected, human_reason = plan_human_identity_rewrite(q, sensor_graph)
    assert human_corrected is None
    assert human_reason is None

    sensor_corrected, sensor_reason = plan_sensor_identity_rewrite(q, sensor_graph)
    assert sensor_reason == RETRY_REASON_HAS_SENSOR_ID
    assert sensor_corrected is not None
    assert "hasSensorID" in sensor_corrected


def test_ambiguous_has_name_does_not_rewrite():
    graph = _FakeHumanGraph(
        {("Caretaker", "hasName", "CONTAINS", "Marie"): 2}
    )
    q = _caretaker_contains("Marie")
    corrected, reason = plan_human_identity_rewrite(q, graph)
    assert corrected is None
    assert reason == RETRY_REASON_AMBIGUOUS

    out = apply_human_identity_retry_if_needed(
        _FakeChain(graph), {"query": "q"}, _empty_result(q)
    )
    assert out["human_identity_retry_used"] is False
    assert out["human_identity_retry_reason"] == RETRY_REASON_AMBIGUOUS
    assert 'a.id CONTAINS "Marie"' in (out.get("final_cypher") or q)


def test_no_has_name_match_does_not_rewrite():
    graph = _FakeHumanGraph({})
    q = _vet_contains("Nobody")
    corrected, reason = plan_human_identity_rewrite(q, graph)
    assert corrected is None
    assert reason == RETRY_REASON_NO_MATCH


def test_membership_predicates_survive_human_retry():
    pred_t = membership_predicate("t")
    pred_a = membership_predicate("a")
    q = (
        f"MATCH (t:PreparationStage WHERE {pred_t})"
        f"-[:INVOLVESACTOR]->(a:Veterinarian WHERE {pred_a}) "
        'WHERE a.id CONTAINS "Martin" '
        "RETURN t"
    )
    graph = _FakeHumanGraph(
        {("Veterinarian", "hasName", "CONTAINS", "Martin"): 1}
    )
    corrected, reason = plan_human_identity_rewrite(q, graph)
    assert reason == RETRY_REASON_HAS_NAME
    assert corrected is not None
    assert pred_t in corrected
    assert pred_a in corrected
    assert "$__kg_scope" in corrected
    assert 'a.hasName CONTAINS "Martin"' in corrected
    assert 'a.id CONTAINS "Martin"' not in corrected
    assert "[:INVOLVESACTOR]" in corrected
    assert "RETURN t" in corrected


def test_does_not_rewrite_unrelated_id_in_same_query():
    q = (
        "MATCH (h:Horse)-[:TRAINSIN]->(t)-[:INVOLVESACTOR]->(a:Rider) "
        'WHERE h.id CONTAINS "Dakota" AND a.id CONTAINS "Alex" '
        "RETURN t, h.id"
    )
    preds = extract_human_id_predicates(q)
    assert len(preds) == 1
    assert preds[0].label == "Rider"
    rewritten = rewrite_human_id_to_has_name(q, preds[0])
    assert 'a.hasName CONTAINS "Alex"' in rewritten
    assert 'h.id CONTAINS "Dakota"' in rewritten
    assert "RETURN t, h.id" in rewritten


def test_sensor_retry_still_runs_before_human_on_empty_sensor_query():
    class _SensorExecGraph:
        def query(self, cypher: str, params: dict | None = None):
            params = params or {}
            if "RETURN count(s) AS n" in cypher or (
                "InertialSensors" in cypher and "count(s)" in cypher
            ):
                lit = params.get("literal")
                if "s.hasSensorID" in cypher:
                    return [{"n": 1 if lit == "CODE-1" else 0}]
                if "s.id" in cypher:
                    return [{"n": 0}]
            if "hasSensorID" in cypher:
                return [{"sid": "CODE-1"}]
            return []

    q = 'MATCH (s:InertialSensors {id: "CODE-1"}) RETURN s'
    chain = _FakeChain(_SensorExecGraph())
    sensor_out = apply_sensor_identity_retry_if_needed(
        chain, {"query": "sensor?"}, _empty_result(q)
    )
    assert sensor_out.get("cypher_retry_used") is True
    human_out = apply_human_identity_retry_if_needed(
        chain, {"query": "sensor?"}, sensor_out
    )
    assert human_out.get("human_identity_retry_used") is False
    assert "hasSensorID" in (human_out.get("final_cypher") or "")
