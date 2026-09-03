"""Unit tests for deterministic Cypher membership injection."""
from __future__ import annotations

from backend.graph_rag.cypher_membership_scope import (
    inject_membership_scope,
    membership_predicate,
    normalize_stable_node_ids,
)


def test_inject_map_and_plain_node_patterns():
    src = (
        'MATCH (s:InertialSensors {id: "X"})-[:ISATTACHEDTO]->(h:Horse {hasName: "Dakota"})\n'
        "RETURN count(s) AS n"
    )
    out = inject_membership_scope(src)
    assert "s:InertialSensors" in out
    assert membership_predicate("s") in out
    assert membership_predicate("h") in out
    assert "[:ISATTACHEDTO]" in out
    assert 'hasName: "Dakota"' in out


def test_inject_preserves_existing_where_on_match():
    src = "MATCH (s:InertialSensors) WHERE s.hasSensorID = '6845' RETURN s"
    out = inject_membership_scope(src)
    assert "hasSensorID = '6845'" in out
    assert membership_predicate("s") in out


def test_normalize_ids():
    assert normalize_stable_node_ids([" a ", "a", "", "b"]) == ["a", "b"]


def test_membership_predicate_is_or_not_coalesce():
    pred = membership_predicate("s")
    assert "coalesce" not in pred.lower()
    assert "s.uri IN $__kg_scope" in pred
    assert "s.id IN $__kg_scope" in pred
    assert "toString(s.hasSensorID) IN $__kg_scope" in pred
    assert " OR " in pred


def test_count_query_gets_scoped_sensors():
    src = (
        'MATCH (h:Horse {hasName: "Dakota"})<-[:ISATTACHEDTO]-(s:InertialSensors)\n'
        "RETURN count(DISTINCT s) AS n"
    )
    out = inject_membership_scope(src)
    assert membership_predicate("h") in out
    assert membership_predicate("s") in out


def test_or_matches_when_membership_stores_id_but_uri_differs():
    """Regression: coalesce(uri,id,...) would miss id-only membership."""
    pred = membership_predicate("n")
    assert "n.id IN $__kg_scope" in pred
    assert "coalesce" not in pred.lower()


def test_unlabeled_and_labeled_involves_actor_both_scoped():
    src = (
        "MATCH (t)-[:INVOLVESACTOR]->(a:Veterinarian)\n"
        'WHERE a.id CONTAINS "Martin"\n'
        "RETURN t.id AS stage"
    )
    out = inject_membership_scope(src)
    assert membership_predicate("t") in out
    assert membership_predicate("a") in out
    assert "[:INVOLVESACTOR]" in out
    assert 'a.id CONTAINS "Martin"' in out
    assert "RETURN t.id AS stage" in out
    assert "count(t WHERE" not in out


def test_unlabeled_stage_after_horse_trainsin_both_scoped():
    src = 'MATCH (h:Horse)-[:TRAINSIN]->(stage) RETURN stage'
    out = inject_membership_scope(src)
    assert membership_predicate("h") in out
    assert membership_predicate("stage") in out
    assert "[:TRAINSIN]" in out


def test_unlabeled_event_and_labeled_season_both_scoped():
    src = (
        "MATCH (event)-[:INSEASON]->(season:CompetitiveSeason)\n"
        "RETURN event.id AS event_id"
    )
    out = inject_membership_scope(src)
    assert membership_predicate("event") in out
    assert membership_predicate("season") in out
    assert "[:INSEASON]" in out
    assert "RETURN event.id AS event_id" in out


def test_labeled_participation_horse_unchanged_and_scoped():
    src = (
        "MATCH (p:EventParticipation)-[:HASHORSE]->(h:Horse)\n"
        "RETURN p, h"
    )
    out = inject_membership_scope(src)
    assert membership_predicate("p") in out
    assert membership_predicate("h") in out
    assert "p:EventParticipation" in out
    assert "h:Horse" in out
    assert "[:HASHORSE]" in out


def test_optional_match_unlabeled_nodes_scoped():
    src = (
        "MATCH (h:Horse)\n"
        "OPTIONAL MATCH (t)-[:INVOLVESACTOR]->(a:Veterinarian)\n"
        "RETURN t, a"
    )
    out = inject_membership_scope(src)
    assert membership_predicate("h") in out
    assert membership_predicate("t") in out
    assert membership_predicate("a") in out
    assert "OPTIONAL MATCH" in out
    assert "[:INVOLVESACTOR]" in out


def test_membership_inject_is_idempotent():
    src = "MATCH (t)-[:INVOLVESACTOR]->(a:Veterinarian) RETURN t"
    once = inject_membership_scope(src)
    twice = inject_membership_scope(once)
    assert once == twice
    assert once.count(membership_predicate("t")) == 1
    assert once.count(membership_predicate("a")) == 1


def test_relationship_variable_is_not_scoped_as_node():
    src = "MATCH (t)-[r:INVOLVESACTOR]->(a:Veterinarian) RETURN r, t"
    out = inject_membership_scope(src)
    assert membership_predicate("t") in out
    assert membership_predicate("a") in out
    assert membership_predicate("r") not in out
    assert "[r:INVOLVESACTOR]" in out


def test_return_count_not_rewritten_as_node_pattern():
    src = "MATCH (t) RETURN count(t) AS n"
    out = inject_membership_scope(src)
    assert membership_predicate("t") in out
    assert "RETURN count(t) AS n" in out
    assert "count(t WHERE" not in out
