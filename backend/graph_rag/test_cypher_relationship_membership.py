"""Unit tests for relationship membership identity + Cypher edge scoping."""
from __future__ import annotations

import pytest

from backend.graph_rag.cypher_membership_scope import (
    MembershipScopeError,
    inject_membership_scope,
    membership_predicate,
    normalize_relationship_ids,
    relationship_membership_predicate,
)
from backend.graph_rag.membership_identity import (
    relationship_membership_cypher_expr,
    relationship_membership_id,
    stable_node_cypher_expr,
    stable_node_id_from_props,
)


def test_stable_node_id_priority_skips_empty_string():
    assert (
        stable_node_id_from_props({"uri": "", "id": "Horse_Dakota"})
        == "Horse_Dakota"
    )
    assert (
        stable_node_id_from_props(
            {"uri": "http://ex/Horses#Horse_Dakota", "id": "Horse_Dakota"}
        )
        == "http://ex/Horses#Horse_Dakota"
    )


def test_relationship_membership_id_matches_writer_format():
    mid = relationship_membership_id(
        "TRAINSIN",
        {"uri": "http://ex/Horses#Horse_Dakota"},
        {"uri": "http://ex/Horses#Prep_01"},
    )
    assert mid == (
        "TRAINSIN:http://ex/Horses#Horse_Dakota:http://ex/Horses#Prep_01"
    )
    # Colon-rich URIs: full-string equality only (never split).
    assert "://" in mid
    assert mid.startswith("TRAINSIN:")
    assert mid == mid  # identity compare by full string


def test_relationship_membership_cypher_uses_start_end_node():
    expr = relationship_membership_cypher_expr("r")
    assert "startNode(r)" in expr
    assert "endNode(r)" in expr
    assert "type(r)" in expr
    assert "hasSensorID" in expr


def test_named_relationship_gets_edge_predicate():
    src = "MATCH (h)-[r:TRAINSIN]->(t) RETURN h, r, t"
    out = inject_membership_scope(src, scope_relationships=True)
    assert membership_predicate("h") in out
    assert membership_predicate("t") in out
    assert relationship_membership_predicate("r") in out
    assert "[r:TRAINSIN WHERE" in out or "[r:TRAINSIN WHERE " in out


def test_anonymous_relationship_binds_generated_var():
    src = "MATCH (h)-[:TRAINSIN]->(t) RETURN h, t"
    out = inject_membership_scope(src, scope_relationships=True)
    assert "[__kg_rel_0:TRAINSIN WHERE" in out
    assert relationship_membership_predicate("__kg_rel_0") in out
    assert "[:TRAINSIN]" not in out


def test_two_hop_scopes_both_relationships():
    src = "MATCH (h)-[:TRAINSIN]->(t)-[:DEPENDSON]->(e) RETURN h, t, e"
    out = inject_membership_scope(src, scope_relationships=True)
    assert "__kg_rel_0:TRAINSIN" in out
    assert "__kg_rel_1:DEPENDSON" in out
    assert relationship_membership_predicate("__kg_rel_0") in out
    assert relationship_membership_predicate("__kg_rel_1") in out


def test_multiple_named_relationships():
    src = "MATCH (h)-[r1:TRAINSIN]->(t)-[r2:DEPENDSON]->(e) RETURN r1, r2"
    out = inject_membership_scope(src, scope_relationships=True)
    assert relationship_membership_predicate("r1") in out
    assert relationship_membership_predicate("r2") in out


def test_optional_match_keeps_optional_keyword_and_scopes_rel():
    src = (
        "MATCH (h:Horse)\n"
        "OPTIONAL MATCH (h)-[:TRAINSIN]->(t)\n"
        "RETURN h, t"
    )
    out = inject_membership_scope(src, scope_relationships=True)
    assert "OPTIONAL MATCH" in out
    assert "__kg_rel_0:TRAINSIN" in out
    assert relationship_membership_predicate("__kg_rel_0") in out


def test_relationship_only_aggregation_scoped():
    src = "MATCH ()-[r]->() RETURN type(r), COUNT(r) AS n"
    out = inject_membership_scope(src, scope_relationships=True)
    assert relationship_membership_predicate("r") in out
    assert "RETURN type(r), COUNT(r) AS n" in out


def test_existing_where_on_match_preserved_with_node_and_edge_scope():
    src = (
        "MATCH (h:Horse)-[r:TRAINSIN]->(t) "
        "WHERE h.hasName = 'Dakota' RETURN t"
    )
    out = inject_membership_scope(src, scope_relationships=True)
    assert "h.hasName = 'Dakota'" in out
    assert membership_predicate("h") in out
    assert relationship_membership_predicate("r") in out


def test_multiple_match_clauses_each_scoped():
    src = (
        "MATCH (h:Horse)-[:TRAINSIN]->(t)\n"
        "MATCH (t)-[:DEPENDSON]->(e)\n"
        "RETURN e"
    )
    out = inject_membership_scope(src, scope_relationships=True)
    assert out.count("WHERE") >= 2
    assert "TRAINSIN" in out
    assert "DEPENDSON" in out


def test_variable_length_relationship_fails_closed():
    src = "MATCH (h)-[:TRAINSIN*1..3]->(t) RETURN h, t"
    with pytest.raises(MembershipScopeError):
        inject_membership_scope(src, scope_relationships=True)


def test_node_only_inject_leaves_anonymous_rel_unchanged():
    """Backward compatible default: no edge rewrite."""
    src = "MATCH (h)-[:TRAINSIN]->(t) RETURN h, t"
    out = inject_membership_scope(src)
    assert "[:TRAINSIN]" in out
    assert "__kg_rel_" not in out


def test_edge_inject_idempotent():
    src = "MATCH (h)-[r:TRAINSIN]->(t) RETURN r"
    once = inject_membership_scope(src, scope_relationships=True)
    twice = inject_membership_scope(once, scope_relationships=True)
    assert once == twice
    assert once.count(relationship_membership_predicate("r")) == 1


def test_normalize_relationship_ids():
    assert normalize_relationship_ids(
        [" TRAINSIN:a:b ", "TRAINSIN:a:b", "", "X:y:z"]
    ) == ["TRAINSIN:a:b", "X:y:z"]


def test_incoming_named_relationship_still_uses_rel_var():
    src = "MATCH (h)<-[r:ISATTACHEDTO]-(s) RETURN h, s"
    out = inject_membership_scope(src, scope_relationships=True)
    assert relationship_membership_predicate("r") in out
    assert "startNode(r)" in relationship_membership_cypher_expr("r")


def test_stable_node_cypher_expr_priority_structure():
    expr = stable_node_cypher_expr("n")
    assert expr.index("n.uri") < expr.index("n.id")
    assert expr.index("n.id") < expr.index("n.hasSensorID")
    assert expr.index("n.hasSensorID") < expr.index("n.hasName")
