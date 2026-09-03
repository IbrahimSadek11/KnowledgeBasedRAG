"""Unit tests for exact-edge visualization membership filtering (no Neo4j)."""
from __future__ import annotations

from backend.graph_rag.membership_identity import relationship_membership_id
from backend.graph_rag.visualization_export import _exact_edge_predicate_cypher


def test_exact_edge_predicate_uses_composite_identity():
    pred = _exact_edge_predicate_cypher()
    assert "IN $rel_ids" in pred
    assert "type(r)" in pred
    assert "a.uri" in pred
    assert "b.uri" in pred
    # Must not use elementId(r) / r.id for membership.
    assert "elementId(r)" not in pred
    assert "r.id" not in pred


def test_non_member_parallel_edge_identity_differs():
    """Two member nodes, two different typed edges → distinct membership ids."""
    a = {"uri": "http://ex/Horses#A"}
    b = {"uri": "http://ex/Horses#B"}
    r1 = relationship_membership_id("TRAINSIN", a, b)
    r2 = relationship_membership_id("DEPENDSON", a, b)
    assert r1 != r2
    selected = {r1}
    assert r2 not in selected


def test_uri_colon_ids_use_full_string_equality():
    a = {"uri": "http://ex/Horses#Horse_Dakota"}
    b = {"uri": "http://ex/Horses#Prep_01"}
    mid = relationship_membership_id("TRAINSIN", a, b)
    assert mid in {mid}
    assert mid not in {"TRAINSIN", "Horse_Dakota", "Prep_01"}
