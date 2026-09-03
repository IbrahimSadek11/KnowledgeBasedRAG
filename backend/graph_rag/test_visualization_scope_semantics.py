"""Regression: None vs [] visualization scope (no Neo4j required)."""

from __future__ import annotations

from backend.graph_rag.visualization_export import (
    export_production_graph,
    resolve_export_scope,
)


def test_no_scope_args_means_global_not_membership_scoped():
    scope = resolve_export_scope()
    assert scope["node_scope_requested"] is False
    assert scope["rel_scope_requested"] is False
    assert scope["membership_scoped"] is False
    assert scope["scoped"] is False
    assert scope["use_exact_edges"] is False


def test_explicit_empty_node_list_is_membership_scoped_zero():
    scope = resolve_export_scope(
        stable_node_ids=[],
        relationship_ids=[],
        exact_edges=True,
    )
    assert scope["node_scope_requested"] is True
    assert scope["rel_scope_requested"] is True
    assert scope["membership_scoped"] is True
    assert scope["scoped"] is True
    assert scope["node_ids"] == []
    assert scope["rel_ids"] == []
    assert scope["use_exact_edges"] is True


def test_blank_string_markers_normalize_to_empty_but_keep_scope():
    """Nest sends stable_node_id='' to signal explicit empty membership."""
    scope = resolve_export_scope(
        stable_node_ids=[""],
        relationship_ids=[""],
        exact_edges=True,
    )
    assert scope["node_scope_requested"] is True
    assert scope["membership_scoped"] is True
    assert scope["node_ids"] == []
    assert scope["rel_ids"] == []


def test_selected_nodes_empty_rels_exact_edges():
    scope = resolve_export_scope(
        stable_node_ids=["http://ex/Horses#Horse_Dakota"],
        relationship_ids=[],
        exact_edges=True,
    )
    assert scope["membership_scoped"] is True
    assert scope["node_ids"] == ["http://ex/Horses#Horse_Dakota"]
    assert scope["rel_ids"] == []
    assert scope["use_exact_edges"] is True


def test_export_explicit_empty_returns_zero_without_full_graph():
    """Must not fall through to MATCH (n) when membership is explicitly empty."""
    result = export_production_graph(
        stable_node_ids=[],
        relationship_ids=[],
        exact_edges=True,
    )
    assert result["scoped"] is True
    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["metadata"]["scopedNodeCount"] == 0
    assert result["metadata"]["scopedEdgeCount"] == 0
    assert result["metadata"]["exactEdges"] is True
