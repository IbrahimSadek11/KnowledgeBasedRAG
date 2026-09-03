"""Tests for invoke_graph_chain_with_membership_scope empty/rel semantics."""
from __future__ import annotations

import pytest

from backend.graph_rag.cypher_membership_scope import (
    MembershipScopeError,
    _REL_SCOPE_PARAM,
    _SCOPE_PARAM,
    invoke_graph_chain_with_membership_scope,
    inject_membership_scope,
)


class _FakeGraph:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def query(self, cypher, params=None, *args, **kwargs):
        self.calls.append((cypher, dict(params or {})))
        return [{"ok": True}]


class _FakeChain:
    def __init__(self, graph):
        self.graph = graph


def test_unscoped_when_both_membership_args_none():
    graph = _FakeGraph()
    chain = _FakeChain(graph)
    seen = {}

    def _invoke(c, inputs, config=None):
        seen["ran"] = True
        # Direct unscoped path — no patch required to succeed.
        return {"result": "full"}

    out = invoke_graph_chain_with_membership_scope(
        chain,
        {"query": "q"},
        stable_node_ids=None,
        relationship_ids=None,
        _invoke=_invoke,
    )
    assert out["result"] == "full"
    assert seen.get("ran") is True
    assert graph.calls == []


def test_empty_membership_lists_do_not_fallback_to_unscoped():
    graph = _FakeGraph()
    chain = _FakeChain(graph)

    def _invoke(c, inputs, config=None):
        # Simulate chain executing Cypher via graph.query
        c.graph.query("MATCH (h)-[:TRAINSIN]->(t) RETURN t")
        return {"result": "empty-scope"}

    out = invoke_graph_chain_with_membership_scope(
        chain,
        {"query": "q"},
        stable_node_ids=[],
        relationship_ids=[],
        _invoke=_invoke,
    )
    assert out["membership_scoped"] is True
    assert out["membership_node_count"] == 0
    assert out["membership_relationship_count"] == 0
    assert len(graph.calls) == 1
    cypher, params = graph.calls[0]
    assert _SCOPE_PARAM in params
    assert params[_SCOPE_PARAM] == []
    assert _REL_SCOPE_PARAM in params
    assert params[_REL_SCOPE_PARAM] == []
    assert "__kg_rel_0:TRAINSIN" in cypher


def test_relationship_ids_enforced_with_nodes():
    graph = _FakeGraph()
    chain = _FakeChain(graph)
    rel_id = "TRAINSIN:http://ex/#H1:http://ex/#S1"

    def _invoke(c, inputs, config=None):
        c.graph.query("MATCH (h)-[r:TRAINSIN]->(t) RETURN r")
        return {"result": "ok"}

    invoke_graph_chain_with_membership_scope(
        chain,
        {"query": "q"},
        stable_node_ids=["http://ex/#H1", "http://ex/#S1"],
        relationship_ids=[rel_id],
        _invoke=_invoke,
    )
    cypher, params = graph.calls[0]
    assert params[_SCOPE_PARAM] == ["http://ex/#H1", "http://ex/#S1"]
    assert params[_REL_SCOPE_PARAM] == [rel_id]
    assert f"IN ${_REL_SCOPE_PARAM}" in cypher or f"IN $__kg_rel_scope" in cypher


def test_unsupported_varlen_raises_during_scoped_query():
    graph = _FakeGraph()
    chain = _FakeChain(graph)

    def _invoke(c, inputs, config=None):
        c.graph.query("MATCH (h)-[:TRAINSIN*1..2]->(t) RETURN t")
        return {"result": "should-not"}

    with pytest.raises(MembershipScopeError):
        invoke_graph_chain_with_membership_scope(
            chain,
            {"query": "q"},
            stable_node_ids=["a"],
            relationship_ids=["TRAINSIN:a:b"],
            _invoke=_invoke,
        )
    assert graph.calls == []


def test_retry_path_uses_same_patched_query():
    """Corrected Cypher must also receive node+relationship params."""
    graph = _FakeGraph()
    chain = _FakeChain(graph)

    def _invoke(c, inputs, config=None):
        c.graph.query("MATCH (h)-[:TRAINSIN]->(t) RETURN t")
        # Simulate sensor/error retry calling graph.query again
        c.graph.query("MATCH (h)-[r:TRAINSIN]->(t) RETURN r")
        return {"result": "retried", "cypher_retry_used": True}

    invoke_graph_chain_with_membership_scope(
        chain,
        {"query": "q"},
        stable_node_ids=["n1"],
        relationship_ids=["TRAINSIN:n1:n2"],
        _invoke=_invoke,
    )
    assert len(graph.calls) == 2
    for cypher, params in graph.calls:
        assert params[_SCOPE_PARAM] == ["n1"]
        assert params[_REL_SCOPE_PARAM] == ["TRAINSIN:n1:n2"]
        assert "$__kg_rel_scope" in cypher or "__kg_rel_" in cypher or "[r:TRAINSIN WHERE" in cypher
