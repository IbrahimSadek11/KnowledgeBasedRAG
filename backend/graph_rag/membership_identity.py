"""
Shared Knowledge Graph membership identity helpers.

Must stay aligned with dynamic_ingestion_writer.stable_node_id priority:
    uri → id → hasSensorID → hasName

Relationship membership id (writer):
    f"{type}:{source_stable}:{target_stable}"

Do NOT split persisted relationship ids on ":"; URIs contain ":".
Compare only by full-string equality / IN.
"""
from __future__ import annotations

from typing import Any


def stable_node_id_from_props(props: dict[str, Any] | None) -> str | None:
    """Python-side stable node id (same priority as dynamic_ingestion_writer)."""
    props = props or {}
    for key in ("uri", "id", "hasSensorID", "hasName"):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def relationship_membership_id(
    rel_type: str,
    start_props: dict[str, Any] | None,
    end_props: dict[str, Any] | None,
) -> str | None:
    """Composite TYPE:source:target membership id, or None if endpoints lack stables."""
    source = stable_node_id_from_props(start_props)
    target = stable_node_id_from_props(end_props)
    if not rel_type or source is None or target is None:
        return None
    return f"{rel_type}:{source}:{target}"


def stable_node_cypher_expr(node_expr: str) -> str:
    """
    Cypher expression for stable node identity of ``node_expr``
    (variable or startNode(r) / endNode(r)).
    """
    n = node_expr
    return (
        f"CASE "
        f"WHEN {n}.uri IS NOT NULL AND toString({n}.uri) <> '' "
        f"THEN toString({n}.uri) "
        f"WHEN {n}.id IS NOT NULL AND toString({n}.id) <> '' "
        f"THEN toString({n}.id) "
        f"WHEN {n}.hasSensorID IS NOT NULL AND toString({n}.hasSensorID) <> '' "
        f"THEN toString({n}.hasSensorID) "
        f"WHEN {n}.hasName IS NOT NULL AND toString({n}.hasName) <> '' "
        f"THEN toString({n}.hasName) "
        f"ELSE null END"
    )


def relationship_membership_cypher_expr(rel_var: str) -> str:
    """
    Cypher expression reconstructing TYPE:source:target for relationship ``rel_var``.

    Uses startNode/endNode so stored Neo4j direction is respected regardless of
    textual left/right pattern order.
    """
    start_stable = stable_node_cypher_expr(f"startNode({rel_var})")
    end_stable = stable_node_cypher_expr(f"endNode({rel_var})")
    return f"(type({rel_var}) + ':' + {start_stable} + ':' + {end_stable})"
