"""
Deterministic Knowledge Graph membership scope for arbitrary read-only Cypher.

When stable_node_ids are provided (same uri/id/hasSensorID keys as visualization
export), every node variable in MATCH / OPTIONAL MATCH patterns is constrained
so MATCH / aggregations only see members of the selected graph — including
unlabeled variables such as ``(t)`` or ``(event)``.

When relationship_ids are provided, every traversed relationship is constrained
to the composite membership id:

    type(r) + ':' + stable(startNode(r)) + ':' + stable(endNode(r))

matching dynamic_ingestion_writer (TYPE:sourceStable:targetStable).

Does not modify prompts, sensor-identity retry, or human-identity retry;
wraps graph.query only.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

from backend.graph_rag.cypher_sensor_identity import is_read_only_cypher
from backend.graph_rag.membership_identity import relationship_membership_cypher_expr

_SCOPE_PARAM = "__kg_scope"
_REL_SCOPE_PARAM = "__kg_rel_scope"

# MATCH / OPTIONAL MATCH keyword (longer alternative first).
_MATCH_KW = re.compile(r"\b(?:OPTIONAL\s+MATCH|MATCH)\b", re.IGNORECASE)

# Clause keywords that end a MATCH pattern when seen at paren/bracket depth 0.
_PATTERN_END = re.compile(
    r"\s+(?:WHERE|RETURN|WITH|UNWIND|ORDER\s+BY|SKIP|LIMIT|UNION|FOREACH|"
    r"OPTIONAL\s+MATCH|MATCH)\b",
    re.IGNORECASE,
)

_NODE_VAR = re.compile(
    r"^\s*(?P<var>[A-Za-z_][\w]*)(?![\w.])",
)

_ALREADY_NODE_SCOPED = re.compile(
    rf"\${_SCOPE_PARAM}\b|IN\s+\${_SCOPE_PARAM}\b", re.IGNORECASE
)

_ALREADY_REL_SCOPED = re.compile(
    rf"\${_REL_SCOPE_PARAM}\b|IN\s+\${_REL_SCOPE_PARAM}\b", re.IGNORECASE
)

# Relationship pattern body: [var?][:types]?[*varlen]? [WHERE ...]?
_REL_BODY = re.compile(
    r"^\s*"
    r"(?P<var>[A-Za-z_][\w]*)?"
    r"(?:\s*:\s*(?P<types>[A-Za-z_][\w]*(?:\s*\|\s*[A-Za-z_][\w]*)*))?"
    r"(?P<varlen>\s*\*)?"
    r"(?P<rest>.*)$",
    re.DOTALL,
)

_CYPHER_IDENT = re.compile(r"\b([A-Za-z_][\w]*)\b")


class MembershipScopeError(ValueError):
    """Selected-graph membership cannot safely scope this Cypher; fail closed."""


def membership_predicate(var: str) -> str:
    """
    Same identity keys as visualization_export / approve stable_id.

    Use OR across uri / id / hasSensorID — NOT coalesce(). Coalesce only tests
    the first non-null property, so a node stored in membership by id or
    hasSensorID would be excluded whenever uri is also set (and differs).
    """
    return (
        f"({var}.uri IN ${_SCOPE_PARAM} "
        f"OR {var}.id IN ${_SCOPE_PARAM} "
        f"OR toString({var}.hasSensorID) IN ${_SCOPE_PARAM})"
    )


def relationship_membership_predicate(rel_var: str) -> str:
    """Full-string IN check against $__kg_rel_scope (never split on ':')."""
    return (
        f"({relationship_membership_cypher_expr(rel_var)} "
        f"IN ${_REL_SCOPE_PARAM})"
    )


def _scan_quoted(text: str, i: int) -> int:
    """Return index after a single- or double-quoted Cypher string at ``i``."""
    quote = text[i]
    i += 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == quote:
            return i + 1
        i += 1
    return n


def _matching_closer(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index after the closer that matches ``text[start] == open_ch``."""
    depth = 1
    i = start + 1
    n = len(text)
    while i < n and depth:
        ch = text[i]
        if ch in ("'", '"'):
            i = _scan_quoted(text, i)
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
        i += 1
    return i


def _pattern_end(cypher: str, start: int) -> int:
    """End index of a MATCH pattern starting at ``start`` (after the keyword)."""
    depth_paren = 0
    depth_brack = 0
    i = start
    n = len(cypher)
    while i < n:
        ch = cypher[i]
        if ch in ("'", '"'):
            i = _scan_quoted(cypher, i)
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif depth_paren == 0 and depth_brack == 0:
            match = _PATTERN_END.match(cypher, i)
            if match:
                return i
        i += 1
    return n


def _iter_node_spans(pattern: str) -> list[tuple[int, int]]:
    """
    Absolute spans of node ``(...)`` patterns inside a MATCH pattern string.

    Skips relationship ``[...]`` blocks so relationship variables are not
    treated as nodes.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch in ("'", '"'):
            i = _scan_quoted(pattern, i)
            continue
        if ch == "[":
            i = _matching_closer(pattern, i, "[", "]")
            continue
        if ch == "(":
            end = _matching_closer(pattern, i, "(", ")")
            spans.append((i, end))
            i = end
            continue
        i += 1
    return spans


def _iter_rel_spans(pattern: str) -> list[tuple[int, int]]:
    """Absolute spans of relationship ``[...]`` patterns (skips node parens)."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch in ("'", '"'):
            i = _scan_quoted(pattern, i)
            continue
        if ch == "(":
            i = _matching_closer(pattern, i, "(", ")")
            continue
        if ch == "[":
            end = _matching_closer(pattern, i, "[", "]")
            spans.append((i, end))
            i = end
            continue
        i += 1
    return spans


def _collect_identifiers(cypher: str) -> set[str]:
    """Best-effort identifier set for collision-safe generated rel vars."""
    return set(_CYPHER_IDENT.findall(cypher or ""))


def _next_rel_var(used: set[str]) -> str:
    idx = 0
    while True:
        name = f"__kg_rel_{idx}"
        if name not in used:
            used.add(name)
            return name
        idx += 1


def _scope_node_body(body: str) -> str | None:
    """
    Return a rewritten node-pattern body with membership applied, or None
    if this is not a bindable node variable / already scoped.
    """
    match = _NODE_VAR.match(body)
    if not match:
        return None
    var = match.group("var")
    if _ALREADY_NODE_SCOPED.search(body):
        return None
    pred = membership_predicate(var)
    stripped = body.rstrip()
    if re.search(r"\bWHERE\b", stripped, re.IGNORECASE):
        return f"{stripped} AND {pred}"
    return f"{stripped} WHERE {pred}"


def _scope_rel_body(body: str, used_idents: set[str]) -> str:
    """
    Rewrite a relationship pattern body so the relationship is membership-scoped.

    Fail closed on variable-length patterns.
    """
    if _ALREADY_REL_SCOPED.search(body):
        return body

    match = _REL_BODY.match(body)
    if not match:
        raise MembershipScopeError(
            "Selected-graph edge scope cannot parse relationship pattern; "
            "refusing unscoped execution"
        )

    if match.group("varlen"):
        raise MembershipScopeError(
            "Selected-graph edge scope does not support variable-length "
            "relationship patterns (*); refusing unscoped execution"
        )

    var = match.group("var")
    types = match.group("types")
    rest = (match.group("rest") or "").strip()

    # Reject leftover junk that is not a WHERE clause (e.g. unsupported syntax).
    if rest and not re.match(r"WHERE\b", rest, re.IGNORECASE):
        raise MembershipScopeError(
            f"Selected-graph edge scope cannot safely handle relationship "
            f"syntax [{body.strip()}]; refusing unscoped execution"
        )

    if not var:
        var = _next_rel_var(used_idents)

    pred = relationship_membership_predicate(var)
    type_part = f":{types}" if types else ""
    if rest:
        # rest starts with WHERE ...
        where_body = re.sub(r"^WHERE\s+", "", rest, count=1, flags=re.IGNORECASE)
        return f"{var}{type_part} WHERE ({where_body}) AND {pred}"
    return f"{var}{type_part} WHERE {pred}"


def _scope_match_pattern(
    pattern: str,
    *,
    scope_nodes: bool,
    scope_relationships: bool,
    used_idents: set[str],
) -> str:
    # Relationships first so generated vars are registered before node pass.
    if scope_relationships:
        pieces: list[str] = []
        last = 0
        for start, end in _iter_rel_spans(pattern):
            body = pattern[start + 1 : end - 1]
            rewritten = _scope_rel_body(body, used_idents)
            pieces.append(pattern[last:start])
            pieces.append(f"[{rewritten}]")
            last = end
        pieces.append(pattern[last:])
        pattern = "".join(pieces)

    if scope_nodes:
        pieces = []
        last = 0
        for start, end in _iter_node_spans(pattern):
            body = pattern[start + 1 : end - 1]
            rewritten = _scope_node_body(body)
            pieces.append(pattern[last:start])
            if rewritten is None:
                pieces.append(pattern[start:end])
            else:
                pieces.append(f"({rewritten})")
            last = end
        pieces.append(pattern[last:])
        pattern = "".join(pieces)

    return pattern


def inject_membership_scope(
    cypher: str,
    *,
    scope_nodes: bool = True,
    scope_relationships: bool = False,
) -> str:
    """
    Rewrite MATCH / OPTIONAL MATCH patterns for membership.

    Nodes (when ``scope_nodes``): each bound node variable must be in $__kg_scope.

    Relationships (when ``scope_relationships``): each relationship must satisfy
    the composite membership id IN $__kg_rel_scope.
    """
    if not cypher or not str(cypher).strip():
        return cypher
    if not scope_nodes and not scope_relationships:
        return cypher

    used = _collect_identifiers(cypher)
    pieces: list[str] = []
    last = 0
    for match in _MATCH_KW.finditer(cypher):
        pat_start = match.end()
        pat_end = _pattern_end(cypher, pat_start)
        pieces.append(cypher[last:pat_start])
        pieces.append(
            _scope_match_pattern(
                cypher[pat_start:pat_end],
                scope_nodes=scope_nodes,
                scope_relationships=scope_relationships,
                used_idents=used,
            )
        )
        last = pat_end
    pieces.append(cypher[last:])
    return "".join(pieces)


def normalize_stable_node_ids(ids: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids or []:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


normalize_relationship_ids = normalize_stable_node_ids


def invoke_graph_chain_with_membership_scope(
    chain,
    inputs: dict,
    *,
    stable_node_ids: Sequence[str] | None = None,
    relationship_ids: Sequence[str] | None = None,
    config=None,
    _invoke: Callable[..., dict] | None = None,
) -> dict:
    """
    Run the existing Graph RAG invoke path, optionally constraining Neo4j reads
    to the given membership id sets (visualization-compatible).

    Scoping is requested when either argument is not None (including empty
    lists). Empty membership must NOT fall through to the full production graph.
    """
    if _invoke is None:
        from backend.graph_rag.cypher_sensor_identity import (
            invoke_graph_chain_with_cypher_retry as _invoke,
        )

    node_requested = stable_node_ids is not None
    rel_requested = relationship_ids is not None
    if not node_requested and not rel_requested:
        return _invoke(chain, inputs, config=config)

    node_scope = normalize_stable_node_ids(stable_node_ids)
    rel_scope = normalize_relationship_ids(relationship_ids)
    # Enforce relationships whenever the caller supplied the field (even []).
    scope_relationships = rel_requested

    graph = chain.graph
    original_query = graph.query

    def _scoped_query(query: str, params: dict | None = None, *args, **kwargs):
        params = dict(params or {})
        cypher = query if isinstance(query, str) else str(query)
        if not is_read_only_cypher(cypher):
            raise PermissionError(
                "Refusing non-read-only Cypher under knowledge-graph scope"
            )
        try:
            rewritten = inject_membership_scope(
                cypher,
                scope_nodes=node_requested,
                scope_relationships=scope_relationships,
            )
        except MembershipScopeError:
            raise
        if node_requested:
            params[_SCOPE_PARAM] = node_scope
        if scope_relationships:
            params[_REL_SCOPE_PARAM] = rel_scope
        return original_query(rewritten, params, *args, **kwargs)

    graph.query = _scoped_query  # type: ignore[method-assign]
    try:
        result = _invoke(chain, inputs, config=config)
        result = {
            **result,
            "membership_scoped": True,
            "membership_node_count": len(node_scope) if node_requested else None,
            "membership_relationship_count": (
                len(rel_scope) if scope_relationships else None
            ),
        }
        return result
    finally:
        graph.query = original_query  # type: ignore[method-assign]
