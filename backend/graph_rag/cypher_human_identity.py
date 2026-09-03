"""
Schema-aware, read-only correction for human identity property misuse.

When generated Cypher filters Rider / Veterinarian / Caretaker on ``id`` with a
string literal that does not match ``id`` but uniquely matches ``hasName``,
rewrite only that variable's identity property and re-execute.

Does not alter prompts, relationship patterns, membership predicates, or the
InertialSensors ``id`` → ``hasSensorID`` retry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.graph_rag.cypher_sensor_identity import (
    _context_is_empty,
    _extract_cypher_and_context,
    _run_qa,
    is_read_only_cypher,
)

HUMAN_LABELS: frozenset[str] = frozenset({"Rider", "Veterinarian", "Caretaker"})
_LABEL_CANON: dict[str, str] = {label.lower(): label for label in HUMAN_LABELS}

_HUMAN_LABEL_ALT = "Rider|Veterinarian|Caretaker"

# Node patterns that declare a human label (with optional variable).
_HUMAN_NODE = re.compile(
    rf"\(\s*(?:(?P<var>[A-Za-z_][\w]*)\s*)?:(?P<label>{_HUMAN_LABEL_ALT})\b(?P<body>[^)]*)\)",
    re.IGNORECASE,
)

# Map-style id filter inside a node pattern body: {id: "..." }
_MAP_ID_LITERAL = re.compile(
    r"\{\s*id\s*:\s*(?P<q>['\"])(?P<lit>(?:\\.|(?!(?P=q)).)*)(?P=q)",
    re.IGNORECASE,
)

# WHERE / AND style: var.id / var.hasName = "..." / CONTAINS / STARTS WITH / ENDS WITH
_PROP_ID_PRED = re.compile(
    r"\b(?P<var>[A-Za-z_][\w]*)\s*\.\s*id\s*"
    r"(?P<op>=|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*"
    r"(?P<q>['\"])(?P<lit>(?:\\.|(?!(?P=q)).)*)(?P=q)",
    re.IGNORECASE,
)

_PROP_IDENTITY_PRED = re.compile(
    r"\b(?P<var>[A-Za-z_][\w]*)\s*\.\s*(?P<prop>id|hasName)\s*"
    r"(?P<op>=|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*"
    r"(?P<q>['\"])(?P<lit>(?:\\.|(?!(?P=q)).)*)(?P=q)",
    re.IGNORECASE,
)

_MAP_IDENTITY_LITERAL = re.compile(
    r"\{\s*(?P<prop>id|hasName)\s*:\s*(?P<q>['\"])(?P<lit>(?:\\.|(?!(?P=q)).)*)(?P=q)",
    re.IGNORECASE,
)

_ALLOWED_OPS = {
    "=": "=",
    "CONTAINS": "CONTAINS",
    "STARTS WITH": "STARTS WITH",
    "ENDS WITH": "ENDS WITH",
}

RETRY_REASON_HAS_NAME = "Human literal matched hasName but not id"
RETRY_REASON_AMBIGUOUS = (
    "Human literal matched multiple hasName values; left unchanged"
)
RETRY_REASON_NO_MATCH = "Human literal did not uniquely match hasName"


@dataclass(frozen=True)
class HumanIdPredicate:
    var: str
    label: str
    op: str
    op_norm: str
    literal: str
    quote: str
    kind: str  # "prop" | "map"
    prop: str = "id"


def _unescape_literal(raw: str) -> str:
    return (
        raw.replace(r"\'", "'")
        .replace(r"\"", '"')
        .replace(r"\\", "\\")
    )


def _normalize_op(op: str) -> str | None:
    compact = re.sub(r"\s+", " ", (op or "").strip().upper())
    return _ALLOWED_OPS.get(compact)


def _human_variables(cypher: str) -> dict[str, str]:
    """Map Cypher variable → canonical human label (first declaration wins)."""
    mapping: dict[str, str] = {}
    for match in _HUMAN_NODE.finditer(cypher):
        var = match.group("var")
        if not var:
            continue
        label = _LABEL_CANON.get((match.group("label") or "").lower())
        if label and var not in mapping:
            mapping[var] = label
    return mapping


def extract_human_id_predicates(cypher: str) -> list[HumanIdPredicate]:
    """
    Collect ``.id`` / map-``id`` filters on Rider, Veterinarian, or Caretaker.

    Ignores Horse, InertialSensors, RETURN projections of ``.id`` without a
    string-literal comparison, and membership ``IN $__kg_scope`` predicates.
    """
    if not cypher:
        return []

    found: list[HumanIdPredicate] = []
    seen: set[tuple[str, str, str, str]] = set()

    def _add(pred: HumanIdPredicate) -> None:
        key = (pred.var, pred.label, pred.op_norm, pred.literal)
        if key in seen:
            return
        seen.add(key)
        found.append(pred)

    human_vars = _human_variables(cypher)

    for match in _HUMAN_NODE.finditer(cypher):
        var = match.group("var")
        label = _LABEL_CANON.get((match.group("label") or "").lower())
        body = match.group("body") or ""
        if not var or not label:
            continue
        for mid in _MAP_ID_LITERAL.finditer(body):
            _add(
                HumanIdPredicate(
                    var=var,
                    label=label,
                    op="=",
                    op_norm="=",
                    literal=_unescape_literal(mid.group("lit")),
                    quote=mid.group("q"),
                    kind="map",
                )
            )

    if human_vars:
        for match in _PROP_ID_PRED.finditer(cypher):
            var = match.group("var")
            label = human_vars.get(var)
            op_norm = _normalize_op(match.group("op"))
            if not label or not op_norm:
                continue
            _add(
                HumanIdPredicate(
                    var=var,
                    label=label,
                    op=match.group("op"),
                    op_norm=op_norm,
                    literal=_unescape_literal(match.group("lit")),
                    quote=match.group("q"),
                    kind="prop",
                )
            )

    return found


def extract_human_identity_lookups(cypher: str) -> list[HumanIdPredicate]:
    """
    Collect id/hasName string predicates on Rider, Veterinarian, or Caretaker.

    Used by QA context enrichment to resolve the queried human. Does not
    rewrite Cypher. Ignores membership ``IN $__kg_scope`` predicates.
    """
    if not cypher:
        return []

    found: list[HumanIdPredicate] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    def _add(pred: HumanIdPredicate) -> None:
        key = (pred.var, pred.label, pred.prop, pred.op_norm, pred.literal)
        if key in seen:
            return
        seen.add(key)
        found.append(pred)

    human_vars = _human_variables(cypher)

    for match in _HUMAN_NODE.finditer(cypher):
        var = match.group("var")
        label = _LABEL_CANON.get((match.group("label") or "").lower())
        body = match.group("body") or ""
        if not var or not label:
            continue
        for mid in _MAP_IDENTITY_LITERAL.finditer(body):
            prop = (mid.group("prop") or "id").lower()
            prop = "hasName" if prop == "hasname" else "id"
            _add(
                HumanIdPredicate(
                    var=var,
                    label=label,
                    op="=",
                    op_norm="=",
                    literal=_unescape_literal(mid.group("lit")),
                    quote=mid.group("q"),
                    kind="map",
                    prop=prop,
                )
            )

    if human_vars:
        for match in _PROP_IDENTITY_PRED.finditer(cypher):
            var = match.group("var")
            label = human_vars.get(var)
            op_norm = _normalize_op(match.group("op"))
            raw_prop = (match.group("prop") or "id")
            prop = "hasName" if raw_prop.lower() == "hasname" else "id"
            if not label or not op_norm:
                continue
            _add(
                HumanIdPredicate(
                    var=var,
                    label=label,
                    op=match.group("op"),
                    op_norm=op_norm,
                    literal=_unescape_literal(match.group("lit")),
                    quote=match.group("q"),
                    kind="prop",
                    prop=prop,
                )
            )

    return found


def rewrite_human_id_to_has_name(cypher: str, pred: HumanIdPredicate) -> str:
    """
    Rewrite only ``pred``'s identity property ``id`` → ``hasName``.

    Leaves other variables, other labels, RETURN ``.id``, operators, literals,
    MATCH patterns, and membership predicates unchanged.
    """
    if not cypher:
        return cypher

    def _repl_node(match: re.Match[str]) -> str:
        var = match.group("var")
        raw_label = match.group("label")
        body = match.group("body") or ""
        label = _LABEL_CANON.get((raw_label or "").lower())
        prefix = f"({var}:{raw_label}" if var else f"(:{raw_label}"
        if pred.kind != "map" or var != pred.var or label != pred.label:
            return match.group(0)

        def _repl_map(m: re.Match[str]) -> str:
            lit = _unescape_literal(m.group("lit"))
            if lit != pred.literal:
                return m.group(0)
            return re.sub(
                r"\bid\s*:",
                "hasName:",
                m.group(0),
                count=1,
                flags=re.IGNORECASE,
            )

        new_body = _MAP_ID_LITERAL.sub(_repl_map, body)
        return f"{prefix}{new_body})"

    rewritten = _HUMAN_NODE.sub(_repl_node, cypher)

    if pred.kind == "prop" or pred.kind == "map":

        def _repl_prop(match: re.Match[str]) -> str:
            if match.group("var") != pred.var:
                return match.group(0)
            lit = _unescape_literal(match.group("lit"))
            if lit != pred.literal:
                return match.group(0)
            op_norm = _normalize_op(match.group("op"))
            if op_norm != pred.op_norm:
                return match.group(0)
            q = match.group("q")
            return (
                f"{match.group('var')}.hasName {match.group('op')} "
                f"{q}{match.group('lit')}{q}"
            )

        rewritten = _PROP_ID_PRED.sub(_repl_prop, rewritten)

    return rewritten


def _count_humans_by_prop(
    graph,
    label: str,
    prop: str,
    op_norm: str,
    literal: str,
) -> int:
    """READ-ONLY existence/count check on a human label."""
    if label not in HUMAN_LABELS:
        raise ValueError(f"Unsupported human label: {label!r}")
    if prop not in {"id", "hasName"}:
        raise ValueError(f"Unsupported human identity property: {prop!r}")
    if op_norm not in _ALLOWED_OPS:
        raise ValueError(f"Unsupported identity operator: {op_norm!r}")
    cypher = (
        f"MATCH (h:{label}) WHERE h.{prop} {op_norm} $literal "
        "RETURN count(h) AS n"
    )
    if not is_read_only_cypher(cypher):
        raise RuntimeError("Refusing non-read-only human identity probe")
    rows = graph.query(cypher, {"literal": literal})
    if not rows:
        return 0
    first = rows[0]
    if isinstance(first, dict):
        return int(first.get("n") or 0)
    return int(first[0] if first else 0)


def plan_human_identity_rewrite(
    cypher: str,
    graph,
) -> tuple[str | None, str | None]:
    """
    Decide whether to rewrite ``cypher``.

    Returns ``(corrected_cypher_or_None, reason_or_None)``.
    """
    if not cypher or not is_read_only_cypher(cypher):
        return None, None

    predicates = extract_human_id_predicates(cypher)
    if not predicates:
        return None, None

    corrected = cypher
    changed = False
    ambiguous = False
    saw_has_name_miss = False

    for pred in predicates:
        id_count = _count_humans_by_prop(
            graph, pred.label, "id", pred.op_norm, pred.literal
        )
        if id_count > 0:
            # Literal is a real internal id — keep that filter as-is.
            continue

        has_count = _count_humans_by_prop(
            graph, pred.label, "hasName", pred.op_norm, pred.literal
        )
        if has_count == 0:
            saw_has_name_miss = True
            continue
        if has_count > 1:
            ambiguous = True
            continue

        new_q = rewrite_human_id_to_has_name(corrected, pred)
        if new_q != corrected:
            corrected = new_q
            changed = True

    if changed:
        if not is_read_only_cypher(corrected):
            return None, None
        return corrected, RETRY_REASON_HAS_NAME
    if ambiguous:
        return None, RETRY_REASON_AMBIGUOUS
    if saw_has_name_miss:
        return None, RETRY_REASON_NO_MATCH
    return None, None


def apply_human_identity_retry_if_needed(
    chain,
    inputs: dict,
    result: dict,
    config=None,
) -> dict:
    """
    After a successful chain invoke with empty Neo4j context, optionally correct
    Rider/Veterinarian/Caretaker ``id``→``hasName`` filters using live evidence.
    """
    original_cypher, context = _extract_cypher_and_context(result)
    if not original_cypher or not _context_is_empty(context):
        if original_cypher and "final_cypher" not in result:
            result = {**result, "final_cypher": original_cypher}
        return {
            **result,
            "human_identity_retry_used": False,
        }

    graph = chain.graph
    corrected, reason = plan_human_identity_rewrite(original_cypher, graph)
    base_original = result.get("original_cypher") or original_cypher
    if not corrected or corrected == original_cypher:
        out = {
            **result,
            "original_cypher": base_original,
            "final_cypher": result.get("final_cypher") or original_cypher,
            "human_identity_retry_used": False,
        }
        if reason in {RETRY_REASON_AMBIGUOUS, RETRY_REASON_NO_MATCH}:
            out["human_identity_retry_reason"] = reason
        return out

    top_k = getattr(chain, "top_k", 50)
    new_context = graph.query(corrected)[:top_k]
    question = inputs.get("query") or inputs.get("question") or ""
    final_answer = _run_qa(chain, question, new_context, config=config)

    chain_output_key = getattr(chain, "output_key", "result")
    out = {
        **result,
        chain_output_key: final_answer,
        "cypher_retry_used": True,
        "original_cypher": base_original,
        "final_cypher": corrected,
        "retry_reason": reason or RETRY_REASON_HAS_NAME,
        "human_identity_retry_used": True,
        "human_identity_retry_reason": reason or RETRY_REASON_HAS_NAME,
    }
    if getattr(chain, "return_intermediate_steps", False):
        out["intermediate_steps"] = [
            {"query": corrected},
            {"context": new_context},
        ]
    return out
