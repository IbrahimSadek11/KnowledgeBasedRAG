"""
Schema-aware, read-only correction for InertialSensors identity property misuse.

When generated Cypher filters InertialSensors on ``id`` with a literal that does
not exist as ``id`` but uniquely matches ``hasSensorID``, rewrite only that
property name and re-execute. Does not alter relationship patterns or prompts.
"""
from __future__ import annotations

import re
from typing import Any, Callable

# Read-only guard: reject any mutation / DDL before re-execution.
_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH\s+DELETE|REMOVE|DROP|LOAD\s+CSV|CALL\s+\w*\.|FOREACH)\b",
    re.IGNORECASE,
)

# Node patterns that declare :InertialSensors (with optional variable).
_INERTIAL_NODE = re.compile(
    r"\(\s*(?:(?P<var>[A-Za-z_][\w]*)\s*)?:InertialSensors\b(?P<body>[^)]*)\)",
    re.IGNORECASE,
)

# Map-style id filter inside a node pattern body: {id: "..." } / { id : '...' }
_MAP_ID_LITERAL = re.compile(
    r"\{\s*id\s*:\s*(?P<q>['\"])(?P<lit>(?:\\.|(?!(?P=q)).)*)(?P=q)",
    re.IGNORECASE,
)

# WHERE / AND style: <var>.id = "..."
_PROP_ID_EQ = re.compile(
    r"\b(?P<var>[A-Za-z_][\w]*)\s*\.\s*id\s*=\s*(?P<q>['\"])(?P<lit>(?:\\.|(?!(?P=q)).)*)(?P=q)",
    re.IGNORECASE,
)

RETRY_REASON_HAS_SENSOR_ID = (
    "Sensor literal matched hasSensorID but not id"
)
RETRY_REASON_AMBIGUOUS = (
    "Sensor literal matched multiple hasSensorID values; left unchanged"
)


def is_read_only_cypher(cypher: str) -> bool:
    """Return True when the query has no write/DDL keywords."""
    if not cypher or not str(cypher).strip():
        return False
    return _WRITE_PATTERN.search(cypher) is None


def _unescape_literal(raw: str) -> str:
    return (
        raw.replace(r"\'", "'")
        .replace(r"\"", '"')
        .replace(r"\\", "\\")
    )


def _inertial_variables(cypher: str) -> set[str]:
    vars_found: set[str] = set()
    for match in _INERTIAL_NODE.finditer(cypher):
        var = match.group("var")
        if var:
            vars_found.add(var)
    return vars_found


def extract_inertial_id_literals(cypher: str) -> list[str]:
    """
    Collect distinct string literals used as InertialSensors ``id`` filters.

    Covers map form ``(:InertialSensors {id: "..."})`` and
    ``WHERE <var>.id = "..."`` when ``<var>`` is bound as InertialSensors.
    Does not treat RETURN projections of ``.id`` as filters.
    """
    literals: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        lit = _unescape_literal(raw)
        if lit not in seen:
            seen.add(lit)
            literals.append(lit)

    for match in _INERTIAL_NODE.finditer(cypher):
        body = match.group("body") or ""
        for mid in _MAP_ID_LITERAL.finditer(body):
            _add(mid.group("lit"))

    inertial_vars = _inertial_variables(cypher)
    if inertial_vars:
        for match in _PROP_ID_EQ.finditer(cypher):
            if match.group("var") in inertial_vars:
                # Skip if this equality sits inside a RETURN clause projection
                # only when it is clearly not a filter — keep simple: equality
                # on id for an InertialSensors var is treated as a filter.
                _add(match.group("lit"))

    return literals


def rewrite_inertial_id_literal_to_has_sensor_id(cypher: str, literal: str) -> str:
    """
    Replace InertialSensors ``id`` filters for ``literal`` with ``hasSensorID``.

    Only touches map ``{id: <literal>}`` inside ``:InertialSensors`` node
    patterns and ``<var>.id = <literal>`` for InertialSensors variables.
    Leaves RETURN ``s.id``, other labels, and unrelated literals unchanged.
    """
    if not cypher or literal is None:
        return cypher

    inertial_vars = _inertial_variables(cypher)

    def _repl_node(match: re.Match[str]) -> str:
        var = match.group("var")
        body = match.group("body") or ""
        prefix = f"({var}:InertialSensors" if var else "(:InertialSensors"

        def _repl_map(m: re.Match[str]) -> str:
            lit = _unescape_literal(m.group("lit"))
            if lit != literal:
                return m.group(0)
            return re.sub(
                r"\bid\s*:",
                "hasSensorID:",
                m.group(0),
                count=1,
                flags=re.IGNORECASE,
            )

        new_body = _MAP_ID_LITERAL.sub(_repl_map, body)
        return f"{prefix}{new_body})"

    rewritten = _INERTIAL_NODE.sub(_repl_node, cypher)

    if inertial_vars:

        def _repl_eq(match: re.Match[str]) -> str:
            if match.group("var") not in inertial_vars:
                return match.group(0)
            lit = _unescape_literal(match.group("lit"))
            if lit != literal:
                return match.group(0)
            q = match.group("q")
            return f"{match.group('var')}.hasSensorID = {q}{match.group('lit')}{q}"

        rewritten = _PROP_ID_EQ.sub(_repl_eq, rewritten)

    return rewritten


def _count_sensors_by_prop(graph, prop: str, literal: str) -> int:
    """READ-ONLY existence/count check. ``prop`` is id or hasSensorID only."""
    if prop not in {"id", "hasSensorID"}:
        raise ValueError(f"Unsupported sensor identity property: {prop!r}")
    cypher = (
        f"MATCH (s:InertialSensors) WHERE s.{prop} = $literal "
        "RETURN count(s) AS n"
    )
    if not is_read_only_cypher(cypher):
        raise RuntimeError("Refusing non-read-only identity probe")
    rows = graph.query(cypher, {"literal": literal})
    if not rows:
        return 0
    first = rows[0]
    if isinstance(first, dict):
        return int(first.get("n") or 0)
    return int(first[0] if first else 0)


def _context_is_empty(context: Any) -> bool:
    if context is None:
        return True
    if isinstance(context, list):
        return len(context) == 0
    return False


def _extract_cypher_and_context(result: dict) -> tuple[str | None, Any]:
    steps = result.get("intermediate_steps") or []
    cypher = None
    context = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        if cypher is None and step.get("query"):
            cypher = step["query"]
        if "context" in step:
            context = step.get("context")
    return cypher, context


def plan_sensor_identity_rewrite(
    cypher: str,
    graph,
) -> tuple[str | None, str | None]:
    """
    Decide whether to rewrite ``cypher``.

    Returns ``(corrected_cypher_or_None, reason_or_None)``.
    """
    if not cypher or not is_read_only_cypher(cypher):
        return None, None

    literals = extract_inertial_id_literals(cypher)
    if not literals:
        return None, None

    corrected = cypher
    changed = False
    ambiguous = False

    for lit in literals:
        id_count = _count_sensors_by_prop(graph, "id", lit)
        if id_count > 0:
            # Literal is a real internal id — keep that filter as-is.
            continue

        has_count = _count_sensors_by_prop(graph, "hasSensorID", lit)
        if has_count == 0:
            continue
        if has_count > 1:
            ambiguous = True
            continue

        new_q = rewrite_inertial_id_literal_to_has_sensor_id(corrected, lit)
        if new_q != corrected:
            corrected = new_q
            changed = True

    if ambiguous and not changed:
        return None, RETRY_REASON_AMBIGUOUS
    if changed:
        if not is_read_only_cypher(corrected):
            return None, None
        return corrected, RETRY_REASON_HAS_SENSOR_ID
    return None, None


def _run_qa(chain, question: str, context: Any, config=None) -> Any:
    qa_callbacks = None
    if config and isinstance(config, dict):
        qa_callbacks = config.get("callbacks")
    qa_inputs = {"question": question, "context": context}
    if qa_callbacks is not None:
        qa_out = chain.qa_chain.invoke(qa_inputs, callbacks=qa_callbacks)
    else:
        qa_out = chain.qa_chain.invoke(qa_inputs)
    output_key = getattr(chain.qa_chain, "output_key", "text")
    if isinstance(qa_out, dict):
        return qa_out.get(output_key, qa_out)
    return qa_out


def apply_sensor_identity_retry_if_needed(
    chain,
    inputs: dict,
    result: dict,
    config=None,
) -> dict:
    """
    After a successful chain invoke with empty Neo4j context, optionally correct
    InertialSensors ``id``→``hasSensorID`` filters using live schema evidence.
    """
    original_cypher, context = _extract_cypher_and_context(result)
    if not original_cypher or not _context_is_empty(context):
        # Preserve prior error-correction metadata; attach final_cypher for clarity.
        if original_cypher and "final_cypher" not in result:
            result = {**result, "final_cypher": original_cypher}
        return result

    graph = chain.graph
    corrected, reason = plan_sensor_identity_rewrite(original_cypher, graph)
    if not corrected or corrected == original_cypher:
        if reason == RETRY_REASON_AMBIGUOUS:
            return {
                **result,
                "original_cypher": result.get("original_cypher") or original_cypher,
                "final_cypher": original_cypher,
                "retry_reason": reason,
            }
        return {
            **result,
            "final_cypher": original_cypher,
        }

    top_k = getattr(chain, "top_k", 50)
    new_context = graph.query(corrected)[:top_k]
    question = inputs.get("query") or inputs.get("question") or ""
    final_answer = _run_qa(chain, question, new_context, config=config)

    chain_output_key = getattr(chain, "output_key", "result")
    out = {
        **result,
        chain_output_key: final_answer,
        "cypher_retry_used": True,
        "original_cypher": result.get("original_cypher") or original_cypher,
        "final_cypher": corrected,
        "retry_reason": reason or RETRY_REASON_HAS_SENSOR_ID,
    }
    if getattr(chain, "return_intermediate_steps", False):
        out["intermediate_steps"] = [
            {"query": corrected},
            {"context": new_context},
        ]
    return out


def invoke_graph_chain_with_cypher_retry(
    chain,
    inputs,
    config=None,
    *,
    _base_invoke: Callable[..., dict] | None = None,
) -> dict:
    """
    Existing Neo4j-error Cypher retry, then schema-aware sensor-identity retry,
    then schema-aware human-identity retry (Rider/Veterinarian/Caretaker).

    ``llm_service.invoke_graph_chain_with_cypher_retry`` is left unchanged; this
    wrapper is the call-site entry used by the API and fusion adapters.
    """
    if _base_invoke is None:
        from backend.graph_rag.llm_service import (
            invoke_graph_chain_with_cypher_retry as _base_invoke,
        )

    result = _base_invoke(chain, inputs, config=config)
    result = apply_sensor_identity_retry_if_needed(
        chain, inputs, result, config=config
    )
    from backend.graph_rag.cypher_human_identity import (
        apply_human_identity_retry_if_needed,
    )

    return apply_human_identity_retry_if_needed(
        chain, inputs, result, config=config
    )
