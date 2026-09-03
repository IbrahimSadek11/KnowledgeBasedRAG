"""
Enrich Neo4j Graph RAG result rows before they become QA context.

When a cell value equals an InertialSensors internal ``id``, attach the
user-facing ``hasSensorID`` alongside it. When generated Cypher filters a
Rider / Veterinarian / Caretaker on id or hasName, attach the resolved
human identity so the frozen QA prompt can see the queried person.

Does not alter Cypher generation or frozen QA prompts — only the execution
result representation.
"""
from __future__ import annotations

from typing import Any, Callable

from backend.graph_rag.cypher_human_identity import (
    HUMAN_LABELS,
    extract_human_identity_lookups,
)
from backend.graph_rag.cypher_sensor_identity import is_read_only_cypher

_LOOKUP_CYPHER = (
    "MATCH (s:InertialSensors) "
    "WHERE s.id IN $ids "
    "RETURN s.id AS id, s.hasSensorID AS hasSensorID"
)

_QUERIED_HUMAN_ID = "queried_human_id"
_QUERIED_HUMAN_NAME = "queried_human_hasName"
_QUERIED_HUMAN_LABEL = "queried_human_label"


def _companion_has_sensor_id_key(key: str) -> str | None:
    """Derive enrichment key for a column that held an internal sensor id."""
    if key in {"hasSensorID", "sensor_hasSensorID"} or key.endswith(
        "_hasSensorID"
    ):
        return None
    if key == "id":
        return "hasSensorID"
    if key.endswith("_id"):
        return f"{key[:-3]}_hasSensorID"
    return f"{key}_hasSensorID"


def _collect_string_candidates(value: Any, out: set[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.add(text)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_string_candidates(item, out)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_string_candidates(item, out)


def _lookup_id_to_has_sensor_id(
    run_query: Callable[..., list],
    candidate_ids: list[str],
) -> dict[str, str]:
    if not candidate_ids:
        return {}
    if not is_read_only_cypher(_LOOKUP_CYPHER):
        raise RuntimeError("Refusing non-read-only sensor identity lookup")
    rows = run_query(_LOOKUP_CYPHER, {"ids": candidate_ids})
    mapping: dict[str, str] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sid = row.get("id")
        has_id = row.get("hasSensorID")
        if sid is None or has_id is None:
            continue
        sid_s = str(sid).strip()
        has_s = str(has_id).strip()
        if sid_s and has_s and sid_s != has_s:
            mapping[sid_s] = has_s
    return mapping


def _enrich_value(
    value: Any,
    id_to_has: dict[str, str],
) -> tuple[Any, Any | None]:
    """
    Return (value_unchanged, optional_hasSensorID_enrichment).

    For lists, enrichment is a parallel list (None where not an internal id).
    """
    if isinstance(value, str):
        text = value.strip()
        return value, id_to_has.get(text)
    if isinstance(value, list):
        parallel: list[Any] = []
        any_hit = False
        for item in value:
            if isinstance(item, str):
                hit = id_to_has.get(item.strip())
                parallel.append(hit)
                if hit is not None:
                    any_hit = True
            else:
                parallel.append(None)
        return value, (parallel if any_hit else None)
    return value, None


def _lookup_humans_by_identity(
    run_query: Callable[..., list],
    label: str,
    op_norm: str,
    literal: str,
) -> list[dict[str, str]]:
    if label not in HUMAN_LABELS:
        raise ValueError(f"Unsupported human label: {label!r}")
    allowed_ops = {"=", "CONTAINS", "STARTS WITH", "ENDS WITH"}
    if op_norm not in allowed_ops:
        raise ValueError(f"Unsupported identity operator: {op_norm!r}")
    cypher = (
        f"MATCH (h:{label}) "
        f"WHERE h.id {op_norm} $literal OR h.hasName {op_norm} $literal "
        "RETURN DISTINCT h.id AS id, h.hasName AS hasName"
    )
    if not is_read_only_cypher(cypher):
        raise RuntimeError("Refusing non-read-only human identity lookup")
    rows = run_query(cypher, {"literal": literal})
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        hid = str(row.get("id") or "").strip()
        name = str(row.get("hasName") or "").strip()
        if not hid and not name:
            continue
        key = (hid, name)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": hid, "hasName": name})
    return out


def resolve_unique_queried_human(
    cypher: str | None,
    run_query: Callable[..., list],
) -> dict[str, str] | None:
    """
    Resolve at most one Rider/Veterinarian/Caretaker from generated Cypher.

    Returns None when zero or multiple humans match (unsafe / ambiguous).
    """
    if not cypher:
        return None
    lookups = extract_human_identity_lookups(cypher)
    if not lookups:
        return None

    unique: dict[str, dict[str, str]] = {}
    for pred in lookups:
        matches = _lookup_humans_by_identity(
            run_query, pred.label, pred.op_norm, pred.literal
        )
        if len(matches) > 1:
            return None
        if len(matches) != 1:
            continue
        hit = matches[0]
        key = hit["id"] or hit["hasName"]
        unique[key] = {
            "id": hit["id"],
            "hasName": hit["hasName"],
            "label": pred.label,
        }

    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _append_queried_human(
    rows: list,
    human: dict[str, str],
) -> list:
    hid = (human.get("id") or "").strip()
    name = (human.get("hasName") or "").strip()
    label = (human.get("label") or "").strip()
    extras = {
        k: v
        for k, v in {
            _QUERIED_HUMAN_ID: hid or None,
            _QUERIED_HUMAN_NAME: name or None,
            _QUERIED_HUMAN_LABEL: label or None,
        }.items()
        if v
    }
    if not extras:
        return rows

    enriched: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            enriched.append(row)
            continue
        new_row = dict(row)
        for key, value in extras.items():
            if key not in new_row:
                new_row[key] = value
        enriched.append(new_row)
    return enriched


def enrich_neo4j_rows_for_qa(
    rows: list | None,
    run_query: Callable[..., list],
    cypher: str | None = None,
) -> list | None:
    """
    Copy rows and add sensor ``*_hasSensorID`` plus queried-human identity
    fields when they can be resolved safely from generated Cypher.
    """
    if not rows or not isinstance(rows, list):
        return rows

    candidates: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            _collect_string_candidates(row, candidates)

    id_to_has = _lookup_id_to_has_sensor_id(run_query, sorted(candidates))
    if not id_to_has:
        enriched_rows: list[Any] = list(rows)
    else:
        enriched_rows = []
        for row in rows:
            if not isinstance(row, dict):
                enriched_rows.append(row)
                continue
            new_row = dict(row)
            extras: dict[str, Any] = {}
            for key, value in row.items():
                companion = _companion_has_sensor_id_key(str(key))
                if companion is None or companion in new_row:
                    continue
                _, enrichment = _enrich_value(value, id_to_has)
                if enrichment is not None:
                    extras[companion] = enrichment
            new_row.update(extras)
            enriched_rows.append(new_row)

    human = resolve_unique_queried_human(cypher, run_query)
    if human:
        enriched_rows = _append_queried_human(enriched_rows, human)
    return enriched_rows


def invoke_graph_chain_with_qa_context_enrichment(
    chain,
    inputs: dict,
    config=None,
    *,
    _invoke: Callable[..., dict] | None = None,
) -> dict:
    """
    Patch ``graph.query`` so Neo4j rows are identity-enriched before QA sees them.
    """
    if _invoke is None:
        from backend.graph_rag.cypher_sensor_identity import (
            invoke_graph_chain_with_cypher_retry as _invoke,
        )

    graph = chain.graph
    original_query = graph.query
    busy = {"on": False}

    def _enriched_query(query: str, params: dict | None = None, *args, **kwargs):
        params = dict(params or {})
        rows = original_query(query, params, *args, **kwargs)
        if busy["on"]:
            return rows
        if not isinstance(rows, list) or not rows:
            return rows
        busy["on"] = True
        try:
            return enrich_neo4j_rows_for_qa(rows, original_query, cypher=query)
        finally:
            busy["on"] = False

    graph.query = _enriched_query  # type: ignore[method-assign]
    try:
        return _invoke(chain, inputs, config=config)
    finally:
        graph.query = original_query  # type: ignore[method-assign]
