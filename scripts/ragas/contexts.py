"""Convert pipeline retrieval artifacts into RAGAS retrieved_contexts.

RAGAS retrieved_contexts must be actual retrieved content:
- Graph / tabular: database result rows
- Textual: retrieved passage text

Never use generated Cypher or SQL as context.
"""
from __future__ import annotations

import json
from typing import Any

_PIPELINE_ALIASES = {
    "graph": "graph",
    "tabular": "tabular_v2",
    "tabular_v2": "tabular_v2",
    "textual": "textual",
}


def to_context_strings(value: Any) -> list[str]:
    """Serialize retrieved rows/passages to List[str] without changing meaning."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return [_dump_row(value)]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(to_context_strings(item))
        return out
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    return [_dump_row(value)]


def _dump_row(row: Any) -> str:
    if isinstance(row, str):
        return row
    try:
        return json.dumps(row, ensure_ascii=False, default=str)
    except TypeError:
        return str(row)


def graph_retrieved_contexts(graph_result: dict[str, Any] | None) -> list[str]:
    """Neo4j result rows from Fusion's Graph adapter. Never Cypher."""
    if not graph_result:
        return []
    rows = graph_result.get("raw_results")
    if rows is None:
        metadata = graph_result.get("metadata") or {}
        steps = metadata.get("raw_intermediate_steps") or []
        for step in steps:
            if isinstance(step, dict) and "context" in step:
                rows = step.get("context")
                break
    return to_context_strings(rows)


def textual_retrieved_contexts(textual_result: dict[str, Any] | None) -> list[str]:
    """Passage text only — not filenames, not empty wrappers."""
    if not textual_result:
        return []
    passages = textual_result.get("retrieved_passages")
    if isinstance(passages, list):
        texts = [p.strip() for p in passages if isinstance(p, str) and p.strip()]
        if texts:
            return texts
    raw = textual_result.get("raw_results")
    if isinstance(raw, list):
        extracted: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                passage = item.get("passage_text")
                if isinstance(passage, str) and passage.strip():
                    extracted.append(passage.strip())
                    continue
            extracted.extend(to_context_strings(item))
        return extracted
    return to_context_strings(raw)


def fusion_retrieved_contexts(
    inference: dict[str, Any],
) -> tuple[list[str], str | None]:
    """Contexts from the Fusion-selected pipeline only. Do not concatenate."""
    fusion = inference.get("fusion") or {}
    selected = fusion.get("selected_pipeline")
    if not selected:
        return [], None
    key = _PIPELINE_ALIASES.get(str(selected), str(selected))
    pipeline = inference.get(key) or {}
    if key == "textual":
        return textual_retrieved_contexts(pipeline), key
    if key == "graph":
        return graph_retrieved_contexts(pipeline), key
    return to_context_strings(pipeline.get("raw_results")), key
