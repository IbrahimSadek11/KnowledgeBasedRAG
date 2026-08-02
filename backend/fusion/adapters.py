"""Live pipeline adapters for Fusion RAG (graph + tabular v2 + textual).

Does not modify existing graph_rag/, tabular_rag/, or textual_rag/ pipeline files
beyond what those modules already expose.
"""
from __future__ import annotations

import time
from typing import Any

from backend.tabular_rag.version2.tabular_chain import (
    DB_PATH as TABULAR_V2_DB_PATH,
)
from backend.tabular_rag.version2.tabular_chain import (
    answer_question as tabular_v2_answer_question,
)
from backend.textual_rag.textual_rag_service import (
    answer_question as textual_answer_question,
)

_KNOWN_TABULAR_FAILURE_ANSWER = "Could not generate a valid query after retries."

_graph_chain = None


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert Neo4j/SQLite-specific objects to JSON-safe types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    try:
        return _to_jsonable(dict(obj))
    except Exception:
        pass
    try:
        return _to_jsonable(list(obj))
    except Exception:
        pass
    return str(obj)


def get_graph_chain():
    """Lazily initialize and cache the GraphCypherQAChain once."""
    global _graph_chain
    if _graph_chain is None:
        from backend.graph_rag.llm_service import init_graph_chain

        chain, _graph = init_graph_chain()
        _graph_chain = chain
    return _graph_chain


def _extract_graph_fields(result: dict) -> tuple[str | None, str | None, Any, Any]:
    """Pull answer, Cypher, raw rows, and intermediate_steps from chain output."""
    answer = result.get("result")
    if answer is None:
        answer = result.get("answer")
    if isinstance(answer, str):
        answer = answer.strip() or None
    elif answer is not None:
        answer = str(answer)

    steps = result.get("intermediate_steps") or []
    cypher = None
    raw_rows = None

    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("query") and cypher is None:
            cypher = step["query"]
        if "context" in step:
            raw_rows = step.get("context")

    if cypher is None and steps and isinstance(steps[0], dict):
        cypher = steps[0].get("query")

    return answer, cypher, raw_rows, steps


def run_graph_live(question: str) -> dict:
    """Invoke the cached Graph RAG chain and normalize to the shared result shape."""
    metadata: dict[str, Any] = {}
    try:
        chain = get_graph_chain()
        t0 = time.perf_counter()
        result = chain.invoke({"query": question})
        elapsed = time.perf_counter() - t0

        answer, cypher, raw_rows, steps = _extract_graph_fields(result)
        success = bool(answer)

        return {
            "pipeline": "graph",
            "question": question,
            "answer": answer,
            "generated_query": cypher,
            "raw_results": _to_jsonable(raw_rows),
            "execution_time_seconds": elapsed,
            "attempts": None,
            "success": success,
            "error": None if success else "empty or missing final answer",
            "retrieved_documents": None,
            "retrieved_passages": None,
            "metadata": {
                "raw_intermediate_steps": _to_jsonable(steps),
                **metadata,
            },
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to caller as error field
        return {
            "pipeline": "graph",
            "question": question,
            "answer": None,
            "generated_query": None,
            "raw_results": None,
            "execution_time_seconds": 0.0,
            "attempts": None,
            "success": False,
            "error": str(exc),
            "retrieved_documents": None,
            "retrieved_passages": None,
            "metadata": metadata,
        }


def run_tabular_v2_live(question: str) -> dict:
    """Invoke tabular RAG version2 and normalize to the shared result shape."""
    metadata: dict[str, Any] = {
        "pipeline_version": "version2",
        "db_path": TABULAR_V2_DB_PATH,
    }
    try:
        t0 = time.perf_counter()
        result = tabular_v2_answer_question(question)
        elapsed = time.perf_counter() - t0

        sql = result.get("sql")
        answer = result.get("answer")
        rows = result.get("rows")
        raw_attempts = result.get("attempts")
        metadata["attempt_log"] = raw_attempts
        if isinstance(raw_attempts, list):
            attempts = len(raw_attempts)
        elif isinstance(raw_attempts, int):
            attempts = raw_attempts
        else:
            attempts = None

        success = sql is not None
        error = None
        if not success:
            error = "SQL generation/execution exhausted retries"

        if success and isinstance(answer, str) and answer.strip() == _KNOWN_TABULAR_FAILURE_ANSWER:
            metadata["answer_matches_known_failure_string"] = True

        return {
            "pipeline": "tabular_v2",
            "question": question,
            "answer": answer,
            "generated_query": sql,
            "raw_results": _to_jsonable(rows),
            "execution_time_seconds": elapsed,
            "attempts": attempts,
            "success": success,
            "error": error,
            "retrieved_documents": None,
            "retrieved_passages": None,
            "metadata": metadata,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to caller as error field
        return {
            "pipeline": "tabular_v2",
            "question": question,
            "answer": None,
            "generated_query": None,
            "raw_results": None,
            "execution_time_seconds": 0.0,
            "attempts": None,
            "success": False,
            "error": str(exc),
            "retrieved_documents": None,
            "retrieved_passages": None,
            "metadata": metadata,
        }


def run_textual_live(question: str) -> dict:
    """Invoke textual RAG and normalize to the shared result shape."""
    metadata: dict[str, Any] = {}
    try:
        t0 = time.perf_counter()
        result = textual_answer_question(question)
        elapsed = time.perf_counter() - t0

        answer = result.get("answer")
        retrieved_documents = result.get("retrieved_docs")
        retrieved_passages = result.get("retrieved_passages")
        retrieved_ids = result.get("retrieved_ids") or []
        retrieved_metadata = result.get("retrieved_metadata") or []
        filenames = retrieved_documents or []
        passages = retrieved_passages or []

        raw_results = _to_jsonable(
            [
                {
                    "id": doc_id,
                    "filename": filename,
                    "entity_type": (meta or {}).get("entity_type"),
                    "passage_text": passage,
                }
                for doc_id, filename, meta, passage in zip(
                    retrieved_ids, filenames, retrieved_metadata, passages
                )
            ]
        )

        success = isinstance(answer, str) and bool(answer.strip())

        return {
            "pipeline": "textual",
            "question": question,
            "answer": answer,
            "generated_query": None,
            "raw_results": raw_results,
            "execution_time_seconds": elapsed,
            "attempts": None,
            "success": success,
            "error": None if success else "empty or missing final answer",
            "retrieved_documents": retrieved_documents,
            "retrieved_passages": retrieved_passages,
            "metadata": metadata,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to caller as error field
        return {
            "pipeline": "textual",
            "question": question,
            "answer": None,
            "generated_query": None,
            "raw_results": None,
            "execution_time_seconds": 0.0,
            "attempts": None,
            "success": False,
            "error": str(exc),
            "retrieved_documents": None,
            "retrieved_passages": None,
            "metadata": metadata,
        }
