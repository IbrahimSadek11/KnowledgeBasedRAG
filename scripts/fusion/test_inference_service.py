"""Live smoke-test for run_fusion_inference (GT-free)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fusion.orchestrator import run_fusion_inference

QUESTION = "Quelle est la race de Dakota ?"
_FORBIDDEN = {"ground_truth", "evaluation_only", "combined_score"}


def _contains_forbidden(obj, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{path}.{k}" if path else str(k)
            if k in _FORBIDDEN:
                hits.append(key_path)
            hits.extend(_contains_forbidden(v, key_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_contains_forbidden(v, f"{path}[{i}]"))
    return hits


def main() -> int:
    print(f"QUESTION: {QUESTION}")
    t0 = time.perf_counter()
    result = run_fusion_inference(QUESTION)
    wall = time.perf_counter() - t0
    print(f"WALL_CLOCK_SECONDS: {wall:.3f}")

    graph = result["graph"]
    tabular = result["tabular_v2"]
    textual = result["textual"]
    evidence = result["evidence_scores"]
    pairwise = result["pairwise_agreements"]
    fusion = result["fusion"]

    print(
        "GRAPH:",
        f"success={graph.get('success')}",
        f"generated_query_nonempty={bool(graph.get('generated_query'))}",
    )
    print(
        "TABULAR_V2:",
        f"success={tabular.get('success')}",
        f"pipeline={tabular.get('pipeline')}",
        f"generated_query_nonempty={bool(tabular.get('generated_query'))}",
    )
    print(
        "TEXTUAL:",
        f"success={textual.get('success')}",
        f"retrieved_documents_nonempty={bool(textual.get('retrieved_documents'))}",
        f"retrieved_documents_count={len(textual.get('retrieved_documents') or [])}",
    )

    assert graph.get("generated_query"), "Graph Cypher missing/empty"
    assert tabular.get("generated_query"), "Tabular SQL missing/empty"
    assert tabular.get("pipeline") == "tabular_v2", "Tabular result not from V2"
    assert textual.get("retrieved_documents") or textual.get(
        "retrieved_passages"
    ), "Textual retrieval empty"
    assert all(k in evidence for k in ("graph", "tabular_v2", "textual")), (
        "Evidence scores incomplete"
    )
    assert isinstance(pairwise, dict) and len(pairwise) == 3, "Pairwise incomplete"
    assert fusion.get("selected_pipeline"), "No pipeline selected"
    assert fusion.get("selected_answer"), "Empty selected answer"

    hits = _contains_forbidden(result)
    assert not hits, f"Forbidden keys present: {hits}"

    serialized = json.dumps(result, ensure_ascii=False)
    assert serialized, "json.dumps failed / empty"

    print("PASS: evidence_scores keys =", sorted(evidence.keys()))
    print("PASS: pairwise_agreements keys =", sorted(pairwise.keys()))
    print("PASS: selected_pipeline =", fusion.get("selected_pipeline"))
    print("PASS: selected_answer =", (fusion.get("selected_answer") or "")[:200])
    print("PASS: no forbidden GT/eval keys")
    print("PASS: json.dumps OK, chars =", len(serialized))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
