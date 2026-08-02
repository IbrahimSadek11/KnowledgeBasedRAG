"""Smoke-test post-selection GT evaluation wrapper on Q2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fusion.adapters import (
    run_graph_live,
    run_tabular_v2_live,
    run_textual_live,
)
from backend.fusion.evaluation_wrapper import (
    evaluate_all_pipelines_for_research,
    evaluate_selected_answer,
)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "test_dataset.json"
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)

    item = data["test_questions"][1]  # Q2
    question = item["question"]
    ground_truth = item["ground_truth"]
    print(f"{item['question_id']}: {question}")
    print(f"ground_truth: {ground_truth}")
    print()

    graph = run_graph_live(question)
    tabular = run_tabular_v2_live(question)
    textual = run_textual_live(question)
    live_results = {
        "graph": graph,
        "tabular_v2": tabular,
        "textual": textual,
    }

    print("--- live answers ---")
    for name, res in live_results.items():
        print(f"{name}: success={res.get('success')} answer={res.get('answer')!r}")
    print()

    # Suppose fusion selected graph (hardcoded for this smoke test).
    selected = evaluate_selected_answer(
        question, "graph", graph.get("answer"), ground_truth
    )
    print("--- evaluate_selected_answer (fusion selected=graph) ---")
    print(json.dumps(selected, indent=2, ensure_ascii=False, default=str))
    print()

    research = evaluate_all_pipelines_for_research(
        question, live_results, ground_truth
    )
    print("--- evaluate_all_pipelines_for_research (evaluation_only) ---")
    print(json.dumps(research, indent=2, ensure_ascii=False, default=str))
    print()

    print("--- combined_score summary ---")
    print(f"selected(graph): {selected.get('combined_score')}")
    for name in ("graph", "tabular_v2", "textual"):
        print(f"{name}: {research[name].get('combined_score')}")


if __name__ == "__main__":
    main()
