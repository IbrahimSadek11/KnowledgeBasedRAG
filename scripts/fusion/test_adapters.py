"""Smoke-test live graph + tabular_v2 + textual fusion adapters on Q1–Q2."""
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


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "test_dataset.json"
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)

    questions = data["test_questions"][:2]

    for item in questions:
        qid = item["question_id"]
        question = item["question"]
        print("=" * 72)
        print(f"{qid}: {question}")
        print("=" * 72)

        print("\n--- PIPELINE: graph ---")
        graph_result = run_graph_live(question)
        print(json.dumps(graph_result, indent=2, ensure_ascii=False, default=str))

        print("\n--- PIPELINE: tabular_v2 ---")
        tabular_result = run_tabular_v2_live(question)
        print(json.dumps(tabular_result, indent=2, ensure_ascii=False, default=str))

        print("\n--- PIPELINE: textual ---")
        textual_result = run_textual_live(question)
        print(json.dumps(textual_result, indent=2, ensure_ascii=False, default=str))
        print()


if __name__ == "__main__":
    main()
