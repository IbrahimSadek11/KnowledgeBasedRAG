"""Smoke-test GT-free evidence judge on Q2 across three live pipelines."""
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
from backend.fusion.evidence_judge import score_evidence


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "test_dataset.json"
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # question_id 2 — "Quelle est la race de Dakota ?"
    item = data["test_questions"][1]
    question = item["question"]
    print(f"{item['question_id']}: {question}")
    print("(ground_truth intentionally not loaded or printed)\n")

    runners = [
        ("graph", run_graph_live),
        ("tabular_v2", run_tabular_v2_live),
        ("textual", run_textual_live),
    ]

    for label, runner in runners:
        print("=" * 72)
        print(f"PIPELINE: {label}")
        print("=" * 72)
        pipeline_result = runner(question)
        print(
            f"adapter success={pipeline_result.get('success')} "
            f"attempts={pipeline_result.get('attempts')}"
        )
        evidence = score_evidence(pipeline_result)
        print(json.dumps(evidence, indent=2, ensure_ascii=False, default=str))
        print()


if __name__ == "__main__":
    main()
