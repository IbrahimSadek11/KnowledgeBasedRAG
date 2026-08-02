"""Smoke-test pairwise agreement judge on synthetic + live answer pairs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fusion.adapters import run_graph_live, run_tabular_v2_live
from backend.fusion.agreement_judge import judge_pairwise_agreement


def _truncate_reason(reason: str, n: int = 15) -> str:
    words = (reason or "").split()
    if len(words) <= n:
        return reason or ""
    return " ".join(words[:n]) + "…"


def _run_pair(
    pair_id: str,
    question: str,
    answer_a: str,
    answer_b: str,
    pipeline_a: str,
    pipeline_b: str,
    expected: bool,
) -> dict:
    print("=" * 72)
    print(f"{pair_id}")
    print("=" * 72)
    print(f"pipeline_a={pipeline_a}")
    print(f"pipeline_b={pipeline_b}")
    print(f"answer_a={answer_a}")
    print(f"answer_b={answer_b}")
    print(f"expected_agreement={expected}")

    result = judge_pairwise_agreement(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
    )
    print("judge_result:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    actual = result.get("agreement")
    met = actual is expected and not result.get("judge_error")
    if pair_id.startswith("PAIR 2") and actual is True:
        print(
            "FINDING: PAIR 2 returned agreement=true — subset/superset "
            "edge case NOT treated as disagreement. Do not silently pass."
        )
        met = False

    print(f"met_expectation={met}")
    print()
    return {
        "pair": pair_id,
        "expected": expected,
        "actual_agreement": actual,
        "reason_truncated": _truncate_reason(str(result.get("reason") or "")),
        "met_expectation": met,
        "judge_error": result.get("judge_error"),
        "full_reason": result.get("reason"),
    }


def main() -> None:
    q_generic = "Qui a participé à la compétition ?"
    q_unavailable = "Quelle est la couleur préférée du cheval Dakota ?"
    q_dakota = "Quelle est la race de Dakota ?"

    rows = []

    rows.append(
        _run_pair(
            "PAIR 1 — True agreement (same fact, different wording)",
            q_generic,
            "Dakota et Orion ont participé à la compétition.",
            "Les participants étaient Orion et Dakota.",
            "graph",
            "tabular_v2",
            True,
        )
    )

    rows.append(
        _run_pair(
            "PAIR 2 — Subset/superset (should NOT be full agreement)",
            q_generic,
            "Dakota a participé.",
            "Dakota, Orion et Vega ont participé.",
            "graph",
            "tabular_v2",
            False,
        )
    )

    rows.append(
        _run_pair(
            "PAIR 3 — Both honestly report unavailable info",
            q_unavailable,
            "L'information n'est pas disponible dans la base de données.",
            "Aucune donnée ne permet de répondre à cette question.",
            "graph",
            "textual",
            True,
        )
    )

    print("=" * 72)
    print("PAIR 4 setup — live graph + tabular_v2 on Q2")
    print("=" * 72)
    graph = run_graph_live(q_dakota)
    tabular = run_tabular_v2_live(q_dakota)
    print(f"graph answer: {graph.get('answer')}")
    print(f"tabular answer: {tabular.get('answer')}")
    print()
    rows.append(
        _run_pair(
            "PAIR 4 — Real pipeline answers (Dakota race)",
            q_dakota,
            graph.get("answer") or "",
            tabular.get("answer") or "",
            "graph",
            "tabular_v2",
            True,
        )
    )

    rows.append(
        _run_pair(
            "PAIR 5 — Direct contradiction",
            q_dakota,
            "Dakota est de race Selle Français.",
            "Dakota est de race Arabe.",
            "graph",
            "tabular_v2",
            False,
        )
    )

    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(
        f"{'pair':<12} | {'expected':<8} | {'actual':<8} | "
        f"{'reason (~15 words)':<60} | met"
    )
    print("-" * 110)
    for row in rows:
        pair_short = row["pair"].split("—")[0].strip()
        actual = row["actual_agreement"]
        actual_s = "None" if actual is None else str(actual)
        print(
            f"{pair_short:<12} | {str(row['expected']):<8} | {actual_s:<8} | "
            f"{row['reason_truncated']:<60} | "
            f"{'yes' if row['met_expectation'] else 'no'}"
        )
        if pair_short.startswith("PAIR 2") and actual is True:
            print("  *** FINDING: PAIR 2 incorrectly agreed on subset/superset ***")

    print("\nFull summary JSON:")
    print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
