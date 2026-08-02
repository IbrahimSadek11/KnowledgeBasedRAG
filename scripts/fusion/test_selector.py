"""Pure unit tests for fusion selector — no LLM, no live pipelines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fusion.selector import select_fusion_answer


def _live(success: bool, answer: str | None = None, error: str | None = None) -> dict:
    return {
        "pipeline": "?",
        "question": "q",
        "answer": answer if success else None,
        "generated_query": "Q" if success else None,
        "raw_results": [{"x": 1}] if success else None,
        "retrieved_documents": None,
        "retrieved_passages": None,
        "success": success,
        "error": error,
        "attempts": None,
    }


def _ev(
    score: float | None,
    groundedness: float = 0.5,
    completeness: float = 0.5,
    relevance: float = 0.5,
) -> dict:
    return {
        "pipeline": "?",
        "groundedness": None if score is None else groundedness,
        "completeness": None if score is None else completeness,
        "relevance": None if score is None else relevance,
        "execution_quality": 1.0,
        "evidence_score": score,
        "reasoning": {},
        "judge_error": score is None,
        "truncated_evidence": False,
    }


def _pair(agreement: bool | None, judge_error: bool = False) -> dict:
    return {
        "agreement": agreement,
        "reason": "mock",
        "judge_error": judge_error or agreement is None,
        "pipeline_a": "a",
        "pipeline_b": "b",
    }


def _run(test_id: int, scenario: str, expected: dict, live, evidence, pairwise) -> dict:
    print("=" * 72)
    print(f"TEST {test_id}: {scenario}")
    print("=" * 72)
    print(
        "inputs:",
        json.dumps(
            {
                "valid": [p for p, r in live.items() if r.get("success")],
                "evidence_scores": {
                    p: (evidence[p] or {}).get("evidence_score") for p in evidence
                },
                "pairwise": {
                    k: (None if v is None else v.get("agreement"))
                    for k, v in pairwise.items()
                },
            },
            ensure_ascii=False,
        ),
    )
    result = select_fusion_answer(live, evidence, pairwise)
    print("result:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    failures = []
    for key, exp_val in expected.items():
        actual = result.get(key)
        if key == "agreement_group" and exp_val is not None:
            if sorted(actual or []) != sorted(exp_val):
                failures.append(f"{key}: expected {exp_val}, got {actual}")
        elif actual != exp_val:
            failures.append(f"{key}: expected {exp_val!r}, got {actual!r}")

    passed = not failures
    print("PASS" if passed else "FAIL: " + "; ".join(failures))
    print()
    return {
        "test": test_id,
        "scenario": scenario,
        "expected": (
            f"{expected.get('decision_rule')} / {expected.get('selected_pipeline')}"
        ),
        "actual": f"{result.get('decision_rule')} / {result.get('selected_pipeline')}",
        "pass": passed,
        "failures": failures,
        "result": result,
    }


def main() -> None:
    rows = []

    # 1. All three agree; pick highest evidence among three
    rows.append(
        _run(
            1,
            "all 3 agree; pick highest evidence",
            {
                "decision_rule": "all_three_agree",
                "selected_pipeline": "graph",
                "agreement_group": ["graph", "tabular_v2", "textual"],
                "agreement_strength": "strongest",
            },
            {
                "graph": _live(True, "A"),
                "tabular_v2": _live(True, "B"),
                "textual": _live(True, "C"),
            },
            {
                "graph": _ev(0.9),
                "tabular_v2": _ev(0.8),
                "textual": _ev(0.7),
            },
            {
                "graph_tabular_v2": _pair(True),
                "graph_textual": _pair(True),
                "tabular_v2_textual": _pair(True),
            },
        )
    )

    # 2. Only graph↔tabular agree; textual has highest evidence but must be excluded
    rows.append(
        _run(
            2,
            "only graph_tabular agree; textual 0.95 excluded",
            {
                "decision_rule": "single_pair_agreement",
                "selected_pipeline": "graph",
                "agreement_group": ["graph", "tabular_v2"],
                "agreement_strength": "moderate",
                "agreement_independence": "shared_upstream_source",
            },
            {
                "graph": _live(True, "Ag"),
                "tabular_v2": _live(True, "At"),
                "textual": _live(True, "Ax"),
            },
            {
                "graph": _ev(0.88),
                "tabular_v2": _ev(0.80),
                "textual": _ev(0.95),
            },
            {
                "graph_tabular_v2": _pair(True),
                "graph_textual": _pair(False),
                "tabular_v2_textual": _pair(False),
            },
        )
    )

    # 3. tabular↔textual agree → strong
    rows.append(
        _run(
            3,
            "tabular_textual agree → strong",
            {
                "decision_rule": "single_pair_agreement",
                "agreement_strength": "strong",
                "agreement_independence": "more_independent_sources",
                "agreement_group": ["tabular_v2", "textual"],
                "selected_pipeline": "textual",
            },
            {
                "graph": _live(True, "Ag"),
                "tabular_v2": _live(True, "At"),
                "textual": _live(True, "Ax"),
            },
            {
                "graph": _ev(0.99),
                "tabular_v2": _ev(0.70),
                "textual": _ev(0.85),
            },
            {
                "graph_tabular_v2": _pair(False),
                "graph_textual": _pair(False),
                "tabular_v2_textual": _pair(True),
            },
        )
    )

    # 4. No agreement → highest evidence overall
    rows.append(
        _run(
            4,
            "no agreement; highest evidence overall",
            {
                "decision_rule": "no_agreement",
                "selected_pipeline": "textual",
                "agreement_group": [],
            },
            {
                "graph": _live(True, "Ag"),
                "tabular_v2": _live(True, "At"),
                "textual": _live(True, "Ax"),
            },
            {
                "graph": _ev(0.6),
                "tabular_v2": _ev(0.7),
                "textual": _ev(0.9),
            },
            {
                "graph_tabular_v2": _pair(False),
                "graph_textual": _pair(False),
                "tabular_v2_textual": _pair(False),
            },
        )
    )

    # 5. Transitivity conflict (2 true, 1 false)
    rows.append(
        _run(
            5,
            "transitivity conflict; highest overall",
            {
                "decision_rule": "transitivity_conflict",
                "transitivity_conflict": True,
                "agreement_group": [],
                "selected_pipeline": "textual",
            },
            {
                "graph": _live(True, "Ag"),
                "tabular_v2": _live(True, "At"),
                "textual": _live(True, "Ax"),
            },
            {
                "graph": _ev(0.7),
                "tabular_v2": _ev(0.75),
                "textual": _ev(0.92),
            },
            {
                "graph_tabular_v2": _pair(True),
                "graph_textual": _pair(True),
                "tabular_v2_textual": _pair(False),
            },
        )
    )

    # 6. Only graph succeeds
    rows.append(
        _run(
            6,
            "only graph valid",
            {
                "decision_rule": "only_one_valid",
                "selected_pipeline": "graph",
            },
            {
                "graph": _live(True, "only graph"),
                "tabular_v2": _live(False, error="sql fail"),
                "textual": _live(False, error="chroma fail"),
            },
            {
                "graph": _ev(0.8),
                "tabular_v2": _ev(None),
                "textual": _ev(None),
            },
            {
                "graph_tabular_v2": None,
                "graph_textual": None,
                "tabular_v2_textual": None,
            },
        )
    )

    # 7. Two valid, they agree
    rows.append(
        _run(
            7,
            "two valid (graph,tabular) agree",
            {
                "decision_rule": "single_pair_agreement_two_valid",
                "agreement_group": ["graph", "tabular_v2"],
                "selected_pipeline": "tabular_v2",
            },
            {
                "graph": _live(True, "Ag"),
                "tabular_v2": _live(True, "At"),
                "textual": _live(False, error="fail"),
            },
            {
                "graph": _ev(0.7),
                "tabular_v2": _ev(0.85),
                "textual": _ev(None),
            },
            {
                "graph_tabular_v2": _pair(True),
                "graph_textual": None,
                "tabular_v2_textual": None,
            },
        )
    )

    # 8. All fail
    rows.append(
        _run(
            8,
            "all pipelines failed",
            {
                "decision_rule": "all_pipelines_failed",
                "selected_pipeline": None,
                "selected_answer": None,
            },
            {
                "graph": _live(False, error="neo4j down"),
                "tabular_v2": _live(False, error="sql down"),
                "textual": _live(False, error="chroma down"),
            },
            {
                "graph": _ev(None),
                "tabular_v2": _ev(None),
                "textual": _ev(None),
            },
            {
                "graph_tabular_v2": None,
                "graph_textual": None,
                "tabular_v2_textual": None,
            },
        )
    )

    # 9. Tie-break by groundedness
    rows.append(
        _run(
            9,
            "tie-break by groundedness",
            {
                "decision_rule": "all_three_agree",
                "selected_pipeline": "tabular_v2",
                "tie_break_note": "tie broken by groundedness",
            },
            {
                "graph": _live(True, "Ag"),
                "tabular_v2": _live(True, "At"),
                "textual": _live(True, "Ax"),
            },
            {
                "graph": _ev(0.9000, groundedness=0.6, completeness=0.9, relevance=0.9),
                "tabular_v2": _ev(
                    0.90005, groundedness=0.95, completeness=0.5, relevance=0.5
                ),
                "textual": _ev(0.90002, groundedness=0.7, completeness=0.7, relevance=0.7),
            },
            {
                "graph_tabular_v2": _pair(True),
                "graph_textual": _pair(True),
                "tabular_v2_textual": _pair(True),
            },
        )
    )

    # 10. One evidence_score None excluded
    rows.append(
        _run(
            10,
            "exclude None evidence_score in group",
            {
                "decision_rule": "single_pair_agreement",
                "selected_pipeline": "graph",
                "agreement_group": ["graph", "tabular_v2"],
            },
            {
                "graph": _live(True, "Ag"),
                "tabular_v2": _live(True, "At"),
                "textual": _live(True, "Ax"),
            },
            {
                "graph": _ev(0.8),
                "tabular_v2": _ev(None),  # judge error — exclude
                "textual": _ev(0.99),
            },
            {
                "graph_tabular_v2": _pair(True),
                "graph_textual": _pair(False),
                "tabular_v2_textual": _pair(False),
            },
        )
    )

    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(f"{'#':<3} | {'scenario':<42} | {'expected':<40} | {'actual':<40} | result")
    print("-" * 140)
    for row in rows:
        print(
            f"{row['test']:<3} | {row['scenario']:<42} | "
            f"{row['expected']:<40} | {row['actual']:<40} | "
            f"{'PASS' if row['pass'] else 'FAIL'}"
        )
        if not row["pass"]:
            print(f"     FINDING: {row['failures']}")

    n_pass = sum(1 for r in rows if r["pass"])
    print(f"\n{n_pass}/{len(rows)} passed")


if __name__ == "__main__":
    main()
