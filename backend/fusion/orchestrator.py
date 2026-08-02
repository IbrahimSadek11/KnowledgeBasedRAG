"""Per-question Fusion RAG orchestrator — live → evidence → agreement → select → eval."""
from __future__ import annotations

import time
from typing import Any

from backend.fusion.adapters import (
    _to_jsonable,
    run_graph_live,
    run_tabular_v2_live,
    run_textual_live,
)
from backend.fusion.agreement_judge import judge_pairwise_agreement
from backend.fusion.evaluation_wrapper import (
    evaluate_all_pipelines_for_research,
)
from backend.fusion.evidence_judge import score_evidence
from backend.fusion.selector import select_fusion_answer

_PAIR_KEYS = (
    ("graph_tabular_v2", "graph", "tabular_v2"),
    ("graph_textual", "graph", "textual"),
    ("tabular_v2_textual", "tabular_v2", "textual"),
)


def run_fusion_inference(question: str) -> dict:
    """GT-free live fusion: three pipelines → evidence → agreement → selection.

    Safe for Streamlit / production inference. Does not import or call
    evaluation_wrapper, ground_truth scoring, combined_score, or evaluation_only.
    Not cached — every call executes all three pipelines live.
    """
    t_total = time.perf_counter()

    graph_result = run_graph_live(question)
    tabular_result = run_tabular_v2_live(question)
    textual_result = run_textual_live(question)

    live_results = {
        "graph": graph_result,
        "tabular_v2": tabular_result,
        "textual": textual_result,
    }

    evidence_scores = {
        "graph": score_evidence(graph_result),
        "tabular_v2": score_evidence(tabular_result),
        "textual": score_evidence(textual_result),
    }

    valid = {
        name for name, result in live_results.items() if result.get("success") is True
    }
    pairwise: dict[str, Any] = {}
    for pair_key, a, b in _PAIR_KEYS:
        if a in valid and b in valid:
            pairwise[pair_key] = judge_pairwise_agreement(
                question=question,
                answer_a=live_results[a].get("answer") or "",
                answer_b=live_results[b].get("answer") or "",
                pipeline_a=a,
                pipeline_b=b,
            )
        else:
            pairwise[pair_key] = None

    selection_result = select_fusion_answer(live_results, evidence_scores, pairwise)

    total_question_seconds = time.perf_counter() - t_total

    return _to_jsonable(
        {
            "graph": graph_result,
            "tabular_v2": tabular_result,
            "textual": textual_result,
            "evidence_scores": evidence_scores,
            "pairwise_agreements": pairwise,
            "fusion": selection_result,
            "timing": {
                "graph_seconds": graph_result.get("execution_time_seconds"),
                "tabular_seconds": tabular_result.get("execution_time_seconds"),
                "textual_seconds": textual_result.get("execution_time_seconds"),
                "total_question_seconds": total_question_seconds,
            },
        }
    )


def run_fusion_for_question(
    question_id,
    question: str,
    ground_truth: str,
    category: str,
    difficulty: str,
) -> dict:
    """Run the full fusion pipeline for one benchmark question.

    Live pipelines + selection come from run_fusion_inference (GT-free).
    Ground truth is used only afterward for evaluation_wrapper scoring.
    ``category`` is recorded in the output only — it does not change control flow.
    """
    t_total = time.perf_counter()

    inference = run_fusion_inference(question)

    graph_result = inference["graph"]
    tabular_result = inference["tabular_v2"]
    textual_result = inference["textual"]
    evidence_scores = inference["evidence_scores"]
    pairwise = inference["pairwise_agreements"]
    selection_result = inference["fusion"]

    live_results = {
        "graph": graph_result,
        "tabular_v2": tabular_result,
        "textual": textual_result,
    }

    # Evaluation AFTER selection (GT allowed only here).
    # Evaluate each live pipeline once via research_eval, then REUSE that
    # result for fusion_eval. Fusion always selects one of the three live
    # pipelines' own answers verbatim, so re-evaluating the selected answer
    # independently would be wasteful and could introduce spurious
    # inconsistency due to LLM judge non-determinism.
    research_eval = evaluate_all_pipelines_for_research(
        question, live_results, ground_truth
    )
    selected_pipeline = selection_result.get("selected_pipeline")
    if selected_pipeline is not None and selected_pipeline in research_eval:
        fusion_eval = research_eval[selected_pipeline]
    else:
        fusion_eval = {
            "semantic_similarity": None,
            "llm_judge_scores": None,
            "combined_score": None,
            "error": "no answer to evaluate",
        }

    total_question_seconds = time.perf_counter() - t_total

    record = {
        "question_id": question_id,
        "question": question,
        "ground_truth": ground_truth,
        "category": category,
        "difficulty": difficulty,
        "graph_rag": {**graph_result, "evidence_judge": evidence_scores["graph"]},
        "tabular_rag": {
            **tabular_result,
            "evidence_judge": evidence_scores["tabular_v2"],
        },
        "textual_rag": {
            **textual_result,
            "evidence_judge": evidence_scores["textual"],
        },
        "pairwise_agreement": pairwise,
        "fusion": {**selection_result, **fusion_eval},
        "evaluation_only": research_eval,
        "timing": {
            "graph_seconds": graph_result.get("execution_time_seconds"),
            "tabular_seconds": tabular_result.get("execution_time_seconds"),
            "textual_seconds": textual_result.get("execution_time_seconds"),
            "total_question_seconds": total_question_seconds,
        },
    }
    return _to_jsonable(record)
