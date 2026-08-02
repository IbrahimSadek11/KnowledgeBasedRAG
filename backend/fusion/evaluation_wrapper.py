"""Post-selection evaluation against ground truth.

Ground truth is allowed ONLY in this module (evaluation phase).
Never import or call this from selector / evidence_judge / agreement_judge.
"""
from __future__ import annotations

from typing import Any

from backend.evaluation_service import (
    calculate_semantic_similarity,
    init_evaluator,
    llm_judge_answer,
)

_evaluator_cache: tuple[Any, Any] | None = None

_PIPELINES = ("graph", "tabular_v2", "textual")

_NO_ANSWER = {
    "semantic_similarity": None,
    "llm_judge_scores": None,
    "combined_score": None,
    "error": "no answer to evaluate",
}


def score_answer_against_ground_truth(
    question: str,
    answer: str,
    ground_truth: str,
    evaluator,
    judge_llm,
) -> dict:
    """Score one answer vs GT using evaluation_service helpers.

    ``evaluator`` is the embeddings object returned by ``init_evaluator()``
    (second element of the ``(judge_llm, embeddings)`` tuple).
    """
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return dict(_NO_ANSWER)

    try:
        semantic_similarity = calculate_semantic_similarity(
            answer, ground_truth, evaluator
        )
        llm_judge_scores = llm_judge_answer(
            question, answer, ground_truth, judge_llm
        )
        overall = float(llm_judge_scores["overall"])
        combined_score = (float(semantic_similarity) + overall) / 2
        return {
            "semantic_similarity": float(semantic_similarity),
            "llm_judge_scores": llm_judge_scores,
            "combined_score": float(combined_score),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "semantic_similarity": None,
            "llm_judge_scores": None,
            "combined_score": None,
            "error": str(exc),
        }


def get_evaluator_and_judge():
    """Lazily init and cache (judge_llm, embeddings) from init_evaluator()."""
    global _evaluator_cache
    if _evaluator_cache is None:
        # init_evaluator returns (judge_llm, embeddings) — same as run_evaluation.py
        judge_llm, embeddings = init_evaluator()
        _evaluator_cache = (judge_llm, embeddings)
    return _evaluator_cache


def evaluate_selected_answer(
    question: str,
    selected_pipeline: str | None,
    selected_answer: str | None,
    ground_truth: str,
) -> dict:
    """Evaluate the fusion-selected answer against GT.

    This result becomes the fusion.combined_score / fusion.semantic_similarity /
    fusion.llm_judge_scores fields in the final per-question output record.
    """
    if selected_answer is None or (
        isinstance(selected_answer, str) and not selected_answer.strip()
    ):
        return dict(_NO_ANSWER)

    judge_llm, embeddings = get_evaluator_and_judge()
    result = score_answer_against_ground_truth(
        question, selected_answer, ground_truth, embeddings, judge_llm
    )
    # Preserve pipeline label for downstream reporting (additive only).
    result = dict(result)
    result["selected_pipeline"] = selected_pipeline
    return result


def evaluate_all_pipelines_for_research(
    question: str,
    live_results: dict,
    ground_truth: str,
) -> dict:
    """Per-pipeline GT scores for research comparison only.

    IMPORTANT: This function's output goes ONLY under an "evaluation_only"
    key in the final per-question record. It must never be passed to
    select_fusion_answer, build_agreement_grouping, or score_evidence.
    It exists purely for research comparison between fusion's choice and
    what each individual pipeline would have scored.
    """
    judge_llm, embeddings = get_evaluator_and_judge()
    out: dict[str, dict] = {}
    for pipeline in _PIPELINES:
        live = live_results.get(pipeline) or {}
        answer = live.get("answer")
        if live.get("success") is True and isinstance(answer, str) and answer.strip():
            out[pipeline] = score_answer_against_ground_truth(
                question, answer, ground_truth, embeddings, judge_llm
            )
        else:
            out[pipeline] = {
                "semantic_similarity": None,
                "llm_judge_scores": None,
                "combined_score": None,
                "error": "pipeline did not succeed",
            }
    return out
