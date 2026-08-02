"""Deterministic Fusion answer selector — no LLM, no ground truth."""
from __future__ import annotations

from typing import Any

# Stable priority used for residual ties and unavailable evidence scores.
_PIPELINE_ORDER = ["graph", "tabular_v2", "textual"]

_PAIR_PIPELINES: dict[str, tuple[str, str]] = {
    "graph_tabular_v2": ("graph", "tabular_v2"),
    "graph_textual": ("graph", "textual"),
    "tabular_v2_textual": ("tabular_v2", "textual"),
}


def _pair_strength_independence(pipelines: list[str]) -> tuple[str, str]:
    """Map an agreeing pair to strength / independence labels."""
    names = set(pipelines)
    if names == {"graph", "tabular_v2"}:
        return "moderate", "shared_upstream_source"
    if "textual" in names:
        return "strong", "more_independent_sources"
    return "moderate", "shared_upstream_source"


def _effective_agreement(pair_result: dict | None) -> bool:
    """Treat missing/error agreement as False for grouping."""
    if pair_result is None:
        return False
    agreement = pair_result.get("agreement")
    if agreement is True:
        return True
    return False


def build_agreement_grouping(pairwise: dict) -> dict:
    """
    pairwise: {
        "graph_tabular_v2": agreement_judge_result_dict | None,
        "graph_textual": agreement_judge_result_dict | None,
        "tabular_v2_textual": agreement_judge_result_dict | None,
    }
    Each present value has {"agreement": bool | None, "reason": str,
    "judge_error": bool}. A pair is None if it was never computed
    (e.g. one of the two pipelines had success=False so no comparison
    was made against it).

    Treat agreement=None (judge_error=True) as agreement=False for
    grouping purposes, but record judge_error_pairs listing which
    pairs had a judge error, so this is visible downstream.
    """
    judge_error_pairs: list[str] = []
    computed: dict[str, dict] = {}
    for key in _PAIR_PIPELINES:
        value = pairwise.get(key)
        if value is None:
            continue
        computed[key] = value
        if value.get("judge_error") or value.get("agreement") is None:
            judge_error_pairs.append(key)

    n_computed = len(computed)
    true_pairs = [
        key for key, value in computed.items() if _effective_agreement(value)
    ]
    true_count = len(true_pairs)

    base = {
        "true_count": true_count,
        "pairs_computed": n_computed,
        "judge_error_pairs": judge_error_pairs,
    }

    if n_computed == 0:
        return {
            **base,
            "agreement_group": [],
            "transitivity_conflict": False,
            "agreement_strength": "none",
            "agreement_independence": "none",
            "decision_rule": "not_applicable",
        }

    if n_computed == 1:
        only_key = next(iter(computed))
        pipes = list(_PAIR_PIPELINES[only_key])
        if _effective_agreement(computed[only_key]):
            strength, independence = _pair_strength_independence(pipes)
            return {
                **base,
                "agreement_group": pipes,
                "transitivity_conflict": False,
                "agreement_strength": strength,
                "agreement_independence": independence,
                "decision_rule": "single_pair_agreement_two_valid",
            }
        return {
            **base,
            "agreement_group": [],
            "transitivity_conflict": False,
            "agreement_strength": "none",
            "agreement_independence": "none",
            "decision_rule": "no_agreement_two_valid",
        }

    # n_computed == 3 (or theoretically 2 if a pair was omitted — treat by true_count)
    if n_computed >= 3:
        if true_count == 3:
            return {
                **base,
                "agreement_group": list(_PIPELINE_ORDER),
                "transitivity_conflict": False,
                "agreement_strength": "strongest",
                "agreement_independence": "all_sources_converge",
                "decision_rule": "all_three_agree",
            }
        if true_count == 2:
            return {
                **base,
                "agreement_group": [],
                "transitivity_conflict": True,
                "agreement_strength": "conflict",
                "agreement_independence": "inconsistent",
                "decision_rule": "transitivity_conflict",
            }
        if true_count == 1:
            only_key = true_pairs[0]
            pipes = list(_PAIR_PIPELINES[only_key])
            strength, independence = _pair_strength_independence(pipes)
            return {
                **base,
                "agreement_group": pipes,
                "transitivity_conflict": False,
                "agreement_strength": strength,
                "agreement_independence": independence,
                "decision_rule": "single_pair_agreement",
            }
        # true_count == 0
        return {
            **base,
            "agreement_group": [],
            "transitivity_conflict": False,
            "agreement_strength": "none",
            "agreement_independence": "none",
            "decision_rule": "no_agreement",
        }

    # Exactly 2 pairs computed (unusual): apply analogous true_count logic.
    if true_count == 2:
        # Both computed pairs agree — group as union of their pipelines.
        group: list[str] = []
        for key in true_pairs:
            for p in _PAIR_PIPELINES[key]:
                if p not in group:
                    group.append(p)
        # Preserve stable order in group
        group = [p for p in _PIPELINE_ORDER if p in group]
        return {
            **base,
            "agreement_group": group,
            "transitivity_conflict": False,
            "agreement_strength": "strong" if "textual" in group else "moderate",
            "agreement_independence": (
                "more_independent_sources"
                if "textual" in group
                else "shared_upstream_source"
            ),
            "decision_rule": "single_pair_agreement",
        }
    if true_count == 1:
        only_key = true_pairs[0]
        pipes = list(_PAIR_PIPELINES[only_key])
        strength, independence = _pair_strength_independence(pipes)
        return {
            **base,
            "agreement_group": pipes,
            "transitivity_conflict": False,
            "agreement_strength": strength,
            "agreement_independence": independence,
            "decision_rule": "single_pair_agreement",
        }
    return {
        **base,
        "agreement_group": [],
        "transitivity_conflict": False,
        "agreement_strength": "none",
        "agreement_independence": "none",
        "decision_rule": "no_agreement",
    }


def _score_value(evidence_result: dict | None) -> float | None:
    if evidence_result is None:
        return None
    score = evidence_result.get("evidence_score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _dim(evidence_result: dict | None, key: str) -> float:
    if evidence_result is None:
        return float("-inf")
    value = evidence_result.get(key)
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _pick_best(candidates: list[str], evidence_scores: dict) -> tuple[str, str]:
    """
    candidates: list of pipeline names to choose among.
    evidence_scores: {pipeline_name: evidence_judge_result_dict}
    Returns (chosen_pipeline_name, tie_break_note) where tie_break_note
    is "" normally or a description if a tie-break rule was invoked.
    """
    if not candidates:
        raise ValueError("_pick_best called with empty candidates")

    scored = [(c, _score_value(evidence_scores.get(c))) for c in candidates]
    usable = [(c, s) for c, s in scored if s is not None]

    if not usable:
        # All evidence_score None — stable pipeline order among candidates.
        for name in _PIPELINE_ORDER:
            if name in candidates:
                return name, "evidence_scores_unavailable, used stable pipeline order"
        return candidates[0], "evidence_scores_unavailable, used stable pipeline order"

    max_score = max(s for _, s in usable)
    near_max = [c for c, s in usable if abs(s - max_score) <= 0.0001]

    if len(near_max) == 1:
        return near_max[0], ""

    # Tie-break among near_max
    # 1. groundedness
    best_g = max(_dim(evidence_scores.get(c), "groundedness") for c in near_max)
    by_g = [
        c
        for c in near_max
        if abs(_dim(evidence_scores.get(c), "groundedness") - best_g) <= 0.0001
    ]
    if len(by_g) == 1:
        return by_g[0], "tie broken by groundedness"

    # 2. completeness
    best_c = max(_dim(evidence_scores.get(c), "completeness") for c in by_g)
    by_c = [
        c
        for c in by_g
        if abs(_dim(evidence_scores.get(c), "completeness") - best_c) <= 0.0001
    ]
    if len(by_c) == 1:
        return by_c[0], "tie broken by completeness"

    # 3. relevance
    best_r = max(_dim(evidence_scores.get(c), "relevance") for c in by_c)
    by_r = [
        c
        for c in by_c
        if abs(_dim(evidence_scores.get(c), "relevance") - best_r) <= 0.0001
    ]
    if len(by_r) == 1:
        return by_r[0], "tie broken by relevance"

    # 4. stable order
    for name in _PIPELINE_ORDER:
        if name in by_r:
            return name, "tie broken by stable order"
    return by_r[0], "tie broken by stable order"


def _summarize_failures(live_results: dict, failed: list[str]) -> str:
    parts = []
    for p in failed:
        err = (live_results.get(p) or {}).get("error")
        parts.append(f"{p}: {err}" if err else f"{p}: (no error detail)")
    return "; ".join(parts) if parts else "none"


def _format_score(evidence_scores: dict, pipeline: str) -> str:
    score = _score_value(evidence_scores.get(pipeline))
    if score is None:
        return "None"
    return f"{score:.4g}"


# ---------------------------------------------------------------------------
# ARCHITECTURAL BOUNDARY (enforced):
# select_fusion_answer must NEVER accept ground_truth, semantic_similarity,
# llm_judge_scores, combined_score, gold_cypher, or gold_sql as parameters.
# Selection is GT-free: live adapter results + evidence scores + pairwise
# agreement only. Evaluation-time GT metrics belong outside this module.
# ---------------------------------------------------------------------------
def select_fusion_answer(
    live_results: dict,
    evidence_scores: dict,
    pairwise: dict,
) -> dict:
    """
    live_results: {"graph": adapter_result, "tabular_v2": adapter_result,
                   "textual": adapter_result} — each the shared shape
                   from adapters.py (has "success", "answer", etc.)
    evidence_scores: {"graph": evidence_judge_result, "tabular_v2": ...,
                      "textual": ...} — each from score_evidence()
    pairwise: {"graph_tabular_v2": ..., "graph_textual": ...,
               "tabular_v2_textual": ...} — each from
               judge_pairwise_agreement() or None if not computed
    """
    valid = [
        p
        for p in _PIPELINE_ORDER
        if (live_results.get(p) or {}).get("success") is True
    ]
    failed = [p for p in _PIPELINE_ORDER if p not in valid]

    empty_meta = {
        "agreement_group": [],
        "transitivity_conflict": False,
        "agreement_strength": "none",
        "agreement_independence": "none",
        "tie_break_note": "",
        "judge_error_pairs": [],
    }

    # Rule 0 — zero valid
    if len(valid) == 0:
        return {
            "selected_pipeline": None,
            "selected_answer": None,
            "selected_evidence_score": None,
            "decision_rule": "all_pipelines_failed",
            "decision_reason": (
                "All three pipelines failed: " + _summarize_failures(live_results, failed)
            ),
            **empty_meta,
        }

    # Exactly one valid
    if len(valid) == 1:
        chosen = valid[0]
        answer = (live_results.get(chosen) or {}).get("answer")
        score = _score_value(evidence_scores.get(chosen))
        return {
            "selected_pipeline": chosen,
            "selected_answer": answer,
            "selected_evidence_score": score,
            "decision_rule": "only_one_valid",
            "decision_reason": (
                f"Only {chosen} succeeded; others failed: "
                + _summarize_failures(live_results, failed)
            ),
            "agreement_group": [],
            "transitivity_conflict": False,
            "agreement_strength": "none",
            "agreement_independence": "none",
            "tie_break_note": "",
            "judge_error_pairs": [],
        }

    # 2 or 3 valid — restrict pairwise to pairs where BOTH pipelines are valid
    restricted: dict[str, Any] = {}
    for key, (a, b) in _PAIR_PIPELINES.items():
        if a in valid and b in valid:
            restricted[key] = pairwise.get(key)
        else:
            restricted[key] = None

    grouping = build_agreement_grouping(restricted)
    agreement_group = list(grouping.get("agreement_group") or [])

    if agreement_group:
        candidates = [p for p in agreement_group if p in valid]
    else:
        candidates = list(valid)

    chosen, tie_break_note = _pick_best(candidates, evidence_scores)
    answer = (live_results.get(chosen) or {}).get("answer")
    score = _score_value(evidence_scores.get(chosen))

    decision_rule = grouping.get("decision_rule") or "no_agreement"
    decision_reason = _build_decision_reason(
        decision_rule=decision_rule,
        agreement_group=agreement_group,
        candidates=candidates,
        chosen=chosen,
        evidence_scores=evidence_scores,
        tie_break_note=tie_break_note,
        grouping=grouping,
    )

    return {
        "selected_pipeline": chosen,
        "selected_answer": answer,
        "selected_evidence_score": score,
        "decision_rule": decision_rule,
        "decision_reason": decision_reason,
        "agreement_group": agreement_group,
        "transitivity_conflict": bool(grouping.get("transitivity_conflict")),
        "agreement_strength": grouping.get("agreement_strength") or "none",
        "agreement_independence": grouping.get("agreement_independence") or "none",
        "tie_break_note": tie_break_note,
        "judge_error_pairs": list(grouping.get("judge_error_pairs") or []),
    }


def _build_decision_reason(
    *,
    decision_rule: str,
    agreement_group: list[str],
    candidates: list[str],
    chosen: str,
    evidence_scores: dict,
    tie_break_note: str,
    grouping: dict,
) -> str:
    chosen_score = _format_score(evidence_scores, chosen)
    others = [p for p in candidates if p != chosen]
    other_scores = ", ".join(
        f"{p}={_format_score(evidence_scores, p)}" for p in others
    )

    if decision_rule == "all_three_agree":
        reason = (
            f"All three pipelines agreed; selected {chosen} with highest "
            f"evidence score ({chosen_score}"
            + (f" vs {other_scores}" if other_scores else "")
            + ")"
        )
    elif decision_rule in ("single_pair_agreement", "single_pair_agreement_two_valid"):
        group_str = " and ".join(agreement_group) if agreement_group else "pair"
        reason = (
            f"{group_str} agreed; selected {chosen} with higher evidence "
            f"score ({chosen_score}"
            + (f" vs {other_scores}" if other_scores else "")
            + ")"
        )
    elif decision_rule == "transitivity_conflict":
        reason = (
            f"Transitivity conflict among pairwise agreements; fell back to "
            f"highest evidence among valid pipelines; selected {chosen} "
            f"({chosen_score}"
            + (f" vs {other_scores}" if other_scores else "")
            + ")"
        )
    elif decision_rule in ("no_agreement", "no_agreement_two_valid"):
        reason = (
            f"No pairwise agreement; selected {chosen} with highest evidence "
            f"score among valid pipelines ({chosen_score}"
            + (f" vs {other_scores}" if other_scores else "")
            + ")"
        )
    else:
        reason = (
            f"decision_rule={decision_rule}; selected {chosen} "
            f"(score={chosen_score})"
        )

    if tie_break_note:
        reason = f"{reason}; {tie_break_note}"
    if grouping.get("judge_error_pairs"):
        reason = (
            f"{reason}; judge_error_pairs={grouping.get('judge_error_pairs')}"
        )
    return reason
