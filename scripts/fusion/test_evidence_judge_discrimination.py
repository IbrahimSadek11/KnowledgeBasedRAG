"""Discrimination stress tests for score_evidence() with synthetic answers."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backend.fusion.evidence_judge as evidence_judge
from backend.fusion.adapters import run_graph_live, run_tabular_v2_live, run_textual_live
from backend.fusion.evidence_judge import (
    PASSAGE_CHAR_CAP,
    _truncate_passages,
    score_evidence,
)


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def test_a(graph_base: dict) -> dict:
    _print_header("TEST A — Groundedness (fabricated fact)")
    corrupted = copy.deepcopy(graph_base)
    corrupted["answer"] = (
        "Dakota est de race Arabe, tout comme Vega et Orion."
    )
    print(f"Corrupted answer: {corrupted['answer']}")
    print(f"Real raw_results: {json.dumps(graph_base.get('raw_results'), ensure_ascii=False)}")

    result = score_evidence(corrupted)
    g = result.get("groundedness")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    expected_met = g is not None and g < 0.3
    soft_met = g is not None and g < 1.0
    print(
        f"EXPECTATION: groundedness << 1.0 (ideally < 0.3). "
        f"actual={g} ideal_met={expected_met} below_1={soft_met}"
    )
    return {
        "test": "A groundedness",
        "expected_direction": "< 0.3 (well below 1.0)",
        "actual_score": g,
        "met_expectation": expected_met,
        "soft_below_1": soft_met,
        "reasoning": (result.get("reasoning") or {}).get("groundedness"),
    }


def test_b() -> dict:
    _print_header("TEST B — Completeness (truncated answer, full evidence)")
    base = run_tabular_v2_live("Quels sont les noms des chevaux dans le système ?")
    print(
        f"adapter success={base.get('success')} "
        f"raw_results_len={len(base.get('raw_results') or [])}"
    )
    corrupted = copy.deepcopy(base)
    corrupted["answer"] = "Il y a un cheval nommé Crepuscule dans le système."
    print(f"Corrupted answer: {corrupted['answer']}")

    result = score_evidence(corrupted)
    c = result.get("completeness")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    expected_met = c is not None and c < 0.3
    soft_met = c is not None and c < 1.0
    print(
        f"EXPECTATION: completeness << 1.0 (ideally < 0.3). "
        f"actual={c} ideal_met={expected_met} below_1={soft_met}"
    )
    return {
        "test": "B completeness",
        "expected_direction": "< 0.3 (well below 1.0)",
        "actual_score": c,
        "met_expectation": expected_met,
        "soft_below_1": soft_met,
        "reasoning": (result.get("reasoning") or {}).get("completeness"),
    }


def test_c(graph_base: dict) -> dict:
    _print_header("TEST C — Relevance (off-topic answer)")
    corrupted = copy.deepcopy(graph_base)
    corrupted["answer"] = (
        "Les compétitions de dressage se déroulent généralement au printemps "
        "et nécessitent un entraînement rigoureux du cheval et du cavalier."
    )
    print(f"Corrupted answer: {corrupted['answer']}")

    result = score_evidence(corrupted)
    r = result.get("relevance")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    expected_met = r is not None and r < 0.3
    soft_met = r is not None and r < 1.0
    print(
        f"EXPECTATION: relevance << 1.0 (ideally < 0.3). "
        f"actual={r} ideal_met={expected_met} below_1={soft_met}"
    )
    return {
        "test": "C relevance",
        "expected_direction": "< 0.3 (well below 1.0)",
        "actual_score": r,
        "met_expectation": expected_met,
        "soft_below_1": soft_met,
        "reasoning": (result.get("reasoning") or {}).get("relevance"),
    }


def test_d() -> dict:
    _print_header("TEST D — success=False skip (no LLM call)")
    mock = {
        "pipeline": "graph",
        "question": "test",
        "answer": None,
        "generated_query": None,
        "raw_results": None,
        "retrieved_documents": None,
        "retrieved_passages": None,
        "success": False,
        "error": "mock Neo4j connection failure",
        "attempts": None,
    }

    call_count = {"n": 0}
    original_call = evidence_judge._call_judge
    original_retry = evidence_judge._call_judge_retry
    original_get = evidence_judge._get_judge_llm

    def _counting_call(*args, **kwargs):
        call_count["n"] += 1
        print("LLM_CALL_DETECTED: _call_judge was invoked (UNEXPECTED for Test D)")
        return original_call(*args, **kwargs)

    def _counting_retry(*args, **kwargs):
        call_count["n"] += 1
        print("LLM_CALL_DETECTED: _call_judge_retry was invoked (UNEXPECTED)")
        return original_retry(*args, **kwargs)

    def _counting_get(*args, **kwargs):
        call_count["n"] += 1
        print("LLM_CALL_DETECTED: _get_judge_llm was invoked (UNEXPECTED)")
        return original_get(*args, **kwargs)

    evidence_judge._call_judge = _counting_call  # type: ignore[assignment]
    evidence_judge._call_judge_retry = _counting_retry  # type: ignore[assignment]
    evidence_judge._get_judge_llm = _counting_get  # type: ignore[assignment]
    try:
        result = score_evidence(mock)
    finally:
        evidence_judge._call_judge = original_call  # type: ignore[assignment]
        evidence_judge._call_judge_retry = original_retry  # type: ignore[assignment]
        evidence_judge._get_judge_llm = original_get  # type: ignore[assignment]

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"OpenAI/LLM helper call count during Test D: {call_count['n']}")

    ok = (
        result.get("execution_quality") == 0.0
        and result.get("groundedness") is None
        and result.get("completeness") is None
        and result.get("relevance") is None
        and result.get("evidence_score") == 0.0
        and call_count["n"] == 0
    )
    print(f"EXPECTATION: early skip, no LLM. met={ok}")
    return {
        "test": "D success=False skip",
        "expected_direction": "eq=0, dims=None, score=0, no LLM",
        "actual_score": result.get("evidence_score"),
        "met_expectation": ok,
        "soft_below_1": True,
        "reasoning": result.get("reasoning"),
        "llm_calls": call_count["n"],
    }


def test_e() -> dict:
    _print_header("TEST E — Truncation sanity (textual)")
    base = run_textual_live("Quelle est la race de Dakota ?")
    passages = list(base.get("retrieved_passages") or [])
    docs = list(base.get("retrieved_documents") or [])
    print(f"passage_count={len(passages)} cap={PASSAGE_CHAR_CAP}")
    total_before = sum(len(p or "") for p in passages)
    print(f"total_chars_before={total_before}")

    capped, truncated_flag = _truncate_passages(passages, PASSAGE_CHAR_CAP)
    total_after = sum(len(p or "") for p in capped)
    shortened = [
        i
        for i, (orig, new) in enumerate(zip(passages, capped))
        if len(new) < len(orig or "")
    ]
    dropped = len(passages) - len(capped)
    print(f"truncated_flag={truncated_flag}")
    print(f"total_chars_after={total_after}")
    print(f"passages_shortened_count={len(shortened)} indices={shortened}")
    print(f"passages_dropped_count={dropped}")
    print(
        "STRATEGY: proportional per-passage shortening (prefix kept, "
        "suffix cut with …[truncated]); does NOT drop list entries."
    )

    # Identify Dakota-specific passage
    dakota_idx = None
    for i, (doc, text) in enumerate(zip(docs, passages)):
        if (doc and "Dakota" in doc) or (text and "Dakota" in text[:200]):
            dakota_idx = i
            break
    print(f"dakota_passage_index={dakota_idx} filename={docs[dakota_idx] if dakota_idx is not None else None}")

    dakota_kept = False
    dakota_still_has_breed = False
    if dakota_idx is not None:
        orig = passages[dakota_idx] or ""
        new = capped[dakota_idx] or ""
        dakota_kept = len(new) > 0 and "Dakota" in new
        # Breed claim often appears early in the horse report
        dakota_still_has_breed = (
            "Selle Français" in new
            or "Selle Francais" in new
            or "race" in new.casefold()
        )
        print(f"dakota_orig_len={len(orig)} dakota_capped_len={len(new)}")
        print(f"dakota_still_contains_name={dakota_kept}")
        print(f"dakota_still_contains_breed_or_race_cue={dakota_still_has_breed}")
        print(f"dakota_capped_prefix={new[:220]!r}")

    evidence = score_evidence(base)
    print("score_evidence on intact textual result:")
    print(json.dumps(evidence, indent=2, ensure_ascii=False, default=str))

    # Expectation for E: truncation uses proportional shortening; Dakota not dropped
    met = (
        dropped == 0
        and (not truncated_flag or len(shortened) > 0)
        and (dakota_idx is None or dakota_kept)
    )
    print(
        f"EXPECTATION: proportional shorten, Dakota passage not dropped. met={met}"
    )
    return {
        "test": "E truncation",
        "expected_direction": "proportional shorten; Dakota kept",
        "actual_score": evidence.get("evidence_score"),
        "met_expectation": met,
        "soft_below_1": True,
        "reasoning": {
            "truncated_evidence": evidence.get("truncated_evidence"),
            "shortened": len(shortened),
            "dropped": dropped,
            "dakota_idx": dakota_idx,
            "dakota_kept": dakota_kept,
        },
    }


def main() -> None:
    _print_header("BASE — run_graph_live for Tests A/C")
    graph_base = run_graph_live("Quelle est la race de Dakota ?")
    print(
        f"success={graph_base.get('success')} "
        f"answer={graph_base.get('answer')!r}"
    )
    print(f"raw_results={json.dumps(graph_base.get('raw_results'), ensure_ascii=False)}")

    rows = [
        test_a(graph_base),
        test_b(),
        test_c(graph_base),
        test_d(),
        test_e(),
    ]

    _print_header("SUMMARY TABLE")
    print(
        f"{'test':<22} | {'expected':<34} | {'actual':<8} | "
        f"{'ideal(<0.3)':<11} | {'below_1.0':<9}"
    )
    print("-" * 100)
    for row in rows:
        actual = row["actual_score"]
        actual_s = "n/a" if actual is None else f"{actual:.3f}"
        print(
            f"{row['test']:<22} | {row['expected_direction']:<34} | "
            f"{actual_s:<8} | {str(row['met_expectation']):<11} | "
            f"{str(row.get('soft_below_1')):<9}"
        )
        if row.get("reasoning") and row["test"].startswith(("A", "B", "C")):
            print(f"  reasoning: {row['reasoning']}")

    print("\nFull row JSON:")
    print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
