"""
🎯 Run Tabular RAG Evaluation (smoke test)
==========================================

Batch evaluator for the Tabular RAG pipeline (backend/tabular_rag/tabular_chain.py).

It reuses Graph RAG's *exact* scoring methodology so the numbers are directly
comparable across pipelines:
  - same embeddings model + cosine similarity  (calculate_semantic_similarity)
  - same French LLM-as-judge prompt            (llm_judge_answer)
  - same combined formula                       (semantic + judge overall) / 2
  - same cost constants                         (backend.config)

For this run it only executes a hand-picked 12-question smoke-test subset to
validate the harness before spending time/cost on the full 100.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 console output (Windows terminals default to cp1252, which
# cannot encode the emoji/accented characters used below).
for _stream in (sys.stdout, sys.stderr):
    reconfig = getattr(_stream, "reconfigure", None)
    if callable(reconfig):
        reconfig(encoding="utf-8", errors="replace")

# Add project root to path so `backend` imports resolve
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.tabular_rag.tabular_chain import answer_question
from backend.evaluation_service import (
    init_evaluator,
    calculate_semantic_similarity,
    llm_judge_answer,
)
from backend.config import (
    COST_PER_1K_INPUT,
    COST_PER_1K_OUTPUT,
    COST_PER_1K_EMBEDDING,
)
from gold_queries import GOLD_QUERIES, EX_NOT_APPLICABLE, compare_execution

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TEST_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "test_dataset.json")
DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "tabular.db"))
RESULTS_DIR = Path(SCRIPT_DIR) / ".." / ".." / "evaluation_results" / "tabular_rag"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The tabular chain's control-flow fallback when no valid SQL could be produced.
FALLBACK_ANSWER = "Could not generate a valid query after retries."

# French phrases that indicate an honest "no information available" answer.
# (The English control-flow fallback is handled separately via produced_answer.)
NO_INFO_MARKERS = [
    "pas disponible",
    "non disponible",
    "aucune information",
    "pas d'information",
    "ne dispose pas",
    "n'est pas disponible",
    "aucune donnée",
    "pas de données",
    "n'y a pas d'informations",
]

# Canonical French no-info answer used to score honest unanswerable outcomes,
# so the English control-flow fallback string is not unfairly penalized.
CANONICAL_NO_INFO = "L'information demandée n'est pas disponible dans le système."


def is_no_info_shaped(answer: str) -> bool:
    low = (answer or "").lower()
    return any(marker in low for marker in NO_INFO_MARKERS)


print("=" * 80)
print("🎯 TABULAR RAG EVALUATION — FULL 100-QUESTION BENCHMARK + EX")
print("=" * 80)

# ══════════════════════════════════════════════════════════════
# Initialize evaluator (reused from Graph RAG)
# ══════════════════════════════════════════════════════════════

print("\n🤖 Initializing evaluation tools (reused from Graph RAG)...")
judge_llm, embeddings = init_evaluator()
print("✅ Evaluation tools ready")

# ══════════════════════════════════════════════════════════════
# Load full test dataset (all questions)
# ══════════════════════════════════════════════════════════════

print("\n📥 Loading test dataset...")
with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

questions_data = test_data["test_questions"]
print(f"✅ Loaded {len(questions_data)} test questions (full benchmark + EX)")

# ══════════════════════════════════════════════════════════════
# Run evaluation
# ══════════════════════════════════════════════════════════════

print("\n🚀 Running Tabular RAG on the full 100-question benchmark...")
print("─" * 80)

results = []
total_time = 0.0
total_query_cost = 0.0
total_eval_cost = 0.0

for i, q_data in enumerate(questions_data, 1):
    question_id = q_data["question_id"]
    question = q_data["question"]
    ground_truth = q_data["ground_truth"]
    category = q_data["category"]
    difficulty = q_data["difficulty"]

    print(f"\n[{i}/{len(questions_data)}] {question_id} ({difficulty}) - {category}")
    print(f"Q: {question}")

    # ── Step 1: get answer from the Tabular RAG chain ──
    start_time = time.perf_counter()
    try:
        chain_result = answer_question(question)
        error = None
    except Exception as exc:  # noqa: BLE001 - record and continue
        chain_result = {
            "question": question,
            "sql": None,
            "rows": None,
            "answer": f"ERROR: {exc}",
            "attempts": [],
        }
        error = str(exc)
    query_time = time.perf_counter() - start_time
    total_time += query_time

    generated_sql = chain_result.get("sql")
    raw_rows = chain_result.get("rows")
    system_answer = chain_result.get("answer", "")
    attempts = chain_result.get("attempts", [])
    num_attempts = len(attempts)

    # "produced any answer at all" vs "could not generate a valid query"
    produced_answer = generated_sql is not None and error is None

    print(f"⏱️  Answered in {query_time:.2f}s ({num_attempts} attempt(s))")
    print(f"SQL: {generated_sql}")
    print(f"ROWS: {raw_rows}")
    print(f"A: {system_answer}")

    # ── Execution Accuracy (EX) — separate axis from answer quality ──
    gold_sql = None
    execution_match = None
    ex_error = None
    ex_na_reason = None
    if question_id in GOLD_QUERIES:
        gold_sql = GOLD_QUERIES[question_id]
        execution_match, ex_error = compare_execution(generated_sql, gold_sql, DB_PATH)
        if execution_match:
            print("EX: MATCH")
        else:
            detail = f" ({ex_error})" if ex_error else ""
            print(f"EX: MISMATCH{detail}")
    elif question_id in EX_NOT_APPLICABLE:
        ex_na_reason = EX_NOT_APPLICABLE[question_id]
        print(f"EX: N/A ({ex_na_reason})")
    else:
        print("EX: N/A (no gold query defined for this question)")

    # Estimate query cost with the SAME formula Graph RAG uses
    query_tokens = len(question.split()) * 1.3 + 1000 + len(system_answer.split()) * 1.3
    query_cost = query_tokens / 1000 * COST_PER_1K_INPUT
    total_query_cost += query_cost

    # ── Special handling for the unanswerable category ──
    # A good result = the system honestly reported "no information", NOT a
    # word-for-word match. Score such honest outcomes against a canonical
    # French no-info answer so the English control-flow fallback string isn't
    # unfairly penalized. (Additive; every other category is scored as-is.)
    scored_answer = system_answer
    honest_unanswerable = False
    scoring_note = None
    if category == "unanswerable":
        honest_unanswerable = (not produced_answer) or is_no_info_shaped(system_answer)
        if not produced_answer:
            # English control-flow fallback: declining to answer IS correct for
            # an unanswerable question. Score against a canonical French no-info
            # statement so the harness artifact isn't unfairly penalized.
            scored_answer = CANONICAL_NO_INFO
            scoring_note = (
                "unanswerable: pipeline declined to answer (no valid query); "
                "scored against a canonical French no-info statement per note."
            )
        elif honest_unanswerable:
            scoring_note = "unanswerable: honest no-info answer, scored as-is."

    # ── Step 2: score (reused Graph RAG methodology) ──
    print("📊 Evaluating answer quality...")
    eval_start = time.perf_counter()
    semantic_score = calculate_semantic_similarity(scored_answer, ground_truth, embeddings)
    judge_scores = llm_judge_answer(question, scored_answer, ground_truth, judge_llm)
    eval_time = time.perf_counter() - eval_start

    eval_tokens = len(question.split() + scored_answer.split() + ground_truth.split()) * 2
    eval_cost = (
        eval_tokens / 1000 * COST_PER_1K_INPUT
        + 200 / 1000 * COST_PER_1K_OUTPUT
        + (len(scored_answer) + len(ground_truth)) / 4 * COST_PER_1K_EMBEDDING
    )
    total_eval_cost += eval_cost

    combined_score = (semantic_score + judge_scores["overall"]) / 2

    print(f"   Semantic Similarity: {semantic_score:.2f}")
    print(f"   LLM Judge Overall: {judge_scores['overall']:.2f}")
    print(f"   Combined Score: {combined_score:.2f}")
    print(f"   Evaluation time: {eval_time:.2f}s")
    if scoring_note:
        print(f"   ℹ️  {scoring_note}")

    results.append({
        "question_id": question_id,
        "question": question,
        "category": category,
        "difficulty": difficulty,
        "ground_truth": ground_truth,
        "generated_sql": generated_sql,
        "gold_sql": gold_sql,
        "execution_match": execution_match,
        "ex_error": ex_error,
        "ex_na_reason": ex_na_reason,
        "raw_rows": [list(r) for r in raw_rows] if raw_rows is not None else None,
        "system_answer": system_answer,
        "num_attempts": num_attempts,
        "attempts": attempts,
        "time_seconds": query_time,
        "produced_answer": produced_answer,
        "honest_unanswerable": honest_unanswerable,
        "scoring_note": scoring_note,
        "semantic_similarity": semantic_score,
        "llm_judge_scores": judge_scores,
        "combined_score": combined_score,
        "query_cost_usd": query_cost,
        "eval_cost_usd": eval_cost,
    })

# ══════════════════════════════════════════════════════════════
# Aggregate
# ══════════════════════════════════════════════════════════════

n = len(results)


def avg(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


ex_applicable = [r for r in results if r["execution_match"] is not None]
ex_matches = sum(1 for r in ex_applicable if r["execution_match"])
ex_rate = (ex_matches / len(ex_applicable)) if ex_applicable else 0.0

overall = {
    "avg_combined_score": avg(r["combined_score"] for r in results),
    "avg_semantic_similarity": avg(r["semantic_similarity"] for r in results),
    "avg_llm_judge_overall": avg(r["llm_judge_scores"]["overall"] for r in results),
    "success_rate": avg(1.0 if r["produced_answer"] else 0.0 for r in results),
    "produced_answer_count": sum(1 for r in results if r["produced_answer"]),
    "no_answer_count": sum(1 for r in results if not r["produced_answer"]),
    "execution_accuracy": {
        "applicable_count": len(ex_applicable),
        "match_count": ex_matches,
        "mismatch_count": len(ex_applicable) - ex_matches,
        "na_count": sum(1 for r in results if r["execution_match"] is None),
        "ex_rate": ex_rate,
    },
}

category_stats = {}
for cat in sorted(set(r["category"] for r in results)):
    rows = [r for r in results if r["category"] == cat]
    category_stats[cat] = {
        "count": len(rows),
        "avg_combined": avg(r["combined_score"] for r in rows),
        "avg_semantic": avg(r["semantic_similarity"] for r in rows),
        "avg_llm_judge": avg(r["llm_judge_scores"]["overall"] for r in rows),
        "success_rate": avg(1.0 if r["produced_answer"] else 0.0 for r in rows),
    }

difficulty_stats = {}
for diff in sorted(set(r["difficulty"] for r in results)):
    rows = [r for r in results if r["difficulty"] == diff]
    difficulty_stats[diff] = {
        "count": len(rows),
        "avg_combined": avg(r["combined_score"] for r in rows),
        "avg_semantic": avg(r["semantic_similarity"] for r in rows),
        "avg_llm_judge": avg(r["llm_judge_scores"]["overall"] for r in rows),
        "success_rate": avg(1.0 if r["produced_answer"] else 0.0 for r in rows),
    }

# ══════════════════════════════════════════════════════════════
# Save detailed JSON report
# ══════════════════════════════════════════════════════════════

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = RESULTS_DIR / f"tabular_eval_full100_ex_{timestamp}.json"

report = {
    "metadata": {
        "timestamp": timestamp,
        "pipeline": "tabular_rag",
        "run_type": "full_benchmark_ex",
        "total_questions": n,
        "total_time_seconds": total_time,
        "avg_time_per_question": total_time / n if n else 0.0,
        "query_cost_usd": total_query_cost,
        "evaluation_cost_usd": total_eval_cost,
        "total_cost_usd": total_query_cost + total_eval_cost,
        "model": "gpt-4o-mini",
        "evaluation_type": "semantic_similarity + llm_judge + execution_accuracy",
        "combined_score_formula": "(semantic_similarity + llm_judge_overall) / 2",
        "ex_formula": "exact result-set match vs gold SQL; N/A excluded from denominator",
        "scoring_reused_from": "backend/evaluation_service.py (Graph RAG)",
    },
    "overall_metrics": overall,
    "category_stats": category_stats,
    "difficulty_stats": difficulty_stats,
    "detailed_results": results,
}

with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# ══════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("📊 TABULAR RAG FULL 100-QUESTION + EX SUMMARY")
print("=" * 80)

print("\nPer-question results:")
print(f"  {'ID':<5} {'CATEGORY':<20} {'DIFFICULTY':<11} {'COMBINED':>8}  {'EX':<10}")
for r in results:
    if r["execution_match"] is True:
        ex_label = "MATCH"
    elif r["execution_match"] is False:
        ex_label = "MISMATCH"
    else:
        ex_label = "N/A"
    print(
        f"  {r['question_id']:<5} {r['category']:<20} {r['difficulty']:<11} "
        f"{r['combined_score']:>8.2f}  {ex_label:<10}"
    )

print("\n📈 Overall:")
print(f"  Questions: {n}")
print(
    f"  Success rate (produced an answer): "
    f"{overall['success_rate'] * 100:.1f}% "
    f"({overall['produced_answer_count']}/{n}; "
    f"{overall['no_answer_count']} could-not-generate)"
)
print(f"  Avg Semantic Similarity: {overall['avg_semantic_similarity']:.3f}")
print(f"  Avg LLM Judge Overall: {overall['avg_llm_judge_overall']:.3f}")
print(f"  Avg Combined Score: {overall['avg_combined_score']:.3f}")
ex_info = overall["execution_accuracy"]
print(
    f"  Execution Accuracy (EX): {ex_info['ex_rate'] * 100:.1f}% "
    f"({ex_info['match_count']}/{ex_info['applicable_count']} applicable; "
    f"{ex_info['na_count']} N/A excluded)"
)

print("\n📊 By Category:")
for cat, s in sorted(category_stats.items(), key=lambda x: x[1]["avg_combined"], reverse=True):
    print(
        f"  {cat:<20} n={s['count']}  combined={s['avg_combined']:.2f}  "
        f"semantic={s['avg_semantic']:.2f}  judge={s['avg_llm_judge']:.2f}  "
        f"success={s['success_rate'] * 100:.0f}%"
    )

print("\n📊 By Difficulty:")
for diff, s in sorted(difficulty_stats.items()):
    print(
        f"  {diff:<11} n={s['count']}  combined={s['avg_combined']:.2f}  "
        f"semantic={s['avg_semantic']:.2f}  judge={s['avg_llm_judge']:.2f}  "
        f"success={s['success_rate'] * 100:.0f}%"
    )

print(f"\n💰 Query cost: ${total_query_cost:.4f}")
print(f"💰 Evaluation cost: ${total_eval_cost:.4f}")
print(f"💰 Total cost: ${total_query_cost + total_eval_cost:.4f}")
print(f"⏱️  Total time: {total_time:.2f}s  (avg {total_time / n:.2f}s/question)")

print(f"\n✅ Report saved to: {report_file}")
print("=" * 80)
