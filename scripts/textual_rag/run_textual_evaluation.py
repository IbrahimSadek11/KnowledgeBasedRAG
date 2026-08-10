"""
Run Textual RAG evaluation on the full 100-question benchmark.

Reuses Graph/Tabular scoring from backend.evaluation_service:
  - calculate_semantic_similarity
  - llm_judge_answer
  - combined = (semantic + llm_judge_overall) / 2

No Execution Accuracy (EX): Textual RAG has no structured query to
compare against gold Cypher/SQL — free-text answers only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    reconfig = getattr(_stream, "reconfigure", None)
    if callable(reconfig):
        reconfig(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from backend.config import (  # noqa: E402
    COST_PER_1K_EMBEDDING,
    COST_PER_1K_INPUT,
    COST_PER_1K_OUTPUT,
)
from backend.evaluation_service import (  # noqa: E402
    calculate_semantic_similarity,
    init_evaluator,
    llm_judge_answer,
)
from backend.textual_rag.textual_rag_service import answer_question  # noqa: E402

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

RESULTS_DIR = REPO_ROOT / "evaluation_results" / "textual_rag"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_dataset_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Textual RAG evaluation (dataset selectable via --30 / --100)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--30",
        dest="use_30",
        action="store_true",
        help="Use data/specialization_test_30.json (30-question specialization benchmark)",
    )
    group.add_argument(
        "--100",
        dest="use_100",
        action="store_true",
        help="Use data/test_dataset.json (full 100-question benchmark; default)",
    )
    return parser.parse_args()


_args = _parse_dataset_args()
if _args.use_30:
    TEST_DATASET_PATH = REPO_ROOT / "data" / "specialization_test_30.json"
    DATASET_LABEL = "specialization30"
    DATASET_EXPECTED = 30
    DATASET_DISPLAY = "specialization 30"
else:
    TEST_DATASET_PATH = REPO_ROOT / "data" / "test_dataset.json"
    DATASET_LABEL = "full100"
    DATASET_EXPECTED = 100
    DATASET_DISPLAY = "full 100"


def _truncate(text: str, n: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "..."


def _avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


print("=" * 80)
print("TEXTUAL RAG EVALUATION")
print("(semantic similarity + LLM judge only — no EX)")
print("=" * 80)
print("=" * 80)
print(f"DATASET: {DATASET_DISPLAY}")
print(f"Path: {TEST_DATASET_PATH}")
print(f"Questions expected: {DATASET_EXPECTED}")
print("=" * 80)

print("\nInitializing evaluation tools (reused from Graph/Tabular RAG)...")
judge_llm, embeddings = init_evaluator()
print("Evaluation tools ready")

print("\nLoading test dataset...")
with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

questions_data = test_data["test_questions"]
print(f"Loaded {len(questions_data)} test questions from {TEST_DATASET_PATH}")

print(f"\nRunning Textual RAG on the {DATASET_DISPLAY} benchmark...")
print("-" * 80)

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
    print(f"Q: {_truncate(question, 160)}")

    start_time = time.perf_counter()
    error = None
    system_answer = ""
    retrieved_docs: list[str] = []
    success = False

    try:
        chain_result = answer_question(question)
        system_answer = (chain_result.get("answer") or "").strip()
        retrieved_docs = list(chain_result.get("retrieved_docs") or [])
        success = bool(system_answer) and not system_answer.lower().startswith("error")
    except Exception as exc:  # noqa: BLE001 — continue benchmark
        error = str(exc)
        system_answer = f"ERROR: {exc}"
        retrieved_docs = []
        success = False
        print(f"FAILED {question_id}: {exc}")

    query_time = time.perf_counter() - start_time
    total_time += query_time

    print(f"Answered in {query_time:.2f}s")
    print(f"Retrieved docs ({len(retrieved_docs)}): {retrieved_docs}")
    print(f"A: {_truncate(system_answer, 200)}")

    # Cost estimate — same style as tabular/graph harness
    query_tokens = len(question.split()) * 1.3 + 1000 + len(system_answer.split()) * 1.3
    query_cost = query_tokens / 1000 * COST_PER_1K_INPUT
    total_query_cost += query_cost

    print("Evaluating answer quality...")
    eval_start = time.perf_counter()
    try:
        semantic_score = calculate_semantic_similarity(
            system_answer, ground_truth, embeddings
        )
        judge_scores = llm_judge_answer(
            question, system_answer, ground_truth, judge_llm
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED scoring {question_id}: {exc}")
        semantic_score = 0.0
        judge_scores = {
            "correctness": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "overall": 0.0,
            "reasoning": f"Evaluation error: {exc}",
        }
        if error is None:
            error = f"scoring_error: {exc}"
        success = False
    eval_time = time.perf_counter() - eval_start

    eval_tokens = len(question.split() + system_answer.split() + ground_truth.split()) * 2
    eval_cost = (
        eval_tokens / 1000 * COST_PER_1K_INPUT
        + 200 / 1000 * COST_PER_1K_OUTPUT
        + (len(system_answer) + len(ground_truth)) / 4 * COST_PER_1K_EMBEDDING
    )
    total_eval_cost += eval_cost

    combined_score = (semantic_score + judge_scores["overall"]) / 2

    print(f"   Semantic Similarity: {semantic_score:.2f}")
    print(f"   LLM Judge Overall:   {judge_scores['overall']:.2f}")
    print(f"   Combined Score:      {combined_score:.2f}")
    print(f"   Evaluation time:     {eval_time:.2f}s")

    results.append(
        {
            "question_id": question_id,
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "ground_truth": ground_truth,
            "system_answer": system_answer,
            "retrieved_docs": retrieved_docs,
            "success": success,
            "error": error,
            "time_seconds": query_time,
            "semantic_similarity": semantic_score,
            "llm_judge_scores": judge_scores,
            "combined_score": combined_score,
            "query_cost_usd": query_cost,
            "eval_cost_usd": eval_cost,
        }
    )

# ══════════════════════════════════════════════════════════════
# Aggregate (no EX)
# ══════════════════════════════════════════════════════════════

n = len(results)

overall = {
    "success_rate": _avg(1.0 if r["success"] else 0.0 for r in results),
    "failed_count": sum(1 for r in results if not r["success"]),
    "avg_semantic_similarity": _avg(r["semantic_similarity"] for r in results),
    "avg_llm_judge_overall": _avg(r["llm_judge_scores"]["overall"] for r in results),
    "avg_llm_judge_correctness": _avg(
        r["llm_judge_scores"]["correctness"] for r in results
    ),
    "avg_llm_judge_completeness": _avg(
        r["llm_judge_scores"]["completeness"] for r in results
    ),
    "avg_llm_judge_accuracy": _avg(r["llm_judge_scores"]["accuracy"] for r in results),
    "avg_combined_score": _avg(r["combined_score"] for r in results),
}

category_stats = {}
for cat in sorted({r["category"] for r in results}):
    rows = [r for r in results if r["category"] == cat]
    category_stats[cat] = {
        "count": len(rows),
        "success_rate": _avg(1.0 if r["success"] else 0.0 for r in rows),
        "avg_semantic_similarity": _avg(r["semantic_similarity"] for r in rows),
        "avg_llm_judge": _avg(r["llm_judge_scores"]["overall"] for r in rows),
        "avg_combined": _avg(r["combined_score"] for r in rows),
    }

difficulty_stats = {}
for diff in sorted({r["difficulty"] for r in results}):
    rows = [r for r in results if r["difficulty"] == diff]
    difficulty_stats[diff] = {
        "count": len(rows),
        "success_rate": _avg(1.0 if r["success"] else 0.0 for r in rows),
        "avg_semantic_similarity": _avg(r["semantic_similarity"] for r in rows),
        "avg_llm_judge": _avg(r["llm_judge_scores"]["overall"] for r in rows),
        "avg_combined": _avg(r["combined_score"] for r in rows),
    }

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = RESULTS_DIR / f"semantic_evaluation_{DATASET_LABEL}_{timestamp}.json"

report = {
    "metadata": {
        "timestamp": timestamp,
        "pipeline": "textual_rag",
        "dataset_label": DATASET_LABEL,
        "dataset_path": str(TEST_DATASET_PATH),
        "question_count": len(questions_data),
        "run_type": "full_benchmark",
        "total_questions": n,
        "total_time_seconds": total_time,
        "avg_time_per_question": (total_time / n) if n else 0.0,
        "query_cost_usd": total_query_cost,
        "evaluation_cost_usd": total_eval_cost,
        "total_cost_usd": total_query_cost + total_eval_cost,
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "evaluation_type": "semantic_similarity + llm_judge",
        "combined_score_formula": "(semantic_similarity + llm_judge_overall) / 2",
        "ex_metric": None,
        "ex_note": (
            "Execution Accuracy is not applicable to Textual RAG "
            "(no structured query / gold result-set comparison)."
        ),
        "scoring_reused_from": "backend/evaluation_service.py (Graph/Tabular RAG)",
        "dataset_path": str(TEST_DATASET_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
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
print("TEXTUAL RAG FULL 100-QUESTION SUMMARY")
print("=" * 80)

print("\nPer-question results:")
print(f"  {'ID':<5} {'CATEGORY':<20} {'DIFFICULTY':<11} {'COMBINED':>8}  {'OK':<6}")
for r in results:
    ok = "YES" if r["success"] else "NO"
    print(
        f"  {r['question_id']:<5} {r['category']:<20} {r['difficulty']:<11} "
        f"{r['combined_score']:>8.2f}  {ok:<6}"
    )

print("\nOverall:")
print(f"  Questions: {n}")
print(
    f"  Success rate: {overall['success_rate'] * 100:.1f}% "
    f"({n - overall['failed_count']}/{n}; {overall['failed_count']} failed)"
)
print(f"  Avg Semantic Similarity: {overall['avg_semantic_similarity']:.3f}")
print(f"  Avg LLM Judge Overall:   {overall['avg_llm_judge_overall']:.3f}")
print(f"  Avg Combined Score:      {overall['avg_combined_score']:.3f}")
print("  Execution Accuracy (EX): N/A (not computed for Textual RAG)")

print("\nBy Category:")
for cat, s in sorted(
    category_stats.items(), key=lambda x: x[1]["avg_combined"], reverse=True
):
    print(
        f"  {cat:<20} n={s['count']}  combined={s['avg_combined']:.2f}  "
        f"semantic={s['avg_semantic_similarity']:.2f}  "
        f"judge={s['avg_llm_judge']:.2f}  "
        f"success={s['success_rate'] * 100:.0f}%"
    )

print("\nBy Difficulty:")
for diff, s in sorted(difficulty_stats.items()):
    print(
        f"  {diff:<11} n={s['count']}  combined={s['avg_combined']:.2f}  "
        f"semantic={s['avg_semantic_similarity']:.2f}  "
        f"judge={s['avg_llm_judge']:.2f}  "
        f"success={s['success_rate'] * 100:.0f}%"
    )

print(f"\nQuery cost: ${total_query_cost:.4f}")
print(f"Evaluation cost: ${total_eval_cost:.4f}")
print(f"Total cost: ${total_query_cost + total_eval_cost:.4f}")
print(f"Total time: {total_time:.2f}s  (avg {total_time / n:.2f}s/question)" if n else "")

successful = [r for r in results if r["success"]]
print("\nBest Answers (Top 3):")
best = sorted(successful, key=lambda x: x["combined_score"], reverse=True)[:3]
if not best:
    print("  (none)")
for r in best:
    print(f"  {r['question_id']}: {_truncate(r['question'], 60)}")
    print(f"    Combined Score: {r['combined_score']:.2f}")

print("\nWorst Answers (Bottom 3):")
worst = sorted(successful, key=lambda x: x["combined_score"])[:3]
if not worst:
    print("  (none)")
for r in worst:
    print(f"  {r['question_id']}: {_truncate(r['question'], 60)}")
    print(f"    Combined Score: {r['combined_score']:.2f}")
    print(f"    Reason: {_truncate(r['llm_judge_scores'].get('reasoning', ''), 80)}")

print(f"\nReport saved to: {report_file}")
print("=" * 80)
