"""
🎯 Run Semantic GraphRAG Evaluation
================================

Evaluates answers using semantic similarity and LLM-as-judge
Perfect for when answers are correct but rephrased differently
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path

# Add repository root + this script's directory to path
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(SCRIPT_DIR))

from backend.graph_rag.llm_service import init_graph_chain, invoke_graph_chain_with_cypher_retry
from backend.evaluation_service import init_evaluator, calculate_semantic_similarity, llm_judge_answer
from backend.config import COST_PER_1K_INPUT, COST_PER_1K_OUTPUT, COST_PER_1K_EMBEDDING
from backend.timing_callback import TimingCallbackHandler
from gold_cypher_queries import (
    GOLD_CYPHER_QUERIES,
    EX_NOT_APPLICABLE,
    AMBIGUOUS_FOR_REVIEW,
    LIST_COMPARE_OVERRIDES,
    compare_cypher_execution,
)

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

RESULTS_DIR = REPO_ROOT / "evaluation_results" / "graph_rag"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
QUESTION_TIMEOUT_SECONDS = 500


def _parse_dataset_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic GraphRAG evaluation (dataset selectable via --30 / --100)."
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
    _dataset_path = REPO_ROOT / "data" / "specialization_test_30.json"
    DATASET_LABEL = "specialization30"
    DATASET_EXPECTED = 30
    DATASET_DISPLAY = "specialization 30"
else:
    _dataset_path = REPO_ROOT / "data" / "test_dataset.json"
    DATASET_LABEL = "full100"
    DATASET_EXPECTED = 100
    DATASET_DISPLAY = "full 100"

TEST_DATASET_PATH = str(_dataset_path)

print("=" * 80)
print("🎯 SEMANTIC GRAPHRAG EVALUATION")
print("=" * 80)
print("=" * 80)
print(f"DATASET: {DATASET_DISPLAY}")
print(f"Path: {TEST_DATASET_PATH}")
print(f"Questions expected: {DATASET_EXPECTED}")
print("=" * 80)

# ══════════════════════════════════════════════════════════════
# Initialize evaluator LLM and embeddings
# ══════════════════════════════════════════════════════════════

print("\n🤖 Initializing evaluation tools...")
judge_llm, embeddings = init_evaluator()
print("✅ Evaluation tools ready")

# ══════════════════════════════════════════════════════════════
# Load test dataset
# ══════════════════════════════════════════════════════════════

print("\n📥 Loading test dataset...")
with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

questions_data = test_data['test_questions']
print(f"✅ Loaded {len(questions_data)} test questions")

# ══════════════════════════════════════════════════════════════
# Initialize GraphRAG
# ══════════════════════════════════════════════════════════════

print("\n🔄 Initializing GraphRAG system...")
chain, graph = init_graph_chain()
print("✅ GraphRAG system initialized")

# ══════════════════════════════════════════════════════════════
# Run evaluation
# ══════════════════════════════════════════════════════════════

print("\n🚀 Running evaluation with semantic analysis...")
print("─" * 80)

results = []
total_time = 0
total_query_cost = 0
total_eval_cost = 0
total_cypher_gen_time = 0
total_answer_gen_time = 0
cypher_gen_time_count = 0
answer_gen_time_count = 0


def attach_timing_callback(graph_chain, callback):
    """Attach callback to both LLM calls inside GraphCypherQAChain."""
    graph_chain.cypher_generation_chain.llm.callbacks = [callback]

    qa_chain = getattr(graph_chain, "qa_chain", None)
    qa_llm = getattr(qa_chain, "llm", None)
    if qa_llm is not None:
        qa_llm.callbacks = [callback]


def extract_cypher_and_context(result):
    """Best-effort extraction of generated Cypher and raw Neo4j context."""
    cypher_query = ""
    raw_context = []
    steps = result.get("intermediate_steps") or []

    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("query"):
            cypher_query = step["query"]
        if "context" in step:
            raw_context = step.get("context") or []

    if not cypher_query and steps and isinstance(steps[0], dict):
        cypher_query = steps[0].get("query", "") or ""

    return cypher_query, raw_context


def score_execution_accuracy(question_id, cypher_query, neo4j_graph):
    """Score EX vs locked gold Cypher result set.

    Returns (execution_match, ex_error, ex_na_reason, gold_cypher).
    execution_match is True/False for applicable questions, None for N/A.
    """
    if question_id in EX_NOT_APPLICABLE:
        return None, None, EX_NOT_APPLICABLE[question_id], None

    if question_id in AMBIGUOUS_FOR_REVIEW:
        return None, None, AMBIGUOUS_FOR_REVIEW[question_id], None

    if question_id not in GOLD_CYPHER_QUERIES:
        return None, None, "no gold Cypher defined for this question", None

    gold_cypher = GOLD_CYPHER_QUERIES[question_id]
    list_mode = LIST_COMPARE_OVERRIDES.get(question_id, "multiset")

    if not cypher_query or not str(cypher_query).strip():
        return False, "generated Cypher is None or empty", None, gold_cypher

    try:
        generated_result = neo4j_graph.query(cypher_query)
    except Exception as exc:  # noqa: BLE001 - surface as EX miss
        return False, f"generated Cypher exception: {exc}", None, gold_cypher

    try:
        gold_result = neo4j_graph.query(gold_cypher)
    except Exception as exc:  # noqa: BLE001 - surface as EX miss
        return False, f"gold Cypher exception: {exc}", None, gold_cypher

    matched, err = compare_cypher_execution(
        generated_result, gold_result, list_mode=list_mode
    )
    return matched, err, None, gold_cypher


for i, q_data in enumerate(questions_data, 1):
    question_id = q_data['question_id']
    question = q_data['question']
    ground_truth = q_data['ground_truth']
    category = q_data['category']
    difficulty = q_data['difficulty']
    
    print(f"\n[{i}/{len(questions_data)}] {question_id} ({difficulty}) - {category}")
    print(f"Q: {question[:80]}...")
    
    # ════════════════════════════════════════════════════
    # Step 1: Get answer from GraphRAG
    # ════════════════════════════════════════════════════
    
    start_time = time.time()

    callback = TimingCallbackHandler()
    attach_timing_callback(chain, callback)
    cypher_retry_used = False
    original_cypher_error = None

    try:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            invoke_graph_chain_with_cypher_retry,
            chain,
            {"query": question},
            {"callbacks": [callback]},
        )
        timed_out = False
        try:
            result = future.result(timeout=QUESTION_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            timed_out = True
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            answer = f"TIMEOUT: question exceeded {QUESTION_TIMEOUT_SECONDS}s"
            cypher_query = ""
            query_time = time.time() - start_time
            success = False
            query_cost = 0
            total_time += query_time
            print(f"⏰ Q{question_id} TIMED OUT after {QUESTION_TIMEOUT_SECONDS}s — skipping")
            execution_match, ex_error, ex_na_reason, gold_cypher = score_execution_accuracy(
                question_id, cypher_query, graph
            )
            if execution_match is True:
                print("EX: MATCH")
            elif execution_match is False:
                detail = f" ({ex_error})" if ex_error else ""
                print(f"EX: MISMATCH{detail}")
            else:
                print(f"EX: N/A ({ex_na_reason})")
            results.append({
                'question_id': question_id,
                'question': question,
                'answer': answer,
                'ground_truth': ground_truth,
                'category': category,
                'difficulty': difficulty,
                'time_seconds': query_time,
                'cypher_query': '',
                'gold_cypher': gold_cypher,
                'execution_match': execution_match,
                'ex_error': ex_error,
                'ex_na_reason': ex_na_reason,
                'success': False,
                'semantic_similarity': 0.0,
                'llm_judge_scores': {
                    'correctness': 0.0,
                    'completeness': 0.0,
                    'accuracy': 0.0,
                    'overall': 0.0,
                    'reasoning': 'Timed out'
                },
                'combined_score': 0.0,
                'cypher_generation_time_seconds': None,
                'answer_generation_time_seconds': None,
                'query_cost_usd': 0.0,
                'eval_cost_usd': 0.0,
                'cypher_retry_used': False,
                'original_error': None,
            })
            callback.reset()
            continue
        finally:
            if not timed_out:
                executor.shutdown(wait=True)
        answer = result.get("result", "")

        cypher_query, _raw_context = extract_cypher_and_context(result)
        cypher_retry_used = bool(result.get("cypher_retry_used", False))
        original_cypher_error = result.get("original_error")

        query_time = time.time() - start_time

        # Per-LLM-call timings captured by the callback
        cypher_generation_time = callback.cypher_generation_time
        answer_generation_time = callback.answer_generation_time
        llm_call_count = len(callback.durations)

        if cypher_generation_time is not None:
            total_cypher_gen_time += cypher_generation_time
            cypher_gen_time_count += 1
        if answer_generation_time is not None:
            total_answer_gen_time += answer_generation_time
            answer_gen_time_count += 1

        # Sanity check: GraphCypherQAChain should make exactly 2 LLM calls
        if llm_call_count != 2:
            print(f"   ⚠️  Expected 2 LLM calls, got {llm_call_count}")

        # Estimate query cost
        query_tokens = len(question.split()) * 1.3 + 1000 + len(answer.split()) * 1.3
        query_cost = query_tokens / 1000 * COST_PER_1K_INPUT
        total_query_cost += query_cost

        success = len(answer) > 0 and 'error' not in answer.lower()

    except Exception as e:
        answer = f"ERROR: {str(e)}"
        cypher_query = ""
        query_time = time.time() - start_time
        cypher_generation_time = callback.cypher_generation_time
        answer_generation_time = callback.answer_generation_time
        success = False
        query_cost = 0
        cypher_retry_used = False
        original_cypher_error = str(e)

    total_time += query_time
    
    print(f"✅ Query answered in {query_time:.2f}s")
    print(f"A: {answer[:100]}...")
    cypher_time_text = (
        f"{callback.cypher_generation_time:.2f}s"
        if callback.cypher_generation_time is not None else "N/A"
    )
    answer_time_text = (
        f"{callback.answer_generation_time:.2f}s"
        if callback.answer_generation_time is not None else "N/A"
    )
    print(f"   ⏱️  Cypher generation : {cypher_time_text}")
    print(f"   ⏱️  Answer generation : {answer_time_text}")
    if cypher_retry_used:
        print(f"   🔁 Cypher retry used (original error: {original_cypher_error})")

    # ── Execution Accuracy (EX) — separate axis from answer quality ──
    execution_match, ex_error, ex_na_reason, gold_cypher = score_execution_accuracy(
        question_id, cypher_query, graph
    )
    if execution_match is True:
        print("EX: MATCH")
    elif execution_match is False:
        detail = f" ({ex_error})" if ex_error else ""
        print(f"EX: MISMATCH{detail}")
    else:
        print(f"EX: N/A ({ex_na_reason})")
    
    # ════════════════════════════════════════════════════
    # Step 2: Evaluate answer quality
    # ════════════════════════════════════════════════════
    
    if success:
        print("📊 Evaluating answer quality...")
        
        eval_start = time.time()
        
        # Semantic similarity using embeddings
        semantic_score = calculate_semantic_similarity(answer, ground_truth, embeddings)
        
        # LLM-as-judge evaluation
        judge_scores = llm_judge_answer(question, answer, ground_truth, judge_llm)
        
        eval_time = time.time() - eval_start
        
        # Estimate evaluation cost
        eval_tokens = len(question.split() + answer.split() + ground_truth.split()) * 2
        eval_cost = (eval_tokens / 1000 * COST_PER_1K_INPUT + 
                    200 / 1000 * COST_PER_1K_OUTPUT +  # Judge response
                    (len(answer) + len(ground_truth)) / 4 * COST_PER_1K_EMBEDDING)  # Embeddings
        total_eval_cost += eval_cost
        
        print(f"   Semantic Similarity: {semantic_score:.2f}")
        print(f"   LLM Judge Overall: {judge_scores['overall']:.2f}")
        print(f"   Evaluation time: {eval_time:.2f}s")
        
    else:
        semantic_score = 0.0
        judge_scores = {
            'correctness': 0.0,
            'completeness': 0.0,
            'accuracy': 0.0,
            'overall': 0.0,
            'reasoning': 'Query failed'
        }
    
    # ════════════════════════════════════════════════════
    # Store results
    # ════════════════════════════════════════════════════
    
    results.append({
        'question_id': question_id,
        'question': question,
        'answer': answer,
        'ground_truth': ground_truth,
        'category': category,
        'difficulty': difficulty,
        'time_seconds': query_time,
        'total_time_seconds': query_time,
        'cypher_query': cypher_query,
        'gold_cypher': gold_cypher,
        'execution_match': execution_match,
        'ex_error': ex_error,
        'ex_na_reason': ex_na_reason,
        'cypher_generation_time_seconds': callback.cypher_generation_time,
        'answer_generation_time_seconds': callback.answer_generation_time,
        'success': success,
        'semantic_similarity': semantic_score,
        'llm_judge_scores': judge_scores,
        'combined_score': (semantic_score + judge_scores['overall']) / 2,  # Average
        'cypher_retry_used': cypher_retry_used,
        'original_error': original_cypher_error,
    })
    callback.reset()

print("\n" + "="*80)
print(" Evaluation completed!")
print(f"⏱️  Total query time: {total_time:.2f}s")
print(f"⏱️  Avg query time: {total_time/len(questions_data):.2f}s")
print(f"💰 Query cost: ${total_query_cost:.4f}")
print(f"💰 Evaluation cost: ${total_eval_cost:.4f}")
print(f"💰 Total cost: ${total_query_cost + total_eval_cost:.4f}")

# ══════════════════════════════════════════════════════════════
# Generate report
# ══════════════════════════════════════════════════════════════

print("\n📊 Generating report...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = RESULTS_DIR / f"semantic_evaluation_{DATASET_LABEL}_{timestamp}.json"

# Calculate statistics by category
category_stats = {}
for category in set(r['category'] for r in results):
    cat_results = [r for r in results if r['category'] == category]
    category_stats[category] = {
        'count': len(cat_results),
        'success_rate': sum(1 for r in cat_results if r['success']) / len(cat_results),
        'avg_time': sum(r['time_seconds'] for r in cat_results) / len(cat_results),
        'avg_semantic_similarity': sum(r['semantic_similarity'] for r in cat_results) / len(cat_results),
        'avg_llm_judge': sum(r['llm_judge_scores']['overall'] for r in cat_results) / len(cat_results),
        'avg_combined': sum(r['combined_score'] for r in cat_results) / len(cat_results),
    }

# Calculate statistics by difficulty
difficulty_stats = {}
for difficulty in set(r['difficulty'] for r in results):
    diff_results = [r for r in results if r['difficulty'] == difficulty]
    difficulty_stats[difficulty] = {
        'count': len(diff_results),
        'success_rate': sum(1 for r in diff_results if r['success']) / len(diff_results),
        'avg_time': sum(r['time_seconds'] for r in diff_results) / len(diff_results),
        'avg_semantic_similarity': sum(r['semantic_similarity'] for r in diff_results) / len(diff_results),
        'avg_llm_judge': sum(r['llm_judge_scores']['overall'] for r in diff_results) / len(diff_results),
    }

ex_applicable = [r for r in results if r.get("execution_match") is not None]
ex_matches = sum(1 for r in ex_applicable if r["execution_match"])
ex_rate = (ex_matches / len(ex_applicable)) if ex_applicable else 0.0
execution_accuracy = {
    "applicable_count": len(ex_applicable),
    "match_count": ex_matches,
    "mismatch_count": len(ex_applicable) - ex_matches,
    "na_count": sum(1 for r in results if r.get("execution_match") is None),
    "ex_rate": ex_rate,
}

retry_used = [r for r in results if r.get("cypher_retry_used")]
cypher_retry_stats = {
    "retry_attempts": len(retry_used),
    "retry_recovered": sum(1 for r in retry_used if r.get("success")),
    "retry_success_rate": (
        sum(1 for r in retry_used if r.get("success")) / len(retry_used)
        if retry_used else None
    ),
}

# Full report
report = {
    'metadata': {
        'timestamp': timestamp,
        'dataset_label': DATASET_LABEL,
        'dataset_path': TEST_DATASET_PATH,
        'question_count': len(questions_data),
        'total_questions': len(results),
        'total_time_seconds': total_time,
        'avg_time_per_question': total_time / len(results),
        'avg_cypher_generation_time_seconds': (
            total_cypher_gen_time / cypher_gen_time_count if cypher_gen_time_count else None
        ),
        'avg_answer_generation_time_seconds': (
            total_answer_gen_time / answer_gen_time_count if answer_gen_time_count else None
        ),
        'avg_cypher_generation_time': round(
            sum(r["cypher_generation_time_seconds"] for r in results
                if r["cypher_generation_time_seconds"] is not None) /
            max(sum(1 for r in results
                if r["cypher_generation_time_seconds"] is not None), 1), 4
        ),
        'avg_answer_generation_time': round(
            sum(r["answer_generation_time_seconds"] for r in results
                if r["answer_generation_time_seconds"] is not None) /
            max(sum(1 for r in results
                if r["answer_generation_time_seconds"] is not None), 1), 4
        ),
        'query_cost_usd': total_query_cost,
        'evaluation_cost_usd': total_eval_cost,
        'total_cost_usd': total_query_cost + total_eval_cost,
        'model': 'gpt-4o-mini',
        'evaluation_type': 'semantic_similarity + llm_judge + execution_accuracy',
        'ex_formula': (
            'exact Neo4j result-set match vs gold Cypher; '
            'N/A excluded from denominator'
        ),
    },
    'overall_metrics': {
        'success_rate': sum(1 for r in results if r['success']) / len(results),
        'failed_count': sum(1 for r in results if not r['success']),
        'avg_semantic_similarity': sum(r['semantic_similarity'] for r in results) / len(results),
        'avg_llm_judge_overall': sum(r['llm_judge_scores']['overall'] for r in results) / len(results),
        'avg_llm_judge_correctness': sum(r['llm_judge_scores']['correctness'] for r in results) / len(results),
        'avg_llm_judge_completeness': sum(r['llm_judge_scores']['completeness'] for r in results) / len(results),
        'avg_llm_judge_accuracy': sum(r['llm_judge_scores']['accuracy'] for r in results) / len(results),
        'avg_combined_score': sum(r['combined_score'] for r in results) / len(results),
        'execution_accuracy': execution_accuracy,
        'cypher_retry': cypher_retry_stats,
    },
    'category_stats': category_stats,
    'difficulty_stats': difficulty_stats,
    'detailed_results': results
}

with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"✅ Report saved to: {report_file}")

# ══════════════════════════════════════════════════════════════
# Print summary
# ══════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📊 SEMANTIC EVALUATION SUMMARY")
print("="*80)

print("\nPer-question EX:")
print(f"  {'ID':<5} {'CATEGORY':<20} {'DIFFICULTY':<11} {'COMBINED':>8}  {'EX':<10}")
for r in results:
    if r.get("execution_match") is True:
        ex_label = "MATCH"
    elif r.get("execution_match") is False:
        ex_label = "MISMATCH"
    else:
        ex_label = "N/A"
    print(
        f"  {r['question_id']:<5} {r['category']:<20} {r['difficulty']:<11} "
        f"{r['combined_score']:>8.2f}  {ex_label:<10}"
    )

print(f"\n📈 Overall Metrics:")
print(f"  Success Rate: {report['overall_metrics']['success_rate']*100:.1f}%")
print(f"  Failed Questions: {report['overall_metrics']['failed_count']}")
print(f"  Avg Time: {report['metadata']['avg_time_per_question']:.2f}s")
print(f"  Total Cost: ${report['metadata']['total_cost_usd']:.4f}")
total_cypher = sum(r["cypher_generation_time_seconds"] for r in results if r["cypher_generation_time_seconds"] is not None)
total_answer = sum(r["answer_generation_time_seconds"] for r in results if r["answer_generation_time_seconds"] is not None)
avg_cypher = total_cypher / max(sum(1 for r in results if r["cypher_generation_time_seconds"] is not None), 1)
avg_answer = total_answer / max(sum(1 for r in results if r["answer_generation_time_seconds"] is not None), 1)
print(f"⏱️  Avg Cypher generation time : {avg_cypher:.2f}s")
print(f"⏱️  Avg Answer generation time : {avg_answer:.2f}s")
print(f"⏱️  Total Cypher generation time : {total_cypher:.2f}s")
print(f"⏱️  Total Answer generation time : {total_answer:.2f}s")

print(f"\n🎯 Quality Scores (0-1 scale):")
print(f"  Semantic Similarity: {report['overall_metrics']['avg_semantic_similarity']:.3f}")
print(f"  LLM Judge Overall: {report['overall_metrics']['avg_llm_judge_overall']:.3f}")
print(f"  LLM Judge Correctness: {report['overall_metrics']['avg_llm_judge_correctness']:.3f}")
print(f"  LLM Judge Completeness: {report['overall_metrics']['avg_llm_judge_completeness']:.3f}")
print(f"  LLM Judge Accuracy: {report['overall_metrics']['avg_llm_judge_accuracy']:.3f}")
print(f"  Combined Score: {report['overall_metrics']['avg_combined_score']:.3f}")
ex_info = report["overall_metrics"]["execution_accuracy"]
print(
    f"  Execution Accuracy (EX): {ex_info['ex_rate'] * 100:.1f}% "
    f"({ex_info['match_count']}/{ex_info['applicable_count']} applicable; "
    f"{ex_info['na_count']} N/A excluded)"
)
retry_info = report["overall_metrics"]["cypher_retry"]
if retry_info["retry_attempts"]:
    rate = retry_info["retry_success_rate"]
    rate_txt = f"{rate * 100:.1f}%" if rate is not None else "n/a"
    print(
        f"  Cypher retries: {retry_info['retry_attempts']} attempted, "
        f"{retry_info['retry_recovered']} recovered ({rate_txt} retry-success)"
    )
else:
    print("  Cypher retries: 0")

print(f"\n📊 By Category:")
for category, stats in sorted(category_stats.items(), key=lambda x: x[1]['avg_combined'], reverse=True):
    print(f"  {category}:")
    print(f"    Success: {stats['success_rate']*100:.1f}%")
    print(f"    Semantic: {stats['avg_semantic_similarity']:.2f}")
    print(f"    LLM Judge: {stats['avg_llm_judge']:.2f}")
    print(f"    Combined: {stats['avg_combined']:.2f}")

print(f"\n📊 By Difficulty:")
for difficulty, stats in sorted(difficulty_stats.items()):
    print(f"  {difficulty}:")
    print(f"    Success: {stats['success_rate']*100:.1f}%")
    print(f"    Semantic: {stats['avg_semantic_similarity']:.2f}")
    print(f"    LLM Judge: {stats['avg_llm_judge']:.2f}")

# Show some high and low scoring examples
print(f"\n✅ Best Answers (Top 3):")
best = sorted([r for r in results if r['success']], key=lambda x: x['combined_score'], reverse=True)[:3]
for r in best:
    print(f"  {r['question_id']}: {r['question'][:60]}...")
    print(f"    Combined Score: {r['combined_score']:.2f}")

print(f"\n⚠️  Worst Answers (Bottom 3):")
worst = sorted([r for r in results if r['success']], key=lambda x: x['combined_score'])[:3]
for r in worst:
    print(f"  {r['question_id']}: {r['question'][:60]}...")
    print(f"    Combined Score: {r['combined_score']:.2f}")
    print(f"    Reason: {r['llm_judge_scores']['reasoning'][:80]}...")

print("\n" + "="*80)
print("✅ Semantic evaluation complete!")
print("="*80)
sys.exit(0)
