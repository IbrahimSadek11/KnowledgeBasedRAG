"""
Run retrieval-only evaluation for the GraphRAG pipeline.

This script evaluates whether generated Cypher and retrieved Neo4j context are
relevant and sufficient for each test question. It does not evaluate final
answer wording.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path

# Add repository root to path so the script works from any current directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from backend.config import OPENAI_API_KEY
from backend.llm_service import init_graph_chain
from backend.timing_callback import TimingCallbackHandler
from langchain_openai import ChatOpenAI


TEST_DATASET_PATH = REPO_ROOT / "data" / "test_dataset.json"
RESULTS_DIR = REPO_ROOT / "evaluation_results"
QUESTION_TIMEOUT_SECONDS = 60


def attach_timing_callback(graph_chain, callback):
    """Attach callback to both LLM calls inside GraphCypherQAChain."""
    graph_chain.cypher_generation_chain.llm.callbacks = [callback]

    qa_chain = getattr(graph_chain, "qa_chain", None)
    qa_llm = getattr(qa_chain, "llm", None)
    if qa_llm is not None:
        qa_llm.callbacks = [callback]


def extract_intermediate_steps(result):
    """Best-effort extraction of generated Cypher and raw context."""
    cypher_query = ""
    raw_context = []
    steps = result.get("intermediate_steps") or []

    if len(steps) > 0 and isinstance(steps[0], dict):
        cypher_query = steps[0].get("query", "") or ""

    if len(steps) > 1 and isinstance(steps[1], dict):
        raw_context = steps[1].get("context", [])
    elif len(steps) > 0 and isinstance(steps[0], dict):
        raw_context = steps[0].get("context", [])

    if raw_context is None:
        raw_context = []

    return cypher_query, raw_context


def context_length(raw_context):
    """Return a stable row count for list/dict/string contexts."""
    if raw_context is None:
        return 0
    if isinstance(raw_context, list):
        return len(raw_context)
    if isinstance(raw_context, dict):
        return 1 if raw_context else 0
    return 1 if raw_context else 0


def parse_judge_response(response_text):
    """Parse retrieval judge JSON, tolerating markdown fences if returned."""
    result_text = response_text.strip()
    if "```json" in result_text:
        result_text = result_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```", 1)[1].split("```", 1)[0].strip()

    parsed = json.loads(result_text)
    relevance = float(parsed.get("relevance", 0.0))
    sufficiency = float(parsed.get("sufficiency", 0.0))
    retrieval_score = float(parsed.get("retrieval_score", (relevance + sufficiency) / 2))
    return {
        "relevance": relevance,
        "sufficiency": sufficiency,
        "retrieval_score": retrieval_score,
        "reasoning": parsed.get("reasoning", ""),
    }


def judge_retrieval(judge_llm, question, raw_context):
    """Use GPT-4o-mini to judge retrieval relevance and sufficiency."""
    if raw_context is None or context_length(raw_context) == 0:
        return {
            "relevance": 0.0,
            "sufficiency": 0.0,
            "retrieval_score": 0.0,
            "reasoning": "Retrieved context is empty.",
        }

    raw_context_text = json.dumps(raw_context, ensure_ascii=False, indent=2, default=str)
    prompt = f"""You are evaluating the RETRIEVAL step of a Graph RAG pipeline.

Question: {question}

Retrieved context from Neo4j (raw graph data):
{raw_context_text}

Evaluate ONLY the retrieved context — ignore how a final answer might be phrased.

Score on these two criteria (0.0 to 1.0):

1. RELEVANCE: Does the retrieved context contain data related to the question?
   - 1.0 = Context is directly relevant to the question
   - 0.5 = Context is partially relevant
   - 0.0 = Context is completely unrelated or empty

2. SUFFICIENCY: Does the retrieved context contain enough data to answer the question?
   - 1.0 = Context fully contains what is needed to answer
   - 0.5 = Context partially covers what is needed
   - 0.0 = Context is missing key information needed to answer

IMPORTANT:
- If raw_context is empty or None, both scores = 0.0
- Judge the RAW DATA only, not any generated answer
- A question asking for a count is sufficient if the context contains the countable items
- A question asking for a comparison is sufficient if context contains both sides

Respond ONLY in JSON (no markdown):
{{
  "relevance": 0.0-1.0,
  "sufficiency": 0.0-1.0,
  "retrieval_score": average of relevance and sufficiency,
  "reasoning": "short explanation"
}}"""

    try:
        response = judge_llm.invoke(prompt)
        return parse_judge_response(response.content)
    except Exception as exc:
        return {
            "relevance": 0.0,
            "sufficiency": 0.0,
            "retrieval_score": 0.0,
            "reasoning": f"Retrieval judge failed: {exc}",
        }


def run_chain_with_timeout(chain, question, callback):
    """Invoke GraphCypherQAChain with a per-question timeout."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        chain.invoke,
        {"query": question},
        config={"callbacks": [callback]},
    )
    timed_out = False
    try:
        return future.result(timeout=QUESTION_TIMEOUT_SECONDS), False, None
    except FuturesTimeoutError:
        timed_out = True
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return None, True, None
    except Exception as exc:
        return None, False, exc
    finally:
        if not timed_out:
            executor.shutdown(wait=True)


def average(values):
    return sum(values) / len(values) if values else 0.0


def build_group_stats(results, group_key):
    stats = {}
    for group_value in sorted({result[group_key] for result in results}):
        group_results = [result for result in results if result[group_key] == group_value]
        stats[group_value] = {
            "count": len(group_results),
            "retrieval_score": average([r["retrieval_score"] for r in group_results]),
            "relevance": average([r["relevance"] for r in group_results]),
            "sufficiency": average([r["sufficiency"] for r in group_results]),
            "cypher_success_rate": average([1.0 if r["cypher_success"] else 0.0 for r in group_results]),
            "context_non_empty_rate": average([0.0 if r["context_empty"] else 1.0 for r in group_results]),
        }
    return stats


def print_group_table(title, stats):
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Group':35} {'Count':>5} {'Retrieval':>10} {'Rel':>7} {'Suff':>7}")
    for group, values in stats.items():
        print(
            f"{group:35} {values['count']:>5} "
            f"{values['retrieval_score']:>10.3f} "
            f"{values['relevance']:>7.3f} "
            f"{values['sufficiency']:>7.3f}"
        )


def evaluate_question(index, total, q_data, chain, judge_llm):
    question_id = q_data["question_id"]
    question = q_data["question"]
    category = q_data["category"]
    difficulty = q_data["difficulty"]
    callback = TimingCallbackHandler()
    attach_timing_callback(chain, callback)

    print(f"\n[{index}/{total}] {question_id} ({difficulty}) - {category}")
    print(f"Q: {question[:80]}...")

    result, timed_out, chain_error = run_chain_with_timeout(chain, question, callback)
    callback.reset()

    if timed_out:
        print(f"⏰ {question_id} timed out after {QUESTION_TIMEOUT_SECONDS}s")
        return {
            "question_id": question_id,
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "cypher_query": "",
            "cypher_success": False,
            "context_empty": True,
            "raw_context": [],
            "relevance": 0.0,
            "sufficiency": 0.0,
            "retrieval_score": 0.0,
            "reasoning": "Timed out",
        }

    if chain_error is not None:
        reasoning = f"Cypher execution error: {str(chain_error)}"
        print(f"❌ {question_id} {reasoning}")
        return {
            "question_id": question_id,
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "cypher_query": "",
            "cypher_success": False,
            "context_empty": True,
            "raw_context": [],
            "relevance": 0.0,
            "sufficiency": 0.0,
            "retrieval_score": 0.0,
            "reasoning": reasoning,
        }

    try:
        cypher_query, raw_context = extract_intermediate_steps(result or {})
        cypher_success = bool(cypher_query)
    except Exception as exc:
        cypher_query = ""
        raw_context = []
        cypher_success = False
        judge = {
            "relevance": 0.0,
            "sufficiency": 0.0,
            "retrieval_score": 0.0,
            "reasoning": f"Malformed intermediate steps: {exc}",
        }
    else:
        judge = judge_retrieval(judge_llm, question, raw_context)

    context_empty = context_length(raw_context) == 0
    print(f"Cypher: {cypher_query[:100]}...")
    print(f"Context rows: {context_length(raw_context)}")
    print(
        f"Relevance: {judge['relevance']:.2f} | "
        f"Sufficiency: {judge['sufficiency']:.2f} | "
        f"Retrieval: {judge['retrieval_score']:.2f}"
    )

    return {
        "question_id": question_id,
        "question": question,
        "category": category,
        "difficulty": difficulty,
        "cypher_query": cypher_query,
        "cypher_success": cypher_success,
        "context_empty": context_empty,
        "raw_context": raw_context,
        "relevance": judge["relevance"],
        "sufficiency": judge["sufficiency"],
        "retrieval_score": judge["retrieval_score"],
        "reasoning": judge["reasoning"],
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Expected retrieval dataset not found: {TEST_DATASET_PATH}"
        )

    print("=" * 80)
    print("RETRIEVAL-ONLY GRAPHRAG EVALUATION")
    print("=" * 80)
    print(f"Dataset: {TEST_DATASET_PATH}")

    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as file:
        test_data = json.load(file)
    questions_data = test_data["test_questions"]
    print(f"Loaded {len(questions_data)} test questions")

    print("\nInitializing GraphRAG chain and retrieval judge...")
    chain, graph = init_graph_chain()
    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    print("Initialization complete")

    results = []
    for index, q_data in enumerate(questions_data, 1):
        results.append(evaluate_question(index, len(questions_data), q_data, chain, judge_llm))

    summary = {
        "total_questions": len(results),
        "overall_retrieval_score": average([r["retrieval_score"] for r in results]),
        "cypher_success_rate": average([1.0 if r["cypher_success"] else 0.0 for r in results]),
        "context_non_empty_rate": average([0.0 if r["context_empty"] else 1.0 for r in results]),
        "avg_relevance": average([r["relevance"] for r in results]),
        "avg_sufficiency": average([r["sufficiency"] for r in results]),
        "by_category": build_group_stats(results, "category"),
        "by_difficulty": build_group_stats(results, "difficulty"),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"retrieval_eval_{timestamp}.json"
    report = {
        "metadata": {
            "timestamp": timestamp,
            "dataset_path": str(TEST_DATASET_PATH),
            "model": "gpt-4o-mini",
            "temperature": 0,
            "timeout_seconds": QUESTION_TIMEOUT_SECONDS,
        },
        "summary": summary,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Overall retrieval score: {summary['overall_retrieval_score']:.3f}")
    print(f"Cypher success rate: {summary['cypher_success_rate'] * 100:.1f}%")
    print(f"Context non-empty rate: {summary['context_non_empty_rate'] * 100:.1f}%")
    print(f"Avg relevance: {summary['avg_relevance']:.3f}")
    print(f"Avg sufficiency: {summary['avg_sufficiency']:.3f}")

    print_group_table("By category", summary["by_category"])
    print_group_table("By difficulty", summary["by_difficulty"])

    best = sorted(results, key=lambda r: r["retrieval_score"], reverse=True)[:5]
    worst = sorted(results, key=lambda r: r["retrieval_score"])[:5]

    print("\nTop 5 best retrieved questions")
    for result in best:
        print(f"{result['question_id']}: {result['retrieval_score']:.3f} - {result['question'][:80]}...")

    print("\nBottom 5 worst retrieved questions")
    for result in worst:
        print(f"{result['question_id']}: {result['retrieval_score']:.3f} - {result['question'][:80]}...")

    print(f"\nSaved retrieval report to: {output_path}")


if __name__ == "__main__":
    main()
