"""
RAGAS evaluation over real Neo4j raw_context — one question at a time.

Merges retrieval + semantic evaluation JSONs, evaluates context_precision and
context_recall per question with a 90s timeout, live progress, and partial saves.
"""

import importlib
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import context_precision, context_recall

from backend.config import OPENAI_API_KEY


REPO_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_SOURCE_FILE = REPO_ROOT / "evaluation_results" / "retrieval_eval_20260624_030926.json"
SEMANTIC_SOURCE_FILE = REPO_ROOT / "evaluation_results" / "semantic_evaluation_20260623_145348.json"
RESULTS_DIR = REPO_ROOT / "evaluation_results"

EVALUATE_TIMEOUT_SECONDS = 300
PARTIAL_SAVE_INTERVAL = 10
_RAGAS_METRICS = None

SCORE_DISTRIBUTION_BUCKETS = [
    ("Perfect", "1.00", "perfect"),
    ("High", "0.75-0.99", "high"),
    ("Mid", "0.25-0.74", "mid"),
    ("Low", "0.01-0.24", "low"),
    ("Zero", "0.00", "zero"),
]


def get_ragas_metrics():
    """Return evaluate()-compatible metric singletons for collections metrics."""
    global _RAGAS_METRICS
    if _RAGAS_METRICS is None:
        # Lowercase collections names resolve to subpackages in current ragas builds.
        _ = (context_precision, context_recall)
        _RAGAS_METRICS = [
            importlib.import_module("ragas.metrics._context_precision").context_precision,
            importlib.import_module("ragas.metrics._context_recall").context_recall,
        ]
    return _RAGAS_METRICS


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def truncate(text, max_chars=100):
    text = " ".join(str(text).split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def raw_context_to_strings(raw_context):
    return [
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        for row in raw_context
    ]


def index_by_question_id(items):
    indexed = {}
    for item in items:
        question_id = item.get("question_id")
        if question_id:
            indexed[question_id] = item
    return indexed


def sorted_intersection_ids(retrieval_by_id, semantic_by_id):
    return sorted(
        set(retrieval_by_id) & set(semantic_by_id),
        key=lambda qid: int(qid.replace("Q", "")),
    )


def validate_row(retrieval, semantic):
    raw_context = retrieval.get("raw_context")
    answer = semantic.get("answer")
    ground_truth = semantic.get("ground_truth")

    if not raw_context:
        return None, "missing_or_empty_raw_context"
    if answer is None or answer == "":
        return None, "missing_answer"
    if ground_truth is None or ground_truth == "":
        return None, "missing_ground_truth"

    return {
        "question_id": retrieval.get("question_id") or semantic.get("question_id"),
        "question": semantic.get("question") or retrieval.get("question", ""),
        "answer": answer,
        "ground_truth": ground_truth,
        "contexts": raw_context_to_strings(raw_context),
        "cypher_query": semantic.get("cypher_query") or retrieval.get("cypher_query", ""),
    }, None


def build_single_dataset(row):
    return Dataset.from_dict(
        {
            "question": [row["question"]],
            "answer": [row["answer"]],
            "contexts": [row["contexts"]],
            "ground_truth": [row["ground_truth"]],
        }
    )


def score_from_result_row(result_row, metric_name):
    if metric_name in result_row:
        value = result_row[metric_name]
    else:
        value = result_row.get(f"ragas_{metric_name}")
    return None if value is None else float(value)


def is_nan(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def run_ragas_evaluate(row):
    dataset = build_single_dataset(row)
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            ragas_results = evaluate(dataset, metrics=get_ragas_metrics())

    result_row = ragas_results.to_pandas().iloc[0]
    return {
        "context_precision": score_from_result_row(result_row, "context_precision"),
        "context_recall": score_from_result_row(result_row, "context_recall"),
    }


def evaluate_with_timeout(row, timeout_seconds=EVALUATE_TIMEOUT_SECONDS):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_ragas_evaluate, row)
        return future.result(timeout=timeout_seconds)


def format_progress_index(current, total):
    return f"[{current:3d}/{total}]"


def format_score(value, decimals=3):
    if is_nan(value):
        return "nan"
    return f"{value:.{decimals}f}"


def valid_metric_values(results, metric_name):
    return [
        row[metric_name]
        for row in results
        if metric_name in row and not is_nan(row[metric_name])
    ]


def average_metric(results, metric_name):
    valid = valid_metric_values(results, metric_name)
    return sum(valid) / len(valid) if valid else 0.0


def count_nan_values(results):
    count = 0
    for row in results:
        if is_nan(row.get("context_precision")) or is_nan(row.get("context_recall")):
            count += 1
    return count


def classify_score(value):
    if is_nan(value):
        return "nan"
    if abs(value - 1.0) < 1e-9:
        return "perfect"
    if value >= 0.75:
        return "high"
    if value >= 0.25:
        return "mid"
    if value >= 0.01:
        return "low"
    return "zero"


def build_score_distribution(results, metric_name):
    bucket_counts = {key: 0 for _, _, key in SCORE_DISTRIBUTION_BUCKETS}
    nan_count = 0
    for row in results:
        value = row.get(metric_name)
        bucket = classify_score(value)
        if bucket == "nan":
            nan_count += 1
        else:
            bucket_counts[bucket] += 1
    return bucket_counts, nan_count


def distribution_bar(count, total, width=20):
    if total == 0:
        filled = 0
    else:
        filled = round(count / total * width)
    return "█" * filled + "░" * (width - filled)


def format_distribution_lines(results, metric_name, use_unicode_bars=True):
    bucket_counts, _ = build_score_distribution(results, metric_name)
    evaluated = len(results)
    lines = []
    for label, range_label, key in SCORE_DISTRIBUTION_BUCKETS:
        count = bucket_counts[key]
        pct = (count / evaluated * 100) if evaluated else 0.0
        if use_unicode_bars:
            bar = distribution_bar(count, evaluated)
            lines.append(
                f"  {label:<8} ({range_label:<9}): {count:3d} questions  "
                f"{bar}  ({pct:5.1f}%)"
            )
        else:
            filled = round(pct / 100 * 20) if evaluated else 0
            bar = "#" * filled + "." * (20 - filled)
            lines.append(
                f"| {label} ({range_label}) | {count} | {pct:.1f}% | `{bar}` |"
            )
    return lines


def skipped_breakdown(skipped_details):
    missing_context = sum(
        1 for item in skipped_details if item.get("reason") == "missing_or_empty_raw_context"
    )
    timeout = sum(1 for item in skipped_details if item.get("reason") == "timeout")
    other = len(skipped_details) - missing_context - timeout
    return missing_context, timeout, other


def skip_reason_counter(skipped_details):
    return dict(Counter(item.get("reason", "unknown") for item in skipped_details))


def sort_key_for_metric(row, metric_name):
    value = row.get(metric_name)
    if is_nan(value):
        return float("-inf")
    return value


def lowest_metric_five(detailed_results, metric_name):
    return sorted(detailed_results, key=lambda row: sort_key_for_metric(row, metric_name))[:5]


def print_header(total):
    print("=" * 80)
    print(" RAGAS V3 Evaluation — raw_context as context")
    print(
        f" Questions: {total}  |  Timeout: {EVALUATE_TIMEOUT_SECONDS}s  |  "
        f"Save every: {PARTIAL_SAVE_INTERVAL}"
    )
    print("=" * 80)


def print_evaluated_line(index, total, question_id, result, running_cp, running_cr):
    prefix = format_progress_index(index, total)
    qid_col = f"{question_id:<4}"
    cp = format_score(result["context_precision"])
    cr = format_score(result["context_recall"])
    elapsed = result["elapsed_seconds"]
    run_cp = format_score(running_cp)
    run_cr = format_score(running_cr)
    print(
        f"{prefix} {qid_col} ✓  CP={cp}  CR={cr}  ({elapsed:.1f}s)   "
        f"running avg → CP={run_cp}  CR={run_cr}"
    )


def print_skipped_line(index, total, question_id, icon, message):
    prefix = format_progress_index(index, total)
    qid_col = f"{question_id:<4}"
    print(f"{prefix} {qid_col} {icon}  {message}")


def print_partial_save_line(question_id, evaluated_count, skipped_count):
    line = (
        f"── partial save at {question_id} "
        f"({evaluated_count} evaluated, {skipped_count} skipped) "
    )
    print(line + "─" * max(1, 80 - len(line)))


def print_final_summary(
    total,
    detailed_results,
    skipped_details,
    avg_context_precision,
    avg_context_recall,
    json_path,
    markdown_path,
):
    nan_count = count_nan_values(detailed_results)
    missing_context, timeout, other = skipped_breakdown(skipped_details)
    evaluated = len(detailed_results)
    skipped = len(skipped_details)

    print("\n" + "=" * 80)
    print(" FINAL RESULTS — RAGAS V3")
    print("=" * 80)
    print(f" Evaluated : {evaluated} / {total}")
    print(
        f" Skipped   : {skipped}  "
        f"({missing_context} missing context, {timeout} timeout, {other} other)"
    )
    print(f" NaN values: {nan_count}   (excluded from averages)")
    print()
    print(" ┌─────────────────────────────────────────┐")
    print(" │  Metric              Score              │")
    print(" │─────────────────────────────────────────│")
    print(f" │  context_precision   {avg_context_precision:>6.4f}             │")
    print(f" │  context_recall      {avg_context_recall:>6.4f}             │")
    print(" └─────────────────────────────────────────┘")
    print()
    print(" Score Distribution:")
    print(" context_precision:")
    for line in format_distribution_lines(detailed_results, "context_precision"):
        print(line)
    print()
    print(" context_recall:")
    for line in format_distribution_lines(detailed_results, "context_recall"):
        print(line)
    print()
    print(" Top 5 Lowest Recall Questions:")
    for row in lowest_metric_five(detailed_results, "context_recall"):
        print(
            f"   {row['question_id']:<4}  "
            f"CP={format_score(row.get('context_precision'))}  "
            f"CR={format_score(row.get('context_recall'))}"
        )
    print()
    print(" Top 5 Lowest Precision Questions:")
    for row in lowest_metric_five(detailed_results, "context_precision"):
        print(
            f"   {row['question_id']:<4}  "
            f"CP={format_score(row.get('context_precision'))}  "
            f"CR={format_score(row.get('context_recall'))}"
        )
    print()
    print(" Output files:")
    print(f"   JSON   → {json_path}")
    print(f"   Report → {markdown_path}")
    print("=" * 80)


def write_partial_save(path, completed, skipped):
    payload = {"completed": completed, "skipped": skipped}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_final_json(
    path,
    timestamp,
    detailed_results,
    skipped_details,
    avg_context_precision,
    avg_context_recall,
):
    report = {
        "timestamp": timestamp,
        "source_files": {
            "retrieval": str(RETRIEVAL_SOURCE_FILE),
            "semantic": str(SEMANTIC_SOURCE_FILE),
        },
        "evaluated_count": len(detailed_results),
        "skipped_count": len(skipped_details),
        "avg_context_precision": avg_context_precision,
        "avg_context_recall": avg_context_recall,
        "detailed_results": detailed_results,
        "skipped_details": skipped_details,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def write_markdown_report(
    path,
    script_path,
    json_path,
    partial_path,
    timestamp,
    total,
    detailed_results,
    skipped_details,
    avg_context_precision,
    avg_context_recall,
):
    nan_count = count_nan_values(detailed_results)
    missing_context, timeout, other = skipped_breakdown(skipped_details)
    reason_counts = skip_reason_counter(skipped_details)
    bottom_recall = lowest_metric_five(detailed_results, "context_recall")
    bottom_precision = lowest_metric_five(detailed_results, "context_precision")

    with open(path, "w", encoding="utf-8") as file:
        file.write("# RAGAS Cypher Evaluation V3 Report\n\n")
        file.write("## Summary\n\n")
        file.write("| Metric | Value |\n")
        file.write("| --- | ---: |\n")
        file.write(f"| Timestamp | {timestamp} |\n")
        file.write(f"| Total questions | {total} |\n")
        file.write(f"| Evaluated | {len(detailed_results)} |\n")
        file.write(f"| Skipped | {len(skipped_details)} |\n")
        file.write(f"| NaN values (excluded from averages) | {nan_count} |\n")
        file.write(f"| avg_context_precision | {avg_context_precision:.4f} |\n")
        file.write(f"| avg_context_recall | {avg_context_recall:.4f} |\n\n")

        file.write("## Skipped Breakdown\n\n")
        file.write("| Reason | Count |\n")
        file.write("| --- | ---: |\n")
        file.write(f"| missing context | {missing_context} |\n")
        file.write(f"| timeout | {timeout} |\n")
        file.write(f"| other | {other} |\n\n")
        if reason_counts:
            file.write("### By reason code\n\n")
            file.write("| reason | count |\n")
            file.write("| --- | ---: |\n")
            for reason, count in sorted(reason_counts.items()):
                file.write(f"| {reason} | {count} |\n")
            file.write("\n")

        file.write("## Score Distribution — context_precision\n\n")
        file.write("| Bucket | Count | Percent | Bar (20 chars) |\n")
        file.write("| --- | ---: | ---: | --- |\n")
        for line in format_distribution_lines(
            detailed_results, "context_precision", use_unicode_bars=False
        ):
            file.write(line + "\n")
        file.write("\n")

        file.write("## Score Distribution — context_recall\n\n")
        file.write("| Bucket | Count | Percent | Bar (20 chars) |\n")
        file.write("| --- | ---: | ---: | --- |\n")
        for line in format_distribution_lines(
            detailed_results, "context_recall", use_unicode_bars=False
        ):
            file.write(line + "\n")
        file.write("\n")

        file.write("## Skipped Questions\n\n")
        if skipped_details:
            file.write("| question_id | reason |\n")
            file.write("| --- | --- |\n")
            for item in skipped_details:
                file.write(f"| {item['question_id']} | {item['reason']} |\n")
        else:
            file.write("None\n")
        file.write("\n")

        file.write("## Lowest 5 Recall Questions\n\n")
        file.write("| question_id | question | context_precision | context_recall |\n")
        file.write("| --- | --- | ---: | ---: |\n")
        for row in bottom_recall:
            file.write(
                f"| {row['question_id']} | {truncate(row.get('question', ''))} | "
                f"{format_score(row.get('context_precision'))} | "
                f"{format_score(row.get('context_recall'))} |\n"
            )
        file.write("\n")

        file.write("## Lowest 5 Precision Questions\n\n")
        file.write("| question_id | question | context_precision | context_recall |\n")
        file.write("| --- | --- | ---: | ---: |\n")
        for row in bottom_precision:
            file.write(
                f"| {row['question_id']} | {truncate(row.get('question', ''))} | "
                f"{format_score(row.get('context_precision'))} | "
                f"{format_score(row.get('context_recall'))} |\n"
            )
        file.write("\n")

        file.write("## Files\n\n")
        file.write(f"- Script: `{script_path}`\n")
        file.write(f"- JSON: `{json_path}`\n")
        file.write(f"- Partial: `{partial_path}`\n")
        file.write(f"- Report: `{path}`\n")


def maybe_save_partial(partial_path, completed, skipped, processed_count, question_id):
    if processed_count % PARTIAL_SAVE_INTERVAL == 0:
        write_partial_save(partial_path, completed, skipped)
        print_partial_save_line(question_id, len(completed), len(skipped))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"ragas_cypher_eval_v3_{timestamp}.json"
    markdown_path = RESULTS_DIR / f"ragas_cypher_eval_v3_{timestamp}_report.md"
    partial_path = RESULTS_DIR / f"ragas_cypher_eval_v3_{timestamp}_partial.json"
    script_path = Path(__file__).resolve()

    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    else:
        print("Warning: OPENAI_API_KEY is not set.")

    retrieval_payload = load_json(RETRIEVAL_SOURCE_FILE)
    semantic_payload = load_json(SEMANTIC_SOURCE_FILE)
    retrieval_by_id = index_by_question_id(retrieval_payload.get("results", []))
    semantic_by_id = index_by_question_id(semantic_payload.get("detailed_results", []))
    question_ids = sorted_intersection_ids(retrieval_by_id, semantic_by_id)
    total = len(question_ids)

    completed = []
    skipped = []
    detailed_results = []
    skipped_details = []

    print_header(total)

    for index, question_id in enumerate(question_ids, start=1):
        retrieval = retrieval_by_id[question_id]
        semantic = semantic_by_id[question_id]
        row, skip_reason = validate_row(retrieval, semantic)

        if skip_reason:
            skip_entry = {"question_id": question_id, "reason": skip_reason}
            skipped.append(skip_entry)
            skipped_details.append(skip_entry)
            print_skipped_line(
                index, total, question_id, "✗", f"SKIPPED ({skip_reason})"
            )
            maybe_save_partial(partial_path, completed, skipped, index, question_id)
            continue

        started = time.time()

        try:
            scores = evaluate_with_timeout(row)
            elapsed = time.time() - started
            result = {
                "question_id": question_id,
                "question": row["question"],
                "context_precision": scores["context_precision"],
                "context_recall": scores["context_recall"],
                "elapsed_seconds": round(elapsed, 1),
            }
            completed.append(result)
            detailed_results.append(result)
            running_cp = average_metric(detailed_results, "context_precision")
            running_cr = average_metric(detailed_results, "context_recall")
            print_evaluated_line(
                index, total, question_id, result, running_cp, running_cr
            )
        except FuturesTimeoutError:
            skip_entry = {
                "question_id": question_id,
                "reason": "timeout",
                "detail": f">{EVALUATE_TIMEOUT_SECONDS}s",
            }
            skipped.append(skip_entry)
            skipped_details.append(skip_entry)
            print_skipped_line(
                index,
                total,
                question_id,
                "⏱",
                f"TIMEOUT (>{EVALUATE_TIMEOUT_SECONDS}s) — skipped",
            )
        except Exception as exc:
            skip_entry = {
                "question_id": question_id,
                "reason": "exception",
                "detail": str(exc),
            }
            skipped.append(skip_entry)
            skipped_details.append(skip_entry)
            print_skipped_line(
                index, total, question_id, "✗", f"ERROR — skipped ({exc})"
            )

        maybe_save_partial(partial_path, completed, skipped, index, question_id)

    avg_context_precision = average_metric(detailed_results, "context_precision")
    avg_context_recall = average_metric(detailed_results, "context_recall")

    write_partial_save(partial_path, completed, skipped)
    write_final_json(
        json_path,
        timestamp,
        detailed_results,
        skipped_details,
        avg_context_precision,
        avg_context_recall,
    )
    write_markdown_report(
        markdown_path,
        script_path,
        json_path,
        partial_path,
        timestamp,
        total,
        detailed_results,
        skipped_details,
        avg_context_precision,
        avg_context_recall,
    )

    print_final_summary(
        total,
        detailed_results,
        skipped_details,
        avg_context_precision,
        avg_context_recall,
        json_path,
        markdown_path,
    )


if __name__ == "__main__":
    main()
