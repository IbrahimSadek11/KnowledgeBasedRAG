"""
RAGAS evaluation using real retrieved Neo4j raw_context.

This script merges an existing retrieval evaluation report with an existing
semantic evaluation report by question_id, then evaluates RAGAS
context_precision and context_recall over real retrieved graph rows.
"""

import json
import os
import sys
import warnings
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

# Add backend to path, matching the local script pattern.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.config import OPENAI_API_KEY


REPO_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_SOURCE_FILE = REPO_ROOT / "evaluation_results" / "retrieval_eval_20260624_030926.json"
SEMANTIC_SOURCE_FILE = REPO_ROOT / "evaluation_results" / "semantic_evaluation_20260623_145348.json"
RESULTS_DIR = REPO_ROOT / "evaluation_results"


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


def build_merged_rows(retrieval_payload, semantic_payload):
    retrieval_by_id = {
        item.get("question_id"): item
        for item in retrieval_payload.get("results", [])
        if item.get("question_id")
    }
    semantic_by_id = {
        item.get("question_id"): item
        for item in semantic_payload.get("detailed_results", [])
        if item.get("question_id")
    }

    rows = []
    skipped = []
    all_question_ids = sorted(set(retrieval_by_id) & set(semantic_by_id))

    for question_id in all_question_ids:
        retrieval = retrieval_by_id[question_id]
        semantic = semantic_by_id[question_id]
        raw_context = retrieval.get("raw_context")
        answer = semantic.get("answer")
        ground_truth = semantic.get("ground_truth")

        if not raw_context:
            skipped.append({"question_id": question_id, "reason": "missing_or_empty_raw_context"})
            continue
        if answer is None or answer == "":
            skipped.append({"question_id": question_id, "reason": "missing_answer"})
            continue
        if ground_truth is None or ground_truth == "":
            skipped.append({"question_id": question_id, "reason": "missing_ground_truth"})
            continue

        rows.append(
            {
                "question_id": question_id,
                "question": semantic.get("question") or retrieval.get("question", ""),
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": raw_context_to_strings(raw_context),
                "cypher_query": semantic.get("cypher_query") or retrieval.get("cypher_query", ""),
            }
        )

    missing_in_semantic = sorted(set(retrieval_by_id) - set(semantic_by_id))
    missing_in_retrieval = sorted(set(semantic_by_id) - set(retrieval_by_id))
    skipped.extend(
        {"question_id": question_id, "reason": "missing_in_semantic_results"}
        for question_id in missing_in_semantic
    )
    skipped.extend(
        {"question_id": question_id, "reason": "missing_in_retrieval_results"}
        for question_id in missing_in_retrieval
    )

    return rows, skipped


def build_dataset(rows):
    return Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row["answer"] for row in rows],
            "contexts": [row["contexts"] for row in rows],
            "ground_truth": [row["ground_truth"] for row in rows],
        }
    )


def score_from_row(row, metric_name):
    if metric_name in row:
        value = row[metric_name]
    else:
        value = row.get(f"ragas_{metric_name}")
    return None if value is None else float(value)


def dataframe_to_results(ragas_results, source_rows):
    df = ragas_results.to_pandas()
    per_question = []

    for index, row in df.iterrows():
        source = source_rows[index]
        per_question.append(
            {
                "question_id": source["question_id"],
                "question": source["question"],
                "context_precision": score_from_row(row, "context_precision"),
                "context_recall": score_from_row(row, "context_recall"),
            }
        )

    precision_values = [
        row["context_precision"]
        for row in per_question
        if row["context_precision"] is not None
    ]
    recall_values = [
        row["context_recall"]
        for row in per_question
        if row["context_recall"] is not None
    ]
    overall = {
        "context_precision": (
            sum(precision_values) / len(precision_values) if precision_values else None
        ),
        "context_recall": (
            sum(recall_values) / len(recall_values) if recall_values else None
        ),
    }
    return overall, per_question


def combined_score(row):
    precision = row.get("context_precision") or 0.0
    recall = row.get("context_recall") or 0.0
    return (precision + recall) / 2


def format_score(value):
    return "N/A" if value is None else f"{value:.4f}"


def lowest_five(per_question):
    return sorted(per_question, key=combined_score)[:5]


def skip_reason_breakdown(skipped):
    return dict(Counter(item["reason"] for item in skipped))


def write_json_report(path, timestamp, rows, skipped, overall, per_question, error_text=""):
    report = {
        "metadata": {
            "timestamp": timestamp,
            "source_files": {
                "retrieval": str(RETRIEVAL_SOURCE_FILE),
                "semantic": str(SEMANTIC_SOURCE_FILE),
            },
            "total_merged": len(rows),
            "total_skipped": len(skipped),
            "skip_reasons": skipped,
            "skip_reason_breakdown": skip_reason_breakdown(skipped),
        },
        "overall_scores": overall,
        "per_question_results": per_question,
    }
    if error_text:
        report["error"] = error_text

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def write_markdown_report(
    path,
    script_path,
    json_path,
    success,
    error_text,
    rows,
    skipped,
    overall,
    bottom_five,
    warnings_text,
):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# RAGAS Cypher Evaluation V2 Report\n\n")
        file.write("## Files\n")
        file.write("Created:\n")
        file.write(f"- `{script_path}`\n")
        file.write(f"- `{json_path}`\n")
        file.write(f"- `{path}`\n")
        file.write("Modified: None\n\n")

        file.write("## Run Status\n")
        if success:
            file.write("- Command: `python scripts/ragas_cypher_eval_v2.py`\n")
            file.write("- Status: completed without errors\n\n")
        else:
            file.write("- Command: `python scripts/ragas_cypher_eval_v2.py`\n")
            file.write(f"- Status: failed\n- Error: `{error_text}`\n\n")

        file.write("## Totals\n")
        file.write(f"- Total merged: {len(rows)}\n")
        file.write(f"- Total skipped: {len(skipped)}\n")
        file.write(f"- Skip reason breakdown: {skip_reason_breakdown(skipped)}\n")
        file.write(
            "- Skipped question_ids: "
            + (", ".join(item["question_id"] for item in skipped) if skipped else "None")
            + "\n\n"
        )

        file.write("## Scores\n")
        file.write(f"- context_precision: {format_score(overall.get('context_precision'))}\n")
        file.write(f"- context_recall: {format_score(overall.get('context_recall'))}\n\n")

        file.write("## Lowest 5 Questions\n")
        file.write("| question_id | question | context_precision | context_recall |\n")
        file.write("| --- | --- | ---: | ---: |\n")
        for row in bottom_five:
            file.write(
                f"| {row['question_id']} | {truncate(row['question'])} | "
                f"{format_score(row.get('context_precision'))} | "
                f"{format_score(row.get('context_recall'))} |\n"
            )

        file.write("\n## Warnings\n")
        file.write("```text\n")
        file.write(warnings_text.strip() if warnings_text.strip() else "None")
        file.write("\n```\n")


def print_summary(rows, skipped, overall, bottom_five):
    print("\nRAGAS Cypher Evaluation V2 Summary")
    print("=" * 80)
    print(f"Total merged: {len(rows)}")
    print(f"Total skipped: {len(skipped)}")
    print(f"Skip reason breakdown: {skip_reason_breakdown(skipped)}")
    print(f"Overall context_precision: {format_score(overall.get('context_precision'))}")
    print(f"Overall context_recall: {format_score(overall.get('context_recall'))}")
    print("\nLowest 5 scoring questions")
    for row in bottom_five:
        print(
            f"{row['question_id']}: "
            f"precision={format_score(row.get('context_precision'))}, "
            f"recall={format_score(row.get('context_recall'))}, "
            f"question={truncate(row['question'])}"
        )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"ragas_cypher_eval_v2_{timestamp}.json"
    markdown_path = RESULTS_DIR / f"ragas_cypher_eval_v2_{timestamp}_report.md"
    script_path = Path(__file__).resolve()

    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    retrieval_payload = load_json(RETRIEVAL_SOURCE_FILE)
    semantic_payload = load_json(SEMANTIC_SOURCE_FILE)
    rows, skipped = build_merged_rows(retrieval_payload, semantic_payload)
    dataset = build_dataset(rows)

    stdout_capture = StringIO()
    stderr_capture = StringIO()
    warning_lines = []
    success = False
    error_text = ""
    overall = {"context_precision": None, "context_recall": None}
    per_question = []

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                ragas_results = evaluate(
                    dataset,
                    metrics=[context_precision, context_recall],
                )
            warning_lines = [
                f"{warning.category.__name__}: {warning.message}"
                for warning in caught_warnings
            ]
        overall, per_question = dataframe_to_results(ragas_results, rows)
        success = True
    except Exception as exc:
        error_text = str(exc)
        print(f"RAGAS evaluation failed: {error_text}")

    bottom_five = lowest_five(per_question)
    warnings_text = "\n".join(
        [
            stdout_capture.getvalue(),
            stderr_capture.getvalue(),
            *warning_lines,
        ]
    ).strip()

    write_json_report(json_path, timestamp, rows, skipped, overall, per_question, error_text)
    write_markdown_report(
        markdown_path,
        script_path,
        json_path,
        success,
        error_text,
        rows,
        skipped,
        overall,
        bottom_five,
        warnings_text,
    )

    if success:
        print_summary(rows, skipped, overall, bottom_five)
        print(f"\nSaved JSON report: {json_path}")
        print(f"Saved Claude report: {markdown_path}")
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
