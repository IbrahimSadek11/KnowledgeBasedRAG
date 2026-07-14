"""
Standalone RAGAS evaluation for generated Cypher as retrieval context.

Reads an existing semantic evaluation JSON and evaluates only the Cypher/query
context quality with RAGAS context_precision and context_recall.
"""

import json
import os
import sys
import warnings
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
SOURCE_FILE = REPO_ROOT / "evaluation_results" / "semantic_evaluation_20260623_145348.json"
RESULTS_DIR = REPO_ROOT / "evaluation_results"


def truncate(text, max_chars=100):
    text = " ".join(str(text).split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def load_detailed_results():
    with open(SOURCE_FILE, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("detailed_results", [])


def build_ragas_dataset(detailed_results):
    rows = []
    skipped_ids = []

    for entry in detailed_results:
        cypher_query = (entry.get("cypher_query") or "").strip()
        question_id = entry.get("question_id")

        if not cypher_query:
            skipped_ids.append(question_id)
            continue

        rows.append(
            {
                "question_id": question_id,
                "question": entry.get("question", ""),
                "answer": entry.get("answer", ""),
                "contexts": [cypher_query],
                "ground_truth": entry.get("ground_truth", ""),
            }
        )

    data = {
        "question": [row["question"] for row in rows],
        "answer": [row["answer"] for row in rows],
        "contexts": [row["contexts"] for row in rows],
        "ground_truth": [row["ground_truth"] for row in rows],
    }
    return rows, skipped_ids, Dataset.from_dict(data)


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


def format_score(value):
    return "N/A" if value is None else f"{value:.4f}"


def combined_score(row):
    precision = row.get("context_precision") or 0.0
    recall = row.get("context_recall") or 0.0
    return (precision + recall) / 2


def get_lowest_five(per_question):
    return sorted(per_question, key=combined_score)[:5]


def write_json_report(path, timestamp, total_evaluated, skipped_ids, overall, per_question):
    report = {
        "metadata": {
            "timestamp": timestamp,
            "source_file": str(SOURCE_FILE),
            "total_questions_evaluated": total_evaluated,
            "questions_skipped": {
                "count": len(skipped_ids),
                "question_ids": skipped_ids,
            },
        },
        "overall_scores": overall,
        "per_question_results": per_question,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def write_markdown_report(
    path,
    script_path,
    json_path,
    success,
    error_text,
    total_evaluated,
    skipped_ids,
    overall,
    lowest_five,
    captured_warnings,
):
    created_files = [str(script_path), str(json_path), str(path)]
    with open(path, "w", encoding="utf-8") as file:
        file.write("# RAGAS Cypher Evaluation Report\n\n")
        file.write("## Files\n")
        file.write("Created:\n")
        for created_file in created_files:
            file.write(f"- `{created_file}`\n")
        file.write("Modified: None\n\n")

        file.write("## Run Status\n")
        if success:
            file.write("- Command: `python scripts/ragas_cypher_eval.py`\n")
            file.write("- Status: completed without errors\n\n")
        else:
            file.write("- Command: `python scripts/ragas_cypher_eval.py`\n")
            file.write(f"- Status: failed\n- Error: `{error_text}`\n\n")

        file.write("## Totals\n")
        file.write(f"- Total questions evaluated: {total_evaluated}\n")
        file.write(f"- Total skipped: {len(skipped_ids)}\n")
        file.write(f"- Skipped question_ids: {', '.join(skipped_ids) if skipped_ids else 'None'}\n\n")

        file.write("## Scores\n")
        file.write(f"- context_precision: {format_score(overall.get('context_precision'))}\n")
        file.write(f"- context_recall: {format_score(overall.get('context_recall'))}\n\n")

        file.write("## Lowest 5 Questions\n")
        file.write("| question_id | question | context_precision | context_recall |\n")
        file.write("| --- | --- | ---: | ---: |\n")
        for row in lowest_five:
            file.write(
                f"| {row['question_id']} | {truncate(row['question'])} | "
                f"{format_score(row.get('context_precision'))} | "
                f"{format_score(row.get('context_recall'))} |\n"
            )

        file.write("\n## Warnings\n")
        file.write("```text\n")
        file.write(captured_warnings.strip() if captured_warnings.strip() else "None")
        file.write("\n```\n")


def print_summary(total_evaluated, skipped_ids, overall, lowest_five):
    print("\nRAGAS Cypher Evaluation Summary")
    print("=" * 80)
    print(f"Total questions evaluated: {total_evaluated}")
    print(f"Total questions skipped: {len(skipped_ids)}")
    print(f"Skipped question_ids: {', '.join(skipped_ids) if skipped_ids else 'None'}")
    print(f"Overall context_precision: {format_score(overall.get('context_precision'))}")
    print(f"Overall context_recall: {format_score(overall.get('context_recall'))}")
    print("\nTop 5 lowest-scoring questions")
    for row in lowest_five:
        print(
            f"{row['question_id']}: "
            f"precision={format_score(row.get('context_precision'))}, "
            f"recall={format_score(row.get('context_recall'))}, "
            f"question={truncate(row['question'])}"
        )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"ragas_cypher_eval_{timestamp}.json"
    markdown_path = RESULTS_DIR / f"ragas_cypher_eval_{timestamp}_report.md"
    script_path = Path(__file__).resolve()

    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    detailed_results = load_detailed_results()
    source_rows, skipped_ids, dataset = build_ragas_dataset(detailed_results)

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
        overall, per_question = dataframe_to_results(ragas_results, source_rows)
        success = True
    except Exception as exc:
        error_text = str(exc)
        print(f"RAGAS evaluation failed: {error_text}")

    lowest_five = get_lowest_five(per_question)
    captured_warnings = "\n".join(
        [
            stdout_capture.getvalue(),
            stderr_capture.getvalue(),
            *warning_lines,
        ]
    ).strip()

    write_json_report(
        json_path,
        timestamp,
        len(source_rows),
        skipped_ids,
        overall,
        per_question,
    )
    write_markdown_report(
        markdown_path,
        script_path,
        json_path,
        success,
        error_text,
        len(source_rows),
        skipped_ids,
        overall,
        lowest_five,
        captured_warnings,
    )

    if success:
        print_summary(len(source_rows), skipped_ids, overall, lowest_five)
        print(f"\nSaved JSON report: {json_path}")
        print(f"Saved Claude report: {markdown_path}")
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
