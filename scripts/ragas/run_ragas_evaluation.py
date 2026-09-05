"""CLI RAGAS evaluation for Graph RAG and Fusion on specialization_test_30.

Does not change live inference, prompts, Fusion rules, or the dataset.

Graph path: backend.fusion.adapters.run_graph_live
  -> backend.graph_rag.cypher_sensor_identity.invoke_graph_chain_with_cypher_retry

Fusion path: backend.fusion.orchestrator.run_fusion_inference
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from contexts import (  # noqa: E402
    fusion_retrieved_contexts,
    graph_retrieved_contexts,
)
from dataset import load_specialization_questions  # noqa: E402
from scoring import (  # noqa: E402
    METRIC_KEYS,
    build_scorers,
    ragas_version,
    score_sample,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation_results" / "ragas"
GRAPH_INVOKE_PATH = (
    "backend.fusion.adapters.run_graph_live -> "
    "backend.graph_rag.cypher_sensor_identity.invoke_graph_chain_with_cypher_retry"
)
FUSION_INVOKE_PATH = "backend.fusion.orchestrator.run_fusion_inference"

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def _p(msg: str = "") -> None:
    print(msg, flush=True)


def _truncate(text: Any, max_chars: int = 180) -> str:
    raw = " ".join(str(text if text is not None else "").split())
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "..."


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RAGAS evaluation of Graph RAG and/or Fusion on "
            "data/specialization_test_30.json"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--graph", action="store_true", help="Evaluate Graph RAG")
    mode.add_argument("--fusion", action="store_true", help="Evaluate Fusion")
    mode.add_argument("--all", action="store_true", help="Evaluate Graph RAG then Fusion")
    parser.add_argument(
        "--question-id",
        dest="question_id",
        default=None,
        help="Evaluate a single question id (e.g. G01)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions after filtering",
    )
    return parser.parse_args()


def _print_startup(scorers: dict[str, Any], questions: list[dict], dataset_path: Path) -> None:
    _p("=" * 60)
    _p("RAGAS EVALUATION")
    _p(f"Dataset: {dataset_path.name}")
    _p(f"Questions: {len(questions)}")
    _p(f"RAGAS version: {scorers['ragas_version']}")
    _p(f"Evaluator LLM: {scorers['judge_llm']}")
    _p(f"Embeddings: {scorers['embeddings']}")
    for key, label in METRIC_LABELS.items():
        _p(f"  {label}: {scorers['class_names'][key]}")
    _p("Answer Relevancy is the collections class for docs name Response Relevancy.")
    _p("=" * 60)


def _base_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "ground_truth": item.get("ground_truth"),
        "expected_best_pipeline": item.get("expected_best_pipeline"),
        "category": item.get("category"),
        "difficulty": item.get("difficulty"),
        "specialization_reason": item.get("specialization_reason"),
    }


def _empty_ragas() -> dict[str, None]:
    return {key: None for key in METRIC_KEYS}


def _score_or_skip(
    scorers: dict[str, Any],
    *,
    inference_ok: bool,
    question: str,
    answer: str | None,
    contexts: list[str],
    reference: str,
) -> tuple[dict[str, float | None], dict[str, str], bool]:
    if not inference_ok or not answer:
        return _empty_ragas(), {}, False
    scores, errors = score_sample(
        scorers,
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference or "",
    )
    ragas_ok = any(scores.get(key) is not None for key in METRIC_KEYS)
    return scores, errors, ragas_ok


def _print_question_progress(
    index: int,
    total: int,
    record: dict[str, Any],
) -> None:
    _p("")
    _p(f"[{index}/{total}] {record.get('question_id')}")
    _p(f"Question: {_truncate(record.get('question'))}")
    if record.get("inference_success"):
        _p(f"Answer: {_truncate(record.get('answer'))}")
        _p(f"Retrieved contexts: {record.get('retrieved_context_count', 0)}")
        if record.get("selected_pipeline"):
            _p(f"Selected pipeline: {record['selected_pipeline']}")
        ragas = record.get("ragas") or {}
        for key, label in METRIC_LABELS.items():
            value = ragas.get(key)
            rendered = f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
            _p(f"{label}: {rendered}")
        errors = record.get("ragas_errors") or {}
        if errors:
            _p(f"RAGAS metric errors: {errors}")
    else:
        _p(f"FAILED inference: {record.get('error')}")


def _group_averages(
    records: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get(field) or "unknown")
        groups.setdefault(key, []).append(record)
    return {name: _metric_averages(items) for name, items in sorted(groups.items())}


def _metric_averages(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = [
            (record.get("ragas") or {}).get(key)
            for record in records
        ]
        valid = [float(v) for v in values if isinstance(v, (int, float))]
        out[key] = {
            "mean": statistics.mean(valid) if valid else None,
            "n_valid": len(valid),
            "n_missing": len(records) - len(valid),
        }
    return out


def _counts(records: list[dict[str, Any]]) -> dict[str, int]:
    n = len(records)
    successful_inference = sum(1 for r in records if r.get("inference_success"))
    ragas_scored = sum(1 for r in records if r.get("ragas_scored"))
    failures = n - ragas_scored
    return {
        "questions": n,
        "successful_inference": successful_inference,
        "ragas_scored": ragas_scored,
        "failures": failures,
    }


def _print_metric_block(averages: dict[str, Any]) -> None:
    for key, label in METRIC_LABELS.items():
        stats = averages.get(key) or {}
        mean = stats.get("mean")
        n_valid = stats.get("n_valid", 0)
        n_missing = stats.get("n_missing", 0)
        rendered = f"{mean:.4f}" if isinstance(mean, (int, float)) else "n/a"
        _p(f"{label:<20}: {rendered}  (n={n_valid} valid, {n_missing} missing)")


def _print_summary(title: str, records: list[dict[str, Any]], fusion: bool = False) -> None:
    counts = _counts(records)
    averages = _metric_averages(records)
    _p("")
    _p("=" * 60)
    _p(title)
    _p("=" * 60)
    _p(f"Questions evaluated : {counts['questions']}")
    _p(f"Successful inference: {counts['successful_inference']}")
    _p(f"RAGAS-scored        : {counts['ragas_scored']}")
    _p(f"Failures            : {counts['failures']}")
    _p(
        "Averages use valid metric values only; failures are not converted to 0."
    )
    inference_only_fail = counts["successful_inference"] - counts["ragas_scored"]
    if inference_only_fail > 0:
        _p(
            f"Note: {inference_only_fail} question(s) inferred an answer but "
            "produced no valid RAGAS metric."
        )
    _p("")
    _print_metric_block(averages)

    by_category = _group_averages(records, "category")
    if by_category:
        _p("")
        _p("By category:")
        for name, stats in by_category.items():
            _p(f"  [{name}]")
            _print_metric_block(stats)

    by_difficulty = _group_averages(records, "difficulty")
    if by_difficulty:
        _p("")
        _p("By difficulty:")
        for name, stats in by_difficulty.items():
            _p(f"  [{name}]")
            _print_metric_block(stats)

    if fusion:
        counter = Counter(
            r.get("selected_pipeline") or "none" for r in records
        )
        n = counts["questions"] or 1
        graph_n = counter.get("graph", 0)
        tabular_n = counter.get("tabular_v2", 0) + counter.get("tabular", 0)
        textual_n = counter.get("textual", 0)
        none_n = counter.get("none", 0)
        _p("")
        _p("Fusion pipeline-selection counts:")
        _p(f"Graph selected   : {graph_n} / {n}")
        _p(f"Tabular selected : {tabular_n} / {n}")
        _p(f"Textual selected : {textual_n} / {n}")
        if none_n:
            _p(f"No selection     : {none_n} / {n}")


def _print_comparison(graph_records: list[dict[str, Any]], fusion_records: list[dict[str, Any]]) -> dict[str, Any]:
    graph_avg = _metric_averages(graph_records)
    fusion_avg = _metric_averages(fusion_records)
    _p("")
    _p("=" * 60)
    _p("GRAPH RAG vs FUSION — RAGAS COMPARISON")
    _p("=" * 60)
    _p(f"{'Metric':<20} {'Graph RAG':>12} {'Fusion':>12} {'Difference':>12}")
    comparison: dict[str, Any] = {}
    for key, label in METRIC_LABELS.items():
        g = (graph_avg.get(key) or {}).get("mean")
        f = (fusion_avg.get(key) or {}).get("mean")
        diff = (f - g) if isinstance(g, (int, float)) and isinstance(f, (int, float)) else None
        comparison[key] = {"graph": g, "fusion": f, "difference_fusion_minus_graph": diff}
        g_s = f"{g:.4f}" if isinstance(g, (int, float)) else "n/a"
        f_s = f"{f:.4f}" if isinstance(f, (int, float)) else "n/a"
        d_s = f"{diff:+.4f}" if isinstance(diff, (int, float)) else "n/a"
        _p(f"{label:<20} {g_s:>12} {f_s:>12} {d_s:>12}")
    return comparison


def _evaluate_graph(
    questions: list[dict[str, Any]],
    scorers: dict[str, Any],
) -> list[dict[str, Any]]:
    from backend.fusion.adapters import get_graph_chain, run_graph_live

    _p("")
    _p("=" * 60)
    _p("RAGAS GRAPH RAG EVALUATION")
    _p(f"Graph invoke path: {GRAPH_INVOKE_PATH}")
    _p("=" * 60)
    get_graph_chain()

    records: list[dict[str, Any]] = []
    total = len(questions)
    for index, item in enumerate(questions, start=1):
        record = _base_record(item)
        record["mode"] = "graph"
        record["selected_pipeline"] = None
        record["decision_rule"] = None
        record["decision_reason"] = None
        record["selected_evidence_score"] = None
        question = item.get("question") or ""
        try:
            result = run_graph_live(question)
        except Exception as exc:  # noqa: BLE001
            result = {
                "success": False,
                "error": str(exc),
                "answer": None,
                "generated_query": None,
                "raw_results": None,
                "execution_time_seconds": None,
            }
        inference_ok = bool(result.get("success")) and bool(result.get("answer"))
        contexts = graph_retrieved_contexts(result)
        scores, errors, ragas_ok = _score_or_skip(
            scorers,
            inference_ok=inference_ok,
            question=question,
            answer=result.get("answer"),
            contexts=contexts,
            reference=item.get("ground_truth") or "",
        )
        record.update(
            {
                "answer": result.get("answer"),
                "retrieved_contexts": contexts,
                "retrieved_context_count": len(contexts),
                "generated_query": result.get("generated_query"),
                "inference_success": inference_ok,
                "success": ragas_ok,
                "error": None if inference_ok else result.get("error"),
                "inference_seconds": result.get("execution_time_seconds"),
                "ragas": scores,
                "ragas_errors": errors,
                "ragas_scored": ragas_ok,
            }
        )
        _print_question_progress(index, total, record)
        records.append(record)
    _print_summary("GRAPH RAG — RAGAS SUMMARY", records)
    return records


def _evaluate_fusion(
    questions: list[dict[str, Any]],
    scorers: dict[str, Any],
) -> list[dict[str, Any]]:
    from backend.fusion.adapters import get_graph_chain
    from backend.fusion.orchestrator import run_fusion_inference

    _p("")
    _p("=" * 60)
    _p("RAGAS FUSION EVALUATION")
    _p(f"Fusion invoke path: {FUSION_INVOKE_PATH}")
    _p(f"Graph path inside Fusion: {GRAPH_INVOKE_PATH}")
    _p("=" * 60)
    get_graph_chain()

    records: list[dict[str, Any]] = []
    total = len(questions)
    for index, item in enumerate(questions, start=1):
        record = _base_record(item)
        record["mode"] = "fusion"
        question = item.get("question") or ""
        try:
            inference = run_fusion_inference(question)
        except Exception as exc:  # noqa: BLE001
            inference = {
                "fusion": {
                    "selected_pipeline": None,
                    "selected_answer": None,
                    "decision_rule": None,
                    "decision_reason": str(exc),
                    "selected_evidence_score": None,
                },
                "error": str(exc),
            }
        fusion = inference.get("fusion") or {}
        answer = fusion.get("selected_answer")
        selected_pipeline = fusion.get("selected_pipeline")
        inference_ok = bool(selected_pipeline) and isinstance(answer, str) and bool(answer.strip())
        contexts, resolved_pipeline = fusion_retrieved_contexts(inference)
        scores, errors, ragas_ok = _score_or_skip(
            scorers,
            inference_ok=inference_ok,
            question=question,
            answer=answer,
            contexts=contexts,
            reference=item.get("ground_truth") or "",
        )
        generated_query = None
        if resolved_pipeline and resolved_pipeline != "textual":
            generated_query = (inference.get(resolved_pipeline) or {}).get("generated_query")
        record.update(
            {
                "answer": answer,
                "retrieved_contexts": contexts,
                "retrieved_context_count": len(contexts),
                "generated_query": generated_query,
                "selected_pipeline": selected_pipeline,
                "decision_rule": fusion.get("decision_rule"),
                "decision_reason": fusion.get("decision_reason"),
                "selected_evidence_score": fusion.get("selected_evidence_score"),
                "inference_success": inference_ok,
                "success": ragas_ok,
                "error": None if inference_ok else (
                    fusion.get("decision_reason") or inference.get("error") or "fusion selection failed"
                ),
                "inference_seconds": (inference.get("timing") or {}).get("total_question_seconds"),
                "ragas": scores,
                "ragas_errors": errors,
                "ragas_scored": ragas_ok,
            }
        )
        _print_question_progress(index, total, record)
        records.append(record)
    _print_summary("FUSION — RAGAS SUMMARY", records, fusion=True)
    return records


def _payload(
    *,
    mode: str,
    dataset_path: Path,
    questions: list[dict[str, Any]],
    scorers: dict[str, Any],
    records: list[dict[str, Any]],
    timestamp: str,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "mode": mode,
        "dataset": dataset_path.name,
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "question_count": len(questions),
        "ragas_version": scorers["ragas_version"],
        "evaluator": {
            "llm": scorers["judge_llm"],
            "embeddings": scorers["embeddings"],
            "metric_classes": scorers["class_names"],
        },
        "graph_invoke_path": GRAPH_INVOKE_PATH,
        "fusion_invoke_path": FUSION_INVOKE_PATH if mode in {"fusion", "all"} else None,
        "counts": _counts(records),
        "averages": _metric_averages(records),
        "averages_by_category": _group_averages(records, "category"),
        "averages_by_difficulty": _group_averages(records, "difficulty"),
        "results": records,
    }


def main() -> int:
    args = _parse_args()
    questions, dataset_path = load_specialization_questions(
        question_id=args.question_id,
        limit=args.limit,
    )
    try:
        version = ragas_version()
    except ImportError as exc:
        _p(str(exc))
        return 1

    _p(f"Working directory root: {PROJECT_ROOT}")
    _p(f"RAGAS {version} — building scorers (no pipeline calls yet)...")
    scorers = build_scorers()
    _print_startup(scorers, questions, dataset_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    graph_records: list[dict[str, Any]] | None = None
    fusion_records: list[dict[str, Any]] | None = None

    if args.graph or args.all:
        graph_records = _evaluate_graph(questions, scorers)
        graph_path = RESULTS_DIR / f"graph_ragas_specialization30_{timestamp}.json"
        _atomic_write_json(
            graph_path,
            _payload(
                mode="graph",
                dataset_path=dataset_path,
                questions=questions,
                scorers=scorers,
                records=graph_records,
                timestamp=timestamp,
            ),
        )
        _p(f"Saved: {graph_path}")

    if args.fusion or args.all:
        fusion_records = _evaluate_fusion(questions, scorers)
        fusion_path = RESULTS_DIR / f"fusion_ragas_specialization30_{timestamp}.json"
        payload = _payload(
            mode="fusion",
            dataset_path=dataset_path,
            questions=questions,
            scorers=scorers,
            records=fusion_records,
            timestamp=timestamp,
        )
        payload["selection_counts"] = {
            "graph": sum(1 for r in fusion_records if r.get("selected_pipeline") == "graph"),
            "tabular_v2": sum(
                1
                for r in fusion_records
                if r.get("selected_pipeline") in {"tabular_v2", "tabular"}
            ),
            "textual": sum(1 for r in fusion_records if r.get("selected_pipeline") == "textual"),
            "none": sum(1 for r in fusion_records if not r.get("selected_pipeline")),
        }
        _atomic_write_json(fusion_path, payload)
        _p(f"Saved: {fusion_path}")

    if args.all and graph_records is not None and fusion_records is not None:
        comparison = _print_comparison(graph_records, fusion_records)
        comparison_path = RESULTS_DIR / f"ragas_comparison_specialization30_{timestamp}.json"
        compact_pairs = []
        for graph_row, fusion_row in zip(graph_records, fusion_records):
            compact_pairs.append(
                {
                    "question_id": graph_row.get("question_id"),
                    "category": graph_row.get("category"),
                    "difficulty": graph_row.get("difficulty"),
                    "graph_ragas": graph_row.get("ragas"),
                    "fusion_ragas": fusion_row.get("ragas"),
                    "fusion_selected_pipeline": fusion_row.get("selected_pipeline"),
                }
            )
        _atomic_write_json(
            comparison_path,
            {
                "timestamp": timestamp,
                "dataset": dataset_path.name,
                "ragas_version": scorers["ragas_version"],
                "evaluator": {
                    "llm": scorers["judge_llm"],
                    "embeddings": scorers["embeddings"],
                    "metric_classes": scorers["class_names"],
                },
                "graph_counts": _counts(graph_records),
                "fusion_counts": _counts(fusion_records),
                "graph_averages": _metric_averages(graph_records),
                "fusion_averages": _metric_averages(fusion_records),
                "comparison": comparison,
                "per_question": compact_pairs,
            },
        )
        _p(f"Saved: {comparison_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
