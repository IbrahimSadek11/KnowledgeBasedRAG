"""
CLI entry point for live Fusion RAG evaluation.

I/O + formatting only — all per-question logic lives in
backend.fusion.orchestrator.run_fusion_for_question.

Resume reads ONLY this runner's own progress files under
--output-dir (default evaluation_results/fusion/). It never reads
evaluation_results/graph_rag/, evaluation_results/tabular_rag/, or
evaluation_results/textual_rag/.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fusion.adapters import _to_jsonable, get_graph_chain
from backend.fusion.evaluation_wrapper import get_evaluator_and_judge
from backend.fusion.orchestrator import run_fusion_for_question

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation_results" / "fusion"
DATASET_PATH_100 = PROJECT_ROOT / "data" / "test_dataset.json"
DATASET_PATH_30 = PROJECT_ROOT / "data" / "specialization_test_30.json"


def _p(msg: str = "") -> None:
    print(msg, flush=True)


def _truncate(obj: Any, max_chars: int = 500) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically: .tmp → fsync → os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    data = _to_jsonable(payload)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _resolve_dataset(args: argparse.Namespace) -> tuple[Path, str, int, str]:
    """Return (path, dataset_label, expected_count, display_name). Default = full100."""
    if args.use_30:
        return DATASET_PATH_30, "specialization30", 30, "specialization 30"
    return DATASET_PATH_100, "full100", 100, "full 100"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Fusion RAG evaluation CLI (orchestrator-backed)."
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--30",
        dest="use_30",
        action="store_true",
        help="Use data/specialization_test_30.json (30-question specialization benchmark)",
    )
    dataset_group.add_argument(
        "--100",
        dest="use_100",
        action="store_true",
        help="Use data/test_dataset.json (full 100-question benchmark; default)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions")
    parser.add_argument("--question-id", type=str, default=None, help="Process only this question_id")
    parser.add_argument(
        "--start-question-id",
        type=str,
        default=None,
        help="Start from this question_id onward",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: evaluation_results/fusion/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the most recent fusion_eval_<dataset>_*.json in --output-dir. "
            # Resume only reads THIS runner's own progress files under --output-dir.
            # Never reads evaluation_results/graph_rag/, tabular_rag/, or textual_rag/.
            "Only reads this runner's own progress file."
        ),
    )
    return parser.parse_args()


def _load_questions(args: argparse.Namespace, dataset_path: Path) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    questions = list(data["test_questions"])

    # Priority: --question-id alone overrides the others
    if args.question_id:
        matched = [q for q in questions if q["question_id"] == args.question_id]
        if not matched:
            raise SystemExit(f"question_id not found: {args.question_id}")
        return matched

    if args.start_question_id:
        start_idx = None
        for i, q in enumerate(questions):
            if q["question_id"] == args.start_question_id:
                start_idx = i
                break
        if start_idx is None:
            raise SystemExit(f"start-question-id not found: {args.start_question_id}")
        questions = questions[start_idx:]

    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be >= 0")
        questions = questions[: args.limit]

    return questions


def _find_latest_progress(output_dir: Path, dataset_label: str) -> Path | None:
    # Prefer dataset-tagged progress files; for full100 also allow legacy
    # fusion_eval_<timestamp>.json from before dataset-tagged naming.
    preferred = sorted(
        (
            p
            for p in output_dir.glob(f"fusion_eval_{dataset_label}_*.json")
            if not p.name.endswith("_summary.json")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return preferred[0]

    if dataset_label != "full100":
        return None

    legacy = sorted(
        (
            p
            for p in output_dir.glob("fusion_eval_*.json")
            if not p.name.endswith("_summary.json")
            and "specialization30" not in p.name
            and "full100" not in p.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return legacy[0] if legacy else None


def _load_completed_ids(progress_path: Path) -> tuple[list[dict], set[str]]:
    with open(progress_path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise SystemExit(f"Progress file is not a JSON list: {progress_path}")
    ids = {r.get("question_id") for r in records if isinstance(r, dict)}
    ids.discard(None)
    return records, set(ids)


def _init_resources() -> None:
    _p("Initializing resources...")
    get_graph_chain()
    get_evaluator_and_judge()
    # Confirm tabular / textual modules import cleanly (they lazy-init internally).
    import backend.tabular_rag.version2.tabular_chain  # noqa: F401
    import backend.textual_rag.textual_rag_service  # noqa: F401
    _p("Resources ready.")
    _p()


def _print_pipeline_sections(record: dict) -> None:
    graph = record.get("graph_rag") or {}
    tabular = record.get("tabular_rag") or {}
    textual = record.get("textual_rag") or {}

    _p("-" * 80)
    _p("GRAPH RAG — LIVE RUN")
    _p("-" * 80)
    _p("[Graph LIVE] Calling Cypher-generation LLM and executing chain...")
    _p(f"Generated Cypher: {graph.get('generated_query')}")
    _p(f"Raw Neo4j results: {_truncate(graph.get('raw_results'))}")
    _p(f"Graph answer: {graph.get('answer')}")
    _p(f"Technical success: {graph.get('success')}")
    _p(f"Error: {graph.get('error')}")
    t = graph.get("execution_time_seconds")
    _p(f"Execution time: {float(t):.2f} seconds" if t is not None else "Execution time: n/a")
    _p()

    _p("-" * 80)
    _p("TABULAR RAG VERSION 2 — LIVE RUN")
    _p("-" * 80)
    _p("[Tabular V2 LIVE] Calling SQL-generation LLM and executing...")
    _p(f"Generated SQL: {tabular.get('generated_query')}")
    _p(f"Raw SQLite rows: {_truncate(tabular.get('raw_results'))}")
    _p(f"Tabular answer: {tabular.get('answer')}")
    _p(f"Attempts: {tabular.get('attempts')}")
    _p(f"Technical success: {tabular.get('success')}")
    _p(f"Error: {tabular.get('error')}")
    t = tabular.get("execution_time_seconds")
    _p(f"Execution time: {float(t):.2f} seconds" if t is not None else "Execution time: n/a")
    _p()

    _p("-" * 80)
    _p("TEXTUAL RAG — LIVE RUN")
    _p("-" * 80)
    _p("[Textual LIVE] Querying Chroma and generating answer...")
    docs = textual.get("retrieved_documents")
    _p(f"Retrieved documents: {_truncate(docs, 400)}")
    passages = textual.get("retrieved_passages") or []
    if isinstance(passages, list) and passages:
        first = passages[0] if isinstance(passages[0], str) else str(passages[0])
        preview = first[:200] + ("..." if len(first) > 200 else "")
        _p(f"Retrieved passages: count={len(passages)}; first_preview={preview!r}")
    else:
        _p("Retrieved passages: none")
    _p(f"Textual answer: {textual.get('answer')}")
    _p(f"Technical success: {textual.get('success')}")
    _p(f"Error: {textual.get('error')}")
    t = textual.get("execution_time_seconds")
    _p(f"Execution time: {float(t):.2f} seconds" if t is not None else "Execution time: n/a")
    _p()


def _print_evidence(record: dict) -> None:
    _p("-" * 80)
    _p("EVIDENCE SCORING — NO GROUND TRUTH")
    _p("-" * 80)
    for label, key in (
        ("GRAPH", "graph_rag"),
        ("TABULAR_V2", "tabular_rag"),
        ("TEXTUAL", "textual_rag"),
    ):
        ev = (record.get(key) or {}).get("evidence_judge") or {}
        _p(f"[{label}]")
        _p(f"  groundedness: {ev.get('groundedness')}")
        _p(f"  completeness: {ev.get('completeness')}")
        _p(f"  relevance: {ev.get('relevance')}")
        _p(f"  execution_quality: {ev.get('execution_quality')}")
        _p(f"  evidence_score: {ev.get('evidence_score')}")
        _p(f"  reasoning: {json.dumps(ev.get('reasoning'), ensure_ascii=False)}")
        _p(f"  judge_error: {ev.get('judge_error')}")
        _p(f"  truncated_evidence: {ev.get('truncated_evidence')}")
    _p()


def _print_pairwise(record: dict) -> None:
    _p("-" * 80)
    _p("PAIRWISE LLM AGREEMENT — NO GROUND TRUTH")
    _p("-" * 80)
    pairwise = record.get("pairwise_agreement") or {}
    for key in ("graph_tabular_v2", "graph_textual", "tabular_v2_textual"):
        val = pairwise.get(key)
        if val is None:
            _p(f"{key}: not computed")
        else:
            _p(
                f"{key}: agreement={val.get('agreement')} "
                f"judge_error={val.get('judge_error')} "
                f"reason={val.get('reason')}"
            )
    fusion = record.get("fusion") or {}
    _p(f"transitivity_conflict: {fusion.get('transitivity_conflict')}")
    _p(f"agreement_group: {fusion.get('agreement_group')}")
    _p(f"agreement_strength: {fusion.get('agreement_strength')}")
    _p()


def _print_decision(record: dict) -> None:
    fusion = record.get("fusion") or {}
    _p("-" * 80)
    _p("FUSION DECISION")
    _p("-" * 80)
    _p(f"Decision rule: {fusion.get('decision_rule')}")
    _p(f"Selected pipeline: {fusion.get('selected_pipeline')}")
    _p(f"Selected evidence score: {fusion.get('selected_evidence_score')}")
    _p(f"Final fused answer: {fusion.get('selected_answer')}")
    _p(f"Decision reason: {fusion.get('decision_reason')}")
    note = fusion.get("tie_break_note") or ""
    _p(f"Tie-break note: {note if note else 'n/a'}")
    _p()


def _print_final_eval(record: dict) -> None:
    fusion = record.get("fusion") or {}
    timing = record.get("timing") or {}
    judge = fusion.get("llm_judge_scores") or {}
    _p("-" * 80)
    _p("FINAL FUSION EVALUATION — GROUND TRUTH USED ONLY HERE")
    _p("-" * 80)
    _p(f"Ground truth: {record.get('ground_truth')}")
    _p(f"Semantic similarity: {fusion.get('semantic_similarity')}")
    _p(f"LLM judge overall: {judge.get('overall')}")
    _p(f"Final fusion combined score: {fusion.get('combined_score')}")
    total_t = timing.get("total_question_seconds")
    if total_t is not None:
        _p(f"Question total time: {float(total_t):.2f} seconds")
    else:
        _p("Question total time: n/a")
    _p()
    _p("Result saved successfully.")
    _p()


def _print_question_banner(item: dict, index: int, total: int) -> None:
    _p("=" * 80)
    _p(f"QUESTION {item['question_id']} / {total}")
    _p("=" * 80)
    _p()
    _p(f"Question ID: {item['question_id']}")
    _p(f"Category: {item.get('category')}")
    _p(f"Difficulty: {item.get('difficulty')}")
    _p(f"Question: {item.get('question')}")
    _p()


def _avg(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _build_summary(
    records: list[dict],
    requested: int,
    wall_seconds: float,
    *,
    dataset_label: str,
    dataset_path: str,
    question_count: int,
) -> dict:
    completed = len(records)
    failed_all = 0
    success_counts = {"graph": 0, "tabular_v2": 0, "textual": 0}
    evidence_vals: dict[str, list[float]] = {
        "graph": [],
        "tabular_v2": [],
        "textual": [],
    }
    fusion_combined: list[float] = []
    fusion_semantic: list[float] = []
    fusion_overall: list[float] = []
    selection_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    pairwise_true = 0
    pairwise_total = 0
    conflict_count = 0
    pipeline_times: dict[str, list[float]] = {
        "graph": [],
        "tabular_v2": [],
        "textual": [],
    }
    total_q_times: list[float] = []
    research_combined: dict[str, list[float]] = {
        "graph": [],
        "tabular_v2": [],
        "textual": [],
    }

    key_map = {
        "graph": "graph_rag",
        "tabular_v2": "tabular_rag",
        "textual": "textual_rag",
    }

    for rec in records:
        fusion = rec.get("fusion") or {}
        timing = rec.get("timing") or {}
        pairwise = rec.get("pairwise_agreement") or {}
        research = rec.get("evaluation_only") or {}

        all_failed = True
        for pipe, block_key in key_map.items():
            block = rec.get(block_key) or {}
            if block.get("success"):
                success_counts[pipe] += 1
                all_failed = False
            ev = block.get("evidence_judge") or {}
            score = ev.get("evidence_score")
            if score is not None:
                evidence_vals[pipe].append(float(score))
            tkey = {
                "graph": "graph_seconds",
                "tabular_v2": "tabular_seconds",
                "textual": "textual_seconds",
            }[pipe]
            tval = timing.get(tkey)
            if tval is not None:
                pipeline_times[pipe].append(float(tval))

        if all_failed:
            failed_all += 1

        sel = fusion.get("selected_pipeline")
        if sel:
            selection_counts[sel] += 1
        else:
            selection_counts["None"] += 1

        rule = fusion.get("decision_rule") or "unknown"
        decision_counts[rule] += 1
        if fusion.get("transitivity_conflict"):
            conflict_count += 1

        cs = fusion.get("combined_score")
        if cs is not None:
            fusion_combined.append(float(cs))
        ss = fusion.get("semantic_similarity")
        if ss is not None:
            fusion_semantic.append(float(ss))
        overall = (fusion.get("llm_judge_scores") or {}).get("overall")
        if overall is not None:
            fusion_overall.append(float(overall))

        for pair_key in ("graph_tabular_v2", "graph_textual", "tabular_v2_textual"):
            val = pairwise.get(pair_key)
            if val is None:
                continue
            pairwise_total += 1
            if val.get("agreement") is True:
                pairwise_true += 1

        tq = timing.get("total_question_seconds")
        if tq is not None:
            total_q_times.append(float(tq))

        for pipe in research_combined:
            c = (research.get(pipe) or {}).get("combined_score")
            if c is not None:
                research_combined[pipe].append(float(c))

    def rate(n: int) -> float | None:
        return (n / completed) if completed else None

    summary = {
        "dataset_label": dataset_label,
        "dataset_path": dataset_path,
        "question_count": question_count,
        "questions_requested": requested,
        "questions_completed": completed,
        "questions_all_pipelines_failed": failed_all,
        "per_pipeline_technical_success_rate": {
            p: rate(success_counts[p]) for p in success_counts
        },
        "average_evidence_score": {p: _avg(evidence_vals[p]) for p in evidence_vals},
        "average_fusion_combined_score": _avg(fusion_combined),
        "average_fusion_semantic_similarity": _avg(fusion_semantic),
        "average_fusion_llm_judge_overall": _avg(fusion_overall),
        "selection_counts": dict(selection_counts),
        "selection_percentages": {
            k: (v / completed if completed else None)
            for k, v in selection_counts.items()
        },
        "decision_rule_counts": dict(decision_counts),
        "pairwise_agreement_rate": (
            pairwise_true / pairwise_total if pairwise_total else None
        ),
        "pairwise_true": pairwise_true,
        "pairwise_total_computed": pairwise_total,
        "transitivity_conflict_count": conflict_count,
        "average_execution_seconds": {
            p: _avg(pipeline_times[p]) for p in pipeline_times
        },
        "average_total_question_seconds": _avg(total_q_times),
        "total_run_wall_seconds": wall_seconds,
        "comparison_avg_combined_score": {
            "fusion": _avg(fusion_combined),
            "graph": _avg(research_combined["graph"]),
            "tabular_v2": _avg(research_combined["tabular_v2"]),
            "textual": _avg(research_combined["textual"]),
        },
    }
    return summary


def _print_summary(summary: dict) -> None:
    _p()
    _p("=" * 80)
    _p("RUN SUMMARY")
    _p("=" * 80)
    _p(
        f"Questions requested/completed/all-failed: "
        f"{summary['questions_requested']} / "
        f"{summary['questions_completed']} / "
        f"{summary['questions_all_pipelines_failed']}"
    )
    _p(f"Per-pipeline success rates: {summary['per_pipeline_technical_success_rate']}")
    _p(f"Avg evidence_score: {summary['average_evidence_score']}")
    _p(f"Avg fusion combined_score: {summary['average_fusion_combined_score']}")
    _p(f"Avg fusion semantic_similarity: {summary['average_fusion_semantic_similarity']}")
    _p(f"Avg fusion llm_judge overall: {summary['average_fusion_llm_judge_overall']}")
    _p(f"Selection counts: {summary['selection_counts']}")
    _p(f"Selection percentages: {summary['selection_percentages']}")
    _p(f"Decision rule counts: {summary['decision_rule_counts']}")
    _p(
        f"Pairwise agreement rate: {summary['pairwise_agreement_rate']} "
        f"({summary['pairwise_true']}/{summary['pairwise_total_computed']})"
    )
    _p(f"Transitivity conflicts: {summary['transitivity_conflict_count']}")
    _p(f"Avg execution seconds: {summary['average_execution_seconds']}")
    _p(f"Avg total question seconds: {summary['average_total_question_seconds']}")
    _p(f"Total run wall seconds: {summary['total_run_wall_seconds']}")
    _p("Comparison (avg combined_score):")
    for k, v in (summary.get("comparison_avg_combined_score") or {}).items():
        _p(f"  {k}: {v}")
    _p("=" * 80)


def main() -> int:
    args = _parse_args()
    dataset_path, dataset_label, dataset_expected, dataset_display = _resolve_dataset(args)

    _p("=" * 80)
    _p(f"DATASET: {dataset_display}")
    _p(f"Path: {dataset_path}")
    _p(f"Questions expected: {dataset_expected}")
    _p("=" * 80)
    _p()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = _load_questions(args, dataset_path)
    completed_records: list[dict] = []
    output_path: Path

    if args.resume:
        # Resume only reads THIS runner's own progress under --output-dir.
        # Never reads evaluation_results/graph_rag/, tabular_rag/, or textual_rag/.
        latest = _find_latest_progress(output_dir, dataset_label)
        if latest is None:
            _p(
                f"Resume: no prior fusion_eval_{dataset_label}_*.json found; "
                "starting a fresh run."
            )
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"fusion_eval_{dataset_label}_{stamp}.json"
        else:
            completed_records, done_ids = _load_completed_ids(latest)
            before = len(questions)
            questions = [q for q in questions if q["question_id"] not in done_ids]
            skipped = before - len(questions)
            _p(f"Resume: loaded {latest}")
            _p(f"Resume: skipping {skipped} already-completed question_id(s).")
            output_path = latest
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"fusion_eval_{dataset_label}_{stamp}.json"

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    requested = len(questions) + (
        len(completed_records) if args.resume else 0
    )
    # For non-resume, requested == questions to process.
    # For resume, report total that will exist after finishing remaining.
    if not args.resume:
        requested = len(questions)
    else:
        requested = len(completed_records) + len(questions)

    _p(f"Output file: {output_path}")
    _p(f"Questions to process this session: {len(questions)}")
    _p()

    _init_resources()

    wall_start = time.perf_counter()
    try:
        total = len(questions)
        for i, item in enumerate(questions, start=1):
            _print_question_banner(item, i, total)

            # Orchestrator runs live pipelines + judges + selection + eval.
            # Formatted sections below reprint fields from the returned record.
            # Analysis-only fields (expected_best_pipeline, specialization_reason)
            # are intentionally NOT passed to inference.
            _p("-" * 80)
            _p("RUNNING FULL FUSION ORCHESTRATOR (live → evidence → agreement → select → eval)")
            _p("-" * 80)
            _p()

            record = run_fusion_for_question(
                question_id=item["question_id"],
                question=item["question"],
                ground_truth=item["ground_truth"],
                category=item.get("category"),
                difficulty=item.get("difficulty"),
            )

            _print_pipeline_sections(record)
            _print_evidence(record)
            _print_pairwise(record)
            _print_decision(record)
            _print_final_eval(record)

            completed_records.append(record)
            _atomic_write_json(output_path, completed_records)

            if i < total:
                _p("=" * 80)
                _p("MOVING TO NEXT QUESTION")
                _p("=" * 80)
                _p()
            else:
                _p("=" * 80)
                _p("RUN COMPLETE")
                _p("=" * 80)
                _p()
    finally:
        wall_seconds = time.perf_counter() - wall_start
        summary = _build_summary(
            completed_records,
            requested,
            wall_seconds,
            dataset_label=dataset_label,
            dataset_path=str(dataset_path),
            question_count=dataset_expected,
        )
        _print_summary(summary)
        try:
            _atomic_write_json(summary_path, summary)
            _p(f"Summary saved to: {summary_path}")
        except Exception as exc:  # noqa: BLE001
            _p(f"WARNING: failed to save summary: {exc}")
        _p(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
