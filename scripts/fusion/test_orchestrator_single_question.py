"""Smoke-test run_fusion_for_question on Q2 + first unanswerable question."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fusion.orchestrator import run_fusion_for_question


def _abbrev_record(record: dict) -> dict:
    """Abbreviate bulky retrieval fields for readable terminal output."""
    out = json.loads(json.dumps(record))  # deep copy via JSON (already jsonable)

    for key in ("graph_rag", "tabular_rag", "textual_rag"):
        block = out.get(key) or {}
        raw = block.get("raw_results")
        if isinstance(raw, list) and len(raw) > 3:
            block["raw_results"] = raw[:3] + [f"...[{len(raw) - 3} more rows abbreviated]"]
        passages = block.get("retrieved_passages")
        if isinstance(passages, list):
            block["retrieved_passages"] = [
                (p[:180] + "...[abbrev]") if isinstance(p, str) and len(p) > 180 else p
                for p in passages[:3]
            ] + (
                [f"...[{len(passages) - 3} more passages abbreviated]"]
                if len(passages) > 3
                else []
            )
        docs = block.get("retrieved_documents")
        if isinstance(docs, list) and len(docs) > 5:
            block["retrieved_documents"] = docs[:5] + [
                f"...[{len(docs) - 5} more filenames abbreviated]"
            ]
        meta = block.get("metadata") or {}
        if "raw_intermediate_steps" in meta:
            meta = dict(meta)
            meta["raw_intermediate_steps"] = "[abbreviated in display]"
            block["metadata"] = meta
        if "attempt_log" in meta and isinstance(meta.get("attempt_log"), list):
            pass  # keep attempt_log — small
        out[key] = block
    return out


def _print_uncut_sections(record: dict) -> None:
    print("\n--- fusion (uncut) ---")
    print(json.dumps(record.get("fusion"), indent=2, ensure_ascii=False))
    print("\n--- pairwise_agreement (uncut) ---")
    print(json.dumps(record.get("pairwise_agreement"), indent=2, ensure_ascii=False))
    print("\n--- evaluation_only (uncut) ---")
    print(json.dumps(record.get("evaluation_only"), indent=2, ensure_ascii=False))
    print("\n--- timing (uncut) ---")
    print(json.dumps(record.get("timing"), indent=2, ensure_ascii=False))


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "test_dataset.json"
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    questions = data["test_questions"]

    q2 = next(q for q in questions if q["question_id"] == "Q2")
    unans = next(q for q in questions if q.get("category") == "unanswerable")
    print(
        f"Unanswerable question found: {unans['question_id']} — {unans['question']}"
    )
    print()

    for item in (q2, unans):
        print("=" * 72)
        print(
            f"RUNNING {item['question_id']} ({item['category']}/{item['difficulty']})"
        )
        print(f"Q: {item['question']}")
        print("=" * 72)

        record = run_fusion_for_question(
            question_id=item["question_id"],
            question=item["question"],
            ground_truth=item["ground_truth"],
            category=item["category"],
            difficulty=item["difficulty"],
        )

        # Prove serialization with no default=str
        serialized = json.dumps(record, indent=2, ensure_ascii=False)
        print(f"json.dumps OK ({len(serialized)} chars, no default=str)")

        display = _abbrev_record(record)
        print("\n--- full record (bulky fields abbreviated) ---")
        print(json.dumps(display, indent=2, ensure_ascii=False))
        _print_uncut_sections(record)

        fusion = record.get("fusion") or {}
        print("\n--- summary ---")
        print(f"selected_pipeline={fusion.get('selected_pipeline')}")
        print(f"decision_rule={fusion.get('decision_rule')}")
        print(f"combined_score={fusion.get('combined_score')}")
        print(
            f"total_question_seconds="
            f"{(record.get('timing') or {}).get('total_question_seconds')}"
        )
        print()


if __name__ == "__main__":
    main()
