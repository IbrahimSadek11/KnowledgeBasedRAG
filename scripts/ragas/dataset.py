"""Load the specialization 30-question dataset. Does not rewrite questions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "specialization_test_30.json"


def load_specialization_questions(
    dataset_path: Path | None = None,
    question_id: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    questions = payload.get("test_questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No test_questions found in {path}")

    if question_id:
        wanted = question_id.strip().upper()
        questions = [
            item
            for item in questions
            if str(item.get("question_id", "")).strip().upper() == wanted
        ]
        if not questions:
            raise ValueError(f"question_id {question_id!r} not found in {path.name}")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        questions = questions[:limit]

    return questions, path
