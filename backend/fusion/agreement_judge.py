"""GT-free pairwise agreement judge for two pipeline answer texts."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI

from backend.config import OPENAI_API_KEY

JUDGE_MODEL = "gpt-4o-mini"

_PIPELINE_LABELS = {
    "graph": "Graph RAG answer",
    "tabular_v2": "Tabular RAG (v2) answer",
    "textual": "Textual RAG answer",
}

_JUDGE_SYSTEM = (
    "You are an agreement judge comparing two RAG answers. "
    "Judge ONLY the two answer texts relative to the question. "
    "Do not use external knowledge or invent facts. "
    "Respond with strict JSON only — no markdown fences, no commentary."
)

_TASK = """\
Determine whether both answers assert the same underlying factual \
conclusion to the question — not just similar wording or topic.

Agreement means a genuine shared factual conclusion. It does NOT \
mean: similar wording, same general topic, or non-contradictory \
statements when the core facts differ.

Specifically: if one answer names a SUBSET or SUPERSET of entities/ \
facts compared to the other (e.g. one says 'Dakota participated' \
and the other says 'Dakota, Orion, and Vega participated'), this \
is usually NOT full agreement, because the factual sets differ — \
unless the question specifically asks about only one entity and \
the extra entities in the other answer are incidental/irrelevant \
context rather than part of the direct answer to what was asked.

Two answers both stating that information is unavailable/unknown \
ARE in agreement if they refer to the same missing information.

Return ONLY valid JSON with this exact schema:
{
  "agreement": true,
  "reason": "..."
}
The "agreement" field must be a JSON boolean (true or false), not a string.
"""


def _get_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )


def _label_for(pipeline: str) -> str:
    return _PIPELINE_LABELS.get(pipeline, f"{pipeline} answer")


def _build_user_prompt(
    question: str,
    answer_a: str,
    answer_b: str,
    pipeline_a: str,
    pipeline_b: str,
) -> str:
    return (
        f"{_TASK}\n\n"
        f"Question:\n{question}\n\n"
        f"{_label_for(pipeline_a)}:\n{answer_a}\n\n"
        f"{_label_for(pipeline_b)}:\n{answer_b}\n"
    )


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _parse_agreement(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        cleaned = _strip_json_fences(raw_text)
        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return None, f"no JSON object found: {raw_text[:300]!r}"
            cleaned = match.group(0)
        data = json.loads(cleaned)
    except Exception as exc:  # noqa: BLE001
        return None, f"JSON parse failed: {exc}; raw={raw_text[:300]!r}"

    if not isinstance(data, dict):
        return None, f"response is not a JSON object: {type(data).__name__}"

    if "agreement" not in data:
        return None, "missing key 'agreement'"

    agreement = data["agreement"]
    # Strict boolean only — reject string "true"/"false" and 0/1.
    if not isinstance(agreement, bool):
        return None, f"'agreement' must be a JSON boolean, got {agreement!r}"

    reason = data.get("reason")
    if reason is None:
        reason = ""
    else:
        reason = str(reason)

    return {"agreement": agreement, "reason": reason}, None


def _invoke(llm: ChatOpenAI, prompt: str) -> str:
    response = llm.invoke(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    )
    return (response.content or "").strip()


def judge_pairwise_agreement(
    question: str,
    answer_a: str,
    answer_b: str,
    pipeline_a: str,
    pipeline_b: str,
) -> dict:
    """Judge whether two answers assert the same factual conclusion (no GT)."""
    prompt = _build_user_prompt(question, answer_a, answer_b, pipeline_a, pipeline_b)

    try:
        llm = _get_judge_llm()
        raw = _invoke(llm, prompt)
        parsed, err = _parse_agreement(raw)
        if parsed is None:
            correction = (
                f"Your previous response was malformed.\n"
                f"Malformed output was:\n{raw[:1500]}\n\n"
                f"Return ONLY valid JSON with a strict boolean "
                f"'agreement' and a string 'reason'.\n\n"
                f"{prompt}"
            )
            raw2 = _invoke(llm, correction)
            parsed, err2 = _parse_agreement(raw2)
            if parsed is None:
                detail = (
                    f"judge_error: invalid after retry; "
                    f"first_error={err}; second_error={err2}"
                )
                return {
                    "pipeline_a": pipeline_a,
                    "pipeline_b": pipeline_b,
                    "agreement": None,
                    "reason": detail,
                    "judge_error": True,
                }
        return {
            "pipeline_a": pipeline_a,
            "pipeline_b": pipeline_b,
            "agreement": parsed["agreement"],
            "reason": parsed["reason"],
            "judge_error": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "pipeline_a": pipeline_a,
            "pipeline_b": pipeline_b,
            "agreement": None,
            "reason": f"judge_error: {exc}",
            "judge_error": True,
        }
