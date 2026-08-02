"""GT-free evidence judge for a single Fusion RAG pipeline result."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI

from backend.config import OPENAI_API_KEY

JUDGE_MODEL = "gpt-4o-mini"
PASSAGE_CHAR_CAP = 6000

# Case-insensitive substrings for honest empty / unavailable answers.
_UNAVAILABLE_PHRASES = (
    "ne contient pas",
    "aucun",
    "pas disponible",
    "impossible de répondre",
)

_JUDGE_SYSTEM = (
    "You are an evidence judge for a RAG system. "
    "Score ONLY from the evidence context provided. "
    "Do not use external knowledge. "
    "Never invent facts. "
    "Respond with strict JSON only — no markdown fences, no commentary."
)

_JUDGE_SCHEMA_HINT = """\
Return ONLY valid JSON with this exact schema:
{
  "groundedness": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "relevance": <float 0.0-1.0>,
  "reasoning": {
    "groundedness": "<short explanation>",
    "completeness": "<short explanation>",
    "relevance": "<short explanation>"
  }
}

Scoring definitions:
- groundedness: is every factual claim in the answer supported by the evidence shown?
- completeness: does the answer include all important information present in the
  evidence that is relevant to the question? (do not penalize for facts not in evidence)
- relevance: does the answer directly address the question, without irrelevant tangents?
"""


def _get_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )


def _answer_indicates_unavailable(answer: Any) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    lower = answer.casefold()
    return any(phrase.casefold() in lower for phrase in _UNAVAILABLE_PHRASES)


def _is_empty_raw_results(raw_results: Any) -> bool:
    if raw_results is None:
        return True
    if isinstance(raw_results, (list, tuple, dict, str)) and len(raw_results) == 0:
        return True
    return False


def _compute_execution_quality(pipeline_result: dict) -> float:
    """Deterministic execution_quality. No LLM."""
    attempts = pipeline_result.get("attempts")

    if attempts is None or attempts == 1:
        return 1.00

    if attempts is not None and attempts > 1:
        return 0.90

    raw_results = pipeline_result.get("raw_results")
    answer = pipeline_result.get("answer")
    if _is_empty_raw_results(raw_results) and _answer_indicates_unavailable(answer):
        return 1.00

    # Default for other success=True shapes; revisit once more edge cases appear.
    return 1.00


def _truncate_passages(passages: list[Any], cap: int = PASSAGE_CHAR_CAP) -> tuple[list[str], bool]:
    """Cap total passage text at ~cap chars; truncate proportionally if needed."""
    texts = ["" if p is None else str(p) for p in passages]
    total = sum(len(t) for t in texts)
    if total <= cap or total == 0:
        return texts, False

    # Proportional shrink; keep at least a few chars per non-empty passage when possible.
    truncated: list[str] = []
    for t in texts:
        if not t:
            truncated.append(t)
            continue
        share = len(t) / total
        keep = max(1, int(cap * share))
        if len(t) > keep:
            truncated.append(t[:keep] + "…[truncated]")
        else:
            truncated.append(t)

    # If rounding left us slightly over, trim from the longest entries.
    while sum(len(t) for t in truncated) > cap and truncated:
        idx = max(range(len(truncated)), key=lambda i: len(truncated[i]))
        if len(truncated[idx]) <= 1:
            break
        truncated[idx] = truncated[idx][:-1]

    return truncated, True


def _build_evidence_context(pipeline_result: dict) -> tuple[str, bool]:
    """Build judge evidence context. Returns (context_text, truncated_evidence)."""
    pipeline = pipeline_result.get("pipeline")
    question = pipeline_result.get("question")
    answer = pipeline_result.get("answer")
    truncated = False

    if pipeline in ("graph", "tabular_v2"):
        query_label = "Cypher query" if pipeline == "graph" else "SQL query"
        context = (
            f"Question:\n{question}\n\n"
            f"{query_label}:\n{pipeline_result.get('generated_query')}\n\n"
            f"Raw results:\n{json.dumps(pipeline_result.get('raw_results'), ensure_ascii=False, default=str)}\n\n"
            f"Answer:\n{answer}\n"
        )
        return context, False

    if pipeline == "textual":
        passages = pipeline_result.get("retrieved_passages") or []
        if not isinstance(passages, list):
            passages = [passages]
        capped, truncated = _truncate_passages(list(passages))
        passage_block = "\n\n---\n\n".join(
            f"[Passage {i}]\n{text}" for i, text in enumerate(capped, start=1)
        )
        context = (
            f"Question:\n{question}\n\n"
            f"Retrieved passages (text only):\n{passage_block}\n\n"
            f"Answer:\n{answer}\n"
        )
        return context, truncated

    # Unknown pipeline type — send minimal context.
    context = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Raw results:\n{json.dumps(pipeline_result.get('raw_results'), ensure_ascii=False, default=str)}\n"
    )
    return context, False


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _parse_and_validate_scores(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse judge JSON; return (parsed, error_message)."""
    try:
        cleaned = _strip_json_fences(raw_text)
        # Tolerate leading/trailing prose by locating the outermost JSON object.
        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return None, f"no JSON object found in judge response: {raw_text[:300]!r}"
            cleaned = match.group(0)
        data = json.loads(cleaned)
    except Exception as exc:  # noqa: BLE001
        return None, f"JSON parse failed: {exc}; raw={raw_text[:300]!r}"

    if not isinstance(data, dict):
        return None, f"judge response is not a JSON object: {type(data).__name__}"

    scores: dict[str, float] = {}
    for key in ("groundedness", "completeness", "relevance"):
        if key not in data:
            return None, f"missing key '{key}' in judge response"
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            return None, f"'{key}' is not a float: {data[key]!r}"
        if not 0.0 <= value <= 1.0:
            return None, f"'{key}' out of range [0.0, 1.0]: {value}"
        scores[key] = value

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, dict):
        reasoning = {
            "groundedness": str(reasoning) if reasoning is not None else "",
            "completeness": "",
            "relevance": "",
        }

    return {
        "groundedness": scores["groundedness"],
        "completeness": scores["completeness"],
        "relevance": scores["relevance"],
        "reasoning": reasoning,
    }, None


def _call_judge(llm: ChatOpenAI, evidence_context: str) -> str:
    prompt = (
        f"{_JUDGE_SCHEMA_HINT}\n\n"
        f"Evidence context (no ground truth is provided — judge only this):\n"
        f"{evidence_context}"
    )
    response = llm.invoke(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    )
    return (response.content or "").strip()


def _call_judge_retry(llm: ChatOpenAI, evidence_context: str, bad_output: str) -> str:
    prompt = (
        f"Your previous response was malformed or invalid for this schema.\n"
        f"Malformed output was:\n{bad_output[:1500]}\n\n"
        f"Return ONLY valid JSON in the exact schema below. No markdown.\n\n"
        f"{_JUDGE_SCHEMA_HINT}\n\n"
        f"Evidence context (no ground truth — judge only this):\n"
        f"{evidence_context}"
    )
    response = llm.invoke(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    )
    return (response.content or "").strip()


def score_evidence(pipeline_result: dict) -> dict:
    """Score one live pipeline result without ground truth."""
    pipeline = pipeline_result.get("pipeline")
    truncated_evidence = False

    # STEP 1 — failed pipeline: no LLM
    if not pipeline_result.get("success"):
        return {
            "pipeline": pipeline,
            "groundedness": None,
            "completeness": None,
            "relevance": None,
            "execution_quality": 0.0,
            "evidence_score": 0.0,
            "reasoning": {
                "skipped": "pipeline success=False, no LLM judging performed"
            },
            "judge_error": None,
            "truncated_evidence": False,
        }

    execution_quality = _compute_execution_quality(pipeline_result)

    # STEP 2 — evidence context
    evidence_context, truncated_evidence = _build_evidence_context(pipeline_result)

    # STEP 3 — LLM judge
    judge_error: str | None = None
    groundedness = completeness = relevance = None
    reasoning: dict[str, Any] = {}

    try:
        llm = _get_judge_llm()
        raw = _call_judge(llm, evidence_context)
        parsed, err = _parse_and_validate_scores(raw)
        if parsed is None:
            raw2 = _call_judge_retry(llm, evidence_context, raw)
            parsed, err2 = _parse_and_validate_scores(raw2)
            if parsed is None:
                judge_error = (
                    f"judge response invalid after retry; "
                    f"first_error={err}; second_error={err2}"
                )
                reasoning = {"judge_error": judge_error}
            else:
                groundedness = parsed["groundedness"]
                completeness = parsed["completeness"]
                relevance = parsed["relevance"]
                reasoning = parsed["reasoning"]
        else:
            groundedness = parsed["groundedness"]
            completeness = parsed["completeness"]
            relevance = parsed["relevance"]
            reasoning = parsed["reasoning"]
    except Exception as exc:  # noqa: BLE001
        judge_error = f"judge LLM call failed: {exc}"
        reasoning = {"judge_error": judge_error}

    # STEP 4 — evidence_score in Python
    evidence_score = None
    if (
        groundedness is not None
        and completeness is not None
        and relevance is not None
    ):
        evidence_score = (
            0.40 * groundedness
            + 0.25 * completeness
            + 0.20 * relevance
            + 0.15 * execution_quality
        )

    return {
        "pipeline": pipeline,
        "groundedness": groundedness,
        "completeness": completeness,
        "relevance": relevance,
        "execution_quality": execution_quality,
        "evidence_score": evidence_score,
        "reasoning": reasoning,
        "judge_error": judge_error,
        "truncated_evidence": truncated_evidence,
    }
