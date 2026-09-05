"""Current RAGAS collections metrics — per-question scoring, no pipeline calls.

Official collections API (ragas 0.3+):
  from ragas.metrics.collections import (
      Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
  )
  from ragas.llms import llm_factory
  from ragas.embeddings.base import embedding_factory

Answer Relevancy is documented as "Response Relevancy"; the collections class
is AnswerRelevancy. JSON keys stay faithfulness / answer_relevancy /
context_precision / context_recall as requested.

Imports are lazy so syntax checks do not construct OpenAI clients.
"""
from __future__ import annotations

import inspect
import math
from typing import Any

from backend.config import OPENAI_API_KEY

JUDGE_LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _metric_value(result: Any) -> float | None:
    if result is None:
        return None
    if isinstance(result, (int, float)):
        value = float(result)
        return value if math.isfinite(value) else None
    if hasattr(result, "value"):
        return _metric_value(result.value)
    try:
        value = float(result)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _import_class(paths: list[tuple[str, str]]):
    last_error: Exception | None = None
    for module_name, attr in paths:
        try:
            module = __import__(module_name, fromlist=[attr])
            return getattr(module, attr)
        except (ImportError, AttributeError) as exc:
            last_error = exc
    raise ImportError(
        "Could not import a RAGAS metric class. Tried: "
        + ", ".join(f"{m}.{a}" for m, a in paths)
        + (f". Last error: {last_error}" if last_error else "")
    )


def _build_embeddings(client: Any) -> Any:
    try:
        from ragas.embeddings.base import embedding_factory

        return embedding_factory(
            provider="openai",
            model=EMBEDDING_MODEL,
            client=client,
        )
    except Exception:
        pass
    try:
        from ragas.embeddings import embedding_factory

        return embedding_factory(
            provider="openai",
            model=EMBEDDING_MODEL,
            client=client,
        )
    except Exception:
        pass
    from ragas.embeddings import OpenAIEmbeddings

    try:
        return OpenAIEmbeddings(client=client, model=EMBEDDING_MODEL)
    except TypeError:
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _construct_scorer(cls: Any, llm: Any, embeddings: Any) -> Any:
    params = inspect.signature(cls.__init__).parameters
    kwargs: dict[str, Any] = {}
    if "llm" in params:
        kwargs["llm"] = llm
    if "embeddings" in params:
        kwargs["embeddings"] = embeddings
    return cls(**kwargs)


def _call_score(scorer: Any, **kwargs: Any) -> float | None:
    """Call the scorer with exactly the kwargs provided. Do not add extras.

    RAGAS 0.4.3 collections `.score()` forwards **kwargs into `.ascore()`,
    whose signatures differ per metric. Passing a universal payload fails.
    """
    score_fn = getattr(scorer, "score", None)
    if score_fn is None:
        raise RuntimeError(f"{type(scorer).__name__} has no .score() method")
    return _metric_value(score_fn(**kwargs))


def ragas_version() -> str:
    try:
        import ragas

        return getattr(ragas, "__version__", "unknown")
    except ImportError as exc:
        raise ImportError(
            "ragas is not installed. Run: pip install -r requirements.txt"
        ) from exc


def build_scorers() -> dict[str, Any]:
    """Build the four collections scorers. Requires OPENAI_API_KEY; no keys printed."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env before running evaluation."
        )

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    # Collections metrics call llm.agenerate() and embeddings.aembed_text().
    # Both require AsyncOpenAI. Keep separate client instances.
    async_llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    async_embedding_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    llm = llm_factory(
        JUDGE_LLM_MODEL,
        provider="openai",
        client=async_llm_client,
    )
    embeddings = _build_embeddings(async_embedding_client)

    faithfulness_cls = _import_class(
        [
            ("ragas.metrics.collections", "Faithfulness"),
            ("ragas.metrics", "Faithfulness"),
        ]
    )
    answer_cls = _import_class(
        [
            ("ragas.metrics.collections", "AnswerRelevancy"),
            ("ragas.metrics.collections", "ResponseRelevancy"),
            ("ragas.metrics", "AnswerRelevancy"),
            ("ragas.metrics", "ResponseRelevancy"),
        ]
    )
    precision_cls = _import_class(
        [
            ("ragas.metrics.collections", "ContextPrecision"),
            ("ragas.metrics", "LLMContextPrecisionWithReference"),
            ("ragas.metrics", "ContextPrecision"),
        ]
    )
    recall_cls = _import_class(
        [
            ("ragas.metrics.collections", "ContextRecall"),
            ("ragas.metrics", "LLMContextRecall"),
            ("ragas.metrics", "ContextRecall"),
        ]
    )

    return {
        "faithfulness": _construct_scorer(faithfulness_cls, llm, embeddings),
        "answer_relevancy": _construct_scorer(answer_cls, llm, embeddings),
        "context_precision": _construct_scorer(precision_cls, llm, embeddings),
        "context_recall": _construct_scorer(recall_cls, llm, embeddings),
        "class_names": {
            "faithfulness": f"{faithfulness_cls.__module__}.{faithfulness_cls.__name__}",
            "answer_relevancy": f"{answer_cls.__module__}.{answer_cls.__name__}",
            "context_precision": f"{precision_cls.__module__}.{precision_cls.__name__}",
            "context_recall": f"{recall_cls.__module__}.{recall_cls.__name__}",
        },
        "judge_llm": JUDGE_LLM_MODEL,
        "embeddings": EMBEDDING_MODEL,
        "ragas_version": ragas_version(),
    }


# RAGAS 0.4.3 collections `.ascore()` / `.score()` argument maps.
# Each metric accepts only these kwargs — never a shared super-payload.
_METRIC_CALLS = {
    "faithfulness": lambda q, a, ctx, ref: {
        "user_input": q,
        "response": a,
        "retrieved_contexts": ctx,
    },
    "answer_relevancy": lambda q, a, ctx, ref: {
        "user_input": q,
        "response": a,
    },
    "context_precision": lambda q, a, ctx, ref: {
        "user_input": q,
        "retrieved_contexts": ctx,
        "reference": ref,
    },
    "context_recall": lambda q, a, ctx, ref: {
        "user_input": q,
        "retrieved_contexts": ctx,
        "reference": ref,
    },
}


def score_sample(
    scorers: dict[str, Any],
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Score one sample. Failed metrics are None — never coerced to 0."""
    scores: dict[str, float | None] = {key: None for key in METRIC_KEYS}
    errors: dict[str, str] = {}
    for key in METRIC_KEYS:
        try:
            kwargs = _METRIC_CALLS[key](
                user_input, response, retrieved_contexts, reference
            )
            scores[key] = _call_score(scorers[key], **kwargs)
            if scores[key] is None:
                errors[key] = "metric returned a non-numeric / non-finite value"
        except Exception as exc:  # noqa: BLE001 - per-metric isolation
            scores[key] = None
            errors[key] = str(exc)
    return scores, errors
