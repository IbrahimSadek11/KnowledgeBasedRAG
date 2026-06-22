"""
⏱️ Timing Callback Handler
==========================

A lightweight LangChain callback that times each LLM invocation inside a
GraphCypherQAChain.invoke() call.

GraphCypherQAChain makes exactly two LLM calls per invoke(), in order:
  1. Cypher generation
  2. QA / final-answer generation

This handler records the wall-clock duration of each LLM call in order, so the
caller can read them off after each invoke. Embedding calls go to the
on_embedding_* hooks (not on_llm_*), so they do not pollute the list.
"""

import time

from langchain_core.callbacks import BaseCallbackHandler


class TimingCallbackHandler(BaseCallbackHandler):
    """Records the duration of each LLM call within a chain invocation, in order."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear recorded timings (call before each chain.invoke)."""
        self.durations = []
        self._start = None

    def on_llm_start(self, *args, **kwargs):
        self._start = time.perf_counter()

    def on_llm_end(self, *args, **kwargs):
        if self._start is not None:
            self.durations.append(time.perf_counter() - self._start)
            self._start = None

    @property
    def cypher_time(self):
        """Duration of the Cypher-generation LLM call (first call), or None."""
        return self.durations[0] if len(self.durations) > 0 else None

    @property
    def answer_time(self):
        """Duration of the QA / final-answer LLM call (second call), or None."""
        return self.durations[1] if len(self.durations) > 1 else None
