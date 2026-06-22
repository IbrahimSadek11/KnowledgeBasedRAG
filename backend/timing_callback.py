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
        self.cypher_generation_time = None
        self.answer_generation_time = None
        self._call_count = 0
        self._starts = {}

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        key = str(run_id) if run_id is not None else "__default__"
        self._starts[key] = time.perf_counter()

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        key = str(run_id) if run_id is not None else "__default__"
        start = self._starts.pop(key, None)
        if start is None:
            return
        duration = time.perf_counter() - start
        self.durations.append(duration)
        if self._call_count == 0:
            self.cypher_generation_time = duration
        elif self._call_count == 1:
            self.answer_generation_time = duration
        self._call_count += 1

    @property
    def cypher_time(self):
        """Duration of the Cypher-generation LLM call (first call), or None."""
        return self.cypher_generation_time

    @property
    def answer_time(self):
        """Duration of the QA / final-answer LLM call (second call), or None."""
        return self.answer_generation_time
