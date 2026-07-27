# ARCHIVE ONLY — reconstructed from bytecode (backend/__pycache__/validated_chain.cpython-314.pyc).
# Never committed to git; never wired into the live Graph RAG pipeline.
# See docs/graph_validator_status.md. Do not import from production code.

"""
Validated replacement for langchain's GraphCypherQAChain.

Same call interface as the original chain - .invoke({"query": question})
returns a dict with "result" and "intermediate_steps" keys in the same
shape - so existing scripts (run_evaluation.py, run_retrieval_eval.py)
work with zero changes.

The difference: generated Cypher is validated against Neo4j's EXPLAIN
before it's ever executed, with one retry that feeds the exact error back
to the model. See cypher_validator.py and cypher_retry.py for the pieces
this builds on.
"""

from backend.graph_rag._archive.cypher_retry import generate_validated_cypher


class _LLMHolder:
    """Tiny shim so external code (e.g. timing callbacks) can reach `.llm`
    the same way it would on the original GraphCypherQAChain's sub-chains."""

    def __init__(self, llm):
        self.llm = llm


class ValidatedGraphCypherQAChain:
    def __init__(
        self,
        graph,
        llm,
        cypher_prompt,
        qa_prompt,
        max_retries: int = 1,
        top_k: int = 50,
    ):
        self.graph = graph
        self.llm = llm
        self.cypher_prompt = cypher_prompt
        self.qa_prompt = qa_prompt
        self.max_retries = max_retries
        self.top_k = top_k
        self.cypher_generation_chain = _LLMHolder(llm)
        self.qa_chain = _LLMHolder(llm)

    def invoke(self, inputs: dict, config: dict | None = None) -> dict:
        question = inputs["query"]

        gen_result = generate_validated_cypher(
            question,
            self.graph,
            self.llm,
            self.cypher_prompt,
            max_retries=self.max_retries,
        )
        cypher = gen_result["cypher"]

        if not cypher:
            return {
                "query": question,
                "result": "Cette information n'est pas disponible.",
                "intermediate_steps": [
                    {
                        "query": (
                            f"[INVALID AFTER {gen_result['attempts']} "
                            f"ATTEMPTS: {gen_result['last_error']}]"
                        )
                    },
                    {"context": []},
                ],
            }

        try:
            context = self.graph.query(cypher)[: self.top_k]
        except Exception:
            return {
                "query": question,
                "result": "Cette information n'est pas disponible.",
                "intermediate_steps": [
                    {"query": cypher},
                    {"context": []},
                ],
            }

        context_str = str(context)
        qa_prompt_text = self.qa_prompt.format(
            question=question, context=context_str
        )
        answer = self.llm.invoke(qa_prompt_text).content.strip()

        return {
            "query": question,
            "result": answer,
            "intermediate_steps": [
                {"query": cypher},
                {"context": context},
            ],
        }
