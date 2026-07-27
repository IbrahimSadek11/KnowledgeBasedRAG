# ARCHIVE ONLY — reconstructed from bytecode (backend/__pycache__/cypher_retry.cpython-314.pyc).
# Never committed to git; never wired into the live Graph RAG pipeline.
# See docs/graph_rag/graph_validator_status.md. Do not import from production code.

"""
Cypher generation with a validated retry loop.

Standalone - not wired into init_graph_chain() yet. Test this on its own
first against questions that previously produced invalid Cypher (e.g. the
HAVING query from Q44) before we integrate it into the real pipeline.
"""

from langchain_core.messages import HumanMessage, AIMessage

from backend.graph_rag._archive.cypher_validator import validate_cypher_with_explain


def generate_validated_cypher(
    question: str,
    graph,
    llm,
    cypher_prompt,
    max_retries: int = 1,
) -> dict:
    """
    Generates Cypher for `question`, validates it via Neo4j's EXPLAIN, and
    retries up to `max_retries` times - feeding the exact Neo4j error back
    to the model so it can fix its own mistake, instead of just re-rolling
    blind.

    Args:
        question: the user's question, in French.
        graph: the Neo4jGraph object from init_graph() - same one your
            chain already uses. Needs a `.schema` attribute.
        llm: a ChatOpenAI instance (or any LangChain chat model).
        cypher_prompt: the PromptTemplate from get_cypher_prompt(), with
            {schema} and {question} placeholders.
        max_retries: how many extra attempts after the first (default 1,
            meaning 2 total attempts).

    Returns a dict:
        {
            "cypher": final query string, or None if never valid,
            "valid": bool,
            "attempts": int,
            "last_error": str, empty if valid,
        }
    """
    schema = graph.schema
    initial_prompt_text = cypher_prompt.format(schema=schema, question=question)
    messages = [HumanMessage(content=initial_prompt_text)]
    attempts = 0
    last_error = ""

    while attempts <= max_retries:
        attempts += 1
        response = llm.invoke(messages)
        candidate_cypher = response.content.strip()

        is_valid, reason = validate_cypher_with_explain(graph, candidate_cypher)

        if is_valid:
            return {
                "cypher": candidate_cypher,
                "valid": True,
                "attempts": attempts,
                "last_error": "",
            }

        last_error = reason
        if attempts <= max_retries:
            messages.append(AIMessage(content=candidate_cypher))
            messages.append(
                HumanMessage(
                    content=(
                        "Cette requête a échoué avec l'erreur Neo4j suivante :\n"
                        f"{reason}"
                        "\n\nCorrige la requête en respectant toutes les règles "
                        "données plus haut. Retourne UNIQUEMENT la requête "
                        "Cypher corrigée, sans explication, sans Markdown."
                    )
                )
            )

    return {
        "cypher": None,
        "valid": False,
        "attempts": attempts,
        "last_error": last_error,
    }
