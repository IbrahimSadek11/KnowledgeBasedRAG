# ARCHIVE ONLY — reconstructed from bytecode (backend/__pycache__/cypher_validator.cpython-314.pyc).
# Never committed to git; never wired into the live Graph RAG pipeline.
# See docs/graph_rag/graph_validator_status.md. Do not import from production code.

"""
Standalone Cypher validation helpers.

Two layers:
1. validate_cypher_syntax() - fast, no database call. Catches obvious
   problems (forbidden keywords, banned SQL-isms like HAVING/GROUP BY,
   missing RETURN, stray markdown fences) before we even touch Neo4j.
2. validate_cypher_with_explain() - the real check. Asks Neo4j itself to
   parse the query via EXPLAIN, without executing it. This catches anything
   Neo4j would reject - undefined variables, invalid syntax, whatever - even
   patterns we haven't seen yet and have no prompt rule for.

Nothing in this file is wired into the chain yet. It's standalone and
testable on its own first.
"""

import re

FORBIDDEN_KEYWORDS = [
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD CSV",
    "FOREACH",
    "CALL",
    "USE",
]

BANNED_SYNTAX = ["HAVING", "GROUP BY"]


def validate_cypher_syntax(query: str) -> tuple[bool, str]:
    """
    Fast, no-database-call checks. Returns (is_valid: bool, reason: str).
    reason is empty string when is_valid is True.
    """
    cleaned = query.strip()
    cleaned = re.sub(
        r"^```(?:cypher)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    if not cleaned:
        return False, "empty query"

    if ";" in cleaned.rstrip(";"):
        return False, "multiple statements (semicolon) detected"

    upper = cleaned.upper()

    for kw in FORBIDDEN_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", upper):
            return False, f"forbidden write/admin keyword: {kw}"

    for kw in BANNED_SYNTAX:
        if re.search(r"\b" + re.escape(kw) + r"\b", upper):
            return False, f"invalid Cypher syntax for this pipeline: {kw}"

    if "RETURN" not in upper:
        return False, "no RETURN clause"

    return True, ""


def validate_cypher_with_explain(graph, query: str) -> tuple[bool, str]:
    """
    Real validation via Neo4j's own parser. `graph` is the same Neo4jGraph
    object your chain already uses (from init_graph()). Runs the fast checks
    first, then asks Neo4j to EXPLAIN the query - this parses and validates
    it WITHOUT executing it, so nothing touches real data.
    Returns (is_valid: bool, reason: str).
    """
    is_valid, reason = validate_cypher_syntax(query)
    if not is_valid:
        return False, reason

    try:
        graph.query(f"EXPLAIN {query}")
        return True, ""
    except Exception as e:
        return False, str(e)
