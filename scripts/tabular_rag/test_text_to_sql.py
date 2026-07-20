"""
Diagnostic: can the LLM generate correct SQL against the Tabular RAG schema?

This script ONLY prints the SQL the model produces — it never executes it.
It reuses the exact same LLM client configuration as the Graph RAG code
(gpt-4o-mini, temperature=0, OPENAI_API_KEY from backend.config) and builds
the schema description live from data/tabular.db so it can never drift out of
sync with the real database.
"""
import os
import sys
import sqlite3

from langchain_openai import ChatOpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from backend.config import OPENAI_API_KEY

DB_PATH = os.path.join(PROJECT_ROOT, "data", "tabular.db")

TABLES = [
    "horses",
    "events",
    "trainings",
    "training_actors",
    "event_participations",
    "sensors",
]

TEST_QUESTION = "What is the training frequency of Dakota for Event_SJ_01?"

INSTRUCTION = (
    "Given the schema above, write a single SQLite SELECT query (and only the "
    "SQL, no explanation, no markdown formatting) that answers the following "
    "question."
)


def build_schema_description():
    """Read the live schema from tabular.db (read-only) via PRAGMA table_info."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        lines = []
        for table in TABLES:
            cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
            # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            col_descs = [f"{c[1]} {c[2]}" for c in cols]
            lines.append(f"Table {table}:")
            for desc in col_descs:
                lines.append(f"  - {desc}")
        return "\n".join(lines)
    finally:
        conn.close()


def main():
    schema_description = build_schema_description()

    print("=== GENERATED SCHEMA DESCRIPTION ===")
    print(schema_description)

    prompt = (
        f"{schema_description}\n\n"
        f"{INSTRUCTION}\n\n"
        f"Question: {TEST_QUESTION}"
    )

    # Same client config as the Graph RAG code (backend/llm_service.py)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    response = llm.invoke(prompt)

    print("\n=== RAW LLM SQL OUTPUT (not executed) ===")
    print(response.content)


if __name__ == "__main__":
    main()
