"""
SQL safety validator for the Tabular RAG pipeline.

Mirrors the purpose of backend/cypher_validator.py, but for SQL: a pure
validation function with no LLM call and no database execution. It enforces
that generated queries are single SELECT statements and cannot stack multiple
statements. Dangerous write operations are blocked at execution time by
sqlite3's set_authorizer in tabular_chain.execute_sql() — not by scanning
keyword substrings here (which falsely rejects values like '%DROP%').
"""
import re


def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate that `sql` is a safe, read-only single SELECT statement.

    Returns (is_valid, message).
    """
    stripped = sql.strip()

    # Must be a SELECT statement.
    if not stripped[:6].upper() == "SELECT":
        return (False, "Only SELECT statements are permitted.")

    # No query stacking: a semicolon may only appear as a trailing terminator.
    # Any non-whitespace content after a semicolon is rejected.
    if re.search(r";\s*\S", stripped):
        return (False, "Query stacking is not permitted.")

    return (True, "OK")


if __name__ == "__main__":
    test_cases = [
        (
            "SELECT frequency FROM trainings JOIN horses ON trainings.horse_id "
            "= horses.horse_id WHERE horses.name = 'Dakota'",
            True,
        ),
        ("DROP TABLE trainings", False),
        ("SELECT * FROM horses; DROP TABLE horses", False),
        ("DELETE FROM trainings WHERE horse_id = 'Horse1'", False),
        ("SELECT * FROM horses WHERE name = 'updated_athlete'", True),
        ("SELECT * FROM horses WHERE race LIKE '%DROP%'", True),
    ]

    for sql, expected_valid in test_cases:
        is_valid, message = validate_sql(sql)
        match = "MATCH" if is_valid == expected_valid else "MISMATCH"
        print(f"INPUT:    {sql}")
        print(f"RESULT:   ({is_valid}, {message!r})")
        print(f"EXPECTED: valid={expected_valid} -> {match}")
        print("-" * 70)
