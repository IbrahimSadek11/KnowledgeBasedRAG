"""
Export the full contents of data/tabular.db to a single JSON file.

Read-only against tabular.db — never modifies the database.
"""
import json
import os
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tabular.db")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "tabular_export_data.json")

TABLES = [
    "horses",
    "events",
    "seasons",
    "trainings",
    "training_actors",
    "event_participations",
    "sensors",
    "event_entries",
    "objectives",
    "horse_rider_associations",
    "people",
]


def export_table(conn: sqlite3.Connection, table: str) -> list[dict]:
    """Return all rows from `table` as a list of column-name dicts."""
    cur = conn.execute(f"SELECT * FROM {table}")
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def main() -> None:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        export_data = {table: export_table(conn, table) for table in TABLES}
    finally:
        conn.close()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print("Row counts per table:")
    for table in TABLES:
        print(f"  {table}: {len(export_data[table])}")

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        parsed = json.load(f)
    print(f"\njson.load() OK — {len(parsed)} top-level tables")
    print(f"Output: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")


if __name__ == "__main__":
    main()
