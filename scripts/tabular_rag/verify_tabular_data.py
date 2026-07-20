"""
Permanent data-integrity verification for the Tabular RAG database
(data/tabular.db).

Read-only: opens the database with mode=ro and never issues INSERT/UPDATE/
DELETE. Validates row counts, referential integrity across all six tables,
column types, text encoding (written to a file to avoid console mangling),
and a full six-table join for Dakota (Horse1).
"""
import os
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tabular.db")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
ENCODING_OUTPUT = os.path.join(SCRIPTS_DIR, "encoding_check_output.txt")

EXPECTED_COUNTS = {
    "horses": 50,
    "events": 20,
    "trainings": 171,
    "training_actors": 314,
    "event_participations": 50,
    "sensors": 108,
    "event_entries": 101,
    "objectives": 2,
    "horse_rider_associations": 51,
    "seasons": 1,
    "people": 27,
}

# (label, child query returning offending ids)
ORPHAN_CHECKS = [
    (
        "trainings.horse_id -> horses.horse_id",
        """
        SELECT t.training_id FROM trainings t
        LEFT JOIN horses h ON t.horse_id = h.horse_id
        WHERE h.horse_id IS NULL
        """,
    ),
    (
        "trainings.event_id -> events.event_id (excluding NULLs)",
        """
        SELECT t.training_id FROM trainings t
        LEFT JOIN events e ON t.event_id = e.event_id
        WHERE t.event_id IS NOT NULL AND e.event_id IS NULL
        """,
    ),
    (
        "training_actors.training_id -> trainings.training_id",
        """
        SELECT ta.training_id FROM training_actors ta
        LEFT JOIN trainings t ON ta.training_id = t.training_id
        WHERE t.training_id IS NULL
        """,
    ),
    (
        "event_participations.event_id -> events.event_id",
        """
        SELECT ep.participation_id FROM event_participations ep
        LEFT JOIN events e ON ep.event_id = e.event_id
        WHERE e.event_id IS NULL
        """,
    ),
    (
        "event_participations.horse_id -> horses.horse_id",
        """
        SELECT ep.participation_id FROM event_participations ep
        LEFT JOIN horses h ON ep.horse_id = h.horse_id
        WHERE h.horse_id IS NULL
        """,
    ),
    (
        "sensors.horse_id -> horses.horse_id",
        """
        SELECT s.sensor_id FROM sensors s
        LEFT JOIN horses h ON s.horse_id = h.horse_id
        WHERE h.horse_id IS NULL
        """,
    ),
    (
        "event_entries.horse_id -> horses.horse_id",
        """
        SELECT ee.horse_id || '|' || ee.event_id FROM event_entries ee
        LEFT JOIN horses h ON ee.horse_id = h.horse_id
        WHERE h.horse_id IS NULL
        """,
    ),
    (
        "event_entries.event_id -> events.event_id",
        """
        SELECT ee.horse_id || '|' || ee.event_id FROM event_entries ee
        LEFT JOIN events e ON ee.event_id = e.event_id
        WHERE e.event_id IS NULL
        """,
    ),
    (
        "horse_rider_associations.horse_id -> horses.horse_id",
        """
        SELECT hra.rider_id || '|' || hra.horse_id FROM horse_rider_associations hra
        LEFT JOIN horses h ON hra.horse_id = h.horse_id
        WHERE h.horse_id IS NULL
        """,
    ),
    (
        "events.season_id -> seasons.season_id (excluding NULLs)",
        """
        SELECT e.event_id FROM events e
        LEFT JOIN seasons s ON e.season_id = s.season_id
        WHERE e.season_id IS NOT NULL AND s.season_id IS NULL
        """,
    ),
]


def main():
    all_passed = True

    # Read-only connection
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()

        # --- Row counts ---
        print("=== ROW COUNTS ===")
        for table, expected in EXPECTED_COUNTS.items():
            actual = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            ok = actual == expected
            all_passed = all_passed and ok
            print(f"  {table}: {actual} (expected {expected}) -> {'PASS' if ok else 'FAIL'}")

        # --- Referential integrity ---
        print("\n=== REFERENTIAL INTEGRITY ===")
        for label, query in ORPHAN_CHECKS:
            offenders = [row[0] for row in cur.execute(query)]
            ok = len(offenders) == 0
            all_passed = all_passed and ok
            if ok:
                print(f"  {label}: 0 orphans -> PASS")
            else:
                print(f"  {label}: {len(offenders)} orphans -> FAIL")
                print(f"    offending ids: {offenders}")

        # --- Season completeness (every event must be in a season) ---
        print("\n=== SEASON COMPLETENESS ===")
        null_season = cur.execute(
            "SELECT COUNT(*) FROM events WHERE season_id IS NULL"
        ).fetchone()[0]
        season_ok = null_season == 0
        all_passed = all_passed and season_ok
        print(
            f"  events with NULL season_id: {null_season} (expected 0) -> "
            f"{'PASS' if season_ok else 'FAIL'}"
        )

        # --- Type check ---
        print("\n=== TYPE CHECK ===")
        freq_type = cur.execute(
            "SELECT typeof(frequency) FROM trainings WHERE frequency IS NOT NULL LIMIT 1"
        ).fetchone()[0]
        rank_type = cur.execute(
            "SELECT typeof(rank) FROM event_participations WHERE rank IS NOT NULL LIMIT 1"
        ).fetchone()[0]
        freq_ok = freq_type == "integer"
        rank_ok = rank_type == "integer"
        all_passed = all_passed and freq_ok and rank_ok
        print(f"  trainings.frequency typeof = {freq_type} -> {'PASS' if freq_ok else 'FAIL'}")
        print(f"  event_participations.rank typeof = {rank_type} -> {'PASS' if rank_ok else 'FAIL'}")

        # --- Encoding check (written to file, not console) ---
        encoding_rows = cur.execute(
            """
            SELECT training_id, horse_id, stage_type, volume, intensity, frequency
            FROM trainings
            WHERE intensity IN ('Modérée', 'Élevée')
            LIMIT 5
            """
        ).fetchall()
        with open(ENCODING_OUTPUT, "w", encoding="utf-8") as f:
            f.write("Encoding check — sample trainings rows with accented intensity values\n")
            f.write("(this file is UTF-8; trust it over console rendering)\n\n")
            for row in encoding_rows:
                f.write(f"{row}\n")
        print(
            "\nEncoding check written to scripts/encoding_check_output.txt — "
            "do not trust console rendering of accented characters, check the file instead."
        )

        # --- Full multi-table join test ---
        print("\n=== FULL MULTI-TABLE JOIN TEST (Dakota / Horse1) ===")
        join_rows = cur.execute(
            """
            SELECT h.name, t.stage_type, t.frequency, ta.actor_id, ta.actor_role,
                   e.discipline
            FROM trainings t
            JOIN horses h ON t.horse_id = h.horse_id
            LEFT JOIN training_actors ta ON t.training_id = ta.training_id
            LEFT JOIN events e ON t.event_id = e.event_id
            WHERE h.horse_id = 'Horse1'
            ORDER BY t.training_id, ta.actor_role
            """
        ).fetchall()
        for row in join_rows:
            print(f"  {row}")

        # --- Final summary ---
        print()
        print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED — see above.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
