"""
Exhaustive full-row verification of data/tabular.db against the Neo4j graph.

For every one of the six tables, this pulls a fresh "ground truth" from Neo4j
using INDEPENDENTLY formulated Cypher (different traversal directions, UNION
per label instead of WHERE...OR) so it genuinely double-checks the ETL rather
than re-running identical queries. It then compares every field of every row,
keyed by primary key, and reports missing rows, orphaned rows, and per-field
mismatches.

The SQLite database is opened read-only (mode=ro).
"""
import os
import sqlite3

from neo4j import GraphDatabase

from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tabular.db")
DIFF_OUTPUT = os.path.join(PROJECT_ROOT, "scripts", "full_diff_output.txt")

SENSOR_TYPES = {"Withers", "Sternum", "CanonOfForelimb", "CanonOfHindlimb"}
ACTOR_ROLES = {"Rider", "Veterinarian", "Caretaker"}

EXPECTED_TOTAL = 50 + 20 + 171 + 314 + 50 + 108  # 713


def date_to_iso(value):
    if value is None:
        return None
    iso = getattr(value, "iso_format", None)
    return iso() if callable(iso) else str(value)


def pick(labels, allowed):
    for label in labels:
        if label in allowed:
            return label
    return None


# --- Independently formulated Cypher (see module docstring) ---

HORSES_CYPHER = """
MATCH (h:Horse)
RETURN h.id AS horse_id, h.hasName AS name, h.hasRace AS race
"""

# UNION per discipline label instead of WHERE ... OR, discipline as a literal.
EVENTS_CYPHER = """
MATCH (e:ShowJumping)
RETURN e.id AS event_id, e.eventLocation AS location, e.category AS category,
       e.eventDate AS event_date, 'ShowJumping' AS discipline
UNION
MATCH (e:Cross)
RETURN e.id AS event_id, e.eventLocation AS location, e.category AS category,
       e.eventDate AS event_date, 'Cross' AS discipline
UNION
MATCH (e:Dressage)
RETURN e.id AS event_id, e.eventLocation AS location, e.category AS category,
       e.eventDate AS event_date, 'Dressage' AS discipline
"""

# UNION per stage label, reverse TRAINSIN direction, literal stage_type.
TRAININGS_CYPHER = """
MATCH (t:PreparationStage)<-[:TRAINSIN]-(h:Horse)
OPTIONAL MATCH (t)-[:DEPENDSON]->(e)
RETURN t.id AS training_id, h.id AS horse_id, e.id AS event_id,
       t.Volume AS volume, t.Intensity AS intensity, t.Frequency AS frequency,
       'PreparationStage' AS stage_type
UNION
MATCH (t:PreCompetitionStage)<-[:TRAINSIN]-(h:Horse)
OPTIONAL MATCH (t)-[:DEPENDSON]->(e)
RETURN t.id AS training_id, h.id AS horse_id, e.id AS event_id,
       t.Volume AS volume, t.Intensity AS intensity, t.Frequency AS frequency,
       'PreCompetitionStage' AS stage_type
UNION
MATCH (t:CompetitionStage)<-[:TRAINSIN]-(h:Horse)
OPTIONAL MATCH (t)-[:DEPENDSON]->(e)
RETURN t.id AS training_id, h.id AS horse_id, e.id AS event_id,
       t.Volume AS volume, t.Intensity AS intensity, t.Frequency AS frequency,
       'CompetitionStage' AS stage_type
UNION
MATCH (t:TransitionStage)<-[:TRAINSIN]-(h:Horse)
OPTIONAL MATCH (t)-[:DEPENDSON]->(e)
RETURN t.id AS training_id, h.id AS horse_id, e.id AS event_id,
       t.Volume AS volume, t.Intensity AS intensity, t.Frequency AS frequency,
       'TransitionStage' AS stage_type
"""

# Reverse INVOLVESACTOR direction.
TRAINING_ACTORS_CYPHER = """
MATCH (a)<-[:INVOLVESACTOR]-(t)
WHERE t:PreparationStage OR t:PreCompetitionStage
   OR t:CompetitionStage OR t:TransitionStage
RETURN t.id AS training_id, a.id AS actor_id, labels(a) AS labels
"""

# Start from EventParticipation and traverse outward (reverse HASPARTICIPATION).
PARTICIPATIONS_CYPHER = """
MATCH (p:EventParticipation)
MATCH (p)<-[:HASPARTICIPATION]-(e)
MATCH (p)-[:HASHORSE]->(h:Horse)
MATCH (p)-[:HASRIDER]->(r:Rider)
RETURN p.id AS participation_id, e.id AS event_id, h.id AS horse_id,
       r.id AS rider_id, p.rank AS rank
"""

# Start from Horse, traverse reverse ISATTACHEDTO to the sensor.
SENSORS_CYPHER = """
MATCH (h:Horse)<-[:ISATTACHEDTO]-(s:InertialSensors)
OPTIONAL MATCH (s)-[:ISUSEDFOR]->(o:ExperimentalObjective)
RETURN s.id AS sensor_id, h.id AS horse_id, labels(s) AS labels,
       s.hasSensorID AS sensor_code, s.hasFormat AS format,
       s.hasSensorOffset AS sensor_offset, s.hasFileSize AS file_size,
       s.hasSensorTime AS sample_rate, o.id AS objective_id
"""


def build_neo4j_dicts(session):
    """Return {table: (fields, {key: value_tuple})} pulled fresh from Neo4j."""
    data = {}

    # horses: key horse_id -> (name, race)
    data["horses"] = (
        ["name", "race"],
        {
            rec["horse_id"]: (rec["name"], rec["race"])
            for rec in session.run(HORSES_CYPHER)
        },
    )

    # events: key event_id -> (location, category, event_date, discipline)
    data["events"] = (
        ["location", "category", "event_date", "discipline"],
        {
            rec["event_id"]: (
                rec["location"],
                rec["category"],
                date_to_iso(rec["event_date"]),
                rec["discipline"],
            )
            for rec in session.run(EVENTS_CYPHER)
        },
    )

    # trainings: key training_id -> (horse_id, event_id, stage_type, volume, intensity, frequency)
    data["trainings"] = (
        ["horse_id", "event_id", "stage_type", "volume", "intensity", "frequency"],
        {
            rec["training_id"]: (
                rec["horse_id"],
                rec["event_id"],
                rec["stage_type"],
                rec["volume"],
                rec["intensity"],
                rec["frequency"],
            )
            for rec in session.run(TRAININGS_CYPHER)
        },
    )

    # training_actors: key (training_id, actor_id) -> (actor_role,)
    ta = {}
    for rec in session.run(TRAINING_ACTORS_CYPHER):
        ta[(rec["training_id"], rec["actor_id"])] = (pick(rec["labels"], ACTOR_ROLES),)
    data["training_actors"] = (["actor_role"], ta)

    # event_participations: key participation_id -> (event_id, horse_id, rider_id, rank)
    data["event_participations"] = (
        ["event_id", "horse_id", "rider_id", "rank"],
        {
            rec["participation_id"]: (
                rec["event_id"],
                rec["horse_id"],
                rec["rider_id"],
                rec["rank"],
            )
            for rec in session.run(PARTICIPATIONS_CYPHER)
        },
    )

    # sensors: key sensor_id -> (horse_id, sensor_type, sensor_code, format,
    #                            sensor_offset, file_size, sample_rate, objective_id)
    sensors = {}
    for rec in session.run(SENSORS_CYPHER):
        sensors[rec["sensor_id"]] = (
            rec["horse_id"],
            pick(rec["labels"], SENSOR_TYPES),
            rec["sensor_code"],
            rec["format"],
            rec["sensor_offset"],
            rec["file_size"],
            rec["sample_rate"],
            rec["objective_id"],
        )
    data["sensors"] = (
        [
            "horse_id",
            "sensor_type",
            "sensor_code",
            "format",
            "sensor_offset",
            "file_size",
            "sample_rate",
            "objective_id",
        ],
        sensors,
    )

    return data


def build_sqlite_dicts(cur):
    """Return {table: {key: value_tuple}} read from tabular.db (read-only)."""
    data = {}

    data["horses"] = {
        row[0]: (row[1], row[2])
        for row in cur.execute("SELECT horse_id, name, race FROM horses")
    }

    data["events"] = {
        row[0]: (row[1], row[2], row[3], row[4])
        for row in cur.execute(
            "SELECT event_id, location, category, event_date, discipline FROM events"
        )
    }

    data["trainings"] = {
        row[0]: (row[1], row[2], row[3], row[4], row[5], row[6])
        for row in cur.execute(
            "SELECT training_id, horse_id, event_id, stage_type, volume, intensity, "
            "frequency FROM trainings"
        )
    }

    data["training_actors"] = {
        (row[0], row[1]): (row[2],)
        for row in cur.execute(
            "SELECT training_id, actor_id, actor_role FROM training_actors"
        )
    }

    data["event_participations"] = {
        row[0]: (row[1], row[2], row[3], row[4])
        for row in cur.execute(
            "SELECT participation_id, event_id, horse_id, rider_id, rank "
            "FROM event_participations"
        )
    }

    data["sensors"] = {
        row[0]: (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8])
        for row in cur.execute(
            "SELECT sensor_id, horse_id, sensor_type, sensor_code, format, "
            "sensor_offset, file_size, sample_rate, objective_id FROM sensors"
        )
    }

    return data


def compare_table(name, fields, neo, sql, sink):
    """Compare one table. Returns (rows_checked, discrepancy_count)."""
    neo_keys = set(neo)
    sql_keys = set(sql)

    missing = sorted(neo_keys - sql_keys, key=str)      # in graph, absent from db
    orphaned = sorted(sql_keys - neo_keys, key=str)     # in db, absent from graph
    mismatches = []
    for key in sorted(neo_keys & sql_keys, key=str):
        nvals, svals = neo[key], sql[key]
        for i, field in enumerate(fields):
            if nvals[i] != svals[i]:
                mismatches.append((key, field, nvals[i], svals[i]))

    discrepancies = len(missing) + len(orphaned) + len(mismatches)
    passed = discrepancies == 0

    sink(f"\n=== TABLE: {name} ===")
    sink(f"  neo4j rows: {len(neo)} | tabular rows: {len(sql)}")
    if passed:
        sink(f"  {name}: PASS ({len(neo)} rows, every field identical)")
    else:
        sink(f"  {name}: FAIL")
        if missing:
            sink(f"  MISSING from tabular.db ({len(missing)}):")
            for k in missing:
                sink(f"    {k} -> neo4j value {neo[k]}")
        if orphaned:
            sink(f"  ORPHANED in tabular.db ({len(orphaned)}):")
            for k in orphaned:
                sink(f"    {k} -> tabular value {sql[k]}")
        if mismatches:
            sink(f"  FIELD MISMATCHES ({len(mismatches)}):")
            for key, field, nval, sval in mismatches:
                sink(f"    key={key} field={field}: neo4j={nval!r} tabular={sval!r}")

    return len(neo), discrepancies


def main():
    lines = []

    def sink(text):
        print(text)
        lines.append(text)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            neo_data = build_neo4j_dicts(session)
        cur = conn.cursor()
        sql_data = build_sqlite_dicts(cur)
    finally:
        driver.close()
        conn.close()

    table_order = [
        "horses",
        "events",
        "trainings",
        "training_actors",
        "event_participations",
        "sensors",
    ]

    total_rows = 0
    total_discrepancies = 0
    for name in table_order:
        fields, neo = neo_data[name]
        sql = sql_data[name]
        rows, discrepancies = compare_table(name, fields, neo, sql, sink)
        total_rows += rows
        total_discrepancies += discrepancies

    sink("\n=== FINAL SUMMARY ===")
    sink(f"  total rows checked across six tables: {total_rows} (expected {EXPECTED_TOTAL})")
    sink(f"  total discrepancies found: {total_discrepancies}")
    if total_discrepancies == 0 and total_rows == EXPECTED_TOTAL:
        sink(f"\nPERFECT MATCH — {total_rows}/{EXPECTED_TOTAL} ROWS VERIFIED IDENTICAL TO GRAPH")
    else:
        sink("\nDISCREPANCIES FOUND — see full details above.")
        # Persist full detail to a UTF-8 file so accented characters are trustworthy.
        with open(DIFF_OUTPUT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nFull UTF-8 detail written to scripts/full_diff_output.txt")


if __name__ == "__main__":
    main()
