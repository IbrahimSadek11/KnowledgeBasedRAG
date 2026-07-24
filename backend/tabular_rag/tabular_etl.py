"""
Tabular RAG data layer — ETL slice #1.

Builds a SQLite database (data/tabular.db) from two Neo4j node types (Horse and
Event subtypes). Reuses the existing Graph RAG Neo4j connection config from
backend/config.py. Read-only against Neo4j; only writes to the local SQLite file.
"""
import os
import re
import sqlite3

from neo4j import GraphDatabase

from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

# Resolve paths relative to the project root (parent of backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "tabular.db")

# Deliberate exception: person nodes are not all reachable via one relationship,
# so PEOPLE_QUERY still enumerates Rider/Caretaker/Veterinarian labels. ACTOR_ROLES
# documents that closed set; do not "clean it up" to match relationship-scoped queries.
ACTOR_ROLES = {"Rider", "Veterinarian", "Caretaker"}

GENERIC_SENSOR_LABEL = "InertialSensors"

HORSE_QUERY = """
MATCH (h:Horse)
RETURN h.id AS horse_id, h.hasName AS name, h.hasRace AS race
"""

EVENT_QUERY = """
MATCH (e)-[:INSEASON]->(:CompetitiveSeason)
RETURN e.id AS event_id, e.eventLocation AS location, e.category AS category,
       e.eventDate AS event_date, labels(e) AS labels
"""

TRAINING_QUERY = """
MATCH (h:Horse)-[:TRAINSIN]->(t)
OPTIONAL MATCH (t)-[:DEPENDSON]->(e)
RETURN t.id AS training_id, h.id AS horse_id, e.id AS event_id,
       t.Volume AS volume, t.Intensity AS intensity,
       t.Frequency AS frequency, labels(t) AS labels
"""

TRAINING_ACTOR_QUERY = """
MATCH (t)-[:INVOLVESACTOR]->(a)
RETURN t.id AS training_id, a.id AS actor_id, labels(a) AS labels
"""

PARTICIPATION_QUERY = """
MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
MATCH (p)-[:HASHORSE]->(h:Horse)
MATCH (p)-[:HASRIDER]->(r:Rider)
RETURN p.id AS participation_id, e.id AS event_id, h.id AS horse_id,
       r.id AS rider_id, p.rank AS rank
"""

SENSOR_QUERY = """
MATCH (s:InertialSensors)-[:ISATTACHEDTO]->(h:Horse)
OPTIONAL MATCH (s)-[:ISUSEDFOR]->(o:ExperimentalObjective)
RETURN s.id AS sensor_id, h.id AS horse_id, labels(s) AS labels,
       s.hasSensorID AS sensor_code, s.hasFormat AS format,
       s.hasSensorOffset AS sensor_offset, s.hasFileSize AS file_size,
       s.hasSensorTime AS sample_rate, o.id AS objective_id
"""

EVENT_ENTRY_QUERY = """
MATCH (h:Horse)-[:COMPETESIN]->(e)
RETURN h.id AS horse_id, e.id AS event_id
"""

OBJECTIVE_QUERY = """
MATCH (o:ExperimentalObjective)
RETURN o.id AS objective_id, o.hasName AS name, o.description AS description
"""

ASSOCIATION_QUERY = """
MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse)
RETURN r.id AS rider_id, h.id AS horse_id
"""

SEASON_QUERY = """
MATCH (s:CompetitiveSeason)
RETURN s.id AS season_id, s.seasonName AS season_name,
       s.seasonStart AS season_start, s.seasonEnd AS season_end
"""

EVENT_SEASON_QUERY = """
MATCH (e)-[:INSEASON]->(s:CompetitiveSeason)
RETURN e.id AS event_id, s.id AS season_id
"""

PEOPLE_QUERY = """
MATCH (p) WHERE p:Rider OR p:Caretaker OR p:Veterinarian
RETURN p.id AS person_id, labels(p) AS labels
"""


def date_to_iso(value):
    """Convert a Neo4j Date (or None) to an ISO YYYY-MM-DD string."""
    if value is None:
        return None
    # neo4j.time.Date exposes iso_format(); fall back to str() otherwise
    iso = getattr(value, "iso_format", None)
    if callable(iso):
        return iso()
    return str(value)


def pick_discipline(labels):
    """Return the (single) discipline label on an event node."""
    return labels[0] if labels else None


def pick_stage_type(labels):
    """Return the (single) training-stage label on a training node."""
    return labels[0] if labels else None


def pick_actor_role(labels):
    """Return the (single) role label on an actor/person node."""
    return labels[0] if labels else None


def pick_sensor_type(labels):
    """Return the position-specific label, skipping the shared InertialSensors label."""
    for label in labels:
        if label != GENERIC_SENSOR_LABEL:
            return label
    return None


def parse_leading_int(text):
    """Parse the leading integer from a unit-suffixed TEXT value (e.g. '45min', '250Hz')."""
    if text is None:
        return None
    match = re.match(r"^\s*(-?\d+)", str(text))
    return int(match.group(1)) if match else None


def parse_real(text):
    """Parse a REAL from a TEXT numeric value (e.g. '0.02')."""
    if text is None:
        return None
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)", str(text))
    return float(match.group(1)) if match else None


def fetch_from_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            horses = [
                (rec["horse_id"], rec["name"], rec["race"])
                for rec in session.run(HORSE_QUERY)
            ]
            event_seasons = {
                rec["event_id"]: rec["season_id"]
                for rec in session.run(EVENT_SEASON_QUERY)
            }
            events = [
                (
                    rec["event_id"],
                    rec["location"],
                    rec["category"],
                    date_to_iso(rec["event_date"]),
                    pick_discipline(rec["labels"]),
                    event_seasons.get(rec["event_id"]),
                )
                for rec in session.run(EVENT_QUERY)
            ]
            trainings = [
                (
                    rec["training_id"],
                    rec["horse_id"],
                    rec["event_id"],
                    pick_stage_type(rec["labels"]),
                    rec["volume"],
                    rec["intensity"],
                    rec["frequency"],
                )
                for rec in session.run(TRAINING_QUERY)
            ]
            training_actors = [
                (
                    rec["training_id"],
                    rec["actor_id"],
                    pick_actor_role(rec["labels"]),
                )
                for rec in session.run(TRAINING_ACTOR_QUERY)
            ]
            event_participations = [
                (
                    rec["participation_id"],
                    rec["event_id"],
                    rec["horse_id"],
                    rec["rider_id"],
                    rec["rank"],
                )
                for rec in session.run(PARTICIPATION_QUERY)
            ]
            sensors = [
                (
                    rec["sensor_id"],
                    rec["horse_id"],
                    pick_sensor_type(rec["labels"]),
                    rec["sensor_code"],
                    rec["format"],
                    rec["sensor_offset"],
                    rec["file_size"],
                    rec["sample_rate"],
                    rec["objective_id"],
                )
                for rec in session.run(SENSOR_QUERY)
            ]
            event_entries = [
                (rec["horse_id"], rec["event_id"])
                for rec in session.run(EVENT_ENTRY_QUERY)
            ]
            objectives = [
                (rec["objective_id"], rec["name"], rec["description"])
                for rec in session.run(OBJECTIVE_QUERY)
            ]
            horse_rider_associations = [
                (rec["rider_id"], rec["horse_id"])
                for rec in session.run(ASSOCIATION_QUERY)
            ]
            seasons = [
                (
                    rec["season_id"],
                    rec["season_name"],
                    date_to_iso(rec["season_start"]),
                    date_to_iso(rec["season_end"]),
                )
                for rec in session.run(SEASON_QUERY)
            ]
            people = [
                (rec["person_id"], pick_actor_role(rec["labels"]))
                for rec in session.run(PEOPLE_QUERY)
            ]
    finally:
        driver.close()
    return (
        horses,
        events,
        trainings,
        training_actors,
        event_participations,
        sensors,
        event_entries,
        objectives,
        horse_rider_associations,
        seasons,
        people,
    )


def build_sqlite(
    horses,
    events,
    trainings,
    training_actors,
    event_participations,
    sensors,
    event_entries,
    objectives,
    horse_rider_associations,
    seasons,
    people,
):
    os.makedirs(DATA_DIR, exist_ok=True)

    # Derived from people — no extra Cypher.
    riders = [(person_id,) for person_id, role in people if role == "Rider"]
    veterinarians = [
        (person_id,) for person_id, role in people if role == "Veterinarian"
    ]
    caretakers = [
        (person_id,) for person_id, role in people if role == "Caretaker"
    ]

    # Enrich trainings/sensors with additive parsed numeric columns (original TEXT kept).
    trainings_rows = [
        (
            training_id,
            horse_id,
            event_id,
            stage_type,
            volume,
            parse_leading_int(volume),
            intensity,
            frequency,
        )
        for (
            training_id,
            horse_id,
            event_id,
            stage_type,
            volume,
            intensity,
            frequency,
        ) in trainings
    ]
    sensors_rows = [
        (
            sensor_id,
            horse_id,
            sensor_type,
            sensor_code,
            format_,
            sensor_offset,
            parse_real(sensor_offset),
            file_size,
            sample_rate,
            parse_leading_int(sample_rate),
            objective_id,
        )
        for (
            sensor_id,
            horse_id,
            sensor_type,
            sensor_code,
            format_,
            sensor_offset,
            file_size,
            sample_rate,
            objective_id,
        ) in sensors
    ]

    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str  # explicit UTF-8 text handling
    try:
        cur = conn.cursor()
        # Drop dependents first so re-runs succeed with declared FKs.
        cur.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "horse_rider_associations",
            "sensors",
            "event_participations",
            "training_actors",
            "trainings",
            "event_entries",
            "riders",
            "veterinarians",
            "caretakers",
            "objectives",
            "people",
            "events",
            "seasons",
            "horses",
        ):
            cur.execute(f"DROP TABLE IF EXISTS {table}")

        # ---- CREATE + INSERT in dependency order (Part D) ----
        # 1. horses
        cur.execute(
            """
            CREATE TABLE horses (
                horse_id TEXT PRIMARY KEY,
                name TEXT,
                race TEXT
            )
            """
        )
        cur.executemany(
            "INSERT INTO horses (horse_id, name, race) VALUES (?, ?, ?)",
            horses,
        )

        # 2. seasons
        cur.execute(
            """
            CREATE TABLE seasons (
                season_id TEXT PRIMARY KEY,
                season_name TEXT,
                season_start TEXT,
                season_end TEXT
            )
            """
        )
        cur.executemany(
            "INSERT INTO seasons (season_id, season_name, season_start, season_end) "
            "VALUES (?, ?, ?, ?)",
            seasons,
        )

        # 3. events
        cur.execute(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                location TEXT,
                category TEXT,
                event_date TEXT,
                discipline TEXT,
                season_id TEXT,
                FOREIGN KEY (season_id) REFERENCES seasons(season_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO events "
            "(event_id, location, category, event_date, discipline, season_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            events,
        )

        # 4. people
        cur.execute(
            """
            CREATE TABLE people (
                person_id TEXT PRIMARY KEY,
                role TEXT
            )
            """
        )
        cur.executemany(
            "INSERT INTO people (person_id, role) VALUES (?, ?)",
            people,
        )

        # 5. riders
        cur.execute(
            """
            CREATE TABLE riders (
                person_id TEXT PRIMARY KEY,
                FOREIGN KEY (person_id) REFERENCES people(person_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO riders (person_id) VALUES (?)",
            riders,
        )

        # 6. veterinarians
        cur.execute(
            """
            CREATE TABLE veterinarians (
                person_id TEXT PRIMARY KEY,
                FOREIGN KEY (person_id) REFERENCES people(person_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO veterinarians (person_id) VALUES (?)",
            veterinarians,
        )

        # 7. caretakers
        cur.execute(
            """
            CREATE TABLE caretakers (
                person_id TEXT PRIMARY KEY,
                FOREIGN KEY (person_id) REFERENCES people(person_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO caretakers (person_id) VALUES (?)",
            caretakers,
        )

        # 8. objectives
        cur.execute(
            """
            CREATE TABLE objectives (
                objective_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT
            )
            """
        )
        cur.executemany(
            "INSERT INTO objectives (objective_id, name, description) "
            "VALUES (?, ?, ?)",
            objectives,
        )

        # 9. event_entries (must precede event_participations for the composite FK)
        cur.execute(
            """
            CREATE TABLE event_entries (
                horse_id TEXT,
                event_id TEXT,
                PRIMARY KEY (horse_id, event_id),
                FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO event_entries (horse_id, event_id) VALUES (?, ?)",
            event_entries,
        )

        # 10. trainings
        cur.execute(
            """
            CREATE TABLE trainings (
                training_id TEXT PRIMARY KEY,
                horse_id TEXT,
                event_id TEXT,
                stage_type TEXT,
                volume TEXT,
                volume_minutes INTEGER,
                intensity TEXT,
                frequency INTEGER,
                FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO trainings "
            "(training_id, horse_id, event_id, stage_type, volume, volume_minutes, "
            "intensity, frequency) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            trainings_rows,
        )

        # 11. training_actors
        cur.execute(
            """
            CREATE TABLE training_actors (
                training_id TEXT,
                actor_id TEXT,
                actor_role TEXT,
                PRIMARY KEY (training_id, actor_id),
                FOREIGN KEY (training_id) REFERENCES trainings(training_id),
                FOREIGN KEY (actor_id) REFERENCES people(person_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO training_actors (training_id, actor_id, actor_role) "
            "VALUES (?, ?, ?)",
            training_actors,
        )

        # 12. event_participations
        cur.execute(
            """
            CREATE TABLE event_participations (
                participation_id TEXT PRIMARY KEY,
                event_id TEXT,
                horse_id TEXT,
                rider_id TEXT,
                rank INTEGER,
                FOREIGN KEY (event_id) REFERENCES events(event_id),
                FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
                FOREIGN KEY (rider_id) REFERENCES riders(person_id),
                FOREIGN KEY (horse_id, event_id) REFERENCES event_entries(horse_id, event_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO event_participations "
            "(participation_id, event_id, horse_id, rider_id, rank) "
            "VALUES (?, ?, ?, ?, ?)",
            event_participations,
        )

        # 13. sensors
        cur.execute(
            """
            CREATE TABLE sensors (
                sensor_id TEXT PRIMARY KEY,
                horse_id TEXT,
                sensor_type TEXT,
                sensor_code TEXT,
                format TEXT,
                sensor_offset TEXT,
                sensor_offset_value REAL,
                file_size INTEGER,
                sample_rate TEXT,
                sample_rate_hz INTEGER,
                objective_id TEXT,
                FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
                FOREIGN KEY (objective_id) REFERENCES objectives(objective_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO sensors "
            "(sensor_id, horse_id, sensor_type, sensor_code, format, "
            "sensor_offset, sensor_offset_value, file_size, sample_rate, "
            "sample_rate_hz, objective_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sensors_rows,
        )

        # 14. horse_rider_associations
        cur.execute(
            """
            CREATE TABLE horse_rider_associations (
                rider_id TEXT,
                horse_id TEXT,
                PRIMARY KEY (rider_id, horse_id),
                FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
                FOREIGN KEY (rider_id) REFERENCES riders(person_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO horse_rider_associations (rider_id, horse_id) "
            "VALUES (?, ?)",
            horse_rider_associations,
        )

        cur.execute("PRAGMA foreign_keys = ON")
        conn.commit()

        horse_count = cur.execute("SELECT COUNT(*) FROM horses").fetchone()[0]
        event_count = cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        training_count = cur.execute("SELECT COUNT(*) FROM trainings").fetchone()[0]
        rider_count = cur.execute("SELECT COUNT(*) FROM riders").fetchone()[0]
        veterinarian_count = cur.execute(
            "SELECT COUNT(*) FROM veterinarians"
        ).fetchone()[0]
        caretaker_count = cur.execute("SELECT COUNT(*) FROM caretakers").fetchone()[0]

        print(f"horses: {horse_count}")
        print(f"events: {event_count}")
        print(f"trainings: {training_count}")
        print(f"riders: {rider_count}")
        print(f"veterinarians: {veterinarian_count}")
        print(f"caretakers: {caretaker_count}")

        print("\nSample horses:")
        for row in cur.execute("SELECT * FROM horses LIMIT 3"):
            print(f"  {row}")

        print("\nSample events:")
        for row in cur.execute("SELECT * FROM events LIMIT 3"):
            print(f"  {row}")

        null_event_rows = cur.execute(
            "SELECT training_id FROM trainings WHERE event_id IS NULL"
        ).fetchall()
        print(f"\ntrainings with NULL event_id: {len(null_event_rows)}")
        if null_event_rows:
            print("  missing-link training_ids:")
            for (training_id,) in null_event_rows:
                print(f"    {training_id}")

        print("\nSample trainings:")
        for row in cur.execute("SELECT * FROM trainings LIMIT 3"):
            print(f"  {row}")

        print("\nDakota / Event_SJ_01 join result:")
        dakota_rows = cur.execute(
            """
            SELECT h.name, e.event_id, t.stage_type, t.frequency, t.intensity, t.volume
            FROM trainings t
            JOIN horses h ON t.horse_id = h.horse_id
            JOIN events e ON t.event_id = e.event_id
            WHERE h.name = 'Dakota' AND e.event_id = 'Event_SJ_01'
            """
        ).fetchall()
        if dakota_rows:
            for row in dakota_rows:
                print(f"  {row}")
        else:
            print("  (no rows)")

        actor_count = cur.execute("SELECT COUNT(*) FROM training_actors").fetchone()[0]
        print(f"\ntraining_actors: {actor_count}")

        print("\nSample training_actors:")
        for row in cur.execute("SELECT * FROM training_actors LIMIT 3"):
            print(f"  {row}")

        print("\nTraining_Prepa_Tonnerre_01 actors:")
        tonnerre_rows = cur.execute(
            """
            SELECT actor_id, actor_role FROM training_actors
            WHERE training_id = 'Training_Prepa_Tonnerre_01'
            ORDER BY actor_role
            """
        ).fetchall()
        for row in tonnerre_rows:
            print(f"  {row}")

        participation_count = cur.execute(
            "SELECT COUNT(*) FROM event_participations"
        ).fetchone()[0]
        print(f"\nevent_participations: {participation_count}")

        print("\nSample event_participations:")
        for row in cur.execute("SELECT * FROM event_participations LIMIT 3"):
            print(f"  {row}")

        print("\nDakota (Horse1) participation check:")
        dakota_part_rows = cur.execute(
            """
            SELECT ep.event_id, ep.horse_id, ep.rider_id, ep.rank
            FROM event_participations ep
            WHERE ep.horse_id = 'Horse1'
            """
        ).fetchall()
        for row in dakota_part_rows:
            print(f"  {row}")

        sensor_count = cur.execute("SELECT COUNT(*) FROM sensors").fetchone()[0]
        print(f"\nsensors: {sensor_count}")

        print("\nSample sensors:")
        for row in cur.execute("SELECT * FROM sensors LIMIT 3"):
            print(f"  {row}")

        print("\nDakota (Horse1) sensor check:")
        dakota_sensor_rows = cur.execute(
            """
            SELECT sensor_id, sensor_type, objective_id
            FROM sensors WHERE horse_id = 'Horse1'
            ORDER BY sensor_type
            """
        ).fetchall()
        for row in dakota_sensor_rows:
            print(f"  {row}")

        entry_count = cur.execute("SELECT COUNT(*) FROM event_entries").fetchone()[0]
        objective_count = cur.execute("SELECT COUNT(*) FROM objectives").fetchone()[0]
        print(f"\nevent_entries: {entry_count}")
        print(f"objectives: {objective_count}")

        print("\nDakota (Horse1) event entries:")
        dakota_entry_rows = cur.execute(
            """
            SELECT event_id FROM event_entries WHERE horse_id = 'Horse1'
            ORDER BY event_id
            """
        ).fetchall()
        for row in dakota_entry_rows:
            print(f"  {row}")

        print("\nAll objectives:")
        for row in cur.execute("SELECT * FROM objectives"):
            print(f"  {row}")

        entered_no_result = cur.execute(
            """
            SELECT COUNT(*) FROM event_entries ee
            LEFT JOIN event_participations ep
              ON ee.horse_id = ep.horse_id AND ee.event_id = ep.event_id
            WHERE ep.participation_id IS NULL
            """
        ).fetchone()[0]
        print(f"\nentered but no ranked result: {entered_no_result}")

        print("\n" + "=" * 60)
        print("SCHEMA-GAP EXTENSION VERIFICATION")
        print("=" * 60)

        assoc_count = cur.execute(
            "SELECT COUNT(*) FROM horse_rider_associations"
        ).fetchone()[0]
        season_count = cur.execute("SELECT COUNT(*) FROM seasons").fetchone()[0]
        people_count = cur.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        events_with_season = cur.execute(
            "SELECT COUNT(*) FROM events WHERE season_id IS NOT NULL"
        ).fetchone()[0]
        total_events = cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        print(f"\n(a) horse_rider_associations: {assoc_count} (expected 51)")
        print(f"(a) seasons: {season_count} (expected 1)")
        print(f"(a) people: {people_count} (expected 27)")
        print(f"(a) riders: {rider_count} (expected 25)")
        print(f"(a) veterinarians: {veterinarian_count} (expected 1)")
        print(f"(a) caretakers: {caretaker_count} (expected 1)")
        print(
            f"(a) events with non-null season_id: {events_with_season} "
            f"of {total_events} (expected 20 of 20)"
        )

        print("\n(b) seasons table:")
        for row in cur.execute("SELECT * FROM seasons"):
            print(f"  {row}")

        print("\n(c) riders associated with Horse1 (expected Rider_Emma, Rider_Manon):")
        for row in cur.execute(
            "SELECT rider_id FROM horse_rider_associations "
            "WHERE horse_id = 'Horse1' ORDER BY rider_id"
        ):
            print(f"  {row}")

        print(
            "\n(d) horses associated with Rider_Alice (expected Horse_Arrow, "
            "Horse_Braise, Horse_Orage, Horse_Soleil):"
        )
        for row in cur.execute(
            "SELECT horse_id FROM horse_rider_associations "
            "WHERE rider_id = 'Rider_Alice' ORDER BY horse_id"
        ):
            print(f"  {row}")
    finally:
        conn.close()


def main():
    print(f"Connexion a Neo4j: {NEO4J_URI} (database: {NEO4J_DATABASE})")
    (
        horses,
        events,
        trainings,
        training_actors,
        event_participations,
        sensors,
        event_entries,
        objectives,
        horse_rider_associations,
        seasons,
        people,
    ) = fetch_from_neo4j()
    print(
        f"Fetched {len(horses)} horses, {len(events)} events, "
        f"{len(trainings)} trainings, {len(training_actors)} training-actor links, "
        f"{len(event_participations)} event participations, {len(sensors)} sensors, "
        f"{len(event_entries)} event entries, {len(objectives)} objectives, "
        f"{len(horse_rider_associations)} horse-rider associations, "
        f"{len(seasons)} seasons, and {len(people)} people "
        f"from Neo4j."
    )
    build_sqlite(
        horses,
        events,
        trainings,
        training_actors,
        event_participations,
        sensors,
        event_entries,
        objectives,
        horse_rider_associations,
        seasons,
        people,
    )
    print(f"\nSQLite ecrit dans: {DB_PATH}")


if __name__ == "__main__":
    main()
