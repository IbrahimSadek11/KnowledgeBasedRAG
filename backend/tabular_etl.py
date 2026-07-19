"""
Tabular RAG data layer — ETL slice #1.

Builds a SQLite database (data/tabular.db) from two Neo4j node types (Horse and
Event subtypes). Reuses the existing Graph RAG Neo4j connection config from
backend/config.py. Read-only against Neo4j; only writes to the local SQLite file.
"""
import os
import sqlite3

from neo4j import GraphDatabase

from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

# Resolve paths relative to the project root (parent of backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "tabular.db")

EVENT_DISCIPLINES = {"ShowJumping", "Cross", "Dressage"}

TRAINING_STAGES = {
    "PreparationStage",
    "PreCompetitionStage",
    "CompetitionStage",
    "TransitionStage",
}

ACTOR_ROLES = {"Rider", "Veterinarian", "Caretaker"}

SENSOR_TYPES = {"Withers", "Sternum", "CanonOfForelimb", "CanonOfHindlimb"}

HORSE_QUERY = """
MATCH (h:Horse)
RETURN h.id AS horse_id, h.hasName AS name, h.hasRace AS race
"""

EVENT_QUERY = """
MATCH (e)
WHERE e:ShowJumping OR e:Cross OR e:Dressage
RETURN e.id AS event_id, e.eventLocation AS location, e.category AS category,
       e.eventDate AS event_date, labels(e) AS labels
"""

TRAINING_QUERY = """
MATCH (h:Horse)-[:TRAINSIN]->(t)
WHERE t:PreparationStage OR t:PreCompetitionStage
   OR t:CompetitionStage OR t:TransitionStage
OPTIONAL MATCH (t)-[:DEPENDSON]->(e)
RETURN t.id AS training_id, h.id AS horse_id, e.id AS event_id,
       t.Volume AS volume, t.Intensity AS intensity,
       t.Frequency AS frequency, labels(t) AS labels
"""

TRAINING_ACTOR_QUERY = """
MATCH (t)-[:INVOLVESACTOR]->(a)
WHERE t:PreparationStage OR t:PreCompetitionStage
   OR t:CompetitionStage OR t:TransitionStage
RETURN t.id AS training_id, a.id AS actor_id, labels(a) AS labels
"""

PARTICIPATION_QUERY = """
MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
WHERE e:ShowJumping OR e:Cross OR e:Dressage
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
    """Return the domain-specific discipline label, ignoring generic ones."""
    for label in labels:
        if label in EVENT_DISCIPLINES:
            return label
    return None


def pick_stage_type(labels):
    """Return the training-stage label, ignoring generic ones."""
    for label in labels:
        if label in TRAINING_STAGES:
            return label
    return None


def pick_actor_role(labels):
    """Return the actor-role label, ignoring generic ones."""
    for label in labels:
        if label in ACTOR_ROLES:
            return label
    return None


def pick_sensor_type(labels):
    """Return the sensor-position label, ignoring generic ones."""
    for label in labels:
        if label in SENSOR_TYPES:
            return label
    return None


def fetch_from_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            horses = [
                (rec["horse_id"], rec["name"], rec["race"])
                for rec in session.run(HORSE_QUERY)
            ]
            events = [
                (
                    rec["event_id"],
                    rec["location"],
                    rec["category"],
                    date_to_iso(rec["event_date"]),
                    pick_discipline(rec["labels"]),
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
    finally:
        driver.close()
    return (
        horses,
        events,
        trainings,
        training_actors,
        event_participations,
        sensors,
    )


def build_sqlite(
    horses, events, trainings, training_actors, event_participations, sensors
):
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str  # explicit UTF-8 text handling
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS horses")
        cur.execute("DROP TABLE IF EXISTS events")
        cur.execute("DROP TABLE IF EXISTS trainings")
        cur.execute("DROP TABLE IF EXISTS training_actors")
        cur.execute("DROP TABLE IF EXISTS event_participations")
        cur.execute("DROP TABLE IF EXISTS sensors")
        cur.execute(
            """
            CREATE TABLE horses (
                horse_id TEXT PRIMARY KEY,
                name TEXT,
                race TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                location TEXT,
                category TEXT,
                event_date TEXT,
                discipline TEXT
            )
            """
        )
        cur.executemany(
            "INSERT INTO horses (horse_id, name, race) VALUES (?, ?, ?)",
            horses,
        )
        cur.execute(
            """
            CREATE TABLE trainings (
                training_id TEXT PRIMARY KEY,
                horse_id TEXT,
                event_id TEXT,
                stage_type TEXT,
                volume TEXT,
                intensity TEXT,
                frequency INTEGER,
                FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO events (event_id, location, category, event_date, discipline) "
            "VALUES (?, ?, ?, ?, ?)",
            events,
        )
        cur.execute(
            """
            CREATE TABLE training_actors (
                training_id TEXT,
                actor_id TEXT,
                actor_role TEXT,
                PRIMARY KEY (training_id, actor_id),
                FOREIGN KEY (training_id) REFERENCES trainings(training_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO trainings "
            "(training_id, horse_id, event_id, stage_type, volume, intensity, frequency) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            trainings,
        )
        cur.execute(
            """
            CREATE TABLE event_participations (
                participation_id TEXT PRIMARY KEY,
                event_id TEXT,
                horse_id TEXT,
                rider_id TEXT,
                rank INTEGER,
                FOREIGN KEY (event_id) REFERENCES events(event_id),
                FOREIGN KEY (horse_id) REFERENCES horses(horse_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO training_actors (training_id, actor_id, actor_role) "
            "VALUES (?, ?, ?)",
            training_actors,
        )
        cur.execute(
            """
            CREATE TABLE sensors (
                sensor_id TEXT PRIMARY KEY,
                horse_id TEXT,
                sensor_type TEXT,
                sensor_code TEXT,
                format TEXT,
                sensor_offset TEXT,
                file_size INTEGER,
                sample_rate TEXT,
                objective_id TEXT,
                FOREIGN KEY (horse_id) REFERENCES horses(horse_id)
            )
            """
        )
        cur.executemany(
            "INSERT INTO event_participations "
            "(participation_id, event_id, horse_id, rider_id, rank) "
            "VALUES (?, ?, ?, ?, ?)",
            event_participations,
        )
        cur.executemany(
            "INSERT INTO sensors "
            "(sensor_id, horse_id, sensor_type, sensor_code, format, "
            "sensor_offset, file_size, sample_rate, objective_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sensors,
        )
        conn.commit()

        horse_count = cur.execute("SELECT COUNT(*) FROM horses").fetchone()[0]
        event_count = cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        training_count = cur.execute("SELECT COUNT(*) FROM trainings").fetchone()[0]

        print(f"horses: {horse_count}")
        print(f"events: {event_count}")
        print(f"trainings: {training_count}")

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
    ) = fetch_from_neo4j()
    print(
        f"Fetched {len(horses)} horses, {len(events)} events, "
        f"{len(trainings)} trainings, {len(training_actors)} training-actor links, "
        f"{len(event_participations)} event participations, "
        f"and {len(sensors)} sensors from Neo4j."
    )
    build_sqlite(
        horses, events, trainings, training_actors, event_participations, sensors
    )
    print(f"\nSQLite ecrit dans: {DB_PATH}")


if __name__ == "__main__":
    main()
