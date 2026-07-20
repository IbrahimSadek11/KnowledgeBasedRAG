"""
Gold SQL queries for Tabular RAG Execution Accuracy (EX).

EX compares the result set of a generated query against a hand-authored gold
query, independently of how the natural-language answer was phrased.
"""
from __future__ import annotations

import sqlite3


GOLD_QUERIES = {
    "Q1": "SELECT DISTINCT name FROM horses;",
    "Q2": "SELECT race FROM horses WHERE name = 'Dakota';",
    "Q3": "SELECT race FROM horses WHERE name = 'Naya';",
    "Q4": "SELECT event_id FROM event_entries WHERE horse_id = 'Horse1';",
    "Q5": "SELECT training_id FROM trainings WHERE horse_id = 'Horse1';",
    "Q6": "SELECT frequency, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type = 'PreparationStage' GROUP BY frequency;",
    "Q7": "SELECT DISTINCT intensity, stage_type FROM trainings WHERE stage_type = 'PreCompetitionStage';",
    "Q8": "SELECT volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type = 'PreparationStage' GROUP BY volume;",
    "Q9": "SELECT volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type = 'PreCompetitionStage' GROUP BY volume;",
    "Q10": "SELECT DISTINCT e.event_id, e.location, e.category, e.event_date, e.discipline, t.stage_type FROM trainings t JOIN events e ON t.event_id = e.event_id WHERE t.horse_id = 'Horse1';",
    "Q11": "SELECT COUNT(*) FROM sensors WHERE horse_id = 'Horse1';",
    "Q12": "SELECT COUNT(DISTINCT sensor_id) AS total_sensors, sensor_type, COUNT(sensor_id) AS count_per_type FROM sensors GROUP BY sensor_type;",
    "Q13": "SELECT DISTINCT sensor_type FROM sensors WHERE horse_id = 'Horse1';",
    "Q14": "SELECT COUNT(DISTINCT sensor_id) AS total_sensors, GROUP_CONCAT(sensor_id) AS example_ids FROM sensors WHERE LOWER(sensor_type) = LOWER('Withers');",
    "Q15": "SELECT DISTINCT sample_rate FROM sensors WHERE sensor_type = 'Sternum';",
    "Q16": "SELECT sensor_id FROM sensors WHERE CAST(REPLACE(sample_rate,'Hz','') AS INTEGER) = (SELECT MAX(CAST(REPLACE(sample_rate,'Hz','') AS INTEGER)) FROM sensors);",
    "Q17": "SELECT rider_id FROM horse_rider_associations WHERE horse_id = 'Horse1';",
    "Q18": "SELECT rider_id FROM horse_rider_associations WHERE horse_id = 'Horse2';",
    "Q19": "SELECT COUNT(*) FROM people WHERE role = 'Rider';",
    "Q20": "SELECT person_id FROM people WHERE role = 'Veterinarian';",
    "Q21": "SELECT person_id FROM people WHERE role = 'Caretaker';",
    "Q22": "SELECT DISTINCT p.person_id, p.role FROM trainings t JOIN training_actors ta ON t.training_id = ta.training_id JOIN people p ON ta.actor_id = p.person_id WHERE LOWER(t.stage_type) = LOWER('PreparationStage');",
    "Q23": "SELECT DISTINCT p.person_id, p.role, t.stage_type FROM trainings t JOIN training_actors ta ON t.training_id = ta.training_id JOIN people p ON ta.actor_id = p.person_id WHERE LOWER(t.stage_type) = LOWER('PreCompetitionStage');",
    "Q24": "SELECT season_start, season_end FROM seasons WHERE season_id = 'Season_2026';",
    "Q25": "SELECT DISTINCT event_id, location, category, event_date, discipline FROM events WHERE LOWER(season_id) = LOWER('Season_2026');",
    "Q26": "SELECT event_date FROM events WHERE event_id = 'Event_SJ_01';",
    "Q27": "SELECT location FROM events WHERE event_id = 'Event_SJ_01';",
    "Q28": "SELECT location FROM events WHERE event_id = 'Event_Dressage_01';",
    "Q29": "SELECT category FROM events WHERE event_id = 'Event_SJ_01';",
    "Q30": "SELECT category FROM events WHERE event_id = 'Event_Dressage_01';",
    "Q31": "SELECT rider_id, rank FROM event_participations WHERE horse_id = 'Horse1' AND event_id = 'Event_SJ_01';",
    "Q32": "SELECT stage_type, frequency, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type IN ('PreparationStage','PreCompetitionStage') GROUP BY stage_type, frequency;",
    "Q33": "SELECT stage_type, frequency, intensity, volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type IN ('PreparationStage','PreCompetitionStage') GROUP BY stage_type, frequency, intensity, volume;",
    "Q34": "SELECT DISTINCT ta.actor_id, ta.actor_role, t.stage_type FROM trainings t JOIN training_actors ta ON t.training_id = ta.training_id WHERE t.stage_type IN ('PreparationStage', 'PreCompetitionStage') ORDER BY ta.actor_id, t.stage_type;",
    "Q36": "SELECT COUNT(*) FROM events;",
    "Q37": "SELECT DISTINCT discipline FROM events;",
    "Q38": "SELECT o.name, o.description FROM sensors s JOIN objectives o ON s.objective_id = o.objective_id WHERE s.sensor_id = 'IMU_Withers_01';",
    "Q39": "SELECT o.name, o.description FROM sensors s JOIN objectives o ON s.objective_id = o.objective_id WHERE s.sensor_id = 'IMU_CanonFore_01';",
    "Q41": "SELECT race, COUNT(*) AS cnt FROM horses GROUP BY race ORDER BY cnt DESC LIMIT 1;",
    "Q42": "SELECT race, COUNT(*) AS cnt FROM horses GROUP BY race HAVING COUNT(*) = 1;",
    "Q43": "SELECT rider_id, COUNT(DISTINCT horse_id) AS horse_count FROM horse_rider_associations GROUP BY rider_id;",
    "Q44": "SELECT horse_id FROM horse_rider_associations GROUP BY horse_id HAVING COUNT(DISTINCT rider_id) > 1;",
    "Q45": "SELECT rider_id, COUNT(DISTINCT horse_id) AS horse_count FROM horse_rider_associations GROUP BY rider_id ORDER BY horse_count DESC LIMIT 1;",
    "Q46": "SELECT sensor_type, COUNT(*) FROM sensors GROUP BY sensor_type;",
    "Q47": "SELECT horse_id, COUNT(*) AS sensor_count FROM sensors GROUP BY horse_id;",
    "Q48": "SELECT objective_id, COUNT(*) FROM sensors GROUP BY objective_id;",
    "Q49": "SELECT DISTINCT format FROM sensors;",
    "Q50": "SELECT sensor_id, horse_id, sensor_type, sample_rate FROM sensors WHERE CAST(REPLACE(sample_rate,'Hz','') AS INTEGER) = (SELECT MAX(CAST(REPLACE(sample_rate,'Hz','') AS INTEGER)) FROM sensors);",
    "Q51": "SELECT sensor_type, sensor_offset, COUNT(*) FROM sensors GROUP BY sensor_type, sensor_offset;",
    "Q52": "SELECT stage_type, COUNT(*) FROM trainings GROUP BY stage_type;",
    "Q53": "SELECT horse_id, COUNT(*) AS stage_count FROM trainings GROUP BY horse_id ORDER BY stage_count DESC LIMIT 1;",
    "Q54": "SELECT DISTINCT volume, intensity, frequency FROM trainings WHERE stage_type = 'CompetitionStage';",
    "Q55": "SELECT volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type = 'TransitionStage' GROUP BY volume;",
    "Q56": "SELECT t.stage_type, ta.actor_role, COUNT(DISTINCT t.training_id) AS cnt FROM trainings t JOIN training_actors ta ON t.training_id = ta.training_id WHERE ta.actor_role IN ('Veterinarian','Caretaker') GROUP BY t.stage_type, ta.actor_role;",
    "Q57": "SELECT training_id FROM trainings WHERE training_id NOT IN (SELECT DISTINCT training_id FROM training_actors);",
    "Q58": "SELECT discipline, COUNT(*) FROM events GROUP BY discipline;",
    "Q59": "SELECT COUNT(DISTINCT event_id) AS competition_count, strftime('%Y-%m', event_date) AS competition_month FROM events WHERE event_date BETWEEN (SELECT season_start FROM seasons WHERE season_id = 'Season_2026') AND (SELECT season_end FROM seasons WHERE season_id = 'Season_2026') GROUP BY competition_month ORDER BY competition_count DESC LIMIT 1;",
    "Q60": "SELECT location, COUNT(*) FROM events GROUP BY location HAVING COUNT(*) > 1;",
    "Q61": "SELECT discipline, COUNT(*) FROM events WHERE category = 'Pro Elite' GROUP BY discipline;",
    "Q62": "SELECT DISTINCT e.event_id, e.location, e.category, e.event_date, e.discipline FROM events e LEFT JOIN event_entries ee ON e.event_id = ee.event_id LEFT JOIN event_participations ep ON e.event_id = ep.event_id WHERE ep.participation_id IS NULL;",
    "Q63": "SELECT event_id, COUNT(*) AS cnt FROM event_participations GROUP BY event_id ORDER BY cnt DESC LIMIT 1;",
    "Q64": "SELECT horse_id, COUNT(DISTINCT event_id) AS event_count FROM event_entries GROUP BY horse_id;",
    "Q65": "SELECT COUNT(DISTINCT e.horse_id) AS unranked, (SELECT COUNT(DISTINCT horse_id) FROM event_entries) AS total FROM event_entries e LEFT JOIN event_participations p ON e.horse_id = p.horse_id AND e.event_id = p.event_id WHERE p.participation_id IS NULL;",
    "Q66": "SELECT event_id, rider_id, horse_id FROM event_participations WHERE (event_id, rider_id) IN (SELECT event_id, rider_id FROM event_participations GROUP BY event_id, rider_id HAVING COUNT(DISTINCT horse_id) > 1);",
    "Q67": "SELECT ee.event_id, COUNT(DISTINCT ee.horse_id) - COALESCE((SELECT COUNT(*) FROM event_participations ep WHERE ep.event_id = ee.event_id), 0) AS gap FROM event_entries ee GROUP BY ee.event_id ORDER BY gap DESC LIMIT 1;",
    "Q74": "SELECT COUNT(*) FROM events WHERE event_date NOT GLOB '____-__-__';",
    "Q75": "SELECT DISTINCT intensity FROM trainings;",
    "Q76": "SELECT DISTINCT p.person_id, p.role FROM training_actors ta JOIN people p ON ta.actor_id = p.person_id WHERE LOWER(ta.actor_role) <> LOWER('Rider');",
    "Q77": "SELECT DISTINCT discipline FROM events;",
    "Q78": "SELECT category, COUNT(*) FROM events GROUP BY category;",
    "Q79": "SELECT DISTINCT discipline FROM events WHERE category = 'Club Elite';",
    "Q80": "SELECT event_id, event_date FROM events WHERE event_date = (SELECT MIN(event_date) FROM events) OR event_date = (SELECT MAX(event_date) FROM events);",
    "Q81": "SELECT objective_id, name, description FROM objectives;",
    "Q82": "SELECT sensor_id, objective_id FROM sensors WHERE horse_id = 'Horse1';",
    "Q83": "SELECT horse_id, stage_type, volume FROM trainings WHERE stage_type = 'PreCompetitionStage' AND volume = (SELECT MAX(volume) FROM trainings WHERE stage_type = 'PreCompetitionStage');",
    "Q84": "SELECT horse_id, volume FROM trainings WHERE stage_type = 'PreparationStage' AND volume = (SELECT MAX(volume) FROM trainings WHERE stage_type = 'PreparationStage');",
    "Q85": "SELECT horse_id, stage_type, volume FROM trainings WHERE (stage_type = 'PreparationStage' AND volume = (SELECT MIN(volume) FROM trainings WHERE stage_type = 'PreparationStage')) OR (stage_type = 'PreCompetitionStage' AND volume = (SELECT MIN(volume) FROM trainings WHERE stage_type = 'PreCompetitionStage'));",
    "Q86": "SELECT t.horse_id, COUNT(*) AS stage_count, (SELECT COUNT(DISTINCT event_id) FROM event_entries ee WHERE ee.horse_id = t.horse_id) AS competition_count FROM trainings t GROUP BY t.horse_id;",
    "Q87": "SELECT COUNT(DISTINCT sensors.sensor_id) AS count_attached, COUNT(DISTINCT sensors.sensor_id) AS total_count FROM sensors LEFT JOIN horses ON sensors.horse_id = horses.horse_id LEFT JOIN objectives ON sensors.objective_id = objectives.objective_id WHERE horses.horse_id IS NOT NULL AND objectives.objective_id IS NOT NULL;",
    "Q90": "SELECT DISTINCT training_id, horse_id, volume FROM trainings WHERE stage_type = 'TransitionStage';",
    "Q91": "SELECT ee.event_id, ep.rank FROM event_entries ee LEFT JOIN event_participations ep ON ee.event_id = ep.event_id AND ee.horse_id = ep.horse_id WHERE ee.horse_id = 'Horse_Auroch';",
    "Q92": "SELECT sensor_count, COUNT(*) AS num_horses FROM (SELECT horse_id, COUNT(*) AS sensor_count FROM sensors GROUP BY horse_id) GROUP BY sensor_count;",
    "Q93": "SELECT t.horse_id, COUNT(*) AS stage_count, (SELECT COUNT(DISTINCT event_id) FROM event_entries ee WHERE ee.horse_id = t.horse_id) AS competition_count FROM trainings t GROUP BY t.horse_id;",
    "Q95": "SELECT discipline, category, COUNT(*) FROM events GROUP BY discipline, category;",
    "Q96": "SELECT event_id, rider_id FROM event_participations WHERE rank IN (1,2) GROUP BY event_id, rider_id HAVING COUNT(DISTINCT rank) = 2;",
    "Q97": "SELECT COUNT(DISTINCT horse_id) AS horses_with_result, (SELECT COUNT(*) FROM horses) AS total_horses FROM event_participations;",
    "Q98": "SELECT COUNT(*) FROM event_entries ee LEFT JOIN event_participations ep ON ee.horse_id = ep.horse_id AND ee.event_id = ep.event_id WHERE ep.participation_id IS NULL;",
    "Q99": "SELECT (SELECT COUNT(*) FROM sensors WHERE horse_id IS NOT NULL) AS horse_links, (SELECT COUNT(*) FROM sensors WHERE objective_id IS NOT NULL) AS objective_links;",
    "Q100": "SELECT (SELECT COUNT(*) FROM horses) AS horses, (SELECT COUNT(*) FROM people WHERE role='Rider') AS riders, (SELECT COUNT(*) FROM events) AS events, (SELECT COUNT(*) FROM sensors) AS sensors;",
}

EX_NOT_APPLICABLE = {
    "Q35": "multi_hop_complex/extreme 'complete analysis' narrative spanning many tables — no single natural result-set shape represents it",
    "Q40": "unanswerable category — no age data exists, correct behavior is declining",
    "Q68": "unanswerable category — no color/weight/phone data exists",
    "Q69": "asks about graph relationship-type vocabulary as named entities, not representable as a SQL result set",
    "Q70": "conceptual schema-explanation question (how would you find X), not a lookup against a specific record",
    "Q71": "conceptual schema-explanation question about how HAS_PARTICIPATION/HAS_HORSE/HAS_RIDER relate",
    "Q72": "conceptual schema-explanation question, same nature as Q70/Q71",
    "Q73": "conceptual schema-explanation question about sensor-horse-objective relationships",
    "Q88": "asks to explain the conceptual DIFFERENCE between two relationship types, not a data lookup",
    "Q89": "conceptual/historical question about a since-removed relationship, not a data lookup",
    "Q94": "detecting naming-convention anomalies has no clean general SQL expression under this schema; known irregular ids are enumerable only by hand",
}


def execute_and_normalize(sql: str, db_path: str) -> list[str]:
    """Run `sql` read-only against tabular.db; return order-independent rows.

    Each row is stringified as a positional tuple; the list of rows is sorted
    so that SELECT row-order differences do not affect comparison. Column
    order within a row is preserved (positional).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    normalized = [str(tuple(row)) for row in rows]
    normalized.sort()
    return normalized


def compare_execution(
    generated_sql: str | None,
    gold_sql: str,
    db_path: str,
) -> tuple[bool, str | None]:
    """Compare normalized result sets of generated vs gold SQL.

    Returns (matched, error_text). error_text is set when either query fails
    to execute (matched is False in that case); otherwise error_text is None.
    """
    if not generated_sql:
        return False, "generated SQL is None or empty"

    try:
        generated_norm = execute_and_normalize(generated_sql, db_path)
    except Exception as exc:  # noqa: BLE001 - surface to caller as EX miss
        return False, f"generated SQL exception: {exc}"

    try:
        gold_norm = execute_and_normalize(gold_sql, db_path)
    except Exception as exc:  # noqa: BLE001 - surface to caller as EX miss
        return False, f"gold SQL exception: {exc}"

    return generated_norm == gold_norm, None


if __name__ == "__main__":
    import os

    _here = os.path.dirname(os.path.abspath(__file__))
    _db = os.path.join(_here, "..", "..", "data", "tabular.db")

    a = execute_and_normalize(
        "SELECT * FROM (SELECT 1 AS n UNION ALL SELECT 2 AS n) ORDER BY n ASC",
        _db,
    )
    b = execute_and_normalize(
        "SELECT * FROM (SELECT 1 AS n UNION ALL SELECT 2 AS n) ORDER BY n DESC",
        _db,
    )
    print(f"normalized A (rows 1 then 2): {a}")
    print(f"normalized B (rows 2 then 1): {b}")
    print(f"equal: {a == b}")
    if a != b:
        raise SystemExit("FAIL: row order still affects comparison")
    print("PASS: row order does not affect comparison")
