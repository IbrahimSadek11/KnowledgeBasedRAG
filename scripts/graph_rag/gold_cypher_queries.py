"""
Gold Cypher queries + Neo4j result-set comparator for Graph RAG Execution Accuracy (EX).

GOLD_CYPHER_QUERIES entries are accepted only after live V9 Neo4j verification
(or after reusing a Cypher+result already recorded in
docs/graph_rag/graph_gt_reconciliation_signoff.md).
"""
from __future__ import annotations

from typing import Any, Literal

ListCompareMode = Literal["multiset", "set"]

# Batch 1 (Q1–Q20) — verified 2026-07-26 against live Neo4j (bolt://127.0.0.1:7687).
GOLD_CYPHER_QUERIES: dict[str, str] = {
    "Q1": "MATCH (h:Horse) RETURN DISTINCT h.hasName AS name",
    "Q2": "MATCH (h:Horse {hasName: 'Dakota'}) RETURN h.hasRace AS race",
    "Q3": "MATCH (h:Horse {hasName: 'Naya'}) RETURN h.hasRace AS race",
    "Q4": (
        "MATCH (h:Horse {hasName: 'Dakota'})-[:COMPETESIN]->(e) "
        "RETURN DISTINCT e.id AS event_id"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q5)
    "Q5": (
        "MATCH (h:Horse {hasName:'Dakota'})-[:TRAINSIN]->(t) "
        "RETURN COUNT(DISTINCT t) AS n, COLLECT(DISTINCT t.id) AS ids"
    ),
    "Q6": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage) "
        "RETURN t.Frequency AS frequency, COUNT(DISTINCT h) AS horse_count, "
        "COLLECT(DISTINCT h.hasName) AS horses"
    ),
    "Q7": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreCompetitionStage) "
        "RETURN DISTINCT t.Intensity AS intensity"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q8)
    "Q8": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage) "
        "RETURN t.Volume AS volume, COUNT(DISTINCT h) AS horses, "
        "COUNT(DISTINCT t) AS stages, COLLECT(DISTINCT h.hasName) AS names"
    ),
    "Q9": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreCompetitionStage) "
        "RETURN t.Volume AS volume, COUNT(DISTINCT h) AS horses, "
        "COLLECT(DISTINCT h.hasName) AS names"
    ),
    "Q10": (
        "MATCH (h:Horse {hasName:'Dakota'})-[:TRAINSIN]->(t)-[:DEPENDSON]->(e) "
        "RETURN COUNT(DISTINCT e) AS n, COLLECT(DISTINCT e.id) AS ids"
    ),
    "Q11": (
        "MATCH (h:Horse {hasName: 'Dakota'})<-[:ISATTACHEDTO]-(s:InertialSensors) "
        "RETURN COUNT(DISTINCT s) AS sensor_count"
    ),
    "Q12": (
        "MATCH (s:InertialSensors) "
        "RETURN labels(s)[1] AS sensor_type, COUNT(s) AS count"
    ),
    "Q13": (
        "MATCH (h:Horse {hasName: 'Dakota'})<-[:ISATTACHEDTO]-(s:InertialSensors) "
        "RETURN DISTINCT labels(s)[1] AS position"
    ),
    "Q14": (
        "MATCH (s:Withers) "
        "RETURN COUNT(s) AS withers_count, COLLECT(s.id) AS ids"
    ),
    "Q15": (
        "MATCH (s:Sternum) "
        "RETURN DISTINCT s.hasSensorTime AS sample_rate"
    ),
    "Q16": (
        "MATCH (s:InertialSensors) "
        "WITH s, toInteger(replace(toString(s.hasSensorTime), 'Hz', '')) AS hz "
        "WITH max(hz) AS max_hz "
        "MATCH (s2:InertialSensors) "
        "WHERE toInteger(replace(toString(s2.hasSensorTime), 'Hz', '')) = max_hz "
        "RETURN s2.id AS sensor_id"
    ),
    "Q17": (
        "MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse {hasName: 'Dakota'}) "
        "RETURN DISTINCT r.id AS rider_id"
    ),
    "Q18": (
        "MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse {hasName: 'Naya'}) "
        "RETURN DISTINCT r.id AS rider_id"
    ),
    "Q19": "MATCH (r:Rider) RETURN COUNT(r) AS rider_count",
    "Q20": "MATCH (v:Veterinarian) RETURN v.id AS veterinarian_id",
    # Batch 2 (Q21–Q40) — verified 2026-07-26 against live Neo4j.
    "Q21": "MATCH (c:Caretaker) RETURN c.id AS caretaker_id",
    "Q22": (
        "MATCH (t:PreparationStage)-[:INVOLVESACTOR]->(a) "
        "WHERE a:Rider OR a:Veterinarian OR a:Caretaker "
        "RETURN DISTINCT a.id AS actor_id, labels(a)[0] AS role"
    ),
    "Q23": (
        "MATCH (t:PreCompetitionStage)-[:INVOLVESACTOR]->(a) "
        "WHERE a:Rider OR a:Veterinarian OR a:Caretaker "
        "RETURN DISTINCT a.id AS actor_id, labels(a)[0] AS role"
    ),
    "Q24": (
        "MATCH (s:CompetitiveSeason) "
        "WHERE s.seasonName = 'Saison 2026' OR s.id = 'Season_2026' "
        "RETURN s.seasonStart AS season_start, s.seasonEnd AS season_end"
    ),
    "Q25": (
        "MATCH (e)-[:INSEASON]->(s:CompetitiveSeason) "
        "WHERE s.seasonName = 'Saison 2026' OR s.id = 'Season_2026' "
        "RETURN DISTINCT e.id AS event_id"
    ),
    "Q26": (
        "MATCH (e) WHERE e.id = 'Event_SJ_01' "
        "RETURN e.eventDate AS event_date"
    ),
    "Q27": (
        "MATCH (e) WHERE e.id = 'Event_SJ_01' "
        "RETURN e.eventLocation AS location"
    ),
    "Q28": (
        "MATCH (e) WHERE e.id = 'Event_Dressage_01' "
        "RETURN e.eventLocation AS location"
    ),
    "Q29": (
        "MATCH (e) WHERE e.id = 'Event_SJ_01' "
        "RETURN e.category AS category"
    ),
    "Q30": (
        "MATCH (e) WHERE e.id = 'Event_Dressage_01' "
        "RETURN e.category AS category"
    ),
    "Q31": (
        "MATCH (e) WHERE e.id = 'Event_SJ_01' "
        "MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)"
        "-[:HASHORSE]->(h:Horse {hasName: 'Dakota'}) "
        "MATCH (p)-[:HASRIDER]->(r:Rider) "
        "RETURN r.id AS rider_id, p.rank AS rank"
    ),
    "Q32": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t) "
        "WHERE t:PreparationStage OR t:PreCompetitionStage "
        "RETURN labels(t)[0] AS stage_type, t.Frequency AS frequency, "
        "COUNT(DISTINCT h) AS horse_count, COLLECT(DISTINCT h.hasName) AS horses"
    ),
    "Q33": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t) "
        "WHERE t:PreparationStage OR t:PreCompetitionStage "
        "RETURN labels(t)[0] AS stage_type, t.Frequency AS frequency, "
        "t.Intensity AS intensity, t.Volume AS volume, "
        "COUNT(DISTINCT h) AS horse_count"
    ),
    "Q34": (
        "MATCH (t)-[:INVOLVESACTOR]->(a) "
        "WHERE (t:PreparationStage OR t:PreCompetitionStage) "
        "AND (a:Rider OR a:Veterinarian OR a:Caretaker) "
        "RETURN DISTINCT labels(t)[0] AS phase, labels(a)[0] AS role, a.id AS actor_id"
    ),
    "Q36": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "RETURN COUNT(e) AS event_count"
    ),
    "Q37": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "RETURN DISTINCT labels(e)[0] AS event_type"
    ),
    # Batch 3 (Q41–Q60) — verified 2026-07-26 against live Neo4j.
    "Q41": (
        "MATCH (h:Horse) "
        "WITH h.hasRace AS race, COUNT(h) AS n, COLLECT(h.hasName) AS names "
        "WITH collect({race:race, n:n, names:names}) AS rows "
        "WITH rows, reduce(m=0, r IN rows | CASE WHEN r.n > m THEN r.n ELSE m END) AS max_n "
        "UNWIND rows AS row "
        "WITH row WHERE row.n = max_n "
        "RETURN row.race AS race, row.n AS horse_count, row.names AS names"
    ),
    "Q42": (
        "MATCH (h:Horse) "
        "WITH h.hasRace AS race, COUNT(h) AS n, COLLECT(h.hasName) AS names "
        "WHERE n = 1 "
        "RETURN race, names"
    ),
    "Q43": (
        "MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse) "
        "WITH r, COUNT(DISTINCT h) AS horse_count "
        "RETURN horse_count, COUNT(r) AS riders, COLLECT(r.id) AS rider_ids"
    ),
    "Q44": (
        "MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse) "
        "WITH h, COUNT(DISTINCT r) AS rider_count, COLLECT(DISTINCT r.id) AS riders "
        "WHERE rider_count > 1 "
        "RETURN h.hasName AS horse, rider_count, riders"
    ),
    "Q45": (
        "MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse) "
        "WITH r, COUNT(DISTINCT h) AS n, COLLECT(DISTINCT h.hasName) AS horses "
        "WITH max(n) AS max_n "
        "MATCH (r2:Rider)-[:ASSOCIATEDWITH]->(h2:Horse) "
        "WITH r2, COUNT(DISTINCT h2) AS n, COLLECT(DISTINCT h2.hasName) AS horses, max_n "
        "WHERE n = max_n "
        "RETURN r2.id AS rider_id, n AS horse_count, horses"
    ),
    "Q46": (
        "MATCH (s:InertialSensors) "
        "RETURN labels(s)[1] AS position, COUNT(s) AS count"
    ),
    "Q47": (
        "MATCH (h:Horse)<-[:ISATTACHEDTO]-(s:InertialSensors) "
        "WITH h, COUNT(s) AS sensor_count "
        "RETURN sensor_count, COUNT(h) AS horses, COLLECT(h.hasName) AS names"
    ),
    "Q48": (
        "MATCH (s:InertialSensors)-[:ISUSEDFOR]->(o) "
        "RETURN o.id AS objective, labels(s)[1] AS position, COUNT(s) AS count"
    ),
    "Q49": (
        "MATCH (s:InertialSensors) "
        "RETURN s.hasFormat AS format, COUNT(s) AS count"
    ),
    "Q50": (
        "MATCH (s:InertialSensors) "
        "WITH s, toInteger(replace(toString(s.hasSensorTime), 'Hz', '')) AS hz "
        "WITH max(hz) AS max_hz "
        "MATCH (s2:InertialSensors) "
        "WHERE toInteger(replace(toString(s2.hasSensorTime), 'Hz', '')) = max_hz "
        "RETURN s2.id AS sensor_id, s2.hasSensorTime AS sample_rate"
    ),
    "Q51": (
        "MATCH (s:InertialSensors) "
        "RETURN labels(s)[1] AS position, s.hasSensorOffset AS offset, COUNT(s) AS count"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q52)
    "Q52": (
        "MATCH (t) "
        "WHERE t:PreparationStage OR t:PreCompetitionStage "
        "OR t:CompetitionStage OR t:TransitionStage "
        "RETURN labels(t)[0] AS stage_type, COUNT(t) AS n"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q53)
    "Q53": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t) "
        "WITH h, COUNT(DISTINCT t) AS n "
        "RETURN n AS stages, COUNT(h) AS horses, COLLECT(h.hasName) AS names"
    ),
    "Q54": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:CompetitionStage) "
        "RETURN DISTINCT t.Volume AS volume, t.Intensity AS intensity, "
        "t.Frequency AS frequency"
    ),
    # Signoff Q55 shape; full name COLLECT (not truncated sample)
    "Q55": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:TransitionStage) "
        "RETURN t.Volume AS volume, COUNT(DISTINCT h) AS horses, "
        "COLLECT(DISTINCT h.hasName) AS names"
    ),
    "Q56": (
        "MATCH (t)-[:INVOLVESACTOR]->(a) "
        "WHERE (t:PreparationStage OR t:PreCompetitionStage "
        "OR t:CompetitionStage OR t:TransitionStage) "
        "AND (a:Veterinarian OR a:Caretaker) "
        "RETURN labels(a)[0] AS role, labels(t)[0] AS phase, "
        "COUNT(DISTINCT t) AS stage_count"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q57)
    "Q57": (
        "MATCH (t) "
        "WHERE t:PreparationStage OR t:PreCompetitionStage "
        "OR t:CompetitionStage OR t:TransitionStage "
        "OPTIONAL MATCH (t)-[:INVOLVESACTOR]->(a) "
        "WHERE a:Rider OR a:Veterinarian OR a:Caretaker "
        "WITH t, COUNT(a) AS supervisors "
        "WHERE supervisors = 0 "
        "RETURN labels(t)[0] AS type, t.id AS stage, supervisors"
    ),
    "Q58": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "RETURN labels(e)[0] AS discipline, COUNT(e) AS count"
    ),
    "Q59": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "WITH substring(toString(e.eventDate), 0, 7) AS ym, "
        "COUNT(e) AS n, COLLECT(e.id) AS ids "
        "WITH collect({ym:ym, n:n, ids:ids}) AS rows "
        "WITH rows, reduce(m=0, r IN rows | CASE WHEN r.n > m THEN r.n ELSE m END) AS max_n "
        "UNWIND rows AS row "
        "WITH row WHERE row.n = max_n "
        "RETURN row.ym AS month, row.n AS event_count, row.ids AS ids"
    ),
    "Q60": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "WITH e.eventLocation AS location, COUNT(e) AS n, COLLECT(e.id) AS ids "
        "WHERE n > 1 "
        "RETURN location, n AS event_count, ids"
    ),
    # Batch 4 (Q61–Q80) — verified 2026-07-26 against live Neo4j.
    "Q61": (
        "MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross) "
        "AND e.category = 'Pro Elite' "
        "RETURN labels(e)[0] AS discipline, e.id AS event_id"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q62)
    "Q62": (
        "MATCH (h:Horse)-[:COMPETESIN]->(e) "
        "OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation) "
        "WITH e, COUNT(DISTINCT h) AS entrants, COUNT(DISTINCT p) AS ranked "
        "WHERE ranked = 0 "
        "RETURN e.id AS event, entrants, ranked "
        "ORDER BY entrants DESC"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q63)
    "Q63": (
        "MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation) "
        "MATCH (e)-[:INSEASON]->(:CompetitiveSeason {seasonName:'Saison 2026'}) "
        "WITH e, COUNT(DISTINCT p) AS result_count "
        "RETURN e.id AS event, result_count "
        "ORDER BY result_count DESC, event "
        "LIMIT 10"
    ),
    "Q64": (
        "MATCH (h:Horse)-[:COMPETESIN]->(e) "
        "WITH h, COUNT(DISTINCT e) AS event_count "
        "RETURN event_count, COUNT(h) AS horses, COLLECT(h.hasName) AS names"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q65)
    "Q65": (
        "MATCH (h:Horse)-[:COMPETESIN]->(e) "
        "OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h) "
        "WITH h, e, COUNT(p) AS ranked "
        "WHERE ranked = 0 "
        "RETURN h.hasName AS horse, e.id AS event, ranked "
        "ORDER BY horse"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q66)
    "Q66": (
        "MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASRIDER]->(r:Rider) "
        "MATCH (p)-[:HASHORSE]->(h:Horse) "
        "WITH e, r, COUNT(DISTINCT h) AS horse_count, "
        "COLLECT(DISTINCT h.hasName) AS horses "
        "WHERE horse_count > 1 "
        "RETURN e.id AS event, r.id AS rider, horse_count, horses "
        "ORDER BY event, rider"
    ),
    # Exact primary Cypher from signoff (Q67): unranked-horse check → empty.
    # Signoff also records a second confirmatory query
    #   MATCH (e {id:'Event_SJ_2026_01'})-[:HASPARTICIPATION]->(p)
    #   RETURN e.id AS event, COUNT(p) AS participations  → 7
    # that was NOT dual-locked as EX gold. If a future Q67 MISMATCH looks
    # surprising, check whether the generated Cypher took that
    # participation-count route (non-empty 1-row result) rather than the
    # ranked-count / unranked-filter route before treating it as a real bug.
    "Q67": (
        "MATCH (h:Horse)-[:COMPETESIN]->(e {id:'Event_SJ_2026_01'}) "
        "OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h) "
        "WITH h, COUNT(p) AS ranked "
        "WHERE ranked = 0 "
        "RETURN h.hasName AS horse, ranked"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q69)
    "Q69": (
        "MATCH ()-[r]->() "
        "RETURN type(r) AS relationship, COUNT(r) AS n "
        "ORDER BY n DESC"
    ),
    "Q74": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "WITH e.eventDate AS d "
        "WHERE d IS NULL OR NOT d =~ '\\\\d{4}-\\\\d{2}-\\\\d{2}' "
        "RETURN d AS bad_date"
    ),
    "Q75": (
        "MATCH (t) "
        "WHERE t:PreparationStage OR t:PreCompetitionStage "
        "OR t:CompetitionStage OR t:TransitionStage "
        "RETURN DISTINCT t.Intensity AS intensity"
    ),
    "Q76": (
        "MATCH (t)-[:INVOLVESACTOR]->(a) "
        "WHERE a:Veterinarian OR a:Caretaker "
        "RETURN DISTINCT labels(a)[0] AS role"
    ),
    "Q77": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "RETURN DISTINCT labels(e)[0] AS discipline"
    ),
    "Q78": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "RETURN e.category AS category, COUNT(e) AS count"
    ),
    "Q79": (
        "MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross) "
        "AND e.category = 'Club Elite' "
        "RETURN DISTINCT labels(e)[0] AS discipline, COUNT(e) AS count"
    ),
    "Q80": (
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "WITH min(e.eventDate) AS first_d, max(e.eventDate) AS last_d "
        "MATCH (e2) WHERE e2.eventDate IN [first_d, last_d] "
        "AND (e2:ShowJumping OR e2:Dressage OR e2:Cross) "
        "RETURN e2.id AS event_id, e2.eventDate AS event_date"
    ),
    # Batch 5 (Q81–Q100) — verified 2026-07-26 against live Neo4j.
    "Q81": (
        "MATCH (o:ExperimentalObjective) "
        "RETURN o.id AS objective_id, o.hasName AS name, o.description AS description"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q82)
    "Q82": (
        "MATCH (h:Horse {hasName:'Dakota'})<-[:ISATTACHEDTO]-(s)-[:ISUSEDFOR]->(o) "
        "RETURN s.id AS sensor, labels(s) AS sensor_labels, o.id AS objective "
        "ORDER BY sensor"
    ),
    "Q83": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreCompetitionStage) "
        "WITH max(t.Volume) AS max_v "
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreCompetitionStage) "
        "WHERE t.Volume = max_v "
        "RETURN h.hasName AS horse, t.Volume AS volume"
    ),
    "Q84": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage) "
        "WITH max(t.Volume) AS max_v "
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage) "
        "WHERE t.Volume = max_v "
        "RETURN DISTINCT h.hasName AS horse, t.Volume AS volume"
    ),
    "Q85": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage) "
        "WITH min(t.Volume) AS min_v "
        "MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage) "
        "WHERE t.Volume = min_v "
        "RETURN DISTINCT h.hasName AS horse, t.Volume AS volume"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q86)
    "Q86": (
        "MATCH (h:Horse {hasName:'Dakota'}) "
        "OPTIONAL MATCH (h)-[:COMPETESIN]->(e) "
        "OPTIONAL MATCH (h)-[:TRAINSIN]->(t) "
        "RETURN COUNT(DISTINCT e) AS comps, COUNT(DISTINCT t) AS stages"
    ),
    "Q87": (
        "MATCH (s:InertialSensors) "
        "OPTIONAL MATCH (s)-[:ISATTACHEDTO]->(h:Horse) "
        "OPTIONAL MATCH (s)-[:ISUSEDFOR]->(o) "
        "WITH s, h, o "
        "WHERE h IS NULL OR o IS NULL "
        "RETURN s.id AS sensor, h.hasName AS horse, o.id AS objective"
    ),
    "Q90": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t:TransitionStage) "
        "RETURN t.Volume AS volume, COUNT(DISTINCT h) AS horses, "
        "COLLECT(DISTINCT h.hasName) AS names"
    ),
    "Q91": (
        "MATCH (h:Horse {hasName:'Auroch'})-[:COMPETESIN]->(e) "
        "OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)"
        "-[:HASHORSE]->(h) "
        "RETURN e.id AS event, p.rank AS rank"
    ),
    "Q92": (
        "MATCH (h:Horse)<-[:ISATTACHEDTO]-(s:InertialSensors) "
        "WITH h, COUNT(s) AS sensor_count "
        "WITH sensor_count, COUNT(h) AS horses, COLLECT(h.hasName) AS names "
        "ORDER BY sensor_count ASC "
        "LIMIT 1 "
        "RETURN sensor_count, horses, names"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q93)
    "Q93": (
        "MATCH (h:Horse) "
        "OPTIONAL MATCH (h)-[:COMPETESIN]->(e) "
        "OPTIONAL MATCH (p:EventParticipation)-[:HASHORSE]->(h) "
        "WITH h, COUNT(DISTINCT e) AS eng, COUNT(DISTINCT p) AS res "
        "RETURN eng, res, COUNT(h) AS horses"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q94)
    "Q94": (
        "MATCH (h:Horse)-[:TRAINSIN]->(t) "
        "WHERE h.hasName IN ['Dakota','Orion'] "
        "RETURN h.hasName AS horse, COLLECT(t.id) AS ids "
        "ORDER BY horse"
    ),
    "Q95": (
        "MATCH (h:Horse)-[:COMPETESIN]->(e)-[:INSEASON]->(s:CompetitiveSeason) "
        "RETURN DISTINCT s.seasonName AS season, e.category AS category, "
        "COUNT(*) AS n"
    ),
    "Q96": (
        "MATCH (e)-[:HASPARTICIPATION]->(p1:EventParticipation)-[:HASRIDER]->(r:Rider) "
        "MATCH (e)-[:HASPARTICIPATION]->(p2:EventParticipation)-[:HASRIDER]->(r) "
        "MATCH (p1)-[:HASHORSE]->(h1:Horse) "
        "MATCH (p2)-[:HASHORSE]->(h2:Horse) "
        "WHERE p1.rank = 1 AND p2.rank = 2 AND h1 <> h2 "
        "RETURN DISTINCT e.id AS event, r.id AS rider, "
        "h1.hasName AS first, h2.hasName AS second"
    ),
    # Exact Cypher from docs/graph_rag/graph_gt_reconciliation_signoff.md (Q97)
    "Q97": (
        "MATCH (h:Horse) "
        "OPTIONAL MATCH (h)-[:COMPETESIN]->(e) "
        "OPTIONAL MATCH (p:EventParticipation)-[:HASHORSE]->(h) "
        "WITH h, COUNT(DISTINCT e) AS engages, COUNT(DISTINCT p) AS results "
        "RETURN engages, results, COUNT(h) AS horses, "
        "COLLECT(h.hasName)[0..5] AS sample "
        "ORDER BY engages, results"
    ),
    # Exact Cypher from signoff Q98 (= same anti-join as Q65)
    "Q98": (
        "MATCH (h:Horse)-[:COMPETESIN]->(e) "
        "OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h) "
        "WITH h, e, COUNT(p) AS ranked "
        "WHERE ranked = 0 "
        "RETURN h.hasName AS horse, e.id AS event, ranked "
        "ORDER BY horse"
    ),
    "Q99": (
        "MATCH (s:InertialSensors) "
        "OPTIONAL MATCH (s)-[:ISATTACHEDTO]->(h:Horse) "
        "OPTIONAL MATCH (s)-[:ISUSEDFOR]->(o) "
        "RETURN COUNT(s) AS sensors, COUNT(h) AS horse_links, "
        "COUNT(o) AS objective_links"
    ),
    "Q100": (
        "MATCH (h:Horse) WITH COUNT(h) AS horses "
        "MATCH (r:Rider) WITH horses, COUNT(r) AS riders "
        "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross "
        "WITH horses, riders, COUNT(e) AS events "
        "MATCH (s:InertialSensors) "
        "RETURN horses, riders, events, COUNT(s) AS sensors"
    ),
}

EX_NOT_APPLICABLE: dict[str, str] = {
    "Q35": (
        "Open-ended multi-hop narrative ('analyse complète') spanning event "
        "attributes, rankings, and multiple training stages — no single "
        "canonical result-set shape"
    ),
    "Q38": (
        "question text references a non-existent legacy identifier; "
        "correct system behavior is 'not found,' which is a different claim "
        "than what gold would need to test"
    ),
    "Q39": (
        "question text references a non-existent legacy identifier; "
        "correct system behavior is 'not found,' which is a different claim "
        "than what gold would need to test"
    ),
    "Q40": (
        "Unanswerable: no age property exists on Horse; correct behavior is "
        "to decline, not return a result set"
    ),
    "Q68": (
        "Unanswerable: schema has no coat/color, weight, or veterinarian "
        "phone properties — correct behavior is to decline"
    ),
    "Q70": (
        "Schema-explanation question ('comment savoir…') with no single "
        "canonical lookup result-set shape"
    ),
    "Q71": (
        "Schema-explanation question about HASPARTICIPATION/HASHORSE/HASRIDER "
        "path — conceptual, not a data lookup"
    ),
    "Q72": (
        "Schema-explanation question ('par quelles informations…') — "
        "conceptual, not a data lookup"
    ),
    "Q73": (
        "Schema-explanation question about ISATTACHEDTO/ISUSEDFOR — "
        "conceptual, not a data lookup"
    ),
    "Q88": (
        "Schema-explanation question contrasting ASSOCIATED_WITH vs "
        "ranked HASRIDER/HASHORSE participation — conceptual, not a "
        "data lookup"
    ),
    "Q89": (
        "Schema-explanation question about how training stages link to "
        "events (DEPENDSON) — conceptual, not a data lookup"
    ),
}

AMBIGUOUS_FOR_REVIEW: dict[str, str] = {}

# Per-question overrides: question_id → "set" when COLLECT duplicate counts
# are intentionally irrelevant. Default for all other questions is multiset.
LIST_COMPARE_OVERRIDES: dict[str, ListCompareMode] = {}


def _value_token(value: Any, list_mode: ListCompareMode = "multiset") -> str:
    """Canonical, order-insensitive token for one cell value.

    COLLECT / list cells default to MULTISET: order ignored, duplicate
    counts must match (missing DISTINCT inflation → MISMATCH). Pass
    list_mode="set" only via an explicit per-question override.
    """
    if isinstance(value, list):
        tokens = [_value_token(item, list_mode) for item in value]
        if list_mode == "set":
            tokens = sorted(set(tokens))
        else:
            tokens = sorted(tokens)
        return "[" + ",".join(tokens) + "]"

    if isinstance(value, dict):
        parts = [
            f"{key}:{_value_token(val, list_mode)}"
            for key, val in sorted(value.items())
        ]
        return "{" + ",".join(parts) + "}"

    if value is None:
        return "null"

    if isinstance(value, bool):
        # bool is a subclass of int — handle before int
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        # Normalize -0.0 and prefer compact int form when exact
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    return str(value)


def _row_value_tokens(row: Any, list_mode: ListCompareMode = "multiset") -> list[str]:
    """Cell-value tokens for one row (column keys discarded)."""
    if isinstance(row, dict):
        return [_value_token(v, list_mode) for v in row.values()]
    if isinstance(row, (list, tuple)):
        return [_value_token(v, list_mode) for v in row]
    return [_value_token(row, list_mode)]


def _normalize_row(row: Any, list_mode: ListCompareMode = "multiset") -> str:
    """One result row → column-order-independent token.

    Neo4j rows are usually dicts (key = RETURN alias). Values are compared
    as a multiset *within the row*: sorted value-tokens, keys discarded so
    RETURN a,b and RETURN b,a MATCH when the values match. Row boundaries
    are preserved — values from different rows are never merged.
    """
    values = _row_value_tokens(row, list_mode)
    return "(" + ",".join(sorted(values)) + ")"


def normalize_cypher_result(
    result: Any,
    list_mode: ListCompareMode = "multiset",
) -> list[str]:
    """Normalize a Cypher result set for order-insensitive comparison.

    Each row is tokenized as a whole (column-order-insensitive value
    multiset *within that row*). The result is then a sorted multiset of
    those whole-row tokens — never a single bag that flattens values
    across rows. Swapping values between two rows therefore MISMATCHes.

    Accepts:
      - list of dict rows (typical Neo4j / LangChain raw_context)
      - list of list/tuple rows
      - empty list / None
      - a bare scalar (treated as one row, one column)
    """
    if result is None:
        rows: list[Any] = []
    elif isinstance(result, list):
        rows = result
    else:
        # Scalar single-value result → 1×1
        rows = [result]

    normalized = [_normalize_row(row, list_mode) for row in rows]
    normalized.sort()
    return normalized


def _as_rows(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def _row_multiset_contained(
    gold_row: Any,
    gen_row: Any,
    list_mode: ListCompareMode = "multiset",
) -> bool:
    """True iff gold_row's value multiset ⊆ gen_row's value multiset.

    Allows generated queries to return *extra columns* (e.g. discipline
    alongside event id) without failing EX when every gold cell value is
    present in the generated row.
    """
    from collections import Counter

    gold_counts = Counter(_row_value_tokens(gold_row, list_mode))
    gen_counts = Counter(_row_value_tokens(gen_row, list_mode))
    return all(gen_counts[k] >= n for k, n in gold_counts.items())


def _containment_match(
    generated_result: Any,
    gold_result: Any,
    list_mode: ListCompareMode = "multiset",
) -> bool:
    """Bipartite gold⊆gen row match with equal cardinality.

    Each gold row must map to a distinct generated row whose value multiset
    is a superset of the gold row's. Extra generated columns are OK; extra
    or missing *rows* are not; missing gold columns are not.
    """
    gen_rows = _as_rows(generated_result)
    gold_rows = _as_rows(gold_result)
    if len(gen_rows) != len(gold_rows):
        return False
    used: set[int] = set()
    for grow in gold_rows:
        found = None
        for i, crow in enumerate(gen_rows):
            if i in used:
                continue
            if _row_multiset_contained(grow, crow, list_mode):
                found = i
                break
        if found is None:
            return False
        used.add(found)
    return True


def compare_cypher_execution(
    generated_result: Any,
    gold_result: Any,
    list_mode: ListCompareMode = "multiset",
) -> tuple[bool, str | None]:
    """Compare Neo4j result sets (generated vs gold).

    Exact normalized equality still MATCHES.

    Column-superset tolerance: when |rows| are equal, MATCH also if every
    gold row's value multiset is a *subset* of some distinct generated row's
    value multiset (bipartite). Extra generated columns are therefore OK;
    missing or wrong gold-required values are not. Row cardinality,
    multiset duplicate counts (T8), and cross-row pairing (T9) stay strict.

    Returns (matched, error_text). error_text is set only for structural
    failures while normalizing; otherwise None. Match is False on any
    normalization error.

    list_mode defaults to "multiset". Use "set" only when a specific gold
    question opts in via LIST_COMPARE_OVERRIDES.
    """
    try:
        generated_norm = normalize_cypher_result(generated_result, list_mode)
    except Exception as exc:  # noqa: BLE001 - surface to caller as EX miss
        return False, f"generated result normalize exception: {exc}"

    try:
        gold_norm = normalize_cypher_result(gold_result, list_mode)
    except Exception as exc:  # noqa: BLE001 - surface to caller as EX miss
        return False, f"gold result normalize exception: {exc}"

    if generated_norm == gold_norm:
        return True, None

    try:
        if _containment_match(generated_result, gold_result, list_mode):
            return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"containment compare exception: {exc}"

    return False, None
