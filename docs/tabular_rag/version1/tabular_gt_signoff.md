# Track 1 (Tabular) — Gold SQL reconciliation (sign-off)

**Date:** 2026-07-26  
**Source file:** `scripts/tabular_rag/gold_queries.py` (`GOLD_QUERIES` only — no separate hardcoded expected-result table)  
**Live DB:** `data/tabular_rag/tabular.db` (re-ETL’d V9: trainings=207, training_actors=355, event_participations=101, NULL ranks=0)  
**Status:** Human approved batch 1 (Q5 / Q10 / Q31 / Q63 / Q67) and batch 2
(Q4 / Q11 / Q13 / Q17 / Q18 / Q82) on 2026-07-26. All **11** applied to
`gold_queries.py`.

Every SQL under **(d)** was executed read-only against live `tabular.db` in this session
(not via the Text-to-SQL chain, not from memory).

### How to read this audit

Unlike Graph RAG’s `test_dataset.json`, tabular EX gold does **not** store prose
answers or numeric literals. Each entry is a gold SQL string; EX compares the
**result set** of generated SQL to the result set of gold SQL on the same DB.

Therefore:

- **Adaptive aggregations** (`COUNT(*) … GROUP BY stage_type`, etc.) already
  return live V9 figures when run on the current DB → **NO CHANGE**.
- **Stale gold** means the SQL itself is wrong for V9 (bad id filter, or a
  shape that answers the V8 phenomenon that no longer exists).

---

## Summary

| Verdict | Count | Queries |
|---|---|---|
| **Checked** (touch `trainings` / `training_actors` / `event_participations`) | **35** | see inventory below |
| **NO CHANGE — gold SQL already V9-correct on live DB** | **30** | Q6–Q9, Q22–Q23, Q32–Q34, Q52–Q57, Q62, Q65–Q66, Q75–Q76, Q83–Q86, Q90–Q91, Q93, Q96–Q98 |
| **CLEAN — approved & applied (batch 1)** | **5** | **Q5, Q10, Q31, Q63, Q67** |
| **CLEAN — approved & applied (batch 2)** | **6** | **Q4, Q11, Q13, Q17, Q18, Q82** (`Horse1`→`Horse_Dakota`, `Horse2`→`Horse_Naya`) |
| **Total gold corrections applied** | **11** | |

\*Batch 2 was flagged in Phase 1 EX triage (stale V8 horse ids outside the
original trainings/actors/participations scope). Same id mapping as batch 1.

---

## Step 1 inventory — verbatim gold SQL (touched tables only)

From `scripts/tabular_rag/gold_queries.py`:

| Q | Gold SQL (verbatim) |
|---|---|
| Q5 | `SELECT training_id FROM trainings WHERE horse_id = 'Horse1';` |
| Q6 | `SELECT frequency, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type = 'PreparationStage' GROUP BY frequency;` |
| Q7 | `SELECT DISTINCT intensity, stage_type FROM trainings WHERE stage_type = 'PreCompetitionStage';` |
| Q8 | `SELECT volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type = 'PreparationStage' GROUP BY volume;` |
| Q9 | `SELECT volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type = 'PreCompetitionStage' GROUP BY volume;` |
| Q10 | `SELECT DISTINCT e.event_id, e.location, e.category, e.event_date, e.discipline, t.stage_type FROM trainings t JOIN events e ON t.event_id = e.event_id WHERE t.horse_id = 'Horse1';` |
| Q22 | `SELECT DISTINCT p.person_id, p.role FROM trainings t JOIN training_actors ta ON t.training_id = ta.training_id JOIN people p ON ta.actor_id = p.person_id WHERE LOWER(t.stage_type) = LOWER('PreparationStage');` |
| Q23 | `SELECT DISTINCT p.person_id, p.role, t.stage_type FROM trainings t JOIN training_actors ta ON t.training_id = ta.training_id JOIN people p ON ta.actor_id = p.person_id WHERE LOWER(t.stage_type) = LOWER('PreCompetitionStage');` |
| Q31 | `SELECT rider_id, rank FROM event_participations WHERE horse_id = 'Horse1' AND event_id = 'Event_SJ_01';` |
| Q32 | `SELECT stage_type, frequency, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type IN ('PreparationStage','PreCompetitionStage') GROUP BY stage_type, frequency;` |
| Q33 | `SELECT stage_type, frequency, intensity, volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type IN ('PreparationStage','PreCompetitionStage') GROUP BY stage_type, frequency, intensity, volume;` |
| Q34 | `SELECT DISTINCT ta.actor_id, ta.actor_role, t.stage_type FROM trainings t JOIN training_actors ta ON t.training_id = ta.training_id WHERE t.stage_type IN ('PreparationStage', 'PreCompetitionStage') ORDER BY ta.actor_id, t.stage_type;` |
| Q52 | `SELECT stage_type, COUNT(*) FROM trainings GROUP BY stage_type;` |
| Q53 | `SELECT horse_id, COUNT(*) AS stage_count FROM trainings GROUP BY horse_id ORDER BY stage_count DESC LIMIT 1;` |
| Q54 | `SELECT DISTINCT volume, intensity, frequency FROM trainings WHERE stage_type = 'CompetitionStage';` |
| Q55 | `SELECT volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings WHERE stage_type = 'TransitionStage' GROUP BY volume;` |
| Q56 | `SELECT t.stage_type, ta.actor_role, COUNT(DISTINCT t.training_id) AS cnt FROM trainings t JOIN training_actors ta ON t.training_id = ta.training_id WHERE ta.actor_role IN ('Veterinarian','Caretaker') GROUP BY t.stage_type, ta.actor_role;` |
| Q57 | `SELECT training_id FROM trainings WHERE training_id NOT IN (SELECT DISTINCT training_id FROM training_actors);` |
| Q62 | `SELECT DISTINCT e.event_id, e.location, e.category, e.event_date, e.discipline FROM events e LEFT JOIN event_entries ee ON e.event_id = ee.event_id LEFT JOIN event_participations ep ON e.event_id = ep.event_id WHERE ep.participation_id IS NULL;` |
| Q63 | `SELECT event_id, COUNT(*) AS cnt FROM event_participations GROUP BY event_id ORDER BY cnt DESC LIMIT 1;` |
| Q65 | `SELECT COUNT(DISTINCT e.horse_id) AS unranked, (SELECT COUNT(DISTINCT horse_id) FROM event_entries) AS total FROM event_entries e LEFT JOIN event_participations p ON e.horse_id = p.horse_id AND e.event_id = p.event_id WHERE p.participation_id IS NULL;` |
| Q66 | `SELECT event_id, rider_id, horse_id FROM event_participations WHERE (event_id, rider_id) IN (SELECT event_id, rider_id FROM event_participations GROUP BY event_id, rider_id HAVING COUNT(DISTINCT horse_id) > 1);` |
| Q67 | `SELECT ee.event_id, COUNT(DISTINCT ee.horse_id) - COALESCE((SELECT COUNT(*) FROM event_participations ep WHERE ep.event_id = ee.event_id), 0) AS gap FROM event_entries ee GROUP BY ee.event_id ORDER BY gap DESC LIMIT 1;` |
| Q75 | `SELECT DISTINCT intensity FROM trainings;` |
| Q76 | `SELECT DISTINCT p.person_id, p.role FROM training_actors ta JOIN people p ON ta.actor_id = p.person_id WHERE LOWER(ta.actor_role) <> LOWER('Rider');` |
| Q83 | `SELECT horse_id, stage_type, volume FROM trainings WHERE stage_type = 'PreCompetitionStage' AND volume = (SELECT MAX(volume) FROM trainings WHERE stage_type = 'PreCompetitionStage');` |
| Q84 | `SELECT horse_id, volume FROM trainings WHERE stage_type = 'PreparationStage' AND volume = (SELECT MAX(volume) FROM trainings WHERE stage_type = 'PreparationStage');` |
| Q85 | `SELECT horse_id, stage_type, volume FROM trainings WHERE (stage_type = 'PreparationStage' AND volume = (SELECT MIN(volume) FROM trainings WHERE stage_type = 'PreparationStage')) OR (stage_type = 'PreCompetitionStage' AND volume = (SELECT MIN(volume) FROM trainings WHERE stage_type = 'PreCompetitionStage'));` |
| Q86 | `SELECT t.horse_id, COUNT(*) AS stage_count, (SELECT COUNT(DISTINCT event_id) FROM event_entries ee WHERE ee.horse_id = t.horse_id) AS competition_count FROM trainings t GROUP BY t.horse_id;` |
| Q90 | `SELECT DISTINCT training_id, horse_id, volume FROM trainings WHERE stage_type = 'TransitionStage';` |
| Q91 | `SELECT ee.event_id, ep.rank FROM event_entries ee LEFT JOIN event_participations ep ON ee.event_id = ep.event_id AND ee.horse_id = ep.horse_id WHERE ee.horse_id = 'Horse_Auroch';` |
| Q93 | `SELECT t.horse_id, COUNT(*) AS stage_count, (SELECT COUNT(DISTINCT event_id) FROM event_entries ee WHERE ee.horse_id = t.horse_id) AS competition_count FROM trainings t GROUP BY t.horse_id;` |
| Q96 | `SELECT event_id, rider_id FROM event_participations WHERE rank IN (1,2) GROUP BY event_id, rider_id HAVING COUNT(DISTINCT rank) = 2;` |
| Q97 | `SELECT COUNT(DISTINCT horse_id) AS horses_with_result, (SELECT COUNT(*) FROM horses) AS total_horses FROM event_participations;` |
| Q98 | `SELECT COUNT(*) FROM event_entries ee LEFT JOIN event_participations ep ON ee.horse_id = ep.horse_id AND ee.event_id = ep.event_id WHERE ep.participation_id IS NULL;` |

---

## Category A — V8 horse id `Horse1` (Dakota is `Horse_Dakota` in V9)

### Q5 — CLEAN

**(a) Question:**  
Quelles étapes d'entraînement Dakota suit-il ?

**(b) Current gold SQL / live result of that SQL:**  
```sql
SELECT training_id FROM trainings WHERE horse_id = 'Horse1';
```
Live result: **0 rows** (`Horse1` does not exist in `horses`).

**(c) Proposed correction:**  
```sql
SELECT training_id FROM trainings WHERE horse_id = 'Horse_Dakota';
```

**(d) Exact SQL + live result (proposed):**
```sql
SELECT training_id FROM trainings WHERE horse_id = 'Horse_Dakota' ORDER BY training_id;
```
Live (8 rows):
- `Training_Comp_Cross2026_Dakota_01`
- `Training_Comp_Dress01_Dakota_01`
- `Training_Competition_SJ_01`
- `Training_PreComp_SJ_01`
- `Training_PreCompetition_SJ_01`
- `Training_Prepa_SJ_01`
- `Training_Preparation_SJ_01`
- `Training_Transition_SJ_01`

Supporting check:
```sql
SELECT horse_id, name FROM horses WHERE name = 'Dakota';
```
Live: `('Horse_Dakota', 'Dakota')`

---

### Q10 — CLEAN

**(a) Question:**  
De quel événement dépendent les étapes d'entraînement de Dakota ?

**(b) Current gold SQL / live result of that SQL:**  
```sql
SELECT DISTINCT e.event_id, e.location, e.category, e.event_date, e.discipline, t.stage_type
FROM trainings t JOIN events e ON t.event_id = e.event_id
WHERE t.horse_id = 'Horse1';
```
Live result: **0 rows**.

**(c) Proposed correction:**  
```sql
SELECT DISTINCT e.event_id, e.location, e.category, e.event_date, e.discipline, t.stage_type
FROM trainings t JOIN events e ON t.event_id = e.event_id
WHERE t.horse_id = 'Horse_Dakota';
```

**(d) Exact SQL + live result (proposed):**
```sql
SELECT DISTINCT e.event_id, e.location, e.category, e.event_date, e.discipline, t.stage_type
FROM trainings t JOIN events e ON t.event_id = e.event_id
WHERE t.horse_id = 'Horse_Dakota'
ORDER BY e.event_id, t.stage_type;
```
Live (8 rows — 4 distinct events):
| event_id | location | category | event_date | discipline | stage_type |
|---|---|---|---|---|---|
| Event_Cross_2026_01 | Bordeaux | Amateur 1 | 2026-08-10 | Cross | CompetitionStage |
| Event_Dressage_01 | Angers | Club Elite | 2026-05-03 | Dressage | CompetitionStage |
| Event_SJ_01 | Saumur | Amateur 1 | 2026-04-12 | ShowJumping | PreCompetitionStage |
| Event_SJ_01 | Saumur | Amateur 1 | 2026-04-12 | ShowJumping | PreparationStage |
| Event_SJ_2026_01 | Paris | Amateur 2 | 2026-06-14 | ShowJumping | CompetitionStage |
| Event_SJ_2026_01 | Paris | Amateur 2 | 2026-06-14 | ShowJumping | PreCompetitionStage |
| Event_SJ_2026_01 | Paris | Amateur 2 | 2026-06-14 | ShowJumping | PreparationStage |
| Event_SJ_2026_01 | Paris | Amateur 2 | 2026-06-14 | ShowJumping | TransitionStage |

**Note (not blocking the Horse_Dakota fix):** Graph `ground_truth` for Q10 still names only
`Event_SJ_01` and `Event_SJ_2026_01`. Live tabular DEPENDS_ON for Dakota also
links competition stages to Cross/Dressage. The id fix is unambiguous; whether
gold should further restrict stage types is a separate product choice.

---

### Q31 — CLEAN

**(a) Question:**  
Quel classement Dakota et Emma ont-ils obtenu à Event_SJ_01 ?

**(b) Current gold SQL / live result of that SQL:**  
```sql
SELECT rider_id, rank FROM event_participations
WHERE horse_id = 'Horse1' AND event_id = 'Event_SJ_01';
```
Live result: **0 rows**.

**(c) Proposed correction:**  
```sql
SELECT rider_id, rank FROM event_participations
WHERE horse_id = 'Horse_Dakota' AND event_id = 'Event_SJ_01';
```

**(d) Exact SQL + live result (proposed):**
```sql
SELECT rider_id, rank FROM event_participations
WHERE horse_id = 'Horse_Dakota' AND event_id = 'Event_SJ_01';
```
Live: `('Rider_Emma', 2)`

---

## Category B — gap query still shaped for V8 “unranked entrants”

### Q67 — CLEAN

**(a) Question:**  
Quelle compétition a le plus grand nombre de chevaux engagés sans résultat officiel enregistré ?

**(b) Current gold SQL / live result of that SQL:**  
```sql
SELECT ee.event_id,
       COUNT(DISTINCT ee.horse_id)
         - COALESCE((SELECT COUNT(*) FROM event_participations ep WHERE ep.event_id = ee.event_id), 0) AS gap
FROM event_entries ee
GROUP BY ee.event_id
ORDER BY gap DESC
LIMIT 1;
```
Live result: `('Event_Versailles_SJ_2026', 0)`  
(All events have gap **0**; `ORDER BY gap DESC LIMIT 1` returns an arbitrary
event with gap 0 — misleading for EX / for the NL question.)

**(c) Proposed correction:**  
```sql
SELECT ee.event_id,
       COUNT(DISTINCT ee.horse_id)
         - COALESCE((SELECT COUNT(*) FROM event_participations ep WHERE ep.event_id = ee.event_id), 0) AS gap
FROM event_entries ee
GROUP BY ee.event_id
HAVING gap > 0
ORDER BY gap DESC
LIMIT 1;
```
(Empty result set = “no such competition” — aligns with corrected Graph GT.)

**(d) Exact SQL + live result (proposed):**
```sql
SELECT ee.event_id,
       COUNT(DISTINCT ee.horse_id)
         - COALESCE((SELECT COUNT(*) FROM event_participations ep WHERE ep.event_id = ee.event_id), 0) AS gap
FROM event_entries ee
GROUP BY ee.event_id
HAVING gap > 0
ORDER BY gap DESC
LIMIT 1;
```
Live: **0 rows**

Supporting check (max gap on V9):
```sql
SELECT MAX(gap) FROM (
  SELECT COUNT(DISTINCT ee.horse_id)
           - COALESCE((SELECT COUNT(*) FROM event_participations ep WHERE ep.event_id = ee.event_id), 0) AS gap
  FROM event_entries ee
  GROUP BY ee.event_id
);
```
Live: `0`

---

## Category C — max participation tie (was AMBIGUOUS; policy approved)

### Q63 — CLEAN (approved: return full max-tie set)

**(a) Question:**  
Quel événement de la saison a réuni le plus de résultats classés ?

**(b) Current gold SQL / live result of that SQL:**  
```sql
SELECT event_id, COUNT(*) AS cnt FROM event_participations
GROUP BY event_id ORDER BY cnt DESC LIMIT 1;
```
Live result (this session, one arbitrary tie winner): `('Event_SJ_2026_01', 7)`  
`LIMIT 1` hides the 3-way tie and makes EX brittle.

**(c) Proposed correction (SQL + expected answer shape):**  
```sql
SELECT event_id, COUNT(*) AS cnt FROM event_participations
GROUP BY event_id
HAVING COUNT(*) = (
  SELECT MAX(c) FROM (SELECT COUNT(*) AS c FROM event_participations GROUP BY event_id)
);
```
**Expected answer text (enumerate all ties, do not name one):**  
Three events are tied at the maximum with **7** ranked results each:
`Event_LeMans_Cross_2026`, `Event_Pau_SJ_2026`, and `Event_SJ_2026_01`
(same three-way tie as Graph RAG’s corrected Q63 ground truth).

**(d) Exact SQL + live result (proposed):**
```sql
SELECT event_id, COUNT(*) AS cnt FROM event_participations
GROUP BY event_id
HAVING COUNT(*) = (
  SELECT MAX(c) FROM (SELECT COUNT(*) AS c FROM event_participations GROUP BY event_id)
)
ORDER BY event_id;
```
Live (verified 2026-07-26 on `data/tabular_rag/tabular.db`):
- `('Event_LeMans_Cross_2026', 7)`
- `('Event_Pau_SJ_2026', 7)`
- `('Event_SJ_2026_01', 7)`

Max count confirmed **7**; matches Graph RAG Q63.

---

## Category D — NO CHANGE (already V9-correct when gold SQL is executed)

These gold strings contain no stale V8 ids and return live V9 result sets.
Representative live evidence (same SQL as gold, run on `tabular.db`):

| Q | Live result summary |
|---|---|
| Q6 | `(4, 36)`, `(5, 14)` |
| Q7 | `('Élevée', 'PreCompetitionStage')` |
| Q8 | `40min:1`, `45min:31`, `50min:16`, `55min:2` |
| Q9 | `55min:1`, `60min:30`, `65min:14`, `70min:4`, `75min:1` |
| Q22 | 26 distinct (person_id, role) on PreparationStage |
| Q23 | 27 distinct (person_id, role, stage_type) on PreCompetitionStage |
| Q32 | Prep freq 4→36 / 5→14; PreComp freq 3→43 / 4→7 |
| Q33 | 12 (stage, freq, intensity, volume, horse_count) groups |
| Q34 | 53 distinct (actor_id, actor_role, stage_type) rows |
| Q52 | Prep 51 / PreComp 51 / Comp **55** / Transition **50** |
| Q53 | `('Horse_Dakota', 8)` |
| Q54 | `('30min', 'Pic', 1)` |
| Q55 | `25min:45`, `30min:5` |
| Q56 | Prep Caretaker 50 / Vet 48; PreComp Caretaker 1 / Vet 49 |
| Q57 | **0** trainings without actors |
| Q62 | **0** events lacking any participation |
| Q65 | `(0, 50)` unranked / total |
| Q66 | 22 (event, rider, horse) rows for multi-horse rider/event |
| Q75 | Faible, Modérée, Pic, Élevée |
| Q76 | Caretaker_Sophie, Vet_DrMartin |
| Q83 | Comet PreComp `75min` |
| Q84 | Comet + Ecume Prep `55min` |
| Q85 | Pixie Prep `40min` + PreComp `55min` |
| Q86 / Q93 | 50 horse rows; Dakota = `(Horse_Dakota, 8, 5)` |
| Q90 | 50 transition rows |
| Q91 | Auroch: Bordeaux rank 4, Clermont rank 5 |
| Q96 | 4 (event, rider) pairs with ranks 1 and 2 |
| Q97 | `(50, 50)` horses_with_result / total_horses |
| Q98 | `(0,)` entries without matching participation |

These are the tabular analogues of Graph questions whose **prose** GT was stale
(Q52/Q53/Q55/Q62/Q65/Q86/Q93/Q97/Q98, etc.): here the gold SQL is adaptive, so
**no edit to `gold_queries.py` is required** for them once the DB is V9.

---

## Category E — V8 horse ids outside trainings scope (batch 2)

Same substitution family as Category A (`Horse1` = Dakota, `Horse2` = Naya).
Approved and applied 2026-07-26 after Phase 1 EX triage.

### Q4 — CLEAN

**(a) Question:** Dans quels événements sportifs Dakota participe-t-il ?

**(b) Current gold SQL / live result:**  
`SELECT event_id FROM event_entries WHERE horse_id = 'Horse1';` → **0 rows**

**(c) Proposed correction:**  
`SELECT event_id FROM event_entries WHERE horse_id = 'Horse_Dakota';`

**(d) Live result (proposed):**  
`Event_Cross_2026_01`, `Event_Dressage_01`, `Event_Dressage_2026_01`, `Event_SJ_01`, `Event_SJ_2026_01` (5 rows)

---

### Q11 — CLEAN

**(a) Question:** Combien de capteurs IMU sont attachés à Dakota ?

**(b) Current:** `... WHERE horse_id = 'Horse1'` → `COUNT(*) = 0`

**(c) Proposed:** `... WHERE horse_id = 'Horse_Dakota'`

**(d) Live:** `(4,)`

---

### Q13 — CLEAN

**(a) Question:** À quelles positions anatomiques les capteurs IMU sont-ils placés sur Dakota ?

**(b) Current:** `Horse1` → **0 rows**

**(c) Proposed:** `Horse_Dakota`

**(d) Live:** Withers, Sternum, CanonOfHindlimb, CanonOfForelimb

---

### Q17 — CLEAN

**(a) Question:** Quels cavaliers sont associés à Dakota ?

**(b) Current:** `Horse1` → **0 rows**

**(c) Proposed:** `Horse_Dakota`

**(d) Live:** `Rider_Emma`, `Rider_Manon`

---

### Q18 — CLEAN

**(a) Question:** Quel cavalier est associé à Naya ?

**(b) Current:** `horse_id = 'Horse2'` → **0 rows**

**(c) Proposed:** `horse_id = 'Horse_Naya'`

**(d) Live:** `Rider_Leo`

---

### Q82 — CLEAN

**(a) Question:** Quels capteurs de Dakota servent à analyser sa démarche, et lesquels servent à surveiller sa fatigue ?

**(b) Current:** `Horse1` → **0 rows**

**(c) Proposed:** `Horse_Dakota`

**(d) Live:**
- `IMU_CanonFore_Dakota_01` → FatigueDetection  
- `IMU_CanonHind_Dakota_01` → GaitClassif_01  
- `IMU_Sternum_Dakota_01` → FatigueDetection  
- `IMU_Withers_Dakota_01` → GaitClassif_01  

---

## Proposed patch set — all 11 approved 2026-07-26

| Q | Change |
|---|---|
| Q5 | `'Horse1'` → `'Horse_Dakota'` |
| Q10 | `'Horse1'` → `'Horse_Dakota'` |
| Q31 | `'Horse1'` → `'Horse_Dakota'` |
| Q63 | remove `ORDER BY cnt DESC LIMIT 1`; keep all rows tied at `MAX(cnt)` |
| Q67 | add `HAVING gap > 0` before `ORDER BY` |
| Q4 | `'Horse1'` → `'Horse_Dakota'` |
| Q11 | `'Horse1'` → `'Horse_Dakota'` |
| Q13 | `'Horse1'` → `'Horse_Dakota'` |
| Q17 | `'Horse1'` → `'Horse_Dakota'` |
| Q18 | `'Horse2'` → `'Horse_Naya'` |
| Q82 | `'Horse1'` → `'Horse_Dakota'` |

All 11 applied to `scripts/tabular_rag/gold_queries.py`.
