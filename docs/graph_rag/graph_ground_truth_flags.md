# Ground-truth flags — questions whose `ground_truth` contradicts the live V9 graph

**Status: FLAGGED ONLY. `data/test_dataset.json` has NOT been modified.**

Every figure below comes from a Cypher query executed directly against the live
Neo4j database (not through the RAG chain). Each entry gives the exact
discrepancy and a proposed replacement, so a human can verify and apply them
independently of prompt-tuning work.

Full supporting inventory: `docs/graph_v9_ground_truth_drift.md`.

Scoring note: these questions cap the achievable combined score. A correct
V9 answer scores ~0 against a v8.7 ground truth, so no prompt change can
recover them. Estimated ceiling cost: **16–18 of 100 questions**.

---

## Category A — sensor identifiers renamed in V9

The bare V8 identifiers return zero rows in V9; identifiers are now
horse-scoped (`IMU_<Position>_<Horse>_01`).

| Q | Ground truth says | Live V9 | Proposed correction |
|---|---|---|---|
| Q38 | `IMU_Withers_01` used for gait classification | no such node; `IMU_Withers_Dakota_01` → `GaitClassif_01` | Replace the identifier with `IMU_Withers_Dakota_01`; the objective (gait classification) stays correct |
| Q39 | `IMU_CanonFore_01` used for fatigue detection | no such node; `IMU_CanonFore_Dakota_01` → `FatigueDetection` | Replace with `IMU_CanonFore_Dakota_01`; objective stays correct |
| Q14 | 50 withers sensors, e.g. `IMU_Withers_01` | 50 withers sensors (correct); example id invalid | Replace the example id with `IMU_Withers_Dakota_01`. The count of 50 is already correct |
| Q82 | four bare ids for Dakota | `IMU_Withers_Dakota_01` + `IMU_CanonHind_Dakota_01` → `GaitClassif_01`; `IMU_CanonFore_Dakota_01` + `IMU_Sternum_Dakota_01` → `FatigueDetection` | Replace all four identifiers with the horse-scoped forms; the objective pairing is correct |

### STAGED FOR HUMAN SIGN-OFF (2026-07-26) — Q38 / Q39 only

Verified again on live Neo4j (direct query, not via the RAG chain):
- `MATCH (s) WHERE s.id IN ["IMU_Withers_01","IMU_CanonFore_01"]` → **0 rows**
- `IMU_Withers_Dakota_01` -[:ISUSEDFOR]-> `GaitClassif_01`
- `IMU_CanonFore_Dakota_01` -[:ISUSEDFOR]-> `FatigueDetection`

Independent eval `semantic_evaluation_20260726_145934.json` confirms both
questions score 0 because the model correctly answers "not available" for
the stale V8 ids. **Do not fix via prompt.** Apply these `ground_truth`
replacements in `data/test_dataset.json` only after explicit sign-off:

**Q38 — proposed `ground_truth`:**
`Le capteur IMU_Withers_Dakota_01 (garrot de Dakota) est utilisé pour la classification de la démarche (GaitClassif_01).`

**Q39 — proposed `ground_truth`:**
`Le capteur IMU_CanonFore_Dakota_01 (canon antérieur de Dakota) est utilisé pour la détection de fatigue (FatigueDetection).`

Status: **STAGED ONLY — `test_dataset.json` not modified.**

## Category B — training-stage inventory grew in V9

| Q | Ground truth says | Live V9 | Proposed correction |
|---|---|---|---|
| Q5 | Dakota follows 6 stages | **8** — adds `Training_Comp_Cross2026_Dakota_01` and `Training_Comp_Dress01_Dakota_01` | Change 6 → 8 and list the two extra competition stages |
| Q52 | 51 prep / 51 pre-comp / 50 competition / 19 transition | 51 / 51 / **55** / **50** | Change competition 50 → 55 and transition 19 → 50 |
| Q53 | Dakota 6 stages; 31 horses ×3, 18 ×4 | Dakota **8**; **46 horses ×4, 3 ×5, 1 ×8** | Replace the whole distribution; Dakota remains the maximum, so the qualitative answer holds |
| Q55 | 19 horses in recovery: 14×25min + 5×30min | **50** horses: **45×25min** + 5×30min | Change 19 → 50 and 14 → 45; the 5×30min group is correct |
| Q86 | Dakota 6 stages | 8 | Change 6 → 8 |
| Q93 | Dakota 6 stages | 8 | Change 6 → 8 |
| Q94 | lists 6 Dakota stage ids | 8 ids | Add the two `Training_Comp_*_Dakota_01` ids |

## Category C — every engagement now has an official result

This is the largest block and the most damaging, because the questions ask
about a phenomenon that **no longer exists**. In V9 each horse's `COMPETESIN`
count equals its `EventParticipation` count exactly (2 horses at 1/1, 47 at
2/2, Dakota at 5/5). Zero events lack participations; zero engagements lack
results.

| Q | Ground truth says | Live V9 | Proposed correction |
|---|---|---|---|
| Q62 | 3 events had entrants but no ranking | **none** | "Non — dans le graphe actuel, chaque engagement a un résultat officiel enregistré." |
| Q65 | 48 of 50 horses have an unranked engagement | **none** | "Non, ce n'est pas fréquent : le cas ne se produit jamais." |
| Q67 | `Event_SJ_2026_01` with 7 unranked entrants | no such case | "Aucune compétition : tous les engagements ont un résultat." |
| Q97 | each horse has exactly 1 result | 2 horses ×1, **47 ×2**, 1 ×5 | "Oui, les 50 chevaux ont au moins un résultat ; 47 en ont deux, Dakota en a cinq." |
| Q98 | entry without ranking is "the most frequent case" | never happens | "Oui — un classement existe toujours pour une inscription." |

**These five are the clearest candidates for correction:** the current prompt
answers them correctly for V9 and is penalised for it.

## Category D — counts that shifted

| Q | Ground truth says | Live V9 | Proposed correction |
|---|---|---|---|
| Q23 | 24 of 25 riders in pre-competition (Emma absent, Manon substitutes) | **25** riders, no exception | Remove the Emma/Manon exception; state 25 |
| Q34 | pre-competition has 24 riders with Manon replacing Emma | **25** riders | Same as Q23 |
| Q57 | 5 stages have no human supervisor | **0** | "Non, chaque étape a au moins un encadrant." |
| Q63 | `Event_Montpellier_Dr_2026` leads with 6 ranked results | **three-way tie at 7**: `Event_Pau_SJ_2026`, `Event_LeMans_Cross_2026`, `Event_SJ_2026_01` | Replace with the three-way tie at 7 |
| Q66 | 9 rider-with-two-horses cases | **11** — adds `Rider_Victor` at `Event_SJ_2026_01` and `Rider_Remi` at `Event_Pau_SJ_2026` | Change 9 → 11 and add the two cases |
| Q69 | `INVOLVES_ACTOR` 314, `DEPENDS_ON` 171, `TRAINS_IN` 171 | **355 / 207 / 207** (`COMPETESIN` 101 is correct) | Update the three counts; the ranking order is unchanged |
| Q8 | 45min for 31 horses | 45min for **32** preparation stages (51 stages, not 50) | Note the horse-vs-stage distinction: 31 horses is right, 32 stages exist |

---

## Ground truths independently confirmed CORRECT on V9

Verified by direct query — these are genuinely achievable and any failure on
them is a prompt problem, not drift:

Q6 (14 horses at 5 sessions/week), Q16 and Q50 (three sensors at 300Hz),
Q22 (24 riders in preparation), Q41 (Selle Français ×3), Q43 (8/9/7/1 rider
distribution), Q47 and Q92 (44/4/2 sensor distribution), Q49 (all CSV),
Q51 (hindlimb canon 7×0.015 and 5×0.02), Q54 (all competition stages
identical: 30min, Pic, ×1), Q56 (Dr Martin 48/49, Sophie 50/1), Q58 (7/7/6 by
discipline), Q59 (September, 5 events), Q60, Q61 (3 SJ + 1 dressage Pro
Elite), Q64, Q78 (6/5/5/4 by category), Q79 (Club Elite all dressage),
Q80 (`Event_SJ_01` → `Event_Pau_SJ_2026`), Q83 (Comet 75min), Q84 (Comet and
Ecume 55min), Q85 (Pixie 40min prep / 55min pre-comp), Q90 (5 horses at
30min), Q91 (corrected earlier with human approval), Q96 (4 cases).
