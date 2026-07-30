# V9 graph vs `test_dataset.json` v8.7 — ground-truth drift report

Produced 2026-07-26 during the Cypher-prompt audit. **No file was modified** —
this documents what `data/test_dataset.json` would need if it were re-audited
against the current V9 graph.

Every figure below was obtained by direct Cypher query against the live Neo4j
database, not through the RAG chain.

## Live V9 graph inventory

516 nodes, 1460 relationships.

| Label | Count | Label | Count |
|---|---|---|---|
| InertialSensors | 108 | Horse | 50 |
| EventParticipation | 101 | TransitionStage | 50 |
| CompetitionStage | 55 | Rider | 25 |
| PreCompetitionStage | 51 | ShowJumping / Cross / Dressage | 7 / 7 / 6 |
| PreparationStage | 51 | ExperimentalObjective | 2 |
| Withers / Sternum / CanonOfForelimb / CanonOfHindlimb | 50 / 26 / 20 / 12 | CompetitiveSeason / Veterinarian / Caretaker | 1 / 1 / 1 |

Relationship counts: INVOLVESACTOR 355, TRAINSIN 207, DEPENDSON 207,
ISATTACHEDTO 108, ISUSEDFOR 108, COMPETESIN 101, HASPARTICIPATION 101,
HASRIDER 101, HASHORSE 101, ASSOCIATEDWITH 51, INSEASON 20.

## Questions whose ground truth no longer matches V9

### 1. Sensor identifiers were renamed (V8 bare form no longer exists)

`IMU_Withers_01`, `IMU_CanonFore_01`, `IMU_CanonHind_01`, `IMU_Sternum_01`
return zero rows. V9 uses horse-scoped identifiers.

| Q | Ground truth references | V9 actual |
|---|---|---|
| Q38 | `IMU_Withers_01` | `IMU_Withers_Dakota_01` → GaitClassif_01 |
| Q39 | `IMU_CanonFore_01` | `IMU_CanonFore_Dakota_01` → FatigueDetection |
| Q14 | `IMU_Withers_01 pour Dakota` | `IMU_Withers_Dakota_01`; also `IMU_Withers_Comet_01`, `IMU_Withers_Apollon_01` (those two are correct) |
| Q82 | all four bare ids | `IMU_Withers_Dakota_01` + `IMU_CanonHind_Dakota_01` → GaitClassif_01; `IMU_CanonFore_Dakota_01` + `IMU_Sternum_Dakota_01` → FatigueDetection |

These four questions cannot score above zero on the identifier detail while the
ground truth keeps the V8 names.

### 2. Training-stage inventory grew

| Q | Ground truth | V9 actual |
|---|---|---|
| Q5 | Dakota follows 6 stages | 8 (adds `Training_Comp_Cross2026_Dakota_01`, `Training_Comp_Dress01_Dakota_01`) |
| Q52 | 51 prep / 51 pre-comp / 50 competition / 19 transition | 51 / 51 / **55** / **50** |
| Q53 | Dakota 6; 31 horses ×3, 18 ×4 | Dakota **8**; **46 horses ×4, 3 ×5, 1 ×8** |
| Q55 | 19 horses in recovery, 14×25min + 5×30min | **50** horses, **45×25min** + 5×30min |
| Q86, Q93 | Dakota 6 stages | 8 |
| Q94 | lists 6 Dakota stage ids | 8 ids (the two `Training_Comp_*_Dakota_01` above) |

### 3. Every engagement now has a matching official result

This is the largest single block. In V9 each horse's `COMPETESIN` count equals
its `EventParticipation` count exactly: 2 horses at 1/1, 47 horses at 2/2,
Dakota at 5/5. There are **zero** events without participations and **zero**
engagements without results.

| Q | Ground truth | V9 actual |
|---|---|---|
| Q62 | 3 events with entrants but no ranking | none |
| Q65 | 48 of 50 horses have an unranked engagement | none |
| Q67 | Event_SJ_2026_01, 7 entrants no result | no such event |
| Q97 | each horse has exactly 1 result | 2 horses ×1, 47 ×2, 1 ×5 |
| Q98 | entry without ranking is "the most frequent case" | never happens in V9 |

Any prompt will answer these "no such case exists", which is correct for V9 and
scores zero against the v8.7 text.

### 4. Counts that shifted

| Q | Ground truth | V9 actual |
|---|---|---|
| Q23, Q34 | 24 of 25 riders in pre-competition (Emma absent) | **25** riders |
| Q57 | 5 stages with no human supervisor | **0** |
| Q63 | Event_Montpellier_Dr_2026, 6 results (unique max) | **three-way tie at 7**: Pau, LeMans, SJ_2026_01 |
| Q66 | 9 rider-two-horses-same-event cases | **11** (adds Rider_Victor at SJ_2026_01, Rider_Remi at Pau) |
| Q69 | INVOLVES_ACTOR 314, DEPENDS_ON 171, TRAINS_IN 171 | **355 / 207 / 207** (COMPETESIN 101 still correct) |
| Q8 | 45min for 31 horses | 45min for **32** stages (51 prep stages, not 50) |

## Ground truths verified still correct on V9

Q6 (14 horses at 5 sessions/week), Q16 and Q50 (300Hz ×3), Q22 (24 riders in
preparation), Q41 (Selle Français ×3), Q43 (8/9/7/1 rider distribution),
Q47 and Q92 (44/4/2 sensor distribution), Q49 (all CSV), Q51 (hindlimb canon
7×0.015, 5×0.02), Q54 (all competition stages identical: 30min, Pic, ×1),
Q56 (Dr Martin 48/49, Sophie 50/1), Q58 (7/7/6 by discipline), Q59 (September,
5 events), Q60, Q61 (3 SJ + 1 dressage Pro Elite), Q64, Q78 (6/5/5/4 by
category), Q79 (Club Elite all dressage), Q80 (Event_SJ_01 → Event_Pau_SJ_2026),
Q83 (Comet 75min), Q84 (Comet and Ecume 55min), Q85 (Pixie 40/55min),
Q90 (5 horses at 30min), Q91 (corrected earlier), Q96 (4 cases).

## Impact

Roughly 16-18 of the 100 questions have ground truths that V9 contradicts.
They cluster in the anti-join and comparative shapes, which is why those
categories score lowest regardless of prompt quality. Correcting them would be
the single largest available score improvement, but requires editing
`test_dataset.json`, which was explicitly out of scope.
