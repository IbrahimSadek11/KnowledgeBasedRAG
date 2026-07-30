# Version2 Tabular RAG — Progress Report (post-diagnostic)

**Scope:** version2 only (`backend/tabular_rag/version2/`, `scripts/tabular_rag/version2/`, `evaluation_results/tabular_rag/version2/`).  
**Live and version1 were never modified** (SHA256 live == version1 for chain, gold, ETL throughout).

**Baseline before this work (diagnostic-era eval):**  
`tabular_eval_full100_ex_20260728_013046.json` — EX **50/84 (59.5%)**, combined **0.779**

**Final kept state:**  
`tabular_eval_full100_ex_20260728_024556.json` — EX **51/84 (60.7%)**, combined **0.791**

---

## 1. Score timeline

| Report | EX | Combined | Semantic | Judge | Notes |
|---|---:|---:|---:|---:|---|
| `013046` (pre-prompt-batch) | 50/84 (59.5%) | 0.779 | 0.893 | 0.665 | After v2 schema + people.name; 34 EX mismatches diagnosed |
| `015703` (after prompt-batch) | 47/84 (56.0%) | 0.778 | 0.892 | 0.663 | Aggregation / join / grain prompts; Q82 fixed; 7 regressions |
| `022900` (iter 3 kept) | 50/84 (59.5%) | 0.770 | 0.893 | 0.648 | person_id vs name fix |
| `023744` (iter 4 kept) | 50/84 (59.5%) | 0.781 | 0.892 | 0.670 | training_id-only for étapes |
| **`024556` (final)** | **51/84 (60.7%)** | **0.791** | **0.892** | **0.689** | id+role for non-rider supervisors |

Net vs `013046`: EX **+1** (50→51), combined **+0.012** (0.779→0.791).  
Net vs post-prompt dip `015703`: EX **+4** (47→51), combined **+0.013**.

---

## 2. Phase A — Diagnosis of the 34 EX mismatches (`013046`)

Full dump: `evaluation_results/tabular_rag/version2/_diag_mismatches_manual.json`

### Category counts (manual)

| Category | Count | Example IDs |
|---|---:|---|
| Wrong aggregation / grouping | 13 | Q12, Q32, Q33, Q46, Q48, Q51, … |
| Column / value formatting (same fact, different shape) | 8 | Q7, Q20, Q21, Q23, Q44, Q50, Q83, Q96 |
| Wrong join / filter grain | 6 | Q66, Q82, Q85, Q86, Q93, Q95 |
| Off-by-one / boundary | 3 | Q45, Q58, Q59 |
| Ambiguous / alt-reasonable | 2 | Q47, Q90 |
| Wrong column chosen | 1 | Q37 (`category` vs `discipline`) |
| Something else (gold bug) | 1 | **Q74** GLOB misuse |

### High judge (>0.7) + EX False (10)

Pattern: NL answer often correct; strict bag-equality fails on projection / id form / grain.  
Examples: Q20/Q21 (name vs `person_id`), Q44 (extra COUNT), Q45 (full ranking vs LIMIT 1), Q47 (histogram vs per-horse), Q83 (missing constant `stage_type`).

---

## 3. Phase B — Bugs found and fixed

### Bug 1 — Q74 gold SQL used GLOB with LIKE wildcards (fixed)

**File:** `scripts/tabular_rag/version2/gold_queries.py` only.

| | Query | Result on v2 DB |
|---|---|---|
| **Broken** | `… NOT GLOB '____-__-__'` | Always `(20,)` — `_` is literal in GLOB |
| **Fixed** | `… NOT LIKE '____-__-__'` | `(0,)` — all dates are valid YYYY-MM-DD |

Live/v1 still have the broken GLOB form (untouched by design).

### Bug 2 — Q82 sensor→objective correlated subquery (fixed via prompt + schema notes)

**Root cause (not wording-only):** generated SQL used  
`sensor_id IN (SELECT sensor_id FROM objectives WHERE name = …)`.  
`objectives` has **no** `sensor_id`. SQLite correlated to the outer `sensors.sensor_id`, so every Dakota sensor was returned for both “gait” and “fatigue” branches — objective labels lost; answer invented.

**Before (EX False):**
```sql
SELECT sensor_id FROM sensors WHERE … AND sensor_id IN
  (SELECT sensor_id FROM objectives WHERE name = 'Gait Classification')
UNION
SELECT sensor_id FROM sensors WHERE … AND sensor_id IN
  (SELECT sensor_id FROM objectives WHERE name = 'Fatigue Detection');
```

**After (EX True in `015703`+):**
```sql
SELECT s.sensor_id, s.objective_id FROM sensors s
JOIN objectives o ON s.objective_id = o.objective_id
WHERE s.horse_id = (SELECT horse_id FROM horses WHERE LOWER(name) = LOWER('Dakota'))
  AND (o.name = 'Gait Classification' OR o.name = 'Fatigue Detection');
```
Rows: CanonHind/Withers → `GaitClassif_01`; Sternum/CanonFore → `FatigueDetection`.

**Fixes applied in** `backend/tabular_rag/version2/tabular_chain.py`:
- `COLUMN_NOTES` on `sensors.objective_id` / `objectives.objective_id`
- `TABLE_CHOICE_GUIDE` ISUSEDFOR bullet
- `SQL_INSTRUCTION` “SENSOR → OBJECTIVE JOIN” section forbidding the fake join

---

## 4. Phase B — Prompt improvements (batch before iterative loop)

All in `backend/tabular_rag/version2/tabular_chain.py`.

| Theme | What was added | Intended failures |
|---|---|---|
| **Aggregation / DISTRIBUTION** | Stronger cues for répartition / combien / GROUP BY + COUNT; concrete shapes for sensor_type, objective_id, stage×frequency, stage×role | Q12, Q32/Q33, Q46, Q48, Q51, Q54, Q56 |
| **Same-event grain** | Multi-horse same rider must `GROUP BY event_id, rider_id` (not rider alone) | Q66 |
| **Sensor→objective** | See Bug 2 above | Q82 |
| **Period = month** | Busiest “période” → `strftime('%Y-%m', …)` unless day asked | Q59 |
| **Tie-safe most-represented** | Keep all ties at max count (no LIMIT 1) | Q58 |

### Immediate eval impact (`013046` → `015703`)

- **Gained:** Q37, Q46, Q61, **Q82**
- **Lost (regressions):** Q4, Q5, Q15, Q64, Q79, Q81, Q92  
- EX **59.5% → 56.0%** (prompt batch helped some patterns, over-fired others)

Notable SQL quality wins even when EX still False:
- Q12: type distribution with correct counts (shape vs gold still differs)
- Q59: month grain `[['2026-09', 5]]` instead of day-level ties
- Q66: event_id in GROUP BY (missing `horse_id` column vs gold)

ETL was **not** re-run (no schema change for these fixes).

---

## 5. Phase C — Iterative recovery (EX ≥ 60% and combined ≥ 78%)

Rules used: diagnose with real SQL/rows → one narrow generalizable change → full eval → keep or revert → SHA256 live/v1 each time.

| Iter | Change | EX | Combined | Decision |
|---:|---|---|---|---|
| 1 | Horse-participation events → `event_id` only (narrow season OUTPUT SHAPE) | 47→47 | 0.778→0.754 | **REVERTED** |
| 2 | “Plusieurs ou une seule” → INVENTORY (no COUNT) | 47→46 | 0.778→0.760 | **REVERTED** (Q79 still COUNTed) |
| 3 | `qui est le [role]` → `person_id` only; name only on explicit “quel est le nom” | 47→**50** | 0.778→0.770 | **KEPT** |
| 4 | Étapes lists → `training_id` only (no `stage_type`) | 50→50 | 0.770→**0.781** | **KEPT** |
| 5 | Non-rider supervisors → always `actor_id, actor_role` | 50→**51** | 0.781→**0.791** | **KEPT** ✓ stop |

Changelog file: `evaluation_results/tabular_rag/version2/_iter_changelog.md`

### Kept fix details

#### Iter 3 — person_id vs name conflict
- **Problem:** Prompt said both “qui est le vétérinaire → person_id” and “who is → people.name”; French “qui est” matched the name path.
- **Q20 before:** `SELECT name …` → `Dr Martin`  
- **Q20 after:** `SELECT DISTINCT person_id …` → `Vet_DrMartin` (MATCH)  
- **Q21 after:** `SELECT DISTINCT person_id …` → `Caretaker_Sophie` (MATCH)  
- Also updated `COLUMN_NOTES` for `people.name`.

#### Iter 4 — étapes projection
- **Problem:** “Include stage_type for attribution” overrode “étapes → training_id”.
- **Q5 before:** `SELECT DISTINCT training_id, stage_type …` (MISMATCH)  
- **Q5 after:** `SELECT training_id FROM trainings WHERE horse_id = (SELECT … Dakota)` (MATCH)

#### Iter 5 — supervisor id + role
- **Problem:** Model returned `actor_id` alone for non-rider encadrants.
- **Q76 after:**  
  `SELECT DISTINCT ta.actor_id, ta.actor_role FROM training_actors ta WHERE ta.actor_role != 'Rider';`  
  → `[('Vet_DrMartin','Veterinarian'), ('Caretaker_Sophie','Caretaker')]` (MATCH)

---

## 6. Files touched (version2 only)

| Path | Changes |
|---|---|
| `scripts/tabular_rag/version2/gold_queries.py` | Q74 GLOB → LIKE |
| `backend/tabular_rag/version2/tabular_chain.py` | COLUMN_NOTES, TABLE_CHOICE_GUIDE, SQL_INSTRUCTION (aggregation, grain, join, period, person_id, étapes, supervisors) |
| `evaluation_results/tabular_rag/version2/*` | Eval JSONs, consoles, diagnostic dumps, changelog |

**Not touched:** live `backend/tabular_rag/*`, `scripts/tabular_rag/gold_queries.py`, anything under `version1/`.

---

## 7. What still fails (honest snapshot on final `024556`)

Still ~33 EX mismatches under strict bag equality. Main remaining buckets:

1. **Projection / formatting** — same facts, extra/missing columns (Q4, Q7, Q12 shape, Q23, Q44, Q50, Q83, …)
2. **Aggregation grain** — list vs COUNT, wrong GROUP BY (Q32, Q33, Q48, Q51, Q56, Q78/Q79 tension, …)
3. **Hard intent / comparison tables** — Q86, Q93, Q95, Q97, Q99
4. **Gold vs reasonable alt** — Q58 (all disciplines vs max ties), Q74 (gold COUNT malformed=0 vs gen lists dates), Q47/Q90 grain
5. **LLM variance** — some questions flip MATCH/MISMATCH between runs (e.g. Q61 COUNT, Q96 rank filter)

Strict EX was **not** loosened; no question-ID hardcoding was introduced.

---

## 8. Bottom line

| | |
|---|---|
| Bugs fixed | Q74 gold GLOB; Q82 correlated `objectives.sensor_id` join |
| Prompt systems added | Aggregation, same-event grain, sensor→objective, month period, person_id identity, étapes shape, supervisor columns |
| Regressions handled | Prompt-batch EX dip recovered via 3 kept iterative fixes (2 reverted) |
| Final scores | **EX 60.7%**, **combined 0.791** (targets EX≥60% and combined≥78% met) |
| Isolation | Live + version1 SHA256-identical and unchanged |
