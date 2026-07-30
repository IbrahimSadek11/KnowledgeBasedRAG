# Tabular prompt changelog — `SQL_INSTRUCTION` in `tabular_chain.py`

Every rule added, changed or reverted, why, and its measured effect.
Only `backend/tabular_rag/tabular_chain.py` is touched for prompt work.
`gold_queries.py` and `data/tabular_rag/tabular.db` are never modified in this track
without separate human sign-off.

## Scoring model

`combined = (semantic_similarity + llm_judge_overall) / 2` over 100 questions.
Execution Accuracy (EX) compares generated vs gold SQL result sets on
`data/tabular_rag/tabular.db` (89 applicable / 11 N/A).

## Full-evaluation history

| # | Run file | Success | Semantic | Judge | Combined | EX | What changed before it |
|---|---|---|---|---|---|---|---|
| 0 | `..._182530` | 1.00 | 0.880 | 0.626 | **0.753** | **42.7%** (38/89) | pre–batch-2 gold; prompt frozen |
| 1 | `..._184212` | 1.00 | 0.881 | 0.629 | **0.755** | **47.2%** (42/89) | **data-corrected baseline** — all 11 gold id/shape fixes; **prompt untouched** |
| 2 | `..._185138` | 0.98 | 0.874 | 0.595 | **0.735** | **51.7%** (46/89) | Changes 1–2 (C aggregation + B shape) |
| 3 | `..._185842` | 0.99 | 0.880 | 0.583 | **0.732** | **52.8%** (47/89) | + Change 3 (E entity/stage) |
| 4 | `..._194814` | 1.00 | 0.885 | 0.652 | **0.768** | **58.4%** (52/89) | Changes 4–5 (regression narrowings + answer exhaustiveness) |
| 5 | `..._200301` | 1.00 | 0.890 | 0.659 | **0.774** | **58.4%** (52/89) | Change 6 (Q34 actor×phase rollup) |
| 6 | `..._202145` | 1.00 | 0.891 | 0.631 | **0.761** | **62.9%** (56/89) | Change 7 (D ties + F dual) |
| 7 | `..._203437` | 1.00 | 0.893 | 0.682 | **0.788** | **62.9%** (56/89) | Change 8 (histogram orientation) — **FINAL** |

Target checkpoint: EX **50–55%** and combined **78–80%**, then stop.

### STOP (2026-07-26) — prompt-only plateau

- **EX target met:** 52.8% ∈ [50, 55].
- **Combined target not met:** 0.732 ≪ 0.78–0.80; trend after baseline is
  **negative** (0.755 → 0.735 → 0.732) while judge falls (0.629 → 0.583).
- **Relocation:** EX gains on distribution/shape questions trade off losses
  elsewhere (same family as Graph RAG WITH-wall “fix here / break there”).
- **Not applying Change 4 (D tie LIMIT 1)** — further EX chasing would not
  recover combined; honest plateau.

**Best prompt-side EX run:** `tabular_eval_full100_ex_20260726_185842.json`
(EX 52.8%, combined 0.732).  
**Best combined under corrected gold:** baseline `..._184212` (0.755 / EX 47.2%).

---

## Change 1 — aggregation / distribution shape (root cause C)

**Problem:** For questions about how frequency/volume/intensity/category
varies across horses, the model returns bare `DISTINCT value` (or `AVG`)
instead of `value, COUNT(DISTINCT horse_id) … GROUP BY value`. Also
conflicts with an older “always SELECT stage_type + DISTINCT” habit when
the stage is already fixed in `WHERE`.

**Rule:** When the question asks how a property is distributed / how many
horses share each value (durée, fréquence, intensité, catégorie,
discipline counts), return one row per distinct value with a horse (or
row) count — never a bare list of values and never a single AVG when the
gold shape is a distribution.

**Correct form (lead with this):**
```sql
SELECT volume, COUNT(DISTINCT horse_id) AS horse_count
FROM trainings
WHERE stage_type = 'PreparationStage'
GROUP BY volume;
```

**Incorrect pattern (do not copy — described in prose only):** returning
only `SELECT DISTINCT volume` (or `AVG(volume_minutes)`) for the same
distribution question.

**Live verification (2026-07-26):**
- Q6 / Q8 / Q9 generated SQL EX-MATCH vs gold; live row counts
  (4→36/5→14), (40/45/50/55min), (55–75min) confirmed on `tabular.db`.
- Fresh: competition intensity COUNT; transition volume COUNT;
  event category COUNT — all distribution-shaped.
- Over-trigger fix: inventory Q75 (`DISTINCT intensity`) PASS after
  narrowing `AGGREGATION_SIGNAL_PHRASES` (removed bare durée/intensité).

**Regression (spot):** Q1–3, Q11, Q13, Q17–18, Q20–21, Q24, Q26–30, Q36,
Q41–42, Q52, Q57, Q60, Q62, Q67, Q75, Q77, Q100 PASS. Remaining FAILs
are pre-existing shape/entity issues (Q37 category vs discipline, Q53
COUNT DISTINCT stage_type, Q55 wrong stage, Q76 projection) — tracked
under B/E, not introduced by C.

**Measured effect:** pending full eval after Changes 1–2 (C+B).

---

## Change 2 — output shape / projection (root cause B)

**Problem:** Model returns related but wrong columns (stage_type instead
of training_id; rank without rider_id; sample_rate_hz instead of
sample_rate; surplus event attributes; actor column order).

**Rule:** Return exactly the columns named by the question; prefer
display text columns for labels; for actor×phase listings use
`(actor_id, actor_role, stage_type)` order.

**Correct form:**
```sql
SELECT training_id FROM trainings
WHERE horse_id = (SELECT horse_id FROM horses WHERE LOWER(name) = LOWER('Dakota'));
```

**Incorrect (prose only):** `SELECT DISTINCT stage_type` for the same
“quelles étapes” question.

**Measured effect (full eval `..._185138` after C+B):**
- EX **47.2% → 51.7%** (42→46/89) — inside 50–55% band.
- Combined **0.755 → 0.735** (down); success 100% → 98% (Q43/Q56
  generation crashes from projecting `stage_type` on `training_actors`
  without JOIN).
- EX gained: Q4,Q5,Q8,Q9,Q10,Q15,Q20,Q21,Q79,Q80.
- EX lost (relocation): Q7,Q12,Q37,Q47,Q65,Q99.

---

## Change 3 — entity / stage routing (root cause E)

**Problem:** Recovery questions hit PreparationStage; “who participates in
a training phase” returns horses instead of actors; 1st+2nd self-joins
omit same `rider_id`.

**Rule:** Recovery → `TransitionStage`; phase actors → people via
`training_actors`; dual-place finish requires same rider.

**Live check:** Q55 EX-MATCH (`TransitionStage` volume distribution
25min:45 / 30min:5). Actor JOIN no longer crashes Q56 generation path.

**Measured effect (full eval `..._185842`):** EX **51.7% → 52.8%** (+1);
combined **0.735 → 0.732** (flat/down). See STOP above — Change 4 (D) skipped.

---

## Change 4 — Part 1 regression narrowings (Q88 / Q53 / Q14 / Q34)

**Diagnoses (from `..._185842`):**
- **Q88:** French apostrophe in SQL string literal → `near "événement": syntax error` on all retries (generation failure, not ambiguity).
- **Q53:** `COUNT(DISTINCT stage_type)` from étape/DISTINCT confusion (OUTPUT SHAPE / inventory tension) → Zephyr/4 vs Dakota/8.
- **Q14:** `GROUP BY sensor_id LIMIT 5` made COUNT=5 instead of 50.
- **Q22:** EX already MATCH — judge drop was synthesis omitting Vet (Part 2).
- **Q34:** missing `DISTINCT` on actor×phase join → 250 vs 53 rows (SQL correctness).

**Rules:** ASCII SQL labels; COUNT(training_id) for programme d'étapes; count+examples via COUNT+GROUP_CONCAT; DISTINCT on phase-actor compare; narrow stage_type projection to true phase comparisons; people for “qui est le vétérinaire”; season event column set; classement horse+event AND-only; per-rider GROUP BY for un/plusieurs; aggregation gate rejects bare GROUP BY on distribution phrases.

**Live:** five targets fixed; prior CBE MATCH list 47/47 retained after collateral narrowings.

## Change 5 — Part 2 answer-synthesis exhaustiveness

**Problem:** EX=MATCH but low judge (Q6/Q8/Q22/Q4/Q5/…) — numbers/names summarized away.

**Rule:** Mirror Graph RAG Change 2 in `answer_question` — cite every number, enumerate all names (incl. Vet/Caretaker), report ties, state whole-group identity, sensor id examples only.

**Measured effect (full eval `..._194814`):** combined **0.732 → 0.768**; judge **0.583 → 0.652**; EX **52.8% → 58.4%** (47→52/89). Success 100%. Combined sits **just under** the 77% stop line — treat as Part-2 plateau / soft ceiling, not a mandate for more prompt churn.

---

## Change 6 — Q34 synthesis (actor×phase rollup; no SQL-shape touch)

**Diagnose:** EX MATCH but judge 0 — model miscounted 53 raw tuples (27/24 invented, Sophie folded into cavaliers, Manon contrast inverted). Siblings EX=MATCH/j<0.5 (Q38/Q39/Q57/Q67) are empty-result polarity cases — **no shared pattern**.

**Fix (answer path only):** `_actor_phase_rollup` feeds exact role×phase counts + set diffs into the answer prompt; one bullet: trust the résumé, count by `actor_role`. No `generate_sql` / gold changes.

**Live pre-check:** Q34 judge 0 → 0.50, combined ~0.46 → ~0.71; EX unchanged.

**Full eval `..._200301`:** Q34 combined **0.457 → 0.961** (j 0→1.00, EX MATCH). Headline: combined **0.768 → 0.774** (+0.6pp), EX **58.4% flat**, judge 0.652→0.659. Within ±1pp noise on the aggregate — stop; no further synthesis lever on this pass.

---

## Change 7 — Category D ties + Category F dual (final Tabular round)

**D (Q63, Q84):** Replace `ORDER BY … LIMIT 1` with `HAVING COUNT = MAX` / `WHERE col = MAX(col)`; `needs_tie_safe_check` retries on LIMIT 1 for superlatives.

**F (Q64, Q92) — not the same as ties:**
- Q64: majority/exceptions → per-horse `event_entries` counts (wrong grain was `GROUP BY event_id`).
- Q92: herd sensor-load → full histogram of per-horse sensor counts (LIMIT 1 / MIN-tie list wrong).

Live: Q63/Q84/Q64/Q92 EX MATCH before full eval.

**Full eval `..._202145`:** EX **58.4% → 62.9%** (52→56/89); all four D/F targets MATCH. Combined **0.774 → 0.761** (judge 0.659→0.631).

## Change 8 — histogram column-orientation (synthesis only; final)

**Judge drops DF vs pre-DF:** 13 questions. Mechanical D/F hit: F histogram SQL (Q47) + Part-2 exhaustiveness without column order → model inverted `(2,44)` as “44 capteurs”. Same pattern Q6 (36/14 as frequencies). Others (Q12, Q34, Q22, …) mostly SQL-same synthesis/judge variance or unrelated SQL drift — not forced.

**Fix:** one answer-prompt bullet: 1st col = level, 2nd = effectif. D/F SQL untouched.

**Full eval `..._203437` (FINAL Tabular number):** combined **0.788**, judge **0.682**, EX **62.9%** (56/89). Combined in 78–80% band; EX floor held. Stop.
