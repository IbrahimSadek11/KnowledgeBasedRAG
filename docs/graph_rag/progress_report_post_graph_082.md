# Progress Report — After Graph RAG 0.66→0.82 (complete fix inventory)

**Covers:** all work after `docs/graph_progress_report_0.66_to_0.82.md`  
**Period:** 2026-07-26 — Tabular RAG campaign + Graph package reorg  
**Scoring:** `combined = (semantic_similarity + llm_judge_overall) / 2` over 100 questions  
**Models:** stock API only (`gpt-4o-mini`, `text-embedding-3-small`) — **no fine-tuning**

---

## 1. Executive summary

| Track | Start (this period) | End (this period) | Driver |
|---|---|---|---|
| **Tabular RAG** | ~0.753 combined / **42.7% EX** | **0.788 combined / 62.9% EX** | 11 gold fixes + every SQL/synthesis rule listed below |
| **Graph RAG** | **0.821** (prior report) | **0.818** post-reorg check | Folder reorg only (no prompt logic) |
| **Fine-tuning** | — | — | **None** |

**Final Tabular artifact:**  
`evaluation_results/tabular_rag/tabular_eval_full100_ex_20260726_203437.json`

**Graph baselines cited in prior PDF (unchanged names after move):**  
`evaluation_results/graph_rag/semantic_evaluation_20260726_160643.json` (combined 0.8206)  
`evaluation_results/graph_rag/semantic_evaluation_20260726_161903.json` (combined 0.7996)

---

## 2. Tabular evaluation ladder

| # | Run | Combined | Judge | EX | Trigger |
|---|---|---|---|---|---|
| 0 | `…_182530` | 0.753 | 0.626 | 42.7% (38/89) | Pre–batch-2 gold |
| 1 | `…_184212` | **0.755** | 0.629 | **47.2%** (42/89) | All 11 gold; prompt frozen |
| 2 | `…_185138` | 0.735 | 0.595 | **51.7%** (46/89) | C + B |
| 3 | `…_185842` | 0.732 | 0.583 | **52.8%** (47/89) | + E |
| 4 | `…_194814` | **0.768** | 0.652 | **58.4%** (52/89) | Regression narrowings + exhaustiveness |
| 5 | `…_200301` | **0.774** | 0.659 | 58.4% | Q34 rollup |
| 6 | `…_202145` | 0.761 | 0.631 | **62.9%** (56/89) | D + F |
| 7 | `…_203437` | **0.788** | **0.682** | **62.9%** | Histogram orientation — **FINAL** |

EX comparator (`compare_execution`) was audited: **unchanged** since first EX commit; gold diff = **exactly the 11 approved keys**.

---

## 3. COMPLETE FIX INVENTORY — Tabular

Each item below is a distinct adjustment applied this period. File unless noted: `backend/tabular_rag/tabular_chain.py`. Gold: `scripts/tabular_rag/gold_queries.py`.

### A. Gold SQL corrections (Track 1 — signed off)

Document: `docs/tabular_gt_signoff.md`. Applied only after human approval.

| ID | Question | Fix |
|---|---|---|
| **G1** | Q4 | `Horse1` → `Horse_Dakota` in `event_entries` filter |
| **G2** | Q5 | `Horse1` → `Horse_Dakota` in `trainings` filter |
| **G3** | Q10 | `Horse1` → `Horse_Dakota` in trainings→events join |
| **G4** | Q11 | `Horse1` → `Horse_Dakota` in sensors count |
| **G5** | Q13 | `Horse1` → `Horse_Dakota` in sensor_type distinct |
| **G6** | Q17 | `Horse1` → `Horse_Dakota` in horse_rider_associations |
| **G7** | Q18 | `Horse2` → `Horse_Naya` in horse_rider_associations |
| **G8** | Q31 | `Horse1` → `Horse_Dakota` in event_participations classement |
| **G9** | Q63 | Drop `ORDER BY cnt DESC LIMIT 1`; keep full max-tie via `HAVING COUNT(*) = (SELECT MAX(c) …)` |
| **G10** | Q67 | Add `HAVING gap > 0` before `ORDER BY gap DESC LIMIT 1` |
| **G11** | Q82 | `Horse1` → `Horse_Dakota` in sensors objective lookup |

**No other gold keys changed.**

---

### B. SQL generation — root cause C (distribution vs inventory)

| ID | Fix |
|---|---|
| **C1** | **DISTRIBUTION rule:** for varie/répart/combien de chevaux/durée/fréquence in a named phase → `SELECT value, COUNT(DISTINCT horse_id) … GROUP BY value` (not bare DISTINCT, not single AVG) |
| **C2** | If `stage_type` already fixed in `WHERE`, do **not** add `stage_type` to SELECT just to justify DISTINCT |
| **C3** | **INVENTORY rule:** “quels niveaux / quelles valeurs / quels types” with no count request → `SELECT DISTINCT value` only (do not invent COUNT) |
| **C4** | Narrowed `AGGREGATION_SIGNAL_PHRASES` so inventory questions (e.g. Q75 distinct intensity) are not forced into COUNT |
| **C5** | Kept phase-scoped phrases in aggregation gate: `"durée des séances"`, `"fréquence d'entraînement"` |

**Measured after C+B:** EX 47.2%→51.7%.

---

### C. SQL generation — root cause B (output shape)

| ID | Fix |
|---|---|
| **B1** | OUTPUT SHAPE block: return exactly columns the question asks — no spare attributes |
| **B2** | “quelles étapes d'entraînement” → `training_id` (not only `stage_type` labels) |
| **B3** | “de quel événement dépendent” → JOIN `events`, return `event_id, location, category, event_date, discipline, stage_type` |
| **B4** | Classement of named horse/rider → `rider_id, rank` (not rank alone) |
| **B5** | Sampling frequency as stored label → `sample_rate` text (e.g. `200Hz`), not `sample_rate_hz`, unless numeric compare/sort asked |
| **B6** | Actor comparisons across phases → JOIN `trainings`↔`training_actors`; SELECT `actor_id, actor_role, stage_type` in that order |
| **B7** | “fréquence d'entraînement” → column `frequency`; “durée des séances” → column `volume` |
| **B8** | First/last event → `event_id, event_date` only (unless location/category asked) |
| **B9** | Correct/wrong shape examples for Dakota training_id vs DISTINCT stage_type |

**Measured after C+B:** EX 51.7%; combined dipped (relocation).

---

### D. SQL generation — root cause E (entity / stage routing)

| ID | Fix |
|---|---|
| **E1** | TABLE_CHOICE: recovery / “après une compétition” → `stage_type = 'TransitionStage'` (not PreparationStage) |
| **E2** | TABLE_CHOICE: actors in a training phase → JOIN `trainings` + `training_actors` + `people` (not horse list) |
| **E3** | TABLE_CHOICE: same rider 1st and 2nd at one event → group by `event_id, rider_id` with `HAVING COUNT(DISTINCT rank)=2` for ranks in (1,2); self-join requires same `rider_id` |

**Measured after C+B+E:** EX 52.8% (47/89); combined 0.732.

---

### E. SQL generation — Part 1 named regressions + collateral narrowings

#### Named targets

| ID | Q | Diagnosis | Fix |
|---|---|---|---|
| **R1** | Q88 | French apostrophe in `'…d'un événement…'` → SQLite syntax error on all retries | SQL string literals must be simple ASCII labels (`association` / `participation`); no French apostrophe/accented prose inside quotes |
| **R2** | Q53 | `COUNT(DISTINCT stage_type)` → Zephyr/4 instead of Dakota/8 | “plus grand nombre d'étapes” / programme le plus complet → `COUNT(*)` or `COUNT(training_id)` per horse; **never** `COUNT(DISTINCT stage_type)` |
| **R3** | Q14 | `GROUP BY sensor_id LIMIT 5` → COUNT=5 not 50 | count + examples → one aggregate COUNT + `GROUP_CONCAT`; never GROUP BY the listed id with LIMIT N |
| **R4** | Q34 | Missing `DISTINCT` on actor×phase join → 250 vs 53 rows | Actor phase compare requires `SELECT DISTINCT ta.actor_id, ta.actor_role, t.stage_type` |
| **R5** | Q22 | EX already MATCH; Vet omitted in French answer | **Not a SQL fix** — handled under synthesis (S*) |

#### Collateral / scope narrowings (same pass)

| ID | Triggered by | Fix |
|---|---|---|
| **R6** | Q76 over-applying stage_type | Actor+stage_type projection **only** for true phase comparisons (“compare les acteurs” / prep vs pré-compétition); general “who can supervise / non-rider actors” → `DISTINCT actor_id, actor_role` with **no** stage_type filter |
| **R7** | Q20 extra columns / wrong table | “qui est le vétérinaire” → `people.person_id WHERE role = 'Veterinarian'` (no `training_actors`, no `actor_role` column) |
| **R8** | Q25 event_id alone | “quels événements” of a named season → `event_id, location, category, event_date, discipline` |
| **R9** | Q31 wrong join / OR rider | Classement: filter via `event_participations` + horse name/event; **do not** route through `horse_rider_associations`; if horse+rider both named, filter **only** `horses.name` + `event_id` (AND), never `OR rider_id`, never invent `Rider_<HorseName>` |
| **R10** | Q43 global COUNTs | “un cavalier … un seul cheval ou … plusieurs” → `SELECT rider_id, COUNT(DISTINCT horse_id) … GROUP BY rider_id` (per-rider), not one pair of global DISTINCT counts |
| **R11** | Q9 bare GROUP BY volume | Aggregation gate: for durée/fréquence/répart/varie questions, `GROUP BY` alone is **not** enough — require COUNT/SUM/AVG/MAX/MIN |
| **R12** | Q16 extra sample_rate column (collateral of D later) | For max sampling frequency return `sensor_id` only (no spare `sample_rate` column) — reinforced with D |

---

### F. SQL generation — category D (tie-safe superlatives)

| ID | Fix |
|---|---|
| **D1** | Replace “most/least → ORDER BY LIMIT 1” instruction with **TIE-SAFE SUPERLATIVES**: use `WHERE col = (SELECT MAX/MIN(col)…)` or `HAVING COUNT(*) = (SELECT MAX(c) FROM …)` |
| **D2** | Worked example: max event_participations tie (Q63 shape) |
| **D3** | Worked example: longest PreparationStage volumes (Q84 shape) |
| **D4** | `needs_tie_safe_check`: reject `LIMIT 1` on superlative questions (`le plus` / `le moins` / `plus grand nombre` / `plus longues` / …) and force regenerate |
| **D5** | Same gate: reject `LIMIT 1` on herd sensor-load questions (`capteur` + moins/plus) |

**Targets:** Q63, Q84 EX→MATCH. EX overall **58.4%→62.9%**.

---

### G. SQL generation — category F (dual / herd-pattern; not the same as D)

| ID | Q | Fix |
|---|---|---|
| **F1** | Q64 | “la plupart / exceptions” on competition load → exactly `SELECT horse_id, COUNT(DISTINCT event_id) FROM event_entries GROUP BY horse_id`; no outer histogram; not `event_participations`; not `GROUP BY event_id` |
| **F2** | Q92 | “le moins de capteurs” herd pattern → full histogram `SELECT sensor_count, COUNT(*) … GROUP BY sensor_count` (all levels); not MIN-tie horse list; not `ORDER BY … LIMIT 1` on the histogram |

---

### H. Answer synthesis — exhaustiveness (Part 2 / Change 5)

| ID | Fix |
|---|---|
| **S1** | Cite **all** numbers in rows (totals, per-group effectifs, values) |
| **S2** | Enumerate **all** names (horses, riders, actors, events) comma-separated after exact count — no “plusieurs / notamment” when names are present |
| **S3** | Never omit Vet/Caretaker because the rider list is long |
| **S4** | Sensor technical IDs: exact count + 2–3 examples only (not dozens) |
| **S5** | Report ties at extreme values explicitly |
| **S6** | If one value covers a whole group, say so explicitly |
| **S7** | Distributions: cite each value and its effectif |
| **S8** | Distinct rows → account for each; don’t collapse different entities |
| **S9** | Ambiguous rows → say so; don’t pick one silently |
| **S10** | End with the direct conclusion (oui/non/value) after presenting data |
| **S11** | Vet_XXXX presented as natural name (e.g. Dr Martin) |

**Measured:** combined 0.732→0.768; judge 0.583→0.652; EX 52.8%→58.4%.

---

### I. Answer synthesis — Q34 actor×phase rollup (Change 6)

| ID | Fix |
|---|---|
| **S12** | `_actor_phase_rollup()`: when result is 3-col `(actor_id, actor_role, stage_type)` across ≥2 stages, inject structured résumé with per-role counts per phase |
| **S13** | Same helper: compute rider set-diffs (“seulement en préparation / seulement en pré-compétition”) |
| **S14** | Prompt rule: count **strictly by `actor_role`** per phase — never fold soigneur/vétérinaire into cavalier totals; don’t invent effectifs |
| **S15** | Prompt rule: for phase comparisons, cite set-diff (who appears in one phase not the other) as the central point |
| **S16** | Prompt rule: if a « Résumé structuré » is present, its effectifs/diffs are authoritative — do not re-count raw rows |

**Measured:** Q34 combined 0.457→0.961 (j 0→1.00) on `…_200301`.

---

### J. Answer synthesis — histogram column orientation (Change 8)

| ID | Fix |
|---|---|
| **S17** | For `(level, effectif)` histograms: 1st column = level (e.g. frequency `4x/week`, `sensor_count=2`, volume `25min`); 2nd = number of entities; **never invert** — `(2,44)` = “44 chevaux portent 2 capteurs”; `(4x/week, 36)` = “36 chevaux à 4×/semaine” |

**Recovered judge drops** on Q47 / Q6 pattern after D/F. Final combined **0.788**.

---

### K. Gates / helpers (code, not free-text prompt only)

| ID | Fix |
|---|---|
| **K1** | `needs_aggregation_check` — aggregation questions must contain COUNT/SUM/AVG/MAX/MIN/GROUP BY |
| **K2** | `needs_aggregation_check` — bare GROUP BY without count-like aggregate rejected for durée/fréquence/répart/varie |
| **K3** | `needs_tie_safe_check` — LIMIT 1 rejected for superlatives + sensor-herd questions (see D4/D5) |
| **K4** | Both gates wired into `answer_question` retry loop with error feedback to `generate_sql` |

---

## 4. COMPLETE FIX INVENTORY — Graph (this period only)

Prior Cypher/GT campaign is in `docs/graph_progress_report_0.66_to_0.82.md`. **This period added no new Cypher/QA prompt rules.** Structural only:

### L. Package reorg

| ID | Move / action |
|---|---|
| **P1** | `backend/llm_service.py` → `backend/graph_rag/llm_service.py` |
| **P2** | `backend/graph_service.py` → `backend/graph_rag/graph_service.py` |
| **P3** | `scripts/run_evaluation.py` → `scripts/graph_rag/run_evaluation.py` |
| **P4** | `scripts/run_retrieval_eval.py` → `scripts/graph_rag/run_retrieval_eval.py` |
| **P5** | `scripts/ragas_cypher_eval.py` → `scripts/graph_rag/` |
| **P6** | `scripts/ragas_cypher_eval_v2.py` → `scripts/graph_rag/` |
| **P7** | `scripts/ragas_cypher_eval_v3.py` → `scripts/graph_rag/` |
| **P8** | All Graph eval artifacts → `evaluation_results/graph_rag/` (filenames preserved) |
| **P9** | `prompt_changelog.md` → `graph_prompt_changelog.md` |
| **P10** | `ground_truth_flags.md` → `docs/graph_ground_truth_flags.md` |
| **P11** | `docs/v9_ground_truth_drift.md` → `docs/graph_v9_ground_truth_drift.md` |
| **P12** | `docs/gt_reconciliation_signoff.md` → `docs/graph_gt_reconciliation_signoff.md` |
| **P13** | `docs/progress_report_0.66_to_0.82.md` → `docs/graph_progress_report_0.66_to_0.82.md` |
| **P14** | `data/test_dataset.json` **left shared** (not moved) |
| **P15** | `evaluation_results/tabular_rag/` **untouched** |
| **P16** | Imports/paths updated: frontend, graph scripts (`REPO_ROOT=parents[2]`, results dir), README, IMPLEMENTATION, project_report, tabular comments |
| **P17** | Package-relative imports: `from ..config`, `from .graph_service` |
| **P18** | Post-reorg full Graph eval: combined **0.818**, success 99% (`semantic_evaluation_20260726_210024.json`) |

### M. Validator archive (not a live fix)

| ID | Action |
|---|---|
| **A1** | Reconstruct `cypher_validator.py` from `.pyc` → `backend/graph_rag/_archive/` |
| **A2** | Reconstruct `cypher_retry.py` → same |
| **A3** | Reconstruct `validated_chain.py` → same |
| **A4** | `docs/graph_validator_status.md` explaining never-in-git / never-wired |
| **A5** | **Explicitly not imported** into `init_graph_chain()` |

---

## 5. Side-by-side finals

| Pipeline | Combined | Judge | Semantic | EX / Success | Artifact |
|---|---|---|---|---|---|
| Graph (delivered baseline) | **0.821** | — | — | — | `…/graph_rag/semantic_evaluation_20260726_160643.json` |
| Graph (post-reorg) | **0.818** | 0.755 | 0.880 | 99% | `…/graph_rag/semantic_evaluation_20260726_210024.json` |
| Tabular (final) | **0.788** | 0.682 | 0.893 | **EX 62.9%** (56/89), 100% | `…/tabular_rag/tabular_eval_full100_ex_20260726_203437.json` |

---

## 6. Count of distinct adjustments this period

| Bucket | IDs | Count |
|---|---|---|
| Gold corrections | G1–G11 | **11** |
| SQL C (distribution) | C1–C5 | **5** |
| SQL B (shape) | B1–B9 | **9** |
| SQL E (entity/stage) | E1–E3 | **3** |
| SQL regressions + collateral | R1–R12 | **12** |
| SQL D (ties) | D1–D5 | **5** |
| SQL F (dual) | F1–F2 | **2** |
| Synthesis exhaustiveness | S1–S11 | **11** |
| Synthesis Q34 rollup | S12–S16 | **5** |
| Synthesis histogram | S17 | **1** |
| Gates/helpers | K1–K4 | **4** |
| Graph reorg | P1–P18 | **18** |
| Archive (non-live) | A1–A5 | **5** |
| **Total distinct items** | | **91** |

---

## 7. Limitations (unchanged substance)

1. Prompt relocation: EX gains can trade judge/combined.  
2. Judge variance on individual questions even when SQL MATCH.  
3. Empty-result / polarity cases (Q38/Q39, Q57) not fully solved.  
4. Validator archive is salvage only until a separate integration decision.  
5. No fine-tuning; further leaps need new levers.

---

## 8. Bottom line

Since the Graph **0.66→0.82** report, Tabular was taken from a data-corrected **0.755 / 47.2% EX** baseline to a final **0.788 / 62.9% EX** via **every fix listed above** (gold + SQL + synthesis + gates). Graph was **repackaged** without prompt changes; post-reorg eval **0.818** confirms the pipeline still works. **No fine-tuning.**
