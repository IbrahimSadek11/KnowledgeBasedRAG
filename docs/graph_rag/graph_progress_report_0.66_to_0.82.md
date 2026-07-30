# GraphRAG Cypher QA — Progress Report: 66% → 82% Combined

**System:** Knowledge Graph RAG over Neo4j (Horse ontology V9)  
**Metric:** `combined = (semantic_similarity + llm_judge_overall) / 2` over 100 questions  
**Primary code touchpoint:** `backend/graph_rag/llm_service.py` (`get_cypher_prompt()`, `get_qa_prompt()`)  
**Hard constraints throughout:** no changes to archived `backend/graph_rag/_archive/{cypher_validator,cypher_retry,validated_chain}.py` without separate approval; live Neo4j verification required for new rules; stop on plateau rather than endless per-question patches.

**Period:** 2026-07-26 (intensive prompt + ground-truth campaign)

---

## 1. Executive summary

| Milestone | Combined | What unlocked it |
|---|---|---|
| Start (pre-rebuild) | **0.660** | Patched but fragmented Cypher prompt |
| After full prompt rebuild + completeness batch | **0.723** | COUNT+COLLECT, QA exhaustiveness, superlatives, uniformity, role lookups |
| After crash repair + domain rules | **0.761** | WITH-scoping, named events, season/schema rules — **75% target met** |
| After WITH-wall generalization | **0.765** | Prompt-only **plateau** (~76%) |
| After V9 ground-truth reconciliation (Track 1) | **0.821** | Corrected 22 GTs + Q34; **prompt unchanged** — **80%+ without Track 2** |

**Net gain:** +16.1 combined points (0.660 → 0.821).  
**Split of the last jump (0.765 → 0.821):** essentially all **Track 1 (ground truth)**; prompt frozen for that run.

**Documented limitation:** WITH-wall / WHERE-after-RETURN / UNION crashes can be reduced but not eliminated by prompt alone — fixing one shape often relocates the same bug (Change 11 plateau).

---

## 2. Scoring model and variance

- Combined score averages semantic similarity and LLM-as-judge overall across 100 questions.
- Semantic similarity was relatively stable (~0.80–0.88); most prompt gains came from **judge** (completeness, correctness) and from **fewer Cypher crashes** (crash → both components 0).
- Same-prompt re-runs showed roughly **±0.01 combined** variance. Gains below that band are not attributed to a change.
- Invalid runs (concurrent evals / API 429 / quota): `133141`, `133244` — excluded from the ladder.

---

## 3. Full evaluation ladder (valid runs only)

All files under `evaluation_results/graph_rag/semantic_evaluation_20260726_*.json`.

| # | Timestamp | Success | Semantic | Judge | Combined | Context |
|---|---|---|---|---|---|---|
| 1 | `011340` | 0.930 | 0.804 | 0.516 | **0.660** | Start — pre-rebuild patched prompt |
| 2 | `013423` | 0.960 | 0.824 | 0.496 | **0.660** | Variance check (same prompt) |
| 3 | `023601` | 0.960 | 0.831 | 0.549 | **0.690** | Full Cypher prompt rebuild (5 sections) |
| 4 | `024254` | 0.960 | 0.834 | 0.560 | **0.697** | Variance |
| 5 | `025408` | 0.970 | 0.841 | 0.577 | **0.709** | Variance (peak before completeness batch) |
| 6 | `131733` | 0.930 | 0.818 | 0.628 | **0.723** | Changes 1–6 (completeness + QA + …); success dipped (COLLECT regressions) |
| 7 | `135110` | 0.960 | 0.838 | 0.619 | **0.728** | Change 7 — crash repair |
| 8 | `142820` | 0.940 | 0.834 | 0.648 | **0.741** | Changes 8–9 |
| 9 | `143958` | 0.940 | 0.838 | 0.684 | **0.761** | Change 10 — **≥75% met** |
| — | `145934` | 0.940 | 0.838 | 0.683 | **0.761** | Independent user confirm; 6 crashes |
| 10 | `152657` | 0.950 | 0.845 | 0.684 | **0.765** | Change 11 — WITH-wall; prompt plateau |
| 11 | `160643` | 0.980 | 0.875 | 0.766 | **0.821** | **Track 1 GT applied; prompt unchanged** |

---

## 4. Phase A — Prompt rebuild and early gains (0.66 → ~0.71)

### Starting point
Fragmented Cypher prompt after prior patching. Combined ~**0.66** with judge ~0.50–0.52.

### Full Cypher prompt rebuild
- Restructured `get_cypher_prompt()` into clear English sections (prohibitions, mandatory shape, schema patterns, few-shots, closing checklist).
- QA prompt remained French (answers must be French).
- Immediate lift: **0.66 → ~0.69–0.71** across rebuild + variance runs (`023601`–`025408`).

### Methodological lesson (recurring)
**Negative examples that look like valid Cypher are copied as templates.** Bad forms must be prose-only or omitted; positive YES forms must lead. This was rediscovered on role lookups, superlatives, and label/path glue.

---

## 5. Phase B — Completeness and answer quality (Changes 1–6 → 0.723)

Measured on run `131733` (**0.7228**). Success fell to 0.930 because COLLECT pairing introduced new WITH-scoping crashes.

### Change 1 — Pair every COUNT with COLLECT
- **Problem:** ~32 questions scored ~0.5: facts correct but judge penalized missing names (GT enumerates; Cypher returned only counts).
- **Fix:** every `COUNT(DISTINCT x)` accompanied by `COLLECT(DISTINCT …)` in the same clause; also compresses lists under `top_k` truncation.
- **Verified** on live Neo4j for major grouping patterns.

### Change 2 — QA exhaustiveness rules
- New exhaustiveness block: cite every number; enumerate names; report ties; state when one value covers the whole population.
- Refined: comma-separated names (not long numbered lists); sensor IDs capped to 2–3 examples (over-enumeration truncated answers).
- Anatomical glossary: Withers→garrot, CanonOfForelimb→canon antérieur, etc. (model had rendered “Withers” as nonsense).

### Change 3 — Superlative on property vs count
- **Problem:** “longest session” answered as “most sessions”.
- **Fix:** property superlatives (`Volume`) vs count superlatives use distinct COLLECT/UNWIND skeletons.
- **Verified:** Comet/Ecume 55min prep (tie), Pixie 40min, Comet 75min pre-comp, 300Hz sensor tie.

### Change 4 — Uniformity by property values
- “Is phase X the same for everyone?” groups by `Volume`/`Intensity`/`Frequency`, never by stage `id`.
- Sibling rule for “does count A always equal count B” with two variables of the same type.

### Change 5 — Simplest role lookups
- “Who is the vet?” → bare `MATCH (v:Veterinarian) RETURN v.id` (no traversal).
- Fixed Q20 empty / “not available” failure.

### Change 6 — Cypher template in English
- User request; QA stays French. No material score change; interpolation verified.

---

## 6. Phase C — Crash repair and domain rules (Changes 7–10 → 0.761)

### Change 7 — Repair COLLECT-induced crashes
- **Problem:** Success 0.970 → 0.930. Pattern: `WITH` drops `h`, then `RETURN … COLLECT(h)` / `COUNT(h)`.
- **Fixes:**
  - After any aggregating `WITH`, `RETURN` must use aliases only (no new aggregates).
  - Stronger ban on WHERE-after-RETURN.
  - Ban path arrow glued to label test in WHERE.
  - Closing block **“THE SIX MISTAKES THAT ACTUALLY HAPPEN”** at end of prompt (recency > early emphasis in a ~35k-char prompt).
- Run `135110`: **0.7284**, success back to 0.960.

### Change 8 — Named events, season listing, Frequency vs count
- Named event: `MATCH (e) WHERE e.id = "…"` — never fake `:Event` label.
- Season listing: bare `INSEASON` path (no glued label WHERE) — fixed Q25-class crash.
- Training frequency/intensity/duration = properties, not `COUNT(DISTINCT t)`.
- Actor name token without spaces (`CONTAINS "Martin"`).
- “Rank at every entered event?” returns positive per-event ranks, not empty anti-join.

### Change 9 — Schema paths + empty yes/no
- Schema path answers use **string literals** for relationship names — never `type(node)` (Type mismatch crash).
- Empty context on yes/no implication → answer « Non », not « information non disponible ».
- Run `142820`: **0.7409**.

### Change 10 — Season period, race, categories
- Season period from `CompetitiveSeason.seasonStart` / `seasonEnd` (not min/max event dates).
- Season categories via INSEASON + `e.category` counts.
- Most-common race COLLECT/UNWIND without reopening dropped `h`.
- QA: actors only in prep/precomp → NON for “all stages”, name missing phases.
- Run `143958`: **0.7611** — **75% target achieved**.

---

## 7. Phase D — WITH-wall generalization and prompt plateau (Change 11 → 0.765)

### Independent confirm
User run `145934` ≈ **0.7605**, matching the 76.1% report. Six Cypher crashes traced:

| Q | Error (surface) | True root cause |
|---|---|---|
| Q59 | `Variable e not defined` | COUNT then COLLECT of `e` across two WITHs |
| Q62 | `Variable h not defined` | Anti-join; `h` dropped before RETURN COUNT(h) |
| Q65 | `Invalid input 'WHERE'` | WHERE after aggregating RETURN |
| Q69 | UNION column mismatch | UNION with incompatible branches |
| Q93 | `Variable h not defined` | Two-level agg; COUNT(h) after drop |
| Q98 | `Variable h not defined` | Same wall as Q62 |

**Grouping:** Q59/Q62/Q93/Q98 (+ variants) = one **WITH wall**; Q65 and Q69 separate.

### Change 11
- Top-of-prompt **RED LINE** (Crash A/B/C) + copy-paste templates (no-result anti-join; training vs competitions).
- SIX MISTAKES #1 rewritten as **THE WITH WALL** with YES shapes (a)(b)(c).
- Strengthened WHERE-before-RETURN; busiest-month without nested COLLECT/UNWIND after dropping `e`.
- Run `152657`: **0.7647**. Original six no longer crash; **new crashes relocated** (Q63, Q66, Q68, Q84, Q86) — same families.
- **Verdict:** prompt-only improvement **plateaued around 76%**. Stop condition applied; no further WITH-wall patching.

Artifacts: `graph_prompt_changelog.md` (Change 11), `docs/graph_ground_truth_flags.md`, `docs/graph_v9_ground_truth_drift.md`.

---

## 8. Phase E — Track 1 ground-truth reconciliation (0.765 → 0.821)

**Rationale:** V8.7 GTs contradicted live V9 graph on ~16–22 questions. Correct V9 answers scored ~0 against stale text — no prompt could recover them.

### Process
1. Inventory in `docs/graph_v9_ground_truth_drift.md` + `docs/graph_ground_truth_flags.md`.
2. Fresh live Neo4j verification for every flagged item (not memory).
3. Sign-off document: `docs/graph_gt_reconciliation_signoff.md` (question, current GT, proposed GT, Cypher + live result).
4. Human approval → apply exact **(c)** texts to `data/test_dataset.json`.

### Applied corrections (22 CLEAN + Q34 partial)

| Category | Questions | Nature of fix |
|---|---|---|
| A — Sensor IDs | Q38, Q39, Q14, Q82 | V8 bare ids → horse-scoped V9 ids |
| B — Training inventory | Q5, Q52, Q53, Q55, Q86, Q93, Q94 | Dakota 6→8 stages; phase counts; transition 19→50; distributions |
| C — Engagements/results | Q62, Q65, Q67, Q97, Q98 | Phenomenon “unranked engagement” no longer exists in V9 |
| D — Shifted counts | Q23, Q34*, Q57, Q63, Q66, Q69 | Rider counts; supervisors; event result tie; 9→11 dual-horse cases; rel counts 355/207/207 |
| No change | Q8 | Horse-level prep durations already correct (31×45min) |
| Prior (kept) | Q91 | Bordeaux rank already corrected earlier |

Also updated `metadata.updated_date` / `audit_note` for the batch.

### Corrected-GT baseline (prompt frozen)
- Run `160643`: **combined 0.8206 ≈ 0.821**
- Success **98%**, semantic **0.875**, judge **0.766**
- Gain vs `152657` (0.765): **+5.6 pp**, entirely from GT alignment
- **80% target already exceeded** before Track 2 prompt work

### Remaining crashes in that run (2)
| Q | Error | Family |
|---|---|---|
| Q66 | `Invalid input 'WHERE'` after RETURN | WHERE-after-RETURN |
| Q84 | `Variable t not defined` after WITH | WITH wall |

These are the documented prompt-limitation families — **not** pursued further under the Track 2 stop rules (no WITH-wall patch-and-relocate).

### Residual low scorers (examples)
- Q38/Q39: question text still names bare V8 ids; model says “not available”; GT now names Dakota-scoped ids → judge still harsh (question/GT intent mismatch).
- Q67: empty anti-join → “not available” vs GT “aucune compétition…”.

---

## 9. Track 2 status

**Not started for score chasing after 0.821.**  
Plan (when resumed): re-baseline already obtained (`160643`); only investigate **new** failure territory outside WITH/WHERE/UNION; one rule + few-shot; stop at 80% (already met) or plateau.

---

## 10. Supporting documentation and artifacts

| Artifact | Role |
|---|---|
| `graph_prompt_changelog.md` | Numbered prompt changes 1–11 + measured effects |
| `docs/graph_v9_ground_truth_drift.md` | V9 vs v8.7 drift inventory |
| `docs/graph_ground_truth_flags.md` | Per-question flag table + staged Q38/Q39 notes |
| `docs/graph_gt_reconciliation_signoff.md` | Full sign-off with Cypher evidence |
| `data/test_dataset.json` | Applied GT batch (22 + Q34; Q8/Q91 as agreed) |
| `backend/graph_rag/llm_service.py` | All Cypher/QA prompt changes |
| Eval JSONs | Ladder in §3 (gitignored under `evaluation_results/*.json`) |

Temporary scratch scripts (`_b*`, `_c*`, `_t1_*`) were used for regression / Neo4j verify / apply — not part of the product.

---

## 11. Methods and guardrails used throughout

1. **Live Neo4j verification** before trusting a rule or GT proposal.  
2. **Fresh self-check questions** (not only the failing benchmark IDs).  
3. **Fixed regression sets** (e.g. Q4, Q17, Q31, Q44, Q66, Q80, Q83, Q84, Q91).  
4. **One structural rule per root cause** — not one patch per question.  
5. **Stop conditions:** ≥ target, or plateau → report plainly, stop iterating.  
6. **GT changes only with explicit human sign-off** (Track 1 process).

---

## 12. Bottom-line attribution

```
0.660  start
  +0.05  prompt rebuild (~0.71)
  +0.01  completeness/QA/superlative/uniformity/roles (~0.72)
  +0.04  crash repair + domain Cypher/QA rules (~0.76)
  +0.00  WITH-wall generalization (plateau ~0.765)
  +0.06  V9 ground-truth reconciliation, prompt frozen (~0.821)
──────
0.821  corrected-GT baseline (latest intended checkpoint)
```

**Conclusion for reporting:** Prompt engineering recovered roughly **10 points** (0.66 → 0.76) and hit a hard ceiling on stochastic Cypher syntax families. Aligning ground truth to the live V9 graph recovered another **~6 points** in a single no-regression Track 1 pass and pushed the system past **80% combined** without further prompt changes.
