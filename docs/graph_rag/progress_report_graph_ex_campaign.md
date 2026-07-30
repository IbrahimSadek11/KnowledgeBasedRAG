# Progress Report — Graph EX campaign (post Tabular final / Graph reorg)

**Covers:** all work after `docs/progress_report_post_graph_082.md`  
**Period:** 2026-07-26 night → 2026-07-27 — Graph execution accuracy (EX) infrastructure, comparator, AMBIGUOUS triage, Cypher prompt rules, clean re-evals  
**Scoring:**  
- Combined = `(semantic_similarity + llm_judge_overall) / 2` over 100 questions  
- EX = Neo4j result-set match vs locked gold Cypher; N/A and AMBIGUOUS excluded from denominator  
**Models:** stock API only (`gpt-4o-mini`, `text-embedding-3-small`) — **no fine-tuning**  
**Hard constraint:** `data/test_dataset.json` NL ground truth **not** modified this period (Graph EX gold lives in `scripts/graph_rag/gold_cypher_queries.py`)

---

## 1. Executive summary

| Track | Start (this period) | End (this period) | Driver |
|---|---|---|---|
| **Graph RAG** | Combined **0.818**, **no EX** | Combined **~0.815–0.828**, **EX 60.9% (53/87)** | Gold Cypher + EX wiring + column-superset comparator + argmax/COLLECT prompt rules |
| **Tabular RAG** | Signed final **0.788 / 62.9% EX** | Reconfirmed **0.787 / 62.9% EX** | Clean re-run only (no new SQL/gold work) |
| **Fine-tuning** | — | — | **None** |

**Headline Graph movement on EX:**

| Milestone | EX | Denom | Notes |
|---|---|---|---|
| First EX baseline | **33.7%** (30/89) | 11 N/A | Strict result-set equality |
| After gold/prompt packaging pass | **61.8%** (55/89) | 11 N/A | Still exact match |
| Peak published % (6 AMBIGUOUS) | **66.3%** (55/83) | 11 N/A + 6 AMB | Comparator live; over-held AMBIGUOUS |
| Clean post–rules A/B (final) | **60.9%** (53/87) | 11 N/A + **Q92/Q95** | Argmax + COLLECT rules; 100% success |
| Same Cypher rescored on 87-denom | **64.4%** (56/87) | — | Offline rescore of `…_005224` (not a live regen) |

**Invalid run (do not cite):** `semantic_evaluation_20260727_013101.json` — success **58%**, EX **41.4%**, heavy API failures / empty Cypher (rate-limit contamination).

**Final clean Graph artifact:**  
`evaluation_results/graph_rag/semantic_evaluation_20260727_015059.json`  
(also confirmed by `…_011802.json` at same EX rate)

**Tabular reconfirm:**  
`evaluation_results/tabular_rag/tabular_eval_full100_ex_20260727_015856.json`  
(= signed `…_203437` within noise: EX still **56/89 = 62.9%**)

---

## 2. Starting point (end of prior report)

From `docs/progress_report_post_graph_082.md`:

| Pipeline | Combined | EX | Artifact |
|---|---|---|---|
| Graph post-reorg | **0.818** | — | `evaluation_results/graph_rag/semantic_evaluation_20260726_210024.json` |
| Tabular signed final | **0.788** | **62.9%** (56/89) | `evaluation_results/tabular_rag/tabular_eval_full100_ex_20260726_203437.json` |

Prior period: Tabular gold + SQL/synthesis campaign; Graph **package reorg only** (no new Cypher prompt logic). This period is the **Graph EX workstream**.

---

## 3. Graph EX evaluation ladder

| # | Run / artifact | Combined | Judge | Sem | Success | EX | N/A | Trigger |
|---|---|---|---|---|---|---|---|---|
| 0 | `…_210024` | **0.818** | 0.755 | 0.880 | 99% | — | — | Post-reorg baseline (prior report) |
| 1 | `…_235509` | 0.804 | 0.751 | 0.857 | 96% | **33.7%** (30/89) | 11 | First EX wiring; exact match |
| 2 | `…_002317` | **0.824** | 0.778 | 0.870 | 98% | **61.8%** (55/89) | 11 | Gold + packaging fixes |
| 3 | `…_005224` | **0.831** | 0.782 | 0.880 | 99% | **66.3%** (55/83) | **17** (11+6 AMB) | Column-superset comparator; 6 AMBIGUOUS |
| 4 | `…_011802` | **0.828** | 0.768 | 0.889 | **100%** | **60.9%** (53/87) | **13** (11+Q92/Q95) | Argmax + COLLECT rules live |
| — | *(invalid)* `…_013101` | 0.481 | 0.438 | 0.525 | **58%** | 41.4% (36/87) | 13 | Rate-limit / empty Cypher — **discard** |
| 5 | `…_015059` | **0.815** | 0.741 | 0.888 | **100%** | **60.9%** (53/87) | 13 | Clean reconfirm of #4 |

**Denom math (current):**  
`applicable = 100 − |EX_NOT_APPLICABLE| − |AMBIGUOUS_FOR_REVIEW| = 100 − 11 − 2 = 87`.

**Why EX % fell from 66.3% → 60.9% even as rules helped:**

1. AMBIGUOUS trimmed from **6 → 2** → denominator grew **83 → 87** (Q53/Q63/Q54/Q93 back into scored set).  
2. Live regen after prompt edits lost some prior MATCHes vs the peak Cypher snapshot (**56/87** offline rescore → **53/87** live).  
3. Net: cleaner policy + honest denom; absolute MATCH count is slightly lower than the over-held peak.

---

## 4. COMPLETE FIX INVENTORY — Graph EX

### A. EX infrastructure (scoring stack)

| ID | Fix |
|---|---|
| **I1** | Authored `GOLD_CYPHER_QUERIES` for applicable questions in `scripts/graph_rag/gold_cypher_queries.py` (live V9 Neo4j verified; locked separately from NL GT) |
| **I2** | Defined `EX_NOT_APPLICABLE` (11) — schema / unanswerable / open narrative (see §5) |
| **I3** | Wired EX into `scripts/graph_rag/run_evaluation.py`: execute gold + generated Cypher, compare, report `execution_accuracy` |
| **I4** | Unit tests `scripts/graph_rag/test_compare_cypher_execution.py` (T1–T13) written before / with comparator evolution |
| **I5** | Probe / triage helpers (non-prod): `_triage_ex.py`, `_classify_mismatches.py`, `_verify_ambiguous.py`, `_verify_rules_ab.py`, `_archive_final_ex.py`, batch verify scripts |
| **I6** | Signoff doc kept: `docs/graph_gt_reconciliation_signoff.md` (NL GT still awaiting human approve; EX gold is Cypher-side) |

**Measured after I1–I3:** first ladder step EX **33.7%**.

---

### B. Bugs found in early EX (and what they meant)

Early triage of MISMATCH buckets (`_classify_mismatches`):

| Class | Meaning | Typical symptom |
|---|---|---|
| **a** | Packaging / extra columns | Same facts; gen returns more aliases or list vs COLLECT shape |
| **b** | Shape / argmax | `ORDER BY … LIMIT 1` on distributions; wrong aggregation grain |
| **c** | Wrong claim / empty / schema | Different answer entirely; empty Cypher; bad entity |

Concrete bugs that drove prompt + comparator work:

| Bug | Symptom | Fix path |
|---|---|---|
| **B1 — Extra-column false fail** | Gen returns gold values **plus** helpful columns → exact match = MISMATCH | Column-superset comparator (§C) |
| **B2 — Argmax hides ties** | “le plus / le moins” → `ORDER BY n DESC LIMIT 1` | Crash D / Rule C: forbid LIMIT 1; full histogram or MAX-all-ties (§D) |
| **B3 — Bare COUNT histograms** | COUNT without member `COLLECT` → incomplete vs gold | Rule 3.2bis / COLLECT-alongside-COUNT (§D) |
| **B4 — Crash A2** | `WITH h, COUNT(t)` then later `COLLECT(t.*)` → variable gone / crash | Merge COLLECT into aggregating WITH; copy-shapes (§D) |
| **B5 — False AMBIGUOUS** | Q53/Q63 held as ambiguous when failures were exhaustiveness / empty Cypher | Removed from AMBIGUOUS after live verify (§E) |
| **B6 — Rate-limit contamination** | 429 / empty Cypher → EX and combined collapse | Discard `…_013101`; require 100% success for publish (§F) |
| **B7 — Regen variance** | Prompt help on some Qs regresses others | Documented; offline rescore shows **56/87** ceiling of prior Cypher |

---

### C. Comparator — column-superset tolerance

| ID | Fix |
|---|---|
| **C1** | Exact normalized equality still MATCH |
| **C2** | Else **containment / column-superset**: equal row cardinality; bipartite pairing where each gold row’s **value multiset ⊆** a distinct gen row’s value multiset |
| **C3** | Extra generated columns OK; missing gold values / wrong values / row-count drift / COLLECT multiset inflation still MISMATCH |
| **C4** | `LIST_COMPARE_OVERRIDES` hook (default multiset; per-Q `"set"` opt-in) — unused by default |
| **C5** | Tests T10–T13 cover extra cols, missing gold value, wrong value with extras, extra COUNT/COLLECT beside core |

**Effect:** packaging-only fails (class **a**) convert to MATCH without changing gold. Peak EX **55/83** after comparator + early AMBIGUOUS set.

> Note: `metadata.ex_formula` in JSON still says “exact … match”; runtime logic also applies containment. Cosmetic metadata lag — behavior is C1–C2.

---

### D. Cypher prompt rules (`backend/graph_rag/llm_service.py` → `get_cypher_prompt()`)

| ID | Rule | Intent |
|---|---|---|
| **P1** | **Crash D / Rule C — no argmax `ORDER BY … LIMIT 1`** | Distributions / “le plus…” must not hide ties |
| **P2** | Superlative → full histogram **or** MAX/MIN-all-ties (`COLLECT` + `UNWIND`) | Q53-class stage histograms |
| **P3** | Event result-count leaderboard → `ORDER BY result_count DESC, event LIMIT 10` | Q63-class; never LIMIT 1; secondary tie-break required |
| **P4** | **3.2bis — COUNT + COLLECT members in the same WITH** | Names/ids alongside counts (prep Volume, riders, sensors, TransitionStage, multi-city) |
| **P5** | **Crash A2** — never `COLLECT(t.*)` after COUNT dropped `t` | Merge COLLECT into first aggregating WITH |
| **P6** | Copy-shapes for stage-count histogram, prep Volume (+ **stages**), rider horse_count, sensor_count, TransitionStage Volume, multi-city events | Reduce inventiveness on high-frequency patterns |
| **P7** | Removed / overridden prior language that “LIMIT 1 OK for single-item” when it contradicted Crash D | Consistency |

**Verification:** `scripts/graph_rag/_verify_rules_ab.py` — live Neo4j check that Q53/Q63 MATCH and `AMBIGUOUS_FOR_REVIEW == {Q92, Q95}`.

**Measured after P1–P7 (clean):** EX **53/87 = 60.9%**, success **100%**, combined **0.828** then reconfirm **0.815**.

---

### E. AMBIGUOUS triage

| Phase | Set | Rationale |
|---|---|---|
| Early (peak %) | Q53, Q54, Q63, Q92, Q93, Q95 | Broad “dual reading” hold |
| After side-by-side verify | **Q92, Q95 only** | Q53/Q63 = exhaustiveness / empty Cypher, not ambiguity |
| Explicitly **not** forced AMBIGUOUS | Q54, Q93 | Dual shape exists but left scored for now |

**Current `AMBIGUOUS_FOR_REVIEW`:**

| ID | Why held |
|---|---|
| **Q92** | NL ≈ fewest IMU / argmin; gold = full sensor-count histogram (2/3/4) — both defensible |
| **Q95** | NL ≈ season vs level; gold = season × category × discipline vs season × category-only — both defensible |

---

### F. Operational / eval hygiene

| ID | Fix |
|---|---|
| **O1** | Treat success ≪ 100% + halved cost as contaminated (rate limits) — do not publish |
| **O2** | Prefer dual clean runs (`…_011802`, `…_015059`) before calling EX “final” |
| **O3** | Offline rescore of older Cypher on new denom when comparing apples-to-apples (**56/87**) |
| **O4** | Soft Tabular run with 429s discarded; clean Tabular reconfirm `…_015856` |

---

## 5. EX_NOT_APPLICABLE (11) — locked exclusions

| ID | Reason (short) |
|---|---|
| **Q35** | Open multi-hop narrative — no single canonical result shape |
| **Q38**, **Q39** | Legacy non-existent sensor IDs; correct behavior is “not found” |
| **Q40** | Unanswerable — no Horse age property |
| **Q68** | Unanswerable — no coat/weight/vet phone properties |
| **Q70–Q73**, **Q88**, **Q89** | Schema-explanation / conceptual — not data lookup |

These mirror Tabular’s 11 N/A spirit but are Graph-specific (Cypher result-set scoring).

---

## 6. Remaining Graph MISMATCH (clean `…_015059`)

**34 MISMATCH** on applicable 87:

`Q1, Q8, Q14, Q16, Q22, Q23, Q34, Q41, Q45, Q47, Q48, Q54, Q56, Q58, Q59, Q61, Q64, Q66, Q74, Q80–Q87, Q90, Q91, Q93, Q94, Q97–Q99`

**Stubborn packaging clusters (post-rules):**

| Bucket | IDs | Pattern |
|---|---|---|
| Prep duration packaging | **Q8** | Gold wants `volume, horses, stages, COLLECT(names)`; gen often omits `stages` |
| COLLECT-alongside-COUNT | **Q47**, **Q90** | Gold requires member lists; gen still bare COUNT |
| Dual interpretation (scored) | **Q54**, **Q93** | Different but defensible shapes — not in AMBIGUOUS |
| Actor / role vs id | **Q56** | Role packaging vs actor id |
| Held out of denom | **Q92**, **Q95** | AMBIGUOUS |

Many remaining misses are still class **b/c** (wrong grain, wrong join, or incomplete packaging) — not comparator false fails.

---

## 7. Tabular parity (this period)

No new Tabular gold or SQL rules. Clean reconfirm after Graph work:

| Artifact | Combined | Judge | Sem | EX | Success |
|---|---|---|---|---|---|
| Signed final `…_203437` | **0.788** | 0.682 | 0.893 | **62.9%** (56/89) | 100% |
| Reconfirm `…_015856` | **0.787** | 0.680 | 0.894 | **62.9%** (56/89) | 100% |

Tabular EX remains the higher of the two pipelines on published clean runs (**62.9%** vs Graph **60.9%**), with **different denominators** (89 vs 87).

---

## 8. Side-by-side finals (publishable)

| Pipeline | Combined | Judge | Semantic | EX | Success | Artifact |
|---|---|---|---|---|---|---|
| Graph (clean EX final) | **0.815** | 0.741 | 0.888 | **60.9%** (53/87) | **100%** | `graph_rag/semantic_evaluation_20260727_015059.json` |
| Graph (same EX, higher combined) | **0.828** | 0.768 | 0.889 | **60.9%** (53/87) | **100%** | `graph_rag/semantic_evaluation_20260727_011802.json` |
| Graph (peak EX %, over-held AMB) | **0.831** | 0.782 | 0.880 | **66.3%** (55/83) | 99% | `graph_rag/semantic_evaluation_20260727_005224.json` — cite with caveat |
| Tabular (signed) | **0.788** | 0.682 | 0.893 | **62.9%** (56/89) | **100%** | `tabular_rag/tabular_eval_full100_ex_20260726_203437.json` |

**Recommended cite for Graph EX:** **60.9% (53/87)** from `…_015059` (or `…_011802`), with N/A = 11 schema/unanswerable + AMBIGUOUS Q92/Q95.

---

## 9. Count of distinct adjustments this period

| Bucket | IDs | Count |
|---|---|---|
| EX infrastructure | I1–I6 | **6** |
| Bugs characterized | B1–B7 | **7** |
| Comparator | C1–C5 | **5** |
| Prompt rules | P1–P7 | **7** |
| AMBIGUOUS policy | E (trim 6→2) | **1** (policy change) |
| Eval hygiene | O1–O4 | **4** |
| **Total distinct items** | | **~30** |

(Plus full gold Cypher corpus authorship — one large deliverable under I1, not counted as N separate prompt lines.)

---

## 10. Limitations (current)

1. **Prompt ↔ EX trade:** rules that fix Q53/Q63/packaging can regress other questions on live regen (56/87 offline → 53/87 live).  
2. **AMBIGUOUS residue:** Q92/Q95 need human gold decision; Q54/Q93 may join that set later.  
3. **Stubborn packaging:** Q8 / Q47 / Q90 / Q56 still often miss gold columns after COLLECT rule.  
4. **NL GT drift:** `docs/graph_gt_reconciliation_signoff.md` corrections still **not** applied to `test_dataset.json` — judge/semantic can disagree with EX gold on drifted items.  
5. **ex_formula metadata** still says “exact match” while comparator includes containment.  
6. **No fine-tuning**; further EX leaps need tighter copy-shapes, selective gold clarification, or gates (validator archive still unwired).  
7. Rate-limited runs must never be mixed into progress charts.

---

## 11. Bottom line

Since the post-0.82 / Tabular-final report, Graph gained a full **execution-accuracy track**: locked gold Cypher, 11 N/A + 2 AMBIGUOUS, a **column-superset** comparator, and two targeted prompt rules (no argmax LIMIT 1; COUNT+COLLECT completeness), verified live on Neo4j.

**Publishable Graph state:** combined **~0.815–0.828**, EX **60.9% (53/87)**, success **100%**.  
**Tabular** remains **0.788 / 62.9% EX** on the signed (and reconfirmed) artifact.  
Peak Graph EX **66.3%** is real but used a temporary 6-question AMBIGUOUS hold; the honest post-triage figure is **60.9%** on 87 applicable questions. **No fine-tuning.**
`)