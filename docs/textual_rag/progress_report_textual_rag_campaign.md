# Progress Report — Textual RAG campaign (post Graph EX)

**Covers:** all work after `docs/graph_rag/progress_report_graph_ex_campaign.md`  
**Period:** 2026-07-27 ~02:21 → ~15:20 (UTC+2)  
**Scoring:** Combined = `(semantic_similarity + llm_judge_overall) / 2` over 100 questions  
**EX:** N/A for Textual RAG (no structured query / gold result-set)  
**Models:** stock API only (`gpt-4o-mini`, `text-embedding-3-small`) — **no fine-tuning**  
**Hard constraints this period:**
- `data/test_dataset.json` **not** modified
- `backend/evaluation_service.py` and scoring formula **not** modified
- No Graph/Tabular pipeline edits for quality gains
- Do **not** raise `n_results` toward full corpus size (70) as a “fix”

---

## 1. Executive summary

| Track | Start (this period) | End (this period) | Driver |
|---|---|---|---|
| **Textual RAG** | Did not exist | Combined **0.703**, success **100%**, EX N/A | Factual corpus → Chroma → grounded GPT-4o-mini + gated retrieval/prompt mechanisms |
| **Graph RAG** | Combined **~0.815–0.828**, EX **60.9% (53/87)** | Unchanged this period | Prior report only |
| **Tabular RAG** | Combined **0.788**, EX **62.9% (56/89)** | Unchanged this period | Prior report only |
| **Fine-tuning** | — | — | **None** |

**Textual combined ladder (full 100):**

| Milestone | Combined | Judge | Sem | Notes |
|---|---|---|---|---|
| First full eval (K=5 + user hedging) | **0.657** | 0.463 | 0.852 | Baseline pipeline |
| K=15 (same prompts) | **0.689** | 0.518 | 0.861 | +0.032 overall; **data_quality 0.66→0.58** |
| After Proposals A/B/C + aggregation + entity-type | **0.700** | 0.533 | 0.866 | data_quality recovered to **0.71** |
| + Value-anchored retrieval (current) | **0.703** | 0.540 | 0.866 | Latest publishable |

**Final Textual artifact:**  
`evaluation_results/textual_rag/semantic_evaluation_20260727_152010.json`  
(console tee: `evaluation_results/textual_rag/_eval_run_console.txt`)

**Still worst category:** **comparison** (~0.56 combined) — synthesis + coverage still hard under top-K.

---

## 2. Starting point (end of prior report)

From `docs/graph_rag/progress_report_graph_ex_campaign.md`:

| Pipeline | Combined | EX | Artifact |
|---|---|---|---|
| Graph (clean EX final) | **0.815** | **60.9% (53/87)** | `…_015059` |
| Tabular (signed) | **0.788** | **62.9% (56/89)** | `…_203437` |
| Textual RAG | — | — | **not started** |

This period builds the **third pipeline** on the **same** `data/test_dataset.json` benchmark and the **same** semantic + LLM-judge scoring.

---

## 3. Chronological workstream

### Phase 0 — Factual corpus (02:21+)

**Goal:** 50 horse + 20 event French prose docs (70 total), facts only from live Neo4j V9 — no LLM invention.

| Deliverable | Path |
|---|---|
| Generator | `scripts/textual_rag/generate_factual_corpus.py` |
| Fact-sheets | `data/textual_rag/textual_corpus/fact_sheets/{horses,events}/*.json` |
| Prose docs | `data/textual_rag/textual_corpus/documents/{horses,events}/*.txt` |
| Manifest / spot-check JSON | `manifest.json`, `verification_spotcheck.json` |
| Report | `data/textual_rag/textual_corpus/PHASE0_REPORT.md` |

**Method:**
1. Read-only Neo4j pull → structured fact-sheets per horse/event.
2. Deterministic French templates interpolate sheet fields only (no generative LLM).
3. Spot-check 10 docs vs sheets: **10/10 PASS** (0 invented claims).

**Corpus length (Python `checklen.py`, not PowerShell):**
| Doc | Chars | Words |
|---|---|---|
| Arrow.txt | 2332 | 277 |
| Dakota.txt (richest) | 3763 | 468 |
| All 70 | min **722** / max **3763** / avg **2116.9** |

---

### Encoding “bug” investigation (false positive)

**Symptom:** Accented French looked garbled in some console views.

**Checks:**
1. `Get-Content -Encoding UTF8` on Arrow.txt  
2. Python `open(..., encoding='utf-8')` + `repr()` of first 200 chars  
3. Byte-level verification

**Verdict:** Files are valid UTF-8. Garble was **console/codepage display** (esp. cmd), **not** a generator write bug. **No corpus regeneration required.**

---

### Indexing (Chroma)

| Step | Detail |
|---|---|
| Dependency | `pip install chromadb` (venv) |
| Script | `scripts/textual_rag/index_corpus.py` |
| Store | `data/textual_rag/chroma_db/` |
| Collection | `equestrian_textual_corpus` |
| Embedding | `text-embedding-3-small`, **whole-doc** (no chunking) |
| Metadata | `entity_type` ∈ {horse, event} from parent folder; `filename` |
| Result | **70/70** indexed; `collection.count() == 70` |

---

### Retrieval smoke test

**Script:** `scripts/textual_rag/test_retrieval.py` (top-3)

| Query intent | Expected signal (observed) |
|---|---|
| Race of Dakota | Dakota horse doc |
| Events at Saumur | Event_SJ_01 (Saumur) |
| IMU fatigue sensors | Horse docs with FatigueDetection |

---

### Textual RAG service (baseline)

**File:** `backend/textual_rag/textual_rag_service.py`

Initial pipeline:
1. Embed question → Chroma top-K  
2. Build French grounded prompt (system: answer only from context; refuse if missing)  
3. GPT-4o-mini `temperature=0`  
4. Return `{answer, retrieved_docs, question}`

Initial default **`n_results=5`**.

---

### Prompt bug / fix — top-K hedging placement

**Problem:** Model treated top-K matches as complete corpus answers (overconfident lists).

**Fix 1:** Append “échantillon top-K / pas exhaustif” hedging instruction to **system** prompt.

**Observed:** Placement on system did not stick well for GPT-4o-mini.

**Fix 2 (A/B placement):** Move **identical wording** to end of **user** prompt (after docs + question). System prompt restored to original rules only.

**Lesson reused later:** All gated synthesis blocks (comparison, aggregation) also append to **user** prompt.

---

### Evaluation harness

**Script:** `scripts/textual_rag/run_textual_evaluation.py`

- Loads same `data/test_dataset.json` (100 Qs)
- Calls `answer_question`
- Reuses `backend.evaluation_service` (`init_evaluator`, `calculate_semantic_similarity`, `llm_judge_answer`)
- Combined = `(semantic + judge) / 2`
- **No EX** (explicitly documented in metadata)
- Writes `evaluation_results/textual_rag/semantic_evaluation_YYYYMMDD_HHMMSS.json`

---

## 4. Evaluation ladder (full detail)

| # | Artifact | Combined | Judge | Sem | Success | Cost | Time | Trigger |
|---|---|---|---|---|---|---|---|---|
| 1 | `…_123543` | **0.657** | 0.463 | 0.852 | 100% | $0.366 | 236s | K=5 + user hedging |
| 2 | `…_125054` | **0.689** | 0.518 | 0.861 | 100% | $0.396 | 363s | Default K **5→15** only |
| 3 | `…_133548` | **0.700** | 0.533 | 0.866 | 100% | $0.445 | 419s | A + B + C + agg + entity-type |
| 4 | `…_152010` | **0.703** | 0.540 | 0.866 | 100% | $0.441 | 445s | + value-anchored retrieval |

### Category movement (combined)

| Category | #1 K=5 | #2 K=15 | #3 mechanisms | #4 +value-anchor |
|---|---|---|---|---|
| unanswerable | 0.93 | 0.93 | 0.93 | **0.93** |
| comparison | **0.53** | **0.55** | **0.52** | **0.56** |
| aggregation | 0.55 | 0.62 | 0.66 | **0.65** |
| data_quality | **0.66** | **0.58** ↓ | **0.71** ↑ | **0.71** |
| consistency_check | 0.59 | 0.61 | 0.60 | **0.57** |
| multi_hop | 0.60 | 0.67 | 0.61 | **0.62** |
| simple_retrieval | 0.78 | 0.78 | 0.79 | **0.81** |
| attribute_retrieval | 0.74 | 0.83 | 0.81 | **0.81** |
| single_hop | 0.72 | 0.83 | 0.81 | **0.83** |
| schema_validation | 0.78 | 0.78 | 0.82 | **0.78** |

**Key diagnostic from #1→#2:** Widening K helped overall (+0.032) and aggregation, but **hurt data_quality** (−0.08). Comparison barely moved → synthesis problem, not coverage alone.

**No isolated A-only full eval** — Proposal A was merged into the #3 batch before a separate 100-Q run.

---

## 5. Diagnosed weaknesses (from evals)

1. **Comparison (worst):** K=5→15 barely moved (~0.53–0.55) → failure in how comparisons are synthesized from retrieved docs.
2. **Aggregation (second-worst):** High-cardinality “how many / which X have Y” cannot be completed by top-K similarity alone (architectural).
3. **data_quality regression at K=15:** Wider context adds noise/contradiction for verification-style questions.
4. **Buried low-salience facts:** Fact exists in one doc but embedding doesn’t surface that doc when the attribute isn’t the doc’s main topic.
5. **Horse↔event type mismatch (found in investigation):** Training/sensor Qs often retrieved mostly `Event_*`; corpus-wide event-count Qs retrieved mostly horses → false “not in documents.”

---

## 6. COMPLETE METHOD / FIX INVENTORY — Textual RAG

### M0 — Corpus generation (Phase 0)
- Deterministic Neo4j → JSON → French template prose
- Spot-check gate before accepting corpus

### M1 — Vector index
- Whole-document `text-embedding-3-small` into Chroma
- `entity_type` metadata for later preferential retrieval

### M2 — Baseline grounded QA
- Retrieve → prompt → GPT-4o-mini; refuse if absent

### F1 — Top-K hedging (prompt)
- Instruct model that context is a sample, not full corpus
- **Bug:** system placement ineffective → **moved to user prompt**

### F2 — Default K 5 → 15
- Pure retrieval-width change
- Net +0.032 combined; **regressed data_quality**

### Proposal A — Comparison synthesis protocol (prompt, gated)
- Gate: `_is_comparative_question` (comparer / différence / versus / plus…que / moins…que / …)
- Appends structured compare protocol to **user** prompt only when gate fires
- Non-comparative prompts unchanged

### Proposal B — Named-entity force-include (retrieval)
- Detect corpus doc ids named in question (word-boundary, longer ids first, max 5)
- `collection.get` those docs first; fill rest with similarity
- Targets buried facts when entity is explicitly named

### Proposal C — Adaptive K for verification (retrieval)
- Gate: `_is_verification_question` (calibr / cohérent / sans résultat / schéma / plusieurs cavaliers / …)
- `effective_k = 5` when gate fires; else default 15
- Targets measured data_quality regression from wide context

### M3 — Entity-type preference (retrieval)
- Infer preferred type horse vs event from cues (+ strong overrides)
- Preferential Chroma `where={"entity_type": …}` query
- Merge **forced → preferred → general**, dedupe, cap at `effective_k`
- Not an exclusive filter; not K→70

### M4 — Aggregation synthesis protocol (prompt, gated)
- Gate: `_is_aggregation_question` (combien / répartition / plus courant / …)
- Instruct: scan all provided docs; aggregate stated facts only; admit partial top-K lists
- **Partial** mitigation only — cannot invent missing entities

### M5 — Value-anchored retrieval (retrieval, additive)
- Taxonomy literals from real corpus text (categories, disciplines, intensities, sensor positions, stage types)
- If question contains a literal, re-rank merge so candidate docs **containing** that literal rise (forced ids stay first)
- No-op if no match; **no new prompt text**
- Does not increase K

### Gate stacking (confirmed in code)
| Mechanism | Effect | Stacks? |
|---|---|---|
| Top-K hedging | Always on user prompt | Always |
| Comparison synthesis | Extra user block | Independent of aggregation |
| Aggregation synthesis | Extra user block | Independent of comparison |
| Verification | Sets `effective_k=5` only | No prompt block |
| Named-entity | Retrieval merge first | With type + value-anchor |
| Entity-type preference | Preferential pool | With forced + general |
| Value-anchor | Re-rank within pool | After classic merge order |

**Q79 probe** (“Club Elite… plusieurs disciplines…”): comparison **False**; entity-type **event**; hedging only; preferred events filled all 15 slots.

---

## 7. Bugs / false alarms / constraints

| ID | Item | Outcome |
|---|---|---|
| B1 | UTF-8 mojibake in corpus | **False alarm** — console display only |
| B2 | Hedging ignored on system prompt | **Fixed** by user-prompt placement |
| B3 | data_quality drop after K=15 | **Mitigated** by adaptive K (Proposal C) + better type retrieval |
| B4 | Horse/event retrieval mismatch | **Mitigated** by entity-type preference |
| C1 | Raising K→70 forbidden | Honored throughout |
| C2 | No GT / scoring edits | Honored throughout |
| C3 | No EX fabrication | Harness metadata sets `ex_metric: null` |

**Invalid / discarded runs:** none for Textual (all four full runs succeeded 100%; no rate-limit collapse this period).

---

## 8. Current pipeline shape (final)

```
question
  → embed (text-embedding-3-small)
  → effective_k = 5 if verification else 15
  → forced docs (named entities)
  → preferred-type similarity (optional)
  → general similarity
  → merge forced → preferred → general
  → optional value-anchor re-rank
  → cap at effective_k
  → user prompt = docs + question + hedging
       + [comparison block?] + [aggregation block?]
  → GPT-4o-mini (system grounded rules, T=0)
  → answer
```

**EX:** not applicable.

---

## 9. Side-by-side publishable (all three pipelines)

| Pipeline | Combined | Judge | Semantic | EX | Success | Artifact |
|---|---|---|---|---|---|---|
| Graph | **0.815** | 0.741 | 0.888 | **60.9% (53/87)** | 100% | `graph_rag/…_015059` |
| Tabular | **0.788** | 0.682 | 0.893 | **62.9% (56/89)** | 100% | `tabular_rag/…_203437` |
| **Textual (new)** | **0.703** | 0.540 | 0.866 | **N/A** | **100%** | `textual_rag/…_152010` |

Textual trails Graph/Tabular on combined/judge — expected for free-text over a 70-doc top-K corpus answering a graph-native 100-Q set — but is a complete third track with shared scoring.

**Latest Textual by category (combined, run #4):**

| Category | n | Combined |
|---|---|---|
| unanswerable | 2 | 0.93 |
| single_hop | 9 | 0.83 |
| hierarchical | 4 | 0.83 |
| simple_retrieval | 8 | 0.81 |
| attribute_retrieval | 10 | 0.81 |
| schema_validation | 8 | 0.78 |
| data_quality | 12 | 0.71 |
| multi_hop_complex | 1 | 0.70 |
| aggregation | 16 | 0.65 |
| multi_hop | 9 | 0.62 |
| consistency_check | 5 | 0.57 |
| **comparison** | **16** | **0.56** |

**Worst answers (run #4):** Q66 (0.29), Q96 (0.35), Q67 (0.38) — often false “not in documents” or contradiction vs GT on cross-event / multi-horse patterns.

---

## 10. Count of distinct adjustments this period

| Bucket | Items | Count |
|---|---|---|
| Corpus / Phase 0 | generator, sheets, prose, spot-check, report | **1** major deliverable |
| Infra | chromadb install, index, retrieval smoke, service, eval harness | **5** |
| Bugs / hygiene | encoding false alarm; hedging placement | **2** |
| Simple config | K 5→15 | **1** |
| Prompt gates | hedging, Proposal A, aggregation synthesis | **3** |
| Retrieval mechanisms | B force-include, C adaptive K, entity-type, value-anchor | **4** |
| Full evals | 4 clean 100-Q runs | **4** |
| **Total distinct work items** | | **~20** |

---

## 11. Limitations (current)

1. **Top-K architecture:** Herd-wide aggregation / complete inventories remain incomplete by design; hedging + aggregation protocol only reduce overclaiming.
2. **Comparison still worst** (~0.56): protocol + value-anchor helped only modestly; many comparison Qs need multi-doc attribute alignment the judge still rejects.
3. **Gate brittleness:** Keyword gates can miss (Q79 “plusieurs…ou…propre à” ≠ comparison) or double-fire (comparison + aggregation both append).
4. **Verification K=5** can under-retrieve when a DQ question actually needs broad event coverage.
5. **No EX** for Textual — not comparable to Graph/Tabular EX columns.
6. **Benchmark mismatch:** Questions were authored for graph/tabular reasoning; textual answers are reconstructed from per-entity prose sheets.
7. **No fine-tuning**; further gains need better retrieval composition, denser corpus design, or selective multi-query — not K→70.

---

## 12. Bottom line

Since the Graph EX campaign report, the project gained a full **Textual RAG** track: Neo4j-grounded factual corpus (70 docs, 10/10 spot-check), Chroma index, grounded GPT-4o-mini QA, shared semantic+judge harness, and an iterative retrieval/prompt stack (hedging placement fix, K=15, comparison/aggregation synthesis, named-entity force-include, adaptive verification K, entity-type preference, value-anchored re-rank).

**Publishable Textual state:** combined **0.703**, judge **0.540**, semantic **0.866**, success **100%**, EX **N/A** — artifact `semantic_evaluation_20260727_152010.json`.  

Lift from first full run: **0.657 → 0.703** (+0.046). Largest category recovery: **data_quality 0.58 → 0.71** after adaptive K + type-biased retrieval. **Comparison remains the open gap.** Graph and Tabular were not reworked this period. **No fine-tuning.**
