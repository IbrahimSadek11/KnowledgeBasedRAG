# How we built Tabular RAG — short report

Parallel Text-to-SQL path over the same Horse ontology as Graph RAG. Not wired into the Streamlit UI; lives under `backend/tabular_rag/` and `scripts/tabular_rag/`.

---

## 1. Idea

Graph RAG asks Neo4j with Cypher. Tabular RAG asks the same facts as **SQL over SQLite**, so we can compare:

- text-to-Cypher vs text-to-SQL on the **same 100 questions** (`data/test_dataset.json`);
- same semantic + LLM-judge metrics, plus SQL **Execution Accuracy (EX)** against gold queries.

---

## 2. Data pipeline

```
Horse RDF (V9)
    → Neo4j          (scripts/setup_database.py)
    → SQLite ETL     (backend/tabular_rag/tabular_etl.py)
    → data/tabular_rag/tabular.db
```

1. Explore Neo4j labels/properties (`scripts/tabular_rag/explore_tabular_source.py`).
2. Extract via Cypher into relational tables (horses, events, seasons, trainings, actors, participations, entries, sensors, objectives, people / riders / vets / caretakers, associations).
3. Keep human-readable TEXT columns; add parsed numeric twins (`volume_minutes`, `sample_rate_hz`, …) so SQL can aggregate cleanly.
4. Verify integrity (`verify_tabular_data.py`, `full_diff_verify.py`) against Neo4j counts/rows before trusting eval.

Tabular DB is a **projection of the graph**, not a second ontology.

---

## 3. Runtime architecture

Core: `backend/tabular_rag/tabular_chain.py`

```
Question
  → live schema from tabular.db (+ sample values)
  → optional horse-name → horse_id resolution
  → LLM (gpt-4o-mini, temp=0) generates one SELECT
  → sql_validator (SELECT-only, no stacked statements)
  → optional aggregation gate (“combien” / “plus …” must use COUNT/GROUP BY)
  → execute on SQLite read-only (+ authorizer)
  → retry on error (feedback loop)
  → second LLM call → short French answer
```

Safety: validator + SQLite authorizer so writes/DDL cannot run even if validation slips.

---

## 4. Evaluation

| Layer | What |
|---|---|
| Semantic + LLM-judge | Same as Graph RAG (`evaluation_service.py`); combined = (semantic + judge) / 2 |
| Execution Accuracy (EX) | Hand-written gold SQL in `scripts/tabular_rag/gold_queries.py`; compare normalized result sets |
| N/A set | Conceptual / unanswerable / schema-explanation questions excluded from EX |
| Runner | `scripts/tabular_rag/run_tabular_evaluation.py` (full 100-Q benchmark) |

---

## 5. Build narrative (bullets)

1. Reused the existing Neo4j Horse graph rather than inventing a separate tabular source.
2. Designed a SQLite schema as a clean relational slice of that graph.
3. Built ETL + verification so table contents stay faithful to Neo4j.
4. Implemented Text-to-SQL with **live schema injection** (no stale prompt schema).
5. Added domain SQL rules (stage types, roles, associations vs ranked participations, case-insensitive filters).
6. Hardened execution (SELECT-only + read-only authorizer + retries).
7. Mirrored Graph RAG models and scoring for a fair head-to-head.
8. Added EX with gold SQL for questions that have a definite result set.
9. Kept the pipeline as a research/eval path alongside Graph RAG (not in the chat UI yet).

---

## 6. Key files

| File | Role |
|---|---|
| `backend/tabular_rag/tabular_etl.py` | Neo4j → SQLite |
| `backend/tabular_rag/tabular_chain.py` | Question → SQL → answer |
| `backend/tabular_rag/sql_validator.py` | SQL safety |
| `data/tabular_rag/tabular.db` | SQLite store |
| `scripts/tabular_rag/gold_queries.py` | Gold SQL + EX helpers |
| `scripts/tabular_rag/run_tabular_evaluation.py` | Full evaluation |
| `scripts/tabular_rag/verify_tabular_data.py` | Integrity checks |
