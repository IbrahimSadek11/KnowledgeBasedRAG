# KnowledgeBasedRAG: Multi-Pipeline RAG for Olympic Equestrian Sports

A natural language system for querying Olympic equestrian sports data across three independent RAG pipelines, with a Fusion layer for comparative evaluation. Built in partnership with the Institut Français du Cheval et de l'Équitation (IFCE).

![Equestrian Knowledge Graph](https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?w=1200&h=400&fit=crop&q=80)

## Overview

KnowledgeBasedRAG answers French-language questions about horses, riders, training stages, inertial sensors, and competitions. It is a **multi-pipeline** system: Graph RAG, Tabular RAG, and Textual RAG each retrieve from a different store. A Fusion layer can run all three, compare their answers, and select one for evaluation.

**Only Graph RAG is integrated into RPHD.** Tabular RAG, Textual RAG, and Fusion remain standalone components in this repository. They are not called by RPHD.

**Key Features:**
- Natural language querying in French
- Three independent RAG pipelines (graph, tabular, textual)
- Fusion evaluation: pairwise agreement plus groundedness, completeness, and relevance
- Graph RAG HTTP API for RPHD (PDF candidate extraction, approved Neo4j writes, scoped chat)
- Knowledge graph with 50 horses, 25 riders, 20 events, and 108 inertial sensors
- Streamlit analytics dashboard and equestrian news page
- Specialization-30 and full-100 evaluation harnesses, including RAGAS

**Technology Stack:**
- Pipelines: Neo4j + Cypher (Graph), SQLite + text-to-SQL (Tabular), Chroma + embeddings (Textual)
- LLM: OpenAI GPT-4o-mini
- Local UI: Streamlit
- RPHD service: FastAPI and Uvicorn
- Data: RDF/OWL ontology (`data/Horse_V9_augmented.rdf`) loaded into Neo4j

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Neo4j Database (Community or Enterprise Edition)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/IbrahimSadek11/KnowledgeBasedRAG.git
cd KnowledgeBasedRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:

Copy `.env.example` to `.env` in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=neo4j
```

RPHD does not read this `.env`. It reaches Graph RAG through `EQUESTRIAN_RAG_URL` (RPHD default: `http://localhost:8500`).

4. Initialize the graph database:
```bash
python scripts/setup_database.py
```

This script parses `data/Horse_V9_augmented.rdf`, creates Neo4j nodes and relationships, and sets up indexes.

### Standalone Streamlit interface (port 8501)

```bash
cd frontend
streamlit run app.py
```

The local chatbot opens at `http://localhost:8501`.

### FastAPI service for RPHD (port 8500)

From the repository root:

```bash
python -m uvicorn api.api_server:app --host 0.0.0.0 --port 8500
```

| Method | Path | Role |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/graph` | Export the current Neo4j graph for visualization |
| `POST` | `/query` | Graph RAG question; optional `stable_node_ids` / `relationship_ids` membership scope |
| `POST` | `/pdf-receive` | Extract a candidate graph from a PDF (no Neo4j write) |
| `POST` | `/dynamic-ingestion/approve` | Human-approved write of a reviewed candidate graph |

## Usage Examples

### Basic Questions
```
Question: "Quels sont les chevaux dans le système ?"
Answer: "Les chevaux dans le système sont Dakota et Naya."

Question: "Quelle est la race de Dakota ?"
Answer: "Dakota est un cheval de race Selle Français."
```

### Relationship Queries
```
Question: "Qui est le cavalier de Naya ?"
Answer: "Le cavalier associé à Naya est Leo."
```

### Complex Queries
```
Question: "Quels capteurs sont attachés à Dakota ?"
Answer: "Dakota a 4 capteurs IMU attachés: au garrot, au sternum, 
         au canon antérieur et au canon postérieur."
```

These examples illustrate question style. The V9 graph contains 50 horses, not only Dakota and Naya.

## Project Structure

```
KnowledgeBasedRAG/
│
├── api/                           # FastAPI service for RPHD (port 8500)
│   └── api_server.py
│
├── backend/                       # Core logic and services
│   ├── config.py                  # Environment and cost constants
│   ├── evaluation_service.py      # Shared semantic + LLM-as-judge scoring
│   ├── news_service.py            # News summary
│   ├── graph_rag/                 # Graph RAG (Neo4j text-to-Cypher)
│   ├── tabular_rag/               # Tabular RAG (SQLite text-to-SQL)
│   ├── textual_rag/               # Textual RAG (Chroma + embeddings)
│   └── fusion/                    # Fusion: adapters, agreement, evidence, selector
│
├── dynamic_kg/                    # PDF candidate extraction for Graph RAG
│
├── frontend/                      # Streamlit UI (port 8501)
│   ├── app.py
│   └── pages/
│       ├── 1_Analytics.py
│       └── 2_News.py
│
├── data/
│   ├── Horse_V9_augmented.rdf     # Final RDF ontology loaded by setup_database.py
│   ├── specialization_test_30.json
│   ├── test_dataset.json          # Full 100-question benchmark
│   ├── tabular_rag/               # SQLite databases for Tabular RAG
│   └── textual_rag/               # Text corpus (Chroma store is local/regenerated)
│
├── scripts/
│   ├── setup_database.py
│   ├── graph_rag/                 # Graph evaluation
│   ├── tabular_rag/               # Tabular evaluation (v2 is the current pipeline)
│   ├── textual_rag/               # Textual evaluation and corpus indexing
│   ├── fusion/                    # Fusion evaluation CLI
│   └── ragas/                     # RAGAS CLI on specialization-30
│
├── evaluation_results/
│   ├── graph_rag/
│   ├── tabular_rag/
│   ├── textual_rag/
│   ├── fusion/
│   └── ragas/
│
├── docs/
│   └── IMPLEMENTATION.md
│
├── requirements.txt
├── .env.example
└── README.md
```

## System Architecture

### Three independent pipelines

**Graph RAG** translates a French question to Cypher, executes it on Neo4j, and writes a grounded natural-language answer.

```
User Question (French)
    ↓
Cypher generation (GPT-4o-mini)
    ↓
Neo4j execution
    ↓
Retrieved rows
    ↓
QA generation (GPT-4o-mini)
    ↓
Natural Language Answer (French)
```

**Tabular RAG** translates the question to SQL over SQLite (`data/tabular_rag/`) and synthesizes an answer from the result set.

**Textual RAG** embeds the question, retrieves passages from a Chroma collection built from `data/textual_rag/textual_corpus/`, and synthesizes an answer from those passages.

### Fusion layer

Fusion is an evaluation orchestrator, not the RPHD production path:

1. Runs Graph, Tabular v2, and Textual on the same question.
2. Scores each answer against its own retrieved evidence (groundedness, completeness, relevance).
3. Judges pairwise agreement among the three answers.
4. Selects one answer using evidence scores, with ties broken by groundedness, then completeness, then relevance.

### RPHD integration

RPHD talks only to the FastAPI Graph RAG service (`EQUESTRIAN_RAG_URL`). PDF upload extracts a candidate graph; a human must approve before Neo4j is written. Chat can optionally constrain Cypher to a selected subgraph.

## Knowledge Graph Schema

Counts from `data/Horse_V9_augmented.rdf`:

**Nodes:**
- Horse (50)
- Rider (25)
- Sporting events (20: 7 Cross, 7 Show Jumping, 6 Dressage)
- InertialSensors (108), with body positions such as withers, sternum, and cannons
- Training stages (Preparation, Pre-Competition, Competition, Transition)

**Relationships:**
- `(Rider)-[:ASSOCIATEDWITH]->(Horse)`
- `(Horse)-[:TRAINSIN]->(TrainingStage)`
- `(TrainingStage)-[:DEPENDSON]->(Event)`
- `(InertialSensors)-[:ISATTACHEDTO]->(Horse)`
- `(Event)-[:HASPARTICIPATION]->(EventParticipation)-[:HASHORSE]->(Horse)`

## Backend Components

### config.py
Loads `.env`, Neo4j connection settings, the OpenAI API key, and token-cost constants.

### graph_rag/
Cypher generation, Neo4j access, retry/identity helpers, visualization export, and dynamic ingestion writes used by the RPHD API.

### tabular_rag/
SQLite ETL and text-to-SQL. Version 2 is the current Tabular pipeline (`backend/tabular_rag/version2/`).

### textual_rag/
Chroma retrieval and grounded answer synthesis from the textual corpus.

### fusion/
Live adapters, evidence judge, pairwise agreement judge, deterministic selector, and per-question orchestrator.

### evaluation_service.py
Semantic similarity and LLM-as-judge scoring shared by Graph, Tabular, Textual, and Fusion evaluation runners.

### news_service.py
RSS aggregation, page fetching, and LLM summarization for the Streamlit News page.

### dynamic_kg/
PDF text extraction to a candidate graph. Used by `POST /pdf-receive`. No Neo4j write until `POST /dynamic-ingestion/approve`.

## Frontend Components

### app.py (MAIN INTERFACE)
Streamlit chat interface, conversation management, and response display.

### pages/1_Analytics.py
Graph statistics and Plotly visualizations (horses, events, sensors).

### pages/2_News.py
Equestrian news aggregation with generated summaries.

## Data Layer

### Horse_V9_augmented.rdf
Final RDF/OWL ontology (4,284 triples) loaded by `scripts/setup_database.py`.

### specialization_test_30.json
30-question specialization benchmark used for the reported internship evaluation.

### test_dataset.json
Full 100-question benchmark (`--100` on the evaluation CLIs).

### tabular_rag/
SQLite databases consumed by Tabular RAG v1/v2.

### textual_rag/textual_corpus/
Fact sheets and documents for Textual RAG. The Chroma directory is regenerated locally with `python scripts/textual_rag/index_corpus.py` and is not required in git.

## Evaluation Results

Figures below are from the selected **specialization-30** final JSON files (30 questions). Combined score is `(semantic_similarity + llm_judge_overall) / 2`. RAGAS metrics are 0–1 means over valid scores only.

**Standalone pipelines (20 August 2026):**

| Pipeline | Combined | Semantic | LLM-as-judge | Technical success | Source |
|---|---|---|---|---|---|
| Tabular RAG v2 | 0.829 | 0.870 | 0.789 | 29/30 (96.7%) | `evaluation_results/tabular_rag/version2/tabular_eval_specialization30_20260820_232347.json` |
| Textual RAG | 0.790 | 0.874 | 0.707 | 30/30 (100%) | `evaluation_results/textual_rag/semantic_evaluation_specialization30_20260820_232741.json` |

**Fusion selection (20 August 2026),** `fusion_eval_specialization30_20260820_232829_summary.json`:

- 30/30 questions completed
- Selected-answer combined score: 0.797 (semantic 0.877, LLM-as-judge 0.717)
- Technical success of the three live pipelines: Graph 96.7%, Tabular v2 96.7%, Textual 100%
- Selection counts: Graph 18, Tabular v2 10, Textual 2
- Pairwise agreement rate: 0.460

**RAGAS (17 August 2026),** same 30-question file:

| Metric | Graph RAG | Fusion | Notes |
|---|---|---|---|
| Faithfulness | 0.788 (n=23) | 0.753 (n=28) | |
| Answer relevancy | 0.712 (n=29) | 0.794 (n=30) | |
| Context precision | 0.423 (n=25) | 0.422 (n=29) | |
| Context recall | 0.710 (n=25) | 0.848 (n=29) | |
| Inference | 29/30 | 30/30 | Graph: `graph_ragas_specialization30_20260817_170421.json`; Fusion: `fusion_ragas_specialization30_20260817_171817.json` |

## Running Evaluations

Specialization-30 (the reported set):

```bash
python scripts/graph_rag/run_evaluation.py --30
python scripts/tabular_rag/version2/run_tabular_evaluation.py --30
python scripts/textual_rag/run_textual_evaluation.py --30
python scripts/fusion/run_fusion_layer.py --30
```

Full 100-question benchmark (default if `--30` is omitted):

```bash
python scripts/graph_rag/run_evaluation.py --100
python scripts/tabular_rag/version2/run_tabular_evaluation.py --100
python scripts/textual_rag/run_textual_evaluation.py --100
python scripts/fusion/run_fusion_layer.py --100
```

RAGAS on `data/specialization_test_30.json` (needs `OPENAI_API_KEY` and Neo4j for Graph/Fusion inference):

```bash
python scripts/ragas/run_ragas_evaluation.py --graph
python scripts/ragas/run_ragas_evaluation.py --fusion
python scripts/ragas/run_ragas_evaluation.py --all
```

JSON is written under `evaluation_results/` (RAGAS under `evaluation_results/ragas/`).

## Features

### Local chatbot
- French questions over Graph, Tabular, and Textual stores
- Grounded answers from retrieved rows or passages
- Streamlit Analytics and News pages

### RPHD Graph RAG API
- Health and graph export
- Scoped Graph RAG query
- PDF candidate extraction and approved ingestion

## Development

### Adding evaluation questions

1. Edit `data/specialization_test_30.json` or `data/test_dataset.json`
2. Run the matching evaluation CLI above
3. Adjust pipeline prompts under `backend/graph_rag/`, `backend/tabular_rag/`, or `backend/textual_rag/` if needed

### Extending the ontology

1. Update `data/Horse_V9_augmented.rdf`
2. Re-run `python scripts/setup_database.py`
3. Refresh Graph RAG schema guidance in `backend/graph_rag/`

## Configuration

### Neo4j Setup

**Local Installation**
```bash
# Download from https://neo4j.com/download/
# Start Neo4j
# Access browser at http://localhost:7474
```

### OpenAI API

Get an API key from https://platform.openai.com/api-keys

## Documentation

For component-level notes, see [IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Contributing

This is an academic research project. For questions or collaboration:
- Open an issue on GitHub
- Contact: amira.boudaoud@efrei.net
- or : Ghofrane.ben-rhaiem@efrei.net

## License

This project is part of academic research conducted at efrei research Lab in partnership with IFCE.

## Acknowledgments

- Institut Français du Cheval et de l'Équitation (IFCE) for domain expertise and data
- Our supervisor Noama Adra for guidance
- LangChain and Neo4j communities for excellent tools

**Status:** Internship evaluation complete (specialization-30 reported above)
