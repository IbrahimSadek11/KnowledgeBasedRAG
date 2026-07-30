# SportLLM: GraphRAG-Based Equestrian Knowledge System

A natural language interface for querying equestrian sports data using Graph Retrieval-Augmented Generation (GraphRAG). Built in partnership with the Institut Français du Cheval et de l'Équitation (IFCE).

![Equestrian Knowledge Graph](https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?w=1200&h=400&fit=crop&q=80)

## Overview

SportLLM is an intelligent chatbot that enables natural French language queries over a structured equestrian knowledge graph. The system combines Neo4j graph database with OpenAI's GPT-4o-mini to translate questions into precise Cypher queries, retrieving accurate information about horses, riders, training sessions, sensors, and competitions.

**Key Features:**
- Natural language querying in French
- GraphRAG architecture for accurate, grounded responses
- Knowledge graph with 2 horses, 3 disciplines, 4 sensors, and comprehensive training data
- 91.8% overall accuracy, 88% success rate on 40 test questions
- Performance analytics dashboard
- News summary from equestrian sources

**Technology Stack:**
- Backend: Python, LangChain, Neo4j
- Frontend: Streamlit
- LLM: OpenAI GPT-4o-mini
- Data: RDF/OWL ontology converted to graph database

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Neo4j Database (Community or Enterprise Edition)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/sportllm-equestrian.git
cd sportllm-equestrian
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:

Create a `.env` file in the project root:
```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
```

4. Initialize the database:
```bash
python scripts/setup_database.py
```

This script will:
- Parse the RDF ontology file (`data/Horse_generatedDataV2.rdf`)
- Create nodes and relationships in Neo4j
- Set up proper indexes

5. Launch the application:
```bash
cd frontend
streamlit run app.py
```

The chatbot interface will open in your browser at `http://localhost:8501`

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

## Project Structure

```
sportllm-equestrian/
│
├── backend/                    # Core logic and services
│   ├── config.py              # Configuration management
│   ├── evaluation_service.py  # Shared quality assessment
│   ├── news_service.py        # News summary
│   ├── graph_rag/             # Graph RAG (Neo4j text-to-Cypher)
│   │   ├── llm_service.py     # GraphRAG pipeline (CORE)
│   │   ├── graph_service.py   # Neo4j operations
│   │   └── _archive/          # Unwired validator trio (bytecode restore)
│   └── tabular_rag/           # Tabular RAG (SQLite text-to-SQL)
│
├── frontend/                   # User interface
│   ├── app.py                 # Main chatbot interface
│   └── pages/
│       ├── 1_Analytics.py     # Statistics dashboard
│       └── 2_News.py          # News feed
│
├── data/                       # Data files
│   ├── test_dataset.json      # Shared evaluation questions
│   └── conversations/         # Chat histories
│
├── scripts/
│   ├── setup_database.py
│   ├── graph_rag/             # Graph eval / retrieval / RAGAS runners
│   └── tabular_rag/           # Tabular eval / ETL helpers
│
├── evaluation_results/
│   ├── graph_rag/
│   └── tabular_rag/
│
├── docs/                       # Documentation
│   └── IMPLEMENTATION.md
│
├── graph_prompt_changelog.md
├── requirements.txt
├── .env.example
└── README.md
```

## System Architecture

### GraphRAG Pipeline

```
User Question (French)
    ↓
Cypher Generation Prompt + GPT-4o-mini
    ↓
Cypher Query
    ↓
Neo4j Execution
    ↓
Retrieved Data
    ↓
QA Prompt + GPT-4o-mini
    ↓
Natural Language Answer (French)
```

### Knowledge Graph Schema

**Nodes:**
- Horse (Dakota, Naya)
- Rider (Emma, Leo, Manon)
- Training Stages (Preparation, Pre-Competition, Competition, Transition)
- Events (ShowJumping, Dressage, Cross)
- Sensors (InertialSensors with body positions)
- Experimental Objectives (Gait Classification, Fatigue Detection)

**Relationships:**
- `(Rider)-[:ASSOCIATEDWITH]->(Horse)`
- `(Horse)-[:TRAINSIN]->(TrainingStage)`
- `(TrainingStage)-[:DEPENDSON]->(Event)`
- `(InertialSensors)-[:ISATTACHEDTO]->(Horse)`
- `(Event)-[:HASPARTICIPATION]->(EventParticipation)-[:HASHORSE]->(Horse)`

## Backend Components

### config.py
Environment variable management, Neo4j credentials, OpenAI API key, cost calculation constants

### graph_service.py
Neo4j connection initialization, graph schema refresh, Cypher query execution

### llm_service.py (CORE COMPONENT)
- Cypher generation prompt (300+ lines of detailed rules)
- QA response prompt (formatting rules)
- GraphCypherQAChain initialization
- LLM configuration (GPT-4o-mini, temp=0)
- Greeting handling and system description
- Error management for SUM() operations on strings

### evaluation_service.py
Semantic similarity calculation, LLM-as-judge evaluation, embedding-based comparison

### news_service.py
RSS feed aggregation, web scraping, LLM-based summarization

## Frontend Components

### app.py (MAIN INTERFACE)
Streamlit chat interface, conversation management, user input handling, response display, chat history persistence

### pages/1_Analytics.py
Graph statistics and visualizations (horses, events, sensors distributions)

### pages/2_News.py
Equestrian news aggregation with AI-generated summaries

## Data Layer

### Horse_generatedDataV2.rdf
Semantic ontology in RDF/OWL format containing all domain knowledge

### test_dataset.json
40 evaluation questions with ground truth answers, categorized by type and difficulty

### conversations/
Saved chat histories in JSON format for persistence

## Evaluation Results

Performance on 40 test questions:
- **Overall Accuracy:** 91.8%
- **Success Rate:** 100% (all questions answered)
- **Average Response Time:** 2.3 seconds
- **Cypher Generation Accuracy:** 95%

**By Question Category:**
- Simple Retrieval: 95% accuracy
- Multi-hop Reasoning: 92% accuracy
- Aggregation Queries: 88% accuracy
- Comparison Queries: 90% accuracy

## Running Evaluations

To test system accuracy:

```bash
python scripts/graph_rag/run_evaluation.py
```

Results are saved to `evaluation_results/` with detailed metrics.

## Features

### Core Chatbot
- Natural language understanding in French
- Accurate query translation (Cypher generation)
- Grounded answers (no hallucination)
- Technical query inspection

### Analytics Dashboard
Navigate to "Analytics" page to view:
- Total counts (horses, events, riders, sensors)
- Sensor distribution by position
- Event type breakdown
- Training intensity analysis

### News Aggregation
Navigate to "News" page for:
- Latest equestrian news from RSS feeds
- AI-generated weekly summaries
- Upcoming event extraction

## Development

### Adding New Questions

1. Add test cases to `data/test_dataset.json`
2. Run evaluation: `python scripts/graph_rag/run_evaluation.py`
3. Adjust prompts in `backend/graph_rag/llm_service.py` if needed

### Extending the Ontology

1. Update RDF file: `data/Horse_generatedDataV2.rdf`
2. Re-run database setup: `python scripts/setup_database.py`
3. Update Cypher prompt with new schema details

## Configuration

### Neo4j Setup

**Local Installation**
```bash
# Download from https://neo4j.com/download/
# Start Neo4j
# Access browser at http://localhost:7474
```

### OpenAI API

Get your API key from https://platform.openai.com/api-keys

**Cost Estimate:**
- Average cost per question: ~$0.003-0.005 USD
- 100 questions ≈ $0.30-0.50 USD


## Documentation

For detailed implementation information, see [IMPLEMENTATION.md](docs/IMPLEMENTATION.md)

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


**Version:** 1.0.0  
**Last Updated:** January 2025  
**Status:** Active Development

