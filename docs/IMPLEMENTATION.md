# Implementation Details

This document provides detailed explanations of each system component, describing what each file does and how it contributes to the overall functionality.


## Table of Contents

1. [Backend Modules](#backend-modules)
2. [Frontend Modules](#frontend-modules)
3. [Scripts](#scripts)
4. [Data Files](#data-files)

---

![complete system architecture](../support_pictures/COMPLETE_SYSTEM_FLOW.png)


## Backend Modules

### config.py - Configuration Management

**Purpose**: Centralizes all system configuration in one location.

**What it does**:
- Loads environment variables from the `.env` file using `python-dotenv`
- Stores Neo4j database connection parameters (URI, username, password, database name)
- Stores OpenAI API key for LLM access
- Defines cost constants for tracking API usage (input/output/embedding token prices)

**Why it matters**: By keeping all configuration in one place, the system remains easy to deploy and modify. Other modules import settings from here, ensuring consistency across the application.

**Location**: `backend/config.py`

---

### graph_service.py - Neo4j Database Operations

**Purpose**: Handles all interactions with the Neo4j knowledge graph database.

**What it does**:
- **Connection Management**: Creates and maintains the connection to Neo4j using LangChain's `Neo4jGraph` wrapper
- **Schema Refresh**: Updates the graph schema metadata that the LLM uses to understand the database structure
- **Query Execution**: Runs Cypher queries against the database and returns results with error handling

**Core Functions**:
```python
def init_graph():
    """Establishes connection to Neo4j and refreshes schema"""
    
def execute_query(graph, query):
    """Executes a Cypher query and returns results or empty list on error"""
```

**Why it matters**: This module abstracts database operations, making the code modular and testable. It isolates database logic from business logic.

**Location**: `backend/graph_rag/graph_service.py`

---

### llm_service.py - GraphRAG Pipeline Core

**Purpose**: Orchestrates the complete GraphRAG pipeline from question to answer.

**What it does**:

1. **Cypher Generation Prompt** (`get_cypher_prompt()`):
   - Provides detailed instructions for translating natural language questions into Cypher queries
   - Specifies exact relationship directions (e.g., `(Rider)-[:ASSOCIATEDWITH]->(Horse)`)
   - Maps domain concepts to graph schema (e.g., "horse name" → `hasName` property)
   - Includes error prevention rules (e.g., avoiding non-existent labels)
   - Contains ~300 lines of carefully crafted instructions
   - Includes rules to prevent SUM() operations on string values (hasDuration, hasFrequency)

2. **Answer Generation Prompt** (`get_qa_prompt()`):
   - Instructs the LLM to generate answers using ONLY retrieved data (no hallucination)
   - Maps technical identifiers to human-readable names (e.g., `Horse1` → `Dakota`)
   - Enforces natural French language formatting
   - Prevents exposing raw data structures to users

3. **Pipeline Initialization** (`init_graph_chain()`):
   - Connects to Neo4j graph database
   - Configures OpenAI GPT-4o-mini with temperature=0 (deterministic output)
   - Loads both prompt templates
   - Assembles everything into LangChain's `GraphCypherQAChain`
   - Returns ready-to-use question-answering pipeline


**Example Pipeline Flow**:
```
User Question → Cypher Prompt + LLM → Cypher Query → Execute on Neo4j 
→ Results → QA Prompt + LLM → Natural Language Answer
```

**Why it matters**: This is the most critical file in the system. It implements the core GraphRAG logic that enables accurate question-answering over the knowledge graph.

**Location**: `backend/graph_rag/llm_service.py`

---

### news_service.py - News Aggregation

**Purpose**: Provides supplementary equestrian news functionality.

**What it does**:
- Fetches articles from RSS feeds of equestrian news sites
- Scrapes web content when necessary
- Uses GPT to generate weekly news summaries
- Extracts upcoming event information from articles

**Note**: This is a secondary feature demonstrating additional LLM capabilities beyond the core GraphRAG chatbot.

**Location**: `backend/news_service.py`

---

### evaluation_service.py - Quality Assessment

**Purpose**: Measures system accuracy and answer quality.

**What it does**:
- **Semantic Similarity**: Calculates how similar the system's answer is to the ground truth using embedding vectors and cosine similarity
- **LLM-as-Judge**: Uses GPT-4o-mini to evaluate answers on three criteria:
  - Correctness (are the facts accurate?)
  - Completeness (are all key points covered?)
  - Accuracy (are specific details like names and numbers correct?)
- **Combined Scoring**: Averages semantic similarity and LLM judge scores for comprehensive evaluation

**Core Functions**:
```python
def calculate_semantic_similarity(answer, ground_truth, embeddings):
    """Returns 0-1 similarity score using OpenAI embeddings"""
    
def llm_judge_answer(question, answer, ground_truth, judge_llm):
    """Returns scoring dictionary with detailed breakdown and reasoning"""
```

**Why it matters**: Enables objective, reproducible measurement of system performance across diverse question types and difficulty levels.

**Location**: `backend/evaluation_service.py`

---

## Frontend Modules

### app.py - Main Chatbot Interface

**Purpose**: Provides the primary user interface for asking questions and viewing answers.

**What it does**:
- **Chat UI**: Renders a ChatGPT-style conversation interface using Streamlit
- **User Input**: Captures questions through `st.chat_input()`
- **Backend Integration**: Calls `init_graph_chain()` and invokes the pipeline with user questions
- **Response Display**: Shows LLM responses in real-time with proper formatting
- **Conversation Management**:
  - Creates new conversation sessions
  - Loads previous conversations from JSON files
  - Saves conversation history automatically
  - Allows renaming and deleting conversations
- **Session State**: Maintains conversation context across page reloads
- **Technical Details Display**: Shows generated Cypher queries in expandable sections

**User Workflow**:
```
1. User opens chatbot interface
2. Types question in French
3. System processes question through GraphRAG pipeline
4. Answer appears in chat
5. Conversation is automatically saved
6. User can continue asking follow-up questions
```

**Key Features**:
- Auto-generates conversation titles from first question
- Sidebar with conversation history
- Delete and rename conversation options
- System information panel showing Neo4j schema

**Why it matters**: This is the main interface users interact with. It bridges the gap between complex backend logic and simple, intuitive user experience.

**Location**: `frontend/app.py`

---

### pages/1_Analytics.py - Dashboard

**Purpose**: Displays statistical insights about the knowledge graph.

**What it does**:
- Shows aggregate metrics (total horses, events, riders, sensors)
- Visualizes data distributions with interactive Plotly charts:
  - Sensor positions distribution
  - Event type breakdown
  - Training intensity analysis
  - Horse participation statistics
- Queries the graph for summary statistics
- Real-time data visualization

**Visualizations**:
- KPI cards showing counts
- Pie charts for categorical distributions
- Bar charts for comparative metrics
- Stacked charts for multi-dimensional data

**Note**: Provides useful visualizations but is secondary to the main chatbot functionality.

**Location**: `frontend/pages/1_Analytics.py`

---

### pages/2_News.py - News Feed

**Purpose**: Displays current equestrian news and events.

**What it does**:
- Fetches recent articles from equestrian news sources
- Generates AI-powered weekly summaries
- Extracts and displays upcoming competition dates
- Allows filtering articles by source
- Caches results to minimize API calls

**Features**:
- Tabbed interface: Summary / Events / All Articles
- Source filtering
- Refresh button for updates
- Event extraction with dates and locations

**Note**: Demonstrates LLM capabilities beyond GraphRAG, but not central to the research contribution.

**Location**: `frontend/pages/2_News.py`

---

## Scripts

### scripts/setup_database.py - Data Loading

**Purpose**: One-time conversion of RDF data into Neo4j database.

**What it does**:

**Step 1: Parse RDF File**
- Uses `rdflib` library to read `Horse_generatedDataV2.rdf`
- Extracts all RDF triples (subject-predicate-object statements)

**Step 2: Transform Entities**
- For each entity (subject):
  - Extracts entity types to become Neo4j labels (e.g., `Horse`, `Rider`)
  - Extracts literal properties (e.g., `hasName`, `hasRace`)
  - Cleans URIs to short identifiers (e.g., `http://...#Horse1` → `Horse1`)

**Step 3: Create Neo4j Nodes**
- Generates Cypher `MERGE` statements for each entity
- Sets all properties on the node
- Executes in Neo4j database

**Step 4: Create Relationships**
- For each RDF relationship triple:
  - Maps to Neo4j relationship
  - Creates connection between nodes
  - Preserves relationship semantics from RDF

**Helper Functions**:
```python
def clean_uri(uri):
    """Extracts short name from full URI"""
    
def get_value(obj):
    """Extracts value from RDF literal or reference"""
```

**Output**: Fully populated Neo4j knowledge graph with proper schema and indexes.

**Why it matters**: Bridges semantic web standards (RDF/OWL) with modern graph databases (Neo4j), enabling the GraphRAG pipeline.

**Location**: `scripts/setup_database.py`

---

### scripts/graph_rag/run_evaluation.py - System Testing

**Purpose**: Automated testing of system accuracy and performance.

**What it does**:

**Step 1: Load Test Questions**
- Reads 40 test questions from `data/test_dataset.json`
- Each question includes ground truth answer, category, and difficulty level

**Step 2: Process Each Question**
- Sends question to GraphRAG system
- Measures response latency
- Tracks API costs

**Step 3: Evaluate Quality**
- Calculates semantic similarity between system answer and ground truth
- Runs LLM-as-judge evaluation for detailed scoring
- Records all metrics

**Step 4: Aggregate Results**
- Groups results by question category (simple, multi-hop, aggregation, etc.)
- Groups results by difficulty level (easy, medium, hard, very hard, extreme)
- Calculates average scores, success rates, and costs

**Step 5: Generate Report**
- Saves detailed JSON report to `evaluation_results/` directory
- Prints summary statistics to console
- Highlights best and worst performing questions

**Output Example**:
```json
{
  "metadata": {
    "total_questions": 40,
    "avg_time_per_question": 2.3,
    "total_cost_usd": 0.15
  },
  "overall_metrics": {
    "success_rate": 0.925,
    "avg_semantic_similarity": 0.847,
    "avg_combined_score": 0.864
  },
  "detailed_results": [...]
}
```

**Why it matters**: Provides objective, reproducible measurements of system quality and enables tracking improvements over time.

**Location**: `scripts/graph_rag/run_evaluation.py`

---

## Data Files

### Horse_generatedDataV2.rdf

**Format**: RDF/XML (Resource Description Framework)

**Content**: Complete equestrian sports ontology including:
- 2 horses (Dakota, Naya) with properties (breed, training intensity)
- 3 riders with associations
- 4 training stages (Preparation, Pre-Competition, Competition, Transition)
- 3 event types (ShowJumping, Dressage, Cross)
- 4 IMU sensors with positions and configurations
- 2 experimental objectives (Gait Classification, Fatigue Detection)

**Purpose**: Source of truth for all domain knowledge

**Location**: `data/Horse_generatedDataV2.rdf`

---

### test_dataset.json

**Format**: JSON array of test cases

**Structure**:
```json
{
  "question": "French natural language question",
  "ground_truth": "Expected correct answer",
  "category": "simple|multi-hop|aggregation|comparison",
  "difficulty": "easy|medium|hard|very_hard|extreme"
}
```

**Content**: 40 carefully designed questions covering:
- Simple property retrieval (15 questions)
- Relationship traversal (10 questions)
- Multi-hop reasoning (8 questions)
- Aggregation queries (4 questions)
- Comparison queries (3 questions)

**Purpose**: Benchmark for automated evaluation

**Location**: `data/test_dataset.json`

---

### conversations/

**Format**: JSON files named `conv_YYYYMMDD_HHMMSS.json`

**Structure**:
```json
{
  "title": "Conversation title",
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp",
  "messages": [
    {"role": "user", "content": "Question text"},
    {"role": "assistant", "content": "Answer text", "cypher": "Generated query"}
  ]
}
```

**Purpose**: Persistent storage of chat histories

**Location**: `data/conversations/`

---

## File Interaction Flow

```
User Input (frontend/app.py)
         ↓
LLM Service (backend/graph_rag/llm_service.py)
         ↓
Graph Service (backend/graph_rag/graph_service.py)
         ↓
Neo4j Database (loaded by scripts/setup_database.py)
```

## Configuration Files

### .env

Contains sensitive credentials (not committed to git):
```env
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

### requirements.txt

Python dependencies with pinned versions for reproducibility

---

## Development Guidelines

### Adding New Features

1. Backend logic → `backend/` modules
2. UI components → `frontend/` pages
3. Data processing → `scripts/`

### Modifying Prompts

- Cypher generation: Edit `get_cypher_prompt()` in `llm_service.py`
- Answer formatting: Edit `get_qa_prompt()` in `llm_service.py`

### Adding New Entities

1. Update RDF file: `data/Horse_generatedDataV2.rdf`
2. Run: `python scripts/setup_database.py`
3. Update prompts with new entity types and relationships



