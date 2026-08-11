"""
Equestrian Graph RAG API — thin HTTP wrapper around the existing pipeline.

Run from the repo root:
    python -m uvicorn api.api_server:app --host 0.0.0.0 --port 8500

Test once running:
    curl -X POST http://localhost:8500/query -H "Content-Type: application/json" -d "{\"question\": \"Quel cheval a le plus de participations?\"}"
"""

from fastapi import FastAPI
from pydantic import BaseModel

from backend.fusion.adapters import get_graph_chain
from backend.graph_rag.llm_service import invoke_graph_chain_with_cypher_retry

app = FastAPI(title="Equestrian Graph RAG API")

_chain = None


@app.on_event("startup")
def load_chain():
    global _chain
    _chain = get_graph_chain()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    cypher_query: str | None = None
    cypher_retry_used: bool = False


def _extract_cypher_query(result: dict) -> str | None:
    steps = result.get("intermediate_steps")
    if steps and isinstance(steps, list) and len(steps) > 0:
        first_step = steps[0]
        if isinstance(first_step, dict):
            return first_step.get("query")
    return None


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = invoke_graph_chain_with_cypher_retry(_chain, {"query": request.question})
    return QueryResponse(
        answer=result.get("result", ""),
        cypher_query=_extract_cypher_query(result),
        cypher_retry_used=result.get("cypher_retry_used", False),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
