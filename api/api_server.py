"""
Equestrian Graph RAG API — thin HTTP wrapper around the existing pipeline.

Run from the repo root:
    python -m uvicorn api.api_server:app --host 0.0.0.0 --port 8500

Test once running:
    curl -X POST http://localhost:8500/query -H "Content-Type: application/json" -d "{\"question\": \"Quel cheval a le plus de participations?\"}"
"""

from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.fusion.adapters import get_graph_chain
from backend.graph_rag.cypher_context_enrichment import (
    invoke_graph_chain_with_qa_context_enrichment,
)
from backend.graph_rag.cypher_membership_scope import (
    invoke_graph_chain_with_membership_scope,
)
from backend.graph_rag.cypher_sensor_identity import (
    invoke_graph_chain_with_cypher_retry,
)

app = FastAPI(title="Equestrian Graph RAG API")

_chain = None


@app.on_event("startup")
def load_chain():
    global _chain
    _chain = get_graph_chain()


class QueryRequest(BaseModel):
    question: str
    # Optional selected Knowledge Graph membership (same ids as GET /graph).
    # When set, Cypher execution is constrained to these nodes / relationships.
    # graph_id is metadata only — enforcement uses the membership arrays.
    stable_node_ids: list[str] | None = None
    relationship_ids: list[str] | None = None
    graph_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    cypher_query: str | None = None
    cypher_retry_used: bool = False
    retry_reason: str | None = None
    original_cypher: str | None = None
    membership_scoped: bool = False
    human_identity_retry_used: bool = False
    human_identity_retry_reason: str | None = None


def _extract_cypher_query(result: dict) -> str | None:
    if result.get("final_cypher"):
        return result["final_cypher"]
    steps = result.get("intermediate_steps")
    if steps and isinstance(steps, list) and len(steps) > 0:
        first_step = steps[0]
        if isinstance(first_step, dict):
            return first_step.get("query")
    return None


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    def _scoped_invoke(chain, inputs, config=None):
        return invoke_graph_chain_with_membership_scope(
            chain,
            inputs,
            stable_node_ids=request.stable_node_ids,
            relationship_ids=request.relationship_ids,
            config=config,
            _invoke=invoke_graph_chain_with_cypher_retry,
        )

    result = invoke_graph_chain_with_qa_context_enrichment(
        _chain,
        {"query": request.question},
        _invoke=_scoped_invoke,
    )
    return QueryResponse(
        answer=result.get("result", ""),
        cypher_query=_extract_cypher_query(result),
        cypher_retry_used=bool(result.get("cypher_retry_used", False)),
        retry_reason=result.get("retry_reason"),
        original_cypher=result.get("original_cypher"),
        membership_scoped=bool(result.get("membership_scoped", False)),
        human_identity_retry_used=bool(
            result.get("human_identity_retry_used", False)
        ),
        human_identity_retry_reason=result.get("human_identity_retry_reason"),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graph")
def export_graph(
    stable_node_id: list[str] | None = Query(
        default=None,
        description="Stable Neo4j node id(s): uri / id / hasSensorID",
    ),
    relationship_id: list[str] | None = Query(
        default=None,
        description="Optional relationship id(s) for membership export",
    ),
    exact_edges: bool = Query(
        default=False,
        description=(
            "When true, relationship_id is exact membership (empty ⇒ no edges). "
            "Presence of stable_node_id (including empty '') means membership "
            "scope was requested — never fall back to the full production graph. "
            "Omit stable_node_id entirely for legacy global export."
        ),
    ),
    rphd_file_id: list[str] | None = Query(
        default=None,
        description="Legacy provenance scope (avoid for reused nodes)",
    ),
    source_document: list[str] | None = Query(
        default=None,
        description="Legacy provenance filename scope",
    ),
):
    """
    Read-only production Neo4j export for RPHD Visualization.
    Prefer stable_node_id membership over rphd_file_id provenance.
    """
    from backend.graph_rag.visualization_export import export_production_graph

    try:
        return export_production_graph(
            stable_node_ids=stable_node_id,
            relationship_ids=relationship_id,
            exact_edges=exact_edges,
            rphd_file_ids=rphd_file_id,
            source_documents=source_document,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Graph export failed: {exc}",
        ) from exc


class PdfReceiveResponse(BaseModel):
    """
    Candidate extraction only (pypdf + gpt-4o-mini Structured Outputs).

    Does NOT review candidates for insertion and does NOT write to Neo4j /
    dynamickg. Distinguishes candidate_extraction from reviewed graph insertion.
    """

    received: bool
    filename: str
    size: int
    pages: int
    extracted_characters: int
    stage: str = "candidate_extraction"
    neo4j_write: bool = False
    candidate_graph: dict[str, Any]
    llm_extraction_calls: int = 0
    extraction_mode: str | None = None


@app.post("/pdf-receive", response_model=PdfReceiveResponse)
async def pdf_receive(file: UploadFile = File(...)):
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are accepted")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Upload does not look like a PDF")

    try:
        from dynamic_kg.extract_facts import (
            extract_candidates_from_pdf_bytes,
            get_last_extraction_stats,
        )

        graph, pages, extracted_characters = extract_candidates_from_pdf_bytes(
            contents
        )
        stats = get_last_extraction_stats()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Candidate extraction failed: {exc}",
        ) from exc

    return PdfReceiveResponse(
        received=True,
        filename=filename,
        size=len(contents),
        pages=pages,
        extracted_characters=extracted_characters,
        stage="candidate_extraction",
        neo4j_write=False,
        candidate_graph=graph.model_dump(mode="json"),
        llm_extraction_calls=int(stats.get("llm_calls") or 0),
        extraction_mode=(
            str(stats["mode"]) if stats.get("mode") is not None else None
        ),
    )


class DynamicIngestionApproveRequest(BaseModel):
    """
    Explicit human approval of a reviewed CandidateGraph.
    Does not call GPT. Does not re-extract PDF.
    """

    approved: bool
    candidate_graph: dict[str, Any]
    source_filename: str | None = None
    rphd_file_id: str | None = None
    source_hash: str | None = None
    dry_run: bool = False


class DynamicIngestionApproveResponse(BaseModel):
    success: bool
    written: bool
    database: str
    preflight: dict[str, Any]
    insert: dict[str, Any] | None = None
    error: str | None = None
    affected_nodes: list[dict[str, Any]] = []
    affected_relationships: list[dict[str, Any]] = []


@app.post(
    "/dynamic-ingestion/approve",
    response_model=DynamicIngestionApproveResponse,
)
def approve_dynamic_ingestion(request: DynamicIngestionApproveRequest):
    """
    Preflight reviewed candidates; optionally write to production NEO4J_DATABASE.

    Requires approved=true to mutate. dry_run=true never writes.
    Returns affected_nodes for created + reused (+ updated) membership tracking.
    """
    from backend.config import NEO4J_DATABASE
    from backend.graph_rag.dynamic_ingestion_writer import (
        DynamicIngestionValidationError,
        Provenance,
        insert_reviewed_candidates,
        preflight_reviewed_candidates,
    )
    from dynamic_kg.extract_facts import CandidateGraph

    database = NEO4J_DATABASE or "neo4j"

    if request.approved is not True:
        return DynamicIngestionApproveResponse(
            success=False,
            written=False,
            database=database,
            preflight={},
            insert=None,
            error="Explicit approval required: set approved=true",
            affected_nodes=[],
            affected_relationships=[],
        )

    try:
        graph = CandidateGraph.model_validate(request.candidate_graph)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Invalid candidate_graph: {exc}",
        ) from exc

    provenance = Provenance(
        source_filename=request.source_filename,
        rphd_file_id=request.rphd_file_id,
        source_hash=request.source_hash,
    )

    try:
        preflight = preflight_reviewed_candidates(graph, provenance)
    except DynamicIngestionValidationError as exc:
        return DynamicIngestionApproveResponse(
            success=False,
            written=False,
            database=database,
            preflight={"valid": False, "rejected": [{"reason": str(exc)}]},
            insert=None,
            error=str(exc),
            affected_nodes=[],
            affected_relationships=[],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Preflight failed: {exc}",
        ) from exc

    preflight_dict = preflight.to_dict()
    affected_nodes = list(preflight.affected_nodes or [])
    affected_relationships = list(preflight.affected_relationships or [])

    if not preflight.valid:
        return DynamicIngestionApproveResponse(
            success=False,
            written=False,
            database=preflight.database,
            preflight=preflight_dict,
            insert=None,
            error="Preflight invalid: conflicts or rejected items prevent write",
            # Still return membership candidates that would have participated
            # only for create/noop/update; conflicts are excluded by builder.
            affected_nodes=affected_nodes,
            affected_relationships=affected_relationships,
        )

    if request.dry_run:
        return DynamicIngestionApproveResponse(
            success=True,
            written=False,
            database=preflight.database,
            preflight=preflight_dict,
            insert=None,
            error=None,
            affected_nodes=affected_nodes,
            affected_relationships=affected_relationships,
        )

    try:
        insert_result = insert_reviewed_candidates(
            graph,
            provenance=provenance,
            confirm_write_to_production=True,
        )
    except PermissionError as exc:
        return DynamicIngestionApproveResponse(
            success=False,
            written=False,
            database=database,
            preflight=preflight_dict,
            insert=None,
            error=str(exc),
            affected_nodes=affected_nodes,
            affected_relationships=affected_relationships,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Insert failed: {exc}",
        ) from exc

    return DynamicIngestionApproveResponse(
        success=insert_result.success,
        written=insert_result.success,
        database=insert_result.database,
        preflight=preflight_dict,
        insert=insert_result.to_dict(),
        error=insert_result.error,
        affected_nodes=insert_result.affected_nodes or affected_nodes,
        affected_relationships=insert_result.affected_relationships
        or affected_relationships,
    )
