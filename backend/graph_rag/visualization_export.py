"""
Read-only production Neo4j export for RPHD Visualization.

Target: NEO4J_DATABASE (default neo4j). Never dynamickg. Never writes.

Preferred scope: explicit stable node IDs from a KnowledgeGraph generation
(uri / id / hasSensorID). Provenance filters remain as a legacy fallback only.

When exact_edges=True (RPHD generated KG), relationships are included only when
their composite membership id TYPE:source:target is in relationship_ids —
same identity as dynamic_ingestion_writer / chat membership scope.

None vs [] semantics (critical):
  stable_node_ids is None  → membership scope NOT requested (legacy/global OK)
  stable_node_ids == []    → membership scope requested with zero members
                             → return empty nodes (NEVER full MATCH (n))
  Same distinction for relationship_ids when combined with exact_edges.
"""

from __future__ import annotations

from typing import Any, Sequence

from neo4j import GraphDatabase

from backend.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from backend.graph_rag.membership_identity import (
    relationship_membership_id,
    stable_node_cypher_expr,
    stable_node_id_from_props,
)

# Prefer domain label over position subclasses for coloring
_PRIMARY_LABEL_PRIORITY = (
    "Horse",
    "Rider",
    "Veterinarian",
    "Caretaker",
    "EventParticipation",
    "CompetitiveSeason",
    "ShowJumping",
    "Dressage",
    "Cross",
    "PreparationStage",
    "PreCompetitionStage",
    "CompetitionStage",
    "TransitionStage",
    "ExperimentalObjective",
    "InertialSensors",
    "Resource",
)


def _stable_node_id(props: dict[str, Any], element_id: str) -> str:
    stable = stable_node_id_from_props(props)
    if stable is not None:
        return stable
    return f"neo4j:{element_id}"


def _display_label(props: dict[str, Any], labels: list[str]) -> str:
    for key in ("hasName", "hasSensorID", "name", "id"):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value)
    if labels:
        return labels[0]
    return "Node"


def _primary_type(labels: list[str]) -> str:
    label_set = set(labels)
    for preferred in _PRIMARY_LABEL_PRIORITY:
        if preferred in label_set:
            return preferred
    for label in labels:
        if label != "Resource":
            return label
    return labels[0] if labels else "Entity"


def _exact_edge_predicate_cypher() -> str:
    """Composite TYPE:source:target IN $rel_ids using start/end node stables."""
    start_stable = stable_node_cypher_expr("a")
    end_stable = stable_node_cypher_expr("b")
    return (
        f"(type(r) + ':' + {start_stable} + ':' + {end_stable}) IN $rel_ids"
    )


def _normalize_id_list(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    return [str(x).strip() for x in values if str(x).strip()]


def resolve_export_scope(
    *,
    stable_node_ids: list[str] | None = None,
    relationship_ids: list[str] | None = None,
    exact_edges: bool = False,
    rphd_file_ids: list[str] | None = None,
    source_documents: list[str] | None = None,
) -> dict[str, Any]:
    """
    Decide export mode without touching Neo4j.

    Presence of the list argument (including []) means scope was requested.
    Absence (None) means the caller did not request that dimension of scope.
    """
    node_scope_requested = stable_node_ids is not None
    rel_scope_requested = relationship_ids is not None
    node_ids = _normalize_id_list(stable_node_ids)
    rel_ids = _normalize_id_list(relationship_ids)
    file_ids = _normalize_id_list(rphd_file_ids)
    documents = _normalize_id_list(source_documents)

    # Explicit empty selected membership must never fall through to global MATCH.
    membership_scoped = node_scope_requested
    provenance_scoped = bool(file_ids or documents) and not membership_scoped
    scoped = membership_scoped or provenance_scoped
    # Exact edges when caller asked for relationship membership mode (incl. []).
    use_exact_edges = bool(exact_edges) and (
        membership_scoped or rel_scope_requested
    )

    return {
        "node_scope_requested": node_scope_requested,
        "rel_scope_requested": rel_scope_requested,
        "node_ids": node_ids,
        "rel_ids": rel_ids,
        "file_ids": file_ids,
        "documents": documents,
        "membership_scoped": membership_scoped,
        "provenance_scoped": provenance_scoped,
        "scoped": scoped,
        "use_exact_edges": use_exact_edges,
    }


def export_production_graph(
    *,
    stable_node_ids: list[str] | None = None,
    relationship_ids: list[str] | None = None,
    exact_edges: bool = False,
    rphd_file_ids: list[str] | None = None,
    source_documents: list[str] | None = None,
    limit_nodes: int | None = None,
) -> dict[str, Any]:
    """
    Export production Neo4j as RPHD/Cytoscape-friendly GraphData.

    Membership-first: when stable_node_ids is not None, return exactly those
    nodes (matched on uri OR id OR hasSensorID). Empty list → zero nodes.

    Relationships:
    - exact_edges=True with membership/rel scope: only edges whose composite
      TYPE:source:target id is in relationship_ids (empty ⇒ no edges).
    - legacy (exact_edges=False): node-induced subgraph between selected
      endpoints when membership has nodes.
    - no scope args at all: full production graph (legacy global consumer).
    """
    database = NEO4J_DATABASE or "neo4j"
    scope = resolve_export_scope(
        stable_node_ids=stable_node_ids,
        relationship_ids=relationship_ids,
        exact_edges=exact_edges,
        rphd_file_ids=rphd_file_ids,
        source_documents=source_documents,
    )
    node_ids = scope["node_ids"]
    rel_ids = scope["rel_ids"]
    file_ids = scope["file_ids"]
    documents = scope["documents"]
    membership_scoped = scope["membership_scoped"]
    provenance_scoped = scope["provenance_scoped"]
    scoped = scope["scoped"]
    use_exact_edges = scope["use_exact_edges"]

    # Explicit empty node membership: do not open Neo4j for a full scan.
    if membership_scoped and not node_ids:
        return {
            "database": database,
            "scoped": True,
            "nodes": [],
            "edges": [],
            "metadata": {
                "totalFiles": 0,
                "processedFiles": 0,
                "extractedEntities": 0,
                "extractedRelations": 0,
                "neo4jNodeCount": None,
                "neo4jEdgeCount": None,
                "scopedNodeCount": 0,
                "scopedEdgeCount": 0,
                "stableNodeIds": [],
                "relationshipIds": rel_ids if scope["rel_scope_requested"] else [],
                "exactEdges": use_exact_edges,
                "rphdFileIds": file_ids,
                "sourceDocuments": documents,
                "ontology": "Horse_V9 + Dynamic PDF ingestion",
                "source": "neo4j",
            },
        }

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=database) as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS n").single()[
                "n"
            ]
            total_edges = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS n"
            ).single()["n"]

            if membership_scoped:
                node_query = """
                MATCH (n)
                WHERE n.uri IN $node_ids
                   OR n.id IN $node_ids
                   OR n.hasSensorID IN $node_ids
                RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props
                """
                params: dict[str, Any] = {"node_ids": node_ids}
            elif provenance_scoped:
                node_query = """
                MATCH (n)
                WHERE ($file_ids <> [] AND n.rphdFileId IN $file_ids)
                   OR ($documents <> [] AND n.sourceDocument IN $documents)
                RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props
                """
                params = {
                    "file_ids": file_ids,
                    "documents": documents,
                }
            else:
                node_query = """
                MATCH (n)
                RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props
                """
                params = {}

            if limit_nodes is not None:
                node_query += f" LIMIT {int(limit_nodes)}"

            node_rows = session.run(node_query, params).data()
            eid_to_stable: dict[str, str] = {}
            eid_to_props: dict[str, dict[str, Any]] = {}
            nodes: list[dict[str, Any]] = []

            for row in node_rows:
                eid = row["eid"]
                labels = list(row["labels"] or [])
                props = dict(row["props"] or {})
                stable = _stable_node_id(props, eid)
                eid_to_stable[eid] = stable
                eid_to_props[eid] = props
                nodes.append(
                    {
                        "id": stable,
                        "label": _display_label(props, labels),
                        "type": _primary_type(labels),
                        "properties": {
                            **props,
                            "labels": labels,
                        },
                    }
                )

            if scoped and eid_to_stable:
                if use_exact_edges:
                    # Exact relationship membership (composite IDs). Empty → no edges.
                    if not rel_ids:
                        rel_rows = []
                    else:
                        edge_pred = _exact_edge_predicate_cypher()
                        rel_rows = session.run(
                            f"""
                            MATCH (a)-[r]->(b)
                            WHERE elementId(a) IN $eids AND elementId(b) IN $eids
                              AND {edge_pred}
                            RETURN elementId(a) AS source_eid,
                                   elementId(b) AS target_eid,
                                   elementId(r) AS rid,
                                   type(r) AS rtype
                            """,
                            {
                                "eids": list(eid_to_stable.keys()),
                                "rel_ids": rel_ids,
                            },
                        ).data()
                else:
                    # Legacy: node-induced subgraph between selected endpoints.
                    rel_rows = session.run(
                        """
                        MATCH (a)-[r]->(b)
                        WHERE elementId(a) IN $eids AND elementId(b) IN $eids
                        RETURN elementId(a) AS source_eid,
                               elementId(b) AS target_eid,
                               elementId(r) AS rid,
                               type(r) AS rtype
                        """,
                        {"eids": list(eid_to_stable.keys())},
                    ).data()
            elif scoped:
                rel_rows = []
            else:
                rel_rows = session.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN elementId(a) AS source_eid,
                           elementId(b) AS target_eid,
                           elementId(r) AS rid,
                           type(r) AS rtype
                    """
                ).data()

            # Defense in depth for exact mode: filter by Python identity too.
            if use_exact_edges and rel_ids:
                rel_id_set = set(rel_ids)
                filtered_rows = []
                for row in rel_rows:
                    mid = relationship_membership_id(
                        row["rtype"],
                        eid_to_props.get(row["source_eid"]),
                        eid_to_props.get(row["target_eid"]),
                    )
                    if mid is not None and mid in rel_id_set:
                        filtered_rows.append(row)
                rel_rows = filtered_rows

            edges: list[dict[str, Any]] = []
            for row in rel_rows:
                source = eid_to_stable.get(row["source_eid"])
                target = eid_to_stable.get(row["target_eid"])
                if not source or not target:
                    continue
                edges.append(
                    {
                        "id": f"rel:{row['rid']}",
                        "source": source,
                        "target": target,
                        "label": row["rtype"],
                        "type": row["rtype"],
                        "weight": 1,
                    }
                )

            return {
                "database": database,
                "scoped": scoped,
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "totalFiles": 0,
                    "processedFiles": 0,
                    "extractedEntities": len(nodes),
                    "extractedRelations": len(edges),
                    "neo4jNodeCount": total_nodes,
                    "neo4jEdgeCount": total_edges,
                    "scopedNodeCount": len(nodes),
                    "scopedEdgeCount": len(edges),
                    "stableNodeIds": node_ids,
                    "relationshipIds": rel_ids,
                    "exactEdges": use_exact_edges,
                    "rphdFileIds": file_ids,
                    "sourceDocuments": documents,
                    "ontology": "Horse_V9 + Dynamic PDF ingestion",
                    "source": "neo4j",
                },
            }
    finally:
        driver.close()
