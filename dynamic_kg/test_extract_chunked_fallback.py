"""Regression: unmarked multi-page PDFs must still invoke LLM extraction."""

from __future__ import annotations

from unittest.mock import patch

from dynamic_kg.extract_facts import (
    CandidateGraph,
    CandidateNode,
    NodeProperties,
    extract_candidates_chunked,
    get_last_extraction_stats,
)


def _fake_invoke(text: str, *, extra_instructions: str = "") -> CandidateGraph:
    return CandidateGraph(
        nodes=[
            CandidateNode(
                local_id="Horse_Bella",
                labels=["Horse"],
                properties=NodeProperties(id="Horse_Bella", hasName="Bella"),
                source_evidence="test",
            )
        ],
        relationships=[],
        rejected_facts=[],
    )


def test_unmarked_pages_use_generic_chunked_llm_not_empty_short_circuit():
    pages = [
        "Horse Bella race Lusitanien participates in Event_Marseille_Dressage_2026.",
        "Participation_Marseille_Bella_Julien rider Julien rank 1.",
        "Training stages and sensors for Bella.",
        "Season 2026 and related objectives.",
        "Extra page five.",
        "Extra page six.",
    ]
    with patch(
        "dynamic_kg.extract_facts._invoke_llm_candidate_graph",
        side_effect=_fake_invoke,
    ) as mocked:
        graph = extract_candidates_chunked(pages, pages_per_chunk=2)

    assert mocked.call_count == 3  # 6 pages / 2
    assert len(graph.nodes) >= 1
    stats = get_last_extraction_stats()
    assert stats["mode"] == "generic-chunked"
    assert stats["llm_calls"] == 3


def test_empty_pages_still_return_empty_without_llm():
    with patch(
        "dynamic_kg.extract_facts._invoke_llm_candidate_graph"
    ) as mocked:
        graph = extract_candidates_chunked(["", "  ", ""], pages_per_chunk=2)
    assert mocked.call_count == 0
    assert graph.nodes == []
    assert graph.relationships == []
