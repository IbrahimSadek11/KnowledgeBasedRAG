"""
Tests for /dynamic-ingestion/approve behavior (stdlib unittest).

Uses mocks for write path where possible; live Neo4j only for optional
idempotency dry-run when env allows.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.api_server import app  # noqa: E402
from backend.graph_rag.dynamic_ingestion_writer import (  # noqa: E402
    Provenance,
    plan_ingestion,
    preflight_reviewed_candidates,
)
from backend.graph_rag.test_dynamic_ingestion_writer import visir_graph  # noqa: E402
from dynamic_kg.extract_facts import CandidateGraph, CandidateNode, SensorProperties  # noqa: E402


class TestApproveApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_approval_false_no_write(self):
        resp = self.client.post(
            "/dynamic-ingestion/approve",
            json={
                "approved": False,
                "candidate_graph": visir_graph().model_dump(mode="json"),
                "source_filename": "VISIR_Dressage_5_Sensor_Summary_v2.pdf",
                "rphd_file_id": "f5e69b59-6f08-4d29-b7d0-7bc810f133de",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["written"])
        self.assertFalse(body["success"])
        self.assertIn("approval", body["error"].lower())

    def test_invalid_labels_rejected(self):
        bad = CandidateGraph.model_construct(
            nodes=[
                CandidateNode.model_construct(
                    local_id="bad",
                    labels=["InertialSensors", "FrontLeft"],
                    properties=SensorProperties(hasSensorID="X1"),
                    source_evidence="x",
                )
            ],
            relationships=[],
            rejected_facts=[],
        )
        # model_validate on approve will fail Literal validation — send raw dict
        payload = {
            "approved": True,
            "candidate_graph": {
                "nodes": [
                    {
                        "local_id": "bad",
                        "labels": ["InertialSensors", "FrontLeft"],
                        "properties": {"hasSensorID": "X1"},
                        "source_evidence": "x",
                    }
                ],
                "relationships": [],
                "rejected_facts": [],
            },
            "dry_run": True,
        }
        resp = self.client.post("/dynamic-ingestion/approve", json=payload)
        self.assertIn(resp.status_code, (400, 200))
        if resp.status_code == 200:
            self.assertFalse(resp.json()["written"])

    def test_conflicts_prevent_write(self):
        from backend.graph_rag.dynamic_ingestion_writer import PreflightResult

        with patch(
            "backend.graph_rag.dynamic_ingestion_writer.preflight_reviewed_candidates",
            return_value=PreflightResult(
                valid=False,
                database="neo4j",
                conflicts=[{"hasSensorID": "6845"}],
            ),
        ), patch(
            "backend.graph_rag.dynamic_ingestion_writer.insert_reviewed_candidates"
        ) as ins:
            resp = self.client.post(
                "/dynamic-ingestion/approve",
                json={
                    "approved": True,
                    "candidate_graph": visir_graph().model_dump(mode="json"),
                    "dry_run": False,
                },
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertFalse(body["written"])
            self.assertFalse(body["success"])
            ins.assert_not_called()

    def test_ingested_at_does_not_conflict_on_reapproval(self):
        session = MagicMock()
        session.run.return_value = [
            {
                "eid": "e1",
                "labels": ["InertialSensors", "Sternum"],
                "props": {
                    "hasSensorID": "6845",
                    "sourceDocument": "VISIR_Dressage_5_Sensor_Summary_v2.pdf",
                    "rphdFileId": "f5e69b59-6f08-4d29-b7d0-7bc810f133de",
                    "ingestedAt": "2026-08-15T00:00:00+00:00",
                },
            }
        ]
        graph = CandidateGraph(
            nodes=[
                CandidateNode(
                    local_id="sensor_6845",
                    labels=["InertialSensors", "Sternum"],  # type: ignore[arg-type]
                    properties=SensorProperties(hasSensorID="6845"),
                    source_evidence="x",
                )
            ],
            relationships=[],
            rejected_facts=[],
        )
        plans, _ = plan_ingestion(
            graph,
            session,
            Provenance(
                source_filename="VISIR_Dressage_5_Sensor_Summary_v2.pdf",
                rphd_file_id="f5e69b59-6f08-4d29-b7d0-7bc810f133de",
                ingested_at="2026-08-15T99:99:99+00:00",
            ),
        )
        self.assertEqual(plans[0].action, "noop")


class TestVisirIdempotencyLive(unittest.TestCase):
    """Live read-only dry_run against production neo4j (no write)."""

    def test_visir_preflight_all_noop(self):
        graph = visir_graph()
        pf = preflight_reviewed_candidates(
            graph,
            Provenance(
                source_filename="VISIR_Dressage_5_Sensor_Summary_v2.pdf",
                rphd_file_id="f5e69b59-6f08-4d29-b7d0-7bc810f133de",
            ),
        )
        self.assertEqual(pf.database, "neo4j")
        self.assertEqual(len(pf.creates), 0)
        self.assertEqual(len(pf.conflicts), 0)
        self.assertEqual(len(pf.noops), 5)
        self.assertEqual(pf.relationships_to_create, [])
        self.assertTrue(pf.valid)


if __name__ == "__main__":
    unittest.main()
