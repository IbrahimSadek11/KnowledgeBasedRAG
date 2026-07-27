"""
Manual retrieval sanity check against the Chroma collection built by
index_corpus.py.

Usage (from repo root, after indexing):
    python scripts/textual_rag/test_retrieval.py

Fails loudly if collection "equestrian_textual_corpus" does not exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import chromadb
from openai import OpenAI

from backend.config import OPENAI_API_KEY

CHROMA_DIR = REPO_ROOT / "data" / "textual_rag" / "chroma_db"
COLLECTION_NAME = "equestrian_textual_corpus"
EMBEDDING_MODEL = "text-embedding-3-small"
N_RESULTS = 3
PREVIEW_CHARS = 150

TEST_QUERIES = [
    "Quelle est la race de Dakota ?",
    "Quels événements ont eu lieu à Saumur ?",
    "Quels capteurs IMU sont utilisés pour détecter la fatigue ?",
]


def embed_query(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def preview(text: str | None, n: int = PREVIEW_CHARS) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= n:
        return compact
    return compact[:n].rstrip() + "..."


def main() -> int:
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set. Load it via .env / backend.config.")
        return 1

    if not CHROMA_DIR.is_dir():
        print(f"ERROR: Chroma directory not found: {CHROMA_DIR}")
        print("Run scripts/textual_rag/index_corpus.py first.")
        return 1

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: collection '{COLLECTION_NAME}' not found at {CHROMA_DIR}")
        print(f"Detail: {exc}")
        print("Run scripts/textual_rag/index_corpus.py first.")
        return 1

    print(f"Connected to collection '{COLLECTION_NAME}' (count={collection.count()})")
    print(f"Chroma path: {CHROMA_DIR}")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    for i, query in enumerate(TEST_QUERIES, start=1):
        print("\n" + "=" * 72)
        print(f"QUERY {i}/{len(TEST_QUERIES)}")
        print("=" * 72)
        print(f"Text: {query}")
        print("-" * 72)

        query_embedding = embed_query(openai_client, query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        if not ids:
            print("  (no results)")
            continue

        for rank, (doc_id, doc_text, meta, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            meta = meta or {}
            filename = meta.get("filename", doc_id)
            entity_type = meta.get("entity_type", "?")
            print(f"  Rank {rank}")
            print(f"    filename:     {filename}")
            print(f"    entity_type:  {entity_type}")
            print(f"    distance:     {distance}")
            print(f"    preview:      {preview(doc_text)}")
            print()

    print("=" * 72)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
