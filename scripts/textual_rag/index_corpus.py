"""
One-time indexer: embed all textual corpus .txt documents into a local
Chroma persistent store.

Usage (from repo root):
    python scripts/textual_rag/index_corpus.py

Does not modify Neo4j, test_dataset.json, or other pipeline code.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import chromadb
from openai import OpenAI

from backend.config import OPENAI_API_KEY

DOCUMENTS_DIR = REPO_ROOT / "data" / "textual_rag" / "textual_corpus" / "documents"
CHROMA_DIR = REPO_ROOT / "data" / "textual_rag" / "chroma_db"
COLLECTION_NAME = "equestrian_textual_corpus"
EMBEDDING_MODEL = "text-embedding-3-small"


def infer_entity_type(path: Path) -> str:
    """Map parent folder name to entity_type metadata."""
    parent = path.parent.name.lower()
    if parent in {"horses", "horse"}:
        return "horse"
    if parent in {"events", "event"}:
        return "event"
    # Fallback: singularize a trailing 's' if present
    if parent.endswith("s") and len(parent) > 1:
        return parent[:-1]
    return parent


def embed_text(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def main() -> int:
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set. Load it via .env / backend.config.")
        return 1

    if not DOCUMENTS_DIR.is_dir():
        print(f"ERROR: documents directory not found: {DOCUMENTS_DIR}")
        return 1

    txt_files = sorted(DOCUMENTS_DIR.rglob("*.txt"))
    print(f"Found {len(txt_files)} .txt file(s) under {DOCUMENTS_DIR}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for path in txt_files:
        filename = path.name
        doc_id = path.stem
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError("file is empty")

            entity_type = infer_entity_type(path)
            embedding = embed_text(openai_client, text)

            collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[
                    {
                        "entity_type": entity_type,
                        "filename": filename,
                    }
                ],
            )
            succeeded.append(filename)
            print(f"[OK]   {filename}  (id={doc_id}, entity_type={entity_type})")
        except Exception as exc:  # noqa: BLE001 — continue indexing other files
            failed.append((filename, str(exc)))
            print(f"[FAIL] {filename}  — {exc}")

    print("\n========== SUMMARY ==========")
    print(f"Total files found:        {len(txt_files)}")
    print(f"Successfully indexed:     {len(succeeded)}")
    print(f"Failed:                   {len(failed)}")
    if failed:
        print("Failed filenames:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print(f"Collection count():       {collection.count()}")
    print(f"Chroma path:              {CHROMA_DIR}")
    print(f"Collection name:          {COLLECTION_NAME}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
