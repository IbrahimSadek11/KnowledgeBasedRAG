# Graph Cypher validator trio — status

**Status:** archived only. Not part of the live Graph RAG pipeline.

## What exists

Under `backend/graph_rag/_archive/`:

- `cypher_validator.py`
- `cypher_retry.py`
- `validated_chain.py`

These were reconstructed from stale `backend/__pycache__/*.cpython-314.pyc` bytecode
(dated 2026-07-14). They were **never committed to git** and their own docstrings
state they were **never wired into `init_graph_chain()`**.

## Live pipeline

`backend/graph_rag/llm_service.py` still uses LangChain’s stock
`GraphCypherQAChain`. Do not import or integrate the archive modules unless
explicitly approved in a separate task.
