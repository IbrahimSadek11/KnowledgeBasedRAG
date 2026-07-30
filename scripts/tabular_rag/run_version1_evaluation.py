"""
Run the full tabular RAG evaluation against the version1 snapshot.

Loads:
  - backend/tabular_rag/version1/{tabular_chain,sql_validator}.py
  - data/tabular_rag/version1/tabular.db
  - scripts/tabular_rag/version1/gold_queries.py

Writes results to:
  - evaluation_results/tabular_rag/version1/

Does not modify live pipeline files.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1_BACKEND = ROOT / "backend" / "tabular_rag" / "version1"
V1_DB = ROOT / "data" / "tabular_rag" / "version1" / "tabular.db"
V1_SCRIPTS = ROOT / "scripts" / "tabular_rag" / "version1"
V1_RESULTS = ROOT / "evaluation_results" / "tabular_rag" / "version1"
LIVE_EVAL = ROOT / "scripts" / "tabular_rag" / "run_tabular_evaluation.py"


def _load_as(fullname: str, path: Path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def bootstrap_version1() -> None:
    if not V1_DB.is_file():
        raise FileNotFoundError(f"version1 DB missing: {V1_DB}")
    V1_RESULTS.mkdir(parents=True, exist_ok=True)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Prefer version1 gold_queries on import.
    sys.path.insert(0, str(V1_SCRIPTS))

    # Ensure package shell exists, then override modules with version1 copies.
    import backend.tabular_rag as tr_pkg  # noqa: F401

    sv = _load_as(
        "backend.tabular_rag.sql_validator",
        V1_BACKEND / "sql_validator.py",
    )
    tr_pkg.sql_validator = sv

    chain = _load_as(
        "backend.tabular_rag.tabular_chain",
        V1_BACKEND / "tabular_chain.py",
    )
    chain.DB_PATH = str(V1_DB)
    tr_pkg.tabular_chain = chain


def main() -> None:
    print("=" * 80)
    print("TABULAR RAG EVALUATION — PIPELINE VERSION1 SNAPSHOT")
    print("=" * 80)
    print(f"chain/validator : {V1_BACKEND}")
    print(f"sqlite DB       : {V1_DB}")
    print(f"gold queries    : {V1_SCRIPTS / 'gold_queries.py'}")
    print(f"results dir     : {V1_RESULTS}")
    print()

    bootstrap_version1()

    code = LIVE_EVAL.read_text(encoding="utf-8")
    # Point EX checks + report output at the version1 snapshot.
    code = code.replace(
        'DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "tabular_rag", "tabular.db"))',
        f'DB_PATH = r"{V1_DB}"',
    )
    code = code.replace(
        'RESULTS_DIR = Path(SCRIPT_DIR) / ".." / ".." / "evaluation_results" / "tabular_rag"',
        f'RESULTS_DIR = Path(r"{V1_RESULTS}")',
    )
    code = code.replace(
        '"pipeline": "tabular_rag"',
        '"pipeline": "tabular_rag_version1"',
    )
    code = code.replace(
        'print("🎯 TABULAR RAG EVALUATION — FULL 100-QUESTION BENCHMARK + EX")',
        'print("🎯 TABULAR RAG EVALUATION — VERSION1 FULL 100-QUESTION BENCHMARK + EX")',
    )

    # Keep __file__ at the live script depth so PROJECT_ROOT resolves correctly.
    ns = {
        "__name__": "__main__",
        "__file__": str(LIVE_EVAL),
    }
    exec(compile(code, str(LIVE_EVAL), "exec"), ns)


if __name__ == "__main__":
    main()
