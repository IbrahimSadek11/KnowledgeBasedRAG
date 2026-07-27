"""
Phase-1 unit tests for compare_cypher_execution — written BEFORE any gold queries.

Hand-crafted result shapes only (no Neo4j, no test_dataset). Run:

    python scripts/graph_rag/test_compare_cypher_execution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "graph_rag"))

from gold_cypher_queries import compare_cypher_execution, normalize_cypher_result


def _check(name: str, matched: bool, expect_match: bool, detail: str = "") -> bool:
    ok = matched is expect_match
    status = "PASS" if ok else "FAIL"
    expected = "MATCH" if expect_match else "MISMATCH"
    got = "MATCH" if matched else "MISMATCH"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {name}: expected {expected}, got {got}{suffix}")
    return ok


def test_same_facts_different_column_order_dict_keys() -> bool:
    """RETURN name, race  vs  RETURN race, name — same values, key order differs."""
    a = [{"name": "Dakota", "race": "Selle Français"}]
    b = [{"race": "Selle Français", "name": "Dakota"}]
    matched, err = compare_cypher_execution(a, b)
    return _check(
        "T1 column-order (dict key order)",
        matched,
        True,
        f"err={err!r} normA={normalize_cypher_result(a)} normB={normalize_cypher_result(b)}",
    )


def test_same_facts_different_column_order_aliases() -> bool:
    """Different RETURN aliases, same cell values — column identity ignored."""
    a = [{"horse": "Naya", "breed": "Anglo-Arabe"}]
    b = [{"name": "Naya", "race": "Anglo-Arabe"}]
    matched, err = compare_cypher_execution(a, b)
    return _check(
        "T2 column-order (different aliases, same values)",
        matched,
        True,
        f"err={err!r}",
    )


def test_collect_same_members_different_order() -> bool:
    """COLLECT multisets with same members+counts, different order → MATCH."""
    a = [{"names": ["Dakota", "Naya", "Orion"]}]
    b = [{"names": ["Orion", "Dakota", "Naya"]}]
    matched, err = compare_cypher_execution(a, b)
    return _check(
        "T3 COLLECT order differs (same multiset)",
        matched,
        True,
        f"err={err!r} norm={normalize_cypher_result(a)}",
    )


def test_collect_same_multiset_including_intentional_dup() -> bool:
    """Same multiset including a genuine duplicate count → MATCH regardless of order."""
    a = [{"ids": ["Event_SJ_01", "Event_SJ_01", "Event_Dressage_01"]}]
    b = [{"ids": ["Event_Dressage_01", "Event_SJ_01", "Event_SJ_01"]}]
    matched, err = compare_cypher_execution(a, b)
    return _check(
        "T4 COLLECT same multiset (intentional dups, order differs)",
        matched,
        True,
        f"err={err!r}",
    )


def test_genuine_value_difference() -> bool:
    """Same shape, different scalar → MISMATCH."""
    a = [{"count": 50}]
    b = [{"count": 49}]
    matched, err = compare_cypher_execution(a, b)
    return _check(
        "T5 genuine value difference",
        matched,
        False,
        f"err={err!r}",
    )


def test_extra_or_missing_row() -> bool:
    """Extra row in generated → MISMATCH."""
    gold = [{"name": "Dakota"}, {"name": "Naya"}]
    generated = [{"name": "Dakota"}, {"name": "Naya"}, {"name": "Orion"}]
    matched, err = compare_cypher_execution(generated, gold)
    return _check(
        "T6 extra/missing row",
        matched,
        False,
        f"err={err!r}",
    )


def test_scalar_degenerate() -> bool:
    """Bare scalar vs 1×1 dict with same value → MATCH."""
    matched, err = compare_cypher_execution(50, [{"n": 50}])
    return _check(
        "T7 scalar degenerate 1×1",
        matched,
        True,
        f"err={err!r}",
    )


def test_collect_inflated_by_missing_distinct() -> bool:
    """Missing DISTINCT inflating COLLECT membership → MISMATCH (multiset default)."""
    gold = [{"names": ["Dakota", "Emma"]}]
    generated = [{"names": ["Dakota", "Dakota", "Emma"]}]
    matched, err = compare_cypher_execution(generated, gold)
    return _check(
        "T8 COLLECT inflated by missing DISTINCT",
        matched,
        False,
        f"err={err!r} "
        f"normG={normalize_cypher_result(generated)} "
        f"normGold={normalize_cypher_result(gold)}",
    )


def test_cross_row_value_swap() -> bool:
    """Values swapped across rows must MISMATCH — rows are not flattened into one bag.

    Gold:  (Dakota, Selle Français), (Naya, Anglo-Arabe)
    Bad:   (Dakota, Anglo-Arabe), (Naya, Selle Français)
    Same global value multiset, wrong per-row pairing.
    """
    gold = [
        {"name": "Dakota", "race": "Selle Français"},
        {"name": "Naya", "race": "Anglo-Arabe"},
    ]
    swapped = [
        {"name": "Dakota", "race": "Anglo-Arabe"},
        {"name": "Naya", "race": "Selle Français"},
    ]
    matched, err = compare_cypher_execution(swapped, gold)
    # Prove we did not flatten: global value bags would be identical if flattened.
    flat_gold = sorted(
        t for row in gold for t in (str(row["name"]), str(row["race"]))
    )
    flat_swapped = sorted(
        t for row in swapped for t in (str(row["name"]), str(row["race"]))
    )
    return _check(
        "T9 cross-row value swap (whole-row multiset, not flat bag)",
        matched,
        False,
        f"err={err!r} flatBagsEqual={flat_gold == flat_swapped} "
        f"normGold={normalize_cypher_result(gold)} "
        f"normSwapped={normalize_cypher_result(swapped)}",
    )


def test_extra_generated_columns_allowed() -> bool:
    """Gold values ⊆ gen values per row (extra columns) → MATCH."""
    gold = [{"event_id": "Event_SJ_01"}, {"event_id": "Event_Dressage_01"}]
    generated = [
        {"event": "Event_SJ_01", "discipline": "ShowJumping"},
        {"event": "Event_Dressage_01", "discipline": "Dressage"},
    ]
    matched, err = compare_cypher_execution(generated, gold)
    return _check(
        "T10 extra generated columns (gold ⊆ gen per row)",
        matched,
        True,
        f"err={err!r}",
    )


def test_missing_gold_column_still_mismatch() -> bool:
    """Generated missing a gold cell value → MISMATCH (containment is one-way)."""
    gold = [{"event_id": "Event_SJ_01", "rank": 1}]
    generated = [{"event": "Event_SJ_01"}]
    matched, err = compare_cypher_execution(generated, gold)
    return _check(
        "T11 missing gold column value still MISMATCH",
        matched,
        False,
        f"err={err!r}",
    )


def test_wrong_value_with_extra_columns_still_mismatch() -> bool:
    """Extra columns must not hide a wrong gold-required value → MISMATCH."""
    gold = [{"event_id": "Event_SJ_01"}]
    generated = [
        {"event": "Event_WRONG_01", "discipline": "ShowJumping"},
    ]
    matched, err = compare_cypher_execution(generated, gold)
    return _check(
        "T12 wrong gold value despite extra columns still MISMATCH",
        matched,
        False,
        f"err={err!r}",
    )


def test_extra_count_column_alongside_core_values() -> bool:
    """Harmless extra COUNT alongside correct core values → MATCH."""
    gold = [{"intensity": "Élevée"}]
    generated = [
        {
            "intensity": "Élevée",
            "horse_count": 50,
            "horses": ["Dakota", "Naya"],
        }
    ]
    matched, err = compare_cypher_execution(generated, gold)
    return _check(
        "T13 extra COUNT/COLLECT columns with correct core value → MATCH",
        matched,
        True,
        f"err={err!r}",
    )


def main() -> int:
    print("Phase-1 compare_cypher_execution unit tests")
    print("=" * 72)
    results = [
        test_same_facts_different_column_order_dict_keys(),
        test_same_facts_different_column_order_aliases(),
        test_collect_same_members_different_order(),
        test_collect_same_multiset_including_intentional_dup(),
        test_genuine_value_difference(),
        test_extra_or_missing_row(),
        test_scalar_degenerate(),
        test_collect_inflated_by_missing_distinct(),
        test_cross_row_value_swap(),
        test_extra_generated_columns_allowed(),
        test_missing_gold_column_still_mismatch(),
        test_wrong_value_with_extra_columns_still_mismatch(),
        test_extra_count_column_alongside_core_values(),
    ]
    passed = sum(results)
    total = len(results)
    print("=" * 72)
    print(f"{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
