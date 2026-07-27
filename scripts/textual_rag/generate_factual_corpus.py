"""
PHASE 0 — Build a factual textual RAG corpus from live V9 Neo4j.

Read-only against Neo4j. Does NOT modify Neo4j, test_dataset.json,
backend/graph_rag, or backend/tabular_rag.

Pipeline:
  1) Pull structured fact-sheets (JSON) for all 50 horses + 20 events
  2) Render French prose documents from those fact-sheets only
     (deterministic templates — every number/name traces to the sheet)
  3) Spot-check 10 random documents and write a verification report

Output root: data/textual_rag/textual_corpus/
"""
from __future__ import annotations

import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.graph_rag.graph_service import init_graph  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "textual_rag" / "textual_corpus"
FS_HORSES = OUT_ROOT / "fact_sheets" / "horses"
FS_EVENTS = OUT_ROOT / "fact_sheets" / "events"
DOC_HORSES = OUT_ROOT / "documents" / "horses"
DOC_EVENTS = OUT_ROOT / "documents" / "events"

STAGE_FR = {
    "PreparationStage": "préparation",
    "PreCompetitionStage": "pré-compétition",
    "CompetitionStage": "compétition (entraînement)",
    "TransitionStage": "transition / récupération",
}

POSITION_FR = {
    "Withers": "garrot",
    "Sternum": "sternum",
    "CanonOfForelimb": "canon antérieur",
    "CanonOfHindlimb": "canon postérieur",
}

DISCIPLINE_FR = {
    "ShowJumping": "saut d'obstacles",
    "Dressage": "dressage",
    "Cross": "cross / concours complet",
}

ROLE_FR = {
    "Rider": "cavalier",
    "Veterinarian": "vétérinaire",
    "Caretaker": "soigneur",
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "iso_format"):
        try:
            return value.iso_format()
        except Exception:  # noqa: BLE001
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable) + "\n",
        encoding="utf-8",
    )


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", text.strip())
    return cleaned.strip("_") or "unknown"


# ── Step 1: fact extraction ─────────────────────────────────────────────


def pull_horse_fact_sheet(graph, horse_name: str) -> dict[str, Any]:
    identity = graph.query(
        """
        MATCH (h:Horse {hasName: $name})
        RETURN h.id AS id, h.hasName AS hasName, h.hasRace AS hasRace
        """,
        {"name": horse_name},
    )[0]

    stages_raw = graph.query(
        """
        MATCH (h:Horse {hasName: $name})-[:TRAINSIN]->(t)
        OPTIONAL MATCH (t)-[:DEPENDSON]->(e)
        WHERE e:ShowJumping OR e:Dressage OR e:Cross
        WITH h, t, collect(DISTINCT e.id) AS linked_events
        OPTIONAL MATCH (t)-[:INVOLVESACTOR]->(a)
        WITH t, linked_events,
             collect(DISTINCT {
               role: labels(a)[0],
               actor_id: coalesce(a.id, a.hasName)
             }) AS actors
        RETURN t.id AS stage_id,
               labels(t)[0] AS stage_type,
               t.Volume AS volume,
               t.Intensity AS intensity,
               t.Frequency AS frequency,
               linked_events,
               [x IN actors WHERE x.actor_id IS NOT NULL | x] AS actors
        ORDER BY stage_type, stage_id
        """,
        {"name": horse_name},
    )

    sensors_raw = graph.query(
        """
        MATCH (s:InertialSensors)-[:ISATTACHEDTO]->(h:Horse {hasName: $name})
        OPTIONAL MATCH (s)-[:ISUSEDFOR]->(o)
        RETURN s.id AS sensor_id,
               s.hasSensorID AS sensor_code,
               [x IN labels(s) WHERE x <> 'InertialSensors'][0] AS position,
               s.hasSensorTime AS sample_rate,
               s.hasSensorOffset AS offset,
               s.hasFormat AS format,
               s.hasFileSize AS file_size,
               o.id AS objective
        ORDER BY position, sensor_id
        """,
        {"name": horse_name},
    )

    participations_raw = graph.query(
        """
        MATCH (h:Horse {hasName: $name})<-[:HASHORSE]-(p:EventParticipation)
              <-[:HASPARTICIPATION]-(e)
        WHERE e:ShowJumping OR e:Dressage OR e:Cross
        OPTIONAL MATCH (p)-[:HASRIDER]->(r:Rider)
        RETURN e.id AS event_id,
               labels(e)[0] AS discipline,
               e.category AS category,
               toString(e.eventDate) AS event_date,
               e.eventLocation AS location,
               p.rank AS rank,
               p.status AS status,
               r.id AS rider_id
        ORDER BY event_date, event_id
        """,
        {"name": horse_name},
    )

    associated_riders = graph.query(
        """
        MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse {hasName: $name})
        RETURN r.id AS rider_id
        ORDER BY rider_id
        """,
        {"name": horse_name},
    )

    return {
        "doc_type": "horse_fact_sheet",
        "source": "live_neo4j_v9",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "horse": {
            "id": identity["id"],
            "hasName": identity["hasName"],
            "hasRace": identity["hasRace"],
        },
        "associated_riders": [row["rider_id"] for row in associated_riders],
        "training_stages": [
            {
                "stage_id": row["stage_id"],
                "stage_type": row["stage_type"],
                "volume": row["volume"],
                "intensity": row["intensity"],
                "frequency": row["frequency"],
                "depends_on_events": row["linked_events"] or [],
                "actors": row["actors"] or [],
            }
            for row in stages_raw
        ],
        "sensors": [
            {
                "sensor_id": row["sensor_id"],
                "sensor_code": row["sensor_code"],
                "position": row["position"],
                "sample_rate": row["sample_rate"],
                "offset": row["offset"],
                "format": row["format"],
                "file_size": row["file_size"],
                "objective": row["objective"],
            }
            for row in sensors_raw
        ],
        "event_participations": [
            {
                "event_id": row["event_id"],
                "discipline": row["discipline"],
                "category": row["category"],
                "event_date": row["event_date"],
                "location": row["location"],
                "rank": row["rank"],
                "status": row["status"],
                "rider_id": row["rider_id"],
            }
            for row in participations_raw
        ],
    }


def pull_event_fact_sheet(graph, event_id: str) -> dict[str, Any]:
    identity = graph.query(
        """
        MATCH (e {id: $eid})
        WHERE e:ShowJumping OR e:Dressage OR e:Cross
        OPTIONAL MATCH (e)-[:INSEASON]->(season)
        RETURN e.id AS id,
               labels(e)[0] AS discipline,
               e.category AS category,
               toString(e.eventDate) AS event_date,
               e.eventLocation AS location,
               season.seasonName AS season
        """,
        {"eid": event_id},
    )[0]

    participants = graph.query(
        """
        MATCH (e {id: $eid})-[:HASPARTICIPATION]->(p:EventParticipation)
        OPTIONAL MATCH (p)-[:HASHORSE]->(h:Horse)
        OPTIONAL MATCH (p)-[:HASRIDER]->(r:Rider)
        RETURN p.id AS participation_id,
               h.hasName AS horse_name,
               h.id AS horse_id,
               h.hasRace AS horse_race,
               r.id AS rider_id,
               p.rank AS rank,
               p.status AS status
        ORDER BY p.rank, h.hasName
        """,
        {"eid": event_id},
    )

    linked_stages = graph.query(
        """
        MATCH (t)-[:DEPENDSON]->(e {id: $eid})
        MATCH (h:Horse)-[:TRAINSIN]->(t)
        RETURN labels(t)[0] AS stage_type,
               count(DISTINCT t) AS stage_count,
               count(DISTINCT h) AS horse_count
        ORDER BY stage_type
        """,
        {"eid": event_id},
    )

    return {
        "doc_type": "event_fact_sheet",
        "source": "live_neo4j_v9",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "event": {
            "id": identity["id"],
            "discipline": identity["discipline"],
            "category": identity["category"],
            "event_date": identity["event_date"],
            "location": identity["location"],
            "season": identity["season"],
        },
        "participants": [
            {
                "participation_id": row["participation_id"],
                "horse_name": row["horse_name"],
                "horse_id": row["horse_id"],
                "horse_race": row["horse_race"],
                "rider_id": row["rider_id"],
                "rank": row["rank"],
                "status": row["status"],
            }
            for row in participants
        ],
        "linked_training_summary": [
            {
                "stage_type": row["stage_type"],
                "stage_count": row["stage_count"],
                "horse_count": row["horse_count"],
            }
            for row in linked_stages
        ],
    }


def extract_all_fact_sheets(graph) -> tuple[list[dict], list[dict]]:
    horse_names = [
        row["name"]
        for row in graph.query(
            "MATCH (h:Horse) RETURN h.hasName AS name ORDER BY h.hasName"
        )
    ]
    event_ids = [
        row["id"]
        for row in graph.query(
            """
            MATCH (e)
            WHERE e:ShowJumping OR e:Dressage OR e:Cross
            RETURN e.id AS id
            ORDER BY e.eventDate, e.id
            """
        )
    ]
    assert len(horse_names) == 50, f"expected 50 horses, got {len(horse_names)}"
    assert len(event_ids) == 20, f"expected 20 events, got {len(event_ids)}"

    horses = []
    for name in horse_names:
        sheet = pull_horse_fact_sheet(graph, name)
        path = FS_HORSES / f"{_slug(name)}.json"
        _write_json(path, sheet)
        sheet["_path"] = str(path.relative_to(REPO_ROOT))
        horses.append(sheet)
        print(f"  horse fact-sheet: {name}")

    events = []
    for eid in event_ids:
        sheet = pull_event_fact_sheet(graph, eid)
        path = FS_EVENTS / f"{_slug(eid)}.json"
        _write_json(path, sheet)
        sheet["_path"] = str(path.relative_to(REPO_ROOT))
        events.append(sheet)
        print(f"  event fact-sheet: {eid}")

    return horses, events


# ── Step 2: prose from fact-sheet only ───────────────────────────────────


def _fmt_actors(actors: list[dict]) -> str:
    if not actors:
        return "aucun acteur renseigné dans le graphe"
    parts = []
    for a in actors:
        role = ROLE_FR.get(a.get("role") or "", a.get("role") or "?")
        parts.append(f"{role} {a.get('actor_id')}")
    return ", ".join(parts)


def render_horse_document(sheet: dict) -> str:
    h = sheet["horse"]
    name = h["hasName"]
    race = h["hasRace"]
    lines: list[str] = []
    lines.append(f"Rapport d'entraînement — {name}")
    lines.append("=" * (len(lines[0]) + 4))
    lines.append("")
    lines.append(
        f"Ce rapport synthétise exclusivement les faits enregistrés dans le "
        f"graphe de connaissances pour le cheval {name} (identifiant {h['id']}), "
        f"race {race}."
    )
    lines.append("")

    riders = sheet.get("associated_riders") or []
    if riders:
        lines.append(
            "Cavaliers associés (relation ASSOCIATEDWITH) : "
            + ", ".join(riders)
            + "."
        )
    else:
        lines.append(
            "Aucun cavalier n'est lié à ce cheval via ASSOCIATEDWITH dans le graphe."
        )
    lines.append("")

    stages = sheet.get("training_stages") or []
    lines.append(f"Programme d'entraînement ({len(stages)} étape(s) enregistrée(s))")
    lines.append("-" * 40)
    if not stages:
        lines.append("Aucune étape d'entraînement n'est liée à ce cheval.")
    else:
        by_type: dict[str, list] = {}
        for st in stages:
            by_type.setdefault(st["stage_type"], []).append(st)
        for stype, group in by_type.items():
            label = STAGE_FR.get(stype, stype)
            lines.append("")
            lines.append(f"Phase {label} ({stype}) — {len(group)} séance(s) :")
            for st in group:
                deps = st.get("depends_on_events") or []
                dep_txt = (
                    ", ".join(deps) if deps else "aucun événement DEPENDSON"
                )
                lines.append(
                    f"  • {st['stage_id']} : volume {st['volume']}, "
                    f"intensité {st['intensity']}, fréquence {st['frequency']} ; "
                    f"acteurs : {_fmt_actors(st.get('actors') or [])} ; "
                    f"événements liés : {dep_txt}."
                )
    lines.append("")

    sensors = sheet.get("sensors") or []
    lines.append(f"Capteurs inertiels ({len(sensors)} capteur(s) attaché(s))")
    lines.append("-" * 40)
    if not sensors:
        lines.append("Aucun capteur InertialSensors n'est attaché à ce cheval.")
    else:
        for s in sensors:
            pos = POSITION_FR.get(s["position"] or "", s["position"] or "?")
            lines.append(
                f"  • {s['sensor_id']} en position {pos} ({s['position']}), "
                f"fréquence d'échantillonnage {s['sample_rate']}, "
                f"offset {s['offset']}, format {s['format']}, "
                f"taille de fichier {s['file_size']}, "
                f"objectif expérimental {s['objective']}."
            )
    lines.append("")

    parts = sheet.get("event_participations") or []
    lines.append(f"Participations en compétition ({len(parts)} résultat(s))")
    lines.append("-" * 40)
    if not parts:
        lines.append(
            "Aucune participation EventParticipation n'est enregistrée pour ce cheval."
        )
    else:
        for p in parts:
            disc = DISCIPLINE_FR.get(p["discipline"] or "", p["discipline"] or "?")
            lines.append(
                f"  • {p['event_id']} ({disc} / {p['discipline']}, "
                f"catégorie « {p['category']} ») "
                f"le {p['event_date']} à {p['location']} : "
                f"rang {p['rank']}, statut « {p['status']} », "
                f"cavalier {p['rider_id']}."
            )
    lines.append("")
    lines.append(
        "Note méthodologique : aucun rang, volume, intensité, fréquence, "
        "capteur ou participant n'a été inventé ; toutes les valeurs ci-dessus "
        "proviennent du fact-sheet extrait de Neo4j V9."
    )
    lines.append("")
    return "\n".join(lines)


def render_event_document(sheet: dict) -> str:
    e = sheet["event"]
    disc = DISCIPLINE_FR.get(e["discipline"] or "", e["discipline"] or "?")
    lines: list[str] = []
    lines.append(f"Compte rendu d'épreuve — {e['id']}")
    lines.append("=" * (len(lines[0]) + 4))
    lines.append("")
    lines.append(
        f"Épreuve de {disc} ({e['discipline']}), catégorie {e['category']}, "
        f"organisée le {e['event_date']} à {e['location']}, "
        f"rattachée à la saison « {e['season']} »."
    )
    lines.append("")

    participants = sheet.get("participants") or []
    lines.append(f"Classement et participants ({len(participants)} engagement(s))")
    lines.append("-" * 40)
    if not participants:
        lines.append("Aucun engagement EventParticipation pour cette épreuve.")
    else:
        for p in participants:
            lines.append(
                f"  • Rang {p['rank']} — cheval {p['horse_name']} "
                f"({p['horse_id']}, race « {p['horse_race']} »), "
                f"cavalier {p['rider_id']}, statut « {p['status']} »."
            )
    lines.append("")

    linked = sheet.get("linked_training_summary") or []
    lines.append("Entraînements liés (étapes DEPENDSON vers cette épreuve)")
    lines.append("-" * 40)
    if not linked:
        lines.append(
            "Aucune étape d'entraînement ne pointe vers cette épreuve via DEPENDSON."
        )
    else:
        for row in linked:
            label = STAGE_FR.get(row["stage_type"], row["stage_type"])
            lines.append(
                f"  • {label} ({row['stage_type']}) : "
                f"{row['stage_count']} étape(s) pour "
                f"{row['horse_count']} cheval(aux)."
            )
    lines.append("")
    lines.append(
        "Note méthodologique : discipline, date, lieu, catégorie, rangs et "
        "cavaliers sont exclusivement issus du fact-sheet Neo4j V9."
    )
    lines.append("")
    return "\n".join(lines)


def render_all_documents(
    horse_sheets: list[dict], event_sheets: list[dict]
) -> tuple[list[dict], list[dict]]:
    horse_docs = []
    for sheet in horse_sheets:
        name = sheet["horse"]["hasName"]
        text = render_horse_document(sheet)
        path = DOC_HORSES / f"{_slug(name)}.txt"
        path.write_text(text, encoding="utf-8")
        meta = {
            "doc_id": f"horse_{_slug(name)}",
            "doc_type": "horse_report",
            "subject": name,
            "path": str(path.relative_to(REPO_ROOT)),
            "fact_sheet_path": sheet["_path"],
            "char_count": len(text),
        }
        horse_docs.append(meta)
        print(f"  horse document: {name}")

    event_docs = []
    for sheet in event_sheets:
        eid = sheet["event"]["id"]
        text = render_event_document(sheet)
        path = DOC_EVENTS / f"{_slug(eid)}.txt"
        path.write_text(text, encoding="utf-8")
        meta = {
            "doc_id": f"event_{_slug(eid)}",
            "doc_type": "event_report",
            "subject": eid,
            "path": str(path.relative_to(REPO_ROOT)),
            "fact_sheet_path": sheet["_path"],
            "char_count": len(text),
        }
        event_docs.append(meta)
        print(f"  event document: {eid}")

    return horse_docs, event_docs


# ── Step 3: verification ─────────────────────────────────────────────────


def _collect_atomic_facts(sheet: dict) -> list[str]:
    """Flatten fact-sheet into comparable string tokens (specific claims)."""
    facts: list[str] = []
    if sheet["doc_type"] == "horse_fact_sheet":
        h = sheet["horse"]
        facts.extend([h["id"], h["hasName"], h["hasRace"]])
        facts.extend(sheet.get("associated_riders") or [])
        for st in sheet.get("training_stages") or []:
            facts.extend(
                [
                    st["stage_id"],
                    st["stage_type"],
                    str(st["volume"]),
                    str(st["intensity"]),
                    str(st["frequency"]),
                ]
            )
            facts.extend(st.get("depends_on_events") or [])
            for a in st.get("actors") or []:
                if a.get("actor_id"):
                    facts.append(a["actor_id"])
        for s in sheet.get("sensors") or []:
            facts.extend(
                [
                    s["sensor_id"],
                    str(s["position"]),
                    str(s["sample_rate"]),
                    str(s["offset"]),
                    str(s["format"]),
                    str(s["file_size"]),
                    str(s["objective"]),
                ]
            )
        for p in sheet.get("event_participations") or []:
            facts.extend(
                [
                    p["event_id"],
                    str(p["discipline"]),
                    str(p["category"]),
                    str(p["event_date"]),
                    str(p["location"]),
                    str(p["rank"]),
                    str(p["status"]),
                    str(p["rider_id"]),
                ]
            )
    else:
        e = sheet["event"]
        facts.extend(
            [
                e["id"],
                str(e["discipline"]),
                str(e["category"]),
                str(e["event_date"]),
                str(e["location"]),
                str(e["season"]),
            ]
        )
        for p in sheet.get("participants") or []:
            facts.extend(
                [
                    str(p["horse_name"]),
                    str(p["horse_id"]),
                    str(p["horse_race"]),
                    str(p["rider_id"]),
                    str(p["rank"]),
                    str(p["status"]),
                ]
            )
        for row in sheet.get("linked_training_summary") or []:
            facts.extend(
                [
                    str(row["stage_type"]),
                    str(row["stage_count"]),
                    str(row["horse_count"]),
                ]
            )
    # unique non-empty
    out = []
    seen = set()
    for f in facts:
        if f is None:
            continue
        token = str(f).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def verify_document(doc_meta: dict) -> dict[str, Any]:
    sheet = json.loads((REPO_ROOT / doc_meta["fact_sheet_path"]).read_text(encoding="utf-8"))
    prose = (REPO_ROOT / doc_meta["path"]).read_text(encoding="utf-8")
    facts = _collect_atomic_facts(sheet)
    fact_set = set(facts)

    # Soft coverage: which sheet tokens are absent from prose (informational).
    # French labels may replace English enums; those show up here but are not fails.
    missing_in_prose = [f for f in facts if f not in prose]

    invented: list[str] = []

    # Graph identifiers mentioned in prose must exist on the sheet.
    id_like = re.findall(
        r"\b(?:Horse_[A-Za-z0-9_]+|Rider_[A-Za-z0-9_]+|Event_[A-Za-z0-9_]+|"
        r"Training_[A-Za-z0-9_]+|IMU_[A-Za-z0-9_]+|Vet_[A-Za-z0-9_]+|"
        r"Caretaker_[A-Za-z0-9_]+|GaitClassif_[A-Za-z0-9_]+|FatigueDetection)\b",
        prose,
    )
    for token in id_like:
        if token not in fact_set:
            invented.append(f"id:{token}")

    for token in re.findall(r"\b\d+Hz\b", prose):
        if token not in fact_set:
            invented.append(f"rate:{token}")

    for token in re.findall(r"\b\d+min\b", prose):
        if token not in fact_set:
            invented.append(f"volume:{token}")

    for rank in re.findall(r"\brang\s+(\d+)\b", prose, flags=re.IGNORECASE):
        if rank not in fact_set:
            invented.append(f"rank:{rank}")

    # Quoted field values (« ... ») must be on the sheet.
    for quoted in re.findall(r"«\s*([^»]+?)\s*»", prose):
        if quoted not in fact_set:
            invented.append(f"quoted:{quoted}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    invented_unique = []
    for item in invented:
        if item not in seen:
            seen.add(item)
            invented_unique.append(item)

    ok = len(invented_unique) == 0
    return {
        "doc_id": doc_meta["doc_id"],
        "doc_type": doc_meta["doc_type"],
        "subject": doc_meta["subject"],
        "path": doc_meta["path"],
        "fact_sheet_path": doc_meta["fact_sheet_path"],
        "atomic_facts_count": len(facts),
        "facts_missing_from_prose": missing_in_prose,
        "facts_missing_count": len(missing_in_prose),
        "invented_or_untraced_claims": invented_unique,
        "invented_count": len(invented_unique),
        "verdict": "PASS" if ok else "FAIL",
        "prose": prose,
        "fact_sheet": sheet,
    }


def spot_check(
    horse_docs: list[dict], event_docs: list[dict], seed: int = 20260727
) -> list[dict]:
    rng = random.Random(seed)
    # 7 horses + 3 events for spread
    horse_sample = rng.sample(horse_docs, 7)
    event_sample = rng.sample(event_docs, 3)
    selected = horse_sample + event_sample
    rng.shuffle(selected)
    results = [verify_document(doc) for doc in selected]
    return results


def write_report(
    horse_sheets: list[dict],
    event_sheets: list[dict],
    horse_docs: list[dict],
    event_docs: list[dict],
    checks: list[dict],
) -> Path:
    # pick 3 full examples: 2 horses + 1 event from the spot-check set if possible
    examples = []
    for c in checks:
        if c["doc_type"] == "horse_report" and sum(1 for e in examples if e["doc_type"] == "horse_report") < 2:
            examples.append(c)
        elif c["doc_type"] == "event_report" and sum(1 for e in examples if e["doc_type"] == "event_report") < 1:
            examples.append(c)
        if len(examples) == 3:
            break
    while len(examples) < 3:
        for c in checks:
            if c not in examples:
                examples.append(c)
            if len(examples) == 3:
                break

    lines: list[str] = []
    lines.append("# PHASE 0 — Factual textual corpus report")
    lines.append("")
    lines.append(f"**Generated at:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Source:** live Neo4j V9 (read-only)")
    lines.append(
        f"**Counts:** {len(horse_docs)} horse docs + {len(event_docs)} event docs "
        f"= {len(horse_docs) + len(event_docs)}"
    )
    lines.append("")
    lines.append("## Fact-sheet format")
    lines.append("")
    lines.append("### Horse fact-sheet (`fact_sheets/horses/<Name>.json`)")
    lines.append("```")
    lines.append(
        json.dumps(
            {
                "doc_type": "horse_fact_sheet",
                "source": "live_neo4j_v9",
                "horse": {"id": "...", "hasName": "...", "hasRace": "..."},
                "associated_riders": ["Rider_..."],
                "training_stages": [
                    {
                        "stage_id": "...",
                        "stage_type": "PreparationStage|PreCompetitionStage|CompetitionStage|TransitionStage",
                        "volume": "...",
                        "intensity": "...",
                        "frequency": "...",
                        "depends_on_events": ["Event_..."],
                        "actors": [{"role": "Rider|Veterinarian|Caretaker", "actor_id": "..."}],
                    }
                ],
                "sensors": [
                    {
                        "sensor_id": "...",
                        "position": "Withers|Sternum|CanonOfForelimb|CanonOfHindlimb",
                        "sample_rate": "...Hz",
                        "offset": "...",
                        "format": "...",
                        "file_size": "...",
                        "objective": "...",
                    }
                ],
                "event_participations": [
                    {
                        "event_id": "...",
                        "discipline": "...",
                        "category": "...",
                        "event_date": "YYYY-MM-DD",
                        "location": "...",
                        "rank": 0,
                        "status": "...",
                        "rider_id": "...",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("### Event fact-sheet (`fact_sheets/events/<EventId>.json`)")
    lines.append("```")
    lines.append(
        json.dumps(
            {
                "doc_type": "event_fact_sheet",
                "event": {
                    "id": "...",
                    "discipline": "ShowJumping|Dressage|Cross",
                    "category": "...",
                    "event_date": "YYYY-MM-DD",
                    "location": "...",
                    "season": "...",
                },
                "participants": [
                    {
                        "horse_name": "...",
                        "horse_id": "...",
                        "horse_race": "...",
                        "rider_id": "...",
                        "rank": 0,
                        "status": "...",
                    }
                ],
                "linked_training_summary": [
                    {
                        "stage_type": "...",
                        "stage_count": 0,
                        "horse_count": 0,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("## Generation method")
    lines.append("")
    lines.append(
        "Prose is rendered by deterministic French templates that interpolate "
        "only fields present on the fact-sheet. No LLM is used, so ranks, "
        "sensor rates, volumes, and participants cannot be invented."
    )
    lines.append("")
    lines.append("## Three full examples (prose ↔ fact-sheet)")
    lines.append("")
    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i} — {ex['doc_id']} ({ex['verdict']})")
        lines.append("")
        lines.append("#### Prose")
        lines.append("```")
        lines.append(ex["prose"].rstrip())
        lines.append("```")
        lines.append("")
        lines.append("#### Source fact-sheet")
        lines.append("```json")
        # drop heavy nested path keys if any
        fs = {k: v for k, v in ex["fact_sheet"].items() if not k.startswith("_")}
        lines.append(json.dumps(fs, ensure_ascii=False, indent=2, default=_jsonable))
        lines.append("```")
        lines.append("")

    lines.append("## Spot-check verification (10 documents)")
    lines.append("")
    lines.append("| # | doc_id | type | subject | invented | facts missing from prose* | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, c in enumerate(checks, 1):
        lines.append(
            f"| {i} | {c['doc_id']} | {c['doc_type']} | {c['subject']} | "
            f"{c['invented_count']} | {c['facts_missing_count']} | **{c['verdict']}** |"
        )
    lines.append("")
    lines.append(
        "\\* « facts missing from prose » lists atomic fact-sheet tokens not "
        "literally present in the document text (e.g. horse URI-less internal "
        "ids already covered by hasName, or stage_type English label when the "
        "French label is used). Invented/untraced claims are the hard fail signal."
    )
    lines.append("")
    for c in checks:
        if c["invented_count"]:
            lines.append(
                f"- **FLAG {c['doc_id']}**: untraced claims = {c['invented_or_untraced_claims']}"
            )
        else:
            lines.append(
                f"- {c['doc_id']}: no untraced numeric/id claims "
                f"({c['atomic_facts_count']} atomic facts in sheet)."
            )
    lines.append("")
    pass_n = sum(1 for c in checks if c["verdict"] == "PASS")
    lines.append(f"**Spot-check summary:** {pass_n}/{len(checks)} PASS")
    lines.append("")

    report_path = OUT_ROOT / "PHASE0_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # machine-readable verification without full prose duplication for all
    compact = []
    for c in checks:
        compact.append(
            {
                "doc_id": c["doc_id"],
                "doc_type": c["doc_type"],
                "subject": c["subject"],
                "path": c["path"],
                "fact_sheet_path": c["fact_sheet_path"],
                "atomic_facts_count": c["atomic_facts_count"],
                "facts_missing_from_prose": c["facts_missing_from_prose"],
                "invented_or_untraced_claims": c["invented_or_untraced_claims"],
                "verdict": c["verdict"],
            }
        )
    _write_json(OUT_ROOT / "verification_spotcheck.json", compact)
    return report_path


def main() -> int:
    for d in (FS_HORSES, FS_EVENTS, DOC_HORSES, DOC_EVENTS):
        d.mkdir(parents=True, exist_ok=True)

    print("Connecting to live Neo4j...")
    graph = init_graph()
    nodes = graph.query("MATCH (n) RETURN count(n) AS n")[0]["n"]
    print(f"Connected. Node count = {nodes}")

    print("\nSTEP 1 — Extract fact-sheets...")
    horse_sheets, event_sheets = extract_all_fact_sheets(graph)

    print("\nSTEP 2 — Render prose documents from fact-sheets only...")
    horse_docs, event_docs = render_all_documents(horse_sheets, event_sheets)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "live_neo4j_v9",
        "horse_documents": len(horse_docs),
        "event_documents": len(event_docs),
        "total_documents": len(horse_docs) + len(event_docs),
        "horses": horse_docs,
        "events": event_docs,
        "generation": "deterministic_french_templates_from_fact_sheets",
    }
    _write_json(OUT_ROOT / "manifest.json", manifest)

    print("\nSTEP 3 — Spot-check 10 documents...")
    checks = spot_check(horse_docs, event_docs)
    report_path = write_report(
        horse_sheets, event_sheets, horse_docs, event_docs, checks
    )

    pass_n = sum(1 for c in checks if c["verdict"] == "PASS")
    print(f"\nDone. Documents: {manifest['total_documents']}")
    print(f"Spot-check: {pass_n}/{len(checks)} PASS")
    print(f"Report: {report_path}")
    return 0 if pass_n == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
