"""
Textual RAG service: retrieve from Chroma + answer with GPT-4o-mini.

Usage (from repo root, after indexing):
    python backend/textual_rag/textual_rag_service.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import chromadb
from openai import OpenAI

from backend.config import OPENAI_API_KEY

CHROMA_DIR = REPO_ROOT / "data" / "textual_rag" / "chroma_db"
COLLECTION_NAME = "equestrian_textual_corpus"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

DEFAULT_N_RESULTS = 15
VERIFICATION_N_RESULTS = 5
MAX_FORCED_ENTITIES = 5

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans le domaine équestre.
Tu dois répondre UNIQUEMENT à partir du contexte fourni ci-dessous.
Règles strictes :
- Réponds en français.
- N'utilise aucune connaissance externe.
- N'invente aucun fait, chiffre, nom, rang, capteur ou événement.
- Si l'information demandée n'apparaît pas clairement dans le contexte,
  dis-le explicitement (par exemple : « L'information n'est pas présente
  dans les documents fournis. »).
"""

_CORPUS_IDS_CACHE: list[str] | None = None


def _get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Load it via .env / backend.config."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def _get_collection():
    if not CHROMA_DIR.is_dir():
        raise RuntimeError(
            f"Chroma directory not found: {CHROMA_DIR}. "
            "Run scripts/textual_rag/index_corpus.py first."
        )
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found at {CHROMA_DIR}. "
            "Run scripts/textual_rag/index_corpus.py first."
        ) from exc


def _corpus_doc_ids(collection) -> list[str]:
    """Cached Chroma document ids (horse names / event ids)."""
    global _CORPUS_IDS_CACHE
    if _CORPUS_IDS_CACHE is None:
        payload = collection.get(include=[])
        _CORPUS_IDS_CACHE = list(payload.get("ids") or [])
    return _CORPUS_IDS_CACHE


def _embed_query(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _normalize_question(question: str) -> str:
    q = (question or "").lower()
    return q.replace("’", "'").replace("`", "'")


def _build_user_prompt(question: str, documents: list[str], filenames: list[str]) -> str:
    parts = ["Contexte (documents récupérés) :", ""]
    for i, (filename, doc) in enumerate(zip(filenames, documents), start=1):
        parts.append(f"--- Document {i}: {filename} ---")
        parts.append(doc.strip())
        parts.append("")
    parts.append("Question :")
    parts.append(question.strip())
    parts.append("")
    parts.append(
        "Réponds uniquement à partir du contexte ci-dessus, en français."
    )
    return "\n".join(parts)


def _is_comparative_question(question: str) -> bool:
    """Lightweight keyword/phrase gate for pairwise comparison questions."""
    q = _normalize_question(question)

    phrase_cues = (
        "comparer",
        "compare",
        "comparaison",
        "comparison",
        "en comparaison",
        "différence",
        "differences",
        "difference",
        "différences",
        "différent de",
        "différente de",
        "différents de",
        "différentes de",
        "versus",
        " vs ",
        " vs.",
        "par rapport à",
        "par rapport aux",
        "contrairement à",
        "contrairement aux",
        "à la différence",
        "au contraire de",
    )
    if any(cue in q for cue in phrase_cues):
        return True

    if re.search(r"\b(?:plus|moins)\b.{0,40}\bque\b", q):
        return True

    return False


def _is_verification_question(question: str) -> bool:
    """Gate for verification / data-quality style questions (Proposal C).

    Tuned from eval failures where wider top-K added noise; avoids bare
    'combien' / generic aggregation phrasing.
    """
    q = _normalize_question(question)
    cues = (
        "calibr",
        "cohérent",
        "coherent",
        "vérifi",
        "verifi",
        "sans résultat",
        "aucun résultat",
        "pas de résultat",
        "résultat officiel",
        "sortent du schéma",
        "hors schéma",
        "identifiants d'entraînement",
        "montés par plusieurs",
        "monte par plusieurs",
        "plusieurs cavaliers différents",
        "même nombre de compétitions",
        "participent-ils au même",
        "engagé dans une compétition sans",
        "engagés mais",
        "engagé mais",
        "ressemble-t-il",
        "correspond-il toujours",
        "correspond-elle toujours",
    )
    return any(cue in q for cue in cues)


def _is_aggregation_question(question: str) -> bool:
    """Gate for count / distribution / 'most common' style questions."""
    q = _normalize_question(question)
    cues = (
        "combien",
        "répart",
        "repart",
        "distribution",
        "la plus courante",
        "le plus courant",
        "les plus courants",
        "représentées par un seul",
        "representees par un seul",
        "quel nombre",
        "nombre de",
    )
    if any(cue in q for cue in cues):
        return True
    # "Y a-t-il des races rares..." style herd inventories
    if "y a-t-il des races" in q or "y a-t-il des chevaux" in q:
        return True
    return False


def _preferred_entity_type(question: str) -> str | None:
    """Bias retrieval toward horse vs event docs when question topic is clear.

    Eval evidence: many training/race/sensor questions retrieved mostly Event_*
    docs; event-count questions retrieved mostly horse docs. Preferential
    merge (not exclusive filter) keeps recall while fixing type mismatch.
    """
    q = _normalize_question(question)

    horse_cues = (
        "race de cheval",
        "races",
        "capteur",
        "capteurs",
        "imu",
        "échantillonnage",
        "echantillonnage",
        "entraînement",
        "entrainement",
        "préparation",
        "preparation",
        "pré-compétition",
        "pre-competition",
        "récupération",
        "recuperation",
        "transition",
        "séance",
        "seance",
        "volume",
        "intensité",
        "intensite",
        "programme d'entraînement",
        "programme d'entrainement",
        "soigneur",
        "vétérinaire",
        "veterinaire",
        "cavaliers différents",
        "cavaliers differents",
        "associés",
        "associes",
    )
    event_cues = (
        "événement sportif",
        "evenement sportif",
        "événements sportifs",
        "evenements sportifs",
        "compétition",
        "competition",
        "compétitions",
        "competitions",
        "classement",
        "catégorie",
        "categorie",
        "résultat officiel",
        "resultat officiel",
        "engagé",
        "engage",
        "engagés",
        "engages",
        "saison 2026",
        "période de la saison",
        "periode de la saison",
        "discipline",
        "lieu",
        "saumur",
        "nantes",
        "bordeaux",
    )

    horse_hit = any(cue in q for cue in horse_cues)
    event_hit = any(cue in q for cue in event_cues)

    # Strong overrides when the question is clearly about corpus-wide event count
    # or competition results rather than training content.
    if any(
        cue in q
        for cue in (
            "combien d'événements",
            "combien d'evenements",
            "événements sportifs sont enregistrés",
            "evenements sportifs sont enregistres",
            "sans résultat officiel",
            "sans resultat officiel",
            "chevaux engagés",
            "chevaux engages",
        )
    ):
        return "event"

    if any(
        cue in q
        for cue in (
            "race de cheval la plus courante",
            "races rares",
            "durée des séances",
            "duree des seances",
            "phase de récupération",
            "phase de recuperation",
            "phase de préparation",
            "phase de preparation",
            "pré-compétition les plus",
            "préparation les plus",
            "capteurs imu",
            "montés par plusieurs cavaliers",
            "montes par plusieurs cavaliers",
        )
    ):
        return "horse"

    if horse_hit and not event_hit:
        return "horse"
    if event_hit and not horse_hit:
        return "event"
    return None


def _named_entities_in_question(question: str, doc_ids: list[str]) -> list[str]:
    """Return corpus doc ids explicitly mentioned in the question (Proposal B)."""
    q = question or ""
    hits: list[str] = []
    # Longer ids first so Event_SJ_2026_01 wins over shorter prefixes if any
    for doc_id in sorted(doc_ids, key=len, reverse=True):
        if len(doc_id) < 3:
            continue
        pattern = r"\b" + re.escape(doc_id) + r"\b"
        if re.search(pattern, q, flags=re.IGNORECASE):
            hits.append(doc_id)
        if len(hits) >= MAX_FORCED_ENTITIES:
            break
    return hits


def _query_by_embedding(
    collection,
    query_embedding: list[float],
    n_results: int,
    where: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], list[dict]]:
    """Run a Chroma similarity query; return (ids, documents, metadatas)."""
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": max(1, n_results),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    return ids, documents, metadatas


def _fetch_docs_by_ids(
    collection, doc_ids: list[str]
) -> tuple[list[str], list[str], list[dict]]:
    if not doc_ids:
        return [], [], []
    payload = collection.get(ids=doc_ids, include=["documents", "metadatas"])
    ids = list(payload.get("ids") or [])
    documents = list(payload.get("documents") or [])
    metadatas = list(payload.get("metadatas") or [])
    return ids, documents, metadatas


# Literal taxonomy strings as they appear in data/textual_rag/textual_corpus documents
# (and English stage/sensor labels used in parentheses in those docs).
# Sourced by scanning horses/*.txt and events/*.txt — not guessed.
TAXONOMY_VALUES: tuple[str, ...] = (
    # Event categories (from "catégorie …" / « … » in event reports)
    "Amateur 1",
    "Amateur 2",
    "Club Elite",
    "Pro Elite",
    # Disciplines — English labels in parentheses + French lead phrases
    "ShowJumping",
    "Dressage",
    "Cross",
    "saut d'obstacles",
    "cross / concours complet",
    "dressage",
    "concours complet",
    # Training intensities (from "intensité …" on horse reports)
    "Élevée",
    "Modérée",
    "Faible",
    "Pic",
    # Sensor positions — English secondary labels + French prose forms
    "Withers",
    "Sternum",
    "CanonOfForelimb",
    "CanonOfHindlimb",
    "garrot",
    "sternum",
    "canon antérieur",
    "canon postérieur",
    # Training stage types — English labels in parentheses
    "PreparationStage",
    "PreCompetitionStage",
    "CompetitionStage",
    "TransitionStage",
)


def _taxonomy_values_in_question(question: str) -> list[str]:
    """Return known taxonomy literals found in the question (longest first)."""
    q = (question or "").casefold()
    hits: list[str] = []
    for value in sorted(TAXONOMY_VALUES, key=len, reverse=True):
        if value.casefold() in q:
            hits.append(value)
    return hits


def _doc_contains_any_value(text: str, values: list[str]) -> bool:
    if not values or not text:
        return False
    hay = text.casefold()
    return any(v.casefold() in hay for v in values)


def _merge_retrieval_results(
    *,
    forced: tuple[list[str], list[str], list[dict]],
    preferred: tuple[list[str], list[str], list[dict]],
    general: tuple[list[str], list[str], list[dict]],
    limit: int,
    anchor_values: list[str] | None = None,
) -> tuple[list[str], list[str], list[str], list[dict]]:
    """Merge forced → preferred → general, dedupe by id, cap at limit.

    Optional value-anchored re-rank: after building the candidate universe
    from the three pools, documents whose text contains a matched taxonomy
    value are moved ahead of non-matching peers (forced ids stay first).
    If no anchor values are provided, or no candidate contains them, order
    is unchanged from the classic merge.

    Returns (ids, documents, filenames, metadatas).
    """
    ordered_ids: list[str] = []
    doc_by_id: dict[str, str] = {}
    meta_by_id: dict[str, dict] = {}
    forced_id_set = set(forced[0] if forced else [])

    for ids, documents, metadatas in (forced, preferred, general):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            if doc_id in doc_by_id:
                continue
            ordered_ids.append(doc_id)
            doc_by_id[doc_id] = doc
            meta_by_id[doc_id] = meta or {}

    if anchor_values:
        anchored = [
            doc_id
            for doc_id in ordered_ids
            if doc_id not in forced_id_set
            and _doc_contains_any_value(doc_by_id[doc_id], anchor_values)
        ]
        if anchored:
            forced_part = [doc_id for doc_id in ordered_ids if doc_id in forced_id_set]
            anchored_set = set(anchored)
            rest = [
                doc_id
                for doc_id in ordered_ids
                if doc_id not in forced_id_set and doc_id not in anchored_set
            ]
            # Preserve relative order within each bucket
            ordered_ids = forced_part + anchored + rest

    ordered_ids = ordered_ids[:limit]
    documents = [doc_by_id[i] for i in ordered_ids]
    filenames = [
        (meta_by_id[i].get("filename") or f"{i}.txt") for i in ordered_ids
    ]
    doc_metadata = [meta_by_id[i] for i in ordered_ids]
    return ordered_ids, documents, filenames, doc_metadata


COMPARISON_SYNTHESIS_INSTRUCTION = (
    "Cette question est une comparaison. Suis ce protocole strictement :\n"
    "a) Nomme explicitement les entités comparées.\n"
    "b) Pour chaque entité, n'utilise que les attributs réellement présents "
    "dans les documents récupérés — n'invente aucun attribut manquant.\n"
    "c) Ne compare que les champs pour lesquels toutes les entités concernées "
    "ont une donnée (champs qui se chevauchent).\n"
    "d) Si une entité n'a pas de donnée pour un attribut que tu voudrais "
    "comparer, dis-le explicitement plutôt que d'omettre silencieusement "
    "ou de deviner."
)

AGGREGATION_SYNTHESIS_INSTRUCTION = (
    "Cette question demande un dénombrement, une distribution ou une "
    "propriété d'ensemble. Suis ce protocole :\n"
    "a) Parcours TOUS les documents fournis (pas seulement le premier).\n"
    "b) Extrais et agrège uniquement les faits explicitement présents.\n"
    "c) Si tu comptes ou listes des entités, base-toi sur l'union des "
    "documents consultés et rappelle que la liste peut être partielle "
    "(échantillon top-K), sans inventer les manquants."
)


def answer_question(question: str, n_results: int = DEFAULT_N_RESULTS) -> dict[str, Any]:
    """Retrieve top-K corpus docs and answer with GPT-4o-mini grounded in them."""
    openai_client = _get_openai_client()
    collection = _get_collection()

    # Proposal C: smaller K for verification / data-quality style questions.
    effective_k = (
        VERIFICATION_N_RESULTS if _is_verification_question(question) else n_results
    )

    try:
        query_embedding = _embed_query(openai_client, question)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: embedding failed for question: {exc}")
        raise

    # Proposal B: force-include docs whose ids are named in the question.
    forced_ids = _named_entities_in_question(question, _corpus_doc_ids(collection))
    try:
        forced = _fetch_docs_by_ids(collection, forced_ids)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: forced-entity fetch failed: {exc}")
        forced = ([], [], [])

    preferred_type = _preferred_entity_type(question)
    preferred: tuple[list[str], list[str], list[dict]] = ([], [], [])
    try:
        if preferred_type is not None:
            preferred = _query_by_embedding(
                collection,
                query_embedding,
                n_results=effective_k,
                where={"entity_type": preferred_type},
            )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: preferred-type Chroma query failed: {exc}")
        preferred = ([], [], [])

    try:
        general = _query_by_embedding(
            collection,
            query_embedding,
            n_results=effective_k,
            where=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Chroma query failed: {exc}")
        raise

    # Value-anchored retrieval: prioritize candidates whose text contains a
    # taxonomy literal found in the question (no-op if none match).
    anchor_values = _taxonomy_values_in_question(question)

    _ids, documents, filenames, doc_metadata = _merge_retrieval_results(
        forced=forced,
        preferred=preferred,
        general=general,
        limit=effective_k,
        anchor_values=anchor_values or None,
    )

    if not documents:
        return {
            "answer": (
                "Aucun document pertinent n'a été récupéré ; "
                "impossible de répondre à partir du contexte."
            ),
            "retrieved_docs": [],
            "question": question,
            "retrieved_passages": [],
            "retrieved_ids": [],
            "retrieved_metadata": [],
        }

    user_prompt = _build_user_prompt(question, documents, filenames)
    user_prompt = (
        user_prompt
        + "\n"
        + "Les documents fournis sont un échantillon récupéré (top-K) et "
        + "peuvent ne pas couvrir toutes les entités ou faits pertinents du "
        + "corpus. Si ta réponse s'appuie sur des correspondances trouvées "
        + "dans cet échantillon, précise-le explicitement — par exemple "
        + "« parmi les documents consultés » ou « cette liste n'est "
        + "peut-être pas exhaustive » — au lieu de présenter ces "
        + "correspondances comme l'ensemble complet des résultats."
    )
    if _is_comparative_question(question):
        user_prompt = user_prompt + "\n\n" + COMPARISON_SYNTHESIS_INSTRUCTION
    if _is_aggregation_question(question):
        user_prompt = user_prompt + "\n\n" + AGGREGATION_SYNTHESIS_INSTRUCTION

    try:
        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = (completion.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: GPT-4o-mini generation failed: {exc}")
        raise

    return {
        "answer": answer,
        "retrieved_docs": filenames,
        "question": question,
        "retrieved_passages": documents,
        "retrieved_ids": _ids,
        "retrieved_metadata": doc_metadata,
    }


TEST_QUESTIONS = [
    "Quelle est la race de Dakota ?",
    "Quels événements ont eu lieu à Saumur ?",
    "Quels capteurs IMU sont utilisés pour détecter la fatigue ?",
]


def main() -> int:
    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "=" * 72)
        print(f"TEST {i}/{len(TEST_QUESTIONS)}")
        print("=" * 72)
        print(f"Question: {question}")
        print("-" * 72)
        try:
            result = answer_question(question)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            continue
        print(f"Retrieved docs ({len(result['retrieved_docs'])}):")
        for name in result["retrieved_docs"]:
            print(f"  - {name}")
        print("-" * 72)
        print("Answer:")
        print(result["answer"])
    print("\n" + "=" * 72)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
