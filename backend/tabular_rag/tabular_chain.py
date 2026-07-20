"""
Core Tabular RAG chain.

Takes a natural-language question and returns a natural-language answer by
wiring together: live schema description -> LLM SQL generation -> safety
validation (backend.tabular_rag.sql_validator) -> retry-on-failure ->
read-only execution -> LLM answer synthesis.

Reuses the same ChatOpenAI configuration as the Graph RAG code
(gpt-4o-mini, temperature=0) and the shared validate_sql() safety check.
"""
import os
import re
import sqlite3

from langchain_openai import ChatOpenAI

from backend.config import OPENAI_API_KEY
from backend.tabular_rag.sql_validator import validate_sql

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tabular.db")

TABLES = [
    "horses",
    "events",
    "seasons",
    "trainings",
    "training_actors",
    "event_participations",
    "event_entries",
    "horse_rider_associations",
    "people",
    "sensors",
    "objectives",
]

SQL_INSTRUCTION = (
    "Given the schema above, write a single SQLite SELECT query (and only the "
    "SQL, no explanation, no markdown formatting) that answers the following "
    "question. When filtering on text columns, always use case-insensitive "
    "comparison, e.g. WHERE LOWER(column) = LOWER('value') or the SQLite "
    "COLLATE NOCASE operator — never assume the exact casing of stored values. "
    "If the question names a specific training phase (e.g. 'preparation', "
    "'pre-competition', 'competition', 'transition'), you MUST filter the "
    "trainings table on its stage_type column using the exact matching value "
    "(PreparationStage, PreCompetitionStage, CompetitionStage, TransitionStage "
    "respectively) — do not search across all of a horse's training stages "
    "when the question specifies one. Also always use SELECT DISTINCT to avoid "
    "duplicate rows from multiple matching join paths. "
    "Apply case-insensitive comparison (LOWER(column) = LOWER('value')) to "
    "EVERY text column filter in the query, with no exceptions — this includes "
    "actor_role, stage_type, discipline, category, and any other text column, "
    "not just names. Additionally, if the question mentions a specific role "
    "(veterinarian, caretaker, or rider), you MUST include an explicit filter "
    "on actor_role for that value — never return rows for all actors when a "
    "specific role was named in the question. "
    "If a horse can plausibly have multiple training records that differ by "
    "training phase (e.g. questions about frequency, intensity, or volume), "
    "always include trainings.stage_type in your SELECT alongside the "
    "requested value, so each row can be correctly attributed to its specific "
    "phase. Never return a bare value column alone when a disambiguating "
    "column exists in the same table that would explain why multiple rows "
    "were returned. "
    "If the question asks for information that has no corresponding column "
    "anywhere in the schema (e.g. age, weight, color, phone number), you MUST "
    "write a query that returns zero rows or acknowledges the absence — NEVER "
    "compute, derive, or approximate an answer using unrelated columns (e.g. "
    "never calculate an 'age' from an event date). If no real column answers "
    "the question, say so; do not invent a proxy calculation. "
    "For questions asking for the most common, most frequent, highest, or "
    "lowest value of something (e.g. 'quelle est la race la plus courante'), "
    "you MUST use GROUP BY with COUNT or an appropriate aggregate, and ORDER BY "
    "... DESC LIMIT 1 — never simply list all distinct values without counting "
    "them. "
    "For questions about riders and horses, use the correct table for what is "
    "actually being asked: use horse_rider_associations for general association "
    "('associé à', 'travaille avec'); use event_participations for a ranked "
    "result at a specific event ('classement', 'résultat'); use event_entries "
    "for whether a horse was entered/competed in an event, regardless of "
    "whether a ranked result exists ('engagé dans', 'participe à', or when "
    "comparing entries vs results). These represent three different real-world "
    "facts and must not be used interchangeably. "
    "NEVER guess a horse's horse_id by pattern-matching its name (e.g. "
    "assuming 'Dakota' -> 'Horse_Dakota'). horse_id values are NOT always "
    "name-based (e.g. some are 'Horse1', 'Horse2'). Always filter horses by "
    "their name column directly (WHERE LOWER(name) = LOWER('...')) and obtain "
    "horse_id via that match, never by string-constructing it. "
    "For yes/no or 'how common/frequent' questions comparing counts "
    "across many rows (e.g. 'is it common for X to happen without Y'), your "
    "query MUST compute a summary count (e.g. COUNT(DISTINCT ...) alongside a "
    "total count) rather than returning every individual matching row for the "
    "answer step to count by hand."
)


def _get_llm():
    """Same client configuration as backend/llm_service.py."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )


def _strip_code_fences(text: str) -> str:
    """Remove ```sql ... ``` or ``` ... ``` fences if the model added them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```sql) and the closing fence.
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def get_schema_description() -> str:
    """Build a plain-text schema description live from tabular.db (read-only).

    For every TEXT column we also pull 2-3 distinct real sample values straight
    from the data so the model can see the actual stored vocabulary (e.g. that
    sensor_type holds body positions like 'Withers', not the literal 'IMU').
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        lines = []
        for table in TABLES:
            cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
            # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            lines.append(f"Table {table}:")
            for col in cols:
                col_name, col_type = col[1], col[2]
                line = f"  - {col_name} {col_type}"
                if "TEXT" in (col_type or "").upper():
                    samples = [
                        str(row[0])
                        for row in cur.execute(
                            f'SELECT DISTINCT "{col_name}" FROM {table} '
                            f'WHERE "{col_name}" IS NOT NULL LIMIT 3'
                        )
                    ]
                    if samples:
                        line += f" (examples: {', '.join(samples)})"
                lines.append(line)
        return "\n".join(lines)
    finally:
        conn.close()


def get_all_horse_names_and_ids() -> list[tuple[str, str]]:
    """Return live (horse_id, name) pairs from tabular.db (read-only)."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT horse_id, name FROM horses").fetchall()
    finally:
        conn.close()


def _resolved_horse_ids_block(question: str) -> str:
    """Build an IDENTIFIANTS RÉSOLUS block for horse names found in `question`.

    Uses whole-word, case-insensitive matching so short names like 'Rio' do not
    match inside unrelated words. Returns "" when no horse is mentioned.
    """
    horses = get_all_horse_names_and_ids()
    # Longer names first so a full name wins over a shorter overlapping one.
    horses_sorted = sorted(horses, key=lambda pair: len(pair[1] or ""), reverse=True)
    found: list[tuple[str, str]] = []
    for horse_id, name in horses_sorted:
        if not name:
            continue
        if re.search(rf"\b{re.escape(name)}\b", question, flags=re.IGNORECASE):
            found.append((horse_id, name))

    if not found:
        return ""

    lines = [
        "IDENTIFIANTS RÉSOLUS (utilise EXACTEMENT ces valeurs, ne devine et "
        "ne construis JAMAIS un horse_id toi-même) :"
    ]
    for horse_id, name in found:
        lines.append(f"- '{name}' -> horse_id = '{horse_id}'")
    return "\n".join(lines)


def generate_sql(question: str, error_feedback: str | None = None) -> str:
    """Ask the LLM for a single SQLite SELECT query for `question`."""
    schema_description = get_schema_description()
    resolved_block = _resolved_horse_ids_block(question)

    # Schema first, then resolved horse_ids (if any), then instruction + question.
    prompt_parts = [schema_description]
    if resolved_block:
        prompt_parts.extend(["", resolved_block])
    prompt_parts.extend(["", SQL_INSTRUCTION, "", f"Question: {question}"])
    prompt = "\n".join(prompt_parts)

    if error_feedback:
        prompt += (
            f"\n\nYour previous attempt failed with this error: "
            f"{error_feedback}. Please provide a corrected query."
        )
    response = _get_llm().invoke(prompt)
    return _strip_code_fences(response.content)


def execute_sql(sql: str) -> list:
    """Execute `sql` against tabular.db read-only; let exceptions propagate."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


AGGREGATION_SIGNAL_PHRASES = [
    "combien",
    "répart",
    "comparer",
    "comparaison",
    "plus fréquent",
    "plus courant",
    "plus courante",
    "moyenne",
    "en moyenne",
]

AGGREGATE_SQL_MARKERS = [
    "COUNT(",
    "SUM(",
    "AVG(",
    "MAX(",
    "MIN(",
    "GROUP BY",
]


def needs_aggregation_check(question: str, sql: str) -> tuple[bool, str]:
    """Rule-based gate: aggregation questions must use COUNT/GROUP BY etc."""
    q_low = (question or "").lower()
    sql_low = (sql or "").lower()
    asks_aggregation = any(phrase in q_low for phrase in AGGREGATION_SIGNAL_PHRASES)
    has_aggregate = any(marker.lower() in sql_low for marker in AGGREGATE_SQL_MARKERS)
    if asks_aggregation and not has_aggregate:
        return (
            False,
            "This question implies a count, distribution, or comparison across "
            "multiple rows, but the query has no aggregate function or GROUP BY. "
            "Add COUNT/GROUP BY as appropriate — do not return a bare list of "
            "distinct values.",
        )
    return True, "OK"


def answer_question(question: str, max_retries: int = 2) -> dict:
    """Full chain: generate -> validate -> execute (with retries) -> answer."""
    attempts = []
    error_feedback = None
    working_sql = None
    rows = None

    for _ in range(max_retries + 1):
        sql = generate_sql(question, error_feedback)

        is_valid, validation_message = validate_sql(sql)
        if not is_valid:
            attempts.append(
                {"sql": sql, "outcome": f"validation failure: {validation_message}"}
            )
            error_feedback = validation_message
            continue

        agg_ok, agg_message = needs_aggregation_check(question, sql)
        if not agg_ok:
            attempts.append(
                {"sql": sql, "outcome": f"aggregation check failure: {agg_message}"}
            )
            error_feedback = agg_message
            continue

        try:
            rows = execute_sql(sql)
        except Exception as exc:  # noqa: BLE001 - surfaced to the LLM as feedback
            attempts.append(
                {"sql": sql, "outcome": f"execution failure: {exc}"}
            )
            error_feedback = str(exc)
            continue

        attempts.append({"sql": sql, "outcome": "success"})
        working_sql = sql
        break

    if working_sql is None:
        return {
            "question": question,
            "sql": None,
            "rows": None,
            "answer": "Could not generate a valid query after retries.",
            "attempts": attempts,
        }

    answer_prompt = (
        "À partir de la question et des lignes brutes de la base de données "
        "ci-dessous, rédige une réponse courte et naturelle.\n\n"
        "RÈGLES DE FORMAT\n"
        "- Réponds en français naturel et fluide.\n"
        "- N'expose jamais les structures de données brutes, les noms de "
        "colonnes, les noms de tables, les URIs ou les identifiants techniques.\n"
        "- Utilise directement les informations dans des phrases naturelles.\n"
        "- Ne dis jamais \"d'après le contexte\" ou \"selon les lignes\".\n\n"
        "RÈGLES DE PRÉSENTATION DES NOMS\n"
        "- Les noms de chevaux sont déjà des noms réels : utilise-les tels "
        "quels (ne les traite pas comme des identifiants).\n"
        "- Les identifiants de cavaliers sont au format Rider_XXXX : présente "
        "naturellement seulement la partie nom.\n"
        "- Les identifiants de vétérinaires sont au format Vet_XXXX : présente "
        "naturellement le nom.\n"
        "- Les identifiants de soigneurs sont au format Caretaker_XXXX : "
        "présente naturellement le nom.\n"
        "- Les phases d'entraînement (PreparationStage, PreCompetitionStage, "
        "CompetitionStage, TransitionStage) doivent être exprimées en français "
        "naturel (phase de préparation, phase de pré-compétition, phase de "
        "compétition, phase de transition).\n"
        "- N'expose jamais les URIs brutes ni les identifiants internes "
        "techniques à l'utilisateur.\n\n"
        "RÈGLES DE COMPLÉTUDE\n"
        "- Si plusieurs lignes distinctes sont retournées, ta réponse doit "
        "rendre compte de chacune d'elles individuellement — ne résume pas à "
        "une seule valeur si les lignes représentent des entités réellement "
        "différentes (par exemple des phases d'entraînement différentes, des "
        "capteurs différents).\n"
        "- Si les lignes sont ambiguës ou si la question ne permet pas de les "
        "départager, dis-le explicitement plutôt que d'en choisir une.\n\n"
        f"Question : {question}\n"
        f"Lignes : {rows}"
    )
    answer = _get_llm().invoke(answer_prompt).content.strip()

    return {
        "question": question,
        "sql": working_sql,
        "rows": rows,
        "answer": answer,
        "attempts": attempts,
    }


if __name__ == "__main__":
    questions = [
        "Quelle est la race de Aurore ?",
        "Quelle est la fréquence d'entraînement de Dakota pour Event_SJ_01 ?",
        "Qui est le vétérinaire impliqué dans la préparation de Tonnerre ?",
        "Quelle est la fréquence d'entraînement de Dakota pour Event_SJ_01 ?",
    ]

    for q in questions:
        result = answer_question(q)
        print("=" * 70)
        print(f"QUESTION: {result['question']}")
        print("\nATTEMPTS:")
        for i, attempt in enumerate(result["attempts"], start=1):
            print(f"  [{i}] SQL: {attempt['sql']}")
            print(f"      OUTCOME: {attempt['outcome']}")
        print(f"\nFINAL SQL: {result['sql']}")
        print(f"RAW ROWS: {result['rows']}")
        print(f"ANSWER: {result['answer']}")
        print()
