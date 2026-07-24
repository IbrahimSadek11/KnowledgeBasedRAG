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

_SCHEMA_CACHE = None

# Inline notes appended to matching PRAGMA columns in get_schema_description().
COLUMN_NOTES = {
    ("trainings", "volume"): (
        "(display text, e.g. '45min' — for numeric comparison, sorting, "
        "MAX/MIN or averages, use volume_minutes instead)"
    ),
    ("trainings", "volume_minutes"): (
        "(INTEGER — parsed from volume; ALWAYS use this one for numeric "
        "comparison, sorting, MAX/MIN, averages)"
    ),
    ("sensors", "sample_rate"): (
        "(display text, e.g. '250Hz' — for numeric comparison use "
        "sample_rate_hz instead)"
    ),
    ("sensors", "sample_rate_hz"): (
        "(INTEGER — parsed from sample_rate; ALWAYS use this one for "
        "numeric comparison, sorting, MAX/MIN)"
    ),
    ("sensors", "sensor_offset"): (
        "(display text — for numeric comparison use sensor_offset_value instead)"
    ),
    ("sensors", "sensor_offset_value"): "(REAL — parsed numeric form)",
}

TABLE_CHOICE_GUIDE = (
    "GUIDE DE CHOIX DE TABLE :\n"
    "- 'participe à', 'engagé dans', 'inscrit à', 'prend part à' (competing "
    "in an event) -> event_entries (NOT trainings — trainings are training "
    "sessions, not competition entries)\n"
    "- 'concourt', 'a concouru' -> event_entries\n"
    "- 'classement', 'résultat', 'rang' -> event_participations\n"
    "- 's'entraîne', 'entraînement', 'séance' -> trainings"
)

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
    "answer step to count by hand. "
    "Pour une question demandant à la fois le premier/plus ancien ET le "
    "dernier/plus récent élément (ex: 'le premier et le dernier événement'), "
    "n'utilise JAMAIS UNION avec un ORDER BY ou LIMIT à l'intérieur de chaque "
    "branche — SQLite rejette cette syntaxe. Utilise plutôt un seul SELECT "
    "avec une clause WHERE combinant MIN et MAX en sous-requêtes, par exemple : "
    "SELECT <colonnes> FROM <table> "
    "WHERE <colonne_date> = (SELECT MIN(<colonne_date>) FROM <table>) "
    "OR <colonne_date> = (SELECT MAX(<colonne_date>) FROM <table>);"
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


def get_schema_description(force_refresh: bool = False) -> str:
    """Build a plain-text schema description live from tabular.db (read-only).

    For every TEXT column we also pull 2-3 distinct real sample values straight
    from the data so the model can see the actual stored vocabulary (e.g. that
    sensor_type holds body positions like 'Withers', not the literal 'IMU').

    Result is cached at module level so subsequent calls in the same process
    reuse the text without re-querying tabular.db (unless force_refresh=True).
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None and not force_refresh:
        return _SCHEMA_CACHE

    print("[schema] querying tabular.db (cache miss)")
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
                note = COLUMN_NOTES.get((table, col_name))
                if note:
                    line += f" {note}"
                lines.append(line)
        lines.append("")
        lines.append(TABLE_CHOICE_GUIDE)
        _SCHEMA_CACHE = "\n".join(lines)
        return _SCHEMA_CACHE
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
    """Execute `sql` against tabular.db read-only; let exceptions propagate.

    Registers a sqlite3 authorizer that allows only read-side actions needed
    for SELECT (and normal JOIN/WHERE/ORDER BY/GROUP BY/aggregate functions).
    Write/DDL actions are denied even if validate_sql() was bypassed.
    """
    # Empirically required for this project's SELECT / JOIN / aggregate queries.
    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }

    def _authorizer(action, arg1, arg2, dbname, source):  # noqa: ARG001
        if action in allowed_actions:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        conn.set_authorizer(_authorizer)
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
    print("=" * 70)
    print("SQL SAFETY VERIFICATION")
    print("=" * 70)

    # 1) False-positive fix at validate_sql layer
    v_sql = "SELECT * FROM horses WHERE race LIKE '%DROP%'"
    v_ok, v_msg = validate_sql(v_sql)
    print(f"1) validate_sql({v_sql!r})")
    print(f"   -> ({v_ok}, {v_msg!r})")
    print()

    # 2) Same query through execute_sql against real DB
    try:
        rows_like = execute_sql(v_sql)
        print(f"2) execute_sql({v_sql!r})")
        print(f"   -> ran OK, {len(rows_like)} row(s): {rows_like}")
    except Exception as exc:  # noqa: BLE001
        print(f"2) execute_sql FAILED: {exc!r}")
    print()

    # 3) Authorizer as real second layer (bypass validate_sql on purpose)
    print("3) execute_sql('DROP TABLE horses') [validate_sql bypassed]")
    try:
        execute_sql("DROP TABLE horses")
        print("   -> UNEXPECTED: DROP succeeded")
    except Exception as exc:  # noqa: BLE001
        print(f"   -> blocked with exception: {type(exc).__name__}: {exc}")
    count_after = execute_sql("SELECT COUNT(*) FROM horses")
    print(f"   SELECT COUNT(*) FROM horses -> {count_after} (expect [(50,)])")
    print()

    # 4) Full sql_validator self-test suite
    print("4) sql_validator self-test suite:")
    from backend.tabular_rag import sql_validator as _sv

    test_cases = [
        (
            "SELECT frequency FROM trainings JOIN horses ON trainings.horse_id "
            "= horses.horse_id WHERE horses.name = 'Dakota'",
            True,
        ),
        ("DROP TABLE trainings", False),
        ("SELECT * FROM horses; DROP TABLE horses", False),
        ("DELETE FROM trainings WHERE horse_id = 'Horse1'", False),
        ("SELECT * FROM horses WHERE name = 'updated_athlete'", True),
        ("SELECT * FROM horses WHERE race LIKE '%DROP%'", True),
    ]
    all_pass = True
    for sql, expected_valid in test_cases:
        is_valid, message = _sv.validate_sql(sql)
        match = is_valid == expected_valid
        all_pass = all_pass and match
        label = "MATCH" if match else "MISMATCH"
        print(f"   INPUT:    {sql}")
        print(f"   RESULT:   ({is_valid}, {message!r})")
        print(f"   EXPECTED: valid={expected_valid} -> {label}")
        print("   " + "-" * 66)
    print(f"   ALL PASS: {all_pass}")
    print()

    # Empirically confirm a real multi-table JOIN still authorizes
    join_sql = (
        "SELECT DISTINCT t.frequency, t.stage_type "
        "FROM trainings t "
        "JOIN horses h ON t.horse_id = h.horse_id "
        "WHERE LOWER(h.name) = LOWER('Dakota') "
        "ORDER BY t.stage_type"
    )
    join_rows = execute_sql(join_sql)
    print(f"JOIN smoke (authorizer allow-list): {len(join_rows)} row(s) -> {join_rows}")
    print()

    # 5) End-to-end answer_question for three real questions
    print("=" * 70)
    print("5) answer_question() end-to-end")
    print("=" * 70)
    questions = [
        "Quelle est la race de Aurore ?",
        "Quelle est la fréquence d'entraînement de Dakota pour Event_SJ_01 ?",
        "Qui est le vétérinaire impliqué dans la préparation de Tonnerre ?",
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
