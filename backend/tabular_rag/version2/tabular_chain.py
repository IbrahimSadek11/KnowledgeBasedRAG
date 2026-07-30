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

# version2: __file__ = .../backend/tabular_rag/version2/tabular_chain.py
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tabular_rag", "version2", "tabular.db")

TABLES = [
    "horses",
    "events",
    "seasons",
    "trainings",
    "training_actors",
    "event_participations",
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
    ("sensors", "objective_id"): (
        "(FK -> objectives.objective_id — Neo4j ISUSEDFOR. ALWAYS join "
        "sensors to objectives ON sensors.objective_id = objectives.objective_id. "
        "objectives has NO sensor_id column — never write "
        "sensor_id IN (SELECT sensor_id FROM objectives ...))"
    ),
    ("objectives", "objective_id"): (
        "(PK — join from sensors.objective_id; do NOT invent a sensor_id "
        "column on this table)"
    ),
    ("people", "name"): (
        "(display name from Neo4j hasName — use ONLY when the question "
        "explicitly asks for the name ('quel est le nom', 'what's the "
        "name of'); for 'qui est le [role]' identity questions use "
        "person_id, not name)"
    ),
}

TABLE_CHOICE_GUIDE = (
    "GUIDE DE CHOIX DE TABLE :\n"
    "- 'participe à', 'engagé dans', 'inscrit à', 'prend part à' (competing "
    "in an event) -> event_participations (NOT trainings — trainings are "
    "training sessions, not competition entries)\n"
    "- 'concourt', 'a concouru' -> event_participations\n"
    "- 'classement', 'résultat', 'rang' -> event_participations\n"
    "- 's'entraîne', 'entraînement', 'séance' -> trainings\n"
    "- 'récupération', 'phase de récupération', 'après une compétition' "
    "(recovery) -> trainings with stage_type = 'TransitionStage' "
    "(NOT PreparationStage)\n"
    "- 'qui participe à la phase' / actors in a training phase "
    "(cavalier, vétérinaire, soigneur) -> JOIN trainings + "
    "training_actors + people (return person_id/role), NOT the horse list\n"
    "- same rider finishing 1st and 2nd at one event -> "
    "event_participations grouped by event_id, rider_id with "
    "HAVING COUNT(DISTINCT rank) = 2 for ranks in (1,2); when self-joining "
    "rows, require the same rider_id on both sides\n"
    "- same rider presenting multiple horses at the SAME event "
    "('lors du même événement', 'à la même compétition') -> "
    "event_participations GROUP BY event_id, rider_id HAVING "
    "COUNT(DISTINCT horse_id) > 1 (event_id MUST be in the grain — "
    "never GROUP BY rider_id alone, which aggregates across all events)\n"
    "- sensor used for / serves objective / démarche vs fatigue "
    "(ISUSEDFOR) -> sensors.objective_id JOIN objectives ON "
    "sensors.objective_id = objectives.objective_id; SELECT sensor_id "
    "AND objective_id (or objective name). NEVER correlate via a "
    "non-existent objectives.sensor_id"
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
    "training phase AND the question does NOT already pin a single phase in "
    "WHERE (and is not asking for a herd-wide distribution/count), AND the "
    "selected columns are attribute values (intensity, volume, frequency) "
    "rather than training identifiers, include trainings.stage_type in your "
    "SELECT alongside the requested value so each row is attributed to its "
    "phase. Do NOT add stage_type when the question asks for the list of "
    "training steps/étapes (those return training_id only). "
    "DISTRIBUTION vs INVENTORY (mandatory): "
    "(1) DISTRIBUTION — the question asks how a property varies across the "
    "herd / how many horses share each value (cues: 'varie', 'varient', "
    "'répart', 'répartition', 'distribution', 'combien de X par Y', "
    "'combien de chevaux', 'quelle est la durée/fréquence' of "
    "sessions in a named phase, 'où sont placés les capteurs' with a "
    "count, 'quels capteurs … fatigue … démarche' as tallies by "
    "objective, calibration/offset counts by type, whether a training "
    "recipe differs, stage×role intervention counts). Return "
    "SELECT <value_col>, COUNT(*) / COUNT(DISTINCT …) … "
    "GROUP BY the relevant column(s). Do NOT return raw ungrouped rows, "
    "bare DISTINCT inventories, or GROUP BY the wrong grain "
    "(e.g. sensor_id when the question asks for sensor_type distribution; "
    "actor_id rows when comparing stage×frequency; sensor_offset alone "
    "when the question is about same placement / sensor_type). "
    "Concrete shapes: "
    "SELECT sensor_type, COUNT(*) FROM sensors GROUP BY sensor_type; "
    "SELECT objective_id, COUNT(*) FROM sensors GROUP BY objective_id; "
    "SELECT sensor_type, sensor_offset, COUNT(*) FROM sensors "
    "GROUP BY sensor_type, sensor_offset; "
    "SELECT stage_type, frequency, COUNT(DISTINCT horse_id) FROM trainings "
    "… GROUP BY stage_type, frequency; "
    "SELECT DISTINCT volume, intensity, frequency FROM trainings "
    "WHERE stage_type = 'CompetitionStage'; "
    "SELECT t.stage_type, ta.actor_role, COUNT(DISTINCT t.training_id) "
    "FROM trainings t JOIN training_actors ta … "
    "GROUP BY t.stage_type, ta.actor_role. "
    "Do NOT use bare DISTINCT and do NOT collapse to a single AVG(...). "
    "If stage_type is already fixed in WHERE, do not add stage_type to "
    "SELECT just to justify DISTINCT — use GROUP BY + COUNT. "
    "Correct distribution example: "
    "SELECT volume, COUNT(DISTINCT horse_id) AS horse_count FROM trainings "
    "WHERE stage_type = 'PreparationStage' GROUP BY volume; "
    "(2) INVENTORY — the question only asks which values exist "
    "('quels niveaux', 'quelles valeurs', 'quels types') with no request "
    "for how many horses per value. Then SELECT DISTINCT <value_col> is "
    "correct; do not invent a COUNT. "
    "Wrong for a distribution question (described only — do not emit): "
    "listing distinct volumes with no COUNT, or averaging into one number. "
    "If the question asks for information that has no corresponding column "
    "anywhere in the schema (e.g. age, weight, color, phone number), you MUST "
    "write a query that returns zero rows or acknowledges the absence — NEVER "
    "compute, derive, or approximate an answer using unrelated columns (e.g. "
    "never calculate an 'age' from an event date). If no real column answers "
    "the question, say so; do not invent a proxy calculation. "
    "TIE-SAFE SUPERLATIVES (mandatory): for the most / least / highest / "
    "longest / plus de / moins de / 'plus représentée' / 'plus fréquente' / "
    "'most common' value (count or stored column), NEVER use "
    "ORDER BY … LIMIT 1 — that hides ties. Keep EVERY row tied at the "
    "extreme (all disciplines/categories sharing the max count), not an "
    "arbitrary single winner and not an incomplete subset of the tie: "
    "filter with WHERE col = (SELECT MAX(col) …) / MIN(col), or for grouped "
    "counts HAVING COUNT(*) = (SELECT MAX(c) FROM (SELECT COUNT(*) AS c … "
    "GROUP BY …)). Example (max participation tie): "
    "SELECT event_id, COUNT(*) AS cnt FROM event_participations "
    "GROUP BY event_id HAVING COUNT(*) = (SELECT MAX(c) FROM ("
    "SELECT COUNT(*) AS c FROM event_participations GROUP BY event_id)); "
    "Example (most represented discipline): "
    "SELECT discipline, COUNT(*) AS event_count FROM events "
    "GROUP BY discipline HAVING COUNT(*) = (SELECT MAX(c) FROM ("
    "SELECT COUNT(*) AS c FROM events GROUP BY discipline)); "
    "Example (longest prep sessions): "
    "SELECT horse_id, volume FROM trainings WHERE stage_type = "
    "'PreparationStage' AND volume = (SELECT MAX(volume) FROM trainings "
    "WHERE stage_type = 'PreparationStage'); "
    "Still aggregate when needed — never list uncounted distinct values "
    "for a 'most common' question. For max sampling frequency return "
    "sensor_id only (no extra sample_rate column). "
    "PERIOD / TIME GRAIN (mandatory): when the question asks for the "
    "busiest 'période', 'mois', 'période la plus chargée', or a "
    "time-based competition load without explicitly saying day / date / "
    "jour, group by month — e.g. strftime('%Y-%m', event_date) — NOT by "
    "raw event_date (day-level). Only use day-level GROUP BY event_date "
    "when the question explicitly asks for a day or a specific date. "
    "SENSOR -> OBJECTIVE JOIN (mandatory, Neo4j ISUSEDFOR): "
    "sensors.objective_id is the FK to objectives.objective_id. Correct: "
    "SELECT s.sensor_id, s.objective_id FROM sensors s "
    "WHERE s.horse_id = …; "
    "or JOIN objectives o ON s.objective_id = o.objective_id. "
    "WRONG (described only — do not emit): "
    "sensor_id IN (SELECT sensor_id FROM objectives …) — objectives has "
    "NO sensor_id column; SQLite silently correlates to the outer "
    "sensors.sensor_id and returns every sensor without the objective "
    "split. Always project objective_id (or o.name) when the question "
    "asks which sensors serve démarche vs fatigue / which objective. "
    "MAJORITY / EXCEPTIONS / SHARED COUNT LEVELS (overrides a bare "
    "MIN/MAX horse list when the question is about the herd pattern): "
    "(a) 'la plupart des chevaux' / 'y a-t-il des exceptions' on how many "
    "compétitions/engagements -> exactly "
    "SELECT horse_id, COUNT(DISTINCT event_id) AS event_count FROM "
    "event_participations GROUP BY horse_id; "
    "Stop there: one row per horse from event_participations. Do NOT wrap "
    "that in an outer histogram, do NOT GROUP BY event_id. "
    "(b) 'quel cheval a le moins de capteurs' / sensor load across the "
    "herd -> exactly the histogram "
    "SELECT sensor_count, COUNT(*) AS num_horses FROM (SELECT horse_id, "
    "COUNT(*) AS sensor_count FROM sensors GROUP BY horse_id) "
    "GROUP BY sensor_count; "
    "Return ALL sensor_count levels (typically three rows). Do NOT add "
    "ORDER BY sensor_count LIMIT 1 on that histogram, do NOT list every "
    "horse tied at MIN, and do NOT keep only the minimum level — the "
    "full effectifs per level are the answer. "
    "For questions about riders and horses, use event_participations: "
    "use event_participations, grouped/deduplicated on horse_id+rider_id, "
    "for general association ('associé à', 'travaille avec'); use "
    "event_participations for a ranked result at a specific event "
    "('classement', 'résultat'); use event_participations for whether a "
    "horse competed in an event ('engagé dans', 'participe à'). "
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
    "OR <colonne_date> = (SELECT MAX(<colonne_date>) FROM <table>); "
    "OUTPUT SHAPE (mandatory): return exactly the columns the question asks "
    "for — no spare attributes. "
    "- 'quelles étapes d\\'entraînement' / training steps -> training_id "
    "ONLY (not stage_type labels, and do not SELECT stage_type alongside "
    "training_id). "
    "- 'de quel événement dépendent' -> JOIN events and return "
    "event_id, location, category, event_date, discipline, stage_type "
    "(not event_id alone). "
    "- classement of a named horse/rider at an event -> rider_id, rank "
    "from event_participations filtered by that horse/event (JOIN horses "
    "on name if needed). "
    "If both a horse name and a rider name appear for a classement at a "
    "named event, filter ONLY horses.name + event_id "
    "(AND-conjoined, never OR rider_id); the rider appears in the selected "
    "rider_id column. Do not invent Rider_<HorseName>. "
    "- 'un cavalier … un seul cheval ou … plusieurs' -> "
    "SELECT rider_id, COUNT(DISTINCT horse_id) … FROM "
    "event_participations GROUP BY rider_id (per-rider distribution), "
    "not a single pair of global COUNT(DISTINCT rider_id/horse_id). "
    "- 'un même cavalier … deux chevaux … lors du même événement' / "
    "same rider, multiple horses at one competition -> "
    "GROUP BY event_id, rider_id HAVING COUNT(DISTINCT horse_id) > 1 "
    "(include event_id in the grain; do NOT GROUP BY rider_id alone). "
    "- sampling frequency as a stored label -> sample_rate text column "
    "(e.g. '200Hz'), not sample_rate_hz, unless the question asks to "
    "compare/sort numerically. "
    "- capteurs de X pour démarche vs fatigue / ISUSEDFOR -> "
    "SELECT sensor_id, objective_id FROM sensors filtered by horse "
    "(JOIN objectives only on objective_id). Never "
    "sensor_id IN (SELECT sensor_id FROM objectives …). "
    "- actor comparisons across named phases (e.g. préparation vs "
    "pré-compétition, or 'compare les acteurs') -> JOIN trainings t to "
    "training_actors ta (stage_type lives on trainings, not on "
    "training_actors) and SELECT DISTINCT ta.actor_id, ta.actor_role, "
    "t.stage_type in that column order (DISTINCT is mandatory — without "
    "it the join duplicates inflate the result set). Do NOT add "
    "stage_type (and do not filter a single phase) when the question "
    "only asks who can supervise / which non-rider actors exist in "
    "general — then SELECT DISTINCT actor_id, actor_role from "
    "training_actors (or person_id, role via people) with no stage_type. "
    "ALWAYS return both columns (id + role); never actor_id/person_id "
    "alone for that question type. "
    "- 'qui est le vétérinaire' / 'qui est le soigneur' / 'qui est le "
    "cavalier' / which person holds a named role -> SELECT person_id "
    "FROM people WHERE role = … (person_id ONLY — do not also SELECT "
    "role or name). "
    "- 'quel est le nom de' / 'what's the name of' a person (explicit "
    "name request only) -> people.name. Do NOT use name for 'qui est "
    "le [role]' identity questions — those use person_id. "
    "- 'quels événements' of a named season -> event_id, location, "
    "category, event_date, discipline (not event_id alone). "
    "- 'fréquence d\\'entraînement' -> column frequency (never substitute "
    "volume/duration). 'durée des séances' -> column volume. "
    "- first/last event -> event_id, event_date only (no location/category "
    "unless asked). "
    "- 'plus grand nombre d\\'étapes' / programme le plus complet -> "
    "COUNT(*) or COUNT(training_id) per horse_id (each training row is "
    "one étape). NEVER COUNT(DISTINCT stage_type): a horse can have "
    "several trainings of the same stage_type, and collapsing to distinct "
    "phase labels undercounts the programme. "
    "- count + a few example identifiers (e.g. 'combien … et quelques "
    "exemples d\\'identifiants') -> one aggregate COUNT over the full "
    "filter, plus GROUP_CONCAT(...) (or a separate list) for examples. "
    "NEVER GROUP BY the identifier being listed with LIMIT N: that makes "
    "the COUNT equal the LIMIT (e.g. 5) instead of the true total. "
    "- SQL string literals must be simple ASCII labels without French "
    "apostrophes or accented prose (e.g. 'association' / 'participation'). "
    "Never put French phrases like \"d'un événement\" inside quoted SQL "
    "strings — they break SQLite parsing. "
    "Correct shape example: "
    "SELECT training_id FROM trainings WHERE horse_id = "
    "(SELECT horse_id FROM horses WHERE LOWER(name) = LOWER('Dakota')); "
    "Wrong for that question (described only — do not emit): selecting "
    "DISTINCT stage_type and dropping the training identifiers."
)


def _get_llm():
    """Same client configuration as backend/graph_rag/llm_service.py."""
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
    "varie",
    "varient",
    # phase-scoped duration/frequency questions (distribution, not inventory)
    "durée des séances",
    "fréquence d'entraînement",
    "fréquence d’entraînement",
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
    # Bare GROUP BY without COUNT/SUM/... is not enough for distribution
    # questions (e.g. GROUP BY volume alone returns labels, not counts).
    has_count_like = any(
        m in sql_low for m in ("count(", "sum(", "avg(", "max(", "min(")
    )
    if asks_aggregation and has_aggregate and not has_count_like and (
        "durée des séances" in q_low
        or "fréquence d'entraînement" in q_low
        or "fréquence d’entraînement" in q_low
        or "répart" in q_low
        or "varie" in q_low
        or "varient" in q_low
    ):
        return (
            False,
            "This distribution question needs COUNT (or another aggregate) "
            "with GROUP BY — do not GROUP BY a value column alone.",
        )
    if asks_aggregation and not has_aggregate:
        return (
            False,
            "This question implies a count, distribution, or comparison across "
            "multiple rows, but the query has no aggregate function or GROUP BY. "
            "Add COUNT/GROUP BY as appropriate — do not return a bare list of "
            "distinct values.",
        )
    return True, "OK"


def needs_tie_safe_check(question: str, sql: str) -> tuple[bool, str]:
    """Reject LIMIT 1 on superlatives / sensor-load histograms (category D/F)."""
    q_low = (question or "").lower()
    sql_low = (sql or "").lower().replace("\n", " ")
    if "limit 1" not in sql_low:
        return True, "OK"

    sensor_herd = (
        "capteur" in q_low
        and ("moins" in q_low or "plus" in q_low)
        and "sensor" in sql_low
    )
    if sensor_herd:
        return (
            False,
            "For herd sensor-load questions, return the full histogram "
            "SELECT sensor_count, COUNT(*) AS num_horses FROM ("
            "SELECT horse_id, COUNT(*) AS sensor_count FROM sensors "
            "GROUP BY horse_id) GROUP BY sensor_count — with NO LIMIT 1.",
        )

    superlative = any(
        p in q_low
        for p in (
            "le plus",
            "la plus",
            "les plus",
            "le moins",
            "la moins",
            "plus grand nombre",
            "plus de résultats",
            "plus longues",
            "les plus longues",
        )
    )
    if superlative:
        return (
            False,
            "Superlative questions must keep ties: use WHERE col = (SELECT "
            "MAX/MIN(col)…) or HAVING COUNT(*) = (SELECT MAX(c) FROM …). "
            "Do not use ORDER BY … LIMIT 1.",
        )
    return True, "OK"


_STAGE_LABELS = {
    "PreparationStage": "préparation",
    "PreCompetitionStage": "pré-compétition",
    "CompetitionStage": "compétition",
    "TransitionStage": "transition",
}
_ROLE_LABELS = {
    "Rider": "cavaliers",
    "Veterinarian": "vétérinaire(s)",
    "Caretaker": "soigneur(s)",
}


def _actor_phase_rollup(rows) -> str | None:
    """Compact role×phase counts + set diffs for answer synthesis.

    Only triggers on 3-column (actor_id, actor_role, stage_type) result
    sets — the shape the model repeatedly miscounts by hand.
    """
    if not rows or not all(isinstance(r, (list, tuple)) and len(r) == 3 for r in rows):
        return None
    roles_seen = {str(r[1]) for r in rows}
    stages_seen = {str(r[2]) for r in rows}
    if not roles_seen.intersection(_ROLE_LABELS) or not stages_seen.intersection(
        _STAGE_LABELS
    ):
        return None
    if len(stages_seen) < 2:
        return None

    from collections import defaultdict

    by_stage_role: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for actor, role, stage in rows:
        by_stage_role[str(stage)][str(role)].add(str(actor))

    lines = [
        "Résumé structuré (effectifs exacts — ne recompte pas les lignes "
        "brutes) :"
    ]
    riders_by_stage: dict[str, set[str]] = {}
    for stage in sorted(by_stage_role, key=lambda s: _STAGE_LABELS.get(s, s)):
        label = _STAGE_LABELS.get(stage, stage)
        parts = []
        for role in ("Rider", "Veterinarian", "Caretaker"):
            names = sorted(by_stage_role[stage].get(role, set()))
            if not names:
                continue
            role_l = _ROLE_LABELS.get(role, role)
            short = [
                n.split("_", 1)[1] if "_" in n else n for n in names
            ]
            parts.append(f"{len(names)} {role_l} ({', '.join(short)})")
            if role == "Rider":
                riders_by_stage[stage] = set(names)
        lines.append(f"- phase de {label} : " + " ; ".join(parts))

    stage_list = sorted(riders_by_stage, key=lambda s: _STAGE_LABELS.get(s, s))
    if len(stage_list) == 2:
        a, b = stage_list
        only_a = sorted(riders_by_stage[a] - riders_by_stage[b])
        only_b = sorted(riders_by_stage[b] - riders_by_stage[a])
        la, lb = _STAGE_LABELS.get(a, a), _STAGE_LABELS.get(b, b)
        if only_a:
            short = [n.split("_", 1)[1] if "_" in n else n for n in only_a]
            lines.append(
                f"- cavaliers seulement en {la} (absents en {lb}) : "
                + ", ".join(short)
            )
        if only_b:
            short = [n.split("_", 1)[1] if "_" in n else n for n in only_b]
            lines.append(
                f"- cavaliers seulement en {lb} (absents en {la}) : "
                + ", ".join(short)
            )
        if not only_a and not only_b:
            lines.append(
                "- mêmes cavaliers dans les deux phases (aucune différence "
                "d'ensemble)."
            )
    return "\n".join(lines)


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

        tie_ok, tie_message = needs_tie_safe_check(question, sql)
        if not tie_ok:
            attempts.append(
                {"sql": sql, "outcome": f"tie-safe check failure: {tie_message}"}
            )
            error_feedback = tie_message
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

    rollup = _actor_phase_rollup(rows)
    rows_block = (
        f"{rollup}\n\nLignes brutes : {rows}" if rollup else f"Lignes : {rows}"
    )

    answer_prompt = (
        "À partir de la question et des lignes brutes de la base de données "
        "ci-dessous, rédige une réponse courte et naturelle.\n\n"
        "RÈGLES DE FORMAT\n"
        "- Réponds en français naturel et fluide.\n"
        "- N'expose jamais les structures de données brutes, les noms de "
        "colonnes, les noms de tables, les URIs ou les identifiants techniques "
        "(sauf identifiants de capteurs / d'événements quand la question les "
        "demande explicitement).\n"
        "- Utilise directement les informations dans des phrases naturelles.\n"
        "- Ne dis jamais \"d'après le contexte\" ou \"selon les lignes\".\n\n"
        "RÈGLES DE PRÉSENTATION DES NOMS\n"
        "- Les noms de chevaux sont déjà des noms réels : utilise-les tels "
        "quels (ne les traite pas comme des identifiants).\n"
        "- Les identifiants de cavaliers sont au format Rider_XXXX : présente "
        "naturellement seulement la partie nom.\n"
        "- Les identifiants de vétérinaires sont au format Vet_XXXX : présente "
        "naturellement le nom (ex. Dr Martin).\n"
        "- Les identifiants de soigneurs sont au format Caretaker_XXXX : "
        "présente naturellement le nom.\n"
        "- Les phases d'entraînement (PreparationStage, PreCompetitionStage, "
        "CompetitionStage, TransitionStage) doivent être exprimées en français "
        "naturel (phase de préparation, phase de pré-compétition, phase de "
        "compétition, phase de transition).\n"
        "- N'expose jamais les URIs brutes ni les identifiants internes "
        "techniques à l'utilisateur.\n\n"
        "RÈGLES D'EXHAUSTIVITÉ — UNE RÉPONSE INCOMPLÈTE EST UNE MAUVAISE "
        "RÉPONSE\n"
        "- Cite TOUS les nombres présents dans les lignes : totaux, "
        "effectifs par groupe, valeurs. Ne te contente jamais du total "
        "global quand les lignes donnent aussi le détail par groupe.\n"
        "- Quand les lignes contiennent une liste de noms (chevaux, "
        "cavaliers, acteurs, événements), énumère-les TOUS explicitement, "
        "sur une seule ligne séparés par des virgules, après le nombre "
        "exact. N'écris pas « plusieurs », « certains », ni « notamment » "
        "si les noms sont là — et n'omets jamais un rôle (vétérinaire, "
        "soigneur) parce que la liste des cavaliers est longue.\n"
        "- Quand les lignes comparent des acteurs par phase (actor_id / "
        "actor_role / stage_type), compte STRICTEMENT par rôle dans chaque "
        "phase d'après actor_role : les cavaliers (Rider) d'un côté, le "
        "vétérinaire et le soigneur de l'autre — ne mélange jamais "
        "soigneur/vétérinaire dans le total des cavaliers et n'invente "
        "aucun effectif. Pour une comparaison, cite explicitement qui "
        "apparaît dans une phase et pas dans l'autre (différence "
        "d'ensemble) : c'est le point central, pas une liste exhaustive "
        "seule.\n"
        "- Pour des identifiants techniques de capteurs, donne le nombre "
        "exact et deux ou trois exemples seulement : n'énumère jamais des "
        "dizaines d'identifiants.\n"
        "- Quand plusieurs lignes partagent la valeur extrême (même max, "
        "même fréquence la plus élevée), mentionne-les TOUTES et signale "
        "l'égalité.\n"
        "- Quand les lignes ne contiennent qu'une seule combinaison de "
        "valeurs pour tout un groupe, dis explicitement que c'est "
        "identique pour tous.\n"
        "- Pour une distribution / histogramme (une ligne par niveau avec "
        "un effectif), la 1re colonne est TOUJOURS le niveau (ex. "
        "fréquence '4x/week', sensor_count=2, volume '25min') et la 2e "
        "est le nombre d'entités qui l'ont. Ne les inverse jamais : "
        "(2, 44) = « 44 chevaux portent 2 capteurs », pas « un cheval a "
        "44 capteurs » ; (4x/week, 36) = « 36 chevaux à 4x/semaine », "
        "pas « fréquence = 36 ». Cite chaque couple niveau→effectif.\n"
        "- Si plusieurs lignes distinctes sont retournées, ta réponse doit "
        "rendre compte de chacune d'elles individuellement — ne résume pas "
        "à une seule valeur si les lignes représentent des entités "
        "réellement différentes (phases, capteurs, acteurs).\n"
        "- Si les lignes sont ambiguës ou si la question ne permet pas de "
        "les départager, dis-le explicitement plutôt que d'en choisir une.\n"
        "- Termine par la conclusion directe attendue par la question "
        "(oui / non / la valeur), après avoir présenté les données.\n"
        "- S'il y a un « Résumé structuré », ses effectifs et différences "
        "d'ensemble font foi : reproduis-les fidèlement, ne recompte pas "
        "les lignes brutes.\n\n"
        f"Question : {question}\n"
        f"{rows_block}"
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
