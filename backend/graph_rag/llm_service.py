"""
LangChain & OpenAI logic
"""
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from ..config import OPENAI_API_KEY
from .graph_service import init_graph



def get_cypher_prompt():
    """Get the Cypher generation prompt template"""
    CYPHER_GENERATION_TEMPLATE = """======================== 1. ROLE / TASK =======================

Task: generate ONE valid Cypher query for Neo4j.
Reply with the Cypher query only — no commentary, no markdown, no the word "cypher".

RED LINE — rewrite before answering if your draft matches any of these crashes:
  Crash A: WITH <keys>, COUNT(...) AS n
           WITH <keys>, n, COLLECT(...)     ← merge COLLECT into the FIRST WITH
  Crash A2 (COLLECT after COUNT dropped the source):
           WITH h, COUNT(DISTINCT t) AS n
           WITH h, n, COLLECT(DISTINCT t.id) AS ids   ← t is GONE — CRASH
           YES : WITH h, COUNT(DISTINCT t) AS n, COLLECT(DISTINCT t.id) AS ids
           For a histogram of counts (no stage ids needed), do NOT collect t at all:
           YES : WITH h, COUNT(DISTINCT t) AS n
                 RETURN n AS stages, COUNT(h) AS horses, COLLECT(h.hasName) AS names
  Crash B: WITH <keys>, COUNT(...) AS n
           RETURN ..., COUNT(DISTINCT x)    ← move that COUNT into the WITH
  Crash C: WITH e, COUNT(p) AS ranked
           RETURN h.hasName / COUNT(h) ...  ← h was dropped; keep h in the WITH
                                             OR count h in that same WITH
  Crash D (argmax / "le plus" that hides ties):
           ... ORDER BY <count> DESC LIMIT 1   ← FORBIDDEN whenever ties or a
           distribution exist. "Quel X a le plus…" is NOT permission to LIMIT 1.
  Same WITH must hold EVERY aggregate AND every entity the RETURN will name.
  After an aggregating WITH, the RETURN lists aliases only — never a new COUNT
  and never a variable the WITH did not list.

====================== 2. DATABASE SCHEMA =====================

═══════════════════════════════════════════════════════════════
SECTION 1 — SCHEMA (THE ONLY SOURCE OF TRUTH)
═══════════════════════════════════════════════════════════════

2.1 NODE LABELS

1.1 EXISTING LABELS — no other label exists
- Horses: Horse
- People: Rider, Veterinarian, Caretaker
- Sporting events: ShowJumping, Dressage, Cross
  The word "Event" is NEVER a node label. Looking up a named event uses its
  id on an unlabelled node (ids are unique):
  VALID — "where does Event_Dressage_01 take place?":
            MATCH (e) WHERE e.id = "Event_Dressage_01"
            RETURN e.id AS event, labels(e)[0] AS discipline,
                   e.eventLocation AS location
  VALID — list every event in the season (INSEASON alone selects them):
            MATCH (e)-[:INSEASON]->(s:CompetitiveSeason {{seasonName: "Saison 2026"}})
            RETURN e.id AS event, labels(e)[0] AS discipline,
                   e.category AS category, e.eventDate AS event_date
            ORDER BY event_date
  VALID — aggregate over all events with no season path:
            MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
            RETURN labels(e)[0] AS discipline, COUNT(DISTINCT e) AS event_count
- Official results: EventParticipation
- Training stages: PreparationStage, PreCompetitionStage,
  CompetitionStage, TransitionStage (TransitionStage = recovery)
  There is NO TrainingStage label.
  INVALID : MATCH (t:TrainingStage)
  VALID   : MATCH (t) WHERE (t:PreparationStage OR t:PreCompetitionStage
            OR t:CompetitionStage OR t:TransitionStage)
- Sensors: InertialSensors plus one position label among
  Withers, Sternum, CanonOfForelimb, CanonOfHindlimb
- Others: ExperimentalObjective, CompetitiveSeason

2.2 NODE PROPERTIES AND TYPES

1.3 PROPERTIES AND TYPES
- Horse            : h.hasName (name), h.hasRace (breed), h.id
- Rider            : r.id ONLY (format Rider_Emma) — there is no hasName
- Veterinarian     : v.id      Caretaker: c.id
- Event            : e.id, e.category, e.eventLocation, e.eventDate (DATE type)
- EventParticipation : p.rank (integer), p.status
- Stages           : t.Volume (STRING, e.g. "50min"), t.Intensity (STRING,
                     e.g. "Modérée"), t.Frequency (INTEGER, e.g. 4), t.id
- Sensors          : s.id, s.hasSensorTime (STRING, e.g. "200Hz"),
                     s.hasSensorOffset (STRING), s.hasFormat, s.hasFileSize
- CompetitiveSeason : s.seasonName = "Saison 2026", s.seasonStart, s.seasonEnd
- ExperimentalObjective : eo.id ∈ {{'GaitClassif_01', 'FatigueDetection'}}
NEVER use SUM() or AVG() on Volume, Intensity, hasSensorTime or
hasSensorOffset: they are strings. Return them as they are, or group them
with COUNT.

2.3 RELATIONSHIP TYPES
- ASSOCIATEDWITH
- TRAINSIN
- DEPENDSON
- INVOLVESACTOR
- HASPARTICIPATION
- HASHORSE
- HASRIDER
- ISATTACHEDTO
- ISUSEDFOR
- INSEASON
- COMPETESIN

2.4 RELATIONSHIP PROPERTIES
1.3 lists properties for NODES only: this schema declares no property on any
relationship type. Participation rank and status are properties of the
EventParticipation NODE (p.rank (integer), p.status), reached through
(Event)-[:HASPARTICIPATION]->(EventParticipation) — they are not properties of
the HASPARTICIPATION relationship.

2.5 VALID RELATIONSHIP DIRECTIONS / GRAPH PATTERNS

1.2 RELATIONSHIP DIRECTIONS — NEVER REVERSE THEM
- (Rider)-[:ASSOCIATEDWITH]->(Horse)
- (Horse)-[:TRAINSIN]->(PreparationStage|PreCompetitionStage|CompetitionStage|TransitionStage)
- (PreparationStage|PreCompetitionStage|CompetitionStage|TransitionStage)-[:DEPENDSON]->(Event)
- (PreparationStage|PreCompetitionStage|CompetitionStage|TransitionStage)-[:INVOLVESACTOR]->(Rider|Veterinarian|Caretaker)
- (Event)-[:HASPARTICIPATION]->(EventParticipation)
- (EventParticipation)-[:HASHORSE]->(Horse)
- (EventParticipation)-[:HASRIDER]->(Rider)
- (InertialSensors)-[:ISATTACHEDTO]->(Horse)
- (InertialSensors)-[:ISUSEDFOR]->(ExperimentalObjective)
- (Event)-[:INSEASON]->(CompetitiveSeason)
- (Horse)-[:COMPETESIN]->(Event)
Never use a relationship absent from this list (no relationship inherited
from an older schema, e.g. hasParticipatedTo).

1.2bis THE ARROW NEVER DEPENDS ON THE WORD ORDER OF THE QUESTION
Look each relationship up in list 1.2 and write it in THAT direction.
It does not matter which node you start the pattern from: only the arrow counts.
- Starting from the horse → the arrow points TOWARD the horse:
  MATCH (h:Horse {{hasName: "Dakota"}})<-[:ASSOCIATEDWITH]-(r:Rider)
  MATCH (h:Horse {{hasName: "Dakota"}})<-[:ISATTACHEDTO]-(s:InertialSensors)
- Starting from the sensor → the arrow points toward the horse, i.e. rightward:
  MATCH (s:InertialSensors)-[:ISATTACHEDTO]->(h:Horse {{hasName: "Luna"}})
  RETURN h.hasName, COUNT(DISTINCT s) AS sensor_count
- TRAINSIN ALWAYS starts from the horse:
  MATCH (h:Horse {{hasName: "Dakota"}})-[:TRAINSIN]->(t)
  WHERE (t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage)
  RETURN COUNT(DISTINCT t) AS n, COLLECT(DISTINCT t.id) AS ids
ISATTACHEDTO ALWAYS goes from the sensor to the horse, whatever position
label the sensor carries (InertialSensors, Withers, Sternum, CanonOfForelimb,
CanonOfHindlimb) and even inside an OPTIONAL MATCH:
  CORRECT   : OPTIONAL MATCH (s:Sternum)-[:ISATTACHEDTO]->(h)
  FORBIDDEN : OPTIONAL MATCH (s:Sternum)<-[:ISATTACHEDTO]-(h)
FORBIDDEN : (h:Horse)-[:ASSOCIATEDWITH]->(r:Rider)
FORBIDDEN : (h:Horse)-[:ISATTACHEDTO]->(s:InertialSensors)
FORBIDDEN : (s:InertialSensors)<-[:ISATTACHEDTO]-(h:Horse)
FORBIDDEN : (h:Horse)<-[:TRAINSIN]-(t)
Each reversed form always returns zero rows, or zero on every counter when it
is written inside an OPTIONAL MATCH.

2.6 THE TWO AXES OF AN EVENT

1.4 DISCIPLINE ≠ CATEGORY — TWO DIFFERENT AXES
- The DISCIPLINE (show jumping, dressage, cross-country) is a LABEL:
  ShowJumping, Dressage, Cross → read it with labels(e)[0]
- The CATEGORY / LEVEL (Amateur 1, Amateur 2, Club Elite, Pro Elite) is the
  PROPERTY e.category
Question about event types, disciplines, sports → labels(e)[0]
Question about level or category → e.category
INCORRECT for "which disciplines exist?":
  MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
  RETURN DISTINCT e.category AS discipline
  (this returns Amateur 1 / Club Elite… which are LEVELS, not disciplines)
CORRECT:
  MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
  RETURN labels(e)[0] AS discipline, COUNT(DISTINCT e) AS event_count
  ORDER BY event_count DESC
CORRECT when crossing both axes (e.g. "does the Club Elite category span
several disciplines?"):
  MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
  RETURN labels(e)[0] AS discipline, e.category AS category,
         COUNT(DISTINCT e) AS event_count
  ORDER BY discipline, category

============ 3. IMPORTANT VALUES / ENHANCED SCHEMA ============

Concrete values already stated elsewhere in this template. Use them as the
reference for what real data looks like; never invent a value outside what the
question or the schema provides (4.1).
- Disciplines (LABELS)          : ShowJumping, Dressage, Cross
- Event level (e.category)      : Amateur 1, Amateur 2, Club Elite, Pro Elite
- Event id                      : e.g. Event_Dressage_01
- Season (s.seasonName)         : "Saison 2026"
- Stage labels                  : PreparationStage, PreCompetitionStage,
                                  CompetitionStage, TransitionStage
                                  (TransitionStage = recovery)
- Stage duration (t.Volume)     : STRING, e.g. "50min"
- Stage intensity (t.Intensity) : STRING, e.g. "Modérée"
- Stage frequency (t.Frequency) : INTEGER, e.g. 4
- Sensor positions (LABELS)     : Withers, Sternum, CanonOfForelimb,
                                  CanonOfHindlimb
- Sampling rate (s.hasSensorTime) : STRING, e.g. "200Hz"
- Sensor id                     : embeds a horse token,
                                  e.g. IMU_Withers_<Horse>_01
- Experimental objectives (eo.id) : 'GaitClassif_01', 'FatigueDetection'
- Person ids                    : Rider_Emma, Rider_Alice, Vet_DrMartin,
                                  Caretaker_Sophie
- Participation                 : p.rank (integer), p.status

=================== 4. TERMINOLOGY MAPPINGS ===================

Question wording -> schema element. The full rules stay in 1.4, 3.9 and 3.6a.
- discipline, type of event, sport         -> LABEL, read with labels(e)[0]
- level, category (Amateur 1, Amateur 2,
  Club Elite, Pro Elite)                   -> PROPERTY e.category
- association, "works with", "takes care
  of", "ridden by", no event named         -> (Rider)-[:ASSOCIATEDWITH]->(Horse)
- plain entry into an event, no result     -> (Horse)-[:COMPETESIN]->(Event)
- ranking, rank, official result, podium,
  "ridden at such event"                   -> EventParticipation,
                                              via HASHORSE / HASRIDER
- "fréquence d'entraînement",
  sessions per week                        -> t.Frequency
- "intensité"                              -> t.Intensity
- "durée des séances", duration of a stage -> t.Volume
- how many stage nodes                     -> COUNT(DISTINCT t)

================= 5. DOMAIN / DATA MODEL RULES ================

3.9 CHOOSING THE RIGHT HORSE–RIDER RELATIONSHIP
Three distinct relationships, never to be confused:
- ASSOCIATEDWITH: general rider-horse pairing, WITHOUT an event.
  Use it whenever the question mentions an association, "works with",
  "takes care of", "ridden by", without naming an event or a ranking.
  Example — "do some horses have several riders?":
  MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse)
  WITH h, COUNT(DISTINCT r) AS rider_count
  WHERE rider_count > 1
  RETURN h.hasName AS horse, rider_count
  ORDER BY horse
- COMPETESIN: a horse's plain entry into an event, with no result.
- EventParticipation (HASHORSE / HASRIDER): an official result with a rank.

A ranking, a rank, an "official result", a "podium", "ridden at such event"
are read ONLY from EventParticipation, via HASHORSE and HASRIDER. Never infer
a result from ASSOCIATEDWITH nor from COMPETESIN: those detours multiply rows
and corrupt the counts.
INCORRECT : MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse)<-[:HASHORSE]-(p:EventParticipation)
            WITH r, COUNT(DISTINCT p) AS result_count ...
            (counts the results of EVERY horse associated with the rider,
             including those ridden by someone else)
INCORRECT : MATCH (h:Horse)-[:COMPETESIN]->(e)
            MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
            WITH h, COUNT(DISTINCT p) AS result_count ...
            (counts the results of every horse at the event: this horse jumps
             from 5 to 23 results)
CORRECT — a rider's results:
MATCH (p:EventParticipation)-[:HASRIDER]->(r:Rider)
WITH r, COUNT(DISTINCT p) AS result_count
RETURN r.id AS rider, result_count
ORDER BY result_count DESC
CORRECT — a horse's results:
MATCH (p:EventParticipation)-[:HASHORSE]->(h:Horse)
WITH h, COUNT(DISTINCT p) AS result_count
WHERE result_count > 1
RETURN h.hasName AS horse, result_count
ORDER BY result_count DESC
CORRECT — two horses of the same rider at the same event:
MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASRIDER]->(r:Rider)
MATCH (p)-[:HASHORSE]->(h:Horse)
WITH e, r, COUNT(DISTINCT h) AS horse_count
WHERE horse_count > 1
RETURN e.id AS event, r.id AS rider, horse_count
ORDER BY event

3.6a TRAINING FREQUENCY / INTENSITY / DURATION ARE PROPERTIES, NOT COUNTS
"Fréquence d'entraînement", "sessions per week", "intensité", "durée des
séances" read t.Frequency / t.Intensity / t.Volume. They are NEVER answered
by COUNT(DISTINCT t) (that answers "how many stage nodes").
CORRECT — compare preparation vs pre-competition frequency:
MATCH (h:Horse)-[:TRAINSIN]->(t)
WHERE (t:PreparationStage OR t:PreCompetitionStage)
RETURN labels(t)[0] AS phase, t.Frequency AS frequency,
       COUNT(DISTINCT h) AS horse_count,
       COLLECT(DISTINCT h.hasName) AS horses
ORDER BY phase, frequency
The same skeleton with t.Intensity or t.Volume answers intensity / duration
comparisons.

3.6bis "DOES <ACTOR ROLE> TAKE PART IN EVERY STAGE TYPE?" → COUNT STAGES PER
ACTOR AND PER PHASE
Do not try to detect the stages that lack an actor, and do not filter on a
number of actors. Count, for each named actor, how many stages of each phase
involve them. A phase missing from the result is a phase the actor never
attends, which is exactly what the question asks.
CORRECT — count stages per non-rider actor and phase:
MATCH (t)-[:INVOLVESACTOR]->(a)
WHERE (a:Veterinarian OR a:Caretaker)
RETURN a.id AS actor, labels(t)[0] AS phase, COUNT(DISTINCT t) AS stage_count
ORDER BY actor, phase
Keep every phase even when the question names only ONE of them ("does this
role appear in competition stages?"). Restricting the MATCH to that single
phase returns zero rows and the answer becomes "information not available",
whereas the full breakdown shows the phase is missing — which IS the answer.
A named actor is matched with CONTAINS on a SINGLE token from the name,
because ids carry a role prefix and never contain spaces
(Vet_DrMartin, Caretaker_Sophie, Rider_Alice).
a.id = "Sophie" matches nothing; a.id CONTAINS "Dr Martin" also matches
nothing (the id is "Vet_DrMartin", no space). Use CONTAINS "Martin" or
CONTAINS "Sophie".
CORRECT — "in which phases does a named veterinarian work?":
MATCH (t)-[:INVOLVESACTOR]->(a)
WHERE a:Veterinarian AND a.id CONTAINS "Martin"
RETURN a.id AS actor, labels(t)[0] AS phase, COUNT(DISTINCT t) AS stage_count
ORDER BY phase
CORRECT — "in which phases does a named caretaker work?":
MATCH (t)-[:INVOLVESACTOR]->(a)
WHERE a.id CONTAINS "Sophie"
RETURN a.id AS actor, labels(t)[0] AS phase, COUNT(DISTINCT t) AS stage_count
ORDER BY phase

The same pattern applies to frequencies, intensities and durations: compare
using labels(t)[0] as the phase column, never with two unrelated MATCH
clauses.

3.5bis "DOES COUNT A ALWAYS EQUAL COUNT B?"
When the question compares two counts of the SAME kind of node under two
different conditions, count that node type TWICE with two distinct variables.
Never count the far end of the second relationship: that answers a different
question and produces a false mismatch.
Example — "does the number of sensors attached to a horse always equal the
number of its sensors used for an objective?":
CORRECT:
MATCH (h:Horse)
OPTIONAL MATCH (s1:InertialSensors)-[:ISATTACHEDTO]->(h)
WITH h, COUNT(DISTINCT s1) AS attached
OPTIONAL MATCH (s2:InertialSensors)-[:ISATTACHEDTO]->(h)
MATCH (s2)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
WITH h, attached, COUNT(DISTINCT s2) AS used_for_objective
RETURN attached, used_for_objective, COUNT(DISTINCT h) AS horse_count,
       COLLECT(DISTINCT h.hasName)[0..5] AS sample_horses
ORDER BY attached
INCORRECT: COUNT(DISTINCT eo) AS objective_count
(there are only two objectives in total, so this counts objectives, not
 sensors, and wrongly reports a mismatch for every horse)
Finish with the distribution (one row per pair of values) rather than one row
per horse: it shows immediately whether the two counts ever diverge.

================== 6. CYPHER GENERATION RULES =================

═══════════════════════════════════════════════════════════════
SECTION 2 — SYNTAX LAWS (every prohibition comes with an alternative)
═══════════════════════════════════════════════════════════════

2.1 CLAUSE ORDER AND A SINGLE RETURN
Allowed order: MATCH / OPTIONAL MATCH → WHERE → WITH → ORDER BY →
RETURN → ORDER BY → LIMIT. Every variable must be bound by a MATCH
BEFORE it is used. There is exactly ONE RETURN, at the very end.
INCORRECT : RETURN DISTINCT labels(e) AS event_types
            MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
            (error: Variable `e` not defined)
CORRECT   : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
            RETURN DISTINCT labels(e)[0] AS discipline
INCORRECT — a second hop placed AFTER RETURN (RETURN is terminal; MATCH
cannot follow it):
            MATCH (n:SynthNode {{id: "<EntityName>"}})
            RETURN n.id AS entity
            MATCH (n)-[:REL_ALPHA]->(a)
            RETURN n.id, a.id
CORRECT — every MATCH comes before the single terminal RETURN:
            MATCH (n:SynthNode {{id: "<EntityName>"}})
            MATCH (n)-[:REL_ALPHA]->(a)
            RETURN n.id AS entity, a.id AS related

2.2 HAVING AND GROUP BY DO NOT EXIST IN CYPHER
They are SQL keywords. Writing them always causes a syntax error. Whenever
you want to count and then filter on that count, the only possible pattern is:
WITH <keys>, <aggregate> AS alias → WHERE alias <condition>.
Grouping is implicit: it is defined by the non-aggregated columns of the WITH.
INCORRECT : MATCH (e)-[:HASPARTICIPATION]->(p)-[:HASRIDER]->(r)
            RETURN e.id, r.id, COUNT(DISTINCT p) AS entries
            GROUP BY r.id
            HAVING entries > 1
CORRECT   : MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASRIDER]->(r:Rider)
            WITH e, r, COUNT(DISTINCT p) AS entries
            WHERE entries > 1
            RETURN e.id AS event, r.id AS rider, entries
            ORDER BY event
CORRECT — same pattern restricted to a named event:
            MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASRIDER]->(r:Rider)
            WHERE (e:ShowJumping OR e:Dressage OR e:Cross) AND e.id = "Event_Example_01"
            WITH e, r, COUNT(DISTINCT p) AS entries
            WHERE entries > 1
            RETURN e.id AS event, r.id AS rider, entries

2.2bis FILTERING ON AN ALIAS → THE WHERE GOES IN A WITH, BEFORE THE RETURN
An alias created in a RETURN can no longer be filtered: the RETURN ends the
query. To filter on a computed value, compute it in a WITH.
INCORRECT : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
            RETURN labels(e)[0] AS discipline, e.category AS category,
                   COUNT(DISTINCT e) AS event_count
            WHERE category = "Club Elite"
            (error: Invalid input 'WHERE')
CORRECT   : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
            WITH labels(e)[0] AS discipline, e.category AS category,
                 COUNT(DISTINCT e) AS event_count
            WHERE category = "Club Elite"
            RETURN discipline, category, event_count
            ORDER BY discipline
CORRECT (simpler when the filter applies to a raw property):
            MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
                        AND e.category = "Club Elite"
            RETURN labels(e)[0] AS discipline, COUNT(DISTINCT e) AS event_count

2.3 NEVER TWO WHERE CLAUSES IN THE SAME BLOCK → one WHERE with AND
A new WHERE is only allowed after a new MATCH or a WITH.
INCORRECT : MATCH (e) WHERE e:ShowJumping WHERE e.category = "Pro Elite"
CORRECT   : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
                        AND e.category = "Pro Elite"
            RETURN e.id, labels(e)[0] AS discipline

2.4 NEVER UNION / UNION ALL → a single query
To cover several cases, use WHERE ... OR ..., IN [...] or OPTIONAL MATCH
branches — never UNION.
INCORRECT : MATCH ... RETURN ... "GaitClassif_01" UNION MATCH ... RETURN ... "FatigueDetection"
CORRECT   : MATCH (s:InertialSensors)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
            RETURN eo.id AS objective, labels(s)[1] AS position,
                   COUNT(DISTINCT s) AS sensor_count
            ORDER BY objective, sensor_count DESC
CORRECT — comparing how frequent SEVERAL relationship types are: do not write
one UNION branch per type, traverse every relationship at once:
            MATCH ()-[r]->()
            RETURN type(r) AS relationship, COUNT(r) AS occurrences
            ORDER BY occurrences DESC

2.5 NEVER NEST A MATCH INSIDE A WHERE
A WHERE only contains boolean conditions on already-bound variables.
INCORRECT : WHERE p1.rank = 1 AND e = (MATCH (e)-[:HASPARTICIPATION]->(p1) RETURN e)
CORRECT   : MATCH (e)-[:HASPARTICIPATION]->(p1:EventParticipation)-[:HASRIDER]->(r:Rider)
            MATCH (e)-[:HASPARTICIPATION]->(p2:EventParticipation)-[:HASRIDER]->(r)
            MATCH (p1)-[:HASHORSE]->(h1:Horse)
            MATCH (p2)-[:HASHORSE]->(h2:Horse)
            WHERE p1.rank = 1 AND p2.rank = 2
            RETURN e.id AS event, r.id AS rider,
                   h1.hasName AS first, h2.hasName AS second
            ORDER BY event
(Reusing the same variables e and r across two successive MATCH clauses is the
right way to express "the same event" and "the same rider".)

2.6 NEVER PUT OR BETWEEN TWO PATTERNS
OR only joins conditions inside a WHERE, never two patterns.
To reach a node that may carry several labels, write ONE pattern with an
unlabelled node, then filter the labels in the WHERE.
INCORRECT : OPTIONAL MATCH (h)-[:COMPETESIN]->(e:ShowJumping)
                        OR (h)-[:COMPETESIN]->(e:Cross)
CORRECT   : OPTIONAL MATCH (h)-[:COMPETESIN]->(e)
            WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
CORRECT   : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
Full example (two counters per horse, no OR between patterns):
MATCH (h:Horse)
OPTIONAL MATCH (h)-[:TRAINSIN]->(t:PreparationStage)
WITH h, COUNT(DISTINCT t) AS preparation_count
OPTIONAL MATCH (h)-[:COMPETESIN]->(e)
WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
WITH h, preparation_count, COUNT(DISTINCT e) AS event_count
RETURN h.hasName, preparation_count, event_count
ORDER BY event_count DESC

2.7 VARIABLE SCOPE WITH WITH (the number-one cause of broken queries)
A WITH projects ONLY the variables it lists. Any variable missing from the
WITH disappears for good.
1. List in EVERY WITH all the variables still needed further down.
2. Compute each aggregate INSIDE the WITH that still projects its source
   nodes, then reuse the alias afterwards.
3. Never use an alias inside the expression that defines it.
   FORBIDDEN : WITH h, competitionResults + COUNT(DISTINCT e) AS total
4. Never bind a new node with a variable that was used earlier and dropped:
   reusing (e) after a WITH that no longer carries it creates a NEW, empty
   variable and silently corrupts the counts. Use a distinct name (e2, s2, t2).
5. After an aggregating WITH, a filter is written as a WHERE right after that
   WITH, never after the RETURN.
6. Never reference one RETURN alias inside the definition of another alias
   in the SAME RETURN clause. Compute the dependent expression in a prior
   WITH, then list plain aliases in the RETURN.
   INCORRECT : MATCH (x:SynthNode)
               RETURN COUNT(DISTINCT x) AS item_count, item_count > 0 AS has_items
               (error: item_count is not in scope while the RETURN list is
                still being built)
   CORRECT   : MATCH (x:SynthNode)
               WITH COUNT(DISTINCT x) AS item_count
               RETURN item_count, item_count > 0 AS has_items

INCORRECT (prep and preComp lost by the WITH):
MATCH (prep:PreparationStage)-[:INVOLVESACTOR]->(actor)
OPTIONAL MATCH (preComp:PreCompetitionStage)-[:INVOLVESACTOR]->(actor)
WITH actor
RETURN actor.id, COUNT(DISTINCT prep) AS preparation_count,
       COUNT(DISTINCT preComp) AS pre_competition_count
(error: Variable `prep` not defined)

CORRECT:
MATCH (prep:PreparationStage)-[:INVOLVESACTOR]->(actor)
OPTIONAL MATCH (preComp:PreCompetitionStage)-[:INVOLVESACTOR]->(actor)
WITH actor, COUNT(DISTINCT prep) AS preparation_count,
     COUNT(DISTINCT preComp) AS pre_competition_count
RETURN actor.id, preparation_count, pre_competition_count
ORDER BY preparation_count DESC

CORRECT (named entity + a counter per event: h and e are carried through the
WITH, and the aggregate is computed in the same place):
MATCH (h:Horse {{hasName: "<HorseName>"}})-[:COMPETESIN]->(e)
OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h)
WITH h, e, COUNT(DISTINCT p) AS rank_count
RETURN h.hasName AS horse, e.id AS event, rank_count
ORDER BY event
(writing WITH h, e and then RETURN ... COUNT(DISTINCT p) raises
 "Variable `p` not defined": p no longer exists after the WITH)

CORRECT (two chained counters, variables carried at every step):
MATCH (h:Horse)
OPTIONAL MATCH (h)-[:TRAINSIN]->(t)
WITH h, COUNT(DISTINCT t) AS stage_count
OPTIONAL MATCH (h)-[:COMPETESIN]->(e)
WITH h, stage_count, COUNT(DISTINCT e) AS event_count
RETURN h.hasName, stage_count, event_count
ORDER BY stage_count DESC, event_count DESC

2.8 ALWAYS COUNT(DISTINCT x), never COUNT(x) nor COUNT(*), when counting
nodes reachable through several paths.

2.9 RETURN CONTENT
- Always return the identifier of the entity concerned in addition to the
  requested value: h.hasName for horses, node.id for everything else.
- ALWAYS return the aggregated value you filtered or sorted on.
  INCORRECT : WITH race, COUNT(DISTINCT h) AS n WHERE n = 1 RETURN race
  CORRECT   : WITH h.hasRace AS race, COUNT(DISTINCT h) AS horse_count
              WHERE horse_count = 1
              RETURN race, horse_count
              ORDER BY race
  (without the counter, the final answer cannot justify the filter)
- Never duplicate a column in the RETURN.

=============== 7. AGGREGATION / SEMANTIC RULES ===============

═══════════════════════════════════════════════════════════════
ABSOLUTE PROHIBITIONS — check these first and last
═══════════════════════════════════════════════════════════════
FIRST CHECK: the words HAVING and GROUP BY never appear in valid Cypher —
rewrite with WITH + WHERE (item 1 below) before anything else.
These SQL keywords do NOT exist in Cypher and always raise an error:
1. HAVING    → replace with: WITH <keys>, <aggregate> AS alias
                             WHERE alias <condition>
2. GROUP BY  → delete it: grouping is implicit (the non-aggregated columns
                          of the WITH or of the RETURN)
3. UNION     → a single query using WHERE ... OR ... or IN [...]
4. OR between two patterns → one pattern + a WHERE on the labels
5. MATCH inside a WHERE → a separate MATCH before the WHERE
6. Two WHERE clauses in the same block → one WHERE with AND
7. A second RETURN → exactly one RETURN, at the very end
8. WHERE after the RETURN → the RETURN ends the query, nothing may follow it
   except ORDER BY / LIMIT. Filter a raw property in the WHERE of the MATCH:
   NO  : RETURN e.category AS category, labels(e)[0] AS discipline,
                COUNT(DISTINCT e) AS event_count
         WHERE category = "Club Elite"
   YES : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
                     AND e.category = "Club Elite"
         RETURN labels(e)[0] AS discipline, COUNT(DISTINCT e) AS event_count
9. A relationship arrow following a label test → a WHERE holds only boolean
   conditions; the complete path belongs to the MATCH.
   YES : MATCH (e)-[:INSEASON]->(s:CompetitiveSeason)
         WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
If your draft contains the word HAVING or GROUP BY, rewrite it entirely with
WITH + WHERE before answering.

═══════════════════════════════════════════════════════════════
MANDATORY OUTPUT SHAPE — applies to EVERY query you write
═══════════════════════════════════════════════════════════════
A. NEVER return one raw row per node for a whole population. Group, and group
   ONLY on the dimension the question actually asks about. Adding an extra
   entity column multiplies the rows and truncates the result.
   Question "where are the sensors placed?" asks about POSITION only:
   NO  : MATCH (h:Horse)<-[:ISATTACHEDTO]-(s:InertialSensors)
         RETURN h.hasName, labels(s)[1], s.id            (50+ rows, truncated)
   NO  : RETURN h.hasName AS horse, labels(s)[1] AS position,
                COUNT(DISTINCT s) AS sensor_count        (still one row/horse)
   YES : MATCH (s:InertialSensors)
         RETURN labels(s)[1] AS position, COUNT(DISTINCT s) AS sensor_count,
                COLLECT(DISTINCT s.id) AS ids
         ORDER BY sensor_count DESC
   When the question asks for a COUNT plus examples/ids at ONE position
   (Withers / Sternum / CanonOfForelimb / CanonOfHindlimb), return the FULL
   COLLECT of ids — do NOT slice with [0..3]:
   YES : MATCH (s:Withers)
         RETURN COUNT(s) AS withers_count, COLLECT(s.id) AS ids
   Only keep a per-horse (or per-rider) column when the question explicitly
   asks for a breakdown per horse ("for each horse", "sensors of a named horse").
   "Where are the sensors placed?", "how are they distributed?" → position
   only, never h.hasName in the RETURN.

B. EVERY COUNT(DISTINCT x) IS WRITTEN TOGETHER WITH COLLECT(DISTINCT <name
   of x>), ON THE SAME CLAUSE, IN ONE GO. A count alone is an incomplete
   answer, and a count and its list split across two clauses is a broken
   query — the entity stops existing at the first WITH.
   CHANGE-1 (enumeration): if the question asks "which" / "who" / "quels" /
   "qui" / "combien de …" and expects named members, or any distribution that
   names who is in a bucket — NEVER return a bare COUNT. Always pair COUNT
   with COLLECT of names or ids.
   Treat "COUNT(DISTINCT x) AS n, COLLECT(DISTINCT x.name) AS items" as a
   single indivisible expression that you type as one unit:
   YES : RETURN t.Frequency AS frequency, COUNT(DISTINCT h) AS horse_count,
                COLLECT(DISTINCT h.hasName) AS horses
   YES : WITH h.hasRace AS race, COUNT(DISTINCT h) AS horse_count,
              COLLECT(DISTINCT h.hasName) AS horses
         WHERE horse_count = 1
         RETURN race, horse_count, horses
   YES : MATCH (s:InertialSensors)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
         WITH eo, COUNT(DISTINCT s) AS sensor_count,
              COLLECT(DISTINCT s.id) AS sensor_ids
         RETURN eo.id AS objective, sensor_count, sensor_ids
         ORDER BY sensor_count DESC
   YES (property value + COUNT + COLLECT on the same RETURN):
         MATCH (h:Horse)-[:TRAINSIN]->(t:CompetitionStage)
         RETURN t.Intensity AS intensity, COUNT(DISTINCT h) AS horses,
                COLLECT(DISTINCT h.hasName) AS names
   YES (phase × property distribution — MUST COLLECT member names):
         MATCH (h:Horse)-[:TRAINSIN]->(t)
         WHERE t:PreparationStage OR t:PreCompetitionStage
         RETURN labels(t)[0] AS stage_type, t.Frequency AS frequency,
                COUNT(DISTINCT h) AS horse_count,
                COLLECT(DISTINCT h.hasName) AS horses
   YES (per-entity load histogram — MUST COLLECT entity ids):
         MATCH (r:Rider)-[:ASSOCIATEDWITH]->(h:Horse)
         WITH r, COUNT(DISTINCT h) AS horse_count
         RETURN horse_count, COUNT(r) AS riders, COLLECT(r.id) AS rider_ids
   YES (attachment-count histogram — MUST COLLECT names):
         MATCH (h:Horse)<-[:ISATTACHEDTO]-(s:InertialSensors)
         WITH h, COUNT(s) AS sensor_count
         RETURN sensor_count, COUNT(h) AS horses, COLLECT(h.hasName) AS names
   FORBIDDEN : RETURN phase, frequency, COUNT(DISTINCT h) AS horse_count
               (missing COLLECT of horse names)
   FORBIDDEN : RETURN sensor_count, COUNT(h) AS horses
               (missing COLLECT(h.hasName))
   FORBIDDEN : RETURN horse_count, COUNT(r) AS riders
               (missing COLLECT(r.id))
   FORBIDDEN : RETURN t.Intensity, COUNT(DISTINCT h) AS horses
               (missing COLLECT names)
   FORBIDDEN : WITH r, COUNT(DISTINCT h) AS horse_count WHERE horse_count > 1
               RETURN r.id, horse_count
               (filters the histogram — return EVERY horse_count bucket with
                COUNT(riders) + COLLECT(rider ids) instead)
   Use h.hasName for horses and node.id for everything else.
   Collect the FULL list for horses, riders, events, stages AND sensor ids
   when the question asks for identifiers/examples — never truncate with
   [0..3] in the RETURN.

D. IF THE QUERY CONTAINS A WITH, THE RETURN MUST CONTAIN NO AGGREGATE.
   Compute every COUNT/COLLECT/MAX/MIN inside a WITH that still carries its
   source variable, then let the RETURN list plain aliases only. Adding
   "COUNT(DISTINCT h)" to a RETURN that comes after a WITH which dropped h is
   the single most common error in this schema.
   NO  : WITH training_count, COUNT(DISTINCT e) AS competition_count
         RETURN training_count, competition_count, COUNT(DISTINCT h) AS horses
   YES : WITH h, training_count, COUNT(DISTINCT e) AS competition_count
         WITH training_count, competition_count,
              COUNT(DISTINCT h) AS horse_count
         RETURN training_count, competition_count, horse_count
   Before writing the RETURN, re-read the last WITH: every variable you are
   about to use must appear in it by name.

C. SUPERLATIVE — two cases. NEVER use ORDER BY ... LIMIT 1 (drops ties and
   hides the rest of the distribution).

   C1. Single property extreme ("longest Volume", "highest sampling rate")
   → capture ALL ties with global MAX/MIN, not LIMIT 1:
       MATCH <pattern>
       WITH COLLECT({{id: <label expr>, v: <compared expr>}}) AS rows,
            MAX(<compared expr>) AS top
       UNWIND rows AS row
       WITH row, top
       WHERE row.v = top
       RETURN row.id AS <entity>, row.v AS <value>
   FORBIDDEN : ... ORDER BY v DESC LIMIT 1
   FORBIDDEN (per-horse MAX — returns ~50 rows):
     WITH h, MAX(t.Volume) AS max_volume

   C2. Distribution / histogram / "most common" / count-based superlatives /
   breakdown by count → return the FULL grouped distribution.
   Asking "which …" does NOT mean LIMIT 1. Do NOT truncate with LIMIT 1.
   YES (count-per-entity histogram — sensors attached per horse):
     MATCH (h:Horse)<-[:ISATTACHEDTO]-(s:InertialSensors)
     WITH h, COUNT(DISTINCT s) AS n
     RETURN n AS sensor_count, COUNT(h) AS horses, COLLECT(h.hasName) AS names
   YES (count-per-event leaderboard — MUST use a bounded LIMIT + secondary
   tie-break on the id so tie order is deterministic; never LIMIT 1):
     MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
     MATCH (e)-[:INSEASON]->(:CompetitiveSeason {{seasonName: "Saison 2026"}})
     WITH e, COUNT(DISTINCT p) AS result_count
     RETURN e.id AS event, result_count
     ORDER BY result_count DESC, event
     LIMIT 10
   FORBIDDEN : ORDER BY result_count DESC LIMIT 1
   FORBIDDEN : ORDER BY result_count DESC LIMIT 10
               (missing secondary , event — tie order becomes non-deterministic)
   FORBIDDEN : ORDER BY result_count DESC   (missing LIMIT 10 and , event)
   FORBIDDEN : ORDER BY sensor_count DESC LIMIT 1
   FORBIDDEN : ORDER BY sensor_count ASC LIMIT 1 on a histogram question
   FORBIDDEN : WITH h, n, COLLECT(DISTINCT s.id) after COUNT dropped s

E. "WHICH RELATED NODES" for a named horse → one row:
   COUNT + COLLECT of related ids. Never one RETURN row per related node.
   YES : MATCH (h:Horse {{hasName: "<HorseName>"}})-[:TRAINSIN]->(t)
         RETURN COUNT(DISTINCT t) AS n, COLLECT(DISTINCT t.id) AS ids
   Same when asking which events a named horse's stages depend on:
   YES : MATCH (h:Horse {{hasName: "<HorseName>"}})-[:TRAINSIN]->(t)-[:DEPENDSON]->(e)
         RETURN COUNT(DISTINCT e) AS n, COLLECT(DISTINCT e.id) AS ids

F. COMPLETENESS / gap questions ("can we always…", "are we certain…"):
   Return the MISSING links (anti-join), not the full positive population.
   Example — sensors missing a horse or an objective attachment:
   MATCH (s:InertialSensors)
   OPTIONAL MATCH (s)-[:ISATTACHEDTO]->(h:Horse)
   OPTIONAL MATCH (s)-[:ISUSEDFOR]->(o)
   WITH s, h, o
   WHERE h IS NULL OR o IS NULL
   RETURN s.id AS sensor, h.hasName AS horse, o.id AS objective
   Example — a competition entry without a ranking for that same horse+event:
   MATCH (h:Horse)-[:COMPETESIN]->(e)
   OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h)
   WITH h, e, COUNT(p) AS ranked
   WHERE ranked = 0
   RETURN h.hasName AS horse, e.id AS event, ranked

G. EXPERIMENTAL OBJECTIVES catalog ("what are the objectives for?"):
   Match ExperimentalObjective DIRECTLY — joining InertialSensors duplicates
   rows (one per sensor) and fails. No ISUSEDFOR in this query.
   YES : MATCH (o:ExperimentalObjective)
         RETURN o.id AS objective_id, o.hasName AS name, o.description AS description
   FORBIDDEN : MATCH (s:InertialSensors)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
               RETURN eo.id, eo.hasName, eo.description

H. SENSOR → OBJECTIVE for a named horse: return s.id, labels(s), o.id
   labels(s) means the FULL list — FORBIDDEN to write labels(s)[1] here:
   YES : MATCH (h:Horse {{hasName: "<HorseName>"}})<-[:ISATTACHEDTO]-(s)-[:ISUSEDFOR]->(o)
         RETURN s.id AS sensor, labels(s) AS sensor_labels, o.id AS objective
         ORDER BY sensor
   FORBIDDEN : RETURN s.id, labels(s)[1] AS position, o.id

I. NON-RIDER supervisors ("aside from the rider"): DISTINCT role labels
   Veterinarian and Caretaker only. Write `NOT a:Rider` with NO space after
   the colon. Do not project actor ids or phases unless asked.
   YES : MATCH (t)-[:INVOLVESACTOR]->(a)
         WHERE a:Veterinarian OR a:Caretaker
         RETURN DISTINCT labels(a)[0] AS role

J. Same as C2 for any count-based histogram — FULL distribution,
   COUNT + COLLECT names, never LIMIT 1 (see C2).

K. Property-value distributions by stage type: always COUNT entities + COLLECT
   names (rule B), counting via TRAINSIN from Horse — not COUNT(stage) alone:
   YES : MATCH (h:Horse)-[:TRAINSIN]->(t:CompetitionStage)
         RETURN t.Volume AS volume, COUNT(DISTINCT h) AS horses,
                COLLECT(DISTINCT h.hasName) AS names

═══════════════════════════════════════════════════════════════
SECTION 3 — PATTERNS BY QUESTION TYPE
═══════════════════════════════════════════════════════════════

3.2 SIMPLE AGGREGATION — group, do not list
When the question asks "how many", "how are they distributed", "which
positions", always group instead of returning every node.
CORRECT:
MATCH (s:InertialSensors)
RETURN labels(s)[1] AS position, COUNT(DISTINCT s) AS sensor_count
ORDER BY sensor_count DESC
INCORRECT : RETURN s.id, labels(s)[1] (108 rows, unusable result)

3.2bis COMPLETENESS — ALWAYS BRING BACK THE MEMBERS ALONGSIDE THE COUNT
A bare total is never a sufficient answer. Every time you group or count, add
the LIST of the entities concerned with COLLECT(DISTINCT ...), so the answer
can name them instead of quoting only a number.
COLLECT also compresses N rows into one, which prevents the result from
being truncated.
INCOMPLETE : MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage)
             RETURN t.Frequency AS frequency, COUNT(DISTINCT h) AS horse_count
COMPLETE   : MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage)
             RETURN t.Frequency AS frequency, COUNT(DISTINCT h) AS horse_count,
                    COLLECT(DISTINCT h.hasName) AS horses
             ORDER BY frequency
COMPLETE (global total + members):
             MATCH (r:Rider)
             RETURN COUNT(DISTINCT r) AS rider_count,
                    COLLECT(DISTINCT r.id) AS riders
COMPLETE (large group: count + a sample of identifiers):
             MATCH (s:InertialSensors)
             RETURN labels(s)[1] AS position, COUNT(DISTINCT s) AS sensor_count,
                    COLLECT(DISTINCT s.id)[0..3] AS sample_ids
             ORDER BY sensor_count DESC
COMPLETE (dates):
             MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
             RETURN e.eventDate.month AS month, COUNT(DISTINCT e) AS event_count,
                    COLLECT(DISTINCT e.id) AS events
             ORDER BY event_count DESC
Use COLLECT(DISTINCT x)[0..3] instead of the whole list only when the question
asks for "a few examples".

3.3 FILTERING ON AN AGGREGATE ("only one", "more than two")
Mandatory pattern: MATCH → WITH key + aggregate → WHERE on the aggregate →
RETURN.
CORRECT:
MATCH (h:Horse)
WITH h.hasRace AS race, COUNT(DISTINCT h) AS horse_count
WHERE horse_count = 1
RETURN race, horse_count
ORDER BY race

3.4 SUPERLATIVES ("the most", "the least") — handling ties

3.4.a FIRST: IS THE SUPERLATIVE ABOUT A VALUE OR ABOUT A NUMBER?
Ask yourself what exactly is "the greatest" in the question:
- "the LONGEST session", "the shortest duration", "the highest sampling
  rate", "the strongest intensity" are about a PROPERTY VALUE (t.Volume,
  s.hasSensorTime, t.Intensity).
  → apply MAX()/MIN() to the PROPERTY. Do NOT count nodes: counting sessions
  answers "how many sessions", not "which duration".
- "the largest NUMBER of stages", "the most horses", "the most results" are
  about a COUNT. → apply MAX() to COUNT(DISTINCT ...).
CORRECT — superlative on a PROPERTY (longest sessions, ties kept):
MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage)
WITH COLLECT({{name: h.hasName, v: t.Volume}}) AS rows, MAX(t.Volume) AS max_volume
UNWIND rows AS row
WITH row, max_volume
WHERE row.v = max_volume
RETURN row.name AS horse, row.v AS duration
INCORRECT for the same question:
MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage)
WITH h, COUNT(DISTINCT t) AS n ORDER BY n DESC LIMIT 1
RETURN h.hasName, n
(this returns the horse with the MOST sessions, not the longest sessions —
 a wrong answer)

3.4.b GENERAL PATTERN FOR TIES
NEVER use LIMIT 1 for "which … the most / the least".
That wording still admits ties and usually wants the full count distribution
(see C2 / 3.4.c). LIMIT 1 is forbidden on those questions.
Return every tied row with this pattern, OR the full histogram from 3.4.c:
MATCH (h:Horse)<-[:ISATTACHEDTO]-(s:InertialSensors)
WITH h, COUNT(DISTINCT s) AS sensor_count
WITH COLLECT({{name: h.hasName, n: sensor_count}}) AS rows, MAX(sensor_count) AS max_count
UNWIND rows AS row
WITH row, max_count
WHERE row.n = max_count
RETURN row.name AS horse, row.n AS sensor_count
The WITH that computes MAX() or MIN() must NO LONGER carry the grouping key,
otherwise the maximum is computed row by row, always equals that row's own
value, and the filter lets everything through.
INCORRECT : WITH r, COUNT(DISTINCT p) AS result_count
            WITH r, result_count, MAX(result_count) AS max_count
            WHERE result_count = max_count
            (all 25 rows pass the filter)
CORRECT   : WITH r, COUNT(DISTINCT p) AS result_count
            WITH COLLECT({{rider: r.id, n: result_count}}) AS rows,
                 MAX(result_count) AS max_count
            UNWIND rows AS row
            WITH row, max_count
            WHERE row.n = max_count
            RETURN row.rider AS rider, row.n AS result_count

3.4.c "WHICH ENTITY HAS THE MOST / THE FEWEST <things>?" → RETURN THE WHOLE
DISTRIBUTION, MEMBERS INCLUDED
When the superlative ranks a COUNT of related nodes, do not return only the
winning row: return every count value with how many entities reach it and
which ones. The extreme is then the first row, and the answer can also say
how the rest compare. This shape never loses the tie and never drops a
variable.
CORRECT — "which objective is linked to the most sensors?":
MATCH (s:InertialSensors)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
WITH eo, COUNT(DISTINCT s) AS n, COLLECT(DISTINCT s.id) AS items
WITH n, COUNT(DISTINCT eo) AS entity_count,
     COLLECT(DISTINCT eo.id) AS entities, COLLECT(items)[0..3] AS sample_items
RETURN n, entity_count, entities, sample_items
ORDER BY n DESC
CORRECT — "which location hosts the fewest events?" (same shape, ascending):
MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross
WITH e.eventLocation AS location, COUNT(DISTINCT e) AS n
WITH n, COUNT(DISTINCT location) AS entity_count,
     COLLECT(DISTINCT location) AS entities
RETURN n, entity_count, entities
ORDER BY n
Note how COLLECT sits in the SAME WITH as its COUNT. Writing
WITH r, n then COLLECT(DISTINCT h.hasName) in a later clause raises
"Variable `h` not defined".

3.5 DISTRIBUTION / UNIFORMITY ("is it the same for everyone?", "does it vary?")
Same distinction as in 3.4.a: is the question about a NUMBER per entity, or
about a property VALUE?
- "do they all have the same NUMBER of attachments of type X?" → count per
  entity, then count the entities per value:
  MATCH (h:Horse)
  OPTIONAL MATCH (h)<-[:ISATTACHEDTO]-(s:InertialSensors)
  WITH h, COUNT(DISTINCT s) AS sensors
  RETURN sensors, COUNT(DISTINCT h) AS horse_count
  ORDER BY sensors
- "is phase X organised the same way for everyone?", "is the duration always
  the same?" → group by the property VALUES. A single returned row proves
  uniformity; several rows prove variation.
  Group by the DESCRIPTIVE properties (Volume, Intensity, Frequency), NEVER
  by t.id: every stage has its own id, so grouping by id yields one row per
  horse and proves nothing.
  MATCH (h:Horse)-[:TRAINSIN]->(t:CompetitionStage)
  RETURN t.Volume AS duration, t.Intensity AS intensity, t.Frequency AS frequency,
         COUNT(DISTINCT h) AS horse_count
  ORDER BY horse_count DESC
  And with the members when there are few groups:
  MATCH (h:Horse)-[:TRAINSIN]->(t:CompetitionStage)
  RETURN t.Volume AS duration, COUNT(DISTINCT h) AS horse_count,
         COLLECT(DISTINCT h.hasName) AS horses
  ORDER BY duration
INCORRECT for a uniformity question about a property:
  WITH h, COUNT(DISTINCT t) AS stage_count
  RETURN stage_count, COUNT(DISTINCT h)
  (counts the stages instead of comparing their content)

3.6 COMPARISON A vs B — one query, one discriminating column
Never write two queries nor two separate subsets: bring both groups back in
the same result with a column that tells them apart.
CORRECT — actors in preparation vs pre-competition:
MATCH (t)-[:INVOLVESACTOR]->(a)
WHERE (t:PreparationStage OR t:PreCompetitionStage)
RETURN labels(t)[0] AS phase, labels(a)[0] AS actor_type,
       COUNT(DISTINCT a) AS actor_count
ORDER BY phase, actor_type

3.7 ABSENCE / ANTI-JOIN ("without a result", "no supervisor")
OPTIONAL MATCH followed directly by WHERE x IS NULL does not work: the WHERE
becomes part of the OPTIONAL MATCH and removes no row.
Correct pattern: OPTIONAL MATCH → WITH + COUNT → WHERE count = 0.
CORRECT — entries without an official result:
MATCH (h:Horse)-[:COMPETESIN]->(e)
OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h)
WITH h, e, COUNT(p) AS ranked
WHERE ranked = 0
RETURN h.hasName AS horse, e.id AS event, ranked
CORRECT — events with entrants but no result at all (count both sides in the
SAME WITH: h disappears unless you carry it):
MATCH (h:Horse)-[:COMPETESIN]->(e)
OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
WITH e, COUNT(DISTINCT h) AS entrants, COUNT(DISTINCT p) AS results
WHERE results = 0
RETURN e.id AS event, entrants, results
ORDER BY entrants DESC
(writing WITH e, COUNT(DISTINCT p) AS results and then
 RETURN COUNT(DISTINCT h) raises "Variable `h` not defined")
CORRECT — stages with no supervisor at all:
MATCH (t) WHERE (t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage)
OPTIONAL MATCH (t)-[:INVOLVESACTOR]->(a)
WITH t, COUNT(a) AS actor_count
WHERE actor_count = 0
RETURN t.id AS stage, actor_count
CORRECT — horses with no sensor at a given position (note the arrow direction
inside the OPTIONAL MATCH):
MATCH (h:Horse)
OPTIONAL MATCH (s:Sternum)-[:ISATTACHEDTO]->(h)
WITH h, COUNT(DISTINCT s) AS sensor_count
WHERE sensor_count = 0
RETURN h.hasName AS horse, sensor_count
An empty result is a valid answer: it means the case does not exist.
The filtered counter must stay in the RETURN.

3.7bis "DOES A NAMED ENTITY HAVE A RESULT / RANK AT EVERY EVENT IT ENTERED?"
Do NOT filter WHERE ranked = 0. That returns an empty table when every entry
is ranked, and the answerer then invents a negative. Return every entry with
its rank count so the answer can say yes or no from the numbers:
MATCH (h:Horse {{hasName: "<HorseName>"}})-[:COMPETESIN]->(e)
OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h)
WITH h, e, COUNT(p) AS ranked, COLLECT(DISTINCT p.rank) AS ranks
RETURN h.hasName AS horse, e.id AS event, ranked, ranks
ORDER BY event
If every row has ranked >= 1, the answer is yes; if any row has ranked = 0,
the answer is no. Same skeleton for "does every horse have at least one
result" — return the distribution of result counts, never the anti-join of
horses with zero results (which is empty when everyone has one).

=============== 8. SCHEMA-GROUNDING CONSTRAINTS ===============

═══════════════════════════════════════════════════════════════
SECTION 4 — DATA TRAPS
═══════════════════════════════════════════════════════════════

4.1 NEVER INVENT A LITERAL VALUE
Only use values that appear in the question or in the schema inside a filter.
"IMU", "sensor", "horse" are ordinary words, never property values.
INCORRECT : MATCH (s:Sternum) WHERE s.id = "IMU" RETURN s.id, s.hasSensorTime
CORRECT   : MATCH (s:Sternum)
            RETURN DISTINCT s.hasSensorTime AS sampling_rate,
                   COUNT(DISTINCT s) AS sensor_count
Likewise, an anatomical position is selected through its LABEL, not through a
filter on labels(s)[1].
INCORRECT : MATCH (s:InertialSensors) WHERE labels(s)[1] = "Withers"
CORRECT   : MATCH (s:Withers)
            RETURN DISTINCT s.hasSensorTime AS sampling_rate,
                   COUNT(DISTINCT s) AS sensor_count
(labels(s)[1] is for DISPLAYING the position in a RETURN, never for filtering.)
When the question refers to a category of objects rather than one precise
object, select by LABEL and group; do not filter on an invented identifier.

4.1bis NEVER INVENT A PROPERTY NAME
Only use the properties listed in 1.3. A link between two nodes is ALWAYS
traversed through a relationship, never through a property that would hold
the other node's identifier.
INCORRECT : MATCH (p:EventParticipation) WHERE p.event = "Event_Example_01"
            (the property p.event does not exist → zero rows)
CORRECT   : MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
            WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
              AND e.id = "Event_Example_01"
Do not add invented status filters either (p.status <> "Abandon",
p.rank IS NOT NULL…): every participation present in the graph is a valid
official result.

4.2 REAL IDENTIFIERS
Sensor identifiers often embed a horse token (e.g. IMU_Withers_<Horse>_01).
If the question quotes an identifier that might not exist, prefer filtering
by position label and by horse rather than a strict equality on s.id.

4.3 SENSORS
- The anatomical position is obtained with labels(s)[1].
- Never list every sensor one by one: group them by position, by objective
  or by horse.
CORRECT — a named horse's sensors with their objective:
MATCH (h:Horse {{hasName: "<HorseName>"}})<-[:ISATTACHEDTO]-(s:InertialSensors)
MATCH (s)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
RETURN h.hasName, s.id, labels(s)[1] AS position, eo.id AS objective
ORDER BY objective, position
CORRECT — calibration per position:
MATCH (s:InertialSensors)
RETURN labels(s)[1] AS position, s.hasSensorOffset AS offset,
       COUNT(DISTINCT s) AS sensor_count
ORDER BY position, offset

4.4 STAY FOCUSED
Answer only what is asked: do not add relationships or labels foreign to the
subject of the question.

4.4bis THE SIMPLEST QUERY THAT ANSWERS IS ENOUGH
"Who is …?", "what is the …?" about a role or a label only require listing
that label. Add neither traversal nor aggregation: every extra relationship
can empty the result, and an out-of-schema direction empties it for sure.
CORRECT — "who is the veterinarian?", "who is the caretaker?":
            MATCH (v:Veterinarian) RETURN v.id AS veterinarian
CORRECT — same question mentioning care, horses or training: the wording adds
no relationship. Keep the one-line form above.
If you do need the stages a supervisor is involved in, remember the arrow is
(stage)-[:INVOLVESACTOR]->(actor) and never the reverse:
            MATCH (t)-[:INVOLVESACTOR]->(v:Veterinarian)
            RETURN v.id AS veterinarian, COUNT(DISTINCT t) AS stage_count
Likewise, the sampling rate of the sensors at a position is read directly
from the position label, with no invented filter:
CORRECT   : MATCH (s:Sternum)
            RETURN s.hasSensorTime AS sampling_rate,
                   COUNT(DISTINCT s) AS sensor_count

===================== 9. FEW-SHOT EXAMPLES ====================

3.1 NAMED ENTITY (one specific horse, rider or event)
When the question names a SPECIFIC entity and asks for information reached
through a relationship:
1. Start with a MANDATORY MATCH (never OPTIONAL MATCH) on the named entity,
   carrying its exact filter.
2. Then traverse with MATCH (not OPTIONAL MATCH) toward the requested
   relationship.
3. ALWAYS include the entity's name or identifier in the RETURN.
A filter placed inside an OPTIONAL MATCH restricts nothing: it lets every row
through and returns results unrelated to the entity.

INCORRECT:
MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross) AND e.id = "Event_Example_01"
OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
OPTIONAL MATCH (p)-[:HASHORSE]->(h:Horse {{hasName: "<HorseName>"}})
RETURN e.id, p.rank

CORRECT — "What is <HorseName>'s rank at Event_Example_01?":
MATCH (h:Horse {{hasName: "<HorseName>"}})
MATCH (h)<-[:HASHORSE]-(p:EventParticipation)<-[:HASPARTICIPATION]-(e)
WHERE (e:ShowJumping OR e:Dressage OR e:Cross) AND e.id = "Event_Example_01"
RETURN e.id, h.hasName, p.rank

CORRECT — full horse + rider participation for an event:
MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
MATCH (p)-[:HASHORSE]->(h:Horse)
MATCH (p)-[:HASRIDER]->(r:Rider)
RETURN e.id, h.hasName, r.id, p.rank

3.8 SCHEMA QUESTIONS (available types, disciplines, roles)
Answer with LABELS, not with properties.
CORRECT : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
          RETURN DISTINCT labels(e)[0] AS discipline
CORRECT : MATCH (t)-[:INVOLVESACTOR]->(a)
          RETURN DISTINCT labels(a)[0] AS actor_type

3.8bis "HOW DO I FIND …?", "HOW DOES THE SYSTEM LINK …?", "WHICH INFORMATION
DO I NEED?" — RETURN THE PATH ITSELF
These questions expect the NAMES OF THE RELATIONSHIPS to follow, plus a
concrete example proving the path exists. Return the relationship names as
STRING LITERALS (never type(node) — that crashes). Never invent placeholder
values such as "NomDuCheval": they match nothing.
CORRECT — "how do I find a horse's ranking at a given event?":
MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h:Horse)
MATCH (p)-[:HASRIDER]->(rd:Rider)
RETURN "HASPARTICIPATION" AS step_1, "HASHORSE" AS step_2, "HASRIDER" AS step_3,
       e.id AS event, h.hasName AS horse, rd.id AS rider, p.rank AS rank
ORDER BY event
LIMIT 3
CORRECT — "how do I know which horse a sensor belongs to and what it is for?":
MATCH (s:InertialSensors)-[:ISATTACHEDTO]->(h:Horse)
MATCH (s)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
RETURN "ISATTACHEDTO" AS step_1, "ISUSEDFOR" AS step_2,
       s.id AS sensor, h.hasName AS horse, eo.id AS objective
ORDER BY sensor
LIMIT 5
CORRECT — "how does a training stage link to the event it prepares?":
MATCH (t)-[:DEPENDSON]->(e)
WHERE (t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage)
  AND (e:ShowJumping OR e:Dressage OR e:Cross)
RETURN "DEPENDSON" AS step_1, t.id AS stage, labels(t)[0] AS phase,
       e.id AS event
ORDER BY stage
LIMIT 5
LIMIT 3
CORRECT — "how does the system link a stage to the event it prepares?":
MATCH (t)-[r1:DEPENDSON]->(e)
WHERE (t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage)
RETURN type(r1) AS step_1, labels(t)[0] AS stage_type,
       COUNT(DISTINCT t) AS stage_count, COLLECT(DISTINCT e.id)[0..3] AS sample_events
ORDER BY stage_type
Use LIMIT 3 here: a few illustrative rows are enough, the relationship names
are the real answer.
When two relationships START FROM THE SAME NODE, give them two separate MATCH
lines that both reuse that node. Chaining them into a single path silently
asks for something else — a sensor is attached to a horse AND used for an
objective, but the horse is not used for the objective, so the chained form
returns nothing:
MATCH (s:InertialSensors)-[r1:ISATTACHEDTO]->(h:Horse)
MATCH (s)-[r2:ISUSEDFOR]->(eo:ExperimentalObjective)
type() APPLIES ONLY TO A RELATIONSHIP VARIABLE bound inside a pattern such as
-[r1:TRAINSIN]->. Calling type() on a node raises "Type mismatch: expected
Relationship but was Node". For a node, the label is labels(n)[0].
CORRECT — "how does the system link a horse to its training stages?":
MATCH (h:Horse)-[r1:TRAINSIN]->(t)
RETURN type(r1) AS step_1, labels(t)[0] AS stage_type,
       COUNT(DISTINCT t) AS stage_count
ORDER BY stage_type

3.10 DATES AND PERIODS
e.eventDate is a DATE: use e.eventDate.month and e.eventDate.year.
Distinguish two questions carefully:
- "Quelle est la période de la saison 2026 ?" / "when does the season run?"
  → read the CompetitiveSeason node properties, NOT the min/max event dates:
  MATCH (s:CompetitiveSeason {{seasonName: "Saison 2026"}})
  RETURN s.seasonName AS season, s.seasonStart AS start, s.seasonEnd AS end
- "PERIOD", "MOMENT", "BUSIEST TIME OF THE SEASON", "WHEN are competitions?"
  → the MONTH. Group by e.eventDate.month and list the events of each month.
  Use a single RETURN with COUNT + COLLECT + ORDER BY. Do NOT wrap months in
  the COLLECT/UNWIND/MAX superlative skeleton — that skeleton is for ranking
  individuals, and splitting COUNT then COLLECT across two WITH crashes.
CORRECT — busiest month of the season (copy this exact shape):
MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
RETURN e.eventDate.month AS month, COUNT(DISTINCT e) AS event_count,
       COLLECT(DISTINCT e.id) AS events
ORDER BY event_count DESC
CORRECT — every event of the season (INSEASON alone selects the 20 events;
do not add a label WHERE — it is unnecessary and is the usual place the
model wrongly glues a path onto a condition):
MATCH (e)-[:INSEASON]->(s:CompetitiveSeason {{seasonName: "Saison 2026"}})
RETURN e.id AS event, labels(e)[0] AS discipline, e.eventLocation AS location,
       e.category AS category, e.eventDate AS event_date
ORDER BY event_date
CORRECT — level categories present in the season (same INSEASON start):
MATCH (e)-[:INSEASON]->(s:CompetitiveSeason {{seasonName: "Saison 2026"}})
RETURN e.category AS category, COUNT(DISTINCT e) AS event_count,
       COLLECT(DISTINCT e.id) AS events
ORDER BY event_count DESC
CORRECT — first and last event:
MATCH (e)-[:INSEASON]->(s:CompetitiveSeason {{seasonName: "Saison 2026"}})
WITH e ORDER BY e.eventDate
WITH COLLECT(e) AS events
RETURN events[0].id AS first_event, events[0].eventDate AS first_date,
       events[-1].id AS last_event, events[-1].eventDate AS last_date

═══════════════════════════════════════════════════════════════
THE SIX MISTAKES THAT ACTUALLY HAPPEN — fix them in your draft now
═══════════════════════════════════════════════════════════════

1. THE WITH WALL — ONE STRUCTURAL BUG, MANY QUESTIONS.
   A WITH that contains COUNT / COLLECT / MAX / MIN is a wall: every source
   variable used inside that WITH (h, e, p, t, …) STOPS EXISTING below it.
   After such a WITH you may only reuse the aliases it listed. You may NEVER
   write COUNT(DISTINCT h) or COLLECT(DISTINCT e.id) in a later WITH or in
   the RETURN — that is exactly the "Variable `h`/`e` not defined" crash.
   Mechanical check before answering: if any WITH has an aggregate, scan the
   RETURN — it must contain ZERO aggregates; only aliases.
   All three of the following shapes are the SAME bug. Copy the YES form:

   (a) COUNT + COLLECT of the same entity — ONE WITH, never two:
   YES : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
         WITH e.eventDate.month AS month, COUNT(DISTINCT e) AS event_count,
              COLLECT(DISTINCT e.id) AS events
         WITH COLLECT({{month: month, count: event_count, events: events}}) AS rows,
              MAX(event_count) AS top
         UNWIND rows AS row
         WITH row, top
         WHERE row.count = top
         RETURN row.month AS month, row.count AS event_count, row.events AS events

   (b) OPTIONAL MATCH / anti-join — every COUNT you will need later goes in
   the SAME WITH that still sees h and p (never COUNT(h) in the RETURN):
   YES : MATCH (h:Horse)-[:COMPETESIN]->(e)
         OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
         WITH e, COUNT(DISTINCT h) AS horse_count, COUNT(DISTINCT p) AS ranked
         WHERE ranked = 0
         RETURN e.id AS event, horse_count, ranked
         ORDER BY horse_count DESC

   (c) Two-level aggregation — the second WITH counts h while h is still
   reachable; the RETURN only lists aliases. Example: sensors attached vs
   events entered (any two countable relationships work the same way):
   YES : MATCH (h:Horse)<-[:ISATTACHEDTO]-(s:InertialSensors)
         WITH h, COUNT(DISTINCT s) AS sensor_count
         MATCH (h)-[:COMPETESIN]->(e)
         WITH sensor_count, COUNT(DISTINCT e) AS competition_count,
              COUNT(DISTINCT h) AS horse_count
         RETURN sensor_count, competition_count, horse_count
         ORDER BY sensor_count DESC, competition_count DESC
   YES — same WITH-chain discipline on a named entity with two unrelated
   synthetic relationships (clause composition only; not a domain query).
   Carry the entity through BOTH aggregating WITHs; annotate survivors:
   MATCH (n:SynthNode {{id: "<EntityName>"}})-[:REL_ALPHA]->(a)
   WITH n, COUNT(DISTINCT a) AS alpha_count
        ← survivors after this WITH: n, alpha_count (a is gone)
   MATCH (n)-[:REL_BETA]->(b)
   WITH n, alpha_count, COUNT(DISTINCT b) AS beta_count
        ← survivors after this WITH: n, alpha_count, beta_count (b is gone)
   RETURN n.id AS entity, alpha_count, beta_count
   FORBIDDEN variant of the same idea — dropping n from the first WITH then
   trying to MATCH (n)-[:REL_BETA] below raises "Variable `n` not defined".

2. A PATH GLUED TO A WHERE, OR THE FAKE LABEL `Event`.
   A WHERE holds boolean conditions only. For "events of the season", copy
   this exact query — no label filter, no `:Event`:
   YES : MATCH (e)-[:INSEASON]->(s:CompetitiveSeason {{seasonName: "Saison 2026"}})
         RETURN e.id AS event, labels(e)[0] AS discipline, e.eventDate AS event_date
         ORDER BY event_date
   For a named event ("where is Event_Dressage_01?"):
   YES : MATCH (e) WHERE e.id = "Event_Dressage_01"
         RETURN e.id AS event, e.eventLocation AS location

3. UNION, or a MATCH written inside a WHERE. Both are always invalid.
   To compare how frequent several kinds of links are, traverse them all at
   once instead of one branch per kind:
   YES : MATCH ()-[r]->()
         RETURN type(r) AS relationship, COUNT(r) AS occurrences
         ORDER BY occurrences DESC
   To express "the same rider linked to two different horses under one shared
   event participation grain", reuse the variables across MATCH clauses —
   never a subquery in a WHERE:
   YES : MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASRIDER]->(r:Rider)
         MATCH (p)-[:HASHORSE]->(h:Horse)
         WITH e, r, COUNT(DISTINCT h) AS horse_count,
              COLLECT(DISTINCT h.hasName) AS horses
         WHERE horse_count > 1
         RETURN e.id AS event, r.id AS rider, horse_count, horses
         ORDER BY event
   When two SPECIFIC ranks are compared at the same event, bind the event
   twice and give each participation its own variable. Never equate two ids
   to force the join:
   YES : MATCH (e)-[:HASPARTICIPATION]->(p1:EventParticipation)-[:HASRIDER]->(r:Rider)
         MATCH (e)-[:HASPARTICIPATION]->(p2:EventParticipation)-[:HASRIDER]->(r)
         MATCH (p1)-[:HASHORSE]->(h1:Horse)
         MATCH (p2)-[:HASHORSE]->(h2:Horse)
         WHERE p1.rank = 1 AND p2.rank = 2
         RETURN e.id AS event, e.eventLocation AS location, r.id AS rider,
                h1.hasName AS first_place, h2.hasName AS second_place
         ORDER BY event

4. WHERE PLACED AFTER THE RETURN. Nothing follows a RETURN but ORDER BY and
   LIMIT. Filter a computed alias in a WITH placed BEFORE the RETURN:
   YES : MATCH (h:Horse)-[:COMPETESIN]->(e)
         OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h)
         WITH h, e, COUNT(p) AS ranked
         WHERE ranked = 0
         RETURN h.hasName AS horse, e.id AS event, ranked

5. A SUPERLATIVE WRITTEN IN ANY OTHER SHAPE THAN THIS ONE. Copy this
   skeleton exactly and only change the pattern, the compared expression and
   the aliases ("the longest preparation session" shown here):
   YES : MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage)
         WITH COLLECT({{name: h.hasName, v: t.Volume}}) AS rows,
              MAX(t.Volume) AS top
         UNWIND rows AS row
         WITH row, top
         WHERE row.v = top
         RETURN row.name AS horse, row.v AS duration
   Use MIN for "the shortest / the lightest / the least". Never invent
   helpers such as length() or size() to rank values, never keep the entity
   in the WITH that computes MAX()/MIN() (that returns everyone), and never
   re-MATCH the entity to compare it with its own maximum.

   When the superlative ranks a NUMBER OF RELATED NODES instead of a
   property ("the most sensors", "the fewest competitions", "the busiest
   month"), the query has EXACTLY TWO aggregation clauses and no more. The
   first names the group and computes its count and its list together; the
   second turns those counts into a distribution. Copy this shape:
   YES : MATCH (s:InertialSensors)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
         WITH eo, COUNT(DISTINCT s) AS n, COLLECT(DISTINCT s.id) AS items
         WITH n, COUNT(DISTINCT eo) AS entity_count,
              COLLECT(DISTINCT eo.id) AS entities
         RETURN n, entity_count, entities
         ORDER BY n DESC
   The extreme is the first row, and the remaining rows give the comparison
   the question implies. A third aggregation clause is always a mistake:
   whatever it wants to collect went out of scope one clause earlier.

   And when the grouping key is a plain expression on the matched node —
   a month, a race, a category, a frequency — there is NO WITH at all. One
   RETURN groups, counts and collects in a single clause:
   YES : MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
         RETURN e.eventDate.month AS month, COUNT(DISTINCT e) AS n,
                COLLECT(DISTINCT e.id) AS events
         ORDER BY n
   Reach for a WITH only when a later clause has to consume the aggregate.

6. A COUNT WITHOUT ITS NAMES. Every COUNT(DISTINCT x) needs a matching
   COLLECT(DISTINCT <name of x>) in the SAME WITH or RETURN, so the answer
   can list the entities and not just a number.

7. TRAINING "FREQUENCY" READ AS A STAGE COUNT. "Fréquence d'entraînement",
   "sessions per week", "intensité", "durée" are the properties
   t.Frequency / t.Intensity / t.Volume. COUNT(DISTINCT t) answers a
   different question. See §3.6a.

8. POSITION DISTRIBUTION GROUPED BY HORSE BY MISTAKE. A question about
   where sensors are placed asks for POSITION totals, not a per-horse
   inventory:
   YES : MATCH (s:InertialSensors)
         RETURN labels(s)[1] AS position, COUNT(DISTINCT s) AS sensor_count,
                COLLECT(DISTINCT s.id)[0..3] AS sample_ids
         ORDER BY sensor_count DESC

9. type(node) ON A NODE. type() crashes on nodes. For schema / "how do I
   find" questions, return relationship names as string literals
   ("HASPARTICIPATION", "ISATTACHEDTO", …) — see §3.8bis. Never write
   type(e), type(p) or type(h).

10. SUPERLATIVE OVER A GROUPED COUNT (e.g. most frequent breed label) — use
    the COLLECT/UNWIND skeleton from rule C. Never reopen COUNT(DISTINCT h)
    after a WITH that already dropped h:
    YES : MATCH (h:Horse)
          WITH h.hasRace AS race, COUNT(DISTINCT h) AS horse_count,
               COLLECT(DISTINCT h.hasName) AS horses
          WITH COLLECT({{race: race, n: horse_count, horses: horses}}) AS rows,
               MAX(horse_count) AS top
          UNWIND rows AS row
          WITH row, top
          WHERE row.n = top
          RETURN row.race AS race, row.n AS horse_count, row.horses AS horses

Also remember: "is there an X with several Y", "at least two", "more than
one" translate into WITH <keys>, COUNT(DISTINCT <item>) AS n → WHERE n > 1,
never into HAVING.

And: only hasName holds a bare name. Every id carries a prefix
(Rider_Alice, Vet_DrMartin, Caretaker_Sophie, Event_Example_01), so a person
named in the question is matched with WHERE a.id CONTAINS "Sophie", while a
horse is matched with {{hasName: "<HorseName>"}} (substitute the name from
the question).

═══════════════════════════════════════════════════════════════
SECTION 5 — CHECKS BEFORE ANSWERING
═══════════════════════════════════════════════════════════════

□ Exactly one RETURN, placed at the end? Is every variable bound by a MATCH
  before it is used?
□ No variable used after a WITH that does not carry it? No alias used inside
  the expression that defines it?
□ Does the word HAVING or GROUP BY appear? → remove them and use
  WITH <keys>, <aggregate> AS alias then WHERE alias <condition>.
□ No UNION, no MATCH inside a WHERE, no OR between two patterns?
□ One WHERE per block, conditions merged with AND?
□ All directions consistent with 1.2, whatever node the pattern starts from
  (horse first or sensor first)?
□ Discipline asked for → labels(e)[0]; level asked for → e.category?
□ Named entity → filtered in a mandatory MATCH and present in the RETURN?
□ Absence question → OPTIONAL MATCH + WITH COUNT + WHERE count = 0?
□ No WHERE after the RETURN? Is every filter on an alias inside a WITH?
□ Ranking or result question → goes through EventParticipation
  (HASHORSE / HASRIDER); association question with no event → ASSOCIATEDWITH?
□ Is MAX()/MIN() computed in a WITH that no longer carries the grouping key?
□ Aggregated value filtered or sorted on → present in the RETURN?
□ No invented literal value in a filter?
□ COUNT(DISTINCT ...) everywhere, no SUM/AVG on a string?
□ Does each clause that computes a COUNT ALREADY contain, on that very same
  clause, the COLLECT(DISTINCT ...) naming the entities counted? If one is
  missing, REWRITE THAT CLAUSE to hold both. Never repair this by appending
  another WITH: the entity is out of scope there and the query will not run.
□ Superlative: is it about a property VALUE (MAX on the property) or about a
  NUMBER (MAX on the COUNT)? Re-read 3.4.a. If the draft has LIMIT 1 on a
  "le plus / the most" count question → DELETE the LIMIT and emit the full
  histogram (C2) or the MAX-tie filter instead.
□ Simple question about a role or a label → a single MATCH line, with no
  superfluous traversal or aggregation?

Return the Cypher query only after these checks.

============== 10. UNANSWERABLE / AMBIGUOUS RULES =============

4.5 NEVER GIVE UP
NEVER generate RETURN "Information not available in the knowledge graph".
Even an abstractly phrased question translates into a query on the graph:
identify the labels closest to the subject and bring back the matching data.
- duration of a stage → t.Volume
- recovery / rest → TransitionStage
- calibration, offset, "calibrated the same way" → s.hasSensorOffset,
  grouped by labels(s)[1] AND by the offset itself:
    MATCH (s:InertialSensors)
    RETURN labels(s)[1] AS position, s.hasSensorOffset AS offset,
           COUNT(DISTINCT s) AS sensor_count
    ORDER BY position, offset
  Two rows for the same position mean the calibration is NOT uniform there.
  Never answer a calibration question by counting horses or objectives.
- sampling rate, frequency in Hz → s.hasSensorTime
- level, category → e.category
- discipline, type of event → labels(e)[0]
- season / competition link → INSEASON + e.category
- date consistency → e.eventDate across all events
- system overview → count every label involved:
  MATCH (h:Horse) WITH COUNT(DISTINCT h) AS horses
  MATCH (r:Rider) WITH horses, COUNT(DISTINCT r) AS riders
  MATCH (s:InertialSensors) WITH horses, riders, COUNT(DISTINCT s) AS sensors
  MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
  RETURN horses, riders, sensors, COUNT(DISTINCT e) AS events

====================== 11. OUTPUT FORMAT ======================

See section 1 — Cypher only, no markdown, no commentary.

========================== 12. SAFETY =========================

This system is READ-ONLY.
NEVER generate any Cypher operation that modifies the database or schema.
STRICTLY FORBIDDEN:
- DELETE
- DETACH DELETE
- DROP
- CREATE
- MERGE
- SET
- REMOVE
Only read/query operations are allowed:
MATCH, OPTIONAL MATCH, WHERE, WITH, UNWIND, RETURN,
ORDER BY, LIMIT and read-only aggregation/functions.
If the user's question asks to modify, delete, create, update,
or alter data/schema, do NOT perform the operation.

=============== 13/14. RUNTIME SCHEMA & QUESTION ==============

Schema: {schema}
Question: {question}
Cypher Query:"""

    return PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )


def get_qa_prompt():
    """Get the QA prompt template"""
    QA_TEMPLATE = """======================== 1. ROLE / TASK =======================

You answer a question about the equestrian knowledge graph using only the
context provided below.

====================== 2. GROUNDING RULE ======================

RÈGLES PRIORITAIRES
1. Ta réponse doit être basée exclusivement sur le context fourni ci-dessous. N'utilise aucune connaissance externe.

RÈGLES DE GROUNDING — NE JAMAIS HALLUCINER
- Si un cheval n'est PAS mentionné dans le context, ne le mentionne pas.
- Si une propriété n'est PAS dans le context, ne l'invente pas.
- N'ajoute aucune information qui n'est pas explicitement présente dans le context.

========================= 3. RELEVANCE ========================

Answer what the question asks; see section 4 for how much supporting detail
to include.

======================= 4. COMPLETENESS =======================

2. Si le context contient des données, extrais-les et présente-les directement et précisément.
5. Pour les questions de comparaison ou d'agrégation, liste explicitement les valeurs du context avant de conclure.

RÈGLES D'EXHAUSTIVITÉ — UNE RÉPONSE INCOMPLÈTE EST UNE MAUVAISE RÉPONSE
- Cite TOUS les nombres présents dans le context : totaux, effectifs par
  groupe, comptes. Ne te contente jamais du total global quand le context
  donne aussi le détail par groupe.
- Quand le context contient une liste de noms (chevaux, cavaliers, événements,
  capteurs), énumère-les explicitement dans la réponse. N'écris pas
  « plusieurs chevaux » ni « certains cavaliers » si les noms sont là.
- Pour une liste de noms de chevaux, de cavaliers ou d'événements, énumère-les
  tous, sur une seule ligne séparés par des virgules (jamais en liste
  numérotée), après avoir donné le nombre exact.
- Pour des identifiants techniques de capteurs, donne le nombre exact et
  deux ou trois exemples seulement : n'énumère jamais des dizaines
  d'identifiants.
- Traduis les positions anatomiques en français : Withers = garrot,
  Sternum = sternum, CanonOfForelimb = canon antérieur,
  CanonOfHindlimb = canon postérieur. Ne traduis jamais un nom propre.
- Quand plusieurs lignes partagent la valeur extrême (même durée maximale,
  même fréquence la plus élevée), mentionne-les TOUTES et signale l'égalité.
- Quand le context contient une colonne d'identifiant (event, stage, sensor,
  rider, objective), cite CET identifiant dans la réponse. Un événement se
  nomme par son id ET sa ville (« Event_Dressage_01, à Angers ») : ne le
  désigne jamais uniquement par sa discipline et sa date.
- Quand le context contient des noms de relations (colonnes step_1, step_2,
  type(...) — par exemple HASPARTICIPATION, ISATTACHEDTO, DEPENDSON,
  INSEASON), la question porte sur le CHEMIN : la réponse DOIT commencer par
  l'énumération ordonnée de ces relations (ex. « on passe par HASPARTICIPATION,
  puis HASHORSE / HASRIDER »), puis illustrer avec l'exemple concret. Une
  réponse qui ne cite aucune relation est incomplète.
- Quand le context donne une distribution (une ligne par valeur, avec le
  nombre d'entités et leurs noms), réponds d'abord par la valeur extrême
  demandée (« le minimum est 2, partagé par 44 chevaux : … »), puis cite
  les autres groupes pour la comparaison. N'écris jamais « tous ont la même
  valeur » alors que le context contient plusieurs lignes avec des valeurs
  différentes, et n'écris jamais « il n'y a pas de minimum » quand une
  ligne donne clairement ce minimum.
- Quand le context ne contient qu'une seule combinaison de valeurs pour tout
  un groupe, dis explicitement que c'est identique pour tous.
- Quand la question demande si un acteur intervient « à toutes les étapes »
  et que le context ne liste que certaines phases (préparation,
  pré-compétition…), la réponse est NON : cite les phases présentes et
  précise celles qui manquent (compétition, transition).
- Termine par la conclusion directe attendue par la question (oui / non /
  la valeur), après avoir présenté les données.

====================== 5. INFERENCE RULE ======================

- N'ajoute aucune information qui n'est pas explicitement présente dans le context.
- Si une propriété n'est PAS dans le context, ne l'invente pas.

=============== 6. EMPTY / INSUFFICIENT CONTEXT ===============

3. Si le context est vide ET que la question porte sur l'existence ou sur
   une généralité, l'absence de résultat EST la réponse : n'écris JAMAIS
   « information non disponible » dans ce cas. Détermine le sens selon la
   formulation de la question :
   - question qui cherche des cas (« y a-t-il des… », « existe-t-il… »,
     « arrive-t-il… », « certains… ? ») → « Non, aucun cas de ce type
     n'existe dans le système », en reformulant le cas recherché.
   - question oui/non sur une implication (« X intervient-il en phase Y ? »,
     « Sophie apparaît-elle aussi en compétition ? ») → « Non ».
   - question qui vérifie une règle générale (« est-ce que TOUS… »,
     « chaque… a-t-il… ? », « … sont-ils tous… ? ») → la requête cherchait
     les exceptions et n'en a trouvé aucune, donc la règle est vérifiée :
     réponds « Oui, sans exception », en reformulant la règle confirmée.
     N'écris jamais « non » ni « aucun » pour ce type de question.
   Si le context est vide et que la question n'est pas une question
   d'existence ni oui/non, réponds : Cette information n'est pas disponible.

====================== 7. PARTIAL ANSWER ======================

   MAIS si le context contient des données partielles, utilise-les pour
   répondre partiellement plutôt que de dire non disponible.

======================= 8. DATA FIDELITY ======================

4. Ne contredis jamais le context récupéré. Si le context dit X, ta réponse doit dire X.

RÈGLES DE PRÉSENTATION DES NOMS
- Les noms de chevaux viennent directement de la propriété hasName dans le context : utilise-les tels quels.
- Les identifiants de cavaliers sont au format Rider_XXXX : présente naturellement seulement la partie nom.
- Les identifiants de vétérinaires sont au format Vet_XXXX : présente naturellement le nom.
- Les identifiants de soigneurs sont au format Caretaker_XXXX : présente naturellement le nom.
- N'utilise aucun mapping codé en dur pour les chevaux : les vrais noms sont déjà dans le context.
- N'expose jamais les URIs brutes ou les identifiants internes techniques à l'utilisateur.

The section 4 bullets on exact numbers and on technical identifiers apply
here as well; see section 4.

========================= 9. LANGUAGE =========================

- Respond in the same language as the question.

====================== 10. RESPONSE STYLE =====================

RÈGLES DE FORMAT
- Ne dis jamais "as indicated in the context" ou "d'après le contexte".
- N'expose jamais les structures de données brutes, les URIs ou les identifiants techniques.
- Utilise directement les informations dans des phrases naturelles.

=============== 11. OPTIONAL FEW-SHOT EXAMPLES ================

None included in this version.

========================= 12. QUESTION ========================

Question: {question}

========================== 13. CONTEXT ========================

Context: {context}

========================== 14. ANSWER =========================

Réponse:"""
    
    return PromptTemplate(
        input_variables=["question", "context"],
        template=QA_TEMPLATE
    )


def init_graph_chain():
    """Initialize the complete GraphRAG chain"""
    # Initialize graph
    graph = init_graph()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Get prompts
    cypher_prompt = get_cypher_prompt()
    qa_prompt = get_qa_prompt()
    
    # Create chain using langchain_neo4j's GraphCypherQAChain
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        top_k=50,
        return_intermediate_steps=True,
        allow_dangerous_requests=True
    )
    
    return chain, graph


# ── Single-retry Cypher error correction (separate from CYPHER_GENERATION_TEMPLATE) ──

CYPHER_CORRECTION_PROMPT = """The following Cypher query failed with this error:
Query: {broken_query}
Error: {error_message}
Question it was meant to answer: {question}
Fix ONLY the specific issue causing this error. Do not change the query's logic or intent otherwise. Return corrected Cypher only, no commentary, no markdown."""


def _strip_cypher_fences(text: str) -> str:
    """Remove optional markdown fences from an LLM Cypher reply."""
    import re

    text = (text or "").strip()
    match = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _is_retryable_cypher_error(exc: BaseException) -> bool:
    """True for Neo4j / Cypher execution failures worth one correction attempt."""
    name = type(exc).__name__
    if name in {
        "CypherSyntaxError",
        "ClientError",
        "CypherTypeError",
        "DatabaseError",
        "Neo4jError",
    }:
        return True
    msg = str(exc).lower()
    markers = (
        "variable",
        "not defined",
        "invalid input",
        "syntax error",
        "cypher",
        "neo4j",
        "unknown function",
        "type mismatch",
        "expected",
        "cannot use",
    )
    return any(m in msg for m in markers)


def _cypher_llm_from_chain(chain):
    """Reuse the chain's existing cypher-generation LLM (same model/temperature)."""
    gen = getattr(chain, "cypher_generation_chain", None)
    if gen is None:
        raise RuntimeError("Graph chain has no cypher_generation_chain")
    llm = getattr(gen, "llm", None)
    if llm is None:
        # LLMChain sometimes nests the model under .llm
        llm = getattr(getattr(gen, "llm_chain", None), "llm", None)
    if llm is None:
        raise RuntimeError("Could not locate cypher-generation LLM on chain")
    return llm


def _correct_cypher_once(llm, question: str, broken_query: str, error_message: str) -> str:
    """One corrective LLM call; returns cleaned Cypher text."""
    prompt = CYPHER_CORRECTION_PROMPT.format(
        broken_query=broken_query,
        error_message=error_message,
        question=question,
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return _strip_cypher_fences(content)


def invoke_graph_chain_with_cypher_retry(chain, inputs, config=None):
    """
    Invoke GraphCypherQAChain with at most one Cypher-fix retry on Neo4j errors.

    Success path: identical to ``chain.invoke`` (result dict unchanged).
    Failure path: if Neo4j rejects the generated Cypher, ask the same LLM to
    fix that error once, re-query, then run the existing QA sub-chain. On a
    second failure, re-raise the original exception.
    """
    graph = chain.graph
    original_query = graph.query
    last_cypher = {"q": None}
    query_error = {"exc": None}

    def _tracking_query(query, *args, **kwargs):
        last_cypher["q"] = query
        try:
            return original_query(query, *args, **kwargs)
        except Exception as exc:
            query_error["exc"] = exc
            raise

    graph.query = _tracking_query  # type: ignore[method-assign]
    try:
        if config is None:
            result = chain.invoke(inputs)
        else:
            result = chain.invoke(inputs, config=config)
        return result
    except Exception as exc:
        broken = last_cypher["q"]
        neo4j_exc = query_error["exc"]
        # Only retry when Neo4j rejected the generated Cypher (not QA / other failures).
        if neo4j_exc is None or not broken or not _is_retryable_cypher_error(neo4j_exc):
            raise

        question = inputs.get("query") or inputs.get("question") or ""
        error_message = str(neo4j_exc)
        try:
            llm = _cypher_llm_from_chain(chain)
            corrected = _correct_cypher_once(llm, question, broken, error_message)
            if not corrected:
                raise neo4j_exc
            context = original_query(corrected)[: getattr(chain, "top_k", 50)]
        except Exception:
            # Cap at one retry: surface the original Neo4j failure unchanged.
            raise neo4j_exc from None

        qa_callbacks = None
        if config and isinstance(config, dict):
            qa_callbacks = config.get("callbacks")

        qa_inputs = {"question": question, "context": context}
        if qa_callbacks is not None:
            qa_out = chain.qa_chain.invoke(qa_inputs, callbacks=qa_callbacks)
        else:
            qa_out = chain.qa_chain.invoke(qa_inputs)

        output_key = getattr(chain.qa_chain, "output_key", "text")
        if isinstance(qa_out, dict):
            final_answer = qa_out.get(output_key, qa_out)
        else:
            final_answer = qa_out

        chain_output_key = getattr(chain, "output_key", "result")
        result = {
            chain_output_key: final_answer,
            "cypher_retry_used": True,
            "original_error": error_message,
            "original_cypher": broken,
        }
        if getattr(chain, "return_intermediate_steps", False):
            result["intermediate_steps"] = [
                {"query": corrected},
                {"context": context},
            ]
        return result
    finally:
        graph.query = original_query  # type: ignore[method-assign]
