"""
LangChain & OpenAI logic
"""
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from .config import OPENAI_API_KEY
from .graph_service import init_graph



def get_cypher_prompt():
    """Get the Cypher generation prompt template"""
    CYPHER_GENERATION_TEMPLATE = """Tâche : Générer une requête Cypher pour Neo4j.

RÈGLE 1 — LABELS D'ÉVÉNEMENTS
Il n'existe PAS de label "Event" dans ce graphe.
Les événements utilisent UNIQUEMENT : ShowJumping, Dressage, ou Cross.
INVALID: MATCH (e:Event)
VALID: MATCH (e) WHERE (e:ShowJumping OR e:Dressage OR e:Cross)
VALID: MATCH (e:ShowJumping)

RÈGLE 2 — RELATIONS SUPPRIMÉES OU ANCIENNES
N'utilise AUCUNE relation supprimée ou héritée d'un ancien schéma.
Utilise uniquement les relations V8 listées ci-dessous.

RÈGLE 3 — DIRECTIONS CORRECTES DES RELATIONS — NE JAMAIS INVERSER
- (Rider)-[:ASSOCIATEDWITH]->(Horse)
- (Horse)-[:TRAINSIN]->(TrainingStage)
- (TrainingStage)-[:DEPENDSON]->(Event)
- (TrainingStage)-[:INVOLVESACTOR]->(Rider|Veterinarian|Caretaker)
- (Event)-[:HASPARTICIPATION]->(EventParticipation)
- (EventParticipation)-[:HASHORSE]->(Horse)
- (EventParticipation)-[:HASRIDER]->(Rider)
- (InertialSensors)-[:ISATTACHEDTO]->(Horse)
- (InertialSensors)-[:ISUSEDFOR]->(ExperimentalObjective)
- (Event)-[:INSEASON]->(CompetitiveSeason)
- (Horse)-[:COMPETESIN]->(Event)

RÈGLE 4 — MAPPINGS DES PROPRIÉTÉS
- Chevaux : h.hasName pour le nom, h.hasRace pour la race
- Cavaliers : r.id pour l'identifiant — il n'existe PAS de propriété hasName sur Rider
- Vétérinaire : v.id
- Soigneur : c.id
- Événements : e.id, e.category, e.eventLocation, e.eventDate
- TrainingStage : t.Volume, t.Intensity, t.Frequency — toutes sont des CHAÎNES
- Capteurs : s.id pour l'identifiant, s.hasSensorTime pour la fréquence d'échantillonnage
- EventParticipation : p.rank pour le classement
- Saison : s.seasonName (= "Saison 2026"), s.seasonStart, s.seasonEnd
- ExperimentalObjective : eo.id (valeurs : 'GaitClassif_01' ou 'FatigueDetection')

RÈGLE 5 — NE JAMAIS UTILISER SUM() SUR DES CHAÎNES
Frequency, Volume, Intensity sont des propriétés STRING.
INVALID: SUM(t.Frequency)
VALID: RETURN t.Frequency directement

RÈGLE 6 — TOUJOURS UTILISER COUNT(DISTINCT)
INVALID: COUNT(s)
VALID: COUNT(DISTINCT s)

RÈGLE 7 — RÈGLES DE RETURN
- Retourne toujours l'identifiant avec les autres propriétés demandées
- INVALID: RETURN e.eventDate
- VALID: RETURN e.id, e.eventDate
- Pour les chevaux : RETURN h.hasName, h.otherProperty
- Pour tous les autres nœuds : RETURN node.id, node.otherProperty
- Ne duplique jamais les colonnes dans RETURN
- INVALID: RETURN e.id, p.rank, h.hasName, p.rank
- VALID: RETURN e.id, p.rank, h.hasName

RÈGLE 8 — RÈGLES DE SYNTAXE
- N'utilise jamais UNION — préfère WHERE ... OR ...
- N'utilise jamais de backticks pour les labels
- Pour les étapes d'entraînement : WHERE (t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage)
- Pour la recherche textuelle : WHERE toLower(h.hasName) CONTAINS toLower("searchterm")

RÈGLE 9 — PARTIE DU CORPS DES CAPTEURS
InertialSensors a toujours 2 labels : InertialSensors + partie du corps.
Sous-classes de partie du corps : Withers, Sternum, CanonOfForelimb, CanonOfHindlimb.
Pour obtenir la partie du corps : utilise labels(s)[1].
Exemple : MATCH (s:InertialSensors)-[:ISATTACHEDTO]->(h:Horse) RETURN s.id, labels(s)[1] as body_part

RÈGLE 10 — PARTICIPATIONS ET CLASSEMENTS
Pour trouver la participation cheval + cavalier dans un événement :
MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
MATCH (p)-[:HASHORSE]->(h:Horse)
MATCH (p)-[:HASRIDER]->(r:Rider)
RETURN e.id, h.hasName, r.id, p.rank

RÈGLE 11 — INVOLVESACTOR
Direction : (TrainingStage)-[:INVOLVESACTOR]->(actor)
Ne jamais inverser cette direction.
L'acteur peut être : Rider, Veterinarian, ou Caretaker.
Exemple : MATCH (t:PreparationStage)-[:INVOLVESACTOR]->(actor) RETURN actor.id

RÈGLE 12 — SAISON
MATCH (e)-[:INSEASON]->(s:CompetitiveSeason)
Valeur de seasonName : "Saison 2026"

RÈGLE 13 — REQUÊTES SIMPLES ET FOCALISÉES
Réponds UNIQUEMENT à ce qui est demandé. Ne mélange pas plusieurs sujets dans une seule requête.
- Question sur les acteurs d'entraînement → utilise TrainingStage + INVOLVESACTOR seulement
- Question sur les événements → utilise les labels d'événement et leurs relations seulement
- Question sur les capteurs → utilise InertialSensors + ISUSEDFOR/ISATTACHEDTO seulement

RÈGLE — AGRÉGATION DES POSITIONS DE CAPTEURS
Quand la question demande où les capteurs sont placés ou combien de capteurs sont à chaque position,
ne retourne jamais les capteurs individuellement. Groupe toujours par position.

CORRECT:
MATCH (s:InertialSensors)
RETURN labels(s)[1] as position, COUNT(DISTINCT s) as count
ORDER BY count DESC

INCORRECT:
MATCH (s:InertialSensors)-[:ISATTACHEDTO]->(h:Horse)
RETURN s.id, labels(s)[1] as body_part
(cela retourne 108 lignes et le résultat est trop grand)

RÈGLE — OBJECTIFS DES CAPTEURS SANS UNION
Quand la question demande quels capteurs servent l'objectif A ou l'objectif B pour un cheval précis,
utilise une seule requête avec l'objectif dans le RETURN, jamais UNION.

CORRECT:
MATCH (s:InertialSensors)-[:ISATTACHEDTO]->(h:Horse {{hasName: "Dakota"}})
MATCH (s)-[:ISUSEDFOR]->(eo:ExperimentalObjective)
RETURN s.id, labels(s)[1] as body_part, eo.id as objective

INCORRECT:
MATCH ... RETURN ... "GaitClassif_01"
UNION
MATCH ... RETURN ... "FatigueDetection"

Schema: {schema}
Question: {question}
Cypher Query:"""
    
    return PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )


def get_qa_prompt():
    """Get the QA prompt template"""
    QA_TEMPLATE = """Question: {question}
Context: {context}

RÈGLES DE GROUNDING — NE JAMAIS HALLUCINER
- Si un cheval n'est PAS mentionné dans le context, ne le mentionne pas.
- Si une propriété n'est PAS dans le context, ne l'invente pas.
- N'ajoute aucune information qui n'est pas explicitement présente dans le context.

RÈGLES DE PRÉSENTATION DES NOMS
- Les noms de chevaux viennent directement de la propriété hasName dans le context : utilise-les tels quels.
- Les identifiants de cavaliers sont au format Rider_XXXX : présente naturellement seulement la partie nom.
- Les identifiants de vétérinaires sont au format Vet_XXXX : présente naturellement le nom.
- Les identifiants de soigneurs sont au format Caretaker_XXXX : présente naturellement le nom.
- N'utilise aucun mapping codé en dur pour les chevaux : les vrais noms sont déjà dans le context.
- N'expose jamais les URIs brutes ou les identifiants internes techniques à l'utilisateur.

RÈGLES DE FORMAT
- Réponds en français naturel et fluide.
- Ne dis jamais "as indicated in the context" ou "d'après le contexte".
- N'expose jamais les structures de données brutes, les URIs ou les identifiants techniques.
- Utilise directement les informations dans des phrases naturelles.

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



