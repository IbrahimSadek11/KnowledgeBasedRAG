# Track 1 — Ground-truth reconciliation (sign-off)

**Date:** 2026-07-26  
**Source of flags:** `docs/graph_ground_truth_flags.md` + `docs/graph_v9_ground_truth_drift.md`  
**Live graph:** 516 nodes / 1460 relationships (verified this session)  
**Status:** `data/test_dataset.json` **NOT modified**. Awaiting explicit human approval.

Every Cypher below was executed directly against live Neo4j in this session
(not via the RAG chain, not from memory).

---

## Summary

| Verdict | Questions |
|---|---|
| **CLEAN — propose correction** | Q38, Q39, Q14, Q82, Q5, Q52, Q53, Q55, Q86, Q93, Q94, Q62, Q65, Q67, Q97, Q98, Q23, Q34*, Q57, Q63, Q66, Q69, **Q10**, **Q64**, **Q89** |
| **SCOPE/PRECISION — extra review** | **Q76** (question excludes rider; GT lists rider anyway) |
| **NO CHANGE — GT still correct** | **Q8** (horse-level durations match; only stage count differs because Dakota has 2× prep) |
| **PARTIAL / note** | **Q34*** — preparation half still correct; pré-compétition half must change |

\*Q34 shares the Emma/Manon story with Q23; only the pré-compétition clause is stale.

**Q10 note:** Flagged **INCORRECT** (not V9-upgrade drift). Live `DEPENDSON` from Dakota's eight training stages yields four distinct events; the NL GT naming only the two ShowJumping events was already wrong relative to the graph edges.

**Q64 note:** Same Dakota stage-count fact as Q5/Q53/Q86 — NL still said “6 étapes”; live = **8**. Missed in the first reconciliation pass.

**Q76 note:** **SCOPE/PRECISION**, not data-drift — see staged section below; needs extra review.

**Q89 note:** Same V8-era stale count / relation-name pattern — NL still says `DEPENDS_ON` / **171**; live = `DEPENDSON` / **207**. Missed because Q89 was outside the original reconciliation pass’s scope.

---

## Category A — sensor identifiers

### Q38 — CLEAN

**(a) Question:**  
Pour quel objectif expérimental le capteur IMU_Withers_01 est-il utilisé ?

**(b) Current ground_truth:**  
Le capteur IMU_Withers_01 est utilisé pour la classification de la démarche (GaitClassif_01).

**(c) Proposed correction:**  
Le capteur IMU_Withers_Dakota_01 (garrot de Dakota) est utilisé pour la classification de la démarche (GaitClassif_01).

**(d) Exact Cypher + live result:**
```cypher
MATCH (s) WHERE s.id IN ['IMU_Withers_01'] RETURN s.id AS id
```
Live: **0 rows**

```cypher
MATCH (s {id:'IMU_Withers_Dakota_01'})-[:ISUSEDFOR]->(o)
RETURN s.id AS sensor, o.id AS objective
```
Live: `{'sensor': 'IMU_Withers_Dakota_01', 'objective': 'GaitClassif_01'}`

---

### Q39 — CLEAN

**(a) Question:**  
Pour quel objectif expérimental le capteur IMU_CanonFore_01 est-il utilisé ?

**(b) Current ground_truth:**  
Le capteur IMU_CanonFore_01 est utilisé pour la détection de fatigue (FatigueDetection).

**(c) Proposed correction:**  
Le capteur IMU_CanonFore_Dakota_01 (canon antérieur de Dakota) est utilisé pour la détection de fatigue (FatigueDetection).

**(d) Exact Cypher + live result:**
```cypher
MATCH (s) WHERE s.id = 'IMU_CanonFore_01' RETURN s.id AS id
```
Live: **0 rows**

```cypher
MATCH (s {id:'IMU_CanonFore_Dakota_01'})-[:ISUSEDFOR]->(o)
RETURN s.id AS sensor, o.id AS objective
```
Live: `{'sensor': 'IMU_CanonFore_Dakota_01', 'objective': 'FatigueDetection'}`

---

### Q14 — CLEAN (example id only; count already correct)

**(a) Question:**  
Combien de capteurs IMU sont placés au garrot, et peux-tu donner quelques exemples d'identifiants ?

**(b) Current ground_truth:**  
Il y a cinquante capteurs IMU placés au garrot dans le système, un par cheval, par exemple IMU_Withers_01 pour Dakota, IMU_Withers_Comet_01 pour Comet et IMU_Withers_Apollon_01 pour Apollon.

**(c) Proposed correction:**  
Il y a cinquante capteurs IMU placés au garrot dans le système, un par cheval, par exemple IMU_Withers_Dakota_01 pour Dakota, IMU_Withers_Comet_01 pour Comet et IMU_Withers_Apollon_01 pour Apollon.

**(d) Exact Cypher + live result:**
```cypher
MATCH (s:Withers)
WITH COUNT(s) AS n, COLLECT(s.id)[0..5] AS sample
MATCH (d {id:'IMU_Withers_Dakota_01'})
OPTIONAL MATCH (bare {id:'IMU_Withers_01'})
RETURN n AS withers_count, sample, d.id AS dakota_id, bare.id AS bare_v8_id
```
Live: `{'withers_count': 50, 'sample': ['IMU_Withers_Thunder_01', 'IMU_Withers_Luminos_01', 'IMU_Withers_Arrow_01', 'IMU_Withers_Sable_01', 'IMU_Withers_Luna_01'], 'dakota_id': 'IMU_Withers_Dakota_01', 'bare_v8_id': None}`

```cypher
MATCH (s) WHERE s.id IN ['IMU_Withers_Comet_01','IMU_Withers_Apollon_01','IMU_Withers_Dakota_01']
RETURN s.id AS id
```
Live: all three present.

---

### Q82 — CLEAN

**(a) Question:**  
Quels capteurs de Dakota servent à analyser sa démarche, et lesquels servent à surveiller sa fatigue ?

**(b) Current ground_truth:**  
Pour Dakota, les capteurs IMU_Withers_01 (garrot) et IMU_CanonHind_01 (canon postérieur) sont utilisés pour l'analyse de la démarche. Les capteurs IMU_CanonFore_01 (canon antérieur) et IMU_Sternum_01 (sternum) sont utilisés pour la détection de fatigue.

**(c) Proposed correction:**  
Pour Dakota, les capteurs IMU_Withers_Dakota_01 (garrot) et IMU_CanonHind_Dakota_01 (canon postérieur) sont utilisés pour l'analyse de la démarche (GaitClassif_01). Les capteurs IMU_CanonFore_Dakota_01 (canon antérieur) et IMU_Sternum_Dakota_01 (sternum) sont utilisés pour la détection de fatigue (FatigueDetection).

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse {hasName:'Dakota'})<-[:ISATTACHEDTO]-(s)-[:ISUSEDFOR]->(o)
RETURN s.id AS sensor, labels(s) AS sensor_labels, o.id AS objective
ORDER BY sensor
```
Live (4 rows):
- `IMU_CanonFore_Dakota_01` → FatigueDetection
- `IMU_CanonHind_Dakota_01` → GaitClassif_01
- `IMU_Sternum_Dakota_01` → FatigueDetection
- `IMU_Withers_Dakota_01` → GaitClassif_01

---

## Category B — training-stage inventory

### Q5 — CLEAN

**(a) Question:**  
Quelles étapes d'entraînement Dakota suit-il ?

**(b) Current ground_truth:**  
Dakota suit six étapes d'entraînement : Training_Prepa_SJ_01 (préparation), Training_PreComp_SJ_01 (pré-compétition), Training_Preparation_SJ_01 (préparation), Training_PreCompetition_SJ_01 (pré-compétition), Training_Competition_SJ_01 (compétition) et Training_Transition_SJ_01 (transition).

**(c) Proposed correction:**  
Dakota suit huit étapes d'entraînement : Training_Prepa_SJ_01 (préparation), Training_Preparation_SJ_01 (préparation), Training_PreComp_SJ_01 (pré-compétition), Training_PreCompetition_SJ_01 (pré-compétition), Training_Competition_SJ_01 (compétition), Training_Comp_Cross2026_Dakota_01 (compétition), Training_Comp_Dress01_Dakota_01 (compétition) et Training_Transition_SJ_01 (transition).

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse {hasName:'Dakota'})-[:TRAINSIN]->(t)
RETURN COUNT(DISTINCT t) AS n, COLLECT(DISTINCT t.id) AS ids
```
Live: `{'n': 8, 'ids': ['Training_Competition_SJ_01', 'Training_Transition_SJ_01', 'Training_PreComp_SJ_01', 'Training_PreCompetition_SJ_01', 'Training_Comp_Cross2026_Dakota_01', 'Training_Preparation_SJ_01', 'Training_Comp_Dress01_Dakota_01', 'Training_Prepa_SJ_01']}`

---

### Q52 — CLEAN

**(a) Question:**  
Comment se décompose le programme d'entraînement dans le système : quelles phases existent et combien d'étapes sont enregistrées pour chacune ?

**(b) Current ground_truth:**  
Le programme d'entraînement comprend quatre types de phases : préparation, pré-compétition, compétition et transition. Dans le système, il y a 51 étapes de préparation, 51 étapes de pré-compétition, 50 étapes de compétition et 19 étapes de transition. Le total supérieur à 50 pour la préparation et la pré-compétition vient du fait que Dakota possède des étapes supplémentaires liées au saut d'obstacles ; le nombre réel de chevaux reste 50.

**(c) Proposed correction:**  
Le programme d'entraînement comprend quatre types de phases : préparation, pré-compétition, compétition et transition. Dans le système, il y a 51 étapes de préparation, 51 étapes de pré-compétition, 55 étapes de compétition et 50 étapes de transition. Le total supérieur à 50 pour la préparation, la pré-compétition et la compétition vient du fait que Dakota possède des étapes supplémentaires ; chaque cheval a désormais une phase de transition.

**(d) Exact Cypher + live result:**
```cypher
MATCH (t)
WHERE t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage
RETURN labels(t)[0] AS stage_type, COUNT(t) AS n
ORDER BY stage_type
```
Live:
- CompetitionStage → 55
- PreCompetitionStage → 51
- PreparationStage → 51
- TransitionStage → 50

---

### Q53 — CLEAN

**(a) Question:**  
Quel cheval suit le programme d'entraînement le plus complet, avec le plus grand nombre d'étapes ?

**(b) Current ground_truth:**  
Dakota suit le programme le plus complet avec 6 étapes d'entraînement au total, contre 3 ou 4 étapes pour la majorité des autres chevaux (31 chevaux suivent 3 étapes, 18 chevaux en suivent 4). Le nombre d'étapes par cheval correspond bien au détail des phases suivies individuellement : 31×3 + 18×4 + 6 = 171.

**(c) Proposed correction:**  
Dakota suit le programme le plus complet avec 8 étapes d'entraînement au total. La répartition est : 46 chevaux suivent 4 étapes, 3 chevaux en suivent 5 (Dune, Pixie, Ecume), et Dakota en suit 8. Vérification : 46×4 + 3×5 + 8 = 207.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)-[:TRAINSIN]->(t)
WITH h, COUNT(DISTINCT t) AS n
RETURN n AS stages, COUNT(h) AS horses, COLLECT(h.hasName) AS names
ORDER BY n
```
Live:
- stages=4, horses=46
- stages=5, horses=3, names=['Dune', 'Pixie', 'Ecume']
- stages=8, horses=1, names=['Dakota']

---

### Q55 — CLEAN

**(a) Question:**  
Combien de temps dure généralement la phase de récupération après une compétition ?

**(b) Current ground_truth:**  
19 chevaux ont une phase de transition (récupération) enregistrée. Elle dure 25 minutes pour 14 d'entre eux, et 30 minutes pour 5 chevaux : Apollon, Atlas, Dakota, Dune et Thunder.

**(c) Proposed correction:**  
50 chevaux ont une phase de transition (récupération) enregistrée. Elle dure 25 minutes pour 45 d'entre eux, et 30 minutes pour 5 chevaux : Apollon, Atlas, Dakota, Dune et Thunder.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)-[:TRAINSIN]->(t:TransitionStage)
RETURN t.Volume AS volume, COUNT(DISTINCT h) AS horses,
       COLLECT(DISTINCT h.hasName)[0..8] AS sample
ORDER BY volume
```
Live:
- `{'volume': '25min', 'horses': 45, ...}`
- `{'volume': '30min', 'horses': 5, 'sample': ['Dakota', 'Thunder', 'Dune', 'Apollon', 'Atlas']}`

```cypher
MATCH (h:Horse)-[:TRAINSIN]->(:TransitionStage)
RETURN COUNT(DISTINCT h) AS horses
```
Live: `{'horses': 50}`

---

### Q86 — CLEAN (number only)

**(a) Question:**  
Le programme d'entraînement et de compétition de Dakota ressemble-t-il à celui des autres chevaux du système ?

**(b) Current ground_truth:**  
Non, Dakota se distingue nettement des autres chevaux : il participe à 5 compétitions et suit 6 étapes d'entraînement, alors que la majorité des chevaux ne participent qu'à 2 compétitions et suivent 3 ou 4 étapes d'entraînement.

**(c) Proposed correction:**  
Non, Dakota se distingue nettement des autres chevaux : il participe à 5 compétitions et suit 8 étapes d'entraînement, alors que la majorité des chevaux ne participent qu'à 2 compétitions et suivent 4 étapes d'entraînement.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse {hasName:'Dakota'})
OPTIONAL MATCH (h)-[:COMPETESIN]->(e)
OPTIONAL MATCH (h)-[:TRAINSIN]->(t)
RETURN COUNT(DISTINCT e) AS comps, COUNT(DISTINCT t) AS stages
```
Live: `{'comps': 5, 'stages': 8}`

(Majority stage count from Q53: 46/50 horses at 4 stages.)

---

### Q93 — CLEAN (number only)

**(a) Question:**  
Les chevaux ayant un programme d'entraînement plus long ont-ils aussi tendance à participer à plus de compétitions ?

**(b) Current ground_truth:**  
Pas nécessairement, à une exception près. Dakota est le seul cheval où les deux coïncident : il a à la fois le plus grand nombre d'étapes d'entraînement (6) et le plus grand nombre de compétitions (5). Pour les autres chevaux, le nombre d'étapes (3 ou 4) et le nombre de compétitions (généralement 2) ne sont pas particulièrement corrélés cas par cas.

**(c) Proposed correction:**  
Pas nécessairement, à une exception près. Dakota est le seul cheval où les deux coïncident : il a à la fois le plus grand nombre d'étapes d'entraînement (8) et le plus grand nombre de compétitions (5). Pour les autres chevaux, le nombre d'étapes (surtout 4, parfois 5) et le nombre de compétitions (généralement 2) ne sont pas particulièrement corrélés cas par cas.

**(d) Exact Cypher + live result:**  
Same Dakota query as Q86 → 8 stages / 5 competitions. Distribution as Q53. Per-horse engage/result counts:
```cypher
MATCH (h:Horse)
OPTIONAL MATCH (h)-[:COMPETESIN]->(e)
OPTIONAL MATCH (p:EventParticipation)-[:HASHORSE]->(h)
WITH h, COUNT(DISTINCT e) AS eng, COUNT(DISTINCT p) AS res
RETURN eng, res, COUNT(h) AS horses
```
Live: (1,1)×2 ; (2,2)×47 ; (5,5)×1

---

### Q94 — CLEAN

**(a) Question:**  
Certains chevaux ont-ils des identifiants d'entraînement qui sortent du schéma habituel ?

**(b) Current ground_truth:**  
Oui, deux cas sortent du schéma de nommage habituel (Training_<Type>_<Cheval>_01) : Orion, dont les étapes sont nommées Training_Orion_LeMans_Preparation, Training_Orion_LeMans_PreComp et Training_Orion_LeMans_Comp, et Dakota, dont les étapes portent des identifiants centrés sur SJ comme Training_Preparation_SJ_01, Training_PreCompetition_SJ_01, Training_PreComp_SJ_01, Training_Prepa_SJ_01, Training_Competition_SJ_01 et Training_Transition_SJ_01.

**(c) Proposed correction:**  
Oui, deux cas sortent du schéma de nommage habituel (Training_<Type>_<Cheval>_01) : Orion, dont les étapes sont nommées Training_Orion_LeMans_Preparation, Training_Orion_LeMans_PreComp, Training_Orion_LeMans_Comp et Training_Transition_Orion_01, et Dakota, dont les étapes portent des identifiants centrés sur SJ ou des ids de compétition : Training_Preparation_SJ_01, Training_Prepa_SJ_01, Training_PreCompetition_SJ_01, Training_PreComp_SJ_01, Training_Competition_SJ_01, Training_Comp_Cross2026_Dakota_01, Training_Comp_Dress01_Dakota_01 et Training_Transition_SJ_01.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)-[:TRAINSIN]->(t)
WHERE h.hasName IN ['Dakota','Orion']
RETURN h.hasName AS horse, COLLECT(t.id) AS ids
ORDER BY horse
```
Live:
- Dakota: 8 ids including `Training_Comp_Cross2026_Dakota_01`, `Training_Comp_Dress01_Dakota_01`
- Orion: `Training_Orion_LeMans_Comp`, `Training_Orion_LeMans_Preparation`, `Training_Orion_LeMans_PreComp`, `Training_Transition_Orion_01`

---

## Category C — engagements always have results

### Q62 — CLEAN

**(a) Question:**  
Y a-t-il des compétitions où des chevaux étaient engagés mais où aucun résultat officiel n'a été enregistré ?

**(b) Current ground_truth:**  
Oui, trois événements ont des chevaux engagés sans qu'aucun classement officiel n'ait été enregistré : Event_SJ_2026_01, Event_Cross_2026_01 et Event_Dressage_01.

**(c) Proposed correction:**  
Non — dans le graphe actuel, chaque engagement a un résultat officiel enregistré.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)-[:COMPETESIN]->(e)
OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
WITH e, COUNT(DISTINCT h) AS entrants, COUNT(DISTINCT p) AS ranked
WHERE ranked = 0
RETURN e.id AS event, entrants, ranked
ORDER BY entrants DESC
```
Live: **0 rows**

---

### Q65 — CLEAN

**(a) Question:**  
Est-il fréquent qu'un cheval soit engagé dans une compétition sans qu'un résultat officiel y soit ensuite enregistré ?

**(b) Current ground_truth:**  
Oui, c'est même le cas le plus courant : 48 chevaux sur 50 ont au moins un engagement en compétition sans résultat officiel correspondant. Seuls Naya et Orion font exception, car ils n'ont chacun qu'une seule compétition et un résultat officiel associé.

**(c) Proposed correction:**  
Non, ce n'est pas fréquent : le cas ne se produit jamais. Pour chaque cheval, le nombre d'engagements COMPETESIN égale le nombre de résultats EventParticipation.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)-[:COMPETESIN]->(e)
OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h)
WITH h, e, COUNT(p) AS ranked
WHERE ranked = 0
RETURN h.hasName AS horse, e.id AS event, ranked
ORDER BY horse
```
Live: **0 rows**

---

### Q67 — CLEAN

**(a) Question:**  
Quelle compétition a le plus grand nombre de chevaux engagés sans résultat officiel enregistré ?

**(b) Current ground_truth:**  
Event_SJ_2026_01 a l'écart le plus important : 7 chevaux y sont engagés, mais aucun résultat officiel n'a été enregistré pour cet événement.

**(c) Proposed correction:**  
Aucune compétition : tous les engagements ont un résultat. Event_SJ_2026_01 a bien 7 participations enregistrées.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)-[:COMPETESIN]->(e {id:'Event_SJ_2026_01'})
OPTIONAL MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h)
WITH h, COUNT(p) AS ranked
WHERE ranked = 0
RETURN h.hasName AS horse, ranked
```
Live: **0 rows**

```cypher
MATCH (e {id:'Event_SJ_2026_01'})-[:HASPARTICIPATION]->(p)
RETURN e.id AS event, COUNT(p) AS participations
```
Live: `{'event': 'Event_SJ_2026_01', 'participations': 7}`

---

### Q97 — CLEAN

**(a) Question:**  
Est-ce que tous les chevaux du système ont au moins un résultat de compétition enregistré ?

**(b) Current ground_truth:**  
Oui, chacun des 50 chevaux a exactement un résultat officiel enregistré, même si beaucoup d'entre eux ont participé à d'autres compétitions pour lesquelles aucun classement n'a été renseigné.

**(c) Proposed correction:**  
Oui, les 50 chevaux ont au moins un résultat. Répartition : 2 chevaux ×1 (Naya, Orion), 47 ×2, Dakota ×5. Il n'y a plus d'engagements sans classement.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)
OPTIONAL MATCH (h)-[:COMPETESIN]->(e)
OPTIONAL MATCH (p:EventParticipation)-[:HASHORSE]->(h)
WITH h, COUNT(DISTINCT e) AS engages, COUNT(DISTINCT p) AS results
RETURN engages, results, COUNT(h) AS horses,
       COLLECT(h.hasName)[0..5] AS sample
ORDER BY engages, results
```
Live:
- `{'engages': 1, 'results': 1, 'horses': 2, 'sample': ['Naya', 'Orion']}`
- `{'engages': 2, 'results': 2, 'horses': 47, ...}`
- `{'engages': 5, 'results': 5, 'horses': 1, 'sample': ['Dakota']}`

---

### Q98 — CLEAN

**(a) Question:**  
Si un cheval est inscrit à une compétition, est-on certain qu'un classement existe pour cette compétition précise ?

**(b) Current ground_truth:**  
Non, ce n'est pas garanti. L'inscription à une compétition et l'existence d'un classement officiel sont deux informations distinctes dans le système : un cheval peut très bien être inscrit (COMPETES_IN) sans qu'un résultat classé (HAS_PARTICIPATION) ne soit enregistré pour cette compétition précise — c'est même le cas le plus fréquent.

**(c) Proposed correction:**  
Oui — dans le graphe actuel, pour chaque inscription COMPETESIN il existe un classement (EventParticipation) pour ce cheval et cet événement.

**(d) Exact Cypher + live result:**  
Same anti-join as Q65 → **0 rows** (no engagement without a matching participation for that horse).

---

## Category D — shifted counts

### Q23 — CLEAN

**(a) Question:**  
Qui participe à la phase pré-compétition de l'entraînement ?

**(b) Current ground_truth:**  
Les participants à la phase pré-compétition incluent Sophie (soigneuse), Dr Martin (vétérinaire), et 24 des 25 cavaliers du système (tous sauf Emma, remplacée par Manon pour la pré-compétition de Dakota) : Alice, Antoine, Baptiste, Camille, Chloe, Clara, Elise, Hugo, Ines, Jade, Julien, Lea, Leo, Lucas, Manon, Marc, Maxime, Nina, Pauline, Remi, Sarah, Theo, Tom et Victor.

**(c) Proposed correction:**  
Les participants à la phase pré-compétition incluent Sophie (soigneuse), Dr Martin (vétérinaire), et les 25 cavaliers du système (Emma et Manon sont toutes deux présentes, chacune sur une étape pré-compétition distincte de Dakota) : Alice, Antoine, Baptiste, Camille, Chloe, Clara, Elise, Emma, Hugo, Ines, Jade, Julien, Lea, Leo, Lucas, Manon, Marc, Maxime, Nina, Pauline, Remi, Sarah, Theo, Tom et Victor.

**(d) Exact Cypher + live result:**
```cypher
MATCH (t:PreCompetitionStage)-[:INVOLVESACTOR]->(r:Rider)
RETURN COUNT(DISTINCT r) AS riders, COLLECT(DISTINCT r.id) AS rider_ids
```
Live: `riders=25`, ids include both `Rider_Emma` and `Rider_Manon`.

```cypher
MATCH (t:PreCompetitionStage)-[:INVOLVESACTOR]->(r:Rider)
WHERE r.id IN ['Rider_Emma','Rider_Manon']
OPTIONAL MATCH (h:Horse)-[:TRAINSIN]->(t)
RETURN r.id AS rider, t.id AS stage, COLLECT(DISTINCT h.hasName) AS horses
ORDER BY rider, stage
```
Live:
- `Rider_Emma` → `Training_PreCompetition_SJ_01` (Dakota)
- `Rider_Manon` → `Training_PreComp_SJ_01` (Dakota)

```cypher
MATCH (t:PreCompetitionStage)-[:INVOLVESACTOR]->(a)
WHERE a:Caretaker OR a:Veterinarian OR a:Rider
RETURN labels(a)[0] AS role, COUNT(DISTINCT a) AS n
```
Live: Rider 25, Caretaker 1, Veterinarian 1

---

### Q34 — PARTIAL (prep OK / pré-comp stale)

**(a) Question:**  
Compare les acteurs impliqués dans les phases de préparation et de pré-compétition.

**(b) Current ground_truth:**  
La phase de préparation implique 24 des 25 cavaliers (tous sauf Manon, y compris Emma pour Dakota), Sophie (soigneuse) et Dr Martin (vétérinaire). La phase pré-compétition implique le même ensemble d'acteurs, à ceci près que Manon remplace Emma pour la pré-compétition de Dakota : les cavaliers y sont donc au nombre de 24 également, mais avec Manon à la place d'Emma.

**(c) Proposed correction:**  
La phase de préparation implique 24 des 25 cavaliers (tous sauf Manon, y compris Emma pour Dakota), Sophie (soigneuse) et Dr Martin (vétérinaire). La phase pré-compétition implique Sophie, Dr Martin et les 25 cavaliers : Emma et Manon y sont toutes deux présentes (chacune sur une étape pré-compétition distincte de Dakota).

**(d) Exact Cypher + live result:**
```cypher
MATCH (t:PreparationStage)-[:INVOLVESACTOR]->(r:Rider)
RETURN COUNT(DISTINCT r) AS riders
```
Live: `{'riders': 24}`

```cypher
MATCH (r:Rider)
OPTIONAL MATCH (t:PreparationStage)-[:INVOLVESACTOR]->(r)
WITH r, COUNT(t) AS n
RETURN r.id AS rider, n
ORDER BY n, rider
```
Live: `Rider_Manon` has n=0 (absent from prep); Emma present on Dakota prep stages.

Pre-competition evidence: same as Q23 (25 riders, both Emma and Manon).

---

### Q57 — CLEAN

**(a) Question:**  
Y a-t-il des étapes d'entraînement où aucun encadrant (cavalier, vétérinaire ou soigneur) n'est mentionné ?

**(b) Current ground_truth:**  
Oui, cinq étapes n'ont aucun encadrant humain enregistré : Training_Preparation_SJ_01, Training_PreCompetition_SJ_01, Training_Competition_SJ_01, Training_Transition_SJ_01 et Training_Transition_Thunder_01.

**(c) Proposed correction:**  
Non — chaque étape d'entraînement a au moins un encadrant (cavalier, vétérinaire ou soigneur).

**(d) Exact Cypher + live result:**
```cypher
MATCH (t)
WHERE t:PreparationStage OR t:PreCompetitionStage
   OR t:CompetitionStage OR t:TransitionStage
OPTIONAL MATCH (t)-[:INVOLVESACTOR]->(a)
WHERE a:Rider OR a:Veterinarian OR a:Caretaker
WITH t, COUNT(a) AS supervisors
WHERE supervisors = 0
RETURN labels(t)[0] AS type, t.id AS stage, supervisors
```
Live: **0 rows**

---

### Q63 — CLEAN

**(a) Question:**  
Quel événement de la saison a réuni le plus de résultats classés ?

**(b) Current ground_truth:**  
Event_Montpellier_Dr_2026 est l'événement avec le plus grand nombre de résultats officiels enregistrés : six chevaux y ont un classement.

**(c) Proposed correction:**  
Trois événements sont à égalité en tête avec 7 résultats classés : Event_Pau_SJ_2026, Event_LeMans_Cross_2026 et Event_SJ_2026_01. Event_Montpellier_Dr_2026 en a 6.

**(d) Exact Cypher + live result:**
```cypher
MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)
MATCH (e)-[:INSEASON]->(:CompetitiveSeason {seasonName:'Saison 2026'})
WITH e, COUNT(DISTINCT p) AS result_count
RETURN e.id AS event, result_count
ORDER BY result_count DESC, event
LIMIT 10
```
Live (top):
- Event_LeMans_Cross_2026 → 7
- Event_Pau_SJ_2026 → 7
- Event_SJ_2026_01 → 7
- Event_Dressage_2026_01 → 6
- Event_Marseille_Dressage_2026 → 6
- Event_Montpellier_Dr_2026 → 6
- Event_Rennes_SJ_2026 → 6
- …

---

### Q66 — CLEAN

**(a) Question:**  
Arrive-t-il qu'un même cavalier présente deux chevaux différents lors du même événement ?

**(b) Current ground_truth:**  
Oui, cela arrive dans neuf cas : Alice à Nantes (Soleil, Arrow), Remi à Versailles (Etoile, Grondre), Lea à Rennes (Rio, Falcon), Camille à Toulouse (Atlas, Iris), Elise à Montpellier (Luminos, Nuage), Maxime à Dijon (Tempete, Ecume), Ines à Clermont-Ferrand (Rafale, Tonnerre), Marc à Strasbourg (Dune, Sable) et Hugo à Lyon (Nova, Storm).

**(c) Proposed correction:**  
Oui, cela arrive dans onze cas : Alice à Nantes (Soleil, Arrow), Remi à Versailles (Etoile, Grondre), Remi à Pau (Grondre, Galop), Lea à Rennes (Rio, Falcon), Camille à Toulouse (Atlas, Iris), Elise à Montpellier (Luminos, Nuage), Maxime à Dijon (Tempete, Ecume), Ines à Clermont-Ferrand (Rafale, Tonnerre), Marc à Strasbourg (Dune, Sable), Hugo à Lyon (Nova, Storm) et Victor à Event_SJ_2026_01 (Cascade, Lumiere).

**(d) Exact Cypher + live result:**
```cypher
MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASRIDER]->(r:Rider)
MATCH (p)-[:HASHORSE]->(h:Horse)
WITH e, r, COUNT(DISTINCT h) AS horse_count, COLLECT(DISTINCT h.hasName) AS horses
WHERE horse_count > 1
RETURN e.id AS event, r.id AS rider, horse_count, horses
ORDER BY event, rider
```
Live (**11 rows**):
1. Event_Clermont_Cross_2026 / Rider_Ines → Rafale, Tonnerre
2. Event_Dijon_Cross_2026 / Rider_Maxime → Tempete, Ecume
3. Event_Lyon_Dressage_2026 / Rider_Hugo → Storm, Nova
4. Event_Montpellier_Dr_2026 / Rider_Elise → Nuage, Luminos
5. Event_Nantes_SJ_2026 / Rider_Alice → Arrow, Soleil
6. Event_Pau_SJ_2026 / Rider_Remi → Grondre, Galop  ← **new**
7. Event_Rennes_SJ_2026 / Rider_Lea → Falcon, Rio
8. Event_SJ_2026_01 / Rider_Victor → Cascade, Lumiere  ← **new**
9. Event_Strasbourg_Cross_2026 / Rider_Marc → Dune, Sable
10. Event_Toulouse_Cross_2026 / Rider_Camille → Iris, Atlas
11. Event_Versailles_SJ_2026 / Rider_Remi → Grondre, Etoile

---

### Q69 — CLEAN

**(a) Question:**  
Quels sont les liens les plus fréquents dans le système : entre l'entraînement et les encadrants, entre les chevaux et les événements, ou ailleurs ?

**(b) Current ground_truth:**  
Le lien le plus fréquent est celui entre une étape d'entraînement et ses encadrants (INVOLVES_ACTOR, 314 occurrences), suivi par les liens entre une étape d'entraînement et l'événement qu'elle prépare (DEPENDS_ON, 171) et entre un cheval et ses étapes d'entraînement (TRAINS_IN, 171). Viennent ensuite les liens entre capteurs et chevaux ou objectifs (IS_ATTACHED_TO et IS_USED_FOR, 108 chacun), puis les engagements en compétition (COMPETES_IN, 101).

**(c) Proposed correction:**  
Le lien le plus fréquent est celui entre une étape d'entraînement et ses encadrants (INVOLVESACTOR, 355 occurrences), suivi par TRAINSIN (207) et DEPENDSON (207). Viennent ensuite ISATTACHEDTO et ISUSEDFOR (108 chacun), puis les engagements en compétition (COMPETESIN, 101).

**(d) Exact Cypher + live result:**
```cypher
MATCH ()-[r]->()
RETURN type(r) AS relationship, COUNT(r) AS n
ORDER BY n DESC
```
Live:
- INVOLVESACTOR 355
- TRAINSIN 207
- DEPENDSON 207
- ISATTACHEDTO 108
- ISUSEDFOR 108
- COMPETESIN 101
- HASPARTICIPATION 101
- HASRIDER 101
- HASHORSE 101
- ASSOCIATEDWITH 51
- INSEASON 20

---

### Q8 — NO CHANGE (do not force a fix)

**(a) Question:**  
Quelle est la durée des séances pendant la phase de préparation ?

**(b) Current ground_truth:**  
Pendant la phase de préparation, les durées de séances varient : 40 minutes pour Pixie (la plus courte), 45 minutes pour 31 chevaux (Dakota, Naya, Thunder, Bella, Zephyr, Nova, Iris, Mistral, Falcon, Sable, Sirius, Brume, Eclair, Ombre, Arrow, Tempete, Nuage, Etoile, Rafale, Cascade, Aurore, Crepuscule, Glacier, Braise, Zephire, Galop, Horizon, Lumiere, Mirage, Orage, Nebule), 50 minutes pour 16 chevaux (Orion, Apollon, Atlas, Auroch, Storm, Rio, Dune, Eclat, Luminos, Grondre, Riviere, Luna, Soleil, Tonnerre, Vega, Volcan), et 55 minutes pour Comet et Ecume, à égalité pour la durée la plus longue.

**(c) Proposed correction:**  
**None — keep current GT.** Horse-level counts and the 31-name / 16-name sets match live exactly. There are 32 PreparationStage nodes at 45min only because Dakota has two prep stages both at 45min; the question is about séance duration per horse.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage)
RETURN t.Volume AS volume, COUNT(DISTINCT h) AS horses, COUNT(DISTINCT t) AS stages,
       COLLECT(DISTINCT h.hasName) AS names
ORDER BY volume
```
Live:
- 40min → 1 horse / 1 stage (Pixie)
- 45min → **31 horses / 32 stages** (name set equals GT list exactly)
- 50min → 16 / 16
- 55min → 2 / 2 (Comet, Ecume)

```cypher
MATCH (h:Horse)-[:TRAINSIN]->(t:PreparationStage)
WITH h, COUNT(t) AS n, COLLECT(t.id) AS ids, COLLECT(t.Volume) AS vols
WHERE n > 1
RETURN h.hasName AS horse, n, ids, vols
```
Live: only Dakota, n=2, both volumes `45min`.

---

## Category — training DEPENDSON events (EX gold lock, 2026-07-26)

### Q10 — CLEAN (INCORRECT NL GT — not V9 drift)

**(a) Question:**  
De quel événement dépendent les étapes d'entraînement de Dakota ?

**(b) Current ground_truth:**  
Les étapes d'entraînement de Dakota dépendent de deux événements : Event_SJ_01 (saut d'obstacles à Saumur le 12 avril 2026) et Event_SJ_2026_01 (saut d'obstacles à Paris le 14 juin 2026).

**(c) Proposed correction:**  
Les étapes d'entraînement de Dakota dépendent de quatre événements : Event_SJ_01, Event_SJ_2026_01, Event_Cross_2026_01, et Event_Dressage_01.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse {hasName:'Dakota'})-[:TRAINSIN]->(t)-[:DEPENDSON]->(e)
RETURN COUNT(DISTINCT e) AS n, COLLECT(DISTINCT e.id) AS ids
```
Live: `{'n': 4, 'ids': ['Event_SJ_01', 'Event_SJ_2026_01', 'Event_Cross_2026_01', 'Event_Dressage_01']}`  
(order of `ids` may vary; EX compares as a multiset)

Full stage→event mapping (8 stages → 4 distinct events because several stages share a target):

| training_id | DEPENDSON → event |
|---|---|
| Training_Prepa_SJ_01 | Event_SJ_01 |
| Training_PreComp_SJ_01 | Event_SJ_01 |
| Training_Preparation_SJ_01 | Event_SJ_2026_01 |
| Training_PreCompetition_SJ_01 | Event_SJ_2026_01 |
| Training_Competition_SJ_01 | Event_SJ_2026_01 |
| Training_Transition_SJ_01 | Event_SJ_2026_01 |
| Training_Comp_Cross2026_Dakota_01 | Event_Cross_2026_01 |
| Training_Comp_Dress01_Dakota_01 | Event_Dressage_01 |

**Classification:** **INCORRECT** — not “stale from a V9 graph upgrade.” The two omitted events (`Event_Cross_2026_01`, `Event_Dressage_01`) are pointed to by Dakota’s competition-stage trainings via ordinary `DEPENDSON` edges; nothing in this discrepancy depends on V9 identifier renames. Staged for sign-off only — **`data/test_dataset.json` not modified**.

---

## Category — EX gold campaign follow-ups (2026-07-26, staging only)

### Q64 — CLEAN (factual; Dakota stage count missed earlier)

**(a) Question:**  
La plupart des chevaux participent-ils au même nombre de compétitions dans la saison, ou y a-t-il des exceptions ?

**(b) Current ground_truth:**  
La majorité des chevaux sont engagés dans deux compétitions. Trois chevaux font exception : Dakota, engagé dans cinq compétitions (sa charge la plus élevée du système, à comparer avec ses 6 étapes d'entraînement, également la plus élevée), tandis que Naya et Orion ne sont engagés que dans une seule compétition chacun.

**(c) Proposed correction:**  
La majorité des chevaux sont engagés dans deux compétitions. Trois chevaux font exception : Dakota, engagée dans cinq compétitions (sa charge la plus élevée du système, à comparer avec ses 8 étapes d'entraînement, également la plus élevée), tandis que Naya et Orion ne sont engagés que dans une seule compétition chacun.

**(d) Exact Cypher + live result:**
```cypher
MATCH (h:Horse)-[:COMPETESIN]->(e)
WITH h, COUNT(DISTINCT e) AS event_count
RETURN event_count, COUNT(h) AS horses, COLLECT(h.hasName) AS names
ORDER BY event_count
```
Live: 1→2 [Naya, Orion]; 2→47; 5→1 [Dakota]

```cypher
MATCH (h:Horse {hasName:'Dakota'})
OPTIONAL MATCH (h)-[:COMPETESIN]->(e)
OPTIONAL MATCH (h)-[:TRAINSIN]->(t)
RETURN COUNT(DISTINCT e) AS comps, COUNT(DISTINCT t) AS stages
```
Live: `{'comps': 5, 'stages': 8}`

**Classification:** Factual correction of the Dakota stage-count aside (same V9 fact as Q5/Q53/Q86). Competition histogram was already correct. Staged only — **`data/test_dataset.json` not modified**.

---

### Q76 — SCOPE/PRECISION (extra review — not data-drift)

**(a) Question:**  
Qui peut encadrer une séance d'entraînement, en dehors du cavalier ?

**(b) Current ground_truth:**  
Une séance d'entraînement peut impliquer trois types d'encadrants : le cavalier (Rider), le vétérinaire (Veterinarian) et le soigneur (Caretaker).

**(c) Proposed correction:**  
Deux autres types d'encadrants peuvent intervenir lors d'une séance d'entraînement, en plus du cavalier : le vétérinaire (Veterinarian) et le soigneur (Caretaker).

**(d) Exact Cypher + live result:**
```cypher
MATCH (t)-[:INVOLVESACTOR]->(a)
WHERE a:Veterinarian OR a:Caretaker
RETURN DISTINCT labels(a)[0] AS role
```
Live: Caretaker, Veterinarian

**Classification:** **SCOPE/PRECISION**, not graph data-drift. The question explicitly excludes the rider (“en dehors du cavalier”); the current GT answers the broader unscoped question and lists Rider. Flagged for **extra review** given the more interpretive wording. Staged only — **`data/test_dataset.json` not modified**.

---

### Q89 — CLEAN (factual; DEPENDSON name + count missed earlier)

**(a) Question:**  
Comment le système relie-t-il une étape d'entraînement à l'événement qu'elle prépare ?

**(b) Current ground_truth:**  
Le lien se fait via la relation DEPENDS_ON, qui relie chaque étape d'entraînement à l'événement pour lequel elle prépare le cheval. C'est ce lien (171 occurrences au total) qui permet de savoir pour quelle compétition un cheval s'entraîne à un moment donné. Une ancienne relation redondante (hasParticipatedTo) exprimait partiellement la même idée mais a été retirée du graphe, car elle n'apportait aucune information que DEPENDS_ON ne couvrait pas déjà.

**(c) Proposed correction:**  
Le lien se fait via la relation DEPENDSON, qui relie chaque étape d'entraînement à l'événement pour lequel elle prépare le cheval. C'est ce lien (207 occurrences au total) qui permet de savoir pour quelle compétition un cheval s'entraîne à un moment donné. Une ancienne relation redondante (hasParticipatedTo) exprimait partiellement la même idée mais a été retirée du graphe, car elle n'apportait aucune information que DEPENDSON ne couvrait pas déjà.

**(d) Exact Cypher + live result:**
```cypher
MATCH ()-[r:DEPENDSON]->() RETURN COUNT(r) AS n
```
Live: `{'n': 207}`

```cypher
MATCH ()-[r:DEPENDS_ON]->() RETURN COUNT(r) AS n
```
Live: `{'n': 0}` (relationship type `DEPENDS_ON` does not exist)

**Classification:** Factual correction of the relation name (`DEPENDS_ON` → `DEPENDSON`) and count (171 → 207) — same V8-era stale-count pattern as Q5/Q52/Q53/Q55/Q64/Q86/Q93/Q94. Missed for Q89 specifically because it was outside the original reconciliation pass’s scope. Staged only — **`data/test_dataset.json` not modified**.

---

## Items NOT proposed (confirmed still correct on V9)

Left as-is per prior inventory + spot checks where relevant:  
Q6, Q16, Q22, Q41, Q43, Q47, Q49, Q50, Q51, Q54, Q56, Q58, Q59, Q60, Q61, Q78, Q79, Q80, Q83, Q84, Q85, Q90, Q91, Q92, Q96 — and **Q8** above.  
(**Q64** removed from this list — now staged above.)

---

## Ask for sign-off

Please reply with which of the following you approve:

1. **Approve all CLEAN corrections** (including Q10, Q64, Q89; Q8 excluded; Q34 partial as written; Q76 SCOPE/PRECISION separately)  
2. **Approve a subset** (list qids)  
3. **Request wording tweaks** before any write to `test_dataset.json`

Only after explicit approval will Track 1 write to `data/test_dataset.json`.  
**Track 2 (prompt work) stays blocked until then.**
