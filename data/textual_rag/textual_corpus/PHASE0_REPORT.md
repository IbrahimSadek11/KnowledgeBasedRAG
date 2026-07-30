# PHASE 0 — Factual textual corpus report

**Generated at:** 2026-07-27T00:24:23.663903+00:00
**Source:** live Neo4j V9 (read-only)
**Counts:** 50 horse docs + 20 event docs = 70

## Fact-sheet format

### Horse fact-sheet (`fact_sheets/horses/<Name>.json`)
```
{
  "doc_type": "horse_fact_sheet",
  "source": "live_neo4j_v9",
  "horse": {
    "id": "...",
    "hasName": "...",
    "hasRace": "..."
  },
  "associated_riders": [
    "Rider_..."
  ],
  "training_stages": [
    {
      "stage_id": "...",
      "stage_type": "PreparationStage|PreCompetitionStage|CompetitionStage|TransitionStage",
      "volume": "...",
      "intensity": "...",
      "frequency": "...",
      "depends_on_events": [
        "Event_..."
      ],
      "actors": [
        {
          "role": "Rider|Veterinarian|Caretaker",
          "actor_id": "..."
        }
      ]
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
      "objective": "..."
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
      "rider_id": "..."
    }
  ]
}
```

### Event fact-sheet (`fact_sheets/events/<EventId>.json`)
```
{
  "doc_type": "event_fact_sheet",
  "event": {
    "id": "...",
    "discipline": "ShowJumping|Dressage|Cross",
    "category": "...",
    "event_date": "YYYY-MM-DD",
    "location": "...",
    "season": "..."
  },
  "participants": [
    {
      "horse_name": "...",
      "horse_id": "...",
      "horse_race": "...",
      "rider_id": "...",
      "rank": 0,
      "status": "..."
    }
  ],
  "linked_training_summary": [
    {
      "stage_type": "...",
      "stage_count": 0,
      "horse_count": 0
    }
  ]
}
```

## Generation method

Prose is rendered by deterministic French templates that interpolate only fields present on the fact-sheet. No LLM is used, so ranks, sensor rates, volumes, and participants cannot be invented.

## Three full examples (prose ↔ fact-sheet)

### Example 1 — horse_Sirius (PASS)

#### Prose
```
Rapport d'entraînement — Sirius
===================================

Ce rapport synthétise exclusivement les faits enregistrés dans le graphe de connaissances pour le cheval Sirius (identifiant Horse_Sirius), race Frison.

Cavaliers associés (relation ASSOCIATEDWITH) : Rider_Baptiste.

Programme d'entraînement (4 étape(s) enregistrée(s))
----------------------------------------

Phase compétition (entraînement) (CompetitionStage) — 1 séance(s) :
  • Training_Comp_Sirius_01 : volume 30min, intensité Pic, fréquence 1 ; acteurs : cavalier Rider_Baptiste ; événements liés : Event_Nice_Dressage_2026.

Phase pré-compétition (PreCompetitionStage) — 1 séance(s) :
  • Training_PreComp_Sirius_01 : volume 60min, intensité Élevée, fréquence 3 ; acteurs : cavalier Rider_Baptiste, vétérinaire Vet_DrMartin ; événements liés : Event_Nice_Dressage_2026.

Phase préparation (PreparationStage) — 1 séance(s) :
  • Training_Prepa_Sirius_01 : volume 45min, intensité Modérée, fréquence 4 ; acteurs : soigneur Caretaker_Sophie, cavalier Rider_Baptiste, vétérinaire Vet_DrMartin ; événements liés : Event_Nice_Dressage_2026.

Phase transition / récupération (TransitionStage) — 1 séance(s) :
  • Training_Transition_Sirius_01 : volume 25min, intensité Faible, fréquence 2 ; acteurs : cavalier Rider_Baptiste ; événements liés : Event_Nice_Dressage_2026.

Capteurs inertiels (2 capteur(s) attaché(s))
----------------------------------------
  • IMU_Sternum_Sirius_01 en position sternum (Sternum), fréquence d'échantillonnage 200Hz, offset 0.02, format CSV, taille de fichier 6400, objectif expérimental FatigueDetection.
  • IMU_Withers_Sirius_01 en position garrot (Withers), fréquence d'échantillonnage 200Hz, offset 0.01, format CSV, taille de fichier 5700, objectif expérimental GaitClassif_01.

Participations en compétition (2 résultat(s))
----------------------------------------
  • Event_Lyon_Dressage_2026 (dressage / Dressage, catégorie « Club Elite ») le 2026-08-23 à Lyon : rang 3, statut « Terminé », cavalier Rider_Baptiste.
  • Event_Nice_Dressage_2026 (dressage / Dressage, catégorie « Pro Elite ») le 2026-09-28 à Nice : rang 1, statut « Terminé », cavalier Rider_Baptiste.

Note méthodologique : aucun rang, volume, intensité, fréquence, capteur ou participant n'a été inventé ; toutes les valeurs ci-dessus proviennent du fact-sheet extrait de Neo4j V9.
```

#### Source fact-sheet
```json
{
  "doc_type": "horse_fact_sheet",
  "source": "live_neo4j_v9",
  "extracted_at": "2026-07-27T00:24:23.323370+00:00",
  "horse": {
    "id": "Horse_Sirius",
    "hasName": "Sirius",
    "hasRace": "Frison"
  },
  "associated_riders": [
    "Rider_Baptiste"
  ],
  "training_stages": [
    {
      "stage_id": "Training_Comp_Sirius_01",
      "stage_type": "CompetitionStage",
      "volume": "30min",
      "intensity": "Pic",
      "frequency": 1,
      "depends_on_events": [
        "Event_Nice_Dressage_2026"
      ],
      "actors": [
        {
          "actor_id": "Rider_Baptiste",
          "role": "Rider"
        }
      ]
    },
    {
      "stage_id": "Training_PreComp_Sirius_01",
      "stage_type": "PreCompetitionStage",
      "volume": "60min",
      "intensity": "Élevée",
      "frequency": 3,
      "depends_on_events": [
        "Event_Nice_Dressage_2026"
      ],
      "actors": [
        {
          "actor_id": "Rider_Baptiste",
          "role": "Rider"
        },
        {
          "actor_id": "Vet_DrMartin",
          "role": "Veterinarian"
        }
      ]
    },
    {
      "stage_id": "Training_Prepa_Sirius_01",
      "stage_type": "PreparationStage",
      "volume": "45min",
      "intensity": "Modérée",
      "frequency": 4,
      "depends_on_events": [
        "Event_Nice_Dressage_2026"
      ],
      "actors": [
        {
          "actor_id": "Caretaker_Sophie",
          "role": "Caretaker"
        },
        {
          "actor_id": "Rider_Baptiste",
          "role": "Rider"
        },
        {
          "actor_id": "Vet_DrMartin",
          "role": "Veterinarian"
        }
      ]
    },
    {
      "stage_id": "Training_Transition_Sirius_01",
      "stage_type": "TransitionStage",
      "volume": "25min",
      "intensity": "Faible",
      "frequency": 2,
      "depends_on_events": [
        "Event_Nice_Dressage_2026"
      ],
      "actors": [
        {
          "actor_id": "Rider_Baptiste",
          "role": "Rider"
        }
      ]
    }
  ],
  "sensors": [
    {
      "sensor_id": "IMU_Sternum_Sirius_01",
      "sensor_code": "IMU-ST-016",
      "position": "Sternum",
      "sample_rate": "200Hz",
      "offset": "0.02",
      "format": "CSV",
      "file_size": 6400,
      "objective": "FatigueDetection"
    },
    {
      "sensor_id": "IMU_Withers_Sirius_01",
      "sensor_code": "IMU-W-021",
      "position": "Withers",
      "sample_rate": "200Hz",
      "offset": "0.01",
      "format": "CSV",
      "file_size": 5700,
      "objective": "GaitClassif_01"
    }
  ],
  "event_participations": [
    {
      "event_id": "Event_Lyon_Dressage_2026",
      "discipline": "Dressage",
      "category": "Club Elite",
      "event_date": "2026-08-23",
      "location": "Lyon",
      "rank": 3,
      "status": "Terminé",
      "rider_id": "Rider_Baptiste"
    },
    {
      "event_id": "Event_Nice_Dressage_2026",
      "discipline": "Dressage",
      "category": "Pro Elite",
      "event_date": "2026-09-28",
      "location": "Nice",
      "rank": 1,
      "status": "Terminé",
      "rider_id": "Rider_Baptiste"
    }
  ]
}
```

### Example 2 — horse_Riviere (PASS)

#### Prose
```
Rapport d'entraînement — Riviere
====================================

Ce rapport synthétise exclusivement les faits enregistrés dans le graphe de connaissances pour le cheval Riviere (identifiant Horse_Riviere), race Camargue.

Cavaliers associés (relation ASSOCIATEDWITH) : Rider_Victor.

Programme d'entraînement (4 étape(s) enregistrée(s))
----------------------------------------

Phase compétition (entraînement) (CompetitionStage) — 1 séance(s) :
  • Training_Comp_Riviere_01 : volume 30min, intensité Pic, fréquence 1 ; acteurs : cavalier Rider_Victor ; événements liés : Event_Dijon_Cross_2026.

Phase pré-compétition (PreCompetitionStage) — 1 séance(s) :
  • Training_PreComp_Riviere_01 : volume 70min, intensité Élevée, fréquence 4 ; acteurs : cavalier Rider_Victor, vétérinaire Vet_DrMartin ; événements liés : Event_Dijon_Cross_2026.

Phase préparation (PreparationStage) — 1 séance(s) :
  • Training_Prepa_Riviere_01 : volume 50min, intensité Modérée, fréquence 5 ; acteurs : cavalier Rider_Victor, soigneur Caretaker_Sophie, vétérinaire Vet_DrMartin ; événements liés : Event_Dijon_Cross_2026.

Phase transition / récupération (TransitionStage) — 1 séance(s) :
  • Training_Transition_Riviere_01 : volume 25min, intensité Faible, fréquence 2 ; acteurs : cavalier Rider_Victor ; événements liés : Event_Dijon_Cross_2026.

Capteurs inertiels (2 capteur(s) attaché(s))
----------------------------------------
  • IMU_CanonFore_Riviere_01 en position canon antérieur (CanonOfForelimb), fréquence d'échantillonnage 250Hz, offset 0.02, format CSV, taille de fichier 7800, objectif expérimental FatigueDetection.
  • IMU_Withers_Riviere_01 en position garrot (Withers), fréquence d'échantillonnage 200Hz, offset 0.01, format CSV, taille de fichier 5500, objectif expérimental GaitClassif_01.

Participations en compétition (2 résultat(s))
----------------------------------------
  • Event_Dijon_Cross_2026 (cross / concours complet / Cross, catégorie « Amateur 1 ») le 2026-06-06 à Dijon : rang 4, statut « Terminé », cavalier Rider_Victor.
  • Event_Strasbourg_Cross_2026 (cross / concours complet / Cross, catégorie « Amateur 1 ») le 2026-07-12 à Strasbourg : rang 4, statut « Terminé », cavalier Rider_Victor.

Note méthodologique : aucun rang, volume, intensité, fréquence, capteur ou participant n'a été inventé ; toutes les valeurs ci-dessus proviennent du fact-sheet extrait de Neo4j V9.
```

#### Source fact-sheet
```json
{
  "doc_type": "horse_fact_sheet",
  "source": "live_neo4j_v9",
  "extracted_at": "2026-07-27T00:24:23.293987+00:00",
  "horse": {
    "id": "Horse_Riviere",
    "hasName": "Riviere",
    "hasRace": "Camargue"
  },
  "associated_riders": [
    "Rider_Victor"
  ],
  "training_stages": [
    {
      "stage_id": "Training_Comp_Riviere_01",
      "stage_type": "CompetitionStage",
      "volume": "30min",
      "intensity": "Pic",
      "frequency": 1,
      "depends_on_events": [
        "Event_Dijon_Cross_2026"
      ],
      "actors": [
        {
          "actor_id": "Rider_Victor",
          "role": "Rider"
        }
      ]
    },
    {
      "stage_id": "Training_PreComp_Riviere_01",
      "stage_type": "PreCompetitionStage",
      "volume": "70min",
      "intensity": "Élevée",
      "frequency": 4,
      "depends_on_events": [
        "Event_Dijon_Cross_2026"
      ],
      "actors": [
        {
          "actor_id": "Rider_Victor",
          "role": "Rider"
        },
        {
          "actor_id": "Vet_DrMartin",
          "role": "Veterinarian"
        }
      ]
    },
    {
      "stage_id": "Training_Prepa_Riviere_01",
      "stage_type": "PreparationStage",
      "volume": "50min",
      "intensity": "Modérée",
      "frequency": 5,
      "depends_on_events": [
        "Event_Dijon_Cross_2026"
      ],
      "actors": [
        {
          "actor_id": "Rider_Victor",
          "role": "Rider"
        },
        {
          "actor_id": "Caretaker_Sophie",
          "role": "Caretaker"
        },
        {
          "actor_id": "Vet_DrMartin",
          "role": "Veterinarian"
        }
      ]
    },
    {
      "stage_id": "Training_Transition_Riviere_01",
      "stage_type": "TransitionStage",
      "volume": "25min",
      "intensity": "Faible",
      "frequency": 2,
      "depends_on_events": [
        "Event_Dijon_Cross_2026"
      ],
      "actors": [
        {
          "actor_id": "Rider_Victor",
          "role": "Rider"
        }
      ]
    }
  ],
  "sensors": [
    {
      "sensor_id": "IMU_CanonFore_Riviere_01",
      "sensor_code": "IMU-CF-017",
      "position": "CanonOfForelimb",
      "sample_rate": "250Hz",
      "offset": "0.02",
      "format": "CSV",
      "file_size": 7800,
      "objective": "FatigueDetection"
    },
    {
      "sensor_id": "IMU_Withers_Riviere_01",
      "sensor_code": "IMU-W-037",
      "position": "Withers",
      "sample_rate": "200Hz",
      "offset": "0.01",
      "format": "CSV",
      "file_size": 5500,
      "objective": "GaitClassif_01"
    }
  ],
  "event_participations": [
    {
      "event_id": "Event_Dijon_Cross_2026",
      "discipline": "Cross",
      "category": "Amateur 1",
      "event_date": "2026-06-06",
      "location": "Dijon",
      "rank": 4,
      "status": "Terminé",
      "rider_id": "Rider_Victor"
    },
    {
      "event_id": "Event_Strasbourg_Cross_2026",
      "discipline": "Cross",
      "category": "Amateur 1",
      "event_date": "2026-07-12",
      "location": "Strasbourg",
      "rank": 4,
      "status": "Terminé",
      "rider_id": "Rider_Victor"
    }
  ]
}
```

### Example 3 — event_Event_Montpellier_Dr_2026 (PASS)

#### Prose
```
Compte rendu d'épreuve — Event_Montpellier_Dr_2026
======================================================

Épreuve de dressage (Dressage), catégorie Club Elite, organisée le 2026-08-15 à Montpellier, rattachée à la saison « Saison 2026 ».

Classement et participants (6 engagement(s))
----------------------------------------
  • Rang 0 — cheval Nebule (Horse_Nebule, race « Haflinger »), cavalier Rider_Theo, statut « NonPartant ».
  • Rang 1 — cheval Crepuscule (Horse_Crepuscule, race « Selle Français »), cavalier Rider_Pauline, statut « Terminé ».
  • Rang 2 — cheval Nuage (Horse_Nuage, race « Pottok »), cavalier Rider_Elise, statut « Terminé ».
  • Rang 3 — cheval Braise (Horse_Braise, race « Appaloosa »), cavalier Rider_Alice, statut « Terminé ».
  • Rang 4 — cheval Luminos (Horse_Luminos, race « Lusitanien »), cavalier Rider_Elise, statut « Terminé ».
  • Rang 5 — cheval Horizon (Horse_Horizon, race « Oldenburg »), cavalier Rider_Ines, statut « Terminé ».

Entraînements liés (étapes DEPENDSON vers cette épreuve)
----------------------------------------
  • compétition (entraînement) (CompetitionStage) : 6 étape(s) pour 6 cheval(aux).
  • pré-compétition (PreCompetitionStage) : 6 étape(s) pour 6 cheval(aux).
  • préparation (PreparationStage) : 6 étape(s) pour 6 cheval(aux).
  • transition / récupération (TransitionStage) : 6 étape(s) pour 6 cheval(aux).

Note méthodologique : discipline, date, lieu, catégorie, rangs et cavaliers sont exclusivement issus du fact-sheet Neo4j V9.
```

#### Source fact-sheet
```json
{
  "doc_type": "event_fact_sheet",
  "source": "live_neo4j_v9",
  "extracted_at": "2026-07-27T00:24:23.545526+00:00",
  "event": {
    "id": "Event_Montpellier_Dr_2026",
    "discipline": "Dressage",
    "category": "Club Elite",
    "event_date": "2026-08-15",
    "location": "Montpellier",
    "season": "Saison 2026"
  },
  "participants": [
    {
      "participation_id": "Participation_Montpellier_Nebule_Theo",
      "horse_name": "Nebule",
      "horse_id": "Horse_Nebule",
      "horse_race": "Haflinger",
      "rider_id": "Rider_Theo",
      "rank": 0,
      "status": "NonPartant"
    },
    {
      "participation_id": "Participation_Montpellier_Crepuscule_Pauline",
      "horse_name": "Crepuscule",
      "horse_id": "Horse_Crepuscule",
      "horse_race": "Selle Français",
      "rider_id": "Rider_Pauline",
      "rank": 1,
      "status": "Terminé"
    },
    {
      "participation_id": "Participation_Montpellier_Nuage_Elise",
      "horse_name": "Nuage",
      "horse_id": "Horse_Nuage",
      "horse_race": "Pottok",
      "rider_id": "Rider_Elise",
      "rank": 2,
      "status": "Terminé"
    },
    {
      "participation_id": "Participation_Montpellier_Braise_Alice",
      "horse_name": "Braise",
      "horse_id": "Horse_Braise",
      "horse_race": "Appaloosa",
      "rider_id": "Rider_Alice",
      "rank": 3,
      "status": "Terminé"
    },
    {
      "participation_id": "Participation_Montpellier_Luminos_Elise",
      "horse_name": "Luminos",
      "horse_id": "Horse_Luminos",
      "horse_race": "Lusitanien",
      "rider_id": "Rider_Elise",
      "rank": 4,
      "status": "Terminé"
    },
    {
      "participation_id": "Participation_Montpellier_Horizon_Ines",
      "horse_name": "Horizon",
      "horse_id": "Horse_Horizon",
      "horse_race": "Oldenburg",
      "rider_id": "Rider_Ines",
      "rank": 5,
      "status": "Terminé"
    }
  ],
  "linked_training_summary": [
    {
      "stage_type": "CompetitionStage",
      "stage_count": 6,
      "horse_count": 6
    },
    {
      "stage_type": "PreCompetitionStage",
      "stage_count": 6,
      "horse_count": 6
    },
    {
      "stage_type": "PreparationStage",
      "stage_count": 6,
      "horse_count": 6
    },
    {
      "stage_type": "TransitionStage",
      "stage_count": 6,
      "horse_count": 6
    }
  ]
}
```

## Spot-check verification (10 documents)

| # | doc_id | type | subject | invented | facts missing from prose* | verdict |
|---|---|---|---|---|---|---|
| 1 | horse_Sirius | horse_report | Sirius | 0 | 0 | **PASS** |
| 2 | horse_Riviere | horse_report | Riviere | 0 | 0 | **PASS** |
| 3 | horse_Mirage | horse_report | Mirage | 0 | 0 | **PASS** |
| 4 | horse_Etoile | horse_report | Etoile | 0 | 0 | **PASS** |
| 5 | horse_Arrow | horse_report | Arrow | 0 | 0 | **PASS** |
| 6 | event_Event_Montpellier_Dr_2026 | event_report | Event_Montpellier_Dr_2026 | 0 | 0 | **PASS** |
| 7 | event_Event_Marseille_Dressage_2026 | event_report | Event_Marseille_Dressage_2026 | 0 | 0 | **PASS** |
| 8 | horse_Brume | horse_report | Brume | 0 | 0 | **PASS** |
| 9 | horse_Grondre | horse_report | Grondre | 0 | 0 | **PASS** |
| 10 | event_Event_Dressage_01 | event_report | Event_Dressage_01 | 0 | 0 | **PASS** |

\* « facts missing from prose » lists atomic fact-sheet tokens not literally present in the document text (e.g. horse URI-less internal ids already covered by hasName, or stage_type English label when the French label is used). Invented/untraced claims are the hard fail signal.

- horse_Sirius: no untraced numeric/id claims (48 atomic facts in sheet).
- horse_Riviere: no untraced numeric/id claims (48 atomic facts in sheet).
- horse_Mirage: no untraced numeric/id claims (48 atomic facts in sheet).
- horse_Etoile: no untraced numeric/id claims (49 atomic facts in sheet).
- horse_Arrow: no untraced numeric/id claims (49 atomic facts in sheet).
- event_Event_Montpellier_Dr_2026: no untraced numeric/id claims (42 atomic facts in sheet).
- event_Event_Marseille_Dressage_2026: no untraced numeric/id claims (41 atomic facts in sheet).
- horse_Brume: no untraced numeric/id claims (48 atomic facts in sheet).
- horse_Grondre: no untraced numeric/id claims (49 atomic facts in sheet).
- event_Event_Dressage_01: no untraced numeric/id claims (13 atomic facts in sheet).

**Spot-check summary:** 10/10 PASS

