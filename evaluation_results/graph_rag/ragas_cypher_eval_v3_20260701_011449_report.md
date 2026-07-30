# RAGAS Cypher Evaluation V3 Report

## Summary

| Metric | Value |
| --- | ---: |
| Timestamp | 20260701_011449 |
| Total questions | 100 |
| Evaluated | 89 |
| Skipped | 11 |
| NaN values (excluded from averages) | 1 |
| avg_context_precision | 0.5055 |
| avg_context_recall | 0.6464 |

## Skipped Breakdown

| Reason | Count |
| --- | ---: |
| missing context | 5 |
| timeout | 6 |
| other | 0 |

### By reason code

| reason | count |
| --- | ---: |
| missing_or_empty_raw_context | 5 |
| timeout | 6 |

## Score Distribution — context_precision

| Bucket | Count | Percent | Bar (20 chars) |
| --- | ---: | ---: | --- |
| Perfect (1.00) | 35 | 39.3% | `########............` |
| High (0.75-0.99) | 7 | 7.9% | `##..................` |
| Mid (0.25-0.74) | 6 | 6.7% | `#...................` |
| Low (0.01-0.24) | 5 | 5.6% | `#...................` |
| Zero (0.00) | 36 | 40.4% | `########............` |

## Score Distribution — context_recall

| Bucket | Count | Percent | Bar (20 chars) |
| --- | ---: | ---: | --- |
| Perfect (1.00) | 50 | 56.2% | `###########.........` |
| High (0.75-0.99) | 1 | 1.1% | `....................` |
| Mid (0.25-0.74) | 13 | 14.6% | `###.................` |
| Low (0.01-0.24) | 0 | 0.0% | `....................` |
| Zero (0.00) | 24 | 27.0% | `#####...............` |

## Skipped Questions

| question_id | reason |
| --- | --- |
| Q32 | timeout |
| Q51 | missing_or_empty_raw_context |
| Q52 | missing_or_empty_raw_context |
| Q53 | missing_or_empty_raw_context |
| Q54 | missing_or_empty_raw_context |
| Q67 | missing_or_empty_raw_context |
| Q72 | timeout |
| Q73 | timeout |
| Q89 | timeout |
| Q94 | timeout |
| Q95 | timeout |

## Lowest 5 Recall Questions

| question_id | question | context_precision | context_recall |
| --- | --- | ---: | ---: |
| Q25 | Quels événements font partie de la saison 2026 ? | 0.000 | nan |
| Q6 | Quelle est la fréquence d'entraînement pendant la phase de préparation ? | 0.955 | 0.000 |
| Q20 | Qui est le vétérinaire impliqué dans les soins des chevaux ? | 1.000 | 0.000 |
| Q22 | Quels acteurs humains sont impliqués dans la phase de préparation de l'entraînement ? | 0.000 | 0.000 |
| Q23 | Qui participe à la phase pré-compétition de l'entraînement ? | 0.000 | 0.000 |

## Lowest 5 Precision Questions

| question_id | question | context_precision | context_recall |
| --- | --- | ---: | ---: |
| Q4 | Dans quels événements sportifs Dakota participe-t-il ? | 0.000 | 1.000 |
| Q5 | Quelles étapes d'entraînement Dakota suit-il ? | 0.000 | 1.000 |
| Q12 | Quels sont les identifiants des capteurs IMU ? | 0.000 | 0.500 |
| Q13 | À quelles positions anatomiques les capteurs IMU sont-ils placés sur Dakota ? | 0.000 | 1.000 |
| Q16 | Quel capteur a la fréquence d'échantillonnage la plus élevée ? | 0.000 | 1.000 |

## Files

- Script: `C:\Users\iTECH\Desktop\Ibrahim\Internships & Courses\9- RAG\KnowledgeBasedRAG\scripts\ragas_cypher_eval_v3.py`
- JSON: `C:\Users\iTECH\Desktop\Ibrahim\Internships & Courses\9- RAG\KnowledgeBasedRAG\evaluation_results\ragas_cypher_eval_v3_20260701_011449.json`
- Partial: `C:\Users\iTECH\Desktop\Ibrahim\Internships & Courses\9- RAG\KnowledgeBasedRAG\evaluation_results\ragas_cypher_eval_v3_20260701_011449_partial.json`
- Report: `C:\Users\iTECH\Desktop\Ibrahim\Internships & Courses\9- RAG\KnowledgeBasedRAG\evaluation_results\ragas_cypher_eval_v3_20260701_011449_report.md`
