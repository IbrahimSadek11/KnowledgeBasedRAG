# RAGAS Cypher Evaluation Report

## Files
Created:
- `C:\Users\iTECH\Desktop\Ibrahim\Internships & Courses\9- RAG\KnowledgeBasedRAG\scripts\ragas_cypher_eval.py`
- `C:\Users\iTECH\Desktop\Ibrahim\Internships & Courses\9- RAG\KnowledgeBasedRAG\evaluation_results\ragas_cypher_eval_20260630_171945.json`
- `C:\Users\iTECH\Desktop\Ibrahim\Internships & Courses\9- RAG\KnowledgeBasedRAG\evaluation_results\ragas_cypher_eval_20260630_171945_report.md`
Modified: None

## Run Status
- Command: `python scripts/ragas_cypher_eval.py`
- Status: completed without errors

## Totals
- Total questions evaluated: 96
- Total skipped: 4
- Skipped question_ids: Q21, Q34, Q67, Q80

## Scores
- context_precision: 0.6354
- context_recall: 0.2278

## Lowest 5 Questions
| question_id | question | context_precision | context_recall |
| --- | --- | ---: | ---: |
| Q2 | Quelle est la race de Dakota ? | 0.0000 | 0.0000 |
| Q3 | Quelle est la race de Naya ? | 0.0000 | 0.0000 |
| Q6 | Quelle est la fréquence d'entraînement pendant la phase de préparation ? | 0.0000 | 0.0000 |
| Q7 | Quelle est l'intensité d'entraînement durant la phase pré-compétition ? | 0.0000 | 0.0000 |
| Q8 | Quelle est la durée des séances pendant la phase de préparation ? | 0.0000 | 0.0000 |

## Warnings
```text
Evaluating:   0%|          | 0/192 [00:00<?, ?it/s]Evaluating:   1%|          | 1/192 [09:57<31:41:30, 597.33s/it]Evaluating: 100%|██████████| 192/192 [09:57<00:00,  3.11s/it]  

ResourceWarning: unclosed file <_io.TextIOWrapper name='C:\\Users\\iTECH\\AppData\\Local\\ragas\\ragas\\uuid.json' mode='r' encoding='cp1252'>
DeprecationWarning: evaluate() is deprecated and will be removed in a future version. Use the @experiment decorator instead. See https://docs.ragas.io/en/latest/concepts/experiment/ for more information.
DeprecationWarning: aevaluate() is deprecated and will be removed in a future version. Use the @experiment decorator instead. See https://docs.ragas.io/en/latest/concepts/experiment/ for more information.
```
