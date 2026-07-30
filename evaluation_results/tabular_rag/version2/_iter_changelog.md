# Version2 iterative EX improvement changelog

**FINAL KEPT:** `tabular_eval_full100_ex_20260728_024556.json`
- **EX 51/84 (60.7%)**, **combined 0.7907** — both targets hit (stop)

Starting point this session: `015703` — EX 47/84 (56.0%), combined 0.7776

| Iteration | Change (1 line) | EX before | EX after | Combined before | Combined after | Kept or reverted |
|---|---|---|---|---|---|---|
| 1 | Narrow season-event OUTPUT SHAPE so horse participation returns event_id only | 47/84 (56.0%) | 47/84 (56.0%) | 0.778 | 0.754 | **REVERTED** |
| 2 | Treat plusieurs/une-seule existence as INVENTORY (no COUNT) | 47/84 (56.0%) | 46/84 (54.8%) | 0.778 | 0.760 | **REVERTED** |
| 3 | Resolve qui-est-[role]→person_id vs who-is→name conflict (person_id only) | 47/84 (56.0%) | 50/84 (59.5%) | 0.778 | 0.770 | **KEPT** |
| 4 | Étapes identifier lists return training_id only (no stage_type) | 50/84 (59.5%) | 50/84 (59.5%) | 0.770 | 0.781 | **KEPT** |
| 5 | Non-rider supervisor lists must return id + role (never id alone) | 50/84 (59.5%) | 51/84 (60.7%) | 0.781 | 0.791 | **KEPT** ✓ targets met |

## Real eval outputs (kept runs)

### Iter 3 (`022900`)
```
Execution Accuracy (EX): 59.5% (50/84 applicable; 16 N/A excluded)
Avg Combined Score: 0.770
```
Q20: `SELECT DISTINCT person_id FROM people WHERE role = 'Veterinarian';` → MATCH
Q21: `SELECT DISTINCT person_id FROM people WHERE role = 'Caretaker';` → MATCH

### Iter 4 (`023744`)
```
Execution Accuracy (EX): 59.5% (50/84 applicable; 16 N/A excluded)
Avg Combined Score: 0.781
```
Q5: `SELECT training_id FROM trainings WHERE horse_id = (SELECT horse_id FROM horses WHERE LOWER(name) = LOWER('Dakota'));` → MATCH

### Iter 5 (`024556`) — FINAL
```
Execution Accuracy (EX): 60.7% (51/84 applicable; 16 N/A excluded)
Avg Combined Score: 0.791
```
Q76: `SELECT DISTINCT ta.actor_id, ta.actor_role FROM training_actors ta WHERE ta.actor_role != 'Rider';` → MATCH
ROWS: `[('Vet_DrMartin', 'Veterinarian'), ('Caretaker_Sophie', 'Caretaker')]`
