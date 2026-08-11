# Frozen predictions on the revised 475-query development set

These files apply the organizer's published 478-to-475 subject-key migration
to the immutable development predictions in `../frozen_20260803/` and then
score them with the official evaluator.

| artifact | old 478-query F1 | revised 475-query F1 | status |
|---|---:|---:|---|
| safe locked system | 0.518450 | 0.525935 | frozen-artifact compatibility replay |
| strict graph candidate | 0.520729 | 0.528228 | development-informed compatibility replay |

The migration renames 20 subject keys and removes the three venue-capacity
queries withdrawn by the organizer. It does not inspect gold answers and does
not change any remaining `ObjectEntities` value. Labels are used only after
the predictions are written, for official scoring.

These are reproducible **development evaluation results**, but they are not a
fresh inference run on the clarified subject strings. In particular, the
strict graph candidate remains development-informed, as documented in the
source manifest.

Reproduce from the repository root with:

```bash
python -m experiments.heterogeneous_agents.rekey_frozen_validation
python -m pytest -q tests/test_rekey_frozen_validation.py
```

`RESULT.json` pins the old and current development files, evaluator, source
predictions, migrated predictions, organizer commit, and per-relation scores.
