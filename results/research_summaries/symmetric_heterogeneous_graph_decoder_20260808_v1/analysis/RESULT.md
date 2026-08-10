# Symmetric heterogeneous graph decoder — train-only audit

No model supplies an incumbent or KEEP state. The empty set and every
complete generated set are peer hypotheses.

| policy | pooled F1 | delta vs staged | parameters |
|---|---:|---:|---:|
| staged pipeline | 0.481128 | — | — |
| exact family vote | 0.468737 | -0.012390 | 0 |
| family-balanced medoid | 0.470260 | -0.010868 | 0 |
| symmetric `full` OOF | 0.476354 | -0.004774 | 127 |
| symmetric `aggregate` OOF | 0.470879 | -0.010249 | 64 |
| symmetric `structure` OOF | 0.316732 | -0.164396 | 22 |
| candidate-set oracle | 0.741738 | +0.260610 | 0 |

Promotion gate passed: **False**

## Primary relation deltas versus staged

- awardWonBy: -0.005281
- companyTradesAtStockExchange: +0.011333
- countryLandBordersCountry: -0.005337
- hasArea: +0.050000
- hasCapacity: -0.070000
- personHasCityOfDeath: -0.010000

## Promotion checks

- minimum_pooled_delta: **False**
- fold_floor: **False**
- beats_aggregate_ablation: **True**
- beats_structure_ablation: **True**
