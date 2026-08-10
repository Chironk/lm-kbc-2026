# Proof-carrying graph decoder — train-only audit

The decoder traverses exact support edges into complete answer-set
hypotheses and applies one relation-agnostic proof obligation.

Promotion gate passed: **True**

| arm | pooled F1 | delta | changed | helped / harmed |
|---|---:|---:|---:|---:|
| support_consensus | 0.504442 | +0.005752 | 51 | 22 / 13 |
| support_cardinality | 0.504233 | +0.005542 | 23 | 13 / 5 |
| support_nonexpanding | 0.513597 | +0.014906 | 35 | 18 / 6 |
| loose_proof_graph | 0.509893 | +0.011202 | 13 | 11 / 1 |
| strict_proof_graph | 0.504811 | +0.006120 | 9 | 7 / 1 |
| strict_proof_graph_cardinality_shifted | 0.500438 | +0.001747 | 3 | 3 / 0 |

## Primary relation deltas

- awardWonBy: +0.000000
- companyTradesAtStockExchange: +0.026000
- countryLandBordersCountry: +0.004766
- hasArea: +0.000000
- hasCapacity: +0.000000
- personHasCityOfDeath: +0.000000

## Primary fold deltas

- fold 0: +0.005263
- fold 1: +0.012835
- fold 2: +0.012500
- fold 3: +0.000000
- fold 4: +0.000000

## Gate checks

- minimum_pooled_delta: **True**
- positive_folds: **True**
- fold_floor: **True**
- relation_floor: **True**
- help_harm_ratio: **True**
- aligned_over_shifted: **True**
