# Symmetric reliability graph decoder — train-only audit

All three families have equal anchor eligibility.  Relation-level
reliability and the downstream proof arm are selected inside the
outer subject-grouped folds; development labels are not opened.

| policy | pooled train F1 | vs staged |
|---|---:|---:|
| staged end-to-end | 0.481128 | — |
| reliability anchor only OOF | 0.482455 | +0.001328 |
| reliability anchor + graph proof OOF | 0.491190 | +0.010063 |
| retained anchor + symmetric graph proof OOF | 0.489513 | +0.008386 |
| fitted diagnostic | 0.505865 | +0.024738 |

Promotion gate passed: **True**

## Frozen deployable relation policy

| relation | anchor source | family | proof arm |
|---|---|---|---|
| awardWonBy | `retained_ensemble` | `graph_ensemble` | `identity` |
| companyTradesAtStockExchange | `retained_ensemble` | `graph_ensemble` | `support_cardinality` |
| countryLandBordersCountry | `retained_ensemble` | `graph_ensemble` | `identity` |
| hasArea | `retained_ensemble` | `graph_ensemble` | `identity` |
| hasCapacity | `retained_ensemble` | `graph_ensemble` | `identity` |
| personHasCityOfDeath | `retained_ensemble` | `graph_ensemble` | `identity` |

## Promotion checks

- beats_or_matches_staged: **True**
- positive_outer_folds: **True**
- outer_fold_floor: **True**
- relation_regression_floor: **True**

## Interpretation

This experiment changes only anchor assignment and uses the existing typed graph proof decoder.  Failure retains the immutable 0.5207 pipeline; passage authorizes exactly one frozen development decode.
