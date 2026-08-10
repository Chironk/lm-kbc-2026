# All-three-first graph decoder — train-only audit

Every exact Qwen, Gemma, and Ministral generation enters one graph
before a relation-conditioned symbolic proof arm is selected.

| policy | pooled F1 | vs Qwen | vs staged |
|---|---:|---:|---:|
| Qwen incumbent | 0.472678 | — | -0.008450 |
| staged end-to-end | 0.481128 | +0.008450 | — |
| all-three-first outer OOF | 0.480694 | +0.008016 | -0.000433 |
| all-three-first fitted diagnostic | 0.481413 | +0.008735 | +0.000286 |

Promotion gate passed: **False**

## Frozen full-train relation policy

- awardWonBy: `identity`
- companyTradesAtStockExchange: `support_cardinality`
- countryLandBordersCountry: `identity`
- hasArea: `identity`
- hasCapacity: `identity`
- personHasCityOfDeath: `identity`

## Promotion checks

- beats_qwen_incumbent: **True**
- beats_staged_end_to_end_by_margin: **False**
- no_relation_regression_vs_staged: **False**

## Interpretation

The clean architecture improves on the Qwen incumbent only if
the OOF delta above is positive, but it does not replace the staged
pipeline unless it also clears the stricter end-to-end gate.  This
distinguishes a real graph-selection gain from a full-system gain.
