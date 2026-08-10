# Minimal graph and structural commitments — train-only audit

Minimal strict-proof parity: **passed**

| arm | train F1 | delta | shifted delta | folds | helped / harmed | gate |
|---|---:|---:|---:|---:|---:|---:|
| strict_support | 0.481128 | +0.000000 | +0.000000 | 0/5 | 0 / 0 | True |
| strict_plus_existence | 0.483224 | +0.002096 | +0.002096 | 2/5 | 2 / 0 | False |
| strict_plus_cardinality | 0.482525 | +0.001398 | -0.000699 | 2/5 | 2 / 1 | False |
| strict_plus_both | 0.482525 | +0.001398 | -0.000699 | 2/5 | 2 / 1 | False |
| loose_plus_existence | 0.485786 | +0.004658 | -0.003728 | 3/5 | 5 / 9 | False |
| loose_plus_cardinality | 0.475304 | -0.005824 | -0.005824 | 1/5 | 2 / 11 | False |
| loose_plus_both | 0.485786 | +0.004658 | -0.005824 | 3/5 | 5 / 9 | False |

Selected arm: **strict_support**
Promotion gate passed: **False**

Validation labels have not been opened by this stage.
