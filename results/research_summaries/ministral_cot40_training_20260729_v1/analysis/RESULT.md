# Ministral CoT × requested-reasoning training ablation

Training-only audit; validation was structurally absent. CoT-20 and CoT-40 use the same checkpoint, subjects, SyntheticCoT examples, seed, temperature, and ordered N=10 sampling stream. The word budget is requested in the prompt and measured, not hard-truncated.

- Competition-pipeline OOF anchor: **0.482454**
- Frozen source-graph oracle: **0.669384**

## Matched SyntheticCoT-5: requested 20 vs 40 words

| N | CoT-20 standalone F1 | CoT-40 standalone F1 | delta | CoT-20 residual F1 | CoT-40 residual F1 | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3511 | 0.3544 | +0.0033 | 0.3942 | 0.3900 | -0.0041 |
| 3 | 0.3714 | 0.3641 | -0.0073 | 0.4632 | 0.4516 | -0.0117 |
| 5 | 0.3536 | 0.3596 | +0.0060 | 0.4835 | 0.4844 | +0.0009 |
| 10 | 0.3508 | 0.3736 | +0.0227 | 0.4769 | 0.4917 | +0.0148 |

## Matched requested 40 words: zero-shot vs SyntheticCoT-5

| N | zero-shot standalone F1 | CoT-5 standalone F1 | delta | zero-shot residual F1 | CoT-5 residual F1 | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2156 | 0.3544 | +0.1388 | 0.2938 | 0.3900 | +0.0963 |
| 3 | 0.2329 | 0.3641 | +0.1311 | 0.3808 | 0.4516 | +0.0708 |

## Decision rule

The train-only screen exposes every support rule and nested prefix. No validation setting is chosen here. A final N/support policy must be frozen from these training diagnostics before one validation confirmation.
