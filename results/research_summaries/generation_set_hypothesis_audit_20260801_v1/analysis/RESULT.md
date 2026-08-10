# Complete-generation answer-set hypothesis audit

Train only; validation was not opened. One exact generation is one
coherent answer-set hypothesis. The candidate-union oracle remains
gold-aware and nondeployable.

| policy | pooled F1 | delta vs incumbent | changed | help / harm |
|---|---:|---:|---:|---:|
| CoT40 incumbent | 0.498690 | -- | -- | -- |
| exact_family_consensus | 0.490811 | -0.007879 | 157 | 31 / 36 |
| family_mean_medoid | 0.491231 | -0.007460 | 156 | 31 / 35 |
| family_minimum_medoid | 0.492452 | -0.006238 | 155 | 33 / 33 |
| independent_family_medoid | 0.474877 | -0.023813 | 173 | 31 / 44 |
| family_mean_medoid_shuffled | 0.267172 | -0.231518 | 376 | 23 / 179 |
| nested_keep_gate | 0.496848 | -0.001842 | 39 | 12 / 7 |
| KEEP-vs-medoid oracle | 0.584383 | +0.085692 | -- | -- |
| whole-generation oracle | 0.742151 | +0.243460 | -- | -- |
| arbitrary candidate-union oracle | 0.754261 | +0.255571 | -- | -- |

- Coherent fraction of candidate headroom: **95.262%**
- Nested fold wins: **3/5**
- Medoid opportunities (better / worse / tie-different): **54 / 197 / 191**
- Coherent-headroom gate: **True**
- Selector gate: **False**
- Next stage: `retain_coherent_set_target_but_reject_current_medoid_selector`
