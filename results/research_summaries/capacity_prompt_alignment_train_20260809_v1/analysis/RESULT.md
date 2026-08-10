# End-to-end heterogeneous system — train

Generated from `/home/hongjing/hongjing_project/dataset2026/lm-kbc-2026/dataset2026/experiments/heterogeneous_agents/runs/capacity_prompt_alignment_train_20260809_v1/input/TRAIN_CAPACITY.jsonl` by this pipeline: four evidence routes produced fresh, one typed graph, one decode pass. No historical incumbent and no inherited prediction artifact.

- Pooled macro-F1: **0.050000**
- Rows: **100**
- Verified parameters: **30,515,165,024** of 32,000,000,000

| relation | macro-P | macro-R | macro-F1 |
|---|---:|---:|---:|
| hasCapacity | 0.0500 | 0.0500 | 0.0500 |
| **pooled** | 0.0500 | 0.0500 | **0.050000** |

## Layers not carried by this run

- `route_residual_company_arm` (1 deployed rows): its train gate selects a pass-through arm under current gold
- `capacity_qwen_precision_veto` (11 deployed rows): requires a review-generation pass with no end-to-end runner
