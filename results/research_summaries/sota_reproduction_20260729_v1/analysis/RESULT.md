# Unified graph-native decoder — consolidation parity

One decoder, one typed evidence graph carrying all three model families as first-class routes, one pass. The deployed artifact was produced by a base decoder plus two post-processing scripts; this run must reproduce it exactly.

- Byte-identical to deployed artifact: **True**
- Divergent rows: **0/478** (0 explained by deployed answer-marker parse fragmentation, 0 unexplained)
- Unified pooled macro-F1: **0.518450**
- Deployed pooled macro-F1: **0.518450**
- Score delta: **+0.000000**

| policy | kind | evidence routes |
|---|---|---|
| `component_surface_residual_ridge` | frozen_learned_model | `gemma:independent`, `qwen:self_consistency`, `qwen:system2` |
| `capacity_qwen_precision_veto` | frozen_policy_ledger | `gemma:independent`, `qwen:self_consistency`, `qwen:system2` |
| `area_unanimous_new_component_replace` | graph_rule | `ministral:self_consistency` |
| `cot40_two_thirds_support_7` | graph_rule | `ministral:cot5_cap40_n10` |
