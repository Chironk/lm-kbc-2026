# Standalone CoT route ablation — current validation release

All four generation routes completed on the current 475-row validation split.
Every response artifact has a complete manifest binding it to the pinned task,
checkpoint revision, seed, configuration, and output hash. The blind test was
not read or scored.

| route | prompt | N | standalone policy or exact-surface diagnostic | macro-F1 |
|---|---|---:|---:|---:|
| Qwen | CoT-5 | 10 | repository-native standalone aggregator | **0.436200** |
| Qwen | CoT-5 | 10 | any exact-surface support (1/10) | 0.334347 |
| Qwen | CoT-5 | 10 | majority (6/10) diagnostic | 0.430257 |
| Qwen | CoT-5 | 10 | two-thirds exact-surface support (7/10) | 0.409862 |
| Qwen | CoT-5 | 10 | unanimous exact-surface support (10/10) | 0.283891 |
| Gemma | CoT-5 | 1 | repository-native standalone aggregator | **0.440345** |
| Gemma | CoT-5 | 1 | raw single-answer diagnostic | 0.425257 |
| Ministral | CoT-5, 40-word reasoning cap | 10 | any exact-surface support (1/10) | 0.297845 |
| Ministral | CoT-5, 40-word reasoning cap | 10 | majority exact-surface support (6/10) | 0.382180 |
| Ministral | CoT-5, 40-word reasoning cap | 10 | deployed residual threshold diagnostic (7/10) | **0.366425** |
| Ministral | CoT-5, 40-word reasoning cap | 10 | unanimous exact-surface support (10/10) | 0.332355 |
| Ministral area only | zero-shot, 20-word reasoning cap | 3 | any (1/3) | 0.200000 |
| Ministral area only | zero-shot, 20-word reasoning cap | 3 | majority/two-thirds exact-surface support (2/3) | 0.200000 |
| Ministral area only | zero-shot, 20-word reasoning cap | 3 | deployed area-admission threshold diagnostic (3/3) | **0.100000** |

## Interpretation boundary

These are **single-route standalone prediction** results. They answer how each
fresh route performs when its supported candidates are used as the entire
answer. They are not leave-one-model-out full-system ablations and do not
measure a route's marginal contribution to the frozen graph decoder.

For Qwen and Gemma, the primary standalone endpoint uses the repository's
existing `prediction_for_agent` policy, including existence commitments,
relation typing, numeric medians, singleton selection, and list support. The
remaining thresholds are explicitly retained as diagnostics, not selected
policies.

The support diagnostics count canonical exact surfaces across distinct
generations. They do not reproduce graph-level equivalence construction or
complete-link numeric components and are labelled accordingly.

In particular, the deployed Ministral CoT rule adds candidates with 7/10
support to an incumbent; its standalone 7/10 score starts from an empty answer.
The deployed zero-shot area rule replaces an incumbent only when there is one
unique, unanimous, typed-new component; its standalone 3/3 score likewise does
not measure that correction's system-level value. Qwen's frozen final route
also includes production aggregation and System-2 behavior that this requested
CoT-only route intentionally excludes.

The historical 0.518450 system artifact contains 478 rows, while this run uses
the current 475-row validation release. Its score is therefore useful context,
not a paired delta. A current-split full-system run plus route-removal decodes
is required for exact component-contribution claims.

## Reproducibility

- Validation SHA-256: `ba86b53ac38eb4b23b80391b291e5987ff4bbfe79827596fc09751b1bb0ce2be`
- Plan SHA-256: `6c94c32eaa7090b8248f79eac4094b3e6b3834a9f8852768b28f98a03a81220f`
- Full local result SHA-256: `b4f995dba65c9ad97fb094b06eb050bab16f74dc4fa8399db3f774447ab06eb4`
- Seed: `20260730`
- Question contract: `official-v1`
- SyntheticCoT pool: `capacity_aligned_v2`
