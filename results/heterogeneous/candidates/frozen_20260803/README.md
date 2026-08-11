# Frozen development candidates — 2026-08-03

This directory preserves the two validation candidates that later experiments
must treat as immutable controls.

- `safe_0_518450_validation.jsonl` is the byte-identical reproduction of the
  locked development system.
- `strict_proof_0_520729_validation.jsonl` adds the conservative proof-carrying
  graph rule. It is the strongest development score, but its strict margin was
  informed by the validation failure of a looser rule.

`MANIFEST.json` pins the prediction bytes, official evaluator, validation gold,
scores, and provenance. These artifacts must never be overwritten by a new
experiment.
