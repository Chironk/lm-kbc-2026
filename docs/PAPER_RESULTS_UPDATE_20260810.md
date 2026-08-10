# Paper results update — 2026-08-10

The final paper reports two result sets separately:

- **Development:** macro-F1 0.5207 on the 478-query validation-phase split
  used while refining the symbolic conditions. These values are preserved in
  `results/research_summaries/proof_carrying_graph_decoder_20260801_v1/validation/RESULT.json`.
- **Official test:** macro-F1 0.4845 on the frozen 475-row Codabench
  submission. The exact archive, its hashes, and the owner-supplied detailed
  metrics are preserved under `submissions/official_test/`.

The paper intentionally uses separate development and test tables and does not
include a difference column. The abstract, contributions, experimental setup,
conclusion, and limitations use the official test result as the headline
end-to-end score. Development-only stage and candidate-oracle analyses are
labeled as such.

The method description also removes the later capacity-only numeric proof
exception. That exception belongs to a post-submission experiment and did not
produce the archived 0.4845 official-test artifact.

## Reproducibility scope

The exact submitted predictions are immutable and hash-pinned. Their official
score can be reproduced by resubmitting the preserved archive. Because the
original sampled intermediate generations were not retained in full, a fresh
GPU inference run is not claimed to reproduce those prediction bytes exactly.
