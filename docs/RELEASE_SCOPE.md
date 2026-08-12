# Release scope

This file records the organization policy used for the paper-code release.
The policy is conservative: no experiment or result is deleted merely because
it is not part of the public entry point.

## Supported paper path

The supported fresh-inference command is
`experiments/heterogeneous_agents/run_paper_system.sh`. It delegates to the
hash-pinned historical implementation associated with the official 0.4845
submission. The exact submitted archive and its manifest are under
`submissions/official_test/`.

## Retained dependencies

`experiments/heterogeneous_agents/components/` is kept intact. A static import
audit found that nearly every component module is reachable from the retained
paper, reproduction, or test runners. Experiment-oriented filenames therefore
must not be interpreted as evidence that a module is obsolete.

The root baseline and inference modules are also retained because they contain
the official baseline, primary Qwen execution, prompt contracts, and evaluator
integration used by the heterogeneous system.

## Retained research records

- `experiments/heterogeneous_agents/analysis/` contains optional diagnostics.
- Later validated runners remain in the heterogeneous package and are clearly
  labeled as research controls.
- `results/research_summaries/` contains compact reviewed outcomes, including
  negative results. These are provenance, not runtime inputs.
- `results/heterogeneous/canonical_runtime/` and
  `results/heterogeneous/candidates/` contain the small frozen inputs and
  predictions required for exact development replay and result verification.

## Generated local state

The following stay on the local machine and are excluded from Git:

- `.env` and credentials;
- `experiments/heterogeneous_agents/runs/**` except its README;
- Python, pytest, editor, and notebook caches;
- model weights and Hugging Face caches;
- local compressed archives and transfer staging.

Generated state is not being deleted by the release cleanup. It is simply not
part of the source repository.
