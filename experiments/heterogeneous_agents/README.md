# Heterogeneous system implementation

## Public release surface

- `run_paper_system.sh`: stable user-facing launcher;
- `historical_sota_test_pipeline.py`: architecture and packaging contract for
  the official 0.4845 submission;
- `run_sota_reproduction.sh` and `sota_reproduction.py`: deterministic replay
  of the tracked development evidence;
- `end_to_end_pipeline.py`: label-free route planning;
- `run_agent.py`: checkpoint-pinned Gemma and Ministral inference;
- `assemble_and_audit.py`, `core.py`, and `frozen_model_loader.py`: shared
  contracts and runtime utilities.

`components/` is the retained dependency closure for graph construction and
decoding. Many filenames record the experiment in which a component was first
introduced; that does not mean the file is safe to delete. `analysis/` contains
optional diagnostics and visualizations rather than production stages.

## Generated and historical outputs

New run outputs belong under `runs/` and are ignored by Git. The exact official
submission is preserved under `submissions/official_test/`. Portable evidence
needed for deterministic replay is under
`results/heterogeneous/canonical_runtime/`; compact negative and ablation
results are under `results/research_summaries/` and are never runtime inputs.

`final_submission_pipeline.py` and `run_final_submission_pipeline.sh` are
retained because later audits and components import them. They describe a
post-submission research lineage and are not the public entry point for the
0.4845 paper result.

The `single_ministral_validation.py` runner and the corresponding
`run_single_ministral_validation.sh` and route-removal analyses preserve the
final one-route ablation. They are research controls, not replacements for the
reported two-route test architecture.
