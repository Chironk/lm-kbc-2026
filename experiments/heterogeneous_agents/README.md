# Heterogeneous pipeline code

The package root contains only the supported execution surface:

- `final_submission_pipeline.py`: graph construction, decoding, scoring, and packaging;
- `end_to_end_pipeline.py`: label-free task planning and evidence assembly;
- `run_agent.py`: checkpoint-pinned model inference;
- `assemble_and_audit.py`, `core.py`, and `frozen_model_loader.py`: shared runtime contracts;
- `preflight.py`: environment and GPU checks;
- `run_final_submission_pipeline.sh`: supported launcher.

`components/` contains the graph builders, decoder stages, and retained research
implementations on which the final pipeline depends. `analysis/` contains optional
post-hoc visualization utilities. Generated outputs belong in `runs/`, which is
ignored except for a local resumable run that may exist on a working machine.

Historical compact experiment outcomes are kept separately in
`results/research_summaries/`; raw responses and logs are not release artifacts.
