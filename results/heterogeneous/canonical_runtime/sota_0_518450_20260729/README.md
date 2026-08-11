# Frozen inputs for the 0.518450 development SOTA

This directory is the portable, minimal evidence snapshot consumed by
`experiments/heterogeneous_agents/sota_reproduction.py`.  It replaces
machine-local references into ignored `runs/` directories.

The snapshot is sufficient to reconstruct the historical L1--L3 incumbent
from its L0 graphs and fitted models, rebuild the unified typed graph, execute
L4--L7, reproduce every retained layer target and the final prediction bytes
under `legacy-20260729` parser compatibility, and independently score all 478
development rows with the tracked official evaluator.

It is not a checkpoint bundle and does not claim bitwise regeneration of the
model responses.  The response and graph artifacts are retained because exact
text generation can vary with CUDA, PyTorch, Transformers, and quantization
versions even when checkpoint revisions and random seeds are fixed.  Model
regeneration is therefore a separate reproducibility tier from deterministic
pipeline replay.

`ARTIFACTS.json` is authoritative.  Every consumer must verify all listed
hashes and row counts before decoding.  The development lineage was selected
after validation had been opened and is not a blind-test estimate.

The exact 478-row validation file used by this historical lineage is retained
at `data/archive/validation_478_20260729.jsonl`.  It is deliberately separate
from the later 475-row task release so that replay cannot silently score
against a different split revision.
