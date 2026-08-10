# Reproducibility

## Environment

The recorded environment uses Python 3.11. `requirements-lock.txt` contains
the exact package versions used for the final implementation;
`requirements.txt` is the portable specification. Model weights are resolved
from immutable Hugging Face revisions recorded in `configs/final/`.

```bash
conda create -n lm-kbc-2026 python=3.11 -y
conda activate lm-kbc-2026
pip install -r requirements-lock.txt
python scripts/verify_release.py
pytest -q
```

## GPU selection

With no override, the launcher detects and uses every visible CUDA GPU up to
each route's verified replica limit. To constrain a run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev \
bash experiments/heterogeneous_agents/run_final_submission_pipeline.sh all
```

The Qwen fp16 route is one sharded process. Four-bit Gemma and Ministral use
one independent worker per visible GPU. On 11-GiB cards, keep generation batch
size at one; the checked-in launcher already does this.

## Staged and resumable execution

Every command below can be rerun safely:

```bash
RUNNER=experiments/heterogeneous_agents/run_final_submission_pipeline.sh
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" plan
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" preflight
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" smoke
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" generate-primary
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" generate-gemma
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" generate-ministral-n3
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" generate-ministral-cot40
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" decode
SPLIT=validation INPUT=data/val.jsonl OUT=experiments/heterogeneous_agents/runs/dev bash "$RUNNER" score
```

Completed response files are checked against their manifests. An incomplete
file resumes from pending task IDs. A conflicting complete artifact is rejected
rather than silently reused.

## Moving to another machine

Commit and push source code only. On the second machine:

```bash
git clone https://github.com/j31040116-boop/lm-kbc-2026.git
cd lm-kbc-2026
git checkout main
```

Create the environment and plan a new `OUT` directory there. Do not copy a
machine-local `PLAN.json` between differently located clones; planning is
cheap and binds paths and hashes for the current checkout. If transferring a
partially completed run, preserve its relative directory layout and place it
under the same `OUT` path before resuming.

## Split discipline

- Train/development may be scored with the official evaluator.
- `score-validation` refuses a test plan.
- The final test launcher checks the official test SHA-256:
  `67c31c8388c585634df55500612f522ad42da6735d4c89eb59a9ef5a39f043f1`.
- Frozen decoder artifacts contain no prediction rows or test answers.

## Expected outputs

After a complete run:

```text
OUT/plan/PLAN.json
OUT/plan/FINAL_POLICY.json
OUT/graph/FINAL_EXACT_EVIDENCE_GRAPH.jsonl
OUT/FINAL_PREDICTIONS.jsonl
OUT/FINAL_MANIFEST.json
OUT/submission/heterogeneous_final_relation_typed_proof_20260809_v3_test.zip
```

All of `OUT` is generated and ignored by Git.
