# Reproducibility

## 1. Environment and integrity check

The recorded environment uses Python 3.11. Exact package versions are in
`requirements-lock.txt`; immutable model revisions are in `configs/final/`.

```bash
git clone https://github.com/Chironk/lm-kbc-2026.git
cd lm-kbc-2026
conda create -n lm-kbc-2026 python=3.11 -y
conda activate lm-kbc-2026
pip install -r requirements-lock.txt
python scripts/verify_release.py
pytest -q
```

`verify_release.py` is model-free. It verifies the official splits, parameter
budget, frozen decoder artifacts, development replay snapshot, and exact
official-test archive. It works both in a Git clone and in an unpacked source
archive.

## 2. Exact result artifacts

### Official test

The exact submitted archive is:

```text
submissions/official_test/heterogeneous_final_strict_proof_20260803_v1_test.zip
```

It contains only `predictions.jsonl`. Its archive SHA-256 is
`3f73d01fe5d4b3c9b9cc7e2f5dba8348d0e1fec19fc0ddb797ff2e0f460b11e4`;
the member SHA-256 is
`73621130839b572a7fdfdc2f8a58c4bf3f00beece4be86ff4a7874c96b63bb53`.
Codabench returned 0.4845 macro-F1. Because test labels are not distributed,
that score is recorded from the official result and is not recomputed locally.

### Development evidence replay

```bash
OUT=experiments/heterogeneous_agents/runs/development_replay \
bash experiments/heterogeneous_agents/run_sota_reproduction.sh all
```

This validates every tracked input hash, reconstructs the staged decoder, and
reproduces the 478-row prediction bytes and macro-F1 0.5184496147269507. The
snapshot is under `results/heterogeneous/canonical_runtime/` and is explicitly
a development-selected lineage. The final graph-corrected development
prediction is also retained and hash-pinned at
`results/heterogeneous/candidates/frozen_20260803/strict_proof_0_520729_validation.jsonl`.
It scores 0.5207285306041929 with the tracked evaluator and is marked in its
manifest as a development-informed graph refinement.

## 3. Fresh paper-system inference

The public launcher delegates to the hash-pinned historical implementation:

```bash
RUNNER=experiments/heterogeneous_agents/run_paper_system.sh
OUT=experiments/heterogeneous_agents/runs/paper_test

OUT="$OUT" bash "$RUNNER" plan
OUT="$OUT" bash "$RUNNER" preflight
OUT="$OUT" bash "$RUNNER" generate-primary
OUT="$OUT" bash "$RUNNER" generate-gemma
OUT="$OUT" bash "$RUNNER" generate-ministral-n3
OUT="$OUT" bash "$RUNNER" generate-ministral-n10
OUT="$OUT" bash "$RUNNER" build
OUT="$OUT" bash "$RUNNER" package
```

Or execute all stages:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUT=experiments/heterogeneous_agents/runs/paper_test \
bash experiments/heterogeneous_agents/run_paper_system.sh all
```

Completed response files are checked against manifests. An interrupted file
resumes from missing task IDs; a conflicting completed artifact is rejected.
The historical-control package is written to:

```text
OUT/submission/historical_sota_replication_control_test.zip
```

The underlying audit runner also emits a paired later area-route experiment.
That second archive is not the submission whose 0.4845 score is reported.

## 4. GPU selection

If `CUDA_VISIBLE_DEVICES` is unset, the launcher enumerates visible GPUs with
`nvidia-smi`. Qwen runs as one process sharded across those devices. Quantized
Gemma and Ministral use one independent worker per visible device. Override
the worker count only when memory pressure requires it:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_WORKERS=2 \
OUT=experiments/heterogeneous_agents/runs/paper_test \
bash experiments/heterogeneous_agents/run_paper_system.sh generate
```

On 11-GiB GPUs, keep generation batch size at one as pinned by the launcher.

## 5. What fresh inference can and cannot reproduce

The fresh runner pins model revisions, prompts, demonstrations, sampling
counts, decoder artifacts, and the historical seed scheme. The sampled model
responses that produced the original official archive were not retained.
Consequently, the repository makes two separate claims:

- the submitted `predictions.jsonl` and its Codabench provenance are exactly
  preserved and hash-verifiable;
- the full inference and decoding procedure can be rerun from the official
  query split.

It does not claim that fresh stochastic generation will recreate the submitted
prediction bytes on every CUDA/software stack.

## 6. Moving or resuming a run

Generated runs are deliberately outside Git. On another host, clone the same
commit, create the environment, and run `plan` in a new `OUT` directory. If a
partial run must be transferred, copy the whole run directory without changing
its internal layout, then invoke the same generation stage; completed task IDs
will be reused after manifest validation.

## 7. Split discipline

- Training and validation may be scored with the official evaluator.
- The blind test path checks the official 475-row SHA-256
  `67c31c8388c585634df55500612f522ad42da6735d4c89eb59a9ef5a39f043f1`.
- The test file contains no nonempty `ObjectEntities` values.
- Frozen decoder artifacts contain no test labels or test-specific decisions.
