# Heterogeneous Parametric Memory for LM-KBC 2026

This is the cleaned paper-release repository for our AKBC/LM-KBC 2026
submission. It contains one supported end-to-end path from an official split
to a validated Codabench archive:

```text
official rows -> Qwen/Gemma/Ministral generations -> typed evidence graph
              -> frozen staged decoder -> symbolic graph correction
              -> predictions.jsonl -> submission zip
```

The three pinned checkpoints contain **30,515,165,024 parameters** in total,
within the shared task's 32B limit. No external retrieval is used at inference
time. The official task data and evaluator are preserved from the upstream
repository; see [the official task documentation](docs/OFFICIAL_TASK.md).

## Repository map

| Path | Purpose |
|---|---|
| `experiments/heterogeneous_agents/final_submission_pipeline.py` | frozen graph construction, decoder, scoring, and packaging |
| `experiments/heterogeneous_agents/run_final_submission_pipeline.sh` | supported end-to-end launcher |
| `configs/final/` | pinned three-model portfolio configurations |
| `artifacts/frozen/` | compact trained decoder artifacts and integrity manifest |
| `data/` | official splits and reviewed SyntheticCoT pools |
| `tests/` | unit and contract tests for the retained runtime closure |
| `docs/ARCHITECTURE.md` | readable architecture and code map |
| `docs/REPRODUCIBILITY.md` | environment, validation, test, and resume instructions |

Generated model responses, logs, smoke outputs, and submission archives are
intentionally ignored. They can be regenerated and should not be committed.

## Setup

Python 3.11 and CUDA-capable PyTorch are required for inference.

```bash
conda create -n lm-kbc-2026 python=3.11 -y
conda activate lm-kbc-2026
pip install -r requirements-lock.txt
```

For tests only:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Verify the release without loading a model

```bash
python scripts/verify_release.py
bash experiments/heterogeneous_agents/run_final_submission_pipeline.sh test
```

The verifier checks the official split sizes and test hash, every frozen
artifact hash, the parameter total, configuration paths, and importability of
the final pipeline.

## Run validation

```bash
SPLIT=validation \
INPUT=data/val.jsonl \
OUT=experiments/heterogeneous_agents/runs/final_validation \
bash experiments/heterogeneous_agents/run_final_submission_pipeline.sh all
```

The launcher detects all visible GPUs unless `CUDA_VISIBLE_DEVICES` is set.
Every generation stage is resumable: rerunning the same command validates and
reuses complete artifacts rather than loading the model again.

## Run the blind test and package a submission

```bash
SPLIT=test \
INPUT=data/test.jsonl \
OUT=experiments/heterogeneous_agents/runs/final_test \
bash experiments/heterogeneous_agents/run_final_submission_pipeline.sh all
```

The final archive is written below `OUT/submission/`. It contains exactly one
root-level file named `predictions.jsonl`. The test path refuses any input
whose SHA-256 does not match the official August 2026 release.

See [reproducibility instructions](docs/REPRODUCIBILITY.md) for staged runs,
two-GPU hosts, artifact transfer, and failure recovery.

## Development policy

- Tune and ablate on train/development data only.
- Never score the blind test locally.
- Do not commit generated `runs/` directories or credentials.
- Add a unit or contract test for any decoder or graph-schema change.
- Update `artifacts/frozen/MANIFEST.json` only when deliberately replacing a
  trained artifact, and record its new hash.

## Upstream and license

The shared-task data, evaluator, and baseline originated in
[`lm-kbc/dataset2026`](https://github.com/lm-kbc/dataset2026). Code and data in
this repository retain the upstream Apache-2.0 license.
