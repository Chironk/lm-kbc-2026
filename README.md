# Heterogeneous Parametric Memory for LM-KBC 2026

This is the paper-release repository for our AKBC/LM-KBC 2026 system. The
supported architecture independently samples Qwen, Gemma, and Ministral,
retains complete answer sets and their model provenance in an evidence graph,
builds a provisional answer with frozen relation-aware stages, and permits a
final rule-based graph correction only when a competing complete set has
stronger cross-model evidence.

```text
official query rows
    -> pinned Qwen, Gemma, and Ministral sampling routes
    -> evidence graph of complete sampled sets and normalized answers
    -> frozen staged set decoder
    -> conservative rule-based graph correction
    -> predictions.jsonl -> Codabench zip
```

The three pinned checkpoints contain **30,515,165,024 parameters** in total,
within the shared task's 32B limit. Inference is closed-book: no external
retrieval is used. The official task data and evaluator are retained from the
upstream repository; see [the task documentation](docs/OFFICIAL_TASK.md).

## What is authoritative

- `experiments/heterogeneous_agents/run_paper_system.sh` is the public launcher
  for fresh inference with the architecture associated with the reported test
  submission.
- `submissions/official_test/` contains the exact 475-row archive submitted to
  Codabench, its SHA-256 manifest, and the owner-recorded official score of
  **0.4845 macro-F1**.
- `experiments/heterogeneous_agents/run_sota_reproduction.sh all`
  deterministically replays the frozen 478-row development evidence and
  reproduces the staged prediction artifact at **0.5184496 macro-F1**. The
  tracked final graph-corrected development prediction is separately pinned at
  **0.5207285 macro-F1** and labeled as a development-informed refinement.
  Applying the organizer's published, label-free 478-to-475 query-key migration
  gives the paper's revised-validation score of **0.5282279 macro-F1**; both the
  migrated predictions and their official-evaluator score are hash-verified.
- `artifacts/frozen/MANIFEST.json` binds the small trained decoder artifacts and
  the 30.515B parameter contract.
- `configs/final/paper_system_contract.json` is the machine-checked source of
  truth for proposal counts, temperatures, auxiliary Qwen inference, model
  revisions, decoder order, and the exact-result versus fresh-rerun boundary.

The exact official archive is intentionally tracked. Newly generated model
responses, logs, smoke outputs, and run-specific submission archives are
ignored and belong under `experiments/heterogeneous_agents/runs/`.

## Repository map

| Path | Purpose |
|---|---|
| `experiments/heterogeneous_agents/run_paper_system.sh` | stable release launcher |
| `experiments/heterogeneous_agents/historical_sota_test_pipeline.py` | hash-pinned paper-system planner, graph builder, decoder, and packager |
| `experiments/heterogeneous_agents/run_sota_reproduction.sh` | deterministic development-evidence replay |
| `experiments/heterogeneous_agents/components/` | retained graph and decoder dependency closure |
| `experiments/heterogeneous_agents/analysis/` | optional post-hoc analyses; not runtime stages |
| `configs/final/` | pinned model portfolio configurations |
| `artifacts/frozen/` | compact decoder artifacts and integrity manifest |
| `submissions/official_test/` | immutable submitted archive and score provenance |
| `results/heterogeneous/canonical_runtime/` | portable development replay inputs |
| `results/research_summaries/` | reviewed experiment outcomes; never runtime inputs |
| `tests/` | unit and artifact-contract tests |
| `docs/ARCHITECTURE.md` | architecture and code map |
| `docs/REPRODUCIBILITY.md` | exact replay, fresh inference, and resume instructions |
| `docs/RELEASE_SCOPE.md` | what is supported, retained, generated, and ignored |

Later research runners remain in the repository for auditability, but are not
the public paper-system entry point. No retained component should be deleted
merely because its filename reflects an earlier experiment: the final runtime
imports much of this tested dependency closure.

## Setup

Python 3.11 and CUDA-capable PyTorch are required for model inference. The
portable direct-dependency lock is `requirements-lock.txt`; the complete
dependency closure recorded for the final Linux/CUDA 13.0 environment is
`requirements-repro-cu130.txt`.

```bash
conda create -n lm-kbc-2026 python=3.11 -y
conda activate lm-kbc-2026
pip install -r requirements-lock.txt
```

On a compatible Linux/CUDA 13.0 host, use the fuller reproduction target:

```bash
pip install -r requirements-repro-cu130.txt
```

Gemma is a gated Hugging Face checkpoint. Accept its model license and log in
with a read token before preflight; never store that token in the repository.

The release is qualified on Linux with four 11-GiB NVIDIA RTX 2080 Ti GPUs.
The launcher detects fewer visible NVIDIA GPUs and can trade throughput for
model sharding, but a successful preflight is required before a full run.
CPU-only, Apple-Silicon, AMD/ROCm, and arbitrary low-memory CUDA inference are
not claimed as supported configurations. Model-free verification, scoring,
planning, and unit tests remain CPU-safe.

For tests only:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Verify the release without loading model weights

```bash
python scripts/verify_release.py
bash experiments/heterogeneous_agents/run_paper_system.sh test
```

The verifier checks split identities, the exact SyntheticCoT pool, the
paper-system route contract, frozen decoder hashes, model revisions and
parameter totals, both retained development candidates, and both the archive
and inner `predictions.jsonl` hashes of the official submission.

## Fresh blind-test inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUT=experiments/heterogeneous_agents/runs/paper_test \
bash experiments/heterogeneous_agents/run_paper_system.sh all
```

The launcher detects the number of visible GPUs and uses one quantized
Gemma/Ministral worker per device. Each stage is resumable. The historical
control package is written to:

```text
OUT/submission/historical_sota_replication_control_test.zip
```

Fresh sampling reproduces the pinned architecture, prompts, revisions, and
seed policy. It is not claimed to reproduce the old sampled text byte for
byte across CUDA, PyTorch, Transformers, and quantization environments. Exact
result reproduction uses the immutable submitted archive instead.

## Development policy

- Tune and ablate on training/validation data only.
- Never score the blind test locally.
- Do not commit generated `runs/` directories or credentials.
- Add a unit or contract test for decoder or graph-schema changes.
- Do not replace a frozen artifact without updating its manifest and
  provenance.

## Upstream and license

The shared-task data, evaluator, and baseline originated in
[`lm-kbc/dataset2026`](https://github.com/lm-kbc/dataset2026). Code and data in
this repository retain the upstream Apache-2.0 license.
