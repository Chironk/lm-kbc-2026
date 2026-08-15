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

## Use these two entry points

For the paper system, use only these two public launchers:

```bash
# Run or resume the complete inference pipeline.
CUDA_VISIBLE_DEVICES=0,1,2,3 OUT=runs/paper_test \
bash scripts/run_paper_system.sh all

# Reproduce the reported validation result from frozen evidence.
OUT=runs/development_replay \
bash scripts/reproduce_validation.sh all
```

`scripts/run_paper_system.sh` is the supported fresh-inference interface, and
`scripts/reproduce_validation.sh` is the supported deterministic validation
replay. Files under `scripts/internal/` are implementation details; files
under `scripts/ablations/` reproduce individual controls only. The Python
modules under `src/lm_kbc/` are imported by these launchers and are not
competing public entry points.

## Repository map

```text
.
├── src/lm_kbc/
│   ├── historical_sota_test_pipeline.py    internal final-system orchestration
│   ├── end_to_end_pipeline.py              internal label-free route planning
│   ├── run_agent.py                        internal checkpoint-pinned inference
│   └── components/
│       ├── route_aware_candidate_graph.py   internal candidate construction
│       ├── relational_candidate_graph.py    internal graph construction
│       └── proof_carrying_graph_decoder.py  internal graph correction
│
├── scripts/
│   ├── run_paper_system.sh                 PUBLIC fresh-inference entry point
│   ├── reproduce_validation.sh             PUBLIC validation-replay entry point
│   ├── verify_release.py                   verifies contracts and result hashes
│   ├── internal/                           internal launcher implementation
│   └── ablations/                          ablation-only launchers
│
├── configs/final/
│   ├── paper_system_contract.json          authoritative frozen system contract
│   └── portfolio_cot.json                  model checkpoints and route settings
│
├── artifacts/frozen/
│   └── MANIFEST.json                       hashes for retained decoder artifacts
│
├── submissions/official_test/              exact submitted archive and manifest
├── results/heterogeneous/candidates/
│   └── frozen_20260811_current_validation/ reported validation predictions
├── results/paper/model_aggregation_ablation/
│                                             reported model/aggregation controls
│
├── tests/                                   unit and reproducibility tests
├── docs/
│   ├── OFFICIAL_TASK.md                     task and evaluator documentation
│   └── figures/                             paper architecture figure and source
│
├── data/                                    official task data
├── models/                                  model loading and inference wrappers
├── evaluate.py                              official prediction evaluator
└── requirements-lock.txt                    portable Python dependencies
```

The installable implementation is in `src/lm_kbc/`, but normal use should go
through the two public shell entry points above. Validation-control launchers
are separated under `scripts/ablations/`.

## Setup

Python 3.11 and CUDA-capable PyTorch are required for model inference. The
portable direct-dependency lock is `requirements-lock.txt`; the complete
dependency closure recorded for the final Linux/CUDA 13.0 environment is
`requirements-repro-cu130.txt`.

```bash
conda create -n lm-kbc-2026 python=3.11 -y
conda activate lm-kbc-2026
pip install -r requirements-lock.txt
pip install -e .
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
bash scripts/run_paper_system.sh verify-release
bash scripts/run_paper_system.sh test
```

The verifier checks split identities, the exact SyntheticCoT pool, the
paper-system route contract, frozen decoder hashes, model revisions and
parameter totals, both retained development candidates, and both the archive
and inner `predictions.jsonl` hashes of the official submission. It also rejects
tracked credential signatures, private-network endpoints, and user-specific
workstation paths. These two commands and the `all` command below are the
supported public interface; reviewers do not need to invoke internal research
runners.

## Validation replay

```bash
OUT=runs/development_replay \
bash scripts/reproduce_validation.sh all

python -m lm_kbc.rekey_frozen_validation
```

The first command reconstructs the retained 478-query staged prediction. The
second verifies the paper's graph-corrected result on the revised 475-query
validation split.

## Fresh blind-test inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUT=runs/paper_test \
bash scripts/run_paper_system.sh all
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
