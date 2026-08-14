# Public release audit (2026-08-13)

## Scope

This audit covers the tracked release tree, all 216 reachable Git commits,
the supported paper-system launcher, frozen result artifacts, dependency
records, and a reconstruction from a Git-only source archive. Local ignored
run directories were not used as inputs.

## Public-data hygiene

- No GitHub, Hugging Face, OpenAI-style, Google, AWS, bearer, or private-key
  credential signature was found in the current tracked tree or reachable Git
  commits.
- No private IP address, SSH host, email address, tracked dotenv file, key, or
  certificate was found in the release tree.
- A local `.env` exists only in the working copy and is ignored by Git. Its
  contents were not read or copied.
- Seventy-two historical provenance strings in 30 tracked files contained a
  user-specific repository prefix. They were changed to repository-relative
  paths. No model coefficient, prediction, label, policy, or decoder decision
  was changed. The five affected artifact hashes and their manifests were
  updated.
- `scripts/verify_release.py` now fails if a public release contains a known
  credential signature, user-specific home path, or private-network endpoint.

Older commits and archival refs still contain historical workstation paths.
Those strings are not credentials. Removing them from Git history would
require a destructive history rewrite, which was deliberately not performed.
The current release tree is clean.

## Reproduction checks

The following checks passed in the recorded Python 3.11 environment:

- `pip check`: no broken requirements;
- complete test suite: 361 passed;
- public runner contract suite: 8 passed;
- shell syntax checks for all supported launchers;
- release verifier: 290 tracked files, 30.515B parameters, exact split and
  artifact hashes, zero public-tree hygiene findings;
- exact official-test archive:
  `3f73d01fe5d4b3c9b9cc7e2f5dba8348d0e1fec19fc0ddb797ff2e0f460b11e4`;
- exact development replay prediction:
  `6c4d4bb1ed60054cb1b2d9a6aa728a1a5f0422c714bdd8486963cc986ca348ae`,
  byte-identical at macro-F1 0.5184496147269507.

The verifier, public tests, development replay, and blind-test planning were
also run from a source archive containing only `git ls-files`. The planner
created all four tracked evidence routes for the correct 475-row blind split
without accessing an ignored local run.

## Supported reviewer commands

Verify every retained result and public-tree contract without loading model
weights:

```bash
bash experiments/heterogeneous_agents/run_paper_system.sh verify-release
bash experiments/heterogeneous_agents/run_paper_system.sh test
```

Replay the exact retained development evidence:

```bash
OUT=experiments/heterogeneous_agents/runs/development_replay \
bash experiments/heterogeneous_agents/run_sota_reproduction.sh all
```

Run fresh blind-test inference and packaging:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUT=experiments/heterogeneous_agents/runs/paper_test \
bash experiments/heterogeneous_agents/run_paper_system.sh all
```

The fresh-inference command is resumable and requires accepted Gemma access.
It reproduces the pinned procedure, not necessarily the original stochastic
response bytes across different CUDA and quantization stacks. Exact test-result
reproduction uses the retained Codabench archive because test labels are not
distributed.

## Hardware boundary

The full inference path is qualified on Linux with four 11-GiB RTX 2080 Ti
GPUs. Fewer visible NVIDIA GPUs are detected and used with lower throughput,
but arbitrary CUDA layouts, CPU-only inference, Apple Silicon, and ROCm are not
claimed as validated. Model-free verification, tests, scoring, and planning are
portable CPU operations.
