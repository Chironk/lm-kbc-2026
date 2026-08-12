# Release checklist

This checklist is intentionally non-destructive. It does not authorize removal
of experiments, results, or untracked work.

## Before committing

- [ ] `python scripts/verify_release.py` passes.
- [ ] `pytest -q` passes in the pinned environment.
- [ ] `bash experiments/heterogeneous_agents/run_paper_system.sh test` passes.
- [ ] The verifier passes from an unpacked `git archive` without `.git`.
- [ ] `git status --short` has been reviewed file by file.
- [ ] No `.env`, credentials, raw `runs/`, caches, or model weights are tracked.
- [ ] The official-test archive and manifest hashes are unchanged.
- [ ] README commands work from the repository root.

## Result claims

- [ ] Official test macro-F1 0.4845 is attributed to Codabench, not locally
      recomputed.
- [ ] Exact artifact replay is distinguished from fresh stochastic inference.
- [ ] Development results are labeled as development-selected.
- [ ] Later research pipelines are not presented as the reported paper system.

## Before pushing

- [ ] Review the staged diff; do not use `git add -A` blindly.
- [ ] Commit only the reviewed release files.
- [ ] Push the intended branch to `https://github.com/Chironk/lm-kbc-2026.git`.
- [ ] Clone or unpack the release in a clean directory and rerun the verifier.
