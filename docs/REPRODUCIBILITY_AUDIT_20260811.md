# Reproducibility audit — 2026-08-11

## Scope and standard

This audit covers the architecture associated with the preserved official-test
submission `heterogeneous_final_strict_proof_20260803_v1_test.zip`, reported by
CodaBench at 0.4845 macro-F1. It distinguishes three claims that must not be
conflated:

1. **Exact result preservation**: the submitted prediction bytes and their
   recorded CodaBench score remain available and unchanged.
2. **Deterministic replay**: retained evidence and frozen decoder artifacts can
   reconstruct a prediction artifact and local development score without model
   sampling.
3. **Fresh inference reproduction**: another host can execute the same model,
   prompting, graph, and decoder procedure from the blind query file.

Fresh stochastic inference is not the same as exact byte reproduction. The raw
sampled responses used to construct the 0.4845 submission were not retained.
The historical seed scheme also does not guarantee identical samples across
CUDA, PyTorch, Transformers, quantization, batching, and GPU-topology changes.
Consequently, a fresh run can reproduce the architecture, but it cannot be
truthfully claimed to reproduce the submitted prediction bytes on arbitrary
GPU hardware.

Status labels below mean:

- **PASS**: verified by a machine check in a clean source export.
- **PARTIAL**: the procedure or provenance is preserved, but an exactness claim
  has a documented limitation.
- **FAIL**: the current paper text does not match the executable system.
- **UNVERIFIED**: the required physical-host experiment was not available in
  this audit environment.

## Executive verdict

| Reproduction claim | Status | Evidence |
|---|---:|---|
| Exact 475-row official prediction artifact is preserved | PASS | Zip SHA-256 `3f73d01f…b11e4`; member SHA-256 `73621130…bb53`; exact key order checked against `data/test.jsonl` |
| Recorded official score provenance is preserved | PASS | CodaBench macro-F1 `0.4845` recorded in the immutable submission manifest |
| Frozen development artifacts rescore exactly | PASS | Historical 478-row scores `0.5184496147` and `0.5207285306`; revised 475-row label-free migrations `0.5259345597` and `0.5282278687` |
| Model portfolio and 32B contract are pinned | PASS | Three immutable revisions; 30,515,165,024 total full-checkpoint parameters |
| Frozen decoder/model artifacts are hash-pinned | PASS | `artifacts/frozen/MANIFEST.json` and `scripts/verify_release.py` |
| Public source archive installs and passes CPU tests | PASS | Clean `git archive` verification and full test suite described below |
| CPU-only planning works without an NVIDIA driver | PASS | Regression test added for failing `nvidia-smi` |
| Fresh inference is runnable from the blind split | PARTIAL | Full launcher and resumable route manifests exist; sampled responses behind the submitted artifact are absent |
| Fresh inference recreates the submitted bytes | FAIL | Impossible from the retained artifacts; the release now explicitly disclaims this |
| Same procedure executes on a second GPU architecture | UNVERIFIED | Placement logic supports variable visible-GPU counts, but a physical second-host smoke/full run is still required |
| Current paper description exactly matches the 0.4845 executable | FAIL | Material route, temperature, commitment, and validation-lineage mismatches are listed below |

The release is therefore **not yet eligible for a “100% paper-aligned and
cross-GPU reproduced” claim**. Its exact submitted artifact, score provenance,
data, model portfolio, decoder artifacts, and CPU replay are preserved. The
remaining blockers are explicit and repairable in documentation, except for
fresh byte identity, which must remain a stated limitation.

## Authoritative executable trace

The public entry point is:

```text
experiments/heterogeneous_agents/run_paper_system.sh
  -> experiments/heterogeneous_agents/run_historical_sota_test_pipeline.sh
  -> experiments.heterogeneous_agents.historical_sota_test_pipeline
```

The execution then has four inference routes:

1. The retained Qwen v0495 pipeline, implemented by `run_submission.py` and
   `run_inference.py`, creates the primary answer and relation-specific
   auxiliary Qwen output.
2. Gemma produces one five-shot proposal and separate structural choices when
   the relation schema does not make the choice constant.
3. Ministral produces three zero-shot samples.
4. Ministral produces ten five-shot samples with a requested 40-word
   reasoning line.

The historical runner reconstructs the evidence graph, applies the frozen
cardinality, numeric, route-residual, component, Ministral, and final graph
correction stages, and packages exactly one `predictions.jsonl` member.
The exact Aug-3 source commit cited by the runner is retained by the annotated
tag `provenance/historical-final-submission-20260803`.

The machine-readable architecture contract is
`configs/final/paper_system_contract.json`. `scripts/verify_release.py` checks
that its split, pool, models, revisions, parameter counts, Qwen temperatures,
Qwen auxiliary configuration, heterogeneous route counts, and official
submission hashes still match executable code and retained artifacts.

## Claim-by-claim paper audit

The paper source audited here was the Overleaf snapshot retrieved on
2026-08-11. No Overleaf edits are made by this audit.

### 1. “Every retained sampling route uses temperature 0.8” — FAIL

The primary Qwen v0495 route uses ten samples at `TEMPERATURE = 1.0` in
`run_inference.py`. Gemma and both Ministral routes use 0.8. The appendix also
lists Qwen at 0.8, so both method and appendix are inaccurate.

Required paper correction: report Qwen primary at 1.0 and Gemma/Ministral at
0.8. Do not describe one global temperature.

### 2. Qwen inference is described as one N=10 route — FAIL

The primary Qwen incumbent also uses an auxiliary route with the same
checkpoint and no additional checkpoint parameters:

- stock exchange: one direct answer used by guarded promotion;
- city of death: five temperature-0.8 reasoned candidates followed by a
  same-model judge call;
- awards: up to four rounds of five temperature-0.8 candidate lists followed
  by same-model judge calls.

This behavior is configured in `configs/baseline-qwen-3.5-9b.yaml` and invoked
by `run_submission.py`. It is operative, not an obsolete experiment. Omitting
it makes the paper architecture and appendix prompts incomplete.

Required paper correction: disclose this as relation-specific auxiliary
inference by the already-counted Qwen checkpoint. Include its actual prompt
contract in the appendix or remove the auxiliary path from the claimed system
and rerun all reported results.

### 3. “Qwen and Gemma answer two structural questions” — FAIL

Gemma has separate structural-choice tasks when needed. In the historical
final runner, Qwen does not make separate candidate-blind calls. Qwen’s
existence and zero/one/many values are deterministically derived from its
already selected primary answer with probability 1.0. The appendix’s shared
“Qwen/Gemma candidate-blind commitment” prompt is therefore false for Qwen.

Required paper correction: distinguish derived Qwen structure from separately
elicited Gemma structure. Ministral supplies proposal sets only in this
historical architecture.

### 4. SyntheticCoT pool description — PARTIAL

The final pool identity is reproducible and verified:

- SHA-256 `72f9974c…1aacb`;
- 1,856 rows;
- 389 of 477 training query keys represented;
- between 1 and 14 retained traces per covered query;
- five deterministic same-relation, target-subject-excluded examples selected
  by each few-shot route.

The paper currently says the teacher generated “up to six” traces for each
labeled query. That is not an accurate description of the final pool because
the pool contains relation-specific top-ups and 88 training queries have no
retained trace. The final pool itself is preserved, but teacher generation is
API-based and stochastic.

The original teacher scripts remain reachable in repository history at commit
`3f27f15` (also retained by the pre-cleanup archive branch), but are not part of
the inference runtime. The teacher was Gemini 3.1 Pro Preview, followed by a
second pass of the same teacher for strategy-faithfulness filtering.

Required paper correction: say that the final filtered pool contains 1,856
traces covering 389 training queries; avoid claiming a uniform maximum per
query. State that five eligible examples are selected per evaluation prompt.

### 5. Validation score and test score are presented as one architecture — FAIL

The paper pairs a revised-validation score of 0.5259 with the official-test
score of 0.4845. The 0.5259 artifact is a label-free rekey of the historical
“safe” development artifact. That lineage includes a validation-subject
capacity ledger and predates the final strict graph correction. The strict
graph-corrected rekey scores 0.5282278687. The unseen-test runner correctly
does not apply the validation-subject ledger.

Therefore 0.5259 and 0.4845 are both real preserved results, but they are not
outputs of one identical executable architecture. The exact 0.4845 test
archive is the authoritative test result for the historical submission.

Required paper correction: either (a) label 0.5259 as a transferred historical
validation artifact and avoid calling it the exact test architecture, or (b)
run one portable frozen architecture on validation and report that matched
score. Do not imply an apples-to-apples architecture match that the artifacts
do not support.

### 6. Appendix Qwen prompt — FAIL

The appendix prompt resembles the newer heterogeneous proposal contract. The
primary v0495 Qwen path actually uses the raw-ChatML prompt assembled by
`run_inference.build_prompt`, including the historical relation-specific
instructions and legacy CoT output protocol. The operative auxiliary Qwen
candidate/judge prompts are also absent. The appendix also reports a 192-token
Qwen limit, whereas the executable defaults are 256 tokens and 384 for awards.

Required paper correction: replace the appendix Qwen prompt with the actual
v0495 primary prompt template, correct its token limits, and disclose the
auxiliary prompt types.

### 7. Ministral N=3 scope — PARTIAL

The plan generates the zero-shot N=3 route for all 475 test queries, although
its named direct replacement action is geographic-area unanimity. Because the
route is attached before graph construction, describing it merely as “area
also N=3” understates the computation and potentially its graph availability.

Required paper correction: say that the route is generated for all queries but
has an explicit area replacement action, or change the executable plan to
generate it only for area and rerun the reported system.

### 8. Model identities and parameter budget — PASS

The following full-checkpoint counts and immutable revisions are verified:

| Model | Revision | Parameters |
|---|---|---:|
| Qwen/Qwen3.5-9B | `c2022362…7b9a` | 9,409,813,744 |
| google/gemma-3-12b-it | `7553b6f3…8c7` | 12,187,325,040 |
| mistralai/Ministral-3-8B-Instruct-2512-BF16 | `f6fae979…734` | 8,918,026,240 |
| **Total** | | **30,515,165,024** |

Unused vision/projector parameters remain included in the legal full-checkpoint
total even where the runtime strips them for text-only inference.

### 9. Blind-test and no-retrieval discipline — PASS

The tracked test split contains 475 unique subject–relation keys, has SHA-256
`67c31c83…43f1`, and contains no nonempty labels. The final decoder artifacts
do not contain test labels or test-subject decision ledgers. Model prompts are
closed-book and no retrieval system is invoked at evaluation time.

## Portability audit

### Software

The recorded host used Python 3.11.15, PyTorch 2.13.0+cu130, Transformers
5.13.0, BitsAndBytes 0.49.2, and CUDA 13.0. Direct portable requirements are
in `requirements-lock.txt`; the full recorded Linux/CUDA dependency closure is
in `requirements-repro-cu130.txt`.

### Hardware placement

The completed reported runs used four 11-GiB RTX 2080 Ti GPUs. The launcher
does not hard-code device UUIDs:

- quantized Gemma and Ministral isolate one visible GPU per worker;
- the 4-bit Qwen border pass can run replicas over visible GPUs;
- fp16 Qwen and its auxiliary path use one process with automatic placement
  over the visible devices;
- CPU-only plan, status, packaging, and tests no longer require working NVML.

The code can plan for fewer visible GPUs, but **support by placement logic is
not empirical cross-GPU reproduction**. Before claiming portability to a
two-GPU RTX 3080 Ti host, that host must run:

```bash
python scripts/verify_release.py
pytest -q
CUDA_VISIBLE_DEVICES=0,1 OUT=/path/to/new/run \
  bash experiments/heterogeneous_agents/run_paper_system.sh plan
CUDA_VISIBLE_DEVICES=0,1 OUT=/path/to/new/run \
  bash experiments/heterogeneous_agents/run_paper_system.sh preflight
```

It must then complete at least one real generation task from each of Qwen,
Gemma, and Ministral while preserving the emitted environment and placement
logs. A full second-host run is needed to evaluate score stability; no local
test labels are available for the blind split.

### Expected determinism

Pinned revisions and seeds make the experiment controlled, not bitwise
portable. GPU kernels, quantization, floating-point reductions, worker
partitioning, and sampling can change token probabilities enough to alter
sampled text. The correct release claim is therefore procedural
reproducibility plus exact preservation of the submitted artifact, not
cross-hardware bit identity.

## Clean-export verification protocol

The release must be validated from a clean committed source export, not the
developer worktree:

```bash
git archive --format=tar HEAD | tar -xf - -C /tmp/lm-kbc-release-audit
cd /tmp/lm-kbc-release-audit
python scripts/verify_release.py
pytest -q
OUT=/tmp/lm-kbc-paper-plan \
  bash experiments/heterogeneous_agents/run_paper_system.sh plan
```

Passing this protocol proves that no ignored local run directory, private
credential, Git object outside `HEAD`, or developer-only import is required by
the CPU release path. GPU preflight and inference remain separate physical-host
checks.

## Preservation and deletion audit

No file or directory was deleted, moved, or cleaned during this audit. Existing
ignored run artifacts and `.env` files were not modified. The changes are
additive or narrow release-safety edits:

- added a machine-readable paper-system contract;
- expanded artifact, route, pool, and result verification;
- added the full recorded CUDA dependency closure;
- made CPU-only launcher stages independent of NVML;
- added a regression test for that behavior;
- documented exact-result, replay, and fresh-inference boundaries.

The official submission archive is intentionally tracked and must not be
removed. Historical research branches and pre-cleanup provenance must also be
kept until the paper and code release are finalized. The annotated historical
source tag must be pushed along with `main`; otherwise a remote clone cannot
resolve the source commit named by the runner.

## Release blockers

1. Correct the paper’s Qwen temperature, auxiliary route, structural-commitment
   description, SyntheticCoT coverage, prompt appendix, and validation-lineage
   wording.
2. Decide whether the paper reports the historical two-Ministral-route 0.4845
   architecture or a later simplified architecture. Do not mix their method
   text and result.
3. Commit the audit changes and rerun the clean-export protocol at that exact
   commit.
4. Push `main` and the
   `provenance/historical-final-submission-20260803` tag.
5. Run the documented smoke/preflight sequence on the second GPU host before
   claiming cross-GPU execution.
6. Preserve the limitation that fresh stochastic inference is not expected to
   recreate the archived prediction bytes.
