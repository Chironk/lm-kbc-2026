# Paper-system architecture and code map

## Supported architecture

The release system has one model-generation phase and one deterministic
decoding phase.

1. **Plan label-free tasks.** The planner strips answer labels, selects
   relation-matched SyntheticCoT demonstrations, binds model and prompt hashes,
   and creates resumable task files.
2. **Sample the three models independently.** Qwen supplies the initial answer
   and ten proposal samples, Gemma supplies one independent proposal, and
   Ministral supplies a zero-shot three-sample route plus a five-shot,
   ten-sample route. The zero-shot route is used only by the retained area
   stage; complete outputs from the other routes remain available to the graph.
3. **Build the evidence graph.** Each sampled complete answer is an evidence
   event. Each normalized answer is a candidate component. A directed
   `supports` edge connects an event to each answer that appeared in that
   complete sampled set. Event attributes preserve the model, route, and sample
   identity, so repetition within one model remains distinct from agreement
   across models.
4. **Construct a provisional set.** Frozen stages begin with Qwen's retained
   answer and apply learned cardinality, numeric, route-residual, surface, and
   high-support Ministral corrections. A failed artifact contract leaves the
   previous answer unchanged.
5. **Apply the rule-based graph correction.** The decoder reconstructs complete
   alternative sets from evidence events. An alternative can replace the
   provisional answer only when at least two model families produced it,
   every family produced a sufficiently compatible complete set, the sampled
   answer structure supports the change, and the result does not expand the
   provisional set. `awardWonBy` keeps the provisional answer because its
   retained Qwen evidence is aggregated differently and is not comparable at
   the sampled-set level.
6. **Package.** The runner verifies row order, split identity, artifact hashes,
   and archive contents before writing one root-level `predictions.jsonl`.

This is a graph-based set decoder, not a GNN. The graph provides explicit,
query-local relationships between sampled answer sets and normalized answers;
the final decision is made by audited deterministic rules.

## Model portfolio

| Model | Pinned checkpoint | Parameters | Generation role |
|---|---|---:|---|
| Qwen | `Qwen/Qwen3.5-9B` | 9,409,813,744 | initial answer and repeated proposals |
| Gemma | `google/gemma-3-12b-it` | 12,187,325,040 | independent proposal |
| Ministral | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 8,918,026,240 | heterogeneous repeated proposals |
| **Total** | | **30,515,165,024** | below the 32B cap |

Exact revisions, runtime settings, and count provenance are stored in
`configs/final/` and `artifacts/frozen/MANIFEST.json`.

## Authoritative entry points

- `run_paper_system.sh` is the public shell interface.
- `historical_sota_test_pipeline.py` pins the architecture associated with the
  official 0.4845 submission and performs graph construction, decoding, and
  packaging.
- `end_to_end_pipeline.py` creates label-free route tasks.
- `run_submission.py` runs the retained Qwen policy.
- `run_agent.py` runs the pinned Gemma and Ministral routes.
- `run_sota_reproduction.sh` replays the tracked development evidence.

The name `historical_sota_test_pipeline.py` is retained deliberately: changing
it would obscure the provenance linking the runner to the archived submission.
`final_submission_pipeline.py` and the modules under `components/` remain
implementation dependencies and records of later audits; they are not a
second advertised paper-system launcher.

## Reproducibility tiers

1. **Exact submitted result:** verify or extract the tracked official zip.
2. **Exact deterministic development replay:** rebuild predictions from tracked
   evidence and frozen small decoder artifacts.
3. **Fresh model inference:** regenerate evidence using pinned checkpoints,
   prompts, and seeds. This reproduces the procedure, while stochastic GPU
   kernels and quantization can produce different sampled text.

These tiers prevent an exact artifact claim from being confused with a claim
of byte-identical generative inference.
