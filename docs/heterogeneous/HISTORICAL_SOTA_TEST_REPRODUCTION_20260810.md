# Historical 0.4845 test lineage and paired paper-system run

## Outcome of the audit

The official-test submission reported as macro-F1 **0.4845** is preserved at:

`submissions/official_test/heterogeneous_final_strict_proof_20260803_v1_test.zip`

Its archive SHA-256 is
`3f73d01fe5d4b3c9b9cc7e2f5dba8348d0e1fec19fc0ddb797ff2e0f460b11e4`.
The only zip member is `predictions.jsonl`; its SHA-256 is
`73621130839b572a7fdfdc2f8a58c4bf3f00beece4be86ff4a7874c96b63bb53`
and it contains 475 rows.

The decoder source that froze this policy is identified by commit
`130b9a0c02ba0d190f9b33cd61cd74156daf6625` and blob
`e199d6c1a09deb41e291d8be7f74933283cd3e73`. The source depends on graph and
proof modules that were uncommitted at that point but were captured in the
later research checkpoint; the release copies differ from those checkpoint
files only by module relocation/import paths.

The archived sampled model responses were not retained. Therefore:

- the exact submitted predictions and score are reproducible by artifact
  replay;
- the architecture, prompts, revisions, seeds, graph, and decoder are
  reproducible by fresh inference; but
- fresh stochastic inference is not claimed to reproduce the archived bytes.

The paired runner reports the row-by-row distance between a fresh historical
control and the archived submission after decoding. It never copies an old
test answer into the fresh output.

## Why the later 0.466 run was not a reproduction

`final_test_release_candidate_20260809_v2` changed both evidence generation
and decoding:

- it used the capacity-aligned SyntheticCoT pool instead of the faithful pool;
- it used the new official capacity question rather than the legacy prompt;
- it added a singleton-numeric capacity graph proof; and
- it included later parser-inventory repair logic.

Its prediction differs from the archived 0.4845 submission on **113/475**
rows: 45 capacity, 29 company, 19 city, 9 borders, 8 awards, and 3 area rows.
It must not be used as the historical control.

## Exact historical control

The reconstructed control pins:

- official 475-row test input SHA-256
  `67c31c8388c585634df55500612f522ad42da6735d4c89eb59a9ef5a39f043f1`;
- Qwen3.5-9B revision `c202236...`, production policy `v0495`, legacy
  row-position seed scheme, CoT-5, N=10;
- Gemma-3-12B revision `7553b6f...`, CoT-5, N=1;
- Ministral-3-8B revision `f6fae97...`, zero-shot N=3 and CoT-5/N=10;
- faithful SyntheticCoT pool SHA-256
  `72f9974c355dd98eab9d13e61a6b2e120a8e9fcc40e39fb8251b54ab8d01aacb`;
- legacy relation-question wording;
- all five frozen learned decoder artifacts; and
- the Aug-3 decoder order ending in the generic strict set proof. Capacity
  does not receive the later singleton-numeric proof.

## Paired paper-system change

The second output uses the same fresh Qwen, Gemma, and Ministral N=10
responses and the same frozen decoder. Its only policy change is:

- remove the zero-shot Ministral N=3 route and its area-unanimity rule;
- for `hasArea`, apply the frozen 7/10 threshold to the unique complete-link
  5%-tolerance numeric component from the CoT-5/N=10 Ministral route.

For every non-area relation, the N=10 rule and all other decoder stages are
unchanged. The runner creates both graphs independently and asserts that the
N=3 route is absent from the paper graph.

## Runner and outputs

Runner:

`experiments/heterogeneous_agents/run_historical_sota_test_pipeline.sh`

Implementation:

`experiments/heterogeneous_agents/historical_sota_test_pipeline.py`

Default run directory:

`experiments/heterogeneous_agents/runs/historical_sota_single_ministral_test_20260810_v1`

The plan is already frozen. To execute all GPU stages and package both blind
outputs:

```bash
cd /path/to/lm-kbc-2026
conda activate lm-kbc-2026

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
experiments/heterogeneous_agents/run_historical_sota_test_pipeline.sh generate

experiments/heterogeneous_agents/run_historical_sota_test_pipeline.sh build
experiments/heterogeneous_agents/run_historical_sota_test_pipeline.sh package
```

The two packages will be:

- `submission/historical_sota_replication_control_test.zip`
- `submission/paper_single_ministral_n10_test.zip`

`FINAL_MANIFEST.json` records hashes, the fresh-control distance from the
archived submission, and the exact rows changed by the single-Ministral policy
without reading test labels.
