# Graph-free CoT ensemble ablation

This development comparison uses the current 475-row validation split and the
completed CoT-5 caches. The blind test was not scored.

Only each model's proposal generations are decoded. The experiment does not
use existence or cardinality commitments, candidate graphs, graph routing, or
the final graph decoder.

| policy | pooled macro-F1 | award | company | borders | area | capacity | city of death |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen only | 0.462866 | 0.003766 | 0.661905 | 0.979903 | 0.340000 | 0.113402 | 0.420000 |
| Gemma only | 0.424555 | 0.043356 | 0.645238 | 0.922152 | 0.310000 | 0.113402 | 0.320000 |
| Ministral only | 0.442345 | 0.023431 | 0.564667 | 0.961958 | 0.420000 | 0.103093 | 0.360000 |
| Qwen + Gemma union | 0.415107 | 0.046990 | 0.644833 | 0.924841 | 0.270000 | 0.113402 | 0.313333 |
| Qwen + Gemma agreement | **0.451424** | 0.000000 | 0.656667 | 0.967053 | 0.270000 | 0.113402 | 0.450000 |
| Three-model union | 0.446181 | 0.058948 | 0.582222 | 0.913589 | 0.430000 | 0.154639 | 0.330000 |
| Three-model majority | **0.504111** | 0.010396 | 0.711905 | 0.972918 | 0.430000 | 0.154639 | 0.440000 |
| Three-model unanimity | 0.473711 | 0.000000 | 0.573333 | 0.965873 | 0.430000 | 0.154639 | 0.440000 |

For string relations, agreement counts distinct models after canonicalization.
For numeric relations, the scorer uses a plain median when the required number
of models returned one valid scalar. It does not cluster values or require the
numbers to agree within the official tolerance.

All predeclared policies are shown. Every row in this table uses the same
current 475-row validation file and the same completed model-response caches.

## Paper interpretation

Among these graph-free controls, Qwen is the strongest individual route at
0.462866. Requiring Qwen and Gemma to agree reaches 0.451424, which is 0.011443
below Qwen alone. Adding Ministral makes a two-of-three rule possible; that arm
reaches 0.504111, improving by 0.041245 over Qwen alone and by 0.052688 over the
two-model agreement arm. The union controls are weaker, consistent with
unsupported alternatives reducing precision.

These comparisons describe complete graph-free systems on validation. They are
not leave-one-model-out estimates of each model's marginal contribution, and
they do not isolate the effect of the graph. No full-system or graph-effect
delta should be reported until the full architecture and a matched graph arm
are scored on these same 475 rows with explicitly matched evidence routes.

## Reproduction

Create the locked project environment described in the repository, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/ablations/run_graphless_cot_ensemble_ablation.sh generate
```

The generation stages are resumable. If the three validated response caches
already exist, reproduce only the decoding and scores with:

```bash
bash scripts/ablations/run_graphless_cot_ensemble_ablation.sh score
```

The exact response caches and all eight graded prediction JSONL files are
published under `../artifacts/`. Its portable manifest records byte sizes, row
counts, hashes, model revisions, seed, prompt contract, and demonstration pool.
The larger local run tree remains ignored.

The checkpoints, prompts, task order, dependencies, and task-derived random
seeds are pinned. The response hashes identify the exact reported run. Fresh
stochastic generation can still differ across hardware or numerical kernels,
so reproducibility here means a fully specified rerun plus exact provenance for
the reported caches, not a claim of cross-hardware bitwise identity.
