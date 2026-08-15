# Published graph-free CoT ablation artifacts

This directory contains the exact model-response caches and prediction files
underlying the scores in `../analysis/RESULT.json`. None of these files contain
development labels; scoring uses the separately versioned `data/val.jsonl`.

## Contents

- `responses/`: complete Qwen, Gemma, and Ministral CoT response caches used by
  the graph-free scorer. Qwen and Gemma include their recorded commitment rows,
  although the graph-free analysis decodes only `phase=propose`.
- `predictions/`: the eight exact 475-row JSONL files scored for the reported
  single-model and ensemble policies.
- `MANIFEST.json`: portable byte sizes, row counts, and SHA-256 hashes.

The original runtime manifests contained machine-local absolute paths. The
portable response manifests retain the fields used by the repository's cache
validator while replacing those path-bearing records.

## Score an exact published prediction

From the repository root:

```bash
python evaluate.py \
  --predictions results/paper/model_aggregation_ablation/artifacts/predictions/qwen_gemma_ministral_majority.jsonl \
  --ground_truth data/val.jsonl
```

## Re-run the decoder from the published responses

First create the deterministic validation plan, then hydrate its response
directory with the published caches and portable manifests:

```bash
OUT=runs/published_graphless_replay \
  SPLIT=validation INPUT=data/val.jsonl QUESTION_CONTRACT=official-v1 \
  SYNTHETIC_COT=data/synthetic_cot_capacity_aligned_v2.jsonl \
  bash scripts/ablations/run_end_to_end_pipeline.sh plan

mkdir -p runs/published_graphless_replay/responses
cp results/paper/model_aggregation_ablation/artifacts/responses/* \
  runs/published_graphless_replay/responses/

OUT=runs/published_graphless_replay \
  bash scripts/ablations/run_graphless_cot_ensemble_ablation.sh score
```
