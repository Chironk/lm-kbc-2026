import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Must be set before torch initializes CUDA. Long runs mix many small
# (64-token) generations with occasional large (256-token, for awardWonBy)
# ones; the default caching allocator can fragment enough after the small
# ones that it fails to find one contiguous block for a later large one,
# even though aggregate free memory would be sufficient. This is exactly
# the failure PyTorch's own OOM message suggests fixing.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
import yaml
from loguru import logger

from evaluate import (
    RELATION_TYPE,
    evaluate_per_sr_pair,
    macro_average_per_relation,
    micro_average_per_relation,
    prediction_statistics,
    read_jsonl_file,
)
from models.baseline_qwen import BaselineQwenModel
from models.user_config import Models

DEFAULT_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def write_jsonl_file(rows: List[Dict], file_path: str):
    """Atomic: write to a temp path, then rename (audit P2 -- a crash between
    sequential writes must not leave a partial final-path artifact)."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{file_path}.tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, file_path)


def order_raw_records(raw_records: List[Dict], inputs: List[Dict]) -> List[Dict]:
    """Return exactly one trace per input in input order.

    Baseline generation groups subjects by relation, so the single-process
    model naturally records traces in relation order rather than original
    JSONL order. Multi-GPU already reconstructed order; applying the same
    contract centrally prevents topology-dependent raw artifacts.
    """
    raw_by_key = {
        (record["SubjectEntity"], record["Relation"]): record
        for record in raw_records
    }
    expected = [(item["SubjectEntity"], item["Relation"]) for item in inputs]
    if len(raw_by_key) != len(raw_records):
        raise RuntimeError("System-2 raw trace contains duplicate subject-relation keys")
    missing = [key for key in expected if key not in raw_by_key]
    extra = set(raw_by_key) - set(expected)
    if missing or extra:
        raise RuntimeError(
            f"System-2 raw key mismatch: missing={missing[:3]} extra={list(extra)[:3]}")
    return [raw_by_key[key] for key in expected]


def _gpu_worker(
    worker_idx: int,
    gpu_ids: List[int],
    config: Dict,
    chunk: List[Dict],
    queue: "mp.Queue",
):
    # Must happen before any torch/transformers CUDA call in this process --
    # this restricts the worker to its assigned GPUs, so device_map="auto"
    # inside BaselineQwenModel shards the (full fp16, unquantized) model
    # across only those GPUs, e.g. a 2-GPU tensor/pipeline shard rather than
    # the full 4-GPU shard used in single-worker mode.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    try:
        model_cls = Models.get_model(config["model"])
        model = model_cls(config)
        preds = model.generate_predictions(chunk)
        queue.put((worker_idx, preds, model.raw_records, None))
    except Exception as exc:
        queue.put((worker_idx, None, None,
                   f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def generate_predictions_multi_gpu(
    config: Dict, inputs: List[Dict], num_workers: int
) -> tuple[List[List[str]], List[Dict]]:
    total_gpus = torch.cuda.device_count()
    if num_workers > total_gpus:
        raise ValueError(f"Requested {num_workers} workers but only {total_gpus} GPU(s) visible.")
    if total_gpus % num_workers != 0:
        logger.warning(
            f"{total_gpus} GPUs do not split evenly across {num_workers} workers; "
            "the last worker will get the extra GPU(s)."
        )

    gpus_per_worker = total_gpus // num_workers
    gpu_groups = [
        list(range(w * gpus_per_worker, (w + 1) * gpus_per_worker))
        for w in range(num_workers)
    ]
    leftover = list(range(num_workers * gpus_per_worker, total_gpus))
    gpu_groups[-1].extend(leftover)

    # Each worker gets a fraction of the GPUs the single-worker path would
    # have used (e.g. 2 of 4), so its memory budget shrinks proportionally.
    # batch_size drives peak activation/logits memory (this model's untied
    # ~2GB lm_head scales with batch_size * vocab_size), so scale it down to
    # match -- otherwise the default config's batch_size (tuned for the
    # full 4-GPU budget) OOMs every worker. Verified: 4 GPUs/1 worker keeps
    # batch_size unchanged; 4 GPUs/2 workers (2 GPUs each) forces batch_size
    # down to 1, which is what made this fit in testing.
    original_batch_size = config.get("batch_size", 2)
    scaled_batch_size = max(1, int(original_batch_size * gpus_per_worker / total_gpus))
    if scaled_batch_size != original_batch_size:
        logger.warning(
            f"Scaling batch_size {original_batch_size} -> {scaled_batch_size} per "
            f"worker (each worker gets {gpus_per_worker}/{total_gpus} GPUs, so its "
            "memory budget is smaller than the single-worker case the original "
            "batch_size was set for)."
        )
    config = {**config, "batch_size": scaled_batch_size}

    # Round-robin by index (not contiguous chunks) so each worker gets a
    # representative mix of every relation -- otherwise a worker could get
    # stuck alone on the slow iterative awardWonBy subjects while the others
    # idle after finishing the faster relations.
    chunks: List[List[Dict]] = [[] for _ in range(num_workers)]
    index_map: List[List[int]] = [[] for _ in range(num_workers)]
    for idx, item in enumerate(inputs):
        w = idx % num_workers
        chunks[w].append(item)
        index_map[w].append(idx)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    for w in range(num_workers):
        p = ctx.Process(target=_gpu_worker, args=(w, gpu_groups[w], config, chunks[w], queue))
        p.start()
        processes.append(p)
        logger.info(f"Started worker {w} on GPU(s) {gpu_groups[w]} with {len(chunks[w])} pairs")

    collected: Dict[int, List[List[str]]] = {}
    collected_raw: Dict[int, List[Dict]] = {}
    errors = []
    remaining = len(processes)
    while remaining:
        # Audit P1-4: a bare queue.get() waits forever if a worker is killed
        # (OOM-killer, driver reset) before posting its result. Bounded waits
        # + liveness checks turn a silent hang into an actionable failure.
        try:
            import queue as queue_module
            w, preds, raw_records, error = queue.get(timeout=900)
        except Exception:
            if not any(p.is_alive() for p in processes):
                dead = [(i, p.exitcode) for i, p in enumerate(processes)]
                raise RuntimeError(
                    f"All System-2 workers exited without delivering results "
                    f"(worker, exitcode): {dead}")
            continue  # workers alive, just slow -- keep waiting
        remaining -= 1
        if error is not None:
            errors.append((w, error))
            logger.error(f"Worker {w} failed:\n{error}")
        else:
            collected[w] = preds
            collected_raw[w] = raw_records
            logger.info(f"Worker {w} finished ({len(preds)} predictions)")

    for p in processes:
        p.join()

    if errors:
        raise RuntimeError(f"{len(errors)} worker(s) failed -- see logged tracebacks above")

    predictions: List[Optional[List[str]]] = [None] * len(inputs)
    for w, idxs in enumerate(index_map):
        for local_i, global_i in enumerate(idxs):
            predictions[global_i] = collected[w][local_i]
    ordered_raw = order_raw_records(
        [record for records in collected_raw.values() for record in records], inputs)
    return predictions, ordered_raw


def main():
    parser = argparse.ArgumentParser(
        description="Run a baseline model and evaluate its predictions"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="Path to the model config YAML file"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to the input jsonl file (e.g. data/val.jsonl)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write predictions to (default: output/<model>.jsonl)"
    )
    parser.add_argument(
        "-w", "--num-workers", type=int, default=1,
        help="Number of data-parallel GPU worker processes. Available GPUs "
             "are split evenly across workers (e.g. 2 workers on 4 GPUs -> "
             "one fp16 model replica per worker, each sharded across 2 "
             "GPUs -- no quantization, so results are identical to the "
             "single-worker run, just faster). Default 1 keeps the "
             "original single-process device_map='auto' behavior."
    )
    parser.add_argument("--raw-cache", default=None,
                        help="Optional JSONL trace of candidates, judge outputs, and numeric "
                             "samples, ordered exactly like the input.")
    parser.add_argument("--manifest", default=None,
                        help="Run manifest path (default <raw-cache>.manifest.json when raw "
                             "tracing is enabled).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Stable per-subject sampling seed recorded in raw traces.")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                        help="Pinned Hugging Face model/tokenizer revision.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = {**config, "seed": args.seed,
              "model_revision": args.model_revision}

    output_path = args.output or f"output/{config['model'].replace('_', '-')}.jsonl"

    input_rows = read_jsonl_file(args.input)
    inputs = [
        {"SubjectEntity": row["SubjectEntity"], "Relation": row["Relation"]}
        for row in input_rows
    ]

    if args.num_workers > 1:
        logger.info(
            f"Generating predictions for {len(inputs)} pairs across "
            f"{args.num_workers} GPU worker processes"
        )
        predictions, raw_records = generate_predictions_multi_gpu(
            config, inputs, args.num_workers)
    else:
        logger.info(f"Loading model `{config['model']}` from config `{args.config}`")
        model_cls = Models.get_model(config["model"])
        model = model_cls(config)
        logger.info(f"Generating predictions for {len(inputs)} subject-relation pairs")
        predictions = model.generate_predictions(inputs)
        raw_records = model.raw_records

    raw_records = order_raw_records(raw_records, inputs)

    pred_rows = [
        {
            "SubjectEntity": item["SubjectEntity"],
            "Relation": item["Relation"],
            "ObjectEntities": objects,
        }
        for item, objects in zip(inputs, predictions)
    ]
    # Audit P1-5: validate IN MEMORY before anything reaches disk. The award
    # iterative path has a forced fallback, but a guaranteed-wrong empty on a
    # never-empty relation must be fatal here, not discovered after writing.
    never_empty_violations = [
        (r["SubjectEntity"], r["Relation"]) for r in pred_rows
        if r["Relation"] in BaselineQwenModel.NEVER_EMPTY_RELATIONS
        and not r["ObjectEntities"]]
    if never_empty_violations:
        raise RuntimeError(
            f"never-empty relation produced empty predictions (fail-closed, "
            f"nothing written): {never_empty_violations[:5]}")
    write_jsonl_file(pred_rows, output_path)
    logger.info(f"Wrote {len(pred_rows)} predictions to {output_path}")

    if args.raw_cache:
        if len(raw_records) != len(inputs):
            raise RuntimeError(
                f"Raw trace count {len(raw_records)} != input count {len(inputs)}")
        write_jsonl_file(raw_records, args.raw_cache)
        logger.info(f"Wrote {len(raw_records)} raw trace rows to {args.raw_cache}")

        def sha256(path: str) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        manifest_path = args.manifest or f"{args.raw_cache}.manifest.json"
        manifest = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "model": config["llm_path"],
            "model_revision": args.model_revision,
            "config": args.config,
            "config_sha256": sha256(args.config),
            "input": args.input,
            "input_sha256": sha256(args.input),
            "output": output_path,
            "output_sha256": sha256(output_path),
            "raw_cache": args.raw_cache,
            "raw_cache_sha256": sha256(args.raw_cache),
            "seed": args.seed,
            "num_workers": args.num_workers,
            "n_rows": len(inputs),
            "argv": sys.argv,
        }
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        logger.info(f"Wrote run manifest to {manifest_path}")

    gt_rows = read_jsonl_file(args.input)
    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows, RELATION_TYPE, tolerance=0.05)

    macro_df = pd.DataFrame(macro_average_per_relation(scores_per_sr_pair)).transpose().round(3)
    micro_df = pd.DataFrame(micro_average_per_relation(scores_per_sr_pair)).transpose().round(3)
    stats_df = pd.DataFrame(prediction_statistics(scores_per_sr_pair)).transpose().round(3)
    stats_df["#empty preds"] = stats_df["#empty preds"].astype(int)

    results = pd.concat([macro_df, micro_df, stats_df], axis=1)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results)


if __name__ == "__main__":
    main()
