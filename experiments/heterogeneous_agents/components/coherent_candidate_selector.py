#!/usr/bin/env python3
"""Joint candidate-level selector over the coherent frozen action graph.

The previous coherent selector ranked a row's alternatives, discarded all but
the top one, and only then decided KEEP/switch.  Its OOF shortlist audit found
helpful coverage of 45/107 at top-1 but 81/107 at top-3 and 103/107 at top-5.
This module removes that lossy boundary:

* every canonical, deduplicated alternative receives a helpfulness score;
* the highest-scoring alternative competes directly with KEEP via a threshold;
* candidate weights preserve equal relation mass, equal row mass, and equal
  mass among candidates within a row;
* model regularization and KEEP threshold are selected inside each outer
  subject fold from inner subject-grouped OOF predictions.

The graph, model generations, reviews, incumbent, and folds are frozen.
Validation is absent.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.coherent_row_selector import (
    HIERARCHICAL_FEATURE_NAMES,
    SHARED_FEATURE_NAMES,
    TRUTH_HIERARCHICAL_FEATURE_NAMES,
    CoherentAction,
    CoherentRow,
    _rank_features,
    load_coherent_rows,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.crossfit_action_utility_selector import (
    DEFAULT_GOLD,
    DEFAULT_SOURCE,
    TRUTH_FEATURE_NAMES,
    _control_predictions,
    _key,
    _write_json,
)
from experiments.heterogeneous_agents.components.multi_challenger_graph_decoder import (
    EPSILON,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
)
from experiments.heterogeneous_agents.components.truth_calibrated_action_decoder import (
    StandardizedLinear,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    RELATIONS,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/heterogeneous_agents/runs"
    / "coherent_candidate_selector_20260727_v2"
)
ARMS: Mapping[str, tuple[bool, bool, bool, bool | str]] = {
    "shared": (False, False, False, False),
    "paired_delta": (False, False, True, False),
    "paired_delta_relation": (False, False, True, True),
}
PRIMARY_ARM = "paired_delta_relation"
L2_GRID = (1.0, 10.0, 100.0)
THRESHOLD_GRID = (
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.60, 0.70,
    0.80, 0.90,
)
MIN_PROMOTION_DELTA = 0.010
MIN_WINNING_FOLDS = 3
MIN_FOLD_DELTA = -0.010
MIN_RELATION_DELTA = -0.015
PAIRED_FEATURE_NAMES = (
    "truth_edit_advantage",
    "truth_edit_expected_correctness",
    "truth_mean_delta",
    "truth_qwen_delta",
    "truth_gemma_delta",
    "truth_delta_agreement",
    "binary_qwen_margin",
    "binary_gemma_margin",
    "binary_mean_margin",
    "binary_direction_agreement",
    "added_joint_support",
    "removed_joint_support",
    "action_empty",
    "action_replace",
    "action_add",
    "action_drop",
)
PAIRED_RELATION_FEATURE_NAMES = (
    PAIRED_FEATURE_NAMES
    + tuple(f"relation:{relation}" for relation in RELATIONS)
)
PAIRED_INTERACTION_SIGNAL_NAMES = (
    "truth_edit_advantage",
    "truth_mean_delta",
    "truth_qwen_delta",
    "truth_gemma_delta",
    "binary_qwen_margin",
    "binary_gemma_margin",
    "added_joint_support",
    "removed_joint_support",
)
PAIRED_HIERARCHICAL_FEATURE_NAMES = (
    PAIRED_RELATION_FEATURE_NAMES
    + tuple(
        f"relation:{relation}*{name}"
        for relation in RELATIONS
        for name in PAIRED_INTERACTION_SIGNAL_NAMES
    )
)
PAIRED_INTERACTION_SCALE = 1.0 / 5.0
_SHARED_INDEX = {
    name: index for index, name in enumerate(SHARED_FEATURE_NAMES)}


def _feature_names(
    *,
    hierarchical: bool,
    include_truth: bool,
    paired: bool,
    relation_bias: bool | str,
) -> tuple[str, ...]:
    if paired:
        if relation_bias == "interactions":
            return PAIRED_HIERARCHICAL_FEATURE_NAMES
        return (
            PAIRED_RELATION_FEATURE_NAMES
            if relation_bias else PAIRED_FEATURE_NAMES
        )
    if hierarchical:
        return (
            TRUTH_HIERARCHICAL_FEATURE_NAMES
            if include_truth else HIERARCHICAL_FEATURE_NAMES
        )
    return (
        SHARED_FEATURE_NAMES + TRUTH_FEATURE_NAMES
        if include_truth else SHARED_FEATURE_NAMES
    )


def _features(
    row: CoherentRow,
    action: CoherentAction,
    *,
    hierarchical: bool,
    include_truth: bool,
    paired: bool,
    relation_bias: bool | str,
) -> list[float]:
    if paired:
        truth = action.source.truth_features
        shared = action.shared_features
        q_delta = float(truth[6] - truth[8])
        g_delta = float(truth[7] - truth[9])
        values = [
            float(truth[13]),
            float(truth[12]),
            float(truth[0] - truth[3]),
            q_delta,
            g_delta,
            float(q_delta * g_delta >= 0.0),
            float(shared[_SHARED_INDEX["binary_qwen_margin"]]),
            float(shared[_SHARED_INDEX["binary_gemma_margin"]]),
            float(shared[_SHARED_INDEX["binary_mean_margin"]]),
            float(shared[_SHARED_INDEX["binary_direction_agreement"]]),
            float(shared[_SHARED_INDEX["added_joint_support"]]),
            float(shared[_SHARED_INDEX["removed_joint_support"]]),
            float(action.source.action["action_type"] == "EMPTY"),
            float(action.source.action["action_type"] == "REPLACE"),
            float(action.source.action["action_type"] == "ADD"),
            float(action.source.action["action_type"] == "DROP"),
        ]
        if relation_bias:
            values.extend(
                float(row.relation == relation) for relation in RELATIONS)
        if relation_bias == "interactions":
            paired_by_name = dict(zip(
                PAIRED_FEATURE_NAMES,
                values[:len(PAIRED_FEATURE_NAMES)],
                strict=True,
            ))
            for relation in RELATIONS:
                active = float(row.relation == relation)
                values.extend(
                    active
                    * paired_by_name[name]
                    * PAIRED_INTERACTION_SCALE
                    for name in PAIRED_INTERACTION_SIGNAL_NAMES
                )
        expected = _feature_names(
            hierarchical=hierarchical,
            include_truth=include_truth,
            paired=paired,
            relation_bias=relation_bias,
        )
        if len(values) != len(expected):
            raise AssertionError("paired delta feature schema drift")
        return values
    return _rank_features(
        row,
        action,
        hierarchical,
        include_truth=include_truth,
    )


def _candidate_weights(rows: Sequence[CoherentRow]) -> list[float]:
    usable = [row for row in rows if row.alternatives]
    relation_rows = Counter(row.relation for row in usable)
    output = []
    for row in usable:
        candidate_mass = (
            1.0
            / len(RELATIONS)
            / relation_rows[row.relation]
            / len(row.alternatives)
        )
        output.extend([candidate_mass] * len(row.alternatives))
    return output


def _fit(
    rows: Sequence[CoherentRow],
    *,
    hierarchical: bool,
    include_truth: bool,
    paired: bool,
    relation_bias: bool | str,
    l2: float,
) -> StandardizedLinear:
    usable = [row for row in rows if row.alternatives]
    matrix = [
        _features(
            row,
            action,
            hierarchical=hierarchical,
            include_truth=include_truth,
            paired=paired,
            relation_bias=relation_bias,
        )
        for row in usable for action in row.alternatives
    ]
    targets = [
        float(action.delta > EPSILON)
        for row in usable for action in row.alternatives
    ]
    return StandardizedLinear(
        _feature_names(
            hierarchical=hierarchical,
            include_truth=include_truth,
            paired=paired,
            relation_bias=relation_bias,
        ),
        l2,
        logistic=True,
    ).fit(matrix, targets, _candidate_weights(usable))


def _probabilities(
    model: StandardizedLinear,
    rows: Sequence[CoherentRow],
    *,
    hierarchical: bool,
    include_truth: bool,
    paired: bool,
    relation_bias: bool | str,
) -> dict[tuple[str, str], list[float]]:
    output = {}
    for row in rows:
        if not row.alternatives:
            output[row.key] = []
            continue
        matrix = [
            _features(
                row,
                action,
                hierarchical=hierarchical,
                include_truth=include_truth,
                paired=paired,
                relation_bias=relation_bias,
            )
            for action in row.alternatives
        ]
        output[row.key] = list(map(float, model.predict(matrix)))
    return output


def _decode(
    rows: Sequence[CoherentRow],
    probabilities: Mapping[tuple[str, str], Sequence[float]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for row in rows:
        values = probabilities[row.key]
        selected_index = None
        selected_probability = 0.0
        if values:
            selected_index = max(
                range(len(values)),
                key=lambda index: (
                    values[index],
                    row.alternatives[index].canonical_output,
                ),
            )
            selected_probability = float(values[selected_index])
        changed = (
            selected_index is not None
            and selected_probability > threshold
        )
        action = (
            row.alternatives[selected_index]
            if selected_index is not None else None
        )
        objects = (
            action.source.action["objects"]
            if changed and action is not None
            else row.source.keep["objects"]
        )
        delta = (
            action.delta if changed and action is not None else 0.0)
        oracle = max([
            0.0, *(candidate.delta for candidate in row.alternatives)])
        predictions.append({
            "SubjectEntity": row.key[0],
            "Relation": row.key[1],
            "ObjectEntities": list(objects),
        })
        diagnostics.append({
            "SubjectEntity": row.key[0],
            "Relation": row.key[1],
            "fold": row.fold,
            "changed": changed,
            "selected_action_id": (
                str(action.source.action["id"])
                if changed and action is not None
                else str(row.source.keep["id"])
            ),
            "top_candidate_action_id": (
                str(action.source.action["id"])
                if action is not None else None
            ),
            "top_candidate_probability": selected_probability,
            "top_candidate_delta": (
                float(action.delta) if action is not None else 0.0),
            "actual_delta": float(delta),
            "shortlist_oracle_delta": float(oracle),
            "threshold": float(threshold),
            "alternative_count": len(row.alternatives),
        })
    return predictions, diagnostics


def _quality(
    predictions: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    rows: Sequence[CoherentRow],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    selected = score(
        list(predictions), [gold_by[row.key] for row in rows])
    control = score(
        _control_predictions([row.source for row in rows]),
        [gold_by[row.key] for row in rows],
    )
    changed = [row for row in diagnostics if row["changed"]]
    return {
        "score": selected["*** All Relations ***"],
        "control": control["*** All Relations ***"],
        "delta": (
            selected["*** All Relations ***"]
            - control["*** All Relations ***"]),
        "relation_deltas": {
            relation: selected[relation] - control[relation]
            for relation in RELATIONS
        },
        "changed": len(changed),
        "helped": sum(
            row["actual_delta"] > EPSILON for row in changed),
        "harmed": sum(
            row["actual_delta"] < -EPSILON for row in changed),
        "neutral": sum(
            abs(row["actual_delta"]) <= EPSILON for row in changed),
        "top_candidate_helpful": sum(
            row["top_candidate_delta"] > EPSILON
            for row in diagnostics),
        "headroom_rows": sum(
            row["shortlist_oracle_delta"] > EPSILON
            for row in diagnostics),
    }


def _inner_select(
    rows: Sequence[CoherentRow],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    hierarchical: bool,
    include_truth: bool,
    paired: bool,
    relation_bias: bool | str,
) -> tuple[float, float, dict[str, Any]]:
    folds = sorted({row.fold for row in rows})
    candidates = {}
    for l2 in L2_GRID:
        probabilities = {}
        for fold in folds:
            fit = [row for row in rows if row.fold != fold]
            hold = [row for row in rows if row.fold == fold]
            model = _fit(
                fit,
                hierarchical=hierarchical,
                include_truth=include_truth,
                paired=paired,
                relation_bias=relation_bias,
                l2=l2,
            )
            probabilities.update(_probabilities(
                model,
                hold,
                hierarchical=hierarchical,
                include_truth=include_truth,
                paired=paired,
                relation_bias=relation_bias,
            ))
        for threshold in THRESHOLD_GRID:
            predictions, diagnostics = _decode(
                rows, probabilities, threshold)
            quality = _quality(
                predictions, diagnostics, rows, gold_by)
            candidates[(l2, threshold)] = quality
    selected = max(
        candidates,
        key=lambda item: (
            candidates[item]["score"],
            -candidates[item]["harmed"],
            -candidates[item]["changed"],
            item[1],
            item[0],
        ),
    )
    return selected[0], selected[1], {
        "selected_l2": selected[0],
        "selected_threshold": selected[1],
        "quality": candidates[selected],
        "grid": [
            {"l2": l2, "threshold": threshold, **candidates[(l2, threshold)]}
            for l2 in L2_GRID for threshold in THRESHOLD_GRID
        ],
    }


def _cross_fit(
    rows: Sequence[CoherentRow],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    hierarchical: bool,
    include_truth: bool,
    paired: bool,
    relation_bias: bool | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics, reports = [], [], []
    for outer in sorted({row.fold for row in rows}):
        fit = [row for row in rows if row.fold != outer]
        hold = [row for row in rows if row.fold == outer]
        l2, threshold, inner = _inner_select(
            fit,
            gold_by,
            hierarchical=hierarchical,
            include_truth=include_truth,
            paired=paired,
            relation_bias=relation_bias,
        )
        model = _fit(
            fit,
            hierarchical=hierarchical,
            include_truth=include_truth,
            paired=paired,
            relation_bias=relation_bias,
            l2=l2,
        )
        probabilities = _probabilities(
            model,
            hold,
            hierarchical=hierarchical,
            include_truth=include_truth,
            paired=paired,
            relation_bias=relation_bias,
        )
        fold_predictions, fold_diagnostics = _decode(
            hold, probabilities, threshold)
        quality = _quality(
            fold_predictions, fold_diagnostics, hold, gold_by)
        predictions.extend(fold_predictions)
        diagnostics.extend(fold_diagnostics)
        reports.append({
            "fold": outer,
            "fit_rows": len(fit),
            "hold_rows": len(hold),
            "inner_selection": inner,
            "holdout": quality,
            "model": model.to_dict(),
        })
    prediction_by = {_key(row): row for row in predictions}
    diagnostic_by = {_key(row): row for row in diagnostics}
    expected = {row.key for row in rows}
    if set(prediction_by) != expected or set(diagnostic_by) != expected:
        raise ContractError("incomplete outer OOF candidate decisions")
    return (
        [prediction_by[row.key] for row in rows],
        [diagnostic_by[row.key] for row in rows],
        reports,
    )


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source = Path(args.source_run).resolve()
    gold_path = Path(args.gold).resolve()
    rows, provenance = load_coherent_rows(source, gold_path)
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    results, artifacts = {}, {}
    for arm, (
        hierarchical, include_truth, paired, relation_bias,
    ) in ARMS.items():
        print(f"coherent candidate selector OOF arm={arm}", flush=True)
        predictions, diagnostics, folds = _cross_fit(
            rows,
            gold_by,
            hierarchical=hierarchical,
            include_truth=include_truth,
            paired=paired,
            relation_bias=relation_bias,
        )
        prediction_path = output / arm / "TRAIN_OOF_PREDICTIONS.jsonl"
        diagnostic_path = output / arm / "TRAIN_OOF_DIAGNOSTICS.jsonl"
        fold_path = output / arm / "FOLDS.json"
        write_jsonl_atomic(prediction_path, predictions)
        write_jsonl_atomic(diagnostic_path, diagnostics)
        _write_json(fold_path, {"arm": arm, "folds": folds})
        quality = _quality(
            predictions, diagnostics, rows, gold_by)
        fold_deltas = [item["holdout"]["delta"] for item in folds]
        results[arm] = {
            **quality,
            "fold_deltas": fold_deltas,
            "winning_folds": sum(value > 0.0 for value in fold_deltas),
            "worst_fold": min(fold_deltas),
            "feature_count": len(_feature_names(
                hierarchical=hierarchical,
                include_truth=include_truth,
                paired=paired,
                relation_bias=relation_bias,
            )),
        }
        artifacts[arm] = {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
            "folds": str(fold_path),
            "folds_sha256": sha256(fold_path),
        }

    primary = results[PRIMARY_ARM]
    checks = {
        "minimum_delta": primary["delta"] >= MIN_PROMOTION_DELTA,
        "minimum_winning_folds": (
            primary["winning_folds"] >= MIN_WINNING_FOLDS),
        "minimum_fold_delta": (
            primary["worst_fold"] >= MIN_FOLD_DELTA),
        "minimum_relation_delta": (
            min(primary["relation_deltas"].values())
            >= MIN_RELATION_DELTA),
        "beats_shared": (
            primary["delta"] > results["shared"]["delta"]),
    }
    result = {
        "schema": "coherent-candidate-selector-result-v1",
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "primary_arm": PRIMARY_ARM,
        "arms": results,
        "artifacts": artifacts,
        "promotion_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "failure_ledger": {
            "headroom_rows": primary["headroom_rows"],
            "top_candidate_helpful": primary["top_candidate_helpful"],
            "helpful_selected": primary["helped"],
            "harmful_selected": primary["harmed"],
            "candidate_identity_loss": (
                primary["headroom_rows"]
                - primary["top_candidate_helpful"]),
            "keep_switch_loss": (
                primary["top_candidate_helpful"] - primary["helped"]),
        },
        "provenance": provenance,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "development_only": True,
        "deployable": False,
        "next_stage": (
            "freeze_and_prepare_one_validation_confirmation"
            if all(checks.values())
            else "revise_candidate_level_selector"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "RESULT.json"
    _write_json(result_path, result)
    lines = [
        "# Joint coherent candidate selector",
        "",
        "Nested subject-grouped train OOF; validation was not opened.",
        "",
        "| Arm | Delta | Top helpful | Changed | Helped | Harmed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = results[arm]
        lines.append(
            f"| {arm} | {item['delta']:+.6f} | "
            f"{item['top_candidate_helpful']} | {item['changed']} | "
            f"{item['helped']} | {item['harmed']} |"
        )
    lines.extend([
        "",
        f"Promotion gate: "
        f"`{'PASS' if all(checks.values()) else 'FAIL'}`.",
    ])
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "primary": primary,
        "arms": {
            arm: {
                "delta": item["delta"],
                "top_candidate_helpful": item["top_candidate_helpful"],
                "changed": item["changed"],
                "helped": item["helped"],
                "harmed": item["harmed"],
            }
            for arm, item in results.items()
        },
        "promotion_gate": result["promotion_gate"],
        "failure_ledger": result["failure_ledger"],
        "result": str(result_path),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-run", default=str(DEFAULT_SOURCE))
    result.add_argument("--gold", default=str(DEFAULT_GOLD))
    result.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    result.set_defaults(function=analyze)
    return result


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
