#!/usr/bin/env python3
"""Train-only audit of singleton-aware symbolic numeric graph correction.

The retained strict graph rule requires a challenger to improve cardinality
agreement.  That is meaningful for variable-length entity sets but impossible
when both incumbent and challenger are valid scalar numeric answers.  This
module predeclares the numeric counterpart:

* incumbent and challenger must both be positive non-empty singletons;
* the challenger must be a coherent complete-link 5% numeric component;
* all three model families must have generated a value in that component;
* at least two more families must support it than support the incumbent; and
* no cardinality/existence advantage is required because both sides are ONE.

There are no fitted parameters.  The rule is evaluated on subject-grouped
training folds and is rejected unless all hard promotion checks pass.  The
default command never reads validation or test labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.capacity_graph_decoder import (
    FAMILIES,
    RELATION,
    TOLERANCE,
    _component_values,
    _correct,
    _format,
    capacity_options,
)
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.graph_event_contract import (
    assert_event_support_invariants,
    repair_unsupported_candidate_set_events,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import _key
from experiments.heterogeneous_agents.sota_pipeline import (
    compose_competition_train_oof,
)
from experiments.heterogeneous_agents.three_model_component_decoder import (
    subject_grouped_folds,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_TRAIN_GRAPH = (
    HERE / "runs/capacity_graph_decoder_20260809_v3/"
    "PREPARED_TRAIN_CAPACITY_GRAPH.jsonl"
)
DEFAULT_VALIDATION_GRAPH = (
    HERE / "runs/cot40_cardinality_validation_confirmation_20260730_v1/"
    "graph/VALIDATION_GRAPH.jsonl"
)
DEFAULT_TRAIN_GOLD = ROOT / "data/train.jsonl"
DEFAULT_VALIDATION_GOLD = ROOT / "data/val.jsonl"
DEFAULT_VALIDATION_INCUMBENT = (
    HERE / "runs/portable_unified_validation_20260809_v1/"
    "VALIDATION_PREDICTIONS.jsonl"
)
DEFAULT_OUTPUT = HERE / "runs/strict_numeric_proof_20260809_v1"

PLAN_SCHEMA = "strict-numeric-proof-plan-v1"
RESULT_SCHEMA = "strict-numeric-proof-train-result-v1"
VALIDATION_MANIFEST_SCHEMA = "strict-numeric-proof-validation-manifest-v1"
VALIDATION_RESULT_SCHEMA = "strict-numeric-proof-validation-result-v1"
POOLED = "*** All Relations ***"
MIN_EXACT_FAMILY_FRACTION = 2.0 / 3.0
MIN_EXACT_FAMILY_ADVANTAGE = 2.0 / 3.0
MIN_POSITIVE_FOLDS = 3
MAX_FOLD_REGRESSION = -0.05  # one capacity row in a 20-row fold
MIN_HELP_HARM_RATIO = 2.0
EPSILON = 1e-12


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _prediction_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    result = {
        _key(row): list(map(str, row.get("ObjectEntities", [])))
        for row in rows
    }
    if len(result) != len(rows):
        raise ContractError("duplicate prediction keys")
    return result


def _coherent_component(
    graph: Mapping[str, Any], option: Mapping[str, Any],
) -> bool:
    values = _component_values(graph)
    members = [values.get(str(component)) for component in option["component_ids"]]
    if not members or any(value is None for value in members):
        return False
    numeric = [float(value) for value in members if value is not None]
    return all(
        abs(left - right) / max(abs(right), 1e-12) <= TOLERANCE + EPSILON
        for left in numeric for right in numeric
    )


def strict_numeric_eligible(
    graph: Mapping[str, Any], challenger: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> bool:
    """Return whether a scalar challenger satisfies the frozen proof."""
    if str(graph["Relation"]) not in {"hasArea", "hasCapacity"}:
        raise ContractError("strict numeric proof applied to nonnumeric relation")
    if not challenger or not incumbent:
        return False
    # Both hypotheses are positive, non-empty singleton numeric components.
    if not (
        math.isfinite(float(challenger["value"]))
        and float(challenger["value"]) > 0.0
        and math.isfinite(float(incumbent["value"]))
        and float(incumbent["value"]) > 0.0
    ):
        return False
    if not _coherent_component(graph, challenger):
        return False

    challenger_fraction = float(challenger["family_fraction"])
    incumbent_fraction = float(incumbent["family_fraction"])
    if challenger_fraction + EPSILON < MIN_EXACT_FAMILY_FRACTION:
        return False
    if (
        challenger_fraction - incumbent_fraction + EPSILON
        < MIN_EXACT_FAMILY_ADVANTAGE
    ):
        return False

    # For scalar events, a family is compatible exactly when at least one of
    # its generations lands in the candidate's 5% component.  Requiring all
    # three families prevents a two-family majority from overriding a third
    # family that generated only incompatible values.
    if any(float(challenger["rates"].get(family, 0.0)) <= 0.0
           for family in FAMILIES):
        return False
    return True


def decode_row(
    graph: Mapping[str, Any], incumbent_objects: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    options = capacity_options(graph, incumbent_objects)
    incumbent = next(option for option in options if option["is_incumbent"])
    eligible = [
        option for option in options
        if not option["is_incumbent"]
        and strict_numeric_eligible(graph, option, incumbent)
    ]
    selected = max(
        eligible,
        key=lambda option: (
            float(option["family_fraction"]),
            float(option["minimum_family_rate"]),
            float(option["mean_family_rate"]),
            float(option["total_event_rate"]),
            -abs(math.log(float(option["value"]) / float(incumbent["value"]))),
            -float(option["value"]),
        ),
        default=incumbent,
    )
    return [_format(float(selected["value"]))], {
        "changed": selected is not incumbent,
        "incumbent": [_format(float(incumbent["value"]))],
        "selected": [_format(float(selected["value"]))],
        "eligible_options": len(eligible),
        "incumbent_family_fraction": float(incumbent["family_fraction"]),
        "selected_family_fraction": float(selected["family_fraction"]),
        "selected_family_rates": dict(selected["rates"]),
        "cardinality_contract": "equal_nonempty_singletons",
    }


def _repair_graph_file(source: Path, output: Path) -> dict[str, Any]:
    rows = read_jsonl(source)
    repaired_rows = []
    repaired_events = 0
    repaired_rows_count = 0
    removed_assertions = 0
    for row in rows:
        repaired, audit = repair_unsupported_candidate_set_events(row)
        assert_event_support_invariants(repaired)
        repaired_rows.append(repaired)
        repaired_events += int(audit["repaired_events"])
        repaired_rows_count += int(audit["repaired_events"] > 0)
        removed_assertions += int(audit["removed_assertion_edges"])
    write_jsonl_atomic(output, repaired_rows)
    return {
        "rows": len(rows),
        "rows_repaired": repaired_rows_count,
        "events_repaired": repaired_events,
        "assertion_edges_removed": removed_assertions,
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
    }


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    implementation = Path(__file__).resolve()
    train_source = Path(args.train_graph).resolve()
    validation_source = Path(args.validation_graph).resolve()
    validation_incumbent = Path(args.validation_incumbent).resolve()
    if not (
        train_source.is_file()
        and validation_source.is_file()
        and validation_incumbent.is_file()
    ):
        raise ContractError("strict numeric proof input graph is missing")
    train_repaired = output / "graph/REPAIRED_TRAIN_CAPACITY_GRAPH.jsonl"
    validation_repaired = output / "graph/REPAIRED_VALIDATION_GRAPH.jsonl"
    train_repair = _repair_graph_file(train_source, train_repaired)
    validation_repair = _repair_graph_file(validation_source, validation_repaired)
    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "validation_labels_opened": False,
        "test_labels_opened": False,
        "implementation": str(implementation),
        "implementation_sha256": sha256(implementation),
        "train_graph": str(train_repaired),
        "train_graph_sha256": sha256(train_repaired),
        "validation_graph": str(validation_repaired),
        "validation_graph_sha256": sha256(validation_repaired),
        "validation_incumbent": str(validation_incumbent),
        "validation_incumbent_sha256": sha256(validation_incumbent),
        "train_repair": train_repair,
        "validation_repair": validation_repair,
        "proof": {
            "numeric_relations": ["hasArea", "hasCapacity"],
            "minimum_exact_family_fraction": MIN_EXACT_FAMILY_FRACTION,
            "minimum_exact_family_advantage": MIN_EXACT_FAMILY_ADVANTAGE,
            "third_family": "must_support_same_5pct_component",
            "cardinality": "equal_nonempty_singletons",
        },
        "promotion_thresholds": {
            "positive_oof_delta": True,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "maximum_fold_regression": MAX_FOLD_REGRESSION,
            "minimum_help_harm_ratio": MIN_HELP_HARM_RATIO,
        },
    }
    path = output / "plan/PLAN.json"
    _write_json(path, plan)
    print(json.dumps({
        "plan": str(path),
        "plan_sha256": sha256(path),
        "train_repair": train_repair,
        "validation_repair": validation_repair,
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(output: Path) -> dict[str, Any]:
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("validation_labels_opened") is not False
        or plan.get("test_labels_opened") is not False
        or plan.get("implementation_sha256") != sha256(Path(__file__).resolve())
        or sha256(Path(plan["train_graph"])) != plan.get("train_graph_sha256")
        or sha256(Path(plan["validation_graph"]))
            != plan.get("validation_graph_sha256")
        or sha256(Path(plan["validation_incumbent"]))
            != plan.get("validation_incumbent_sha256")
    ):
        raise ContractError("strict numeric proof plan contract failed")
    return plan


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    graphs = read_jsonl(Path(plan["train_graph"]))
    if len(graphs) != 100 or any(str(row["Relation"]) != RELATION for row in graphs):
        raise ContractError("expected 100 train capacity graphs")
    incumbent_rows, _ = compose_competition_train_oof()
    incumbents = _prediction_map(incumbent_rows)
    gold_path = Path(args.train_gold).resolve()
    gold = {_key(row): row for row in read_jsonl(gold_path)}
    folds = subject_grouped_folds(graphs, n_folds=5)
    decisions = []
    for graph in graphs:
        key = _key(graph)
        if key not in incumbents or key not in gold:
            raise ContractError(f"train coverage mismatch: {key}")
        selected, detail = decode_row(graph, incumbents[key])
        before = _correct(incumbents[key], gold[key])
        after = _correct(selected, gold[key])
        decisions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "fold": int(folds[key]),
            "before_correct": before,
            "after_correct": after,
            "delta": after - before,
            **detail,
        })

    baseline = sum(float(row["before_correct"]) for row in decisions) / len(decisions)
    selected = sum(float(row["after_correct"]) for row in decisions) / len(decisions)
    changed = sum(bool(row["changed"]) for row in decisions)
    helped = sum(float(row["delta"]) > 0 for row in decisions)
    harmed = sum(float(row["delta"]) < 0 for row in decisions)
    neutral = changed - helped - harmed
    fold_deltas = {}
    for fold in range(5):
        subset = [row for row in decisions if int(row["fold"]) == fold]
        fold_deltas[str(fold)] = (
            sum(float(row["delta"]) for row in subset) / len(subset)
        )
    checks = {
        "positive_oof_delta": selected > baseline + EPSILON,
        "minimum_positive_folds": sum(
            delta > EPSILON for delta in fold_deltas.values()
        ) >= MIN_POSITIVE_FOLDS,
        "maximum_fold_regression": min(fold_deltas.values())
            >= MAX_FOLD_REGRESSION - EPSILON,
        "minimum_help_harm_ratio": helped
            >= MIN_HELP_HARM_RATIO * max(1, harmed),
    }
    passed = all(checks.values())
    write_jsonl_atomic(output / "analysis/TRAIN_DECISIONS.jsonl", decisions)
    result = {
        "schema": RESULT_SCHEMA,
        "contains_labels": True,
        "gold_aware": True,
        "train_only": True,
        "validation_labels_opened": False,
        "test_labels_opened": False,
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "baseline_oof": baseline,
        "strict_numeric_proof_oof": selected,
        "oof_delta": selected - baseline,
        "changed": changed,
        "helped": helped,
        "harmed": harmed,
        "neutral": neutral,
        "fold_deltas": fold_deltas,
        "promotion_checks": checks,
        "promotion_gate_passed": passed,
        "next_stage": (
            "freeze_numeric_proof_before_validation" if passed
            else "reject_structural_numeric_proof"
        ),
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)
    lines = [
        "# Strict singleton-numeric graph proof — train-only audit", "",
        f"Promotion gate passed: **{passed}**", "",
        f"- production train OOF capacity: **{baseline:.4f}**",
        f"- strict numeric proof capacity: **{selected:.4f}**",
        f"- delta: **{selected - baseline:+.4f}**",
        f"- changed: **{changed}**",
        f"- helpful / harmful / neutral: **{helped} / {harmed} / {neutral}**",
        "", "## Fold deltas", "",
        *[f"- fold {fold}: {delta:+.4f}"
          for fold, delta in fold_deltas.items()],
        "", "## Promotion checks", "",
        *[f"- {name}: **{passed_check}**"
          for name, passed_check in checks.items()],
        "", "Validation and test labels were not opened.",
    ]
    (output / "analysis/RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "promotion_gate_passed": passed,
        "baseline_oof": baseline,
        "strict_numeric_proof_oof": selected,
        "oof_delta": selected - baseline,
        "changed": changed,
        "helped": helped,
        "harmed": harmed,
        "result": str(result_path),
    }, indent=2, sort_keys=True))
    return 0


def decode_validation_diagnostic(args: argparse.Namespace) -> int:
    """Freeze a post-hoc validation decode without opening validation gold."""
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    train_result = _json(output / "analysis/RESULT.json")
    if (
        train_result.get("schema") != RESULT_SCHEMA
        or train_result.get("plan_sha256") != sha256(output / "plan/PLAN.json")
    ):
        raise ContractError("strict numeric train result contract failed")
    graphs = read_jsonl(Path(plan["validation_graph"]))
    incumbent_rows = read_jsonl(Path(plan["validation_incumbent"]))
    if len(graphs) != 478 or len(incumbent_rows) != 478:
        raise ContractError("validation coverage mismatch")
    incumbents = _prediction_map(incumbent_rows)
    graph_by = {_key(row): row for row in graphs}
    if set(graph_by) != set(incumbents):
        raise ContractError("validation graph/incumbent key mismatch")

    decisions = []
    replacements: dict[tuple[str, str], list[str]] = {}
    for key, graph in graph_by.items():
        if key[1] != RELATION:
            continue
        selected, detail = decode_row(graph, incumbents[key])
        replacements[key] = selected
        decisions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "posthoc_after_failed_train_gate": not bool(
                train_result.get("promotion_gate_passed")),
            **detail,
        })
    if len(decisions) != 100:
        raise ContractError("expected 100 capacity validation decisions")
    predictions = [{
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
        "ObjectEntities": list(map(str, replacements.get(
            _key(row), row.get("ObjectEntities", [])))),
    } for row in incumbent_rows]
    prediction_path = output / "validation/VALIDATION_PREDICTIONS.jsonl"
    decision_path = output / "validation/VALIDATION_DECISIONS.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(decision_path, decisions)
    manifest = {
        "schema": VALIDATION_MANIFEST_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "deployable": False,
        "posthoc_after_failed_train_gate": not bool(
            train_result.get("promotion_gate_passed")),
        "validation_labels_opened": False,
        "rows": len(predictions),
        "capacity_rows": len(decisions),
        "capacity_rows_changed": sum(
            bool(row["changed"]) for row in decisions),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "decisions": str(decision_path),
        "decisions_sha256": sha256(decision_path),
        "incumbent": plan["validation_incumbent"],
        "incumbent_sha256": plan["validation_incumbent_sha256"],
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_result_sha256": sha256(output / "analysis/RESULT.json"),
    }
    manifest_path = prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def score_validation_diagnostic(args: argparse.Namespace) -> int:
    """Open validation labels only after diagnostic predictions are frozen."""
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    prediction_path = output / "validation/VALIDATION_PREDICTIONS.jsonl"
    manifest_path = prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != VALIDATION_MANIFEST_SCHEMA
        or manifest.get("contains_labels") is not False
        or manifest.get("predictions_sha256") != sha256(prediction_path)
        or manifest.get("plan_sha256") != sha256(output / "plan/PLAN.json")
    ):
        raise ContractError("validation diagnostic manifest contract failed")
    predictions = read_jsonl(prediction_path)
    incumbents = read_jsonl(Path(plan["validation_incumbent"]))
    gold_path = Path(args.validation_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    selected_scores = score(predictions, gold_rows)
    incumbent_scores = score(incumbents, gold_rows)
    gold = {_key(row): row for row in gold_rows}
    incumbent_by = _prediction_map(incumbents)
    selected_by = _prediction_map(predictions)
    capacity_keys = [key for key in selected_by if key[1] == RELATION]
    paired = []
    for key in capacity_keys:
        before = _correct(incumbent_by[key], gold[key])
        after = _correct(selected_by[key], gold[key])
        if incumbent_by[key] != selected_by[key]:
            paired.append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "before": incumbent_by[key],
                "after": selected_by[key],
                "before_correct": before,
                "after_correct": after,
                "delta": after - before,
            })
    result = {
        "schema": VALIDATION_RESULT_SCHEMA,
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "posthoc_after_failed_train_gate": True,
        "validation_labels_opened_after_prediction_freeze": True,
        "validation_labels_used_for_policy_selection": False,
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "validation_gold": str(gold_path),
        "validation_gold_sha256": sha256(gold_path),
        "incumbent_scores": incumbent_scores,
        "scores": selected_scores,
        "deltas": {
            key: selected_scores[key] - incumbent_scores[key]
            for key in selected_scores
        },
        "capacity_changed": len(paired),
        "capacity_helpful": sum(row["delta"] > 0 for row in paired),
        "capacity_harmful": sum(row["delta"] < 0 for row in paired),
        "capacity_neutral": sum(row["delta"] == 0 for row in paired),
        "capacity_paired_edits": paired,
    }
    result_path = output / "validation/RESULT.json"
    _write_json(result_path, result)
    lines = [
        "# Strict singleton-numeric graph proof — validation diagnostic", "",
        "This diagnostic was run after the train gate failed and is not a",
        "deployable or independently selected result.", "",
        f"- incumbent pooled F1: **{incumbent_scores[POOLED]:.6f}**",
        f"- numeric-proof pooled F1: **{selected_scores[POOLED]:.6f}**",
        f"- pooled delta: **{selected_scores[POOLED] - incumbent_scores[POOLED]:+.6f}**",
        f"- incumbent capacity F1: **{incumbent_scores[RELATION]:.4f}**",
        f"- numeric-proof capacity F1: **{selected_scores[RELATION]:.4f}**",
        f"- capacity delta: **{selected_scores[RELATION] - incumbent_scores[RELATION]:+.4f}**",
        f"- capacity helpful / harmful / neutral: "
        f"**{result['capacity_helpful']} / {result['capacity_harmful']} / "
        f"{result['capacity_neutral']}**",
    ]
    (output / "validation/RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "incumbent_pooled": incumbent_scores[POOLED],
        "selected_pooled": selected_scores[POOLED],
        "pooled_delta": selected_scores[POOLED] - incumbent_scores[POOLED],
        "incumbent_capacity": incumbent_scores[RELATION],
        "selected_capacity": selected_scores[RELATION],
        "capacity_delta": selected_scores[RELATION] - incumbent_scores[RELATION],
        "capacity_changed": len(paired),
        "capacity_helpful": result["capacity_helpful"],
        "capacity_harmful": result["capacity_harmful"],
        "result": str(result_path),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--train-graph", default=str(DEFAULT_TRAIN_GRAPH))
    prepare_parser.add_argument(
        "--validation-graph", default=str(DEFAULT_VALIDATION_GRAPH))
    prepare_parser.add_argument(
        "--validation-incumbent", default=str(DEFAULT_VALIDATION_INCUMBENT))
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.set_defaults(function=prepare)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    analyze_parser.add_argument("--train-gold", default=str(DEFAULT_TRAIN_GOLD))
    analyze_parser.set_defaults(function=analyze)
    decode_parser = sub.add_parser("decode-validation-diagnostic")
    decode_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    decode_parser.set_defaults(function=decode_validation_diagnostic)
    score_parser = sub.add_parser("score-validation-diagnostic")
    score_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    score_parser.add_argument(
        "--validation-gold", default=str(DEFAULT_VALIDATION_GOLD))
    score_parser.set_defaults(function=score_validation_diagnostic)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
