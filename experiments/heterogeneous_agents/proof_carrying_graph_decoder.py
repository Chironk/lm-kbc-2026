#!/usr/bin/env python3
"""Conservative complete-set decoding over the minimal evidence graph.

This experiment uses only graph evidence that survived the preceding audits:

* exact evidence-event -> candidate support;
* independent model-family provenance;
* complete generated answer-set hypotheses; and
* cardinality/existence compatibility derived from event support degree.

The decoder is deliberately symbolic.  A challenger may replace KEEP only
when it carries a proof consisting of multi-family exact support, agreement
from every family, strictly better exact-cardinality compatibility, no worse
existence compatibility, and no expansion of the incumbent answer set.  The
last condition is a precision guard: unsupported additions are the dominant
failure mode of the unfiltered candidate union.

There are no fitted parameters and no relation-specific rules.  Training
labels are opened only by ``analyze`` to audit the frozen proof rule.  The
validation decode is fail-closed and is available only after the train gate
passes; scoring is a separate command.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.cot40_graph_native_decoder import (
    POOLED,
    RELATIONS,
    cot40_count_anchor,
)
from experiments.heterogeneous_agents.generation_set_hypothesis_audit import (
    FAMILIES,
    _audit,
    _key,
    _row_f1,
    _validate_plan as validate_generation_source,
    build_hypotheses,
    hypothesis_stats,
)
from experiments.heterogeneous_agents.sota_pipeline import (
    compose_competition_train_oof,
)
from experiments.heterogeneous_agents.three_model_component_decoder import (
    subject_grouped_folds,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"
DEFAULT_TRAIN_SOURCE = RUNS / "generation_set_hypothesis_audit_20260801_v1"
DEFAULT_VALIDATION_GRAPH = (
    RUNS / "cot40_cardinality_validation_confirmation_20260730_v1/"
    "graph/VALIDATION_GRAPH.jsonl"
)
DEFAULT_VALIDATION_BASELINE = (
    RUNS / "sota_reproduction_20260729_v1/VALIDATION_PREDICTIONS.jsonl"
)
DEFAULT_OUTPUT = RUNS / "proof_carrying_graph_decoder_20260801_v1"
DEFAULT_TRAIN_GOLD = ROOT / "data/train.jsonl"
DEFAULT_VALIDATION_GOLD = ROOT / "data/val.jsonl"

PLAN_SCHEMA = "proof-carrying-graph-decoder-plan-v1"
RESULT_SCHEMA = "proof-carrying-graph-decoder-result-v1"
PREDICTION_MANIFEST_SCHEMA = (
    "proof-carrying-graph-validation-predictions-manifest-v1"
)

ARMS = (
    "support_consensus",
    "support_cardinality",
    "support_nonexpanding",
    "loose_proof_graph",
    "strict_proof_graph",
    "strict_proof_graph_cardinality_shifted",
)
PRIMARY_ARM = "strict_proof_graph"
IDENTITY_RELATIONS = frozenset({"awardWonBy"})

# Frozen proof constants.  They express graph semantics rather than learned
# thresholds: at least two of three independent families must have generated
# the exact set, and every family must overlap it by at least set-F1 0.5.
MIN_EXACT_FAMILY_FRACTION = 2.0 / 3.0
MIN_EXACT_FAMILY_ADVANTAGE = 2.0 / 3.0
MIN_FAMILY_SIMILARITY = 0.5
EPSILON = 1e-12

MIN_POOLED_DELTA = 0.005
# The strict proof is intentionally sparse.  Requiring four *positive* folds
# rejects policies that simply have no legal edit in two folds.  Three active
# fold wins plus a zero-regression floor tests the intended invariant.
MIN_POSITIVE_FOLDS = 3
MAX_FOLD_REGRESSION = -EPSILON
MAX_RELATION_REGRESSION = -EPSILON
MIN_HELP_HARM_RATIO = 3.0
MIN_ALIGNED_OVER_SHIFTED = 0.003


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


def _artifact_matches(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def _prediction(
    row: Mapping[str, Any], hypothesis: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
        "ObjectEntities": list(hypothesis["objects"]),
    }


def _event_records(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Traverse exact graph edges into typed complete event records."""
    relational = graph["relational_graph"]
    nodes = {
        str(node["id"]): node
        for node in relational.get("nodes", [])
        if node.get("node_type") == "evidence_event"
    }
    support: dict[str, set[str]] = {event: set() for event in nodes}
    asserted_cardinality: dict[str, str] = {}
    asserted_existence: dict[str, str] = {}
    for edge in relational.get("edges", []):
        source, target = str(edge["source"]), str(edge["target"])
        edge_type = str(edge.get("edge_type"))
        if edge_type == "supports":
            if source not in support:
                raise ContractError(f"{_key(graph)}: orphan support source")
            support[source].add(target)
        elif edge_type == "asserts_cardinality":
            asserted_cardinality[source] = target.removeprefix("cardinality:")
        elif edge_type == "asserts_existence":
            asserted_existence[source] = target.removeprefix("existence:")

    records = []
    for event_id, node in sorted(nodes.items()):
        status = str(node.get("status"))
        if status not in ("candidate_set", "explicit_none"):
            continue
        family = str(node.get("model_family"))
        if family not in FAMILIES:
            raise ContractError(f"{_key(graph)}: unknown family {family}")
        tokens = frozenset(support[event_id])
        if (status == "candidate_set") != bool(tokens):
            raise ContractError(f"{_key(graph)}: event status/support mismatch")
        coarse = "ZERO" if not tokens else "ONE" if len(tokens) == 1 else "MANY"
        if asserted_cardinality.get(event_id) != coarse:
            raise ContractError(
                f"{_key(graph)}: cardinality assertion/support mismatch")
        expected_existence = "EMPTY" if not tokens else "NONEMPTY"
        if (
            event_id in asserted_existence
            and asserted_existence[event_id] != expected_existence
        ):
            raise ContractError(
                f"{_key(graph)}: existence assertion/support mismatch")
        records.append({
            "id": event_id,
            "family": family,
            "tokens": tokens,
            # Exact count is more informative than ZERO/ONE/MANY and is
            # recovered losslessly from the degree of the supports edges.
            "exact_cardinality": len(tokens),
            "exists": bool(tokens),
        })
    if not records or {value["family"] for value in records} != set(FAMILIES):
        raise ContractError(f"{_key(graph)}: incomplete family evidence")
    return records


def _proof_metrics(
    hypothesis: Mapping[str, Any],
    stats: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    tokens = frozenset(map(str, hypothesis["tokens"]))
    cardinality = len(tokens)
    cardinality_match = sum(
        int(record["exact_cardinality"]) == cardinality
        for record in records
    ) / len(records)
    existence_match = sum(
        bool(record["exists"]) == bool(tokens)
        for record in records
    ) / len(records)
    result = {
        "exact_family_fraction": float(stats["exact_family_fraction"]),
        "minimum_similarity": float(stats["minimum_similarity"]),
        "mean_similarity": float(stats["mean_similarity"]),
        "within_family_exact_rate_mean": float(
            stats["within_family_exact_rate_mean"]),
        "independent_similarity": float(stats["independent_similarity"]),
        "exact_cardinality_match_rate": float(cardinality_match),
        "existence_match_rate": float(existence_match),
        "set_size": float(cardinality),
    }
    if not all(map(math.isfinite, result.values())):
        raise ContractError("non-finite proof metric")
    return result


def _candidate_is_eligible(
    arm: str,
    challenger: Mapping[str, float],
    incumbent: Mapping[str, float],
) -> bool:
    if (
        challenger["exact_family_fraction"] + EPSILON
            < MIN_EXACT_FAMILY_FRACTION
        or challenger["exact_family_fraction"]
            <= incumbent["exact_family_fraction"] + EPSILON
        or challenger["minimum_similarity"] + EPSILON
            < MIN_FAMILY_SIMILARITY
    ):
        return False
    if arm in ("support_cardinality", "loose_proof_graph",
               "strict_proof_graph",
               "strict_proof_graph_cardinality_shifted"):
        if (
            challenger["exact_cardinality_match_rate"]
            <= incumbent["exact_cardinality_match_rate"] + EPSILON
            or challenger["existence_match_rate"] + EPSILON
            < incumbent["existence_match_rate"]
        ):
            return False
    if arm in ("support_nonexpanding", "loose_proof_graph",
               "strict_proof_graph",
               "strict_proof_graph_cardinality_shifted"):
        if challenger["set_size"] > incumbent["set_size"] + EPSILON:
            return False
    if arm in ("strict_proof_graph",
               "strict_proof_graph_cardinality_shifted"):
        # A two-family exact-support *total* was insufficient: 2-vs-1 is only
        # a one-family margin and proved unstable when pruning true minority
        # objects.  Strict proof requires 2-vs-0 or 3-vs-1/0.
        if (
            challenger["exact_family_fraction"]
            - incumbent["exact_family_fraction"]
            + EPSILON < MIN_EXACT_FAMILY_ADVANTAGE
        ):
            return False
    return True


def _selection_key(
    hypothesis: Mapping[str, Any], metrics: Mapping[str, float],
) -> tuple[Any, ...]:
    return (
        metrics["exact_family_fraction"],
        metrics["minimum_similarity"],
        metrics["exact_cardinality_match_rate"],
        metrics["mean_similarity"],
        metrics["within_family_exact_rate_mean"],
        metrics["independent_similarity"],
        -metrics["set_size"],
        tuple(sorted(map(str, hypothesis["tokens"]))),
    )


def _row_context(
    graph: Mapping[str, Any], incumbent_objects: Sequence[str],
    *, cardinality_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    hypotheses, family_events, statuses = build_hypotheses(
        graph, incumbent_objects)
    own_records = _event_records(graph)
    records = list(cardinality_records or own_records)
    metrics = [
        _proof_metrics(hypothesis, hypothesis_stats(hypothesis, family_events),
                       records)
        for hypothesis in hypotheses
    ]
    keeps = [
        index for index, hypothesis in enumerate(hypotheses)
        if hypothesis["is_incumbent"]
    ]
    if len(keeps) != 1:
        raise ContractError(f"{_key(graph)}: expected one KEEP hypothesis")
    return {
        "hypotheses": hypotheses,
        "metrics": metrics,
        "keep_index": keeps[0],
        "records": own_records,
        "status_counts": statuses,
    }


def _select(
    context: Mapping[str, Any], arm: str,
) -> tuple[int, dict[str, Any]]:
    hypotheses = context["hypotheses"]
    metrics = context["metrics"]
    keep = int(context["keep_index"])
    incumbent = metrics[keep]
    eligible = [
        index for index in range(len(hypotheses))
        if index != keep and _candidate_is_eligible(
            arm, metrics[index], incumbent)
    ]
    selected = (
        max(eligible, key=lambda index: _selection_key(
            hypotheses[index], metrics[index]))
        if eligible else keep
    )
    challenger = metrics[selected]
    return selected, {
        "keep_index": keep,
        "selected_index": selected,
        "eligible_hypotheses": len(eligible),
        "changed": selected != keep,
        "exact_family_advantage": (
            challenger["exact_family_fraction"]
            - incumbent["exact_family_fraction"]
        ),
        "exact_cardinality_advantage": (
            challenger["exact_cardinality_match_rate"]
            - incumbent["exact_cardinality_match_rate"]
        ),
        "existence_advantage": (
            challenger["existence_match_rate"]
            - incumbent["existence_match_rate"]
        ),
        "set_size_delta": challenger["set_size"] - incumbent["set_size"],
        "minimum_family_similarity": challenger["minimum_similarity"],
    }


def _shifted_records(
    graphs: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Same-relation subject shift preserving event-count distributions."""
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for graph in graphs:
        by_relation[str(graph["Relation"])].append(graph)
    output = {}
    for relation, values in by_relation.items():
        ordered = sorted(values, key=lambda row: _key(row))
        if len(ordered) < 2:
            raise ContractError(f"cannot shift singleton relation: {relation}")
        for index, graph in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            output[_key(graph)] = _event_records(donor)
    return output


def _decode(
    graphs: Sequence[Mapping[str, Any]],
    incumbents: Mapping[tuple[str, str], Sequence[str]],
    arm: str,
    *,
    fail_closed_invalid_evidence: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shifted = (
        _shifted_records(graphs)
        if arm == "strict_proof_graph_cardinality_shifted" else {}
    )
    predictions, decisions = [], []
    for graph in graphs:
        key = _key(graph)
        if key not in incumbents:
            raise ContractError(f"missing incumbent: {key}")
        try:
            context = _row_context(
                graph, incumbents[key],
                cardinality_records=shifted.get(key),
            )
        except ContractError as exc:
            if not fail_closed_invalid_evidence:
                raise
            predictions.append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": list(incumbents[key]),
            })
            decisions.append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "arm": arm,
                "changed": False,
                "evidence_invalid_fallback": True,
                "fallback_reason": str(exc),
            })
            continue
        selected, decision = _select(context, arm)
        predictions.append(_prediction(
            graph, context["hypotheses"][selected]))
        decisions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "arm": arm,
            **decision,
        })
    return predictions, decisions


def _fold_audit(
    graphs: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assignment = subject_grouped_folds(graphs)
    output = []
    for fold in range(5):
        indices = [
            index for index, graph in enumerate(graphs)
            if assignment[_key(graph)] == fold
        ]
        fold_gold = [gold[_key(graphs[index])] for index in indices]
        selected_score = score(
            [predictions[index] for index in indices], fold_gold)[POOLED]
        control_score = score(
            [controls[index] for index in indices], fold_gold)[POOLED]
        output.append({
            "fold": fold,
            "rows": len(indices),
            "control": control_score,
            "selected": selected_score,
            "delta": selected_score - control_score,
        })
    return output


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source = Path(args.train_source).resolve()
    source_plan, graphs = validate_generation_source(source)
    validation_graph = Path(args.validation_graph).resolve()
    validation_baseline = Path(args.validation_baseline).resolve()
    implementation = Path(__file__).resolve()
    if len(graphs) != 477:
        raise ContractError("expected 477 train graph rows")
    if not validation_graph.is_file() or not validation_baseline.is_file():
        raise ContractError("validation decode inputs are absent")
    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "deployable": False,
        # The strict family-margin refinement was added after auditing the
        # first loose-rule development confirmation.  Preserve that history
        # explicitly; this is not a fresh blind confirmation.
        "validation_opened": True,
        "validation_labels_used": True,
        "validation_informed_refinement": True,
        "train_source": str(source),
        "train_source_plan": str(source / "plan/PLAN.json"),
        "train_source_plan_sha256": sha256(source / "plan/PLAN.json"),
        "train_source_graph": source_plan["source_graph"],
        "train_source_graph_sha256": source_plan["source_graph_sha256"],
        "validation_graph": str(validation_graph),
        "validation_graph_sha256": sha256(validation_graph),
        "validation_baseline": str(validation_baseline),
        "validation_baseline_sha256": sha256(validation_baseline),
        "implementation": str(implementation),
        "implementation_sha256": sha256(implementation),
        "arms": list(ARMS),
        "primary_arm": PRIMARY_ARM,
        "identity_relations": sorted(IDENTITY_RELATIONS),
        "proof_contract": {
            "minimum_exact_family_fraction": MIN_EXACT_FAMILY_FRACTION,
            "minimum_exact_family_advantage": MIN_EXACT_FAMILY_ADVANTAGE,
            "minimum_family_similarity": MIN_FAMILY_SIMILARITY,
            "cardinality_advantage": "strictly_positive",
            "existence_advantage": "nonnegative",
            "set_size_delta": "nonpositive",
        },
    }
    path = output / "plan/PLAN.json"
    _write_json(path, plan)
    print(json.dumps({
        "plan": str(path), "plan_sha256": sha256(path),
        "train_rows": len(graphs),
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _json(output / "plan/PLAN.json")
    required = (
        ("train_source_plan", "train_source_plan_sha256"),
        ("train_source_graph", "train_source_graph_sha256"),
        ("validation_graph", "validation_graph_sha256"),
        ("validation_baseline", "validation_baseline_sha256"),
        ("implementation", "implementation_sha256"),
    )
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("validation_opened") is not True
        or plan.get("validation_labels_used") is not True
        or plan.get("validation_informed_refinement") is not True
        or plan.get("arms") != list(ARMS)
        or plan.get("primary_arm") != PRIMARY_ARM
        or plan.get("identity_relations") != sorted(IDENTITY_RELATIONS)
        or any(not _artifact_matches(Path(plan[path]), plan[digest])
               for path, digest in required)
    ):
        raise ContractError("proof decoder plan contract failed")
    _, graphs = validate_generation_source(Path(plan["train_source"]))
    return plan, graphs


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan, graphs = _validate_plan(output)
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold = {_key(row): row for row in gold_rows}
    control_rows, _ = compose_competition_train_oof()
    raw_controls = {
        _key(row): list(row["ObjectEntities"]) for row in control_rows
    }
    # The frozen CoT40 train incumbent includes its relation-independent count
    # anchor.  Omitting this step silently evaluates an older 0.48245 control
    # instead of the 0.49869 architecture that produced the source graph.
    controls = {
        _key(graph): cot40_count_anchor(
            graph, raw_controls[_key(graph)])
        for graph in graphs
    }
    if set(controls) != set(gold) or {_key(row) for row in graphs} != set(gold):
        raise ContractError("train coverage mismatch")
    ordered_controls = [{
        "SubjectEntity": str(graph["SubjectEntity"]),
        "Relation": str(graph["Relation"]),
        "ObjectEntities": list(controls[_key(graph)]),
    } for graph in graphs]
    control_scores = score(ordered_controls, gold_rows)
    results = {}
    for arm in ARMS:
        predictions, decisions = _decode(graphs, controls, arm)
        scores = score(predictions, gold_rows)
        audit = _audit(predictions, controls, gold)
        relation_deltas = {
            relation: scores[relation] - control_scores[relation]
            for relation in RELATIONS
        }
        results[arm] = {
            "scores": scores,
            "delta": scores[POOLED] - control_scores[POOLED],
            "relation_deltas": relation_deltas,
            "audit": audit,
            "folds": _fold_audit(
                graphs, predictions, ordered_controls, gold),
            "changed": sum(value["changed"] for value in decisions),
        }
        write_jsonl_atomic(
            output / f"analysis/{arm}_OOF_PREDICTIONS.jsonl", predictions)
        write_jsonl_atomic(
            output / f"analysis/{arm}_DECISIONS.jsonl", decisions)

    primary = results[PRIMARY_ARM]
    audit = primary["audit"]
    checks = {
        "minimum_pooled_delta": primary["delta"] >= MIN_POOLED_DELTA,
        "positive_folds": sum(
            value["delta"] > EPSILON for value in primary["folds"]
        ) >= MIN_POSITIVE_FOLDS,
        "fold_floor": min(value["delta"] for value in primary["folds"])
            >= MAX_FOLD_REGRESSION,
        "relation_floor": min(primary["relation_deltas"].values())
            >= MAX_RELATION_REGRESSION,
        "help_harm_ratio": int(audit.get("helped", 0))
            >= MIN_HELP_HARM_RATIO * max(1, int(audit.get("harmed", 0))),
        "aligned_over_shifted": primary["delta"] - results[
            "strict_proof_graph_cardinality_shifted"]["delta"]
            >= MIN_ALIGNED_OVER_SHIFTED,
    }
    gate = all(checks.values())
    result = {
        "schema": RESULT_SCHEMA,
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "validation_opened": True,
        "validation_labels_used": True,
        "validation_informed_refinement": True,
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "control_scores": control_scores,
        "arms": results,
        "promotion_checks": checks,
        "promotion_gate_passed": gate,
        "next_stage": (
            "freeze_and_decode_validation" if gate
            else "reject_proof_carrying_graph_decoder"
        ),
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)
    lines = [
        "# Proof-carrying graph decoder — train-only audit", "",
        "The decoder traverses exact support edges into complete answer-set",
        "hypotheses and applies one relation-agnostic proof obligation.", "",
        f"Promotion gate passed: **{gate}**", "",
        "| arm | pooled F1 | delta | changed | helped / harmed |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = results[arm]
        arm_audit = value["audit"]
        lines.append(
            f"| {arm} | {value['scores'][POOLED]:.6f} | "
            f"{value['delta']:+.6f} | {value['changed']} | "
            f"{arm_audit.get('helped', 0)} / {arm_audit.get('harmed', 0)} |"
        )
    lines.extend(["", "## Primary relation deltas", ""])
    for relation in RELATIONS:
        lines.append(
            f"- {relation}: {primary['relation_deltas'][relation]:+.6f}")
    lines.extend(["", "## Primary fold deltas", ""])
    for value in primary["folds"]:
        lines.append(f"- fold {value['fold']}: {value['delta']:+.6f}")
    lines.extend(["", "## Gate checks", ""])
    for name, passed in checks.items():
        lines.append(f"- {name}: **{passed}**")
    (output / "analysis/RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "promotion_gate_passed": gate,
        "primary_delta": primary["delta"],
        "changed": primary["changed"],
        "helped": audit.get("helped", 0),
        "harmed": audit.get("harmed", 0),
        "result": str(result_path),
    }, indent=2, sort_keys=True))
    return 0


def decode_validation(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan, _ = _validate_plan(output)
    result_path = output / "analysis/RESULT.json"
    result = _json(result_path)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("plan_sha256") != sha256(output / "plan/PLAN.json")
        or result.get("promotion_gate_passed") is not True
        or result.get("validation_opened") is not True
        or result.get("validation_informed_refinement") is not True
    ):
        raise ContractError("train gate did not authorize validation decode")
    graphs = read_jsonl(Path(plan["validation_graph"]))
    baselines = read_jsonl(Path(plan["validation_baseline"]))
    incumbent = {_key(row): list(row["ObjectEntities"]) for row in baselines}
    if len(graphs) != 478 or len(incumbent) != 478:
        raise ContractError("validation coverage mismatch")
    predictions, decisions = _decode(
        graphs, incumbent, PRIMARY_ARM,
        fail_closed_invalid_evidence=True,
    )
    # Exact per-generation Qwen award artifacts were not preserved.  Preserve
    # the frozen SOTA output instead of treating an aggregate as an event.
    for index, graph in enumerate(graphs):
        if str(graph["Relation"]) in IDENTITY_RELATIONS:
            predictions[index] = {
                "SubjectEntity": str(graph["SubjectEntity"]),
                "Relation": str(graph["Relation"]),
                "ObjectEntities": list(incumbent[_key(graph)]),
            }
            decisions[index]["identity_fallback"] = True
            decisions[index]["changed"] = False
    prediction_path = output / "VALIDATION_PREDICTIONS.jsonl"
    decision_path = output / "VALIDATION_DECISIONS.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(decision_path, decisions)
    manifest = {
        "schema": PREDICTION_MANIFEST_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "deployable": False,
        "validation_labels_used": True,
        "validation_informed_refinement": True,
        "rows": len(predictions),
        "changed": sum(value.get("changed", False) for value in decisions),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "decisions": str(decision_path),
        "decisions_sha256": sha256(decision_path),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_result_sha256": sha256(result_path),
    }
    manifest_path = prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def evaluate_validation(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan, _ = _validate_plan(output)
    prediction_path = output / "VALIDATION_PREDICTIONS.jsonl"
    manifest_path = prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != PREDICTION_MANIFEST_SCHEMA
        or manifest.get("predictions_sha256") != sha256(prediction_path)
        or manifest.get("validation_labels_used") is not True
    ):
        raise ContractError("validation prediction manifest mismatch")
    predictions = read_jsonl(prediction_path)
    baselines = read_jsonl(Path(plan["validation_baseline"]))
    gold_path = Path(args.validation_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    scores = score(predictions, gold_rows)
    baseline_scores = score(baselines, gold_rows)
    gold = {_key(row): row for row in gold_rows}
    controls = {_key(row): list(row["ObjectEntities"]) for row in baselines}
    paired = _audit(predictions, controls, gold)
    result = {
        "schema": "proof-carrying-graph-validation-result-v1",
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "validation_opened": True,
        "validation_labels_used_for_policy_selection": True,
        "validation_informed_refinement": True,
        "validation_lineage_previously_selected": True,
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "validation_gold": str(gold_path),
        "validation_gold_sha256": sha256(gold_path),
        "baseline_scores": baseline_scores,
        "scores": scores,
        "deltas": {
            key: scores[key] - baseline_scores[key] for key in scores
        },
        "paired_audit": paired,
    }
    path = output / "validation/RESULT.json"
    _write_json(path, result)
    lines = [
        "# Proof-carrying graph decoder — validation confirmation", "",
        f"Baseline F1: **{baseline_scores[POOLED]:.6f}**",
        f"Proof-graph F1: **{scores[POOLED]:.6f}**",
        f"Delta: **{scores[POOLED] - baseline_scores[POOLED]:+.6f}**", "",
        "| relation | baseline | proof graph | delta |",
        "|---|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        lines.append(
            f"| {relation} | {baseline_scores[relation]:.6f} | "
            f"{scores[relation]:.6f} | "
            f"{scores[relation] - baseline_scores[relation]:+.6f} |")
    lines.extend(["", (
        "The strict two-family-margin rule is a development-informed refinement "
        "of a looser rule whose failure ledger was inspected on this validation "
        "split. It is therefore validation-tuned evidence, not a fresh blind-test "
        "confirmation; its real generalization test is the competition test."
    )])
    (output / "validation/RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "baseline": baseline_scores[POOLED],
        "selected": scores[POOLED],
        "delta": scores[POOLED] - baseline_scores[POOLED],
        "changed": paired.get("changed", 0),
        "helped": paired.get("helped", 0),
        "harmed": paired.get("harmed", 0),
        "result": str(path),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--train-source", default=str(DEFAULT_TRAIN_SOURCE))
    prepare_parser.add_argument("--validation-graph", default=str(DEFAULT_VALIDATION_GRAPH))
    prepare_parser.add_argument("--validation-baseline", default=str(DEFAULT_VALIDATION_BASELINE))
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.set_defaults(function=prepare)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    analyze_parser.add_argument("--train-gold", default=str(DEFAULT_TRAIN_GOLD))
    analyze_parser.set_defaults(function=analyze)
    decode_parser = sub.add_parser("decode-validation")
    decode_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    decode_parser.set_defaults(function=decode_validation)
    evaluate_parser = sub.add_parser("evaluate-validation")
    evaluate_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    evaluate_parser.add_argument("--validation-gold", default=str(DEFAULT_VALIDATION_GOLD))
    evaluate_parser.set_defaults(function=evaluate_validation)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
