#!/usr/bin/env python3
"""Audit exact model generations as coherent answer-set hypotheses.

The candidate-union oracle may splice arbitrary components from many noisy
generations.  This train-only experiment measures a stricter ceiling: choose
one complete exact generation (or KEEP) per row.  It then tests whether a
family-balanced medoid over Qwen, Gemma, and Ministral generation sets can
identify a useful coherent challenger.

No model is called, no validation path is accepted, and the learned arm is a
single row-level KEEP-versus-one-challenger ridge evaluated with nested,
subject-grouped out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluate import try_parse_number

from experiments.heterogeneous_agents.assemble_and_audit import (
    oracle_rows,
    score,
)
from experiments.heterogeneous_agents.baseline_relative_route_decoder import (
    ResidualRidge,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    stable_seed,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.cot40_evidence_edge_ablation import (
    DEFAULT_OUTPUT as DEFAULT_SOURCE,
    FAMILIES,
    _validate_plan as _validate_source,
)
from experiments.heterogeneous_agents.cot40_graph_native_decoder import (
    DEFAULT_GOLD,
    NUMERIC_RELATIONS,
    POOLED,
    RELATIONS,
    _component_ids,
    _objects_for_ids,
    cot40_count_anchor,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    _key,
)
from experiments.heterogeneous_agents.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.sota_pipeline import (
    compose_competition_train_oof,
)
from experiments.heterogeneous_agents.three_model_component_decoder import (
    subject_grouped_folds,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "generation_set_hypothesis_audit_20260801_v1"
)
PLAN_SCHEMA = "generation-set-hypothesis-plan-v1"
RESULT_SCHEMA = "generation-set-hypothesis-result-v1"
ARMS = (
    "exact_family_consensus",
    "family_mean_medoid",
    "family_minimum_medoid",
    "independent_family_medoid",
    "family_mean_medoid_shuffled",
    "nested_keep_gate",
)

# Fixed before the learned OOF arm is evaluated.
L2 = 10.0
GUARDS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
MIN_COHERENT_ORACLE_DELTA = 0.03
MIN_OOF_DELTA = 0.005
MIN_FOLD_WINS = 3
MIN_HELP_HARM_RATIO = 2.0
MAX_RELATION_REGRESSION = -0.01
MIN_ALIGNED_OVER_SHUFFLED = 0.02

STAT_NAMES = (
    "mean_similarity",
    "minimum_similarity",
    "exact_family_fraction",
    "within_family_exact_rate_mean",
)
FEATURE_NAMES = (
    *tuple(f"relation_{relation}" for relation in RELATIONS),
    *tuple(
        f"{scope}_{name}"
        for name in STAT_NAMES
        for scope in ("challenger", "incumbent", "delta")
    ),
    *tuple(f"family_similarity_delta_{family}" for family in FAMILIES),
    *tuple(f"family_exact_rate_delta_{family}" for family in FAMILIES),
    "incumbent_size",
    "challenger_size",
    "size_delta",
    "incumbent_challenger_similarity",
    "incumbent_empty",
    "challenger_empty",
    "opens_empty",
    "closes_nonempty",
)


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


def set_f1(left: Iterable[str], right: Iterable[str]) -> float:
    """Symmetric set F1 used only between graph component identities."""
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def _incumbent_tokens(
    graph: Mapping[str, Any], objects: Sequence[str],
) -> frozenset[str]:
    """Map KEEP to components, with a guarded numeric-tolerance fallback."""
    relation = str(graph["Relation"])
    tokens: list[str] = []
    for index, item in enumerate(objects):
        exact = _component_ids(graph, [str(item)])
        token: str | None = None
        if len(exact) == 1:
            token = next(iter(exact))
        elif relation in NUMERIC_RELATIONS:
            target = try_parse_number(str(item))
            candidates: list[tuple[float, str]] = []
            if target is not None and target > 0:
                for component in graph["relational_graph"]["components"]:
                    numbers = [
                        try_parse_number(str(value))
                        for value in (
                            *component.get("member_items", []),
                            component.get("representative", ""),
                        )
                    ]
                    distances = [
                        abs(value - target) / abs(target)
                        for value in numbers
                        if value is not None and value > 0
                    ]
                    if distances and min(distances) <= 0.05 + 1e-12:
                        candidates.append((
                            min(distances), str(component["id"])))
            candidates.sort()
            if candidates and (
                len(candidates) == 1
                or candidates[0][0] < candidates[1][0] - 1e-12
            ):
                token = candidates[0][1]
        if token is None:
            token = (
                f"incumbent:{index}:"
                f"{canonical_key(str(item), relation)}"
            )
        if token not in tokens:
            tokens.append(token)
    return frozenset(tokens)


def build_hypotheses(
    graph: Mapping[str, Any], incumbent: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, list[frozenset[str]]], dict[str, int]]:
    """Traverse exact event->component edges into unique complete sets."""
    relational = graph["relational_graph"]
    events = {
        str(node["id"]): node
        for node in relational["nodes"]
        if node.get("node_type") == "evidence_event"
    }
    support: dict[str, set[str]] = {event: set() for event in events}
    for edge in relational["edges"]:
        if edge.get("edge_type") != "supports":
            continue
        source, target = str(edge["source"]), str(edge["target"])
        if source not in support:
            raise ContractError(f"{_key(graph)}: orphan supports edge")
        support[source].add(target)

    incumbent_tokens = _incumbent_tokens(graph, incumbent)
    by_tokens: dict[frozenset[str], dict[str, Any]] = {
        incumbent_tokens: {
            "tokens": incumbent_tokens,
            "objects": list(incumbent),
            "is_incumbent": True,
            "proposer_families": set(),
            "event_ids": [],
        }
    }
    family_events: dict[str, list[frozenset[str]]] = {
        family: [] for family in FAMILIES
    }
    status_counts: Counter[str] = Counter()
    for event_id, node in sorted(events.items()):
        status = str(node.get("status"))
        status_counts[status] += 1
        if status not in ("candidate_set", "explicit_none"):
            continue
        family = str(node.get("model_family"))
        if family not in family_events:
            raise ContractError(f"{_key(graph)}: unknown family {family}")
        tokens = frozenset(support[event_id])
        if status == "candidate_set" and not tokens:
            raise ContractError(f"{_key(graph)}: empty candidate-set event")
        if status == "explicit_none" and tokens:
            raise ContractError(f"{_key(graph)}: supported explicit-none event")
        family_events[family].append(tokens)
        hypothesis = by_tokens.setdefault(tokens, {
            "tokens": tokens,
            "objects": _objects_for_ids(graph, tokens),
            "is_incumbent": False,
            "proposer_families": set(),
            "event_ids": [],
        })
        hypothesis["proposer_families"].add(family)
        hypothesis["event_ids"].append(event_id)

    if any(not values for values in family_events.values()):
        raise ContractError(f"{_key(graph)}: family has no parseable event")
    hypotheses = sorted(by_tokens.values(), key=lambda value: (
        not bool(value["is_incumbent"]),
        tuple(sorted(value["tokens"])),
    ))
    return hypotheses, family_events, dict(sorted(status_counts.items()))


def hypothesis_stats(
    hypothesis: Mapping[str, Any],
    family_events: Mapping[str, Sequence[frozenset[str]]],
) -> dict[str, Any]:
    tokens = frozenset(map(str, hypothesis["tokens"]))
    similarities = {
        family: max(
            (set_f1(tokens, event) for event in family_events[family]),
            default=0.0,
        )
        for family in FAMILIES
    }
    exact_rates = {
        family: (
            sum(event == tokens for event in family_events[family])
            / len(family_events[family])
        )
        for family in FAMILIES
    }
    proposers = set(map(str, hypothesis["proposer_families"]))
    reviewers = [family for family in FAMILIES if family not in proposers]
    independent = (
        1.0 if len(proposers) == len(FAMILIES)
        else statistics.mean(similarities[family] for family in reviewers)
        if reviewers else 0.5
    )
    return {
        "mean_similarity": statistics.mean(similarities.values()),
        "minimum_similarity": min(similarities.values()),
        "exact_family_fraction": len(proposers) / len(FAMILIES),
        "within_family_exact_rate_mean": statistics.mean(exact_rates.values()),
        "independent_similarity": independent,
        "family_similarities": similarities,
        "family_exact_rates": exact_rates,
    }


def _selection_key(
    hypothesis: Mapping[str, Any], stats: Mapping[str, Any], arm: str,
    override: float | None = None,
) -> tuple[Any, ...]:
    primary = (
        float(override) if override is not None else
        float(stats["exact_family_fraction"])
        if arm == "exact_family_consensus" else
        float(stats["minimum_similarity"])
        if arm == "family_minimum_medoid" else
        float(stats["independent_similarity"])
        if arm == "independent_family_medoid" else
        float(stats["mean_similarity"])
    )
    return (
        primary,
        float(stats["exact_family_fraction"]),
        float(stats["mean_similarity"]),
        bool(hypothesis["is_incumbent"]),
        -len(hypothesis["tokens"]),
        tuple(sorted(hypothesis["tokens"])),
    )


def select_hypothesis(
    hypotheses: Sequence[Mapping[str, Any]],
    stats: Sequence[Mapping[str, Any]],
    arm: str,
    overrides: Sequence[float] | None = None,
) -> int:
    if not hypotheses or len(hypotheses) != len(stats):
        raise ContractError("invalid coherent hypothesis menu")
    if overrides is not None and len(overrides) != len(hypotheses):
        raise ContractError("invalid coherent score override")
    return max(range(len(hypotheses)), key=lambda index: _selection_key(
        hypotheses[index],
        stats[index],
        arm,
        None if overrides is None else float(overrides[index]),
    ))


def shuffled_values(
    values: Sequence[float], *, subject: str, relation: str,
) -> list[float]:
    result = list(map(float, values))
    if len(result) < 2:
        return result
    shift = 1 + stable_seed(
        "generation-set-medoid-shuffle", subject, relation,
    ) % (len(result) - 1)
    return result[shift:] + result[:shift]


def challenger_features(
    relation: str,
    incumbent: Mapping[str, Any],
    incumbent_stats: Mapping[str, Any],
    challenger: Mapping[str, Any],
    challenger_stats: Mapping[str, Any],
) -> list[float]:
    values: list[float] = [
        *[float(relation == value) for value in RELATIONS],
    ]
    for name in STAT_NAMES:
        challenger_value = float(challenger_stats[name])
        incumbent_value = float(incumbent_stats[name])
        values.extend((
            challenger_value,
            incumbent_value,
            challenger_value - incumbent_value,
        ))
    values.extend(
        float(challenger_stats["family_similarities"][family])
        - float(incumbent_stats["family_similarities"][family])
        for family in FAMILIES
    )
    values.extend(
        float(challenger_stats["family_exact_rates"][family])
        - float(incumbent_stats["family_exact_rates"][family])
        for family in FAMILIES
    )
    incumbent_tokens = set(map(str, incumbent["tokens"]))
    challenger_tokens = set(map(str, challenger["tokens"]))
    values.extend((
        min(1.0, len(incumbent_tokens) / 10.0),
        min(1.0, len(challenger_tokens) / 10.0),
        max(-1.0, min(
            1.0, (len(challenger_tokens) - len(incumbent_tokens)) / 10.0)),
        set_f1(incumbent_tokens, challenger_tokens),
        float(not incumbent_tokens),
        float(not challenger_tokens),
        float(not incumbent_tokens and bool(challenger_tokens)),
        float(bool(incumbent_tokens) and not challenger_tokens),
    ))
    if len(values) != len(FEATURE_NAMES) or not all(map(math.isfinite, values)):
        raise ContractError("invalid generation-set challenger features")
    return values


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, Any]]:
    return [{
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
        "ObjectEntities": list(selected[_key(row)]),
    } for row in rows]


def _audit(
    predictions: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    for row in predictions:
        key = _key(row)
        before = list(controls[key])
        after = list(row["ObjectEntities"])
        if tuple(before) == tuple(after):
            continue
        old = _row_f1(before, gold[key], key[1])
        new = _row_f1(after, gold[key], key[1])
        outcome = (
            "helped" if new > old + 1e-12 else
            "harmed" if new < old - 1e-12 else "neutral"
        )
        counts.update(("changed", outcome))
        by_relation[key[1]].update(("changed", outcome))
    return {
        **dict(counts),
        "by_relation": {
            relation: dict(by_relation[relation]) for relation in RELATIONS
        },
    }


def _fit_model(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int],
) -> ResidualRidge:
    return ResidualRidge(FEATURE_NAMES, L2).fit(
        [records[index]["features"] for index in indices],
        [records[index]["target_delta"] for index in indices],
        [1.0 for _ in indices],
    )


def nested_keep_gate(
    records: Sequence[Mapping[str, Any]],
    graphs: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) != len(graphs):
        raise ContractError("generation-set record coverage mismatch")
    gold = {_key(row): row for row in gold_rows}
    outer_assignment = subject_grouped_folds(graphs)
    selected: dict[tuple[str, str], list[str]] = {}
    fold_diagnostics = []
    for outer in range(5):
        outer_fit = [
            index for index, graph in enumerate(graphs)
            if outer_assignment[_key(graph)] != outer
        ]
        outer_hold = [
            index for index, graph in enumerate(graphs)
            if outer_assignment[_key(graph)] == outer
        ]
        inner_graphs = [graphs[index] for index in outer_fit]
        inner_assignment = subject_grouped_folds(inner_graphs, n_folds=4)
        inner_scores: dict[int, float] = {}
        for inner in range(4):
            fit_indices = [
                index for index in outer_fit
                if inner_assignment[_key(graphs[index])] != inner
            ]
            hold_indices = [
                index for index in outer_fit
                if inner_assignment[_key(graphs[index])] == inner
            ]
            model = _fit_model(records, fit_indices)
            estimates = model.predict([
                records[index]["features"] for index in hold_indices
            ])
            inner_scores.update({
                index: float(value)
                for index, value in zip(hold_indices, estimates)
            })
        if set(inner_scores) != set(outer_fit):
            raise ContractError("nested generation-set inner coverage mismatch")
        candidates = []
        for guard in GUARDS:
            predictions = []
            for index in outer_fit:
                record = records[index]
                objects = (
                    record["challenger_objects"]
                    if inner_scores[index] > guard
                    else record["incumbent_objects"]
                )
                predictions.append({
                    "SubjectEntity": record["SubjectEntity"],
                    "Relation": record["Relation"],
                    "ObjectEntities": list(objects),
                })
            fit_gold = [gold[_key(row)] for row in predictions]
            candidates.append((score(predictions, fit_gold)[POOLED], guard))
        _, guard = max(candidates, key=lambda value: (value[0], value[1]))
        model = _fit_model(records, outer_fit)
        estimates = model.predict([
            records[index]["features"] for index in outer_hold
        ])
        hold_predictions = []
        for index, estimate in zip(outer_hold, estimates):
            record = records[index]
            objects = (
                record["challenger_objects"]
                if float(estimate) > guard
                else record["incumbent_objects"]
            )
            selected[_key(record)] = list(objects)
            hold_predictions.append({
                "SubjectEntity": record["SubjectEntity"],
                "Relation": record["Relation"],
                "ObjectEntities": list(objects),
            })
        hold_controls = [{
            "SubjectEntity": records[index]["SubjectEntity"],
            "Relation": records[index]["Relation"],
            "ObjectEntities": list(records[index]["incumbent_objects"]),
        } for index in outer_hold]
        hold_gold = [gold[_key(row)] for row in hold_predictions]
        fold_diagnostics.append({
            "outer_fold": outer,
            "selected_guard": guard,
            "control_score": score(hold_controls, hold_gold)[POOLED],
            "selected_score": score(hold_predictions, hold_gold)[POOLED],
            "rows": len(outer_hold),
        })
    if len(selected) != len(records):
        raise ContractError("nested generation-set outer coverage mismatch")
    return _prediction_rows(records, selected), fold_diagnostics


def prepare(args: argparse.Namespace) -> int:
    source = Path(args.source_run).resolve()
    output = Path(args.output_dir).resolve()
    source_plan, graphs = _validate_source(source)
    if len(graphs) != 477:
        raise ContractError("generation-set source coverage mismatch")
    source_result = source / "analysis/RESULT.json"
    artifacts = (
        source / "plan/PLAN.json",
        Path(source_plan["typed_graph"]),
        Path(source_plan["typed_manifest"]),
        source_result,
    )
    if not all(path.exists() for path in artifacts):
        raise ContractError("generation-set source artifact missing")
    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "source_run": str(source),
        "source_plan": str(artifacts[0]),
        "source_plan_sha256": sha256(artifacts[0]),
        "source_graph": str(artifacts[1]),
        "source_graph_sha256": sha256(artifacts[1]),
        "source_manifest": str(artifacts[2]),
        "source_manifest_sha256": sha256(artifacts[2]),
        "source_result": str(artifacts[3]),
        "source_result_sha256": sha256(artifacts[3]),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "arms": list(ARMS),
        "families": list(FAMILIES),
        "feature_names": list(FEATURE_NAMES),
        "l2": L2,
        "guards": list(GUARDS),
        "folding": "nested_subject_grouped_5x4",
        "coherent_oracle_minimum_delta": MIN_COHERENT_ORACLE_DELTA,
        "oof_minimum_delta": MIN_OOF_DELTA,
    }
    path = output / "plan/PLAN.json"
    _write_json(path, plan)
    print(json.dumps({
        "plan": str(path),
        "plan_sha256": sha256(path),
        "source_rows": len(graphs),
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _json(output / "plan/PLAN.json")
    required = (
        ("source_plan", "source_plan_sha256"),
        ("source_graph", "source_graph_sha256"),
        ("source_manifest", "source_manifest_sha256"),
        ("source_result", "source_result_sha256"),
        ("implementation", "implementation_sha256"),
    )
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("validation_opened") is not False
        or plan.get("arms") != list(ARMS)
        or plan.get("families") != list(FAMILIES)
        or plan.get("feature_names") != list(FEATURE_NAMES)
        or not artifact_contract_matches(plan, required)
    ):
        raise ContractError("generation-set plan contract failed")
    _, graphs = _validate_source(Path(plan["source_run"]))
    return plan, graphs


def artifact_contract_matches(
    plan: Mapping[str, Any],
    required: Sequence[tuple[str, str]],
) -> bool:
    """Verify path/digest fields without confusing field names for paths."""
    try:
        return all(
            sha256(Path(str(plan[path_field]))) == plan[digest_field]
            for path_field, digest_field in required
        )
    except (KeyError, OSError, TypeError):
        return False


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan, graphs = _validate_plan(output)
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold = {_key(row): row for row in gold_rows}
    base_rows, _ = compose_competition_train_oof()
    base = {_key(row): row for row in base_rows}
    if not (len(gold) == len(graphs) == 477 and set(gold) == set(base)):
        raise ContractError("generation-set train coverage mismatch")

    controls: dict[tuple[str, str], list[str]] = {}
    predictions: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARMS if arm != "nested_keep_gate"
    }
    event_oracle_rows = []
    family_oracle_rows = {family: [] for family in FAMILIES}
    records = []
    inventory = Counter()
    oracle_winners = Counter()
    for graph in graphs:
        key = _key(graph)
        incumbent = cot40_count_anchor(
            graph, list(base[key].get("ObjectEntities", [])))
        controls[key] = list(incumbent)
        hypotheses, family_events, statuses = build_hypotheses(
            graph, incumbent)
        stats = [
            hypothesis_stats(hypothesis, family_events)
            for hypothesis in hypotheses
        ]
        incumbent_index = next(
            index for index, hypothesis in enumerate(hypotheses)
            if hypothesis["is_incumbent"]
        )
        challenger_indices = [
            index for index in range(len(hypotheses))
            if index != incumbent_index
        ]
        challenger_index = (
            select_hypothesis(
                [hypotheses[index] for index in challenger_indices],
                [stats[index] for index in challenger_indices],
                "family_mean_medoid",
            )
            if challenger_indices else 0
        )
        challenger_index = (
            challenger_indices[challenger_index]
            if challenger_indices else incumbent_index
        )

        for arm in (
            "exact_family_consensus",
            "family_mean_medoid",
            "family_minimum_medoid",
            "independent_family_medoid",
        ):
            selected_index = select_hypothesis(hypotheses, stats, arm)
            predictions[arm].append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": list(hypotheses[selected_index]["objects"]),
            })
        values = [float(value["mean_similarity"]) for value in stats]
        shifted = shuffled_values(values, subject=key[0], relation=key[1])
        shuffled_index = select_hypothesis(
            hypotheses, stats, "family_mean_medoid", shifted)
        predictions["family_mean_medoid_shuffled"].append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": list(hypotheses[shuffled_index]["objects"]),
        })

        event_best = max(range(len(hypotheses)), key=lambda index: (
            _row_f1(hypotheses[index]["objects"], gold[key], key[1]),
            bool(hypotheses[index]["is_incumbent"]),
        ))
        event_oracle_rows.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": list(hypotheses[event_best]["objects"]),
        })
        winner_families = hypotheses[event_best]["proposer_families"]
        oracle_winners.update(winner_families or ("incumbent",))
        for family in FAMILIES:
            available = [
                index for index, hypothesis in enumerate(hypotheses)
                if hypothesis["is_incumbent"]
                or family in hypothesis["proposer_families"]
            ]
            best = max(available, key=lambda index: (
                _row_f1(hypotheses[index]["objects"], gold[key], key[1]),
                bool(hypotheses[index]["is_incumbent"]),
            ))
            family_oracle_rows[family].append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": list(hypotheses[best]["objects"]),
            })

        incumbent_hypothesis = hypotheses[incumbent_index]
        challenger_hypothesis = hypotheses[challenger_index]
        target_delta = (
            _row_f1(challenger_hypothesis["objects"], gold[key], key[1])
            - _row_f1(incumbent, gold[key], key[1])
        )
        records.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "incumbent_objects": list(incumbent),
            "challenger_objects": list(challenger_hypothesis["objects"]),
            "features": challenger_features(
                key[1],
                incumbent_hypothesis,
                stats[incumbent_index],
                challenger_hypothesis,
                stats[challenger_index],
            ),
            "target_delta": target_delta,
        })
        inventory.update({
            "rows": 1,
            "events": sum(len(values) for values in family_events.values()),
            "unique_hypotheses": len(hypotheses),
            "rows_with_challenger": int(challenger_index != incumbent_index),
            **{f"event_status_{name}": count
               for name, count in statuses.items()},
        })

    control_rows = _prediction_rows(graphs, controls)
    keep_medoid_oracle_rows = [{
        "SubjectEntity": record["SubjectEntity"],
        "Relation": record["Relation"],
        "ObjectEntities": list(
            record["challenger_objects"]
            if record["target_delta"] > 0.0
            else record["incumbent_objects"]
        ),
    } for record in records]
    challenger_outcomes: Counter[str] = Counter()
    challenger_outcomes_by_relation: dict[str, Counter[str]] = defaultdict(
        Counter)
    for record in records:
        outcome = (
            "identical" if tuple(record["challenger_objects"])
                == tuple(record["incumbent_objects"]) else
            "better" if record["target_delta"] > 1e-12 else
            "worse" if record["target_delta"] < -1e-12 else
            "tie_different"
        )
        challenger_outcomes.update((outcome,))
        challenger_outcomes_by_relation[record["Relation"]].update((outcome,))
    nested_rows, fold_diagnostics = nested_keep_gate(
        records, graphs, gold_rows)
    predictions["nested_keep_gate"] = nested_rows
    candidate_oracle_rows = oracle_rows(graphs, gold_rows)
    control_scores = score(control_rows, gold_rows)
    arm_results = {
        arm: {
            "scores": score(rows, gold_rows),
            "audit": _audit(rows, controls, gold),
        }
        for arm, rows in predictions.items()
    }
    event_scores = score(event_oracle_rows, gold_rows)
    keep_medoid_oracle_scores = score(keep_medoid_oracle_rows, gold_rows)
    candidate_scores = score(candidate_oracle_rows, gold_rows)
    family_oracles = {
        family: score(rows, gold_rows)
        for family, rows in family_oracle_rows.items()
    }
    nested = arm_results["nested_keep_gate"]
    fold_wins = sum(
        value["selected_score"] > value["control_score"] + 1e-12
        for value in fold_diagnostics
    )
    relation_deltas = {
        relation: nested["scores"][relation] - control_scores[relation]
        for relation in RELATIONS
    }
    nested_audit = nested["audit"]
    harm = int(nested_audit.get("harmed", 0))
    help_count = int(nested_audit.get("helped", 0))
    aligned = arm_results["family_mean_medoid"]["scores"][POOLED]
    shuffled = arm_results[
        "family_mean_medoid_shuffled"]["scores"][POOLED]
    coherent_gate = (
        event_scores[POOLED] - control_scores[POOLED]
        >= MIN_COHERENT_ORACLE_DELTA
    )
    selector_checks = {
        "minimum_oof_delta": (
            nested["scores"][POOLED] - control_scores[POOLED]
            >= MIN_OOF_DELTA),
        "minimum_fold_wins": fold_wins >= MIN_FOLD_WINS,
        "minimum_help_harm_ratio": (
            help_count >= MIN_HELP_HARM_RATIO * max(1, harm)),
        "relation_floor": min(relation_deltas.values())
            >= MAX_RELATION_REGRESSION,
        "aligned_beats_shuffled": aligned - shuffled
            >= MIN_ALIGNED_OVER_SHUFFLED,
    }
    selector_gate = coherent_gate and all(selector_checks.values())
    result = {
        "schema": RESULT_SCHEMA,
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "inventory": dict(sorted(inventory.items())),
        "control_scores": control_scores,
        "event_set_oracle_scores": event_scores,
        "keep_vs_medoid_oracle_scores": keep_medoid_oracle_scores,
        "candidate_union_oracle_scores": candidate_scores,
        "family_event_oracle_scores": family_oracles,
        "event_oracle_winner_families": dict(sorted(oracle_winners.items())),
        "coherent_oracle_delta": event_scores[POOLED] - control_scores[POOLED],
        "keep_vs_medoid_oracle_delta": (
            keep_medoid_oracle_scores[POOLED] - control_scores[POOLED]
        ),
        "candidate_oracle_delta": candidate_scores[POOLED] - control_scores[POOLED],
        "coherent_fraction_of_candidate_headroom": (
            (event_scores[POOLED] - control_scores[POOLED])
            / (candidate_scores[POOLED] - control_scores[POOLED])
        ),
        "arms": arm_results,
        "nested_fold_diagnostics": fold_diagnostics,
        "nested_fold_wins": fold_wins,
        "nested_relation_deltas": relation_deltas,
        "challenger_outcomes": dict(sorted(challenger_outcomes.items())),
        "challenger_outcomes_by_relation": {
            relation: dict(sorted(challenger_outcomes_by_relation[relation].items()))
            for relation in RELATIONS
        },
        "coherent_headroom_gate_passed": coherent_gate,
        "selector_gate_checks": selector_checks,
        "selector_gate_passed": selector_gate,
        "next_stage": (
            "freeze_for_validation_confirmation" if selector_gate else
            "retain_coherent_set_target_but_reject_current_medoid_selector"
            if coherent_gate else
            "stop_graph_set_selection_and_improve_generation"
        ),
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)
    write_jsonl_atomic(output / "analysis/NESTED_OOF_PREDICTIONS.jsonl", nested_rows)
    lines = [
        "# Complete-generation answer-set hypothesis audit",
        "",
        "Train only; validation was not opened. One exact generation is one",
        "coherent answer-set hypothesis. The candidate-union oracle remains",
        "gold-aware and nondeployable.",
        "",
        "| policy | pooled F1 | delta vs incumbent | changed | help / harm |",
        "|---|---:|---:|---:|---:|",
        f"| CoT40 incumbent | {control_scores[POOLED]:.6f} | -- | -- | -- |",
    ]
    for arm in ARMS:
        report = arm_results[arm]
        audit = report["audit"]
        lines.append(
            f"| {arm} | {report['scores'][POOLED]:.6f} | "
            f"{report['scores'][POOLED] - control_scores[POOLED]:+.6f} | "
            f"{audit.get('changed', 0)} | "
            f"{audit.get('helped', 0)} / {audit.get('harmed', 0)} |"
        )
    lines.extend((
        f"| KEEP-vs-medoid oracle | {keep_medoid_oracle_scores[POOLED]:.6f} | "
        f"{keep_medoid_oracle_scores[POOLED] - control_scores[POOLED]:+.6f} | "
        f"-- | -- |",
        f"| whole-generation oracle | {event_scores[POOLED]:.6f} | "
        f"{event_scores[POOLED] - control_scores[POOLED]:+.6f} | -- | -- |",
        f"| arbitrary candidate-union oracle | {candidate_scores[POOLED]:.6f} | "
        f"{candidate_scores[POOLED] - control_scores[POOLED]:+.6f} | -- | -- |",
        "",
        f"- Coherent fraction of candidate headroom: "
        f"**{result['coherent_fraction_of_candidate_headroom']:.3%}**",
        f"- Nested fold wins: **{fold_wins}/5**",
        f"- Medoid opportunities (better / worse / tie-different): "
        f"**{challenger_outcomes['better']} / {challenger_outcomes['worse']} / "
        f"{challenger_outcomes['tie_different']}**",
        f"- Coherent-headroom gate: **{coherent_gate}**",
        f"- Selector gate: **{selector_gate}**",
        f"- Next stage: `{result['next_stage']}`",
    ))
    (output / "analysis/RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "control": control_scores[POOLED],
        "whole_generation_oracle": event_scores[POOLED],
        "candidate_union_oracle": candidate_scores[POOLED],
        "coherent_fraction": result["coherent_fraction_of_candidate_headroom"],
        "family_mean_medoid": aligned,
        "nested_keep_gate": nested["scores"][POOLED],
        "selector_gate_passed": selector_gate,
        "result": str(output / "analysis/RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    analyze_parser.add_argument("--train-gold", type=Path, default=DEFAULT_GOLD)
    return value


def main() -> int:
    args = parser().parse_args()
    return prepare(args) if args.command == "prepare" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
