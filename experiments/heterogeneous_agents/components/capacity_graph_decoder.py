#!/usr/bin/env python3
"""Train-only capacity selection over the current three-family evidence graph.

The historical numeric decoder was trained on an older Qwen/Gemma graph and
correctly disabled ``hasCapacity`` after an unstable outer-fold result.  This
module gives capacity a fair, current-pipeline decoder without changing model
generation or any other relation:

* numeric candidate components are grouped with the evaluator's 5% tolerance;
* exact generation-event support is aggregated separately for Qwen, Gemma,
  and Ministral;
* a small option-correctness calibrator ranks the incumbent and graph options;
* L2 strength and the incumbent guard are selected inside each outer training
  fold; and
* validation predictions are frozen before validation labels are opened.

The deployment gate is deliberately fail-closed.  If nested train OOF does not
show a material, fold-stable gain, the emitted gated prediction is byte-wise
equivalent in answers to its incumbent.  No test labels or Codabench feedback
are consumed by this module.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate import RELATION_TYPE, true_positives, try_parse_number
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    proposal_parse_status,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    LogisticCalibrator,
    _key,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    compose_competition_train_oof,
)
from experiments.heterogeneous_agents.components.three_model_component_decoder import (
    subject_grouped_folds,
)


ROOT = Path(__file__).resolve().parents[3]
RUNS = Path(__file__).resolve().parents[1] / "runs"
DEFAULT_TRAIN_GRAPH = (
    RUNS / "minimal_structural_commitment_dev_20260804_v1/base/train/"
    "graph/FINAL_EXACT_EVIDENCE_GRAPH.jsonl"
)
DEFAULT_REPAIRED_COT_RESPONSE = (
    RUNS / "capacity_prompt_alignment_train_20260809_v1/responses/"
    "ministral__cot5_cap40_n10.jsonl"
)
DEFAULT_VALIDATION_GRAPH = (
    RUNS / "cot40_cardinality_validation_confirmation_20260730_v1/"
    "graph/VALIDATION_GRAPH.jsonl"
)
DEFAULT_VALIDATION_INCUMBENT = (
    ROOT / "results/heterogeneous/candidates/frozen_20260803/"
    "strict_proof_0_520729_validation.jsonl"
)
DEFAULT_OUTPUT = RUNS / "capacity_graph_decoder_20260809_v1"
DEFAULT_TRAIN_GOLD = ROOT / "data/train.jsonl"
DEFAULT_VALIDATION_GOLD = ROOT / "data/val.jsonl"

RELATION = "hasCapacity"
FAMILIES = (
    "qwen_recall",
    "gemma_independent",
    "ministral_independent",
)
FAMILY_PREFIX = {
    "qwen_recall": "qwen",
    "gemma_independent": "gemma",
    "ministral_independent": "ministral",
}
L2_VALUES = (0.5, 1.0, 2.0, 4.0, 8.0)
GUARD_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
TOLERANCE = 0.05
QWEN_ROUTE = "qwen:self_consistency"
GEMMA_ROUTE = "gemma:independent"
MINISTRAL_ROUTE = "ministral:cot5_cap40_n10"

# Predeclared train-only promotion contract.  Three additional correct rows
# out of 100 is material for the weakest relation, while the fold and paired
# guards prevent a single lucky partition from enabling the decoder.
MIN_OOF_DELTA = 0.03
MIN_POSITIVE_FOLDS = 3
MAX_FOLD_REGRESSION = -0.05
MIN_HELP_HARM_RATIO = 1.5

FEATURE_NAMES = (
    "intercept",
    "is_incumbent",
    "family_fraction",
    "three_family",
    "qwen_rate",
    "gemma_rate",
    "ministral_rate",
    "mean_family_rate",
    "minimum_family_rate",
    "maximum_family_rate",
    "total_event_rate",
    "family_fraction_advantage",
    "mean_rate_advantage",
    "minimum_rate_advantage",
    "component_fraction",
    "log_distance_from_incumbent",
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


def _gold_aliases(gold: Mapping[str, Any]) -> list[list[str]]:
    values = gold.get("ObjectEntities", [])
    if values and isinstance(values[0], str):
        return [[str(value)] for value in values]
    return [[str(alias) for alias in value] for value in values]


def _correct(objects: Sequence[str], gold: Mapping[str, Any]) -> float:
    return float(true_positives(
        list(map(str, objects)),
        _gold_aliases(gold),
        RELATION_TYPE[RELATION],
        TOLERANCE,
    ) > 0)


def _near(left: float, right: float) -> bool:
    if left <= 0 or right <= 0:
        return False
    return abs(left - right) / max(abs(right), 1e-12) <= TOLERANCE


def _format(value: float) -> str:
    return format(float(value), ".12g")


def _validate_graphs(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise ContractError(f"{path}: expected {expected_rows} rows")
    if any(bool(row.get("contains_labels")) or bool(row.get("gold_aware"))
           for row in rows):
        raise ContractError(f"graph contains labels: {path}")
    keys = [_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ContractError(f"duplicate graph keys: {path}")
    return rows


def _response_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if manifest.get("output_sha256") != sha256(path):
        raise ContractError(f"repaired CoT response hash mismatch: {path}")
    return manifest


def _numeric_objects(text: str) -> list[str]:
    status, items = proposal_parse_status(text, RELATION)
    if status != "parsed_nonempty":
        return []
    values = []
    for item in items:
        value = try_parse_number(str(item))
        if value is not None and math.isfinite(value) and value > 0:
            values.append(_format(float(value)))
    return list(dict.fromkeys(values))


def _old_event_objects(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read only the unchanged Qwen/Gemma exact events from production.

    The repaired experiment must not silently regenerate either family or
    replace the production incumbent.  The existing exact graph is therefore
    the authoritative source for these two routes.
    """
    relational = row["relational_graph"]
    components = {
        str(component["id"]): component
        for component in relational.get("components", [])
    }
    supports: dict[str, list[str]] = defaultdict(list)
    for edge in relational.get("edges", []):
        if str(edge.get("edge_type")) != "supports":
            continue
        target = str(edge["target"])
        if target in components:
            representative = str(components[target].get("representative", ""))
            value = try_parse_number(representative)
            if value is not None and math.isfinite(value) and value > 0:
                supports[str(edge["source"])].append(_format(float(value)))
    records = []
    allowed = {QWEN_ROUTE, GEMMA_ROUTE}
    for node in relational.get("nodes", []):
        if (
            node.get("node_type") != "evidence_event"
            or str(node.get("route")) not in allowed
        ):
            continue
        records.append({
            "route": str(node["route"]),
            "family": str(node["model_family"]),
            "generation_index": int(node.get("generation_index", 0) or 0),
            "objects": list(dict.fromkeys(supports.get(str(node["id"]), []))),
            "status": str(node.get("status", "unparsed_or_no_candidate")),
        })
    families = {record["family"] for record in records}
    if not {"qwen_recall", "gemma_independent"} <= families:
        raise ContractError(f"{_key(row)}: production graph lacks Qwen/Gemma events")
    return records


def build_repaired_train_graph(
    source_path: Path,
    repaired_cot_path: Path,
    output_path: Path,
    *,
    expected_source_rows: int | None = 477,
    expected_capacity_rows: int | None = 100,
) -> dict[str, Any]:
    """Build the isolated three-family capacity graph, without labels.

    Only Ministral's CoT event stream is replaced.  Production Qwen/Gemma
    events and the production baseline are copied exactly.  Components are
    scalar numeric surfaces; the decoder performs deterministic complete-link
    5% grouping, so no transitive equivalence is baked into this artifact.
    """
    source_rows = read_jsonl(source_path)
    if expected_source_rows is not None and len(source_rows) != expected_source_rows:
        raise ContractError(
            f"{source_path}: expected {expected_source_rows} rows, "
            f"got {len(source_rows)}")
    if any(bool(row.get("contains_labels")) or bool(row.get("gold_aware"))
           for row in source_rows):
        raise ContractError(f"graph contains labels: {source_path}")
    source_keys = [_key(row) for row in source_rows]
    if len(set(source_keys)) != len(source_keys):
        raise ContractError(f"duplicate graph keys: {source_path}")
    source = {
        _key(row): row for row in source_rows if str(row["Relation"]) == RELATION
    }
    if expected_capacity_rows is not None and len(source) != expected_capacity_rows:
        raise ContractError(
            f"production source must contain {expected_capacity_rows} "
            f"capacity rows, got {len(source)}")
    if not source:
        raise ContractError("production source has no capacity rows")
    response_manifest = _response_manifest(repaired_cot_path)
    responses = {}
    for row in read_jsonl(repaired_cot_path):
        key = (str(row.get("subject")), str(row.get("relation")))
        if key[1] != RELATION or str(row.get("phase")) != "propose":
            continue
        if key in responses:
            raise ContractError(f"duplicate repaired CoT row: {key}")
        generations = list(map(str, row.get("generations", [])))
        if len(generations) != 10:
            raise ContractError(f"{key}: expected ten repaired CoT generations")
        responses[key] = generations
    if set(responses) != set(source):
        raise ContractError("repaired CoT/source capacity coverage mismatch")

    output_rows = []
    route_counts: Counter[str] = Counter()
    for key in sorted(source):
        old = source[key]
        records = _old_event_objects(old)
        for index, text in enumerate(responses[key]):
            objects = _numeric_objects(text)
            parse_status, _ = proposal_parse_status(text, RELATION)
            records.append({
                "route": MINISTRAL_ROUTE,
                "family": "ministral_independent",
                "generation_index": index,
                "objects": objects,
                "status": (
                    "candidate_set" if objects
                    else "explicit_none" if parse_status == "explicit_none"
                    else "unparsed_or_no_candidate"
                ),
            })

        values = sorted({
            value for record in records for value in record["objects"]
        }, key=lambda value: (float(value), value))
        component_id = {
            value: f"component:{index}" for index, value in enumerate(values)
        }
        components = [{
            "id": component_id[value],
            "node_type": "candidate_component",
            "representative": value,
            "member_items": [value],
        } for value in values]
        nodes: list[dict[str, Any]] = list(components)
        edges: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            event_id = f"evidence:{record['route']}:generation:{record['generation_index']}"
            objects = list(record["objects"])
            status = "candidate_set" if objects else str(record["status"])
            nodes.append({
                "id": event_id,
                "node_type": "evidence_event",
                "evidence_kind": "exact_generation",
                "route": record["route"],
                "model_family": record["family"],
                "generation_index": record["generation_index"],
                "status": status,
            })
            route_counts[str(record["route"])] += 1
            for value in objects:
                edges.append({
                    "source": event_id,
                    "target": component_id[value],
                    "edge_type": "supports",
                    "evidence_kind": "exact_generation",
                })
        output_rows.append({
            "schema": "capacity-isolated-three-family-evidence-row-v1",
            "SubjectEntity": key[0],
            "Relation": key[1],
            "baseline_objects": list(map(str, old.get("baseline_objects", []))),
            "contains_labels": False,
            "gold_aware": False,
            "relational_graph": {
                "schema": "capacity-isolated-three-family-evidence-graph-v1",
                "components": components,
                "nodes": nodes,
                "edges": edges,
            },
        })
    write_jsonl_atomic(output_path, output_rows)
    manifest = {
        "schema": "capacity-isolated-three-family-evidence-manifest-v1",
        "rows": len(output_rows),
        "contains_labels": False,
        "gold_aware": False,
        "production_source": str(source_path),
        "production_source_sha256": sha256(source_path),
        "repaired_cot_response": str(repaired_cot_path),
        "repaired_cot_response_sha256": sha256(repaired_cot_path),
        "repaired_cot_manifest_schema": response_manifest.get("schema"),
        "routes": dict(sorted(route_counts.items())),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "isolation_contract": (
            "production incumbent and Qwen/Gemma exact events are unchanged; "
            "only Ministral CoT events are replaced"
        ),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


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


def _component_values(graph: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for component in graph["relational_graph"].get("components", []):
        value = try_parse_number(str(component.get("representative", "")))
        if value is None or not math.isfinite(value) or value <= 0:
            continue
        values[str(component["id"])] = float(value)
    return values


def _event_support(
    graph: Mapping[str, Any], component_values: Mapping[str, float],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    relational = graph["relational_graph"]
    nodes = {
        str(node["id"]): node
        for node in relational.get("nodes", [])
        if node.get("node_type") == "evidence_event"
    }
    targets: dict[str, set[str]] = defaultdict(set)
    for edge in relational.get("edges", []):
        if str(edge.get("edge_type")) != "supports":
            continue
        source, target = str(edge["source"]), str(edge["target"])
        if source in nodes and target in component_values:
            targets[source].add(target)
    events: list[dict[str, Any]] = []
    denominators: Counter[str] = Counter()
    for event_id, node in sorted(nodes.items()):
        family = str(node.get("model_family"))
        if family not in FAMILIES:
            continue
        # Every exact generation belongs in the denominator.  Counting only
        # parsed non-empty scalars would erase abstentions and parse failures,
        # artificially inflating a route's apparent confidence.
        denominators[family] += 1
        supported = targets.get(event_id, set())
        if str(node.get("status")) != "candidate_set" or not supported:
            continue
        # Capacity is single-valued.  A malformed multi-object event is not
        # silently converted into several independent votes.
        if len(supported) != 1:
            continue
        component_id = next(iter(supported))
        events.append({
            "id": event_id,
            "family": family,
            "component": component_id,
            "value": float(component_values[component_id]),
        })
    if set(denominators) != set(FAMILIES):
        raise ContractError(f"{_key(graph)}: incomplete family evidence")
    return events, denominators


def _complete_link_groups(
    component_values: Mapping[str, float],
) -> list[list[str]]:
    """Deterministic, non-transitive 5% groups over observed components."""
    groups: list[list[str]] = []
    ordered = sorted(component_values, key=lambda key: (
        component_values[key], key))
    for component_id in ordered:
        compatible = [
            index for index, group in enumerate(groups)
            if all(_near(component_values[component_id],
                         component_values[member]) for member in group)
        ]
        if not compatible:
            groups.append([component_id])
            continue
        # Prefer the group whose log-median is closest.  Complete-link avoids
        # the A~B~C transitivity error that can merge values >5% apart.
        index = min(compatible, key=lambda candidate: abs(
            math.log(component_values[component_id])
            - statistics.median(
                math.log(component_values[member])
                for member in groups[candidate])
        ))
        groups[index].append(component_id)
    return groups


def capacity_options(
    graph: Mapping[str, Any], incumbent_objects: Sequence[str],
) -> list[dict[str, Any]]:
    if str(graph["Relation"]) != RELATION:
        raise ContractError(f"capacity options requested for {_key(graph)}")
    if len(incumbent_objects) != 1:
        raise ContractError(f"{_key(graph)}: capacity incumbent must be scalar")
    incumbent = try_parse_number(str(incumbent_objects[0]))
    if incumbent is None or not math.isfinite(incumbent) or incumbent <= 0:
        raise ContractError(f"{_key(graph)}: invalid capacity incumbent")
    components = _component_values(graph)
    events, denominators = _event_support(graph, components)
    groups = _complete_link_groups(components)
    compatible_incumbent_groups = [
        index for index, group in enumerate(groups)
        if all(_near(incumbent, components[member]) for member in group)
    ]
    if not compatible_incumbent_groups:
        pseudo = "incumbent:pseudo"
        components[pseudo] = float(incumbent)
        groups = _complete_link_groups(components)
        compatible_incumbent_groups = [
            index for index, group in enumerate(groups)
            if pseudo in group
        ]
    incumbent_group = min(
        compatible_incumbent_groups,
        key=lambda index: (
            abs(
                math.log(incumbent)
                - statistics.median(
                    math.log(components[member]) for member in groups[index])
            ),
            index,
        ),
    )

    total_events = sum(denominators.values())
    options: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        group_set = set(group)
        grouped_events = [
            event for event in events if event["component"] in group_set]
        counts = Counter(event["family"] for event in grouped_events)
        rates = {
            family: counts[family] / denominators[family]
            for family in FAMILIES
        }
        supported_values = [float(event["value"]) for event in grouped_events]
        observed_values = [components[member] for member in group]
        representative_pool = supported_values or observed_values
        # Preserve an observed surface: select the support-weighted medoid.
        representative = min(
            sorted(set(representative_pool)),
            key=lambda value: (
                sum(abs(math.log(value / other)) for other in representative_pool),
                value,
            ),
        )
        is_incumbent = group_index == incumbent_group
        if is_incumbent:
            representative = float(incumbent)
        family_fraction = sum(counts[family] > 0 for family in FAMILIES) / 3.0
        family_rates = [rates[family] for family in FAMILIES]
        options.append({
            "value": float(representative),
            "component_ids": sorted(group),
            "is_incumbent": bool(is_incumbent),
            "family_fraction": family_fraction,
            "rates": rates,
            "mean_family_rate": statistics.mean(family_rates),
            "minimum_family_rate": min(family_rates),
            "maximum_family_rate": max(family_rates),
            "total_event_rate": len(grouped_events) / total_events,
            "component_fraction": len(group) / max(1, len(components)),
        })
    incumbents = [index for index, option in enumerate(options)
                  if option["is_incumbent"]]
    if len(incumbents) != 1:
        raise ContractError(f"{_key(graph)}: expected one incumbent option")
    return options


def option_features(
    option: Mapping[str, Any], incumbent: Mapping[str, Any],
) -> list[float]:
    rates = option["rates"]
    values = [
        1.0,
        float(option["is_incumbent"]),
        float(option["family_fraction"]),
        float(option["family_fraction"] == 1.0),
        float(rates["qwen_recall"]),
        float(rates["gemma_independent"]),
        float(rates["ministral_independent"]),
        float(option["mean_family_rate"]),
        float(option["minimum_family_rate"]),
        float(option["maximum_family_rate"]),
        float(option["total_event_rate"]),
        float(option["family_fraction"] - incumbent["family_fraction"]),
        float(option["mean_family_rate"] - incumbent["mean_family_rate"]),
        float(option["minimum_family_rate"] - incumbent["minimum_family_rate"]),
        float(option["component_fraction"]),
        min(1.0, abs(math.log(float(option["value"])
                              / float(incumbent["value"]))) / 5.0),
    ]
    if len(values) != len(FEATURE_NAMES) or not all(map(math.isfinite, values)):
        raise ContractError("capacity feature schema failure")
    return values


class CapacityGraphModel:
    def __init__(self, l2: float):
        self.l2 = float(l2)
        self.model = LogisticCalibrator(FEATURE_NAMES, l2=self.l2)

    def fit(
        self,
        graphs: Sequence[Mapping[str, Any]],
        incumbents: Mapping[tuple[str, str], Sequence[str]],
        gold: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> "CapacityGraphModel":
        x: list[list[float]] = []
        y: list[float] = []
        weights: list[float] = []
        for graph in graphs:
            key = _key(graph)
            options = capacity_options(graph, incumbents[key])
            incumbent = next(option for option in options if option["is_incumbent"])
            for option in options:
                x.append(option_features(option, incumbent))
                y.append(_correct([_format(option["value"])], gold[key]))
                weights.append(1.0 / len(options))
        self.model.fit(x, y, weights)
        return self

    def decode(
        self,
        graph: Mapping[str, Any],
        incumbent_objects: Sequence[str],
        margin: float,
    ) -> tuple[list[str], dict[str, Any]]:
        options = capacity_options(graph, incumbent_objects)
        incumbent_index = next(index for index, option in enumerate(options)
                               if option["is_incumbent"])
        incumbent = options[incumbent_index]
        probabilities = self.model.predict([
            option_features(option, incumbent) for option in options])
        best_index = max(range(len(options)), key=lambda index: (
            float(probabilities[index]),
            float(options[index]["family_fraction"]),
            float(options[index]["minimum_family_rate"]),
            -index,
        ))
        improvement = float(
            probabilities[best_index] - probabilities[incumbent_index])
        selected_index = best_index if improvement > margin else incumbent_index
        selected = options[selected_index]
        return [_format(selected["value"])], {
            "incumbent": list(map(str, incumbent_objects)),
            "selected": [_format(selected["value"])],
            "changed": selected_index != incumbent_index,
            "guard_margin": float(margin),
            "estimated_improvement": improvement,
            "selected_probability": float(probabilities[selected_index]),
            "incumbent_probability": float(probabilities[incumbent_index]),
            "options": [
                {
                    **option,
                    "probability": float(probability),
                }
                for option, probability in zip(options, probabilities)
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "capacity-three-family-graph-model-v1",
            "l2": self.l2,
            "features": list(FEATURE_NAMES),
            "calibrator": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapacityGraphModel":
        """Restore a frozen capacity model without refitting or labels."""
        if value.get("schema") != "capacity-three-family-graph-model-v1":
            raise ContractError("invalid frozen capacity graph model schema")
        if tuple(value.get("features", [])) != FEATURE_NAMES:
            raise ContractError("frozen capacity feature schema mismatch")
        l2 = float(value.get("l2", float("nan")))
        if not math.isfinite(l2) or l2 <= 0.0:
            raise ContractError("invalid frozen capacity L2")
        restored = cls(l2)
        restored.model = LogisticCalibrator.from_dict(value.get("calibrator", {}))
        if tuple(restored.model.names) != FEATURE_NAMES:
            raise ContractError("restored capacity feature schema mismatch")
        return restored


def _capacity_graphs(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    result = [row for row in rows if str(row["Relation"]) == RELATION]
    if len(result) != 100:
        raise ContractError(f"expected 100 capacity rows, got {len(result)}")
    return result


def _evaluate_rows(
    graphs: Sequence[Mapping[str, Any]],
    incumbents: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    model: CapacityGraphModel,
    margin: float,
) -> tuple[float, float, list[dict[str, Any]]]:
    before, after, details = [], [], []
    for graph in graphs:
        key = _key(graph)
        objects, detail = model.decode(graph, incumbents[key], margin)
        prior = _correct(incumbents[key], gold[key])
        current = _correct(objects, gold[key])
        before.append(prior)
        after.append(current)
        details.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "before_correct": prior,
            "after_correct": current,
            "delta": current - prior,
            **detail,
        })
    return statistics.mean(before), statistics.mean(after), details


def _inner_config(
    graphs: Sequence[Mapping[str, Any]],
    incumbents: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    folds: Mapping[tuple[str, str], int],
    outer_fold: int,
) -> tuple[float, float, dict[str, Any]]:
    candidates: dict[tuple[float, float], list[float]] = defaultdict(list)
    inner_folds = sorted({folds[_key(graph)] for graph in graphs
                          if folds[_key(graph)] != outer_fold})
    for inner_fold in inner_folds:
        fit = [graph for graph in graphs
               if folds[_key(graph)] not in {outer_fold, inner_fold}]
        holdout = [graph for graph in graphs
                   if folds[_key(graph)] == inner_fold]
        for l2 in L2_VALUES:
            model = CapacityGraphModel(l2).fit(fit, incumbents, gold)
            for margin in GUARD_MARGINS:
                before, after, _ = _evaluate_rows(
                    holdout, incumbents, gold, model, margin)
                candidates[(l2, margin)].append(after - before)
    selected = max(candidates, key=lambda config: (
        statistics.mean(candidates[config]),
        config[1],  # conservative incumbent guard on ties
        config[0],
    ))
    return selected[0], selected[1], {
        f"l2={l2:g},margin={margin:g}": {
            "fold_deltas": values,
            "mean_delta": statistics.mean(values),
        }
        for (l2, margin), values in sorted(candidates.items())
    }


def _merge(
    incumbent_rows: Sequence[Mapping[str, Any]],
    replacements: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, Any]]:
    return [{
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
        "ObjectEntities": list(map(str, replacements.get(
            _key(row), row.get("ObjectEntities", [])))),
    } for row in incumbent_rows]


def _option_oracle_diagnostic(
    graphs: Sequence[Mapping[str, Any]],
    incumbents: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    incumbent_correct = 0
    option_oracle_correct = 0
    wrong_incumbent_with_correct_option = 0
    option_counts = []
    for graph in graphs:
        key = _key(graph)
        options = capacity_options(graph, incumbents[key])
        baseline = _correct(incumbents[key], gold[key])
        oracle = max(
            _correct([_format(option["value"])], gold[key])
            for option in options
        )
        incumbent_correct += int(baseline)
        option_oracle_correct += int(oracle)
        wrong_incumbent_with_correct_option += int(not baseline and oracle)
        option_counts.append(len(options))
    return {
        "rows": len(graphs),
        "incumbent_correct": incumbent_correct,
        "option_oracle_correct": option_oracle_correct,
        "incumbent_accuracy": incumbent_correct / len(graphs),
        "option_oracle_accuracy": option_oracle_correct / len(graphs),
        "wrong_incumbent_with_correct_option": (
            wrong_incumbent_with_correct_option),
        "mean_options": statistics.mean(option_counts),
        "warning": (
            "Gold-aware diagnostic only; it measures supply and is never a "
            "decoder or deployment policy."),
    }


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_train_graph = Path(args.train_graph).resolve()
    train_graph_path = output / "PREPARED_TRAIN_CAPACITY_GRAPH.jsonl"
    preparation = build_repaired_train_graph(
        source_train_graph,
        Path(args.repaired_cot_response).resolve(),
        train_graph_path,
    )
    train_rows = _validate_graphs(train_graph_path, 100)
    train_graphs = _capacity_graphs(train_rows)

    train_incumbent_rows, train_incumbent_manifest = (
        compose_competition_train_oof())
    train_incumbents = _prediction_map(train_incumbent_rows)
    train_keys = {_key(row) for row in train_rows}
    if not train_keys <= set(train_incumbents):
        raise ContractError("train incumbent/graph coverage mismatch")

    train_gold_path = Path(args.train_gold).resolve()
    train_gold = {_key(row): row for row in read_jsonl(train_gold_path)}
    folds = subject_grouped_folds(train_rows, n_folds=5)
    outer_records: list[dict[str, Any]] = []
    oof_details: list[dict[str, Any]] = []
    selected_configs: list[tuple[float, float]] = []
    for outer_fold in range(5):
        fit = [graph for graph in train_graphs
               if folds[_key(graph)] != outer_fold]
        holdout = [graph for graph in train_graphs
                   if folds[_key(graph)] == outer_fold]
        l2, margin, inner = _inner_config(
            train_graphs, train_incumbents, train_gold, folds, outer_fold)
        selected_configs.append((l2, margin))
        model = CapacityGraphModel(l2).fit(fit, train_incumbents, train_gold)
        before, after, details = _evaluate_rows(
            holdout, train_incumbents, train_gold, model, margin)
        for detail in details:
            detail["outer_fold"] = outer_fold
            detail["selected_l2"] = l2
            oof_details.append(detail)
        outer_records.append({
            "fold": outer_fold,
            "rows": len(holdout),
            "l2": l2,
            "guard_margin": margin,
            "baseline_accuracy": before,
            "selected_accuracy": after,
            "delta": after - before,
            "inner_selection": inner,
        })

    baseline_oof = statistics.mean(row["before_correct"] for row in oof_details)
    selected_oof = statistics.mean(row["after_correct"] for row in oof_details)
    fold_deltas = [float(row["delta"]) for row in outer_records]
    helped = sum(float(row["delta"]) > 0 for row in oof_details)
    harmed = sum(float(row["delta"]) < 0 for row in oof_details)
    changed = sum(bool(row["changed"]) for row in oof_details)
    ratio = helped / max(1, harmed)
    gate_checks = {
        "minimum_oof_delta": selected_oof - baseline_oof >= MIN_OOF_DELTA,
        "minimum_positive_folds": sum(delta > 0 for delta in fold_deltas)
        >= MIN_POSITIVE_FOLDS,
        "maximum_fold_regression": min(fold_deltas) >= MAX_FOLD_REGRESSION,
        "minimum_help_harm_ratio": ratio >= MIN_HELP_HARM_RATIO,
    }
    gate_passed = all(gate_checks.values())
    train_supply = _option_oracle_diagnostic(
        train_graphs, train_incumbents, train_gold)

    # Median outer-selected hyperparameters are robust to a single fold and do
    # not use development labels.
    final_l2 = float(statistics.median(config[0] for config in selected_configs))
    final_margin = float(statistics.median(
        config[1] for config in selected_configs))
    final_model = CapacityGraphModel(final_l2).fit(
        train_graphs, train_incumbents, train_gold)
    model_payload = {
        **final_model.to_dict(),
        "guard_margin": final_margin,
        "gate_passed": gate_passed,
        "gate_checks": gate_checks,
        "train_graph_sha256": sha256(train_graph_path),
        "train_gold_sha256": sha256(train_gold_path),
        "validation_labels_used": False,
        "test_labels_used": False,
    }
    model_path = output / "FROZEN_MODEL.json"
    _write_json(model_path, model_payload)

    write_jsonl_atomic(output / "TRAIN_OOF_DECISIONS.jsonl", oof_details)
    gate_payload = {
        "schema": "capacity-three-family-graph-gate-v1",
        "gate_passed": gate_passed,
        "checks": gate_checks,
        "thresholds": {
            "minimum_oof_delta": MIN_OOF_DELTA,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "maximum_fold_regression": MAX_FOLD_REGRESSION,
            "minimum_help_harm_ratio": MIN_HELP_HARM_RATIO,
        },
        "baseline_oof": baseline_oof,
        "selected_oof": selected_oof,
        "oof_delta": selected_oof - baseline_oof,
        "changed": changed,
        "helped": helped,
        "harmed": harmed,
        "help_harm_ratio": ratio,
        "outer_folds": outer_records,
        "final_l2": final_l2,
        "final_guard_margin": final_margin,
        "train_incumbent": train_incumbent_manifest,
        "prepared_graph": preparation,
        "train_supply_diagnostic": train_supply,
        "validation_labels_used": False,
        "test_labels_used": False,
    }
    _write_json(output / "TRAIN_GATE.json", gate_payload)

    if args.train_only:
        print(json.dumps({
            "gate_passed": gate_passed,
            "train_oof_baseline": baseline_oof,
            "train_oof_selected": selected_oof,
            "train_oof_delta": selected_oof - baseline_oof,
            "changed": changed,
            "helped": helped,
            "harmed": harmed,
            "train_gate": str((output / "TRAIN_GATE.json").resolve()),
            "model": str(model_path.resolve()),
        }, indent=2, sort_keys=True))
        return 0

    validation_graph_path = Path(args.validation_graph).resolve()
    validation_rows = _validate_graphs(validation_graph_path, 478)
    validation_graphs = _capacity_graphs(validation_rows)
    validation_incumbent_path = Path(args.validation_incumbent).resolve()
    validation_incumbent_rows = read_jsonl(validation_incumbent_path)
    validation_incumbents = _prediction_map(validation_incumbent_rows)
    if set(validation_incumbents) != {_key(row) for row in validation_rows}:
        raise ContractError("validation incumbent/graph coverage mismatch")

    replacements: dict[tuple[str, str], list[str]] = {}
    validation_details: list[dict[str, Any]] = []
    for graph in validation_graphs:
        key = _key(graph)
        objects, detail = final_model.decode(
            graph, validation_incumbents[key], final_margin)
        replacements[key] = objects
        validation_details.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            **detail,
        })
    ungated_rows = _merge(validation_incumbent_rows, replacements)
    gated_rows = (
        ungated_rows if gate_passed
        else _merge(validation_incumbent_rows, {}))
    ungated_path = output / "VALIDATION_UNGATED_PREDICTIONS.jsonl"
    gated_path = output / "VALIDATION_GATED_PREDICTIONS.jsonl"
    write_jsonl_atomic(ungated_path, ungated_rows)
    write_jsonl_atomic(gated_path, gated_rows)
    write_jsonl_atomic(
        output / "VALIDATION_DECISIONS.jsonl", validation_details)

    # Leakage boundary: validation gold is opened only after both prediction
    # alternatives and the gate have been serialized.
    validation_gold_path = Path(args.validation_gold).resolve()
    validation_gold = read_jsonl(validation_gold_path)
    validation_gold_by_key = {_key(row): row for row in validation_gold}
    validation_supply = _option_oracle_diagnostic(
        validation_graphs, validation_incumbents, validation_gold_by_key)
    incumbent_scores = score(validation_incumbent_rows, validation_gold)
    ungated_scores = score(ungated_rows, validation_gold)
    gated_scores = score(gated_rows, validation_gold)
    result = {
        "schema": "capacity-three-family-graph-result-v1",
        "development_only": True,
        "gate_passed": gate_passed,
        "validation_labels_used_for_selection": False,
        "validation_labels_used_for_posthoc_scoring": True,
        "test_labels_used": False,
        "train_gate": str((output / "TRAIN_GATE.json").resolve()),
        "train_gate_sha256": sha256(output / "TRAIN_GATE.json"),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "validation_incumbent_sha256": sha256(validation_incumbent_path),
        "validation_graph_sha256": sha256(validation_graph_path),
        "validation_gold_sha256": sha256(validation_gold_path),
        "validation_changed_ungated": sum(
            bool(row["changed"]) for row in validation_details),
        "train_supply_diagnostic": train_supply,
        "validation_supply_diagnostic": validation_supply,
        "scores": {
            "incumbent": incumbent_scores,
            "ungated": ungated_scores,
            "gated": gated_scores,
        },
        "capacity_deltas": {
            "ungated": ungated_scores[RELATION] - incumbent_scores[RELATION],
            "gated": gated_scores[RELATION] - incumbent_scores[RELATION],
        },
        "pooled_deltas": {
            "ungated": ungated_scores["*** All Relations ***"]
            - incumbent_scores["*** All Relations ***"],
            "gated": gated_scores["*** All Relations ***"]
            - incumbent_scores["*** All Relations ***"],
        },
    }
    _write_json(output / "RESULT.json", result)
    (output / "RESULT.md").write_text(
        "# Three-family capacity graph decoder\n\n"
        f"- Train nested-OOF capacity: **{baseline_oof:.4f} -> "
        f"{selected_oof:.4f} ({selected_oof - baseline_oof:+.4f})**\n"
        f"- Train gate passed: **{gate_passed}**\n"
        f"- Validation incumbent capacity: "
        f"**{incumbent_scores[RELATION]:.4f}**\n"
        f"- Validation ungated capacity: **{ungated_scores[RELATION]:.4f} "
        f"({ungated_scores[RELATION] - incumbent_scores[RELATION]:+.4f})**\n"
        f"- Validation gated capacity: **{gated_scores[RELATION]:.4f} "
        f"({gated_scores[RELATION] - incumbent_scores[RELATION]:+.4f})**\n"
        f"- Validation ungated edits: **{sum(bool(row['changed']) for row in validation_details)}**\n\n"
        "The ungated score is a post-hoc diagnostic. Only the gated artifact is "
        "eligible for integration, and only when the train-only contract passes.\n"
    )
    print(json.dumps({
        "gate_passed": gate_passed,
        "train_oof_delta": selected_oof - baseline_oof,
        "validation_capacity_ungated_delta": (
            ungated_scores[RELATION] - incumbent_scores[RELATION]),
        "validation_capacity_gated_delta": (
            gated_scores[RELATION] - incumbent_scores[RELATION]),
        "result": str((output / "RESULT.md").resolve()),
    }, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-graph", default=str(DEFAULT_TRAIN_GRAPH))
    parser.add_argument(
        "--repaired-cot-response", default=str(DEFAULT_REPAIRED_COT_RESPONSE))
    parser.add_argument("--validation-graph", default=str(DEFAULT_VALIDATION_GRAPH))
    parser.add_argument(
        "--validation-incumbent", default=str(DEFAULT_VALIDATION_INCUMBENT))
    parser.add_argument("--train-gold", default=str(DEFAULT_TRAIN_GOLD))
    parser.add_argument("--validation-gold", default=str(DEFAULT_VALIDATION_GOLD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--train-only", action="store_true",
        help="fit and gate on train only; do not open validation artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
