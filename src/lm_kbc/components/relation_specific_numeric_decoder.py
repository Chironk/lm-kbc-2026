#!/usr/bin/env python3
"""Frozen production numeric-decoder helpers.

Only symbols reached by public inference and deterministic replay are retained.
The complete research source is preserved on the local recovery branch
``archive/pre-consolidation-20260814`` and in the ignored recovery copy.
"""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
import numpy as np
from evaluate import RELATION_TYPE, true_positives
from lm_kbc.core import ContractError, NUMERIC_RELATIONS
from lm_kbc.components.dual_model_validation import GEMMA, QWEN
from lm_kbc.components.heterogeneous_memory_selector import LogisticCalibrator, _key, _numeric_value, _weighted_median
RELATIONS = tuple(sorted(NUMERIC_RELATIONS))
OPTION_SCHEMA = 'relation-specific-numeric-option-v1'
FEATURE_SCHEMA = 'relation-specific-numeric-features-v1'

def _within_tolerance(left: float, right: float, tolerance: float=0.05) -> bool:
    """Symmetric neighborhood used only to construct inference-time clusters."""
    if left <= 0 or right <= 0:
        return False
    return abs(left - right) / max(abs(right), 1e-12) <= tolerance

def _same_option(left: float, right: float) -> bool:
    return abs(math.log(left / right)) <= 1e-10

def _source_support(node: Mapping[str, Any], agent: str) -> float:
    source = node.get('sources', {}).get(agent)
    return float(source.get('support', 0.0)) if source else 0.0

def _source_samples(graph: Mapping[str, Any], agent: str) -> float:
    value = float(graph['agents'][agent].get('n_samples', 0.0))
    return max(1.0, value)

def _numeric_mad(graph: Mapping[str, Any], agent: str) -> float:
    value = graph['agents'][agent].get('numeric_log_mad')
    if value is None:
        return 0.0
    value = float(value)
    return min(1.0, max(0.0, value) / 2.0)

def _add_option(options: list[dict], value: float | None, kind: str) -> None:
    if value is None or not math.isfinite(value) or value <= 0:
        return
    for option in options:
        if _same_option(float(option['value']), value):
            option['kinds'].add(kind)
            return
    options.append({'schema': OPTION_SCHEMA, 'value': float(value), 'kinds': {kind}})

def numeric_options(graph: Mapping[str, Any]) -> list[dict]:
    """Enumerate legal outputs without labels.

    Weighted medians preserve an observed value.  Weighted geometric means add
    a smooth representative on the positive numeric scale and are restricted
    to a local 5%-neighborhood so they cannot average unrelated magnitudes.
    """
    if graph['Relation'] not in NUMERIC_RELATIONS:
        raise ContractError(f'numeric option request for nonnumeric row: {_key(graph)}')
    options: list[dict] = []
    for item in graph.get('baseline_objects', []):
        _add_option(options, _numeric_value(item), 'baseline')
    nodes: list[tuple[float, Mapping[str, Any]]] = []
    for node in graph.get('candidates', []):
        value = _numeric_value(node.get('item'))
        if value is None:
            continue
        nodes.append((value, node))
        _add_option(options, value, 'node')
    for anchor, _ in nodes:
        members = [(value, node) for value, node in nodes if _within_tolerance(value, anchor)]
        weights = [sum((_source_support(node, agent) for agent in (QWEN, GEMMA))) for _, node in members]
        if not members or sum(weights) <= 0:
            continue
        _add_option(options, _weighted_median([(value, weight) for (value, _), weight in zip(members, weights)]), 'cluster_median')
        log_mean = sum((math.log(value) * weight for (value, _), weight in zip(members, weights))) / sum(weights)
        _add_option(options, math.exp(log_mean), 'cluster_geomean')
    if not options:
        raise ContractError(f'numeric graph has no legal output option: {_key(graph)}')
    if not any(('baseline' in option['kinds'] for option in options)):
        raise ContractError(f'numeric graph has no legal baseline: {_key(graph)}')
    return options

def feature_names() -> list[str]:
    return ['intercept', 'is_baseline', 'is_raw_node', 'is_cluster_median', 'is_cluster_geomean', 'qwen_support_mass', 'gemma_support_mass', 'cross_model_neighborhood', 'qwen_selected_neighborhood', 'gemma_selected_neighborhood', 'neighborhood_node_count', 'neighborhood_total_mass', 'qwen_numeric_log_mad', 'gemma_numeric_log_mad', 'log_distance_from_baseline', 'candidate_count']

def option_features(graph: Mapping[str, Any], option: Mapping[str, Any]) -> list[float]:
    value = float(option['value'])
    near = []
    for node in graph.get('candidates', []):
        candidate = _numeric_value(node.get('item'))
        if candidate is not None and _within_tolerance(candidate, value):
            near.append(node)
    qmass = sum((_source_support(node, QWEN) for node in near))
    gmass = sum((_source_support(node, GEMMA) for node in near))
    qrate = min(1.0, qmass / _source_samples(graph, QWEN))
    grate = min(1.0, gmass / _source_samples(graph, GEMMA))
    baseline_values = [_numeric_value(item) for item in graph.get('baseline_objects', [])]
    baseline_values = [item for item in baseline_values if item is not None]
    if not baseline_values:
        raise ContractError(f'missing numeric baseline: {_key(graph)}')
    baseline = baseline_values[0]
    kinds = set(option['kinds'])
    values = [1.0, float('baseline' in kinds), float('node' in kinds), float('cluster_median' in kinds), float('cluster_geomean' in kinds), qrate, grate, float(qmass > 0 and gmass > 0), float(any((node.get('selected_by', {}).get(QWEN, False) for node in near))), float(any((node.get('selected_by', {}).get(GEMMA, False) for node in near))), min(1.0, len(near) / 5.0), min(1.0, (qrate + grate) / 2.0), _numeric_mad(graph, QWEN), _numeric_mad(graph, GEMMA), min(1.0, abs(math.log(value / baseline)) / 5.0), min(1.0, len(graph.get('candidates', [])) / 11.0)]
    if len(values) != len(feature_names()):
        raise AssertionError('numeric option feature schema drift')
    if not all((math.isfinite(item) for item in values)):
        raise ContractError(f'non-finite numeric option features: {_key(graph)}')
    return values

def _gold_aliases(gold: Mapping[str, Any]) -> list[list[str]]:
    values = gold.get('ObjectEntities', [])
    return [[str(item)] for item in values] if values and isinstance(values[0], str) else values

def option_label(graph: Mapping[str, Any], option: Mapping[str, Any], gold: Mapping[str, Any]) -> float:
    prediction = format(float(option['value']), '.12g')
    return float(true_positives([prediction], _gold_aliases(gold), RELATION_TYPE[str(graph['Relation'])], 0.05) > 0)

class RelationSpecificNumericModel:
    """One option-correctness calibrator per numeric relation."""

    def __init__(self, l2: float=2.0):
        self.l2 = float(l2)
        self.models: dict[str, LogisticCalibrator] = {}

    def fit(self, graphs: Sequence[Mapping[str, Any]], gold_by_key: Mapping[tuple[str, str], Mapping[str, Any]]) -> 'RelationSpecificNumericModel':
        for relation in RELATIONS:
            subset = [graph for graph in graphs if graph['Relation'] == relation]
            if not subset:
                raise ContractError(f'no numeric training rows for {relation}')
            x, y, weights = ([], [], [])
            for graph in subset:
                if _key(graph) not in gold_by_key:
                    raise ContractError(f'missing training gold for {_key(graph)}')
                options = numeric_options(graph)
                row_weight = 1.0 / len(options)
                for option in options:
                    x.append(option_features(graph, option))
                    y.append(option_label(graph, option, gold_by_key[_key(graph)]))
                    weights.append(row_weight)
            self.models[relation] = LogisticCalibrator(feature_names(), l2=self.l2).fit(x, y, weights)
        return self

    def score_options(self, graph: Mapping[str, Any]) -> tuple[list[dict], np.ndarray]:
        relation = str(graph['Relation'])
        if relation not in self.models:
            raise ContractError(f'numeric model missing relation {relation}')
        options = numeric_options(graph)
        probabilities = self.models[relation].predict([option_features(graph, option) for option in options])
        return (options, probabilities)

    def decode(self, graph: Mapping[str, Any], margin: float) -> tuple[list[str], dict[str, Any]]:
        options, probabilities = self.score_options(graph)
        baseline_index = next((index for index, option in enumerate(options) if 'baseline' in option['kinds']))
        best_index = max(range(len(options)), key=lambda index: (float(probabilities[index]), -index))
        improvement = float(probabilities[best_index]) - float(probabilities[baseline_index])
        selected_index = best_index if improvement > margin else baseline_index
        selected = options[selected_index]
        return ([format(float(selected['value']), '.12g')], {'relation': graph['Relation'], 'selected_value': float(selected['value']), 'selected_kinds': sorted(selected['kinds']), 'selected_probability': float(probabilities[selected_index]), 'best_probability': float(probabilities[best_index]), 'baseline_value': float(options[baseline_index]['value']), 'baseline_probability': float(probabilities[baseline_index]), 'estimated_improvement': improvement, 'guard_margin': float(margin), 'used_baseline': selected_index == baseline_index, 'options': [{'value': float(option['value']), 'kinds': sorted(option['kinds']), 'probability': float(probability)} for option, probability in zip(options, probabilities)]})

    def to_dict(self) -> dict[str, Any]:
        return {'schema': 'relation-specific-numeric-model-v1', 'feature_schema': FEATURE_SCHEMA, 'feature_names': feature_names(), 'l2': self.l2, 'models': {relation: model.to_dict() for relation, model in sorted(self.models.items())}}

def _merge_numeric(control_rows: Sequence[Mapping[str, Any]], numeric_rows: Mapping[tuple[str, str], Sequence[str]]) -> list[dict]:
    merged = []
    control_keys = {_key(row) for row in control_rows}
    if not set(numeric_rows) <= control_keys:
        raise ContractError('numeric predictions are not covered by control')
    for row in control_rows:
        key = _key(row)
        merged.append({'SubjectEntity': row['SubjectEntity'], 'Relation': row['Relation'], 'ObjectEntities': list(numeric_rows.get(key, row.get('ObjectEntities', [])))})
    return merged
