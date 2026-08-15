#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence
from lm_kbc.core import ContractError
from lm_kbc.components.generation_set_hypothesis_audit import FAMILIES, _key, build_hypotheses, hypothesis_stats

PRIMARY_ARM = "strict_proof_graph"

IDENTITY_RELATIONS = frozenset({"awardWonBy"})

MIN_EXACT_FAMILY_FRACTION = 2.0 / 3.0

MIN_EXACT_FAMILY_ADVANTAGE = 2.0 / 3.0

MIN_FAMILY_SIMILARITY = 0.5

EPSILON = 1e-12

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
