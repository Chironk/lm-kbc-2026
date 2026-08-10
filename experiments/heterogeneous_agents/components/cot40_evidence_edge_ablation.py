#!/usr/bin/env python3
"""Test which additions to the minimal evidence graph are actually useful.

This is a train-only, fail-closed graph experiment.  It starts from the
CoT40 candidate/component inventory, reconstructs exact Qwen and routed Gemma
generation membership from immutable raw artifacts, and materializes five
small edge families:

* ``supports``: evidence event -> candidate component;
* ``co_occurs_with``: candidate component <-> component when both appeared in
  the same exact generation;
* ``asserts_cardinality``: event -> ZERO/ONE/MANY state;
* ``asserts_existence``: event -> EMPTY/NONEMPTY state; and
* ``asserts_claim``: event -> deterministic relation-scope claim.

The analysis changes one edge family at a time over a matched component-table
selector.  Every arm uses identical rows, candidate components, legal actions,
targets, nested subject-grouped folds, ridge model, regularization, and action
guards.  The all-edge arm also has a subject-shifted negative control.

There is intentionally no validation command and no deployable prediction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sample_evidence import (
    ENTITY,
    EXPLICIT_ABSTENTION,
    classify_samples,
)
from evaluate import normalize_string

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.baseline_relative_route_decoder import (
    ResidualRidge,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    proposal_parse_status,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.cot40_graph_native_decoder import (
    GUARDS,
    L2_VALUES,
    LOCAL_NAMES,
    POOLED,
    RELATIONS,
    _prediction_rows,
    _row_f1,
    _score_predictions,
    _validate_prepared,
    cot40_count_anchor,
    legal_actions,
)
from experiments.heterogeneous_agents.components.cot40_minimal_evidence_graph import (
    EVENT_NAMES,
    _paired_audit,
    component_features,
    event_features,
    minimalize_row,
    parity_audit,
)
from experiments.heterogeneous_agents.components.graph_event_contract import (
    repair_unsupported_candidate_set_events,
)
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    _key,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    compose_competition_train_oof,
)
from experiments.heterogeneous_agents.components.three_model_component_decoder import (
    subject_grouped_folds,
)


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
DEFAULT_SOURCE = RUNS / "cot40_graph_native_decoder_20260729_v1"
DEFAULT_MINIMAL = RUNS / "cot40_minimal_evidence_graph_20260730_v1"
DEFAULT_OUTPUT = RUNS / "cot40_evidence_edge_ablation_20260730_v1"
DEFAULT_GOLD = ROOT / "data/train.jsonl"

PRODUCTION_AUDIT = (
    RUNS / "production_matched_oof_20260723_v1/"
    "PRODUCTION_MATCH_AUDIT.json"
)
EXPANDED_PLAN = (
    RUNS / "expanded_calibration_n1_20260723_v1/plan/PLAN.json"
)
GEMMA_BASE = (
    RUNS / "expanded_calibration_n1_20260723_v1/"
    "responses/gemma_evidence.jsonl"
)
TARGETED_GEMMA_PLAN = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/plan/PLAN.json"
)
TARGETED_GEMMA_RESPONSE = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/"
    "responses/gemma_extra2.jsonl"
)

PLAN_SCHEMA = "cot40-evidence-edge-ablation-plan-v1"
ROW_SCHEMA = "typed-minimal-evidence-row-v1"
GRAPH_SCHEMA = "typed-minimal-evidence-graph-v1"
MANIFEST_SCHEMA = "typed-minimal-evidence-graph-manifest-v1"
RESULT_SCHEMA = "cot40-evidence-edge-ablation-result-v1"

ROUTE_QWEN = "qwen:self_consistency"
ROUTE_SYSTEM2 = "qwen:system2"
ROUTE_GEMMA = "gemma:independent"
ROUTE_MINISTRAL = "ministral:cot5_cap40_n10"

FAMILIES = (
    "qwen_recall",
    "gemma_independent",
    "ministral_independent",
)

ARMS = (
    "component_table",
    "exact_support",
    "exact_support_cooccurrence",
    "exact_support_cardinality",
    "exact_support_existence",
    "exact_support_claims",
    "all_typed_edges",
    "all_typed_edges_shifted",
)

FAMILY_SUPPORT_NAMES = tuple(
    f"support_{family}_{metric}"
    for family in FAMILIES
    for metric in ("jaccard_mean", "set_rate", "selected_coverage")
)
SUPPORT_NAMES = (*EVENT_NAMES, *FAMILY_SUPPORT_NAMES)

COOCCURRENCE_NAMES = (
    "co_selected_pair_rate_mean",
    "co_selected_pair_rate_max",
    "co_selected_pair_rate_min",
    "co_selected_pair_coverage",
    "co_selected_internal_mass_share",
    "co_boundary_mass_share",
    "co_omitted_internal_mass_share",
)

CARDINALITY_NAMES = (
    "cardinality_match_rate",
    "cardinality_distance_mean",
    *tuple(f"cardinality_match_{family}" for family in FAMILIES),
)

EXISTENCE_NAMES = (
    "existence_match_rate",
    "existence_disagreement_rate",
    *tuple(f"existence_match_{family}" for family in FAMILIES),
)

RISK_CLAIMS = (
    "area_non_km_unit",
    "award_partial_scope",
    "border_maritime_only",
    "capacity_variant",
    "city_noncity_scope",
    "listing_historical",
    "listing_inactive",
)
POSITIVE_CLAIMS = (
    "area_km2_unit",
    "border_land",
    "capacity_total",
    "city_explicit",
    "listing_current",
)
CLAIM_NAMES = (
    "claim_selected_clean_mass_share",
    "claim_selected_risk_mass_share",
    "claim_omitted_risk_mass_share",
    "claim_risky_event_rate",
    "claim_positive_event_rate",
    *tuple(f"claim_selected_{claim}" for claim in RISK_CLAIMS),
)

TYPED_NAMES = (
    *SUPPORT_NAMES,
    *COOCCURRENCE_NAMES,
    *CARDINALITY_NAMES,
    *EXISTENCE_NAMES,
    *CLAIM_NAMES,
)

MIN_INCREMENT = 0.003
MIN_FOLD_WINS = 3
MAX_RELATION_REGRESSION = -0.01
MIN_ALIGNED_OVER_SHIFTED = 0.001


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _manifest(path: Path) -> dict[str, Any]:
    return _json(path.with_suffix(path.suffix + ".manifest.json"))


def _response_map(path: Path, *, agent: str) -> dict[tuple[str, str], dict]:
    manifest = _manifest(path)
    if (
        manifest.get("agent_id") != agent
        or manifest.get("output_sha256") != sha256(path)
    ):
        raise ContractError(f"response manifest mismatch: {path}")
    result: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(path):
        if (
            row.get("agent_id") != agent
            or row.get("phase") != "propose"
            or row.get("mode") != "generate"
        ):
            continue
        key = (str(row["subject"]), str(row["relation"]))
        if key in result:
            raise ContractError(f"duplicate proposal response: {key}")
        result[key] = row
    return result


def _qwen_raw_sources() -> tuple[
    dict[tuple[str, str], list[str]], dict[str, Any],
]:
    """Load the exact raw samples that created the production-matched graph."""
    audit = _json(PRODUCTION_AUDIT)
    if audit.get("labels_opened") is not False:
        raise ContractError("production raw audit does not exclude labels")
    result: dict[tuple[str, str], list[str]] = {}
    sources: dict[str, Any] = {}
    for name in ("fp16_system1", "border_system1"):
        source = audit["sources"][name]
        path = Path(source["path"])
        if sha256(path) != source["sha256"]:
            raise ContractError(f"Qwen production raw hash mismatch: {path}")
        sources[name] = {"path": str(path), "sha256": source["sha256"]}
        for row in read_jsonl(path):
            key = _key(row)
            if (
                name == "border_system1"
                and key[1] != "countryLandBordersCountry"
            ):
                continue
            if (
                name == "fp16_system1"
                and key[1] == "countryLandBordersCountry"
            ):
                continue
            samples = [str(value) for value in row.get("raw_samples", [])]
            if len(samples) != 10 or key in result:
                raise ContractError(f"invalid Qwen production row: {key}")
            result[key] = samples

    # The production-matched transform deliberately preserved the ten award
    # rows.  Their exact N=10 samples are pinned in the expanded OOF plan.
    expanded = _json(EXPANDED_PLAN)
    provenance = expanded["qwen_oof_provenance"]
    award_sources = []
    for fold in range(5):
        record = provenance[str(fold)]
        path = Path(record["raw"])
        if sha256(path) != record["raw_sha256"]:
            raise ContractError(f"Qwen award raw hash mismatch: {path}")
        award_sources.append({
            "path": str(path), "sha256": record["raw_sha256"]})
        for row in read_jsonl(path):
            key = _key(row)
            if key[1] != "awardWonBy":
                continue
            samples = [str(value) for value in row.get("raw_samples", [])]
            if len(samples) != 10 or key in result:
                raise ContractError(f"invalid Qwen award row: {key}")
            result[key] = samples
    sources["award_oof"] = award_sources
    if len(result) != 477:
        raise ContractError(
            f"Qwen exact raw coverage is {len(result)}, expected 477")
    return result, sources


def _ministral_sources(
    source_plan: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict], dict[str, Any]]:
    path = Path(source_plan["responses"])
    if sha256(path) != source_plan["responses_sha256"]:
        raise ContractError("Ministral source response hash mismatch")
    result = _response_map(path, agent="ministral_independent")
    if len(result) != 477:
        raise ContractError("Ministral response coverage mismatch")
    return result, {"path": str(path), "sha256": sha256(path)}


def _gemma_sources() -> tuple[
    dict[tuple[str, str], dict],
    dict[tuple[str, str], list[str]],
    dict[str, Any],
]:
    base = _response_map(GEMMA_BASE, agent="gemma_independent")
    if len(base) != 477:
        raise ContractError("Gemma N=1 response coverage mismatch")
    plan = _json(TARGETED_GEMMA_PLAN)
    if (
        plan.get("schema") != "targeted-company-gemma-n3-plan-v1"
        or sha256(Path(plan["inputs"])) != plan["inputs_sha256"]
        or sha256(TARGETED_GEMMA_RESPONSE)
            != _manifest(TARGETED_GEMMA_RESPONSE)["output_sha256"]
    ):
        raise ContractError("targeted Gemma plan/response mismatch")
    inputs = read_jsonl(Path(plan["inputs"]))
    responses = {
        str(row["task_id"]): row
        for row in read_jsonl(TARGETED_GEMMA_RESPONSE)
    }
    extras: dict[tuple[str, str], list[str]] = {}
    for index, source in enumerate(inputs):
        if source.get("_split") != "train":
            continue
        key = _key(source)
        response = responses.get(f"gemma_independent::{index}::proposal")
        if response is None:
            raise ContractError(f"missing targeted Gemma response: {key}")
        generations = [str(value) for value in response["generations"]]
        if len(generations) != 2 or key in extras:
            raise ContractError(f"invalid targeted Gemma generations: {key}")
        extras[key] = generations
    return base, extras, {
        "base": {"path": str(GEMMA_BASE), "sha256": sha256(GEMMA_BASE)},
        "targeted_plan": {
            "path": str(TARGETED_GEMMA_PLAN),
            "sha256": sha256(TARGETED_GEMMA_PLAN),
        },
        "targeted_response": {
            "path": str(TARGETED_GEMMA_RESPONSE),
            "sha256": sha256(TARGETED_GEMMA_RESPONSE),
        },
    }


def _reasoning(text: str) -> str:
    value = str(text)
    match = re.search(r"<think>(.*?)(?:</think>|$)", value, re.I | re.S)
    if match:
        return match.group(1).strip()
    before = re.split(r"\bANSWER\s*:", value, maxsplit=1, flags=re.I)[0]
    before = re.sub(r"^\s*REASONING\s*:\s*", "", before, flags=re.I)
    return before.strip()


def reasoning_claims(text: str, relation: str) -> list[str]:
    """Extract only explicit, relation-scoped claims from model reasoning.

    These are lexical observations, not truth labels or confidence scores.
    Risk/positive interpretation is applied only by the train-side selector.
    """
    value = re.sub(r"\s+", " ", _reasoning(text).casefold())
    claims: set[str] = set()
    if relation == "hasArea":
        has_km = bool(re.search(
            r"\b(square\s+kilomet(?:er|re)s?|sq\.?\s*km|km\s*[²2])\b",
            value,
        ))
        has_other = bool(re.search(
            r"\b(square\s+miles?|sq\.?\s*mi|acres?|hectares?)\b", value))
        if has_km:
            claims.add("area_km2_unit")
        if has_other and not has_km and "convert" not in value:
            claims.add("area_non_km_unit")
    elif relation == "hasCapacity":
        if re.search(
            r"\b(seated|seating|standing|concert|temporary|configuration)\b",
            value,
        ):
            claims.add("capacity_variant")
        if re.search(
            r"\b(total|official|spectator)\s+(?:seating\s+)?capacity\b",
            value,
        ):
            claims.add("capacity_total")
    elif relation == "companyTradesAtStockExchange":
        if re.search(
            r"\b(formerly|previously|historically|used to|at one time)\b",
            value,
        ):
            claims.add("listing_historical")
        if re.search(
            r"\b(delisted|no longer (?:listed|traded)|taken private|"
            r"privately held|not publicly traded|acquired)\b",
            value,
        ):
            claims.add("listing_inactive")
        if (
            re.search(
                r"\b(currently|shares (?:are )?traded|shares trade|"
                r"listed on|publicly traded)\b",
                value,
            )
            and not ({"listing_historical", "listing_inactive"} & claims)
        ):
            claims.add("listing_current")
    elif relation == "awardWonBy":
        if (
            re.search(r"\b(?:19|20)\d{2}\b", value)
            or re.search(
                r"\b(recent|latest|for example|focus on|specific year)\b",
                value,
            )
        ):
            claims.add("award_partial_scope")
    elif relation == "countryLandBordersCountry":
        has_land = bool(re.search(r"\bland border", value))
        has_maritime = bool(re.search(r"\bmaritime (?:border|boundary)", value))
        if has_land:
            claims.add("border_land")
        if has_maritime and not has_land:
            claims.add("border_maritime_only")
    elif relation == "personHasCityOfDeath":
        if re.search(r"\b(city of|died in|death (?:in|at))\b", value):
            claims.add("city_explicit")
        if (
            re.search(r"\b(province|state|region|country)\b", value)
            and not re.search(r"\b(city|town|municipality)\b", value)
        ):
            claims.add("city_noncity_scope")
    return sorted(claims)


def _component_lookup(
    graph: Mapping[str, Any],
) -> tuple[dict[str, str], set[str]]:
    relation = str(graph["Relation"])
    lookup: dict[str, str] = {}
    invalid = set()
    for component in graph["relational_graph"]["components"]:
        component_id = str(component["id"])
        keys = {
            str(value) for value in component.get("member_keys", [])
        } | {
            canonical_key(str(value), relation)
            for value in component.get("member_items", [])
        } | {
            canonical_key(str(component.get("representative", "")), relation)
        }
        if relation in {"hasArea", "hasCapacity"}:
            keys |= {
                value.removeprefix("numeric:")
                for value in list(keys)
                if value.startswith("numeric:")
            }
        for key in keys:
            if not key:
                continue
            if key in lookup and lookup[key] != component_id:
                invalid.add(key)
            lookup[key] = component_id
    if invalid:
        raise ContractError(
            f"{_key(graph)}: ambiguous component keys {sorted(invalid)}")
    return lookup, invalid


def _map_items(
    graph: Mapping[str, Any],
    items: Iterable[str],
) -> tuple[list[str], list[str]]:
    relation = str(graph["Relation"])
    lookup, _ = _component_lookup(graph)
    components: list[str] = []
    unknown: list[str] = []
    seen = set()
    for item in items:
        key = canonical_key(str(item), relation)
        # Historical numeric candidate tables retained a small number of
        # invalid/zero surfaces under their literal legacy key.  They remain
        # legal components in the frozen action inventory, so exact event
        # recovery must preserve their observed membership even though the
        # current positive-number canonicalizer intentionally rejects them.
        candidates = [key, str(item).strip(), normalize_string(str(item))]
        if key.startswith("numeric:"):
            candidates.append(key.removeprefix("numeric:"))
        component = next(
            (lookup[value] for value in candidates if value in lookup), None)
        if component is None:
            if key:
                unknown.append(str(item))
            continue
        if component not in seen:
            components.append(component)
            seen.add(component)
    return components, unknown


def _generic_record(
    graph: Mapping[str, Any], text: str,
) -> tuple[str, list[str], list[str]]:
    status, items = proposal_parse_status(str(text), str(graph["Relation"]))
    components, unknown = _map_items(graph, items)
    if unknown:
        raise ContractError(
            f"{_key(graph)}: generation has unmapped candidates {unknown}")
    mapped_status = (
        "candidate_set" if components else
        "explicit_none" if status == "explicit_none" else
        "unparsed_or_no_candidate"
    )
    return mapped_status, components, reasoning_claims(
        str(text), str(graph["Relation"]))


def _qwen_records(
    graph: Mapping[str, Any], samples: Sequence[str],
) -> list[tuple[str, list[str], list[str]]]:
    relation = str(graph["Relation"])
    evidence = classify_samples(samples, relation, "legacy-cot")
    records = []
    for raw, parsed in zip(samples, evidence):
        components, unknown = _map_items(graph, parsed.items)
        if unknown:
            raise ContractError(
                f"{_key(graph)}: Qwen generation has unmapped {unknown}")
        status = (
            "candidate_set" if components else
            "explicit_none"
            if parsed.kind == EXPLICIT_ABSTENTION else
            "unparsed_or_no_candidate"
        )
        records.append((
            status,
            components,
            reasoning_claims(raw, relation),
        ))
    return records


def _event_node(
    *, route: str, family: str, generation: int, samples: int,
    status: str, claims: Sequence[str], provenance: str, text: str,
) -> dict[str, Any]:
    return {
        "id": f"evidence:{route}:generation:{generation}",
        "node_type": "evidence_event",
        "evidence_kind": "exact_generation",
        "route": route,
        "model_family": family,
        "generation_index": generation,
        "samples": samples,
        "status": status,
        "claims": list(claims),
        "provenance_mode": provenance,
        "reasoning_sha256": hashlib.sha256(
            _reasoning(text).encode()).hexdigest(),
    }


def _replace_route_events(
    graph: dict[str, Any],
    *, route: str, family: str,
    records: Sequence[tuple[str, list[str], list[str]]],
    raw_texts: Sequence[str],
    provenance: str,
) -> dict[str, Any]:
    if len(records) != len(raw_texts):
        raise ContractError("event record/raw count mismatch")
    relational = graph["relational_graph"]
    old_nodes = list(relational["nodes"])
    removed_ids = {
        str(node["id"])
        for node in old_nodes
        if (
            node.get("node_type") == "evidence_event"
            and node.get("route") == route
        )
    }
    relational["nodes"] = [
        node for node in old_nodes if str(node.get("id")) not in removed_ids]
    relational["edges"] = [
        edge for edge in relational["edges"]
        if (
            str(edge.get("source")) not in removed_ids
            and str(edge.get("target")) not in removed_ids
        )
    ]
    samples = len(records)
    recovered: Counter[str] = Counter()
    for generation, ((status, components, claims), raw) in enumerate(
        zip(records, raw_texts)
    ):
        node = _event_node(
            route=route,
            family=family,
            generation=generation,
            samples=samples,
            status=status,
            claims=claims,
            provenance=provenance,
            text=raw,
        )
        relational["nodes"].append(node)
        for component in components:
            recovered[component] += 1
            metadata = next(
                value["routes"][route]
                for value in relational["components"]
                if (
                    str(value["id"]) == component
                    and route in value.get("routes", {})
                )
            )
            relational["edges"].append({
                "source": node["id"],
                "target": component,
                "edge_type": "supports",
                "evidence_kind": "exact_generation",
                "route": route,
                "model_family": family,
                "generation_index": generation,
                "weight": 1.0 / samples,
                "selected": bool(metadata.get("selected", False)),
            })

    # Exact component-union support may be greater than the historical maximum
    # surface support, but can never be less.  This catches wrong raw lineage.
    for component in relational["components"]:
        metadata = component.get("routes", {}).get(route)
        if metadata is None:
            continue
        lower_bound = int(math.ceil(
            float(metadata.get(
                "max_support_rate",
                metadata.get("component_support_rate", 0.0),
            )) * samples - 1e-9
        ))
        actual = recovered[str(component["id"])]
        if actual < lower_bound or actual > samples:
            raise ContractError(
                f"{_key(graph)}: {route} exact support violates "
                f"historical lower bound for {component['id']}: "
                f"{actual} not in [{lower_bound}, {samples}]")
    return {
        "events_replaced": len(removed_ids),
        "exact_events": samples,
        "support_edges": sum(recovered.values()),
        "support_union_increase": sum(
            max(
                0,
                recovered[str(component["id"])]
                - int(round(float(
                    component.get("routes", {}).get(route, {}).get(
                        "component_support_rate", 0.0)
                ) * samples)),
            )
            for component in relational["components"]
        ),
    }


def _attach_existing_claims(
    graph: dict[str, Any],
    *, route: str, raw_texts: Sequence[str],
) -> int:
    changed = 0
    for node in graph["relational_graph"]["nodes"]:
        if (
            node.get("node_type") != "evidence_event"
            or node.get("route") != route
            or node.get("evidence_kind") != "exact_generation"
        ):
            continue
        index = int(node["generation_index"])
        if index >= len(raw_texts):
            raise ContractError(f"{_key(graph)}: missing raw text for {route}")
        claims = reasoning_claims(
            raw_texts[index], str(graph["Relation"]))
        node["claims"] = claims
        node["reasoning_sha256"] = hashlib.sha256(
            _reasoning(raw_texts[index]).encode()).hexdigest()
        changed += 1
    return changed


def _state_and_relation_edges(graph: dict[str, Any]) -> dict[str, int]:
    # A candidate-set status without a mapped support edge is a parse/mapping
    # failure, not evidence for an empty answer.  Repair it before deriving
    # cardinality and existence assertions.
    _, repair = repair_unsupported_candidate_set_events(graph, in_place=True)
    relational = graph["relational_graph"]
    events = {
        str(node["id"]): node
        for node in relational["nodes"]
        if node.get("node_type") == "evidence_event"
    }
    supports: dict[str, set[str]] = {event: set() for event in events}
    for edge in relational["edges"]:
        if edge.get("edge_type") == "supports":
            supports[str(edge["source"])].add(str(edge["target"]))

    state_nodes = [
        {
            "id": f"cardinality:{value}",
            "node_type": "cardinality_state",
            "value": value,
        }
        for value in ("ZERO", "ONE", "MANY")
    ] + [
        {
            "id": f"existence:{value}",
            "node_type": "existence_state",
            "value": value,
        }
        for value in ("EMPTY", "NONEMPTY")
    ]
    claims = sorted({
        str(claim)
        for event in events.values()
        for claim in event.get("claims", [])
    })
    claim_nodes = [{
        "id": f"claim:{claim}",
        "node_type": "claim_state",
        "claim": claim,
        "risk": claim in RISK_CLAIMS,
    } for claim in claims]
    relational["nodes"].extend([*state_nodes, *claim_nodes])

    extra_edges: list[dict[str, Any]] = []
    pair_events: dict[tuple[str, str], list[str]] = defaultdict(list)
    parseable_events = 0
    for event_id, event in sorted(events.items()):
        status = str(event.get("status"))
        members = sorted(supports[event_id])
        if status not in ("candidate_set", "explicit_none"):
            continue
        parseable_events += 1
        cardinality = (
            "ZERO" if not members else "ONE" if len(members) == 1 else "MANY")
        existence = "EMPTY" if not members else "NONEMPTY"
        extra_edges.extend([
            {
                "source": event_id,
                "target": f"cardinality:{cardinality}",
                "edge_type": "asserts_cardinality",
            },
            {
                "source": event_id,
                "target": f"existence:{existence}",
                "edge_type": "asserts_existence",
            },
        ])
        for claim in event.get("claims", []):
            extra_edges.append({
                "source": event_id,
                "target": f"claim:{claim}",
                "edge_type": "asserts_claim",
                "risk": claim in RISK_CLAIMS,
            })
        for left, right in itertools.combinations(members, 2):
            pair_events[(left, right)].append(event_id)
    for (left, right), event_ids in sorted(pair_events.items()):
        family_counts = Counter(
            str(events[event]["model_family"]) for event in event_ids)
        route_counts = Counter(str(events[event]["route"]) for event in event_ids)
        extra_edges.append({
            "source": left,
            "target": right,
            "edge_type": "co_occurs_with",
            "directed": False,
            "count": len(event_ids),
            "rate": len(event_ids) / max(1, parseable_events),
            "event_ids": event_ids,
            "family_counts": dict(sorted(family_counts.items())),
            "route_counts": dict(sorted(route_counts.items())),
        })
    relational["edges"].extend(extra_edges)
    return {
        "unsupported_candidate_events_repaired": repair["repaired_events"],
        "parseable_events": parseable_events,
        "co_occurrence_edges": sum(
            edge["edge_type"] == "co_occurs_with"
            for edge in extra_edges
        ),
        "cardinality_edges": sum(
            edge["edge_type"] == "asserts_cardinality"
            for edge in extra_edges
        ),
        "existence_edges": sum(
            edge["edge_type"] == "asserts_existence"
            for edge in extra_edges
        ),
        "claim_nodes": len(claim_nodes),
        "claim_edges": sum(
            edge["edge_type"] == "asserts_claim"
            for edge in extra_edges
        ),
    }


def enrich_graph(
    source: Mapping[str, Any],
    qwen_samples: Sequence[str],
    gemma_base: Mapping[str, Any],
    gemma_extra: Sequence[str] | None,
    ministral: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = minimalize_row(source)
    graph["schema"] = ROW_SCHEMA
    graph["relational_graph"]["schema"] = GRAPH_SCHEMA
    audit: dict[str, Any] = {}

    qwen_records = _qwen_records(graph, qwen_samples)
    audit["qwen"] = _replace_route_events(
        graph,
        route=ROUTE_QWEN,
        family="qwen_recall",
        records=qwen_records,
        raw_texts=qwen_samples,
        provenance="immutable_production_raw",
    )

    base_generations = [
        str(value) for value in gemma_base.get("generations", [])]
    if len(base_generations) != 1:
        raise ContractError(f"{_key(graph)}: Gemma N=1 response invalid")
    if gemma_extra is not None:
        gemma_texts = [*base_generations, *gemma_extra]
        gemma_records = [
            _generic_record(graph, value) for value in gemma_texts]
        audit["gemma"] = _replace_route_events(
            graph,
            route=ROUTE_GEMMA,
            family="gemma_independent",
            records=gemma_records,
            raw_texts=gemma_texts,
            provenance="immutable_n1_plus_targeted_extra2_raw",
        )
    else:
        audit["gemma_claim_events"] = _attach_existing_claims(
            graph, route=ROUTE_GEMMA, raw_texts=base_generations)

    ministral_texts = [
        str(value) for value in ministral.get("generations", [])]
    if len(ministral_texts) != 10:
        raise ContractError(f"{_key(graph)}: Ministral N=10 response invalid")
    audit["ministral_claim_events"] = _attach_existing_claims(
        graph, route=ROUTE_MINISTRAL, raw_texts=ministral_texts)

    audit["typed_edges"] = _state_and_relation_edges(graph)
    relational = graph["relational_graph"]
    event_nodes = [
        value for value in relational["nodes"]
        if value.get("node_type") == "evidence_event"
    ]
    if any(
        value.get("evidence_kind") != "exact_generation"
        for value in event_nodes
    ):
        raise ContractError(f"{_key(graph)}: aggregate event survived recovery")
    support_edge_count = sum(
        value.get("edge_type") == "supports"
        for value in relational["edges"]
    )
    graph["minimal_evidence_contract"].update({
        "exact_event_count": len(event_nodes),
        "aggregate_event_count": 0,
        "final_support_edge_count": support_edge_count,
        "all_sampled_routes_have_exact_events": True,
    })
    graph["typed_evidence_contract"] = {
        "all_sampled_routes_have_exact_events": True,
        "raw_reasoning_not_stored": True,
        "claim_extraction_is_label_free_lexical": True,
        "edge_types": sorted({
            str(value["edge_type"]) for value in relational["edges"]}),
        "node_types": sorted({
            str(value["node_type"]) for value in relational["nodes"]}),
    }
    return graph, audit


def _typed_index(
    graph: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, float]],
    dict[tuple[str, str], float],
    dict[str, str],
    dict[str, str],
    dict[str, set[str]],
]:
    relational = graph["relational_graph"]
    events = {
        str(node["id"]): node
        for node in relational["nodes"]
        if node.get("node_type") == "evidence_event"
    }
    support = {event: {} for event in events}
    co_occurrence: dict[tuple[str, str], float] = {}
    cardinality: dict[str, str] = {}
    existence: dict[str, str] = {}
    claims: dict[str, set[str]] = {event: set() for event in events}
    allowed = {
        "supports",
        "co_occurs_with",
        "asserts_cardinality",
        "asserts_existence",
        "asserts_claim",
    }
    for edge in relational["edges"]:
        edge_type = str(edge["edge_type"])
        if edge_type not in allowed:
            raise ContractError(f"unexpected typed edge: {edge_type}")
        source, target = str(edge["source"]), str(edge["target"])
        if edge_type == "supports":
            if source not in events or target in support[source]:
                raise ContractError("invalid/duplicate supports edge")
            support[source][target] = float(edge["weight"])
        elif edge_type == "co_occurs_with":
            key = tuple(sorted((source, target)))
            if key in co_occurrence:
                raise ContractError("duplicate co-occurrence edge")
            co_occurrence[key] = float(edge["rate"])
        elif edge_type == "asserts_cardinality":
            cardinality[source] = target.removeprefix("cardinality:")
        elif edge_type == "asserts_existence":
            existence[source] = target.removeprefix("existence:")
        else:
            claims[source].add(target.removeprefix("claim:"))
    return events, support, co_occurrence, cardinality, existence, claims


def _support_view(graph: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(graph)
    relational = graph["relational_graph"]
    value["relational_graph"] = {
        "schema": "support-feature-view-v1",
        "relation": graph["Relation"],
        "components": relational["components"],
        "nodes": [
            node for node in relational["nodes"]
            if node.get("node_type") in (
                "candidate_component", "evidence_event")
        ],
        "edges": [
            edge for edge in relational["edges"]
            if edge.get("edge_type") == "supports"
        ],
    }
    return value


def _overlap(
    selected: set[str], supported: set[str],
) -> tuple[float, float]:
    union = selected | supported
    intersection = selected & supported
    return (
        len(intersection) / len(union) if union else 1.0,
        len(intersection) / len(selected) if selected else float(not supported),
    )


def typed_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
) -> list[float]:
    cached = action.get("_typed_edge_features")
    if cached is not None:
        if len(cached) != len(TYPED_NAMES):
            raise ContractError("typed feature cache schema drift")
        return list(cached)
    selected = {str(value) for value in action["component_ids"]}
    (
        events,
        support,
        co_occurrence,
        cardinality,
        existence,
        claims,
    ) = _typed_index(graph)

    support_values = list(event_features(_support_view(graph), action))
    family_values = []
    for family in FAMILIES:
        records = []
        coverage = set()
        for event_id, node in events.items():
            if (
                str(node.get("model_family")) != family
                or str(node.get("status"))
                    not in ("candidate_set", "explicit_none")
            ):
                continue
            supported = set(support[event_id])
            jaccard, recall = _overlap(selected, supported)
            records.append((jaccard, float(supported == selected)))
            coverage |= supported
        family_values.extend([
            (
                math.fsum(value[0] for value in records) / len(records)
                if records else 0.0
            ),
            (
                math.fsum(value[1] for value in records) / len(records)
                if records else 0.0
            ),
            (
                len(selected & coverage) / len(selected)
                if selected else float(any(
                    str(events[event].get("status")) == "explicit_none"
                    for event in events
                    if str(events[event].get("model_family")) == family
                ))
            ),
        ])
    support_values.extend(family_values)

    selected_pairs = [
        tuple(sorted(value))
        for value in itertools.combinations(sorted(selected), 2)]
    selected_pair_rates = [
        co_occurrence.get(value, 0.0) for value in selected_pairs]
    all_mass = math.fsum(co_occurrence.values())
    selected_mass = boundary_mass = omitted_mass = 0.0
    for pair, weight in co_occurrence.items():
        overlap_count = len(set(pair) & selected)
        if overlap_count == 2:
            selected_mass += weight
        elif overlap_count == 1:
            boundary_mass += weight
        else:
            omitted_mass += weight
    co_values = [
        (
            math.fsum(selected_pair_rates) / len(selected_pair_rates)
            if selected_pair_rates else 0.0
        ),
        max(selected_pair_rates, default=0.0),
        min(selected_pair_rates, default=0.0),
        (
            sum(value > 0.0 for value in selected_pair_rates)
            / len(selected_pair_rates)
            if selected_pair_rates else 0.0
        ),
        selected_mass / all_mass if all_mass else 0.0,
        boundary_mass / all_mass if all_mass else 0.0,
        omitted_mass / all_mass if all_mass else 0.0,
    ]

    action_cardinality = (
        "ZERO" if not selected else "ONE" if len(selected) == 1 else "MANY")
    cardinality_records = [
        (event, value)
        for event, value in cardinality.items()
        if event in events
    ]
    cardinality_values = [
        (
            sum(value == action_cardinality for _, value in cardinality_records)
            / len(cardinality_records)
            if cardinality_records else 0.0
        ),
        (
            math.fsum(
                abs(
                    {"ZERO": 0, "ONE": 1, "MANY": 2}[value]
                    - {"ZERO": 0, "ONE": 1, "MANY": 2}[action_cardinality]
                ) / 2.0
                for _, value in cardinality_records
            ) / len(cardinality_records)
            if cardinality_records else 0.0
        ),
    ]
    for family in FAMILIES:
        values = [
            value for event, value in cardinality_records
            if str(events[event].get("model_family")) == family]
        cardinality_values.append(
            sum(value == action_cardinality for value in values) / len(values)
            if values else 0.0
        )

    action_existence = "EMPTY" if not selected else "NONEMPTY"
    existence_records = [
        (event, value)
        for event, value in existence.items()
        if event in events
    ]
    existence_match = [
        value == action_existence for _, value in existence_records]
    existence_values = [
        (
            sum(existence_match) / len(existence_match)
            if existence_match else 0.0
        ),
        (
            1.0 - sum(existence_match) / len(existence_match)
            if existence_match else 0.0
        ),
    ]
    for family in FAMILIES:
        values = [
            value for event, value in existence_records
            if str(events[event].get("model_family")) == family]
        existence_values.append(
            sum(value == action_existence for value in values) / len(values)
            if values else 0.0
        )

    total_selected = clean_selected = risk_selected = risk_omitted = 0.0
    risky_events = positive_events = parseable = 0
    by_risk = Counter()
    for event_id, node in events.items():
        if str(node.get("status")) not in (
            "candidate_set", "explicit_none"):
            continue
        parseable += 1
        event_claims = claims[event_id]
        risks = event_claims & set(RISK_CLAIMS)
        positives = event_claims & set(POSITIVE_CLAIMS)
        risky_events += int(bool(risks))
        positive_events += int(bool(positives))
        edges = support[event_id]
        event_selected = math.fsum(
            value for component, value in edges.items()
            if component in selected
        )
        event_omitted = math.fsum(
            value for component, value in edges.items()
            if component not in selected
        )
        total_selected += event_selected
        if risks:
            risk_selected += event_selected
            risk_omitted += event_omitted
            for risk in risks:
                by_risk[risk] += event_selected
        else:
            clean_selected += event_selected
    denominator = max(total_selected, 1e-12)
    claim_values = [
        clean_selected / denominator,
        risk_selected / denominator,
        risk_omitted / max(
            risk_selected + risk_omitted, 1e-12),
        risky_events / parseable if parseable else 0.0,
        positive_events / parseable if parseable else 0.0,
        *[
            by_risk[claim] / denominator
            for claim in RISK_CLAIMS
        ],
    ]
    values = [
        *support_values,
        *co_values,
        *cardinality_values,
        *existence_values,
        *claim_values,
    ]
    if (
        len(values) != len(TYPED_NAMES)
        or not all(math.isfinite(value) for value in values)
    ):
        raise ContractError("invalid typed edge feature vector")
    action["_typed_edge_features"] = list(values)
    return values


def _arm_names(arm: str) -> tuple[str, ...]:
    if arm == "component_table":
        extra: tuple[str, ...] = ()
    elif arm == "exact_support":
        extra = tuple(SUPPORT_NAMES)
    elif arm == "exact_support_cooccurrence":
        extra = (*SUPPORT_NAMES, *COOCCURRENCE_NAMES)
    elif arm == "exact_support_cardinality":
        extra = (*SUPPORT_NAMES, *CARDINALITY_NAMES)
    elif arm == "exact_support_existence":
        extra = (*SUPPORT_NAMES, *EXISTENCE_NAMES)
    elif arm == "exact_support_claims":
        extra = (*SUPPORT_NAMES, *CLAIM_NAMES)
    elif arm in ("all_typed_edges", "all_typed_edges_shifted"):
        extra = tuple(TYPED_NAMES)
    else:
        raise ContractError(f"unknown typed edge arm: {arm}")
    return (*LOCAL_NAMES, *extra)


def _arm_values(action: Mapping[str, Any], arm: str) -> list[float]:
    if arm == "component_table":
        return []
    field = (
        "_typed_edge_features_shifted"
        if arm == "all_typed_edges_shifted"
        else "_typed_edge_features"
    )
    values = action.get(field)
    if not isinstance(values, list) or len(values) != len(TYPED_NAMES):
        raise ContractError(f"{arm}: typed feature cache not prepared")
    slices = {
        "exact_support": (0, len(SUPPORT_NAMES)),
        "exact_support_cooccurrence": (
            0, len(SUPPORT_NAMES) + len(COOCCURRENCE_NAMES)),
    }
    if arm in slices:
        left, right = slices[arm]
        return list(values[left:right])
    offset = len(SUPPORT_NAMES)
    co_end = offset + len(COOCCURRENCE_NAMES)
    card_end = co_end + len(CARDINALITY_NAMES)
    existence_end = card_end + len(EXISTENCE_NAMES)
    if arm == "exact_support_cardinality":
        return [
            *values[:offset],
            *values[co_end:card_end],
        ]
    if arm == "exact_support_existence":
        return [
            *values[:offset],
            *values[card_end:existence_end],
        ]
    if arm == "exact_support_claims":
        return [
            *values[:offset],
            *values[existence_end:],
        ]
    return list(values)


def action_features(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    action: Mapping[str, Any],
    arm: str,
) -> list[float]:
    values = [
        *component_features(graph, incumbent, action),
        *_arm_values(action, arm),
    ]
    if (
        len(values) != len(_arm_names(arm))
        or not all(math.isfinite(value) for value in values)
    ):
        raise ContractError(f"{arm}: invalid action feature vector")
    return values


def _prepare_feature_cache(
    graphs: Sequence[dict[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
) -> dict[str, Any]:
    strata: dict[
        tuple[str, str],
        list[tuple[dict[str, Any], dict[str, Any], list[float]]],
    ] = defaultdict(list)
    total = 0
    for graph in sorted(graphs, key=_key):
        incumbent = controls[_key(graph)]
        for action in legal_actions(graph, incumbent):
            values = typed_features(graph, action)
            strata[(
                str(graph["Relation"]),
                str(action["action_type"]),
            )].append((graph, action, values))
            total += 1
    changed = same_subject = 0
    for stratum in sorted(strata):
        values = sorted(
            strata[stratum],
            key=lambda item: (
                str(item[0]["SubjectEntity"]),
                tuple(str(value) for value in item[1]["component_ids"]),
            ),
        )
        size = len(values)
        offset = 0
        for candidate in range(1, size):
            if all(
                str(values[index][0]["SubjectEntity"])
                != str(values[(index + candidate) % size][0][
                    "SubjectEntity"])
                for index in range(size)
            ):
                offset = candidate
                break
        if size > 1 and offset == 0:
            offset = max(1, size // 2)
        for index, (graph, action, original) in enumerate(values):
            source_graph, _, shifted = values[(index + offset) % size]
            action["_typed_edge_features_shifted"] = list(shifted)
            changed += int(original != shifted)
            same_subject += int(
                str(graph["SubjectEntity"])
                == str(source_graph["SubjectEntity"])
            )
    return {
        "actions": total,
        "strata": len(strata),
        "changed_vectors": changed,
        "same_subject_assignments": same_subject,
    }


def _fit_model(
    graphs: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    arm: str,
    l2: float,
) -> ResidualRidge:
    x, y, weights = [], [], []
    for graph in graphs:
        key = _key(graph)
        incumbent = list(controls[key])
        actions = legal_actions(graph, incumbent)
        baseline = _row_f1(incumbent, gold[key], key[1])
        row_weight = 1.0 / len(actions)
        for action in actions:
            x.append(action_features(graph, incumbent, action, arm))
            y.append(
                _row_f1(action["objects"], gold[key], key[1]) - baseline)
            weights.append(row_weight)
    return ResidualRidge(_arm_names(arm), l2).fit(x, y, weights)


def _propose(
    model: ResidualRidge,
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    arm: str,
) -> tuple[list[str], float, str, int]:
    actions = legal_actions(graph, incumbent)
    estimates = model.predict([
        action_features(graph, incumbent, action, arm)
        for action in actions
    ])
    keep = next(
        index for index, action in enumerate(actions)
        if action["action_type"] == "KEEP"
    )
    best = max(range(len(actions)), key=lambda index: (
        float(estimates[index]),
        actions[index]["action_type"] == "KEEP",
        -len(actions[index]["objects"]),
        -index,
    ))
    return (
        list(actions[best]["objects"]),
        float(estimates[best] - estimates[keep]),
        str(actions[best]["action_type"]),
        len(actions),
    )


def _nested_oof(
    graphs: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    folds: Mapping[tuple[str, str], int],
    arm: str,
) -> dict[str, Any]:
    control_by = {_key(row): row for row in control_rows}
    fold_ids = sorted(set(folds.values()))
    oof = {}
    diagnostics = []
    for outer in fold_ids:
        print(f"{arm}: outer fold {outer + 1}/{len(fold_ids)}", flush=True)
        outer_fit = [row for row in graphs if folds[_key(row)] != outer]
        outer_hold = [row for row in graphs if folds[_key(row)] == outer]
        fit_keys = {_key(row) for row in outer_fit}
        inner_controls = [control_by[key] for key in sorted(fit_keys)]
        candidates = []
        for l2 in L2_VALUES:
            inner_proposals = {}
            for inner in fold_ids:
                if inner == outer:
                    continue
                model = _fit_model(
                    [
                        row for row in outer_fit
                        if folds[_key(row)] != inner
                    ],
                    controls,
                    gold,
                    arm,
                    l2,
                )
                for graph in outer_fit:
                    if folds[_key(graph)] != inner:
                        continue
                    key = _key(graph)
                    inner_proposals[key] = _propose(
                        model, graph, controls[key], arm)
            if set(inner_proposals) != fit_keys:
                raise ContractError("inner OOF coverage mismatch")
            for guard in GUARDS:
                scores, audit, _ = _score_predictions(
                    inner_controls, inner_proposals, guard, gold)
                candidates.append((scores[POOLED], guard, l2, scores, audit))
        _, guard, l2, inner_scores, inner_audit = max(
            candidates, key=lambda value: (value[0], value[1], value[2]))
        model = _fit_model(outer_fit, controls, gold, arm, l2)
        hold_proposals = {
            _key(graph): _propose(
                model, graph, controls[_key(graph)], arm)
            for graph in outer_hold
        }
        hold_controls = [control_by[_key(row)] for row in outer_hold]
        hold_scores, hold_audit, _ = _score_predictions(
            hold_controls, hold_proposals, guard, gold)
        for key, (proposal, advantage, action_type, count) in (
            hold_proposals.items()
        ):
            oof[key] = (
                proposal if advantage > guard else list(controls[key]),
                1.0 if advantage > guard else 0.0,
                action_type,
                count,
            )
        diagnostics.append({
            "outer_fold": outer,
            "selected_l2": l2,
            "selected_guard": guard,
            "inner_scores": inner_scores,
            "inner_audit": inner_audit,
            "hold_scores": hold_scores,
            "hold_audit": hold_audit,
        })
    if set(oof) != {_key(row) for row in graphs}:
        raise ContractError("outer OOF coverage mismatch")
    scores, audit, decisions = _score_predictions(
        control_rows, oof, 0.5, gold)
    return {
        "scores": scores,
        "audit": audit,
        "decisions": decisions,
        "fold_diagnostics": diagnostics,
        "fold_scores": [
            value["hold_scores"][POOLED] for value in diagnostics],
    }


def prepare(args: argparse.Namespace) -> int:
    source_run = Path(args.source_run).resolve()
    output = Path(args.output_dir).resolve()
    source_plan, source_rows = _validate_prepared(source_run)
    base_rows, _ = compose_competition_train_oof()
    controls = {
        _key(row): list(row.get("ObjectEntities", []))
        for row in base_rows
    }
    qwen, qwen_sources = _qwen_raw_sources()
    gemma, gemma_extra, gemma_sources = _gemma_sources()
    ministral, ministral_sources = _ministral_sources(source_plan)
    keys = {_key(row) for row in source_rows}
    if not (
        keys == set(qwen)
        == set(gemma)
        == set(ministral)
        and set(gemma_extra) <= keys
    ):
        raise ContractError("raw evidence/source graph coverage mismatch")

    graphs = []
    recovery = {
        "qwen_support_union_increase": 0,
        "gemma_support_union_increase": 0,
        "qwen_exact_events": 0,
        "gemma_exact_events": 0,
        "typed_edges": Counter(),
    }
    for source in source_rows:
        key = _key(source)
        graph, audit = enrich_graph(
            source,
            qwen[key],
            gemma[key],
            gemma_extra.get(key),
            ministral[key],
        )
        graphs.append(graph)
        recovery["qwen_support_union_increase"] += int(
            audit["qwen"]["support_union_increase"])
        recovery["qwen_exact_events"] += int(
            audit["qwen"]["exact_events"])
        if isinstance(audit.get("gemma"), Mapping):
            recovery["gemma_support_union_increase"] += int(
                audit["gemma"]["support_union_increase"])
            recovery["gemma_exact_events"] += int(
                audit["gemma"]["exact_events"])
        else:
            recovery["gemma_exact_events"] += int(
                audit["gemma_claim_events"])
        recovery["typed_edges"].update(audit["typed_edges"])

    parity = parity_audit(source_rows, graphs, controls)
    graph_path = output / "graph/TYPED_MINIMAL_EVIDENCE_GRAPH.jsonl"
    write_jsonl_atomic(graph_path, graphs)
    edge_counts = Counter(
        str(edge["edge_type"])
        for graph in graphs
        for edge in graph["relational_graph"]["edges"]
    )
    node_counts = Counter(
        str(node["node_type"])
        for graph in graphs
        for node in graph["relational_graph"]["nodes"]
    )
    parity.update({
        "exact_generation_events": node_counts["evidence_event"],
        "aggregate_route_events": 0,
        "support_edges": edge_counts["supports"],
    })
    source_artifacts = {
        "qwen": qwen_sources,
        "gemma": gemma_sources,
        "ministral": ministral_sources,
    }
    manifest_path = graph_path.with_suffix(
        graph_path.suffix + ".manifest.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "rows": len(graphs),
        "node_counts": dict(sorted(node_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
        "source_graph": source_plan["graph"],
        "source_graph_sha256": sha256(Path(source_plan["graph"])),
        "source_artifacts": source_artifacts,
        "output": str(graph_path),
        "output_sha256": sha256(graph_path),
        "parity": parity,
        "recovery": {
            **{key: value for key, value in recovery.items()
               if key != "typed_edges"},
            "typed_edges": dict(sorted(recovery["typed_edges"].items())),
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(manifest_path, manifest)
    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "source_run": str(source_run),
        "source_plan": str(source_run / "plan/PLAN.json"),
        "source_plan_sha256": sha256(source_run / "plan/PLAN.json"),
        "source_graph": source_plan["graph"],
        "source_graph_sha256": sha256(Path(source_plan["graph"])),
        "prior_minimal_run": str(Path(args.minimal_run).resolve()),
        "typed_graph": str(graph_path),
        "typed_graph_sha256": sha256(graph_path),
        "typed_manifest": str(manifest_path),
        "typed_manifest_sha256": sha256(manifest_path),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "arms": list(ARMS),
        "component_feature_names": list(LOCAL_NAMES),
        "typed_feature_names": list(TYPED_NAMES),
        "feature_families": {
            "supports": list(SUPPORT_NAMES),
            "co_occurs_with": list(COOCCURRENCE_NAMES),
            "asserts_cardinality": list(CARDINALITY_NAMES),
            "asserts_existence": list(EXISTENCE_NAMES),
            "asserts_claim": list(CLAIM_NAMES),
        },
        "l2_values": list(L2_VALUES),
        "guards": list(GUARDS),
        "folding": "strict_subject_grouped_5_fold_nested_cv",
        "negative_control":
            "deterministic subject shift within relation/action_type",
        "parity": parity,
        "recovery": manifest["recovery"],
        "source_artifacts": source_artifacts,
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, plan)
    print(json.dumps({
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "typed_graph": str(graph_path),
        "typed_graph_sha256": sha256(graph_path),
        "parity": parity,
        "recovery": manifest["recovery"],
        "node_counts": dict(sorted(node_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _json(output / "plan/PLAN.json")
    required = (
        ("source_plan", "source_plan_sha256"),
        ("source_graph", "source_graph_sha256"),
        ("typed_graph", "typed_graph_sha256"),
        ("typed_manifest", "typed_manifest_sha256"),
        ("implementation", "implementation_sha256"),
    )
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or plan.get("arms") != list(ARMS)
        or not all(
            sha256(Path(plan[path_field])) == plan[hash_field]
            for path_field, hash_field in required
        )
    ):
        raise ContractError("typed evidence plan contract failed")
    graph_path = Path(plan["typed_graph"])
    rows = read_jsonl(graph_path)
    manifest = _json(Path(plan["typed_manifest"]))
    if (
        len(rows) != 477
        or len({_key(row) for row in rows}) != 477
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("contains_labels") is not False
        or manifest.get("output_sha256") != plan["typed_graph_sha256"]
        or manifest.get("parity", {}).get("parity_passed") is not True
        or any(
            not row.get("typed_evidence_contract", {}).get(
                "all_sampled_routes_have_exact_events")
            for row in rows
        )
    ):
        raise ContractError("typed evidence graph contract failed")
    return plan, rows


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan, graphs = _validate_plan(output)
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold = {_key(row): row for row in gold_rows}
    base_rows, base_detail = compose_competition_train_oof()
    base_controls = {
        _key(row): list(row.get("ObjectEntities", []))
        for row in base_rows
    }
    graph_by = {_key(row): row for row in graphs}
    if (
        len(gold) != 477
        or set(gold) != set(base_controls)
        or set(gold) != set(graph_by)
    ):
        raise ContractError("typed evidence analysis coverage mismatch")
    control_rows = [{
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
        "ObjectEntities": cot40_count_anchor(
            graph_by[_key(row)], base_controls[_key(row)]),
    } for row in base_rows]
    controls = {
        _key(row): list(row["ObjectEntities"]) for row in control_rows}
    for graph in graphs:
        graph["baseline_objects"] = list(controls[_key(graph)])
        graph.pop("_graph_native_action_cache", None)
    shift_audit = _prepare_feature_cache(graphs, controls)
    folds = subject_grouped_folds(graphs)
    write_jsonl_atomic(
        output / "analysis/SUBJECT_GROUPED_FOLDS.jsonl",
        [{
            "SubjectEntity": key[0],
            "Relation": key[1],
            "fold": fold,
        } for key, fold in sorted(folds.items())],
    )
    incumbent_scores = score(control_rows, gold_rows)
    reports = {
        arm: _nested_oof(
            graphs, control_rows, controls, gold, folds, arm)
        for arm in ARMS
    }
    for arm, report in reports.items():
        replacements = {
            (str(value["SubjectEntity"]), str(value["Relation"])):
                value["selected"]
            for value in report["decisions"]
        }
        path = output / f"analysis/{arm.upper()}_OOF.jsonl"
        write_jsonl_atomic(
            path, _prediction_rows(control_rows, replacements))
        report["predictions"] = str(path)
        report["predictions_sha256"] = sha256(path)

    component = reports["component_table"]
    exact = reports["exact_support"]
    shifted = reports["all_typed_edges_shifted"]
    comparisons = {}
    for arm, report in reports.items():
        fold_delta_component = [
            right - left for left, right in zip(
                component["fold_scores"], report["fold_scores"])
        ]
        fold_delta_exact = [
            right - left for left, right in zip(
                exact["fold_scores"], report["fold_scores"])
        ]
        comparisons[arm] = {
            "delta_vs_component_table":
                report["scores"][POOLED] - component["scores"][POOLED],
            "delta_vs_exact_support":
                report["scores"][POOLED] - exact["scores"][POOLED],
            "delta_vs_incumbent":
                report["scores"][POOLED] - incumbent_scores[POOLED],
            "fold_deltas_vs_component_table": fold_delta_component,
            "fold_wins_vs_component_table": sum(
                value > 1e-12 for value in fold_delta_component),
            "fold_deltas_vs_exact_support": fold_delta_exact,
            "fold_wins_vs_exact_support": sum(
                value > 1e-12 for value in fold_delta_exact),
            "relation_deltas_vs_exact_support": {
                relation:
                    report["scores"][relation] - exact["scores"][relation]
                for relation in RELATIONS
            },
            "paired_audit_vs_exact_support":
                _paired_audit(exact, report, gold),
        }
    all_report = reports["all_typed_edges"]
    all_comparison = comparisons["all_typed_edges"]
    paired = all_comparison["paired_audit_vs_exact_support"]
    gate_passed = bool(
        all_comparison["delta_vs_exact_support"] >= MIN_INCREMENT
        and all_comparison["fold_wins_vs_exact_support"] >= MIN_FOLD_WINS
        and min(
            all_comparison["relation_deltas_vs_exact_support"].values()
        ) >= MAX_RELATION_REGRESSION
        and paired["helped"] > paired["harmed"]
        and (
            all_report["scores"][POOLED] - shifted["scores"][POOLED]
            >= MIN_ALIGNED_OVER_SHIFTED
        )
    )
    marginal_arms = (
        "exact_support_cooccurrence",
        "exact_support_cardinality",
        "exact_support_existence",
        "exact_support_claims",
    )
    edge_family_ranking = sorted(
        marginal_arms,
        key=lambda arm: (
            comparisons[arm]["delta_vs_exact_support"],
            comparisons[arm]["fold_wins_vs_exact_support"],
        ),
        reverse=True,
    )
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
        "typed_graph": plan["typed_graph"],
        "typed_graph_sha256": plan["typed_graph_sha256"],
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "incumbent_detail": base_detail,
        "incumbent_scores": incumbent_scores,
        "shift_audit": shift_audit,
        "arms": reports,
        "comparisons": comparisons,
        "edge_family_ranking": edge_family_ranking,
        "all_typed_edges_gate_passed": gate_passed,
        "gate": {
            "minimum_increment_over_exact_support": MIN_INCREMENT,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "maximum_relation_regression": MAX_RELATION_REGRESSION,
            "minimum_aligned_over_shifted": MIN_ALIGNED_OVER_SHIFTED,
            "helpful_edits_must_exceed_harmful": True,
        },
        "methodology":
            "label-free exact provenance reconstruction and typed edge "
            "materialization; one-edge-family-at-a-time matched strict-"
            "subject nested OOF ablation with shifted-evidence control",
        "next_stage": (
            "freeze_successful_edge_families_for_separate_confirmation"
            if gate_passed else
            "retain_only_edge_families_with_repeatable_marginal_signal"
        ),
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)
    lines = [
        "# CoT40 evidence-edge ablation",
        "",
        "Train-only nested subject-grouped audit. Validation was not opened.",
        "",
        "| arm | OOF F1 | vs component | vs exact support | folds vs exact | helped | harmed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        ARMS,
        key=lambda arm: reports[arm]["scores"][POOLED],
        reverse=True,
    )
    for arm in ranked:
        comparison = comparisons[arm]
        audit = comparison["paired_audit_vs_exact_support"]
        lines.append(
            f"| {arm} | {reports[arm]['scores'][POOLED]:.6f} | "
            f"{comparison['delta_vs_component_table']:+.6f} | "
            f"{comparison['delta_vs_exact_support']:+.6f} | "
            f"{comparison['fold_wins_vs_exact_support']}/5 | "
            f"{audit['helped']} | {audit['harmed']} |"
        )
    lines.extend([
        "",
        f"- Incumbent: **{incumbent_scores[POOLED]:.6f}**",
        f"- Exact Qwen events recovered: "
        f"**{plan['recovery']['qwen_exact_events']}**",
        f"- Exact Gemma events represented: "
        f"**{plan['recovery']['gemma_exact_events']}**",
        f"- Qwen component-union support recovered beyond old maximum-"
        f"surface summaries: "
        f"**{plan['recovery']['qwen_support_union_increase']}**",
        f"- Edge-family ranking: `{', '.join(edge_family_ranking)}`",
        f"- All-edge gate passed: **{gate_passed}**",
        f"- Next stage: `{result['next_stage']}`",
    ])
    result_md = output / "analysis/RESULT.md"
    result_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "result": str(result_path),
        "result_md": str(result_md),
        "incumbent": incumbent_scores[POOLED],
        "component_table": component["scores"][POOLED],
        "exact_support": exact["scores"][POOLED],
        "all_typed_edges": all_report["scores"][POOLED],
        "all_typed_edges_shifted": shifted["scores"][POOLED],
        "edge_family_ranking": edge_family_ranking,
        "all_typed_edges_gate_passed": gate_passed,
        "validation_opened": False,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source-run", default=str(DEFAULT_SOURCE))
    prepare_parser.add_argument(
        "--minimal-run", default=str(DEFAULT_MINIMAL))
    prepare_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.set_defaults(function=prepare)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT))
    analyze_parser.add_argument(
        "--train-gold", default=str(DEFAULT_GOLD))
    analyze_parser.set_defaults(function=analyze)
    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
