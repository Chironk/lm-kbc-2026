#!/usr/bin/env python3
"""Assemble the anonymized claim graph and audit portfolio complementarity.

Gold labels are optional and are used only by the final audit section. They
are never written into task, response, claim-graph, or review-task artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from evaluate import (RELATION_TYPE, evaluate_per_sr_pair,
                      macro_average_per_relation, true_positives)
from experiments.heterogeneous_agents.core import (
    ContractError, NUMERIC_RELATIONS, NULLABLE_RELATIONS, SINGLE_RELATIONS,
    build_review_tasks, canonical_key, load_agent_config, proposal_candidates,
    proposal_parse_status, read_jsonl, sha256, validate_inputs, validate_task_response,
    write_jsonl_atomic,
)


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return row["SubjectEntity"], row["Relation"]


def _response_files(response_dir: Path, agents: Sequence[Mapping[str, Any]]) -> dict:
    return {agent["id"]: response_dir / f"{agent['id']}.jsonl" for agent in agents}


def load_responses(response_dir: Path, agents: Sequence[Mapping[str, Any]]) -> dict:
    by_agent = {}
    for agent_id, path in _response_files(response_dir, agents).items():
        if not path.is_file():
            raise ContractError(f"missing response file for {agent_id}: {path}")
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise ContractError(f"missing response manifest for {agent_id}: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid response manifest {manifest_path}: {exc}") from exc
        if manifest.get("agent_id") != agent_id:
            raise ContractError(f"{manifest_path}: agent id mismatch")
        if manifest.get("output_sha256") != sha256(path):
            raise ContractError(f"{manifest_path}: response hash mismatch")
        rows = read_jsonl(path)
        keyed = {}
        for row in rows:
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or task_id in keyed:
                raise ContractError(f"{path}: invalid/duplicate task_id {task_id!r}")
            if row.get("agent_id") != agent_id:
                raise ContractError(f"{path}: foreign agent response {task_id}")
            keyed[task_id] = row
        by_agent[agent_id] = keyed
    return by_agent


def _phase(response_map: Mapping[str, dict], subject: str, relation: str,
           phase: str) -> dict:
    matches = [row for row in response_map.values()
               if row.get("subject") == subject and row.get("relation") == relation
               and row.get("phase") == phase]
    if len(matches) != 1:
        raise ContractError(
            f"expected one {phase} response for {(subject, relation)}, got {len(matches)}")
    return matches[0]


def assemble_graphs(inputs: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]],
                    responses: Mapping[str, Mapping[str, dict]]) -> List[dict]:
    validate_inputs(inputs)
    graphs = []
    for index, source in enumerate(inputs):
        subject, relation = source["SubjectEntity"], source["Relation"]
        commitments = {}
        candidates: Dict[str, dict] = {}
        proposal_sample_counts = {}
        proposal_parse_diagnostics = {}
        for agent in agents:
            agent_id = agent["id"]
            response_map = responses[agent_id]
            existence = _phase(response_map, subject, relation, "commit_existence")
            cardinality = _phase(response_map, subject, relation, "commit_cardinality")
            proposal = _phase(response_map, subject, relation, "propose")
            proposal_sample_counts[agent_id] = len(proposal.get("generations", []))
            parse_counts = Counter(
                proposal_parse_status(str(generation), relation)[0]
                for generation in proposal.get("generations", []))
            proposal_parse_diagnostics[agent_id] = dict(parse_counts)
            commitments[agent_id] = {
                "existence": existence["selected_choice"],
                "existence_probabilities": existence.get("choice_probabilities", {}),
                "cardinality": cardinality["selected_choice"],
                "cardinality_probabilities": cardinality.get("choice_probabilities", {}),
            }
            for candidate in proposal_candidates(proposal):
                node = candidates.setdefault(candidate["key"], {
                    "key": candidate["key"], "item": candidate["item"],
                    "proposer_agents": [], "proposal_support": {}, "reviews": {},
                })
                node["proposer_agents"].append(agent_id)
                node["proposal_support"][agent_id] = candidate["support"]
        nodes = list(candidates.values())
        nodes.sort(key=lambda node: (-len(node["proposer_agents"]),
                                    -sum(node["proposal_support"].values()), node["key"]))
        graphs.append({
            "SubjectEntity": subject, "Relation": relation, "input_index": index,
            "commitments": commitments, "proposal_sample_counts": proposal_sample_counts,
            "proposal_parse_diagnostics": proposal_parse_diagnostics,
            "candidates": nodes,
        })
    return graphs


def attach_reviews(graphs: List[dict], review_dir: Optional[Path],
                   agents: Sequence[Mapping[str, Any]]) -> None:
    if review_dir is None:
        return
    review_responses = load_responses(review_dir, agents)
    graph_by_key = {_key(graph): graph for graph in graphs}
    for agent_id, responses in review_responses.items():
        for response in responses.values():
            if response.get("phase") != "blind_review":
                continue
            graph = graph_by_key.get((response.get("subject"), response.get("relation")))
            if graph is None:
                raise ContractError(f"review references unknown graph: {response.get('task_id')}")
            matches = [node for node in graph["candidates"]
                       if node["key"] == response.get("candidate_key")]
            if len(matches) != 1:
                raise ContractError(f"review references unknown candidate: {response.get('task_id')}")
            node = matches[0]
            if agent_id in node["proposer_agents"]:
                raise ContractError(f"agent reviewed its own candidate: {response.get('task_id')}")
            node["reviews"][agent_id] = {
                "label": response["selected_choice"],
                "probabilities": response.get("choice_probabilities", {}),
            }


def write_review_tasks(graphs: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]],
                       output_dir: Path) -> dict:
    tasks = build_review_tasks(graphs, agents)
    by_agent: Dict[str, List[dict]] = defaultdict(list)
    for task in tasks:
        by_agent[task["agent_id"]].append(task)
    result = {}
    for agent in agents:
        path = output_dir / f"{agent['id']}.jsonl"
        write_jsonl_atomic(path, by_agent.get(agent["id"], []))
        result[agent["id"]] = {"path": str(path), "sha256": sha256(path),
                               "tasks": len(by_agent.get(agent["id"], []))}
    return result


def _numeric_values(node_list: Sequence[Mapping[str, Any]],
                    allowed_agents: Optional[set[str]] = None) -> List[float]:
    values = []
    for node in node_list:
        if allowed_agents is not None and not (set(node["proposer_agents"]) & allowed_agents):
            continue
        try:
            value = float(str(node["item"]).replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(value) and value > 0:
            support = sum(
                count for agent, count in node["proposal_support"].items()
                if allowed_agents is None or agent in allowed_agents)
            values.extend([value] * max(1, support))
    return values


def _existence_yes(graph: Mapping[str, Any], allowed_agents: Optional[set[str]] = None) -> bool:
    labels = [value["existence"] for agent, value in graph["commitments"].items()
              if allowed_agents is None or agent in allowed_agents]
    if graph["Relation"] not in NULLABLE_RELATIONS:
        return True
    return labels.count("YES") > labels.count("NO")


def _cardinality_consensus(graph: Mapping[str, Any]) -> str:
    labels = [value.get("cardinality", "UNKNOWN")
              for value in graph["commitments"].values()]
    usable = [label for label in labels if label != "UNKNOWN"]
    if not usable:
        return "UNKNOWN"
    counts = Counter(usable)
    best_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == best_count]
    return winners[0] if len(winners) == 1 else "UNKNOWN"


def prediction_for_agent(graph: Mapping[str, Any], agent_id: str) -> List[str]:
    relation = graph["Relation"]
    if not _existence_yes(graph, {agent_id}):
        return []
    nodes = [node for node in graph["candidates"] if agent_id in node["proposer_agents"]]
    if relation in NUMERIC_RELATIONS:
        values = _numeric_values(nodes, {agent_id})
        return [format(statistics.median(values), ".12g")] if values else []
    nodes.sort(key=lambda node: (-node["proposal_support"][agent_id], node["key"]))
    if relation in SINGLE_RELATIONS:
        return [nodes[0]["item"]] if nodes else []
    # Majority within this agent's actual N proposal samples.
    n = int(graph.get("proposal_sample_counts", {}).get(agent_id, 0))
    threshold = max(1, math.ceil(n / 2))
    return [node["item"] for node in nodes
            if node["proposal_support"][agent_id] >= threshold]


def heterogeneous_prediction(graph: Mapping[str, Any], *, use_reviews: bool,
                             use_cardinality: bool = False) -> List[str]:
    relation = graph["Relation"]
    if not _existence_yes(graph):
        return []
    cardinality = _cardinality_consensus(graph) if use_cardinality else "UNKNOWN"
    if cardinality == "ZERO":
        return []
    if relation in NUMERIC_RELATIONS:
        values = _numeric_values(graph["candidates"])
        return [format(statistics.median(values), ".12g")] if values else []
    eligible = []
    for node in graph["candidates"]:
        evidence_agents = set(node["proposer_agents"])
        contradictions = set()
        if use_reviews:
            for agent_id, review in node["reviews"].items():
                if review["label"] == "SUPPORTED":
                    evidence_agents.add(agent_id)
                elif review["label"] == "CONTRADICTED":
                    contradictions.add(agent_id)
        # This is the predeclared conservative baseline router, not a tuned
        # final policy: two independent model families and no contradiction.
        if len(evidence_agents) >= 2 and not contradictions:
            eligible.append((node, len(evidence_agents)))
    eligible.sort(key=lambda pair: (-pair[1], -len(pair[0]["proposer_agents"]),
                                    -sum(pair[0]["proposal_support"].values()), pair[0]["key"]))
    if relation in SINGLE_RELATIONS:
        return [eligible[0][0]["item"]] if eligible else []
    if cardinality == "ONE":
        return [eligible[0][0]["item"]] if eligible else []
    return [node["item"] for node, _ in eligible]


def prediction_rows(graphs: Sequence[Mapping[str, Any]], policy: str,
                    agent_ids: Sequence[str]) -> List[dict]:
    rows = []
    for graph in graphs:
        if policy.startswith("agent:"):
            agent_id = policy.split(":", 1)[1]
            if agent_id not in agent_ids:
                raise ContractError(f"unknown prediction agent {agent_id}")
            objects = prediction_for_agent(graph, agent_id)
        elif policy == "heterogeneous_proposal_consensus":
            objects = heterogeneous_prediction(graph, use_reviews=False)
        elif policy == "heterogeneous_proposal_cardinality_consensus":
            objects = heterogeneous_prediction(
                graph, use_reviews=False, use_cardinality=True)
        elif policy == "blind_review_consensus":
            objects = heterogeneous_prediction(graph, use_reviews=True)
        elif policy == "blind_review_cardinality_consensus":
            objects = heterogeneous_prediction(
                graph, use_reviews=True, use_cardinality=True)
        elif policy == "candidate_union":
            objects = [node["item"] for node in graph["candidates"]]
        else:
            raise ContractError(f"unknown policy {policy}")
        rows.append({"SubjectEntity": graph["SubjectEntity"], "Relation": graph["Relation"],
                     "ObjectEntities": objects})
    return rows


def _gold_aliases(row: Mapping[str, Any]) -> list:
    gold = row.get("ObjectEntities", [])
    if gold and isinstance(gold[0], str):
        return [[item] for item in gold]
    return gold


def oracle_rows(graphs: Sequence[Mapping[str, Any]], gold_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    gold_by_key = {_key(row): row for row in gold_rows}
    output = []
    for graph in graphs:
        gold = gold_by_key[_key(graph)]
        aliases = _gold_aliases(gold)
        relation = graph["Relation"]
        selected = []
        if aliases:
            matched = 0
            for node in graph["candidates"]:
                new_matched = true_positives(
                    selected + [node["item"]], aliases,
                    RELATION_TYPE.get(relation, "string"), 0.05)
                if new_matched > matched:
                    selected.append(node["item"])
                    matched = new_matched
                    if relation in SINGLE_RELATIONS:
                        break
        output.append({"SubjectEntity": graph["SubjectEntity"], "Relation": relation,
                       "ObjectEntities": selected})
    return output


def score(rows: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]) -> dict:
    per_pair = evaluate_per_sr_pair(list(rows), list(gold), RELATION_TYPE, tolerance=0.05)
    return {relation: values["macro-f1"]
            for relation, values in macro_average_per_relation(per_pair).items()}


def portfolio_diagnostics(graphs: Sequence[Mapping[str, Any]], gold_rows: Sequence[Mapping[str, Any]],
                          agent_ids: Sequence[str]) -> dict:
    gold_by_key = {_key(row): row for row in gold_rows}
    unique_correct_rows = Counter()
    correct_vectors: Dict[str, List[float]] = {agent: [] for agent in agent_ids}
    null_false_positives = Counter()
    parse_diagnostics: Dict[str, Counter] = {
        agent: Counter() for agent in agent_ids}
    cardinality_diagnostics: Dict[str, Counter] = {
        agent: Counter() for agent in agent_ids}
    for graph in graphs:
        gold = gold_by_key[_key(graph)]
        aliases = _gold_aliases(gold)
        true_cardinality = ("ZERO" if len(aliases) == 0 else
                            "ONE" if len(aliases) == 1 else "MANY")
        relation_type = RELATION_TYPE.get(graph["Relation"], "string")
        for agent in agent_ids:
            parse_diagnostics[agent].update(
                graph.get("proposal_parse_diagnostics", {}).get(agent, {}))
            predicted_cardinality = graph["commitments"][agent]["cardinality"]
            cardinality_diagnostics[agent][
                f"{true_cardinality}->{predicted_cardinality}"] += 1
            cardinality_diagnostics[agent]["total"] += 1
            if predicted_cardinality == true_cardinality:
                cardinality_diagnostics[agent]["correct"] += 1
            if predicted_cardinality == "UNKNOWN":
                cardinality_diagnostics[agent]["unknown"] += 1
        agent_correct_candidate = {}
        for agent in agent_ids:
            candidates = [node["item"] for node in graph["candidates"]
                          if agent in node["proposer_agents"]]
            has_correct = bool(aliases and true_positives(
                candidates, aliases, relation_type, 0.05) > 0)
            agent_correct_candidate[agent] = has_correct
            correct_vectors[agent].append(float(has_correct))
            if not aliases and candidates:
                null_false_positives[agent] += 1
        winners = [agent for agent, value in agent_correct_candidate.items() if value]
        if len(winners) == 1:
            unique_correct_rows[winners[0]] += 1

    correlations = {}
    for left_index, left in enumerate(agent_ids):
        for right in agent_ids[left_index + 1:]:
            x, y = correct_vectors[left], correct_vectors[right]
            mx, my = statistics.mean(x), statistics.mean(y)
            numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
            dx = math.sqrt(sum((a - mx) ** 2 for a in x))
            dy = math.sqrt(sum((b - my) ** 2 for b in y))
            correlation = numerator / (dx * dy) if dx and dy else None
            correlations[f"{left}__{right}"] = correlation
    return {
        "unique_correct_candidate_rows": dict(unique_correct_rows),
        "null_rows_with_any_candidate": dict(null_false_positives),
        "candidate_correctness_correlations": correlations,
        "proposal_parse_diagnostics": {
            agent: dict(counts) for agent, counts in parse_diagnostics.items()},
        "cardinality_commitment_diagnostics": {
            agent: {
                **dict(counts),
                "accuracy": (counts["correct"] / counts["total"]
                             if counts["total"] else None),
            }
            for agent, counts in cardinality_diagnostics.items()},
    }


def markdown_report(scores: Mapping[str, Mapping[str, float]], diagnostics: Mapping[str, Any],
                    config: Mapping[str, Any], has_reviews: bool) -> str:
    relations = sorted({relation for values in scores.values() for relation in values})
    lines = [
        "# Heterogeneous parametric-memory portfolio audit", "",
        "This is a development audit, not a production or blind-test result.", "",
        f"Legally counted full-checkpoint total: "
        f"**{config['declared_parameter_total'] / 1e9:.6f}B / "
        f"{config['parameter_cap'] / 1e9:.1f}B**.",
        f"Active text modules after verified stripping: "
        f"**{config['active_text_inference_parameter_total'] / 1e9:.6f}B** "
        "(not used to claim additional parameter-cap headroom).",
        f"Blind review responses present: **{has_reviews}**.", "",
        "## Scores", "",
        "| policy | " + " | ".join(relations) + " |",
        "|---|" + "|".join(["---:"] * len(relations)) + "|",
    ]
    for policy, values in scores.items():
        lines.append("| " + policy + " | " + " | ".join(
            f"{values.get(relation, float('nan')):.4f}" for relation in relations) + " |")
    lines.extend(["", "## Complementarity diagnostics", "", "```json",
                  json.dumps(diagnostics, indent=2, sort_keys=True), "```", "",
                  "The `candidate_union_oracle` is gold-aware and non-deployable. It measures "
                  "whether heterogeneous models add reservoir coverage; it is not a selector.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--review-response-dir", default=None)
    parser.add_argument("--gold", default=None,
                        help="Optional labeled train file for the final audit only")
    parser.add_argument("--agents", default=str(Path(__file__).with_name("agents.json")))
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_agent_config(Path(args.agents).resolve())
    agents = config["agents"]
    inputs = read_jsonl(plan_dir / "input_subset.jsonl")
    responses = load_responses(Path(args.response_dir).resolve(), agents)
    graphs = assemble_graphs(inputs, agents, responses)
    review_dir = Path(args.review_response_dir).resolve() if args.review_response_dir else None
    attach_reviews(graphs, review_dir, agents)
    graph_path = output_dir / "claim_graph.jsonl"
    write_jsonl_atomic(graph_path, graphs)
    review_tasks = write_review_tasks(graphs, agents, output_dir / "review_tasks")

    manifest = {
        "schema": "heterogeneous-claim-graph-v1",
        "plan": str(plan_dir / "PLAN.json"), "plan_sha256": sha256(plan_dir / "PLAN.json"),
        "claim_graph": str(graph_path), "claim_graph_sha256": sha256(graph_path),
        "declared_parameter_total": config["declared_parameter_total"],
        "verified_parameter_total": config["verified_parameter_total"],
        "active_text_inference_parameter_total":
            config["active_text_inference_parameter_total"],
        "review_tasks": review_tasks, "contains_gold_labels": False,
    }

    if args.gold:
        gold_all = read_jsonl(Path(args.gold).resolve())
        gold_by_key = {_key(row): row for row in gold_all}
        gold = [gold_by_key[_key(row)] for row in inputs]
        agent_ids = [agent["id"] for agent in agents]
        policies = [f"agent:{agent_id}" for agent_id in agent_ids]
        policies.extend(["heterogeneous_proposal_consensus",
                         "heterogeneous_proposal_cardinality_consensus",
                         "candidate_union"])
        if review_dir:
            policies.extend(["blind_review_consensus",
                             "blind_review_cardinality_consensus"])
        scores = {policy: score(prediction_rows(graphs, policy, agent_ids), gold)
                  for policy in policies}
        scores["candidate_union_oracle"] = score(oracle_rows(graphs, gold), gold)
        diagnostics = portfolio_diagnostics(graphs, gold, agent_ids)
        audit = {"scores": scores, "diagnostics": diagnostics,
                 "gold_path": str(Path(args.gold).resolve()),
                 "gold_sha256": sha256(Path(args.gold).resolve())}
        (output_dir / "AUDIT.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n")
        (output_dir / "RESULT.md").write_text(
            markdown_report(scores, diagnostics, config, review_dir is not None))
        manifest["audit"] = str(output_dir / "AUDIT.json")
        manifest["audit_sha256"] = sha256(output_dir / "AUDIT.json")
        # Gold hash belongs only to the audit manifest, never claim_graph/tasks.
        manifest["audit_gold_sha256"] = sha256(Path(args.gold).resolve())

    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"claim graphs: {len(graphs)} -> {graph_path}")
    print(f"review tasks: {sum(value['tasks'] for value in review_tasks.values())}")
    if args.gold:
        print(f"audit: {output_dir / 'RESULT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
