#!/usr/bin/env python3
"""Direct heterogeneous tournament over multiple complete graph actions.

The previous cascade collapsed every row to KEEP plus one Qwen-selected
challenger before learning whether to switch.  This module preserves up to six
complete, label-free graph actions.  It compares them in two anchored groups,
each containing the same KEEP action, so within-group log odds remain
comparable without discarding challengers.

``prepare`` never opens labels.  It ranks actions only with frozen
candidate-truth responses and writes hash-addressed choice tasks.  ``analyze``
opens train labels after responses are complete and evaluates predeclared,
parameter-free aggregation rules.  Validation is deliberately absent until
the train-only promotion gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.component_expected_f1_set_decoder import (
    _validated_truth_evidence,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    balanced_choice_codebooks,
    canonical_key,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.row_grouped_action_ranker import _utility
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    validate_registered_predictions,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import _key


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_REVIEW_RUN = RUNS / "baseline_conditioned_action_review_20260727_v2"
DEFAULT_TRUTH_RUN = RUNS / "candidate_truth_evidence_20260727_v1"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_OUTPUT = RUNS / "multi_challenger_graph_decoder_20260727_v1"

AGENTS = (QWEN, GEMMA)
ABSTAIN = "ABSTAIN"
# Preserve six alternatives.  A train-only, label-free shortlist audit showed
# that four alternatives retained only 62.5% of hasCapacity action-oracle
# gain, while six retained at least 91.7% of the gain for every relation.
# Each review remains small: KEEP, two challengers, and ABSTAIN.
MAX_CHALLENGERS = 6
GROUP_CHALLENGERS = 2
PRIMARY_ARM = "dual_argmax"
ARMS = (
    "qwen_argmax",
    "gemma_argmax",
    "arithmetic_mean",
    "geometric_mean",
    "minimum",
    "dual_argmax",
)
MIN_INCREMENTAL_DELTA = 0.010
MIN_HELP_HARM_RATIO = 1.0
MIN_SHORTLIST_GAIN_CAPTURE = 0.75
MIN_RELATION_DELTA = -0.010
EPSILON = 1e-12


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        dict(value), indent=2, sort_keys=True) + "\n")


def _validated_plan(output: Path) -> tuple[Path, dict[str, Any]]:
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    review_plan_path = Path(plan["review_run"]) / "plan/PLAN.json"
    checks = (
        plan.get("schema") == "multi-challenger-graph-decoder-plan-v1",
        not plan.get("contains_labels"),
        not plan.get("gold_aware"),
        not plan.get("validation_opened"),
        plan.get("starting_pipeline_id") == COMPETITION_PIPELINE_ID,
        sha256(Path(plan["implementation"]))
        == plan.get("implementation_sha256"),
        sha256(review_plan_path) == plan.get("review_plan_sha256"),
        sha256(Path(plan["action_registry"]))
        == plan.get("action_registry_sha256"),
        sha256(Path(plan["shortlists"]))
        == plan.get("shortlists_sha256"),
    )
    if not all(checks):
        raise ContractError("invalid or stale multi-challenger plan")
    return plan_path, plan


def _stable_id(*values: object) -> str:
    return hashlib.sha256(
        "\x1f".join(map(str, values)).encode()).hexdigest()[:20]


def _format_objects(objects: Sequence[str]) -> str:
    if not objects:
        return "None"
    return "[" + "; ".join(
        json.dumps(str(item), ensure_ascii=False) for item in objects) + "]"


def _canonical_objects(
    objects: Sequence[str], relation: str,
) -> set[str]:
    return {
        canonical_key(str(item), relation)
        for item in objects
    }


def _component_scores(
    graph: Mapping[str, Any],
    truth_evidence: Mapping[
        tuple[str, str, str], Mapping[str, float]],
) -> dict[str, float]:
    """Map every component surface to a heterogeneous mean truth score."""
    relation = str(graph["Relation"])
    output: dict[str, float] = {}
    for node in graph["nodes"]:
        if node.get("node_type") != "candidate_component":
            continue
        identity = (*_key(graph), str(node["id"]))
        if identity not in truth_evidence:
            raise ContractError(f"missing truth evidence: {identity}")
        evidence = truth_evidence[identity]
        probability = (
            float(evidence[QWEN]) + float(evidence[GEMMA])) / 2.0
        if not 0.0 <= probability <= 1.0:
            raise ContractError(f"invalid component score: {identity}")
        for item in [
            *node.get("member_items", []),
            node["representative"],
        ]:
            surface = canonical_key(str(item), relation)
            if (
                surface in output
                and abs(output[surface] - probability) > EPSILON
            ):
                raise ContractError(
                    f"{_key(graph)}: surface belongs to multiple components: "
                    f"{item}")
            output[surface] = probability
    return output


def action_shortlist_score(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    keep: Mapping[str, Any],
    probability_by_surface: Mapping[str, float],
) -> float:
    """Rank edits by heterogeneous truth mass added minus mass removed.

    This is an identity rule only.  It is not the final switch decision.
    Missing incumbent-only components receive a neutral 0.5 score, which
    prevents an absent truth task from becoming implicit evidence to delete.
    """
    relation = str(graph["Relation"])
    selected = _canonical_objects(action["objects"], relation)
    incumbent = _canonical_objects(keep["objects"], relation)
    added = selected - incumbent
    removed = incumbent - selected
    return (
        sum(float(probability_by_surface.get(item, 0.5)) for item in added)
        - sum(float(probability_by_surface.get(item, 0.5)) for item in removed)
    )


def shortlist_actions(
    graph: Mapping[str, Any],
    truth_evidence: Mapping[
        tuple[str, str, str], Mapping[str, float]],
    limit: int = MAX_CHALLENGERS,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    actions = list(graph["actions"])
    keep = [
        action for action in actions if action["action_type"] == "KEEP"]
    if len(keep) != 1:
        raise ContractError(f"{_key(graph)}: expected exactly one KEEP")
    probability_by_surface = _component_scores(graph, truth_evidence)
    challengers = sorted(
        (action for action in actions if action is not keep[0]),
        key=lambda action: (
            action_shortlist_score(
                graph, action, keep[0], probability_by_surface),
            str(action["id"]),
        ),
        reverse=True,
    )[:limit]
    return keep[0], challengers


def _edit_description(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    keep: Mapping[str, Any],
) -> str:
    relation = str(graph["Relation"])
    before = _canonical_objects(keep["objects"], relation)
    after = _canonical_objects(action["objects"], relation)
    before_render = {
        canonical_key(str(item), relation): str(item)
        for item in keep["objects"]}
    after_render = {
        canonical_key(str(item), relation): str(item)
        for item in action["objects"]}
    added = [after_render[item] for item in sorted(after - before)]
    removed = [before_render[item] for item in sorted(before - after)]
    edits = []
    if added:
        edits.append("ADD " + _format_objects(added))
    if removed:
        edits.append("REMOVE " + _format_objects(removed))
    if not edits:
        edits.append("NO CHANGE")
    return (
        f"{'; '.join(edits)}; COMPLETE OUTPUT "
        f"{_format_objects(action['objects'])}"
    )


def tournament_prompt(
    graph: Mapping[str, Any],
    keep: Mapping[str, Any],
    challengers: Sequence[Mapping[str, Any]],
    codebook: Mapping[str, str],
) -> str:
    options = [
        (
            f"{codebook[str(keep['id'])]} = KEEP CURRENT OUTPUT "
            f"{_format_objects(keep['objects'])}"
        )
    ]
    options.extend(
        f"{codebook[str(action['id'])]} = "
        f"{_edit_description(graph, action, keep)}"
        for action in challengers
    )
    options.append(
        f"{codebook[ABSTAIN]} = ABSTAIN because no edit is reliably better")
    return (
        "Act as the final editor of a closed-book knowledge-base answer. "
        "Choose the single COMPLETE output expected to receive the highest "
        "official F1 score. Compare all edits directly in the same context. "
        "A false addition and deletion of a true item are both harmful. For "
        "numeric relations, an answer is correct only within the task's 5% "
        "tolerance. Do not choose an edit merely because it is new, detailed, "
        "or proposed by the options. Keep the current output or abstain unless "
        "your factual memory supports an improvement. Option order and code "
        "letters contain no information.\n"
        f"SUBJECT: {graph['SubjectEntity']}\n"
        f"RELATION: {graph['Relation']}\n"
        "OPTIONS:\n- " + "\n- ".join(options) + "\n"
        "Return exactly one option code and nothing else.\nCODE:"
    )


def _task(
    graph: Mapping[str, Any],
    keep: Mapping[str, Any],
    challengers: Sequence[Mapping[str, Any]],
    agent: str,
    row_index: int,
    group_index: int,
) -> dict[str, Any]:
    choices = (
        str(keep["id"]),
        *(str(action["id"]) for action in challengers),
        ABSTAIN,
    )
    salt = (
        "multi-challenger-graph-decoder-v1",
        agent,
        graph["SubjectEntity"],
        graph["Relation"],
        group_index,
    )
    codebooks = balanced_choice_codebooks(choices, *salt)
    variants = [
        {
            "choice_codes": dict(codebook),
            "prompt": tournament_prompt(
                graph, keep, challengers, codebook),
        }
        for codebook in codebooks
    ]
    return {
        "task_id": (
            f"{agent}::multi_challenger::{row_index}::"
            f"{group_index}::"
            f"{_stable_id(graph['SubjectEntity'], graph['Relation'])}"
        ),
        "agent_id": agent,
        "subject": str(graph["SubjectEntity"]),
        "relation": str(graph["Relation"]),
        "phase": "multi_challenger_graph_tournament",
        "mode": "choice",
        "prompt": variants[0]["prompt"],
        "choices": list(choices),
        "choice_codes": variants[0]["choice_codes"],
        "choice_variants": variants,
        "keep_action_id": str(keep["id"]),
        "group_index": group_index,
        "challenger_action_ids": [
            str(action["id"]) for action in challengers],
        "contains_labels": False,
        "gold_aware": False,
        "prompt_masks_provenance": True,
        "prompt_compares_complete_outputs": True,
    }


def _validated_prepare_inputs(
    review_run: Path,
    truth_run: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[
        tuple[str, str, str], Mapping[str, float]]]:
    review_plan = _json(review_run / "plan/PLAN.json")
    registry = Path(review_plan["registry"])
    if (
        review_plan.get("schema")
        != "baseline-conditioned-action-review-plan-v1"
        or review_plan.get("contains_labels")
        or review_plan.get("gold_aware")
        or review_plan.get("starting_pipeline_id")
        != COMPETITION_PIPELINE_ID
        or sha256(registry) != review_plan["registry_sha256"]
    ):
        raise ContractError("invalid action-review source")
    graphs = read_jsonl(registry)
    if len(graphs) != int(review_plan["rows"]):
        raise ContractError("action registry row count mismatch")
    control_path = Path(review_plan["starting_predictions"]).resolve()
    validate_registered_predictions(
        control_path,
        pipeline_id=COMPETITION_PIPELINE_ID,
        split="train",
    )
    if sha256(control_path) != review_plan["starting_predictions_sha256"]:
        raise ContractError("stale registered SOTA control")
    control = {_key(row): row for row in read_jsonl(control_path)}
    if {_key(graph) for graph in graphs} != set(control):
        raise ContractError("action registry/control coverage mismatch")
    for graph in graphs:
        keep = [
            action for action in graph["actions"]
            if action["action_type"] == "KEEP"
        ]
        if (
            len(keep) != 1
            or _canonical_objects(
                keep[0]["objects"], str(graph["Relation"]))
            != _canonical_objects(
                control[_key(graph)]["ObjectEntities"],
                str(graph["Relation"]),
            )
        ):
            raise ContractError(
                f"{_key(graph)}: KEEP is not exact registered SOTA")
    truth, _ = _validated_truth_evidence(
        truth_run, str(review_plan["source_graph_sha256"]))
    return review_plan, graphs, truth


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    review_run = Path(args.review_run).resolve()
    truth_run = Path(args.truth_run).resolve()
    review_plan, graphs, truth = _validated_prepare_inputs(
        review_run, truth_run)

    shortlist_rows = []
    tasks_by_agent: dict[str, list[dict[str, Any]]] = {
        agent: [] for agent in AGENTS}
    for row_index, graph in enumerate(graphs):
        keep, challengers = shortlist_actions(graph, truth)
        groups = [
            challengers[index:index + GROUP_CHALLENGERS]
            for index in range(0, len(challengers), GROUP_CHALLENGERS)
        ]
        shortlist_rows.append({
            "SubjectEntity": str(graph["SubjectEntity"]),
            "Relation": str(graph["Relation"]),
            "keep_action_id": str(keep["id"]),
            "challenger_action_ids": [
                str(action["id"]) for action in challengers],
            "action_ids": [
                str(keep["id"]),
                *(str(action["id"]) for action in challengers),
            ],
            "groups": [
                [str(action["id"]) for action in group]
                for group in groups
            ],
            "contains_labels": False,
            "gold_aware": False,
        })
        for agent in AGENTS:
            for group_index, group in enumerate(groups):
                tasks_by_agent[agent].append(_task(
                    graph, keep, group, agent, row_index, group_index))

    shortlist_path = output / "plan/SHORTLISTS.jsonl"
    write_jsonl_atomic(shortlist_path, shortlist_rows)
    jobs = {}
    for agent, tasks in tasks_by_agent.items():
        validate_tasks(tasks, agent)
        task_path = output / f"plan/tasks/{agent}.jsonl"
        smoke_path = output / f"plan/smoke/{agent}.jsonl"
        response_path = output / f"responses/{agent}.jsonl"
        write_jsonl_atomic(task_path, tasks)
        write_jsonl_atomic(smoke_path, tasks[:2])
        jobs[agent] = {
            "tasks": len(tasks),
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256(smoke_path),
            "response_path": str(response_path),
        }

    plan = {
        "schema": "multi-challenger-graph-decoder-plan-v1",
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "review_run": str(review_run),
        "review_plan_sha256": sha256(review_run / "plan/PLAN.json"),
        "action_registry": str(Path(review_plan["registry"])),
        "action_registry_sha256": review_plan["registry_sha256"],
        "truth_run": str(truth_run),
        "shortlists": str(shortlist_path),
        "shortlists_sha256": sha256(shortlist_path),
        "rows": len(graphs),
        "max_challengers": MAX_CHALLENGERS,
        "primary_arm": PRIMARY_ARM,
        "arms": list(ARMS),
        "jobs": jobs,
        "agents": review_plan["agents"],
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "development_only": True,
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, plan)
    print(f"plan: {plan_path}")
    for agent, job in jobs.items():
        print(f"{agent}: {job['tasks']} tasks -> {job['task_path']}")
    return 0


def _validated_responses(
    plan: Mapping[str, Any],
) -> dict[
    tuple[tuple[str, str], str, int],
    Mapping[str, float],
]:
    output = {}
    for agent in AGENTS:
        job = plan["jobs"][agent]
        task_path = Path(job["task_path"])
        response_path = Path(job["response_path"])
        if sha256(task_path) != job["task_sha256"]:
            raise ContractError(f"{agent}: task hash mismatch")
        manifest_path = response_path.with_suffix(
            response_path.suffix + ".manifest.json")
        tasks = read_jsonl(task_path)
        by_id = validate_tasks(tasks, agent)
        if not response_path.is_file() or not manifest_path.is_file():
            raise ContractError(f"{agent}: missing responses or manifest")
        manifest = _json(manifest_path)
        if (
            manifest.get("task_sha256") != job["task_sha256"]
            or manifest.get("output_sha256") != sha256(response_path)
            or manifest.get("agent_id") != agent
            or int(manifest.get("tasks", -1)) != int(job["tasks"])
            or manifest.get("contains_labels")
            or manifest.get("gold_aware")
        ):
            raise ContractError(f"{agent}: response manifest mismatch")
        responses = read_jsonl(response_path)
        if len(responses) != len(tasks):
            raise ContractError(f"{agent}: incomplete responses")
        response_by_id = {}
        for response in responses:
            task_id = str(response["task_id"])
            if task_id not in by_id or task_id in response_by_id:
                raise ContractError(f"{agent}: invalid response id")
            validate_task_response(by_id[task_id], response)
            response_by_id[task_id] = response
        for task in tasks:
            response = response_by_id[str(task["task_id"])]
            probabilities = {
                str(choice): float(value)
                for choice, value in response[
                    "choice_probabilities"].items()}
            if (
                set(probabilities) != set(map(str, task["choices"]))
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in probabilities.values())
                or abs(sum(probabilities.values()) - 1.0) > 1e-6
            ):
                raise ContractError(f"{agent}: invalid choice probabilities")
            key = str(task["subject"]), str(task["relation"])
            identity = key, agent, int(task["group_index"])
            if identity in output:
                raise ContractError(f"{agent}: duplicate response group")
            output[identity] = probabilities
    return output


def _aggregate(
    arm: str,
    qwen_margins: Mapping[str, float],
    gemma_margins: Mapping[str, float],
    keep_action_id: str,
) -> str:
    choices = tuple(qwen_margins)
    if set(choices) != set(gemma_margins):
        raise ContractError("heterogeneous action coverage mismatch")
    if not choices:
        return keep_action_id
    scores: dict[str, float]
    if arm == "qwen_argmax":
        scores = dict(qwen_margins)
    elif arm == "gemma_argmax":
        scores = dict(gemma_margins)
    elif arm == "arithmetic_mean":
        scores = {
            choice: math.log(max(EPSILON, (
                math.exp(float(qwen_margins[choice]))
                + math.exp(float(gemma_margins[choice]))
            ) / 2.0))
            for choice in choices
        }
    elif arm == "geometric_mean":
        # Margins are log ratios. Their arithmetic mean is the log geometric
        # mean of the two agents' action-versus-KEEP odds.
        scores = {
            choice: (
                float(qwen_margins[choice])
                + float(gemma_margins[choice])
            ) / 2.0
            for choice in choices
        }
    elif arm == "minimum":
        scores = {
            choice: min(
                float(qwen_margins[choice]),
                float(gemma_margins[choice]),
            )
            for choice in choices
        }
    elif arm == "dual_argmax":
        q_choice = max(choices, key=lambda choice: (
            qwen_margins[choice], choice))
        g_choice = max(choices, key=lambda choice: (
            gemma_margins[choice], choice))
        if (
            q_choice == g_choice
            and qwen_margins[q_choice] > 0.0
            and gemma_margins[g_choice] > 0.0
        ):
            return q_choice
        return keep_action_id
    else:
        raise ContractError(f"unknown aggregation arm: {arm}")
    selected = max(
        choices, key=lambda choice: (scores[choice], choice))
    return selected if scores[selected] > 0.0 else keep_action_id


def _result_summary(
    predictions: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_scores = score(list(predictions), list(gold))
    control_scores = score(list(control), list(gold))
    changed = [row for row in diagnostics if row["changed"]]
    relation_deltas = {
        relation: selected_scores[relation] - control_scores[relation]
        for relation in control_scores if relation != "*** All Relations ***"
    }
    return {
        "scores": selected_scores,
        "control_scores": control_scores,
        "incremental_delta": (
            selected_scores["*** All Relations ***"]
            - control_scores["*** All Relations ***"]),
        "relation_deltas": relation_deltas,
        "changed_rows": len(changed),
        "helped_rows": sum(
            float(row["utility_delta"]) > EPSILON for row in changed),
        "harmed_rows": sum(
            float(row["utility_delta"]) < -EPSILON for row in changed),
        "neutral_changed_rows": sum(
            abs(float(row["utility_delta"])) <= EPSILON for row in changed),
    }


def _write_result_markdown(
    path: Path,
    result: Mapping[str, Any],
) -> None:
    primary = result["arms"][PRIMARY_ARM]
    lines = [
        "# Multi-challenger heterogeneous graph decoder",
        "",
        "This is a train-only development audit. Validation was not opened.",
        "",
        "## Primary result",
        "",
        f"- Arm: `{PRIMARY_ARM}`",
        f"- Control macro-F1: "
        f"`{primary['control_scores']['*** All Relations ***']:.9f}`",
        f"- Selected macro-F1: "
        f"`{primary['scores']['*** All Relations ***']:.9f}`",
        f"- Incremental delta: `{primary['incremental_delta']:+.9f}`",
        f"- Changed/helped/harmed/neutral: "
        f"`{primary['changed_rows']}/{primary['helped_rows']}/"
        f"{primary['harmed_rows']}/{primary['neutral_changed_rows']}`",
        f"- Promotion gate: "
        f"`{'PASS' if result['promotion_gate']['passed'] else 'FAIL'}`",
        "",
        "## Relation deltas",
        "",
        "| Relation | Delta | Shortlist gain capture |",
        "|---|---:|---:|",
    ]
    for relation in sorted(primary["relation_deltas"]):
        lines.append(
            f"| {relation} | "
            f"{primary['relation_deltas'][relation]:+.6f} | "
            f"{result['shortlist_supply'][relation]['gain_capture']:.1%} |"
        )
    lines.extend([
        "",
        "## Failure ledger",
        "",
        "```json",
        json.dumps(
            result["failure_ledgers"][PRIMARY_ARM],
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        f"Next stage: `{result['next_stage']}`.",
        "",
    ])
    path.write_text("\n".join(lines))


def audit_supply(args: argparse.Namespace) -> int:
    """Measure label-free shortlist coverage; never fit or select a decoder."""
    output = Path(args.output_dir).resolve()
    plan_path, plan = _validated_plan(output)
    review_plan = _json(Path(plan["review_run"]) / "plan/PLAN.json")
    graphs = read_jsonl(Path(plan["action_registry"]))
    shortlists = read_jsonl(Path(plan["shortlists"]))
    gold_path = Path(args.gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    control = read_jsonl(Path(review_plan["starting_predictions"]))
    if (
        len(graphs) != len(shortlists)
        or {_key(graph) for graph in graphs} != set(gold_by)
    ):
        raise ContractError("supply-audit row coverage mismatch")
    full_oracle, shortlist_oracle = [], []
    relation_rows: dict[str, list[dict[str, float]]] = {}
    for graph, shortlist in zip(graphs, shortlists, strict=True):
        key = _key(graph)
        if key != _key(shortlist):
            raise ContractError("graph/shortlist order mismatch")
        actions = {
            str(action["id"]): action for action in graph["actions"]}
        keep = actions[str(shortlist["keep_action_id"])]
        base = _utility(graph, keep, gold_by)
        full = max(
            graph["actions"],
            key=lambda action: _utility(graph, action, gold_by))
        candidates = [
            keep,
            *(actions[str(action_id)]
              for action_id in shortlist["challenger_action_ids"]),
        ]
        short = max(
            candidates,
            key=lambda action: _utility(graph, action, gold_by))
        full_gain = _utility(graph, full, gold_by) - base
        short_gain = _utility(graph, short, gold_by) - base
        relation_rows.setdefault(
            str(graph["Relation"]), []).append({
                "full_gain": full_gain,
                "short_gain": short_gain,
            })
        full_oracle.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": list(full["objects"]),
        })
        shortlist_oracle.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": list(short["objects"]),
        })
    by_relation = {}
    for relation, rows in relation_rows.items():
        full_gain = sum(row["full_gain"] for row in rows)
        short_gain = sum(row["short_gain"] for row in rows)
        full_rows = sum(row["full_gain"] > EPSILON for row in rows)
        short_rows = sum(row["short_gain"] > EPSILON for row in rows)
        by_relation[relation] = {
            "rows": len(rows),
            "full_oracle_rows": full_rows,
            "shortlist_oracle_rows": short_rows,
            "row_capture": (
                short_rows / full_rows if full_rows else 1.0),
            "full_oracle_gain": full_gain,
            "shortlist_oracle_gain": short_gain,
            "gain_capture": (
                short_gain / full_gain if full_gain > EPSILON else 1.0),
        }
    result = {
        "schema": "multi-challenger-shortlist-supply-audit-v1",
        "control_scores": score(control, gold_rows),
        "full_action_oracle_scores": score(full_oracle, gold_rows),
        "shortlist_oracle_scores": score(shortlist_oracle, gold_rows),
        "relations": by_relation,
        "minimum_gain_capture": min(
            row["gain_capture"] for row in by_relation.values()),
        "shortlist_gate_passed": min(
            row["gain_capture"] for row in by_relation.values())
        >= MIN_SHORTLIST_GAIN_CAPTURE,
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "gold": str(gold_path),
        "gold_sha256": sha256(gold_path),
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "development_only": True,
        "deployable": False,
    }
    result_path = output / "SUPPLY_AUDIT.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path, plan = _validated_plan(output)
    responses = _validated_responses(plan)
    review_plan = _json(Path(plan["review_run"]) / "plan/PLAN.json")
    graphs = read_jsonl(Path(plan["action_registry"]))
    shortlists = read_jsonl(Path(plan["shortlists"]))
    if len(graphs) != len(shortlists):
        raise ContractError("graph/shortlist row mismatch")
    gold_rows = read_jsonl(Path(args.gold).resolve())
    gold_by = {_key(row): row for row in gold_rows}
    control = read_jsonl(Path(review_plan["starting_predictions"]))
    control_by = {_key(row): row for row in control}
    if (
        {_key(graph) for graph in graphs}
        != set(gold_by)
        or set(control_by) != set(gold_by)
    ):
        raise ContractError("analysis row coverage mismatch")

    artifacts = {}
    arm_results = {}
    failure_ledgers = {}
    for arm in ARMS:
        predictions = []
        diagnostics = []
        ledger = {
            "no_helpful_action_in_graph": 0,
            "helpful_action_discarded_by_shortlist": 0,
            "helpful_action_in_shortlist_but_not_selected": 0,
            "harmful_action_accepted": 0,
            "helpful_action_selected": 0,
        }
        for graph, shortlist in zip(graphs, shortlists, strict=True):
            key = _key(graph)
            if key != _key(shortlist):
                raise ContractError("graph/shortlist order mismatch")
            actions = {
                str(action["id"]): action for action in graph["actions"]}
            keep = actions[str(shortlist["keep_action_id"])]
            shortlisted = [
                actions[str(action_id)]
                for action_id in shortlist["challenger_action_ids"]]
            qwen_margins = {}
            gemma_margins = {}
            qwen_groups = {}
            gemma_groups = {}
            for group_index, group in enumerate(shortlist["groups"]):
                qwen = responses[(key, QWEN, group_index)]
                gemma = responses[(key, GEMMA, group_index)]
                qwen_groups[str(group_index)] = dict(qwen)
                gemma_groups[str(group_index)] = dict(gemma)
                keep_id = str(keep["id"])
                for action_id in map(str, group):
                    qwen_margins[action_id] = (
                        math.log(max(qwen[action_id], EPSILON))
                        - math.log(max(qwen[keep_id], EPSILON))
                    )
                    gemma_margins[action_id] = (
                        math.log(max(gemma[action_id], EPSILON))
                        - math.log(max(gemma[keep_id], EPSILON))
                    )
            selected_id = _aggregate(
                arm, qwen_margins, gemma_margins, str(keep["id"]))
            if selected_id not in actions:
                raise ContractError(
                    f"{key}: response selected unknown action")
            selected = actions[selected_id]
            before = _utility(graph, keep, gold_by)
            after = _utility(graph, selected, gold_by)
            all_deltas = [
                _utility(graph, action, gold_by) - before
                for action in graph["actions"]]
            shortlist_deltas = [
                _utility(graph, action, gold_by) - before
                for action in shortlisted]
            full_best = max(all_deltas, default=0.0)
            shortlist_best = max([0.0, *shortlist_deltas])
            delta = after - before
            if full_best <= EPSILON:
                ledger["no_helpful_action_in_graph"] += 1
            elif shortlist_best <= EPSILON:
                ledger["helpful_action_discarded_by_shortlist"] += 1
            elif delta <= EPSILON:
                ledger["helpful_action_in_shortlist_but_not_selected"] += 1
            if selected is not keep and delta < -EPSILON:
                ledger["harmful_action_accepted"] += 1
            if selected is not keep and delta > EPSILON:
                ledger["helpful_action_selected"] += 1
            predictions.append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": list(selected["objects"]),
            })
            diagnostics.append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "arm": arm,
                "selected_action_id": str(selected["id"]),
                "keep_action_id": str(keep["id"]),
                "changed": selected is not keep,
                "utility_delta": delta,
                "full_oracle_delta": full_best,
                "shortlist_oracle_delta": shortlist_best,
                "shortlist_contains_best_improvement": (
                    shortlist_best >= full_best - EPSILON),
                "qwen_margins": qwen_margins,
                "gemma_margins": gemma_margins,
                "qwen_group_probabilities": qwen_groups,
                "gemma_group_probabilities": gemma_groups,
            })
        prediction_path = output / f"analysis/{arm}_PREDICTIONS.jsonl"
        diagnostic_path = output / f"analysis/{arm}_DIAGNOSTICS.jsonl"
        write_jsonl_atomic(prediction_path, predictions)
        write_jsonl_atomic(diagnostic_path, diagnostics)
        artifacts[arm] = {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
        }
        arm_results[arm] = _result_summary(
            predictions, control, diagnostics, gold_rows)
        failure_ledgers[arm] = ledger

    # Supply quality is independent of the aggregation arm.
    primary_diagnostics = read_jsonl(
        Path(artifacts[PRIMARY_ARM]["diagnostics"]))
    relations = sorted({row["Relation"] for row in primary_diagnostics})
    shortlist_supply = {}
    for relation in relations:
        subset = [
            row for row in primary_diagnostics
            if row["Relation"] == relation]
        full_gain = sum(float(row["full_oracle_delta"]) for row in subset)
        shortlist_gain = sum(
            float(row["shortlist_oracle_delta"]) for row in subset)
        shortlist_supply[relation] = {
            "rows": len(subset),
            "full_oracle_gain": full_gain,
            "shortlist_oracle_gain": shortlist_gain,
            "gain_capture": (
                shortlist_gain / full_gain if full_gain > EPSILON else 1.0),
            "full_oracle_rows": sum(
                float(row["full_oracle_delta"]) > EPSILON for row in subset),
            "shortlist_oracle_rows": sum(
                float(row["shortlist_oracle_delta"]) > EPSILON
                for row in subset),
        }

    primary = arm_results[PRIMARY_ARM]
    help_harm_ratio = (
        float(primary["helped_rows"]) / max(1, primary["harmed_rows"]))
    passed = bool(
        primary["incremental_delta"] >= MIN_INCREMENTAL_DELTA
        and help_harm_ratio >= MIN_HELP_HARM_RATIO
        and min(primary["relation_deltas"].values()) >= MIN_RELATION_DELTA
        and min(
            item["gain_capture"]
            for item in shortlist_supply.values())
        >= MIN_SHORTLIST_GAIN_CAPTURE
    )
    result = {
        "schema": "multi-challenger-graph-decoder-result-v1",
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "primary_arm": PRIMARY_ARM,
        "arms": arm_results,
        "artifacts": artifacts,
        "failure_ledgers": failure_ledgers,
        "shortlist_supply": shortlist_supply,
        "promotion_gate": {
            "passed": passed,
            "minimum_incremental_delta": MIN_INCREMENTAL_DELTA,
            "minimum_help_harm_ratio": MIN_HELP_HARM_RATIO,
            "minimum_relation_delta": MIN_RELATION_DELTA,
            "minimum_shortlist_gain_capture": MIN_SHORTLIST_GAIN_CAPTURE,
            "observed_help_harm_ratio": help_harm_ratio,
        },
        "next_stage": (
            "freeze_validation_confirmation"
            if passed else "reject_or_revise_multiway_reviewer"),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "gold": str(Path(args.gold).resolve()),
        "gold_sha256": sha256(Path(args.gold).resolve()),
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "development_only": True,
        "deployable": False,
    }
    result_path = output / "RESULT.json"
    _write_json(result_path, result)
    markdown_path = output / "RESULT.md"
    _write_result_markdown(markdown_path, result)
    print(json.dumps({
        "primary_arm": PRIMARY_ARM,
        "primary": primary,
        "promotion_gate": result["promotion_gate"],
        "shortlist_supply": shortlist_supply,
        "result": str(result_path),
        "result_markdown": str(markdown_path),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--review-run", default=str(DEFAULT_REVIEW_RUN))
    prepare_parser.add_argument(
        "--truth-run", default=str(DEFAULT_TRUTH_RUN))
    prepare_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.set_defaults(function=prepare)
    supply_parser = subparsers.add_parser("audit-supply")
    supply_parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    supply_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT))
    supply_parser.set_defaults(function=audit_supply)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    analyze_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT))
    analyze_parser.set_defaults(function=analyze)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
