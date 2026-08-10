#!/usr/bin/env python3
"""Cross-fitted action decoding for candidate-conditioned truth evidence.

Raw TRUE/FALSE logits from the heterogeneous memories rank candidate
components well but are not calibrated probabilities.  This module repairs
that decoder without consulting validation:

* one shared logistic calibrator maps Qwen/Gemma truth evidence to component
  correctness probabilities;
* calibration is stacked: action-model training rows receive probabilities
  from a calibrator that excluded their subjects;
* the accepted upstream selector owns cardinality and abstention;
* a fixed-cardinality state graph supplies identity substitutions only;
* a within-question pairwise ranker proposes candidate identities;
* one shared expected-utility gate learns from label-free Qwen/Gemma/route
  counterfactual incumbents whether replacing the incumbent improves F1;
* outer subject-grouped folds measure end-to-end row F1.

There are no per-relation models, relation-specific thresholds, or validation
labels.  Relation identity is a feature so one pooled model can learn that
the same raw score has different reliability under different fact semantics.
The fixed-cardinality factorization is important: candidate-truth evidence
answers "which candidate?", not the distinct question "does a fact exist?".
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.candidate_truth_evidence import (
    AGENTS as TRUTH_AGENTS,
    _truth_scores,
    _validated_responses,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    COMPETITION_VALIDATION,
    DEFAULT_TRAIN_OOF,
    validate_registered_predictions,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    DEFAULT_AGENTS,
    DEFAULT_GOLD,
    DEFAULT_GRAPH,
    FEATURE_NAMES,
    OUTER_FOLDS,
    PARAMETER_CAP,
    RELATIONS,
    ROW_GATE_FEATURE_NAMES,
    RUNS,
    _agent_parameter_total,
    _component_values,
    _deployment_gate,
    _key,
    _legal_actions,
    _relation_deltas,
    action_features,
    build_hierarchical_row,
    grouped_relation_folds,
    relation_family,
    row_gate_features,
)
from experiments.heterogeneous_agents.components.walking_memory_graph_selector import (
    MAX_WALK_STEPS,
    expand_states,
    state_key,
    walk_with_chooser,
)


DEFAULT_TRUTH_PLAN = (
    RUNS / "candidate_truth_evidence_20260727_v1/plan/PLAN.json")
DEFAULT_LIKELIHOOD_RUN = (
    RUNS / "likelihood_evidence_full_20260725_v1")
DEFAULT_VALIDATION_LIKELIHOOD_RUN = (
    RUNS / "likelihood_evidence_full_validation_20260725_v1")
DEFAULT_VALIDATION_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/"
    "graphs/validation_graph.jsonl")
DEFAULT_VALIDATION_TRUTH_RUN = (
    RUNS / "candidate_truth_evidence_validation_20260727_v1")
DEFAULT_CONTROL = DEFAULT_TRAIN_OOF
DEFAULT_STARTING_PREDICTIONS = COMPETITION_VALIDATION
DEFAULT_OUTPUT = RUNS / "truth_calibrated_action_decoder_20260727_v1"

INNER_CALIBRATION_FOLDS = 3
CALIBRATOR_L2 = 10.0
ACTION_L2 = 100.0
MIN_INCREMENTAL_DELTA = 0.005
DECISION_MODES = ("fixed-cardinality", "free")
DECODER_MODES = (
    "counterfactual-gated-rank",
    "set-aware-gated-rank",
    "direct-utility-action",
    "pairwise-component-rank", "calibrated-rank", "learned-action")
COUNTERFACTUAL_GATE_L2 = 10.0
DIRECT_UTILITY_L2 = 100.0
SET_RANK_HIDDEN = 16
SET_RANK_EPOCHS = 120
SET_RANK_L2 = 1e-3


LIKELIHOOD_FEATURE_NAMES = (
    "likelihood_available",
    "qwen_subject_rank", "qwen_subject_z",
    "qwen_masked_rank", "qwen_masked_z",
    "qwen_pmi_rank", "qwen_pmi_z",
    "gemma_subject_rank", "gemma_subject_z",
    "gemma_masked_rank", "gemma_masked_z",
    "gemma_pmi_rank", "gemma_pmi_z",
    "mean_subject_rank", "minimum_subject_rank",
    "subject_rank_disagreement", "subject_top_agreement",
    "mean_pmi_rank", "minimum_pmi_rank",
    "pmi_rank_disagreement", "pmi_top_agreement",
)

CALIBRATOR_FEATURE_NAMES = (
    "qwen_true", "gemma_true",
    "qwen_logit", "gemma_logit",
    "mean_true", "minimum_true", "maximum_true",
    "model_disagreement", "model_product",
    "proposed_by_qwen", "proposed_by_gemma", "proposed_by_both",
    *(f"relation:{relation}" for relation in RELATIONS),
    "qwen_x_numeric", "gemma_x_numeric",
    "qwen_x_single", "gemma_x_single",
    "qwen_x_list", "gemma_x_list",
    *LIKELIHOOD_FEATURE_NAMES,
)

TRUTH_ACTION_FEATURE_NAMES = (
    "truth_output_max", "truth_output_mean", "truth_output_min",
    "truth_incumbent_max", "truth_incumbent_mean", "truth_incumbent_min",
    "truth_added_max", "truth_added_mean", "truth_added_min",
    "truth_removed_max", "truth_removed_mean", "truth_removed_min",
    "truth_output_expected_true", "truth_output_expected_false",
    "truth_added_expected_true", "truth_added_expected_false",
    "truth_removed_expected_true", "truth_removed_expected_false",
    "truth_challenger_advantage",
    "truth_replace_gain_proxy", "truth_add_gain_proxy",
    "truth_drop_gain_proxy", "truth_empty_gain_proxy",
    "truth_added_qwen_max", "truth_added_gemma_max",
    "truth_removed_qwen_max", "truth_removed_gemma_max",
    "truth_added_model_disagreement",
    "truth_removed_model_disagreement",
)

TRUTH_GATE_FEATURE_NAMES = (
    "truth_candidate_max", "truth_candidate_mean", "truth_candidate_min",
    "truth_incumbent_max", "truth_incumbent_mean", "truth_incumbent_min",
    "truth_challenger_max", "truth_challenger_advantage",
    "truth_candidate_above_half_rate",
    "truth_qwen_candidate_max", "truth_gemma_candidate_max",
    "truth_candidate_model_disagreement",
)

ACTION_FEATURE_NAMES = FEATURE_NAMES + TRUTH_ACTION_FEATURE_NAMES
GATE_FEATURE_NAMES = ROW_GATE_FEATURE_NAMES + TRUTH_GATE_FEATURE_NAMES
PAIRWISE_FEATURE_NAMES = (
    *(f"action:{name}" for name in ACTION_FEATURE_NAMES),
    *(f"delta_from_keep:{name}" for name in ACTION_FEATURE_NAMES),
)
RANK_GATE_FEATURE_NAMES = (
    *PAIRWISE_FEATURE_NAMES,
    "rank_proposal_max", "rank_proposal_mean", "rank_proposal_min",
    "rank_incumbent_max", "rank_incumbent_mean", "rank_incumbent_min",
    "rank_max_advantage", "rank_mean_advantage",
    "rank_proposal_above_incumbent_min_rate",
    "proposal_incumbent_overlap_rate",
    *(f"relation:{relation}" for relation in RELATIONS),
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return math.log(clipped / (1.0 - clipped))


class StandardizedLinear:
    """Deterministic weighted ridge or logistic model with named features."""

    def __init__(
        self, feature_names: Sequence[str], l2: float, *, logistic: bool,
    ):
        self.feature_names = tuple(str(item) for item in feature_names)
        self.l2 = float(l2)
        self.logistic = bool(logistic)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    @property
    def parameter_count(self) -> int:
        return len(self.feature_names) + 1

    def fit(
        self, x: Sequence[Sequence[float]], y: Sequence[float],
        weights: Sequence[float],
    ) -> "StandardizedLinear":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        if (
            matrix.shape != (len(target), len(self.feature_names))
            or weight.shape != target.shape
            or len(target) < 2
            or np.any(weight <= 0)
            or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(target))
        ):
            raise ValueError("invalid standardized linear training arrays")
        if self.logistic and (
            not set(np.unique(target)) <= {0.0, 1.0}
            or len(np.unique(target)) != 2
        ):
            raise ValueError("logistic target must contain both classes")
        weight = weight * (len(weight) / weight.sum())
        self.mean = np.average(matrix, axis=0, weights=weight)
        variance = np.average(
            (matrix - self.mean) ** 2, axis=0, weights=weight)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = np.column_stack(
            [np.ones(len(target)), (matrix - self.mean) / self.scale])
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.l2
        penalty[0, 0] = 0.0
        if self.logistic:
            beta = np.zeros(design.shape[1], dtype=np.float64)
            for _ in range(100):
                logits = np.clip(design @ beta, -30.0, 30.0)
                probability = 1.0 / (1.0 + np.exp(-logits))
                curvature = np.maximum(
                    probability * (1.0 - probability), 1e-8)
                gradient = design.T @ (
                    weight * (probability - target)) + penalty @ beta
                hessian = (
                    design.T @ (design * (weight * curvature)[:, None])
                    + penalty)
                step = np.linalg.solve(hessian, gradient)
                beta -= step
                if float(np.max(np.abs(step))) < 1e-9:
                    break
            self.coef = beta
        else:
            root = np.sqrt(weight)[:, None]
            self.coef = np.linalg.solve(
                (design * root).T @ (design * root) + penalty,
                (design * root).T @ (target * root[:, 0]),
            )
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("standardized linear prediction shape mismatch")
        design = np.column_stack(
            [np.ones(len(matrix)), (matrix - self.mean) / self.scale])
        values = design @ self.coef
        if self.logistic:
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
        return values

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("model is not fitted")
        return {
            "schema": "named-standardized-linear-v1",
            "feature_names": list(self.feature_names),
            "l2": self.l2,
            "logistic": self.logistic,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coef.tolist(),
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def from_dict(cls, artifact: Mapping[str, Any]) -> "StandardizedLinear":
        if artifact.get("schema") != "named-standardized-linear-v1":
            raise ContractError("invalid standardized linear artifact")
        model = cls(
            artifact["feature_names"], float(artifact["l2"]),
            logistic=bool(artifact["logistic"]),
        )
        model.mean = np.asarray(artifact["mean"], dtype=np.float64)
        model.scale = np.asarray(artifact["scale"], dtype=np.float64)
        model.coef = np.asarray(artifact["coefficients"], dtype=np.float64)
        expected = len(model.feature_names)
        if (
            model.mean.shape != (expected,)
            or model.scale.shape != (expected,)
            or model.coef.shape != (expected + 1,)
            or np.any(model.scale <= 0)
            or not np.all(np.isfinite(model.mean))
            or not np.all(np.isfinite(model.scale))
            or not np.all(np.isfinite(model.coef))
            or int(artifact.get("parameter_count", -1))
            != model.parameter_count
        ):
            raise ContractError("invalid standardized linear parameters")
        return model


class _DeepSetScorer(nn.Module):
    """Small permutation-equivariant scorer for one question's components."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(values)
        context = encoded.mean(dim=0, keepdim=True).expand_as(encoded)
        combined = torch.cat(
            [encoded, context, encoded - context], dim=1)
        return self.score(combined).squeeze(1)


class SetAwareRanker:
    """Permutation-equivariant within-question component ranker."""

    def __init__(
        self, feature_names: Sequence[str], hidden_size: int = SET_RANK_HIDDEN,
    ):
        self.feature_names = tuple(str(item) for item in feature_names)
        self.hidden_size = int(hidden_size)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.state: dict[str, torch.Tensor] | None = None

    @property
    def parameter_count(self) -> int:
        model = _DeepSetScorer(len(self.feature_names), self.hidden_size)
        return sum(item.numel() for item in model.parameters())

    def _model(self) -> _DeepSetScorer:
        model = _DeepSetScorer(len(self.feature_names), self.hidden_size)
        if self.state is not None:
            model.load_state_dict(self.state)
        return model

    def fit(
        self, rows: Sequence[tuple[Sequence[Sequence[float]],
                                   Sequence[float]]], *, seed: int,
    ) -> "SetAwareRanker":
        matrices = [
            np.asarray(features, dtype=np.float64)
            for features, _ in rows if features
        ]
        if (
            not matrices
            or any(
                matrix.ndim != 2
                or matrix.shape[1] != len(self.feature_names)
                or not np.all(np.isfinite(matrix))
                for matrix in matrices)
        ):
            raise ValueError("invalid set-aware ranker rows")
        flat = np.concatenate(matrices, axis=0)
        self.mean = flat.mean(axis=0)
        self.scale = flat.std(axis=0)
        self.scale = np.where(self.scale > 1e-12, self.scale, 1.0)
        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True)
        model = self._model().double()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.01, weight_decay=SET_RANK_L2)
        prepared = []
        for features, labels in rows:
            if not features:
                continue
            matrix = np.asarray(features, dtype=np.float64)
            target = np.asarray(labels, dtype=np.float64)
            if target.shape != (len(matrix),):
                raise ValueError("set-aware target shape mismatch")
            positive = np.flatnonzero(target > 0.5)
            negative = np.flatnonzero(target <= 0.5)
            if len(positive) and len(negative):
                prepared.append((
                    torch.from_numpy(
                        (matrix - self.mean) / self.scale).double(),
                    torch.from_numpy(positive),
                    torch.from_numpy(negative),
                ))
        if len(prepared) < 2:
            raise ValueError("insufficient informative set-aware rows")
        for _ in range(SET_RANK_EPOCHS):
            optimizer.zero_grad()
            losses = []
            for values, positive, negative in prepared:
                scores = model(values)
                differences = (
                    scores[positive][:, None] - scores[negative][None, :])
                losses.append(torch.nn.functional.softplus(
                    -differences).mean())
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
        self.state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        return self

    def predict_rows(
        self, rows: Sequence[Sequence[Sequence[float]]],
    ) -> list[np.ndarray]:
        if self.mean is None or self.scale is None or self.state is None:
            raise RuntimeError("set-aware ranker is not fitted")
        model = self._model().double().eval()
        output = []
        with torch.no_grad():
            for features in rows:
                if not features:
                    output.append(np.asarray([], dtype=np.float64))
                    continue
                matrix = np.asarray(features, dtype=np.float64)
                values = torch.from_numpy(
                    (matrix - self.mean) / self.scale).double()
                output.append(model(values).cpu().numpy())
        return output

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None or self.state is None:
            raise RuntimeError("set-aware ranker is not fitted")
        return {
            "schema": "set-aware-component-ranker-v1",
            "feature_names": list(self.feature_names),
            "hidden_size": self.hidden_size,
            "architecture": "DeepSets(candidate,mean,candidate-minus-mean)",
            "training_loss": "row-balanced pairwise logistic",
            "epochs": SET_RANK_EPOCHS,
            "weight_decay": SET_RANK_L2,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "state": {
                key: value.numpy().tolist()
                for key, value in self.state.items()
            },
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def from_dict(cls, artifact: Mapping[str, Any]) -> "SetAwareRanker":
        if artifact.get("schema") != "set-aware-component-ranker-v1":
            raise ContractError("invalid set-aware ranker artifact")
        model = cls(
            artifact["feature_names"], int(artifact["hidden_size"]))
        model.mean = np.asarray(artifact["mean"], dtype=np.float64)
        model.scale = np.asarray(artifact["scale"], dtype=np.float64)
        reference = model._model().state_dict()
        state = {
            key: torch.as_tensor(
                artifact["state"][key], dtype=reference[key].dtype)
            for key in reference
        }
        model.state = state
        if (
            model.mean.shape != (len(model.feature_names),)
            or model.scale.shape != model.mean.shape
            or np.any(model.scale <= 0)
            or int(artifact.get("parameter_count", -1))
            != model.parameter_count
        ):
            raise ContractError("invalid set-aware ranker parameters")
        model._model()
        return model


def load_truth_records(
    plan_path: Path,
    *, expected_split: str = "train",
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    """Load label-free truth evidence from task/response artifacts."""
    plan = _json(plan_path)
    if (
        plan.get("schema") != "candidate-truth-evidence-plan-v1"
        or plan.get("split", "train") != expected_split
        or plan.get("contains_labels")
        or plan.get("gold_aware")
        or plan.get("validation_labels_used", False)
        or sha256(Path(plan["inventory"])) != plan["inventory_sha256"]
        or sha256(Path(plan.get("graph", plan.get("train_graph"))))
        != plan.get("graph_sha256", plan.get("train_graph_sha256"))
    ):
        raise ContractError("invalid label-free candidate truth plan")
    responses, tasks = _validated_responses(plan)
    scores = _truth_scores(responses, tasks)
    inventory = read_jsonl(Path(plan["inventory"]))
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for component in inventory:
        component_key = str(component["component_key"])
        key = (
            str(component["SubjectEntity"]),
            str(component["Relation"]),
            str(component["component_id"]),
        )
        if key in records or component_key not in scores:
            raise ContractError("candidate truth inventory coverage failure")
        records[key] = {
            "component_key": component_key,
            "proposer_agents": list(component["proposer_agents"]),
            "raw": dict(scores[component_key]),
        }
    if len(records) != int(plan["components"]):
        raise ContractError("candidate truth component count mismatch")
    return records, plan


def _within_row_rank_z(
    values: Mapping[str, float],
) -> dict[str, tuple[float, float]]:
    """Normalize scores without comparing logits across models or rows."""
    if not values:
        return {}
    ordered = list(values.values())
    mean = statistics.mean(ordered)
    std = statistics.pstdev(ordered)
    denominator = max(len(ordered) - 1, 1)
    output = {}
    for key, value in values.items():
        # Ties deliberately receive the same rank.  This avoids arbitrary
        # component-id ordering becoming evidence.
        rank = (
            sum(candidate < value for candidate in ordered)
            + 0.5 * (sum(candidate == value for candidate in ordered) - 1)
        ) / denominator
        z = (
            max(-1.0, min(1.0, (value - mean) / (3.0 * std)))
            if std > 1e-12 else 0.0
        )
        output[key] = (float(rank), float(z))
    return output


def attach_likelihood_records(
    records: dict[tuple[str, str, str], dict[str, Any]],
    graphs: Sequence[Mapping[str, Any]],
    run_dir: Path,
    graph_path: Path,
    *, expected_split: str = "train",
) -> dict[str, Any]:
    """Attach label-free subject/masked likelihood messages to components.

    Scores are reduced to within-question ranks and z-scores.  Consequently,
    Qwen/Gemma tokenizer scales and answer-length priors cannot be mistaken
    for cross-row confidence.  The subject-minus-masked PMI feature measures
    whether naming the subject specifically raises support for a candidate.
    """
    run_dir = Path(run_dir).resolve()
    plan_path = run_dir / "plan/PLAN.json"
    if not plan_path.is_file():
        raise ContractError("missing likelihood evidence plan")
    plan = _json(plan_path)
    if (
        plan.get("schema") != "likelihood-evidence-plan-v1"
        or plan.get("split") != expected_split
        or plan.get("labels_in_model_tasks") is not False
        or plan.get("source_train_graph_sha256") != sha256(graph_path)
        or not {"subject", "masked"} <= set(plan.get("contexts", []))
    ):
        raise ContractError("invalid likelihood evidence provenance")

    responses: dict[
        tuple[str, str, str, str], Mapping[str, Any]
    ] = {}
    for agent in (QWEN, GEMMA):
        job = plan.get("jobs", {}).get(agent)
        if not isinstance(job, dict):
            raise ContractError(f"likelihood plan missing {agent} job")
        task_path = run_dir / f"plan/tasks/{agent}.jsonl"
        response_path = run_dir / f"responses/{agent}.jsonl"
        manifest_path = response_path.with_suffix(
            response_path.suffix + ".manifest.json")
        if not (
            task_path.is_file()
            and response_path.is_file()
            and manifest_path.is_file()
        ):
            raise ContractError(f"incomplete likelihood artifacts: {agent}")
        manifest = _json(manifest_path)
        tasks = read_jsonl(task_path)
        rows = read_jsonl(response_path)
        if (
            manifest.get("schema") != "likelihood-evidence-responses-v1"
            or manifest.get("agent_id") != agent
            or manifest.get("task_sha256") != sha256(task_path)
            or manifest.get("output_sha256") != sha256(response_path)
            or job.get("task_sha256") != sha256(task_path)
            or int(job.get("tasks", -1)) != len(tasks)
            or int(manifest.get("tasks", -1)) != len(rows)
            or {str(row["task_id"]) for row in tasks}
            != {str(row["task_id"]) for row in rows}
        ):
            raise ContractError(
                f"invalid likelihood task/response contract: {agent}")
        for row in rows:
            context = str(row.get("context"))
            if context not in {"subject", "masked"}:
                continue
            key = (
                agent, str(row["subject"]), str(row["relation"]), context)
            if key in responses:
                raise ContractError(f"duplicate likelihood response: {key}")
            responses[key] = row

    covered_rows = covered_components = 0
    covered_relations: Counter[str] = Counter()
    for graph in graphs:
        subject, relation = _key(graph)
        by_agent = {}
        for agent in (QWEN, GEMMA):
            subject_row = responses.get(
                (agent, subject, relation, "subject"))
            masked_row = responses.get(
                (agent, subject, relation, "masked"))
            if (subject_row is None) != (masked_row is None):
                raise ContractError(
                    f"partial likelihood contexts: {agent}/{subject}/{relation}")
            if subject_row is not None:
                by_agent[agent] = (subject_row, masked_row)
        if not by_agent:
            continue
        if set(by_agent) != {QWEN, GEMMA}:
            raise ContractError(
                f"partial likelihood agents: {subject}/{relation}")

        components = graph["_source"]["relational_graph"]["components"]
        raw_by_agent: dict[
            str, dict[str, dict[str, float]]
        ] = {}
        for agent, (subject_row, masked_row) in by_agent.items():
            subject_scores = subject_row.get("continuation_scores", {})
            masked_scores = masked_row.get("continuation_scores", {})
            component_scores = {
                "subject": {}, "masked": {}, "pmi": {},
            }
            for component in components:
                component_id = str(component["id"])
                surface_values = {
                    "subject": [], "masked": [], "pmi": [],
                }
                for raw_item in component.get("member_items", []):
                    item = str(raw_item)
                    text = f"ANSWER: {item}"
                    subject_entry = subject_scores.get(text)
                    masked_entry = masked_scores.get(text)
                    if not (
                        isinstance(subject_entry, Mapping)
                        and isinstance(masked_entry, Mapping)
                    ):
                        continue
                    subject_count = int(subject_entry.get("token_count", 0))
                    masked_count = int(masked_entry.get("token_count", 0))
                    subject_sum = float(
                        subject_entry.get("sum_logprob", math.nan))
                    masked_sum = float(
                        masked_entry.get("sum_logprob", math.nan))
                    if (
                        subject_count < 1 or masked_count < 1
                        or not math.isfinite(subject_sum)
                        or not math.isfinite(masked_sum)
                    ):
                        raise ContractError(
                            f"invalid likelihood score: "
                            f"{agent}/{subject}/{relation}/{item}")
                    surface_values["subject"].append(
                        subject_sum / subject_count)
                    surface_values["masked"].append(
                        masked_sum / masked_count)
                    surface_values["pmi"].append(
                        subject_sum - masked_sum)
                if not surface_values["subject"]:
                    raise ContractError(
                        f"unscored likelihood component: "
                        f"{agent}/{subject}/{relation}/{component_id}")
                for signal in component_scores:
                    # Alias surfaces are alternative spellings, so the best
                    # supported spelling represents the component.
                    component_scores[signal][component_id] = max(
                        surface_values[signal])
            raw_by_agent[agent] = component_scores

        normalized = {
            agent: {
                signal: _within_row_rank_z(values)
                for signal, values in signals.items()
            }
            for agent, signals in raw_by_agent.items()
        }
        for component in components:
            component_id = str(component["id"])
            likelihood = {"available": 1.0}
            for agent in (QWEN, GEMMA):
                likelihood[agent] = {}
                for signal in ("subject", "masked", "pmi"):
                    rank, z = normalized[agent][signal][component_id]
                    likelihood[agent][signal] = {
                        "rank": rank, "z": z,
                    }
            records[_record_key(graph, component)]["likelihood"] = likelihood
            covered_components += 1
        covered_rows += 1
        covered_relations[relation] += 1
    return {
        "run_dir": str(run_dir),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "rows": covered_rows,
        "components": covered_components,
        "relation_rows": dict(covered_relations),
        "normalization": "within_question_rank_and_z",
        "alias_aggregation": "max",
        "subject_masked_pmi": True,
        "labels_in_evidence": False,
    }


def _record_key(
    graph: Mapping[str, Any], component: Mapping[str, Any],
) -> tuple[str, str, str]:
    return (
        str(graph["SubjectEntity"]),
        str(graph["Relation"]),
        str(component["id"]),
    )


def validate_graph_truth_coverage(
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> None:
    expected = {
        _record_key(graph, component)
        for graph in graphs
        for component in graph["_source"]["relational_graph"]["components"]
    }
    if expected != set(records):
        missing = len(expected - set(records))
        extra = len(set(records) - expected)
        raise ContractError(
            f"truth/graph component mismatch: missing={missing}, extra={extra}")


def calibrator_features(
    relation: str, record: Mapping[str, Any],
) -> list[float]:
    raw = record["raw"]
    q = float(raw[QWEN])
    g = float(raw[GEMMA])
    proposers = set(record["proposer_agents"])
    family = relation_family(relation)
    likelihood = record.get("likelihood", {})
    available = float(likelihood.get("available", 0.0))
    likelihood_values = {}
    for agent in (QWEN, GEMMA):
        agent_values = likelihood.get(agent, {})
        for signal in ("subject", "masked", "pmi"):
            signal_values = agent_values.get(signal, {})
            likelihood_values[(agent, signal, "rank")] = float(
                signal_values.get("rank", 0.0))
            likelihood_values[(agent, signal, "z")] = float(
                signal_values.get("z", 0.0))
    q_subject = likelihood_values[(QWEN, "subject", "rank")]
    g_subject = likelihood_values[(GEMMA, "subject", "rank")]
    q_pmi = likelihood_values[(QWEN, "pmi", "rank")]
    g_pmi = likelihood_values[(GEMMA, "pmi", "rank")]
    values = [
        q, g, _logit(q), _logit(g),
        (q + g) / 2.0, min(q, g), max(q, g), abs(q - g), q * g,
        float(QWEN in proposers), float(GEMMA in proposers),
        float(proposers == set(TRUTH_AGENTS)),
        *(float(relation == item) for item in RELATIONS),
        q * float(family == "numeric"), g * float(family == "numeric"),
        q * float(family == "single"), g * float(family == "single"),
        q * float(family == "list"), g * float(family == "list"),
        available,
        likelihood_values[(QWEN, "subject", "rank")],
        likelihood_values[(QWEN, "subject", "z")],
        likelihood_values[(QWEN, "masked", "rank")],
        likelihood_values[(QWEN, "masked", "z")],
        likelihood_values[(QWEN, "pmi", "rank")],
        likelihood_values[(QWEN, "pmi", "z")],
        likelihood_values[(GEMMA, "subject", "rank")],
        likelihood_values[(GEMMA, "subject", "z")],
        likelihood_values[(GEMMA, "masked", "rank")],
        likelihood_values[(GEMMA, "masked", "z")],
        likelihood_values[(GEMMA, "pmi", "rank")],
        likelihood_values[(GEMMA, "pmi", "z")],
        (q_subject + g_subject) / 2.0,
        min(q_subject, g_subject),
        abs(q_subject - g_subject),
        float(available and q_subject >= 1.0 and g_subject >= 1.0),
        (q_pmi + g_pmi) / 2.0,
        min(q_pmi, g_pmi),
        abs(q_pmi - g_pmi),
        float(available and q_pmi >= 1.0 and g_pmi >= 1.0),
    ]
    if len(values) != len(CALIBRATOR_FEATURE_NAMES):
        raise AssertionError("truth calibrator feature schema drift")
    return values


def _component_label(
    graph: Mapping[str, Any], component: Mapping[str, Any],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    relation = str(graph["Relation"])
    return float(_row_f1(
        [str(component["representative"])],
        gold_by[_key(graph)],
        relation,
    ) > 0.0)


def fit_calibrator(
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> StandardizedLinear:
    x, y, weights = [], [], []
    for graph in graphs:
        components = graph["_source"]["relational_graph"]["components"]
        row_weight = 1.0 / max(len(components), 1)
        for component in components:
            record = records[_record_key(graph, component)]
            x.append(calibrator_features(str(graph["Relation"]), record))
            y.append(_component_label(graph, component, gold_by))
            weights.append(row_weight)
    return StandardizedLinear(
        CALIBRATOR_FEATURE_NAMES, CALIBRATOR_L2, logistic=True,
    ).fit(x, y, weights)


def predict_calibrated(
    model: StandardizedLinear,
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], float]:
    keys, x = [], []
    for graph in graphs:
        for component in graph["_source"]["relational_graph"]["components"]:
            key = _record_key(graph, component)
            keys.append(key)
            x.append(calibrator_features(
                str(graph["Relation"]), records[key]))
    values = model.predict(x)
    return {key: float(value) for key, value in zip(keys, values)}


def fit_pairwise_component_ranker(
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[StandardizedLinear, dict[str, Any]]:
    """Fit balanced within-row positive-vs-negative component contrasts."""
    x, y, weights = [], [], []
    informative_rows = pair_count = 0
    for graph in graphs:
        components = graph["_source"]["relational_graph"]["components"]
        positives = [
            item for item in components
            if _component_label(graph, item, gold_by) > 0.0]
        negatives = [
            item for item in components
            if _component_label(graph, item, gold_by) == 0.0]
        if not positives or not negatives:
            continue
        informative_rows += 1
        pairs = len(positives) * len(negatives)
        pair_weight = 1.0 / (2.0 * pairs)
        for positive in positives:
            positive_features = calibrator_features(
                str(graph["Relation"]),
                records[_record_key(graph, positive)],
            )
            for negative in negatives:
                negative_features = calibrator_features(
                    str(graph["Relation"]),
                    records[_record_key(graph, negative)],
                )
                difference = [
                    left - right for left, right in zip(
                        positive_features, negative_features)
                ]
                x.extend((difference, [-value for value in difference]))
                y.extend((1.0, 0.0))
                weights.extend((pair_weight, pair_weight))
                pair_count += 1
    model = StandardizedLinear(
        CALIBRATOR_FEATURE_NAMES, CALIBRATOR_L2, logistic=True,
    ).fit(x, y, weights)
    return model, {
        "informative_rows": informative_rows,
        "unordered_component_pairs": pair_count,
        "directed_training_pairs": len(y),
        "row_balanced": True,
        "within_question_contrasts": True,
    }


def predict_component_ranks(
    model: StandardizedLinear | SetAwareRanker,
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], float]:
    """Produce scores whose ordering follows the pairwise linear ranker."""
    if isinstance(model, SetAwareRanker):
        return predict_set_aware_component_ranks(model, graphs, records)
    return predict_calibrated(model, graphs, records)


def _set_rank_rows(
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> list[tuple[list[list[float]], list[float]]]:
    rows = []
    for graph in graphs:
        components = graph["_source"]["relational_graph"]["components"]
        features = [
            calibrator_features(
                str(graph["Relation"]),
                records[_record_key(graph, component)])
            for component in components
        ]
        labels = (
            [
                _component_label(graph, component, gold_by)
                for component in components
            ]
            if gold_by is not None else [0.0] * len(components)
        )
        rows.append((features, labels))
    return rows


def fit_set_aware_component_ranker(
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]], *, seed: int,
) -> tuple[SetAwareRanker, dict[str, Any]]:
    rows = _set_rank_rows(graphs, records, gold_by)
    informative = sum(
        bool(any(labels) and not all(labels)) for _, labels in rows)
    model = SetAwareRanker(CALIBRATOR_FEATURE_NAMES).fit(rows, seed=seed)
    return model, {
        "rows": len(rows),
        "informative_rows": informative,
        "components": sum(len(features) for features, _ in rows),
        "permutation_equivariant": True,
        "row_balanced_pairwise_loss": True,
        "hidden_size": SET_RANK_HIDDEN,
        "epochs": SET_RANK_EPOCHS,
        "weight_decay": SET_RANK_L2,
        "parameter_count": model.parameter_count,
    }


def predict_set_aware_component_ranks(
    model: SetAwareRanker,
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], float]:
    rows = _set_rank_rows(graphs, records)
    values = model.predict_rows([features for features, _ in rows])
    output: dict[tuple[str, str, str], float] = {}
    for graph, scores in zip(graphs, values):
        components = graph["_source"]["relational_graph"]["components"]
        if len(scores) != len(components):
            raise ContractError("set-aware prediction coverage failure")
        for component, value in zip(components, scores):
            output[_record_key(graph, component)] = float(value)
    return output


def crossfit_set_aware_component_ranker(
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    *, seed: int,
) -> tuple[
    dict[tuple[str, str, str], float], SetAwareRanker, dict[str, Any],
]:
    folds = grouped_relation_folds(
        graphs, INNER_CALIBRATION_FOLDS, seed=seed)
    stacked: dict[tuple[str, str, str], float] = {}
    details = []
    for fold in range(INNER_CALIBRATION_FOLDS):
        fit_rows = [row for row in graphs if folds[_key(row)] != fold]
        hold_rows = [row for row in graphs if folds[_key(row)] == fold]
        ranker, fit_detail = fit_set_aware_component_ranker(
            fit_rows, records, gold_by, seed=seed + 7919 * (fold + 1))
        predictions = predict_set_aware_component_ranks(
            ranker, hold_rows, records)
        if set(stacked) & set(predictions):
            raise ContractError("duplicate stacked set-aware prediction")
        stacked.update(predictions)
        details.append({
            "fold": fold,
            "fit_rows": len(fit_rows),
            "hold_rows": len(hold_rows),
            "fit_ranker": fit_detail,
        })
    expected = {
        _record_key(graph, component)
        for graph in graphs
        for component in graph["_source"]["relational_graph"]["components"]
    }
    if set(stacked) != expected:
        raise ContractError("stacked set-aware coverage failure")
    final, final_detail = fit_set_aware_component_ranker(
        graphs, records, gold_by, seed=seed + 99991)
    return stacked, final, {
        "subject_grouped": True,
        "each_training_score_excludes_subject": True,
        "components": len(stacked),
        "folds": details,
        "final_ranker": final_detail,
    }


def crossfit_component_ranker(
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    *, seed: int,
) -> tuple[
    dict[tuple[str, str, str], float],
    StandardizedLinear,
    dict[str, Any],
]:
    """Stack rank scores so every gate-training row excludes its subject."""
    folds = grouped_relation_folds(
        graphs, INNER_CALIBRATION_FOLDS, seed=seed)
    stacked: dict[tuple[str, str, str], float] = {}
    details = []
    for fold in range(INNER_CALIBRATION_FOLDS):
        fit_rows = [row for row in graphs if folds[_key(row)] != fold]
        hold_rows = [row for row in graphs if folds[_key(row)] == fold]
        ranker, fit_detail = fit_pairwise_component_ranker(
            fit_rows, records, gold_by)
        predictions = predict_component_ranks(
            ranker, hold_rows, records)
        overlap = set(stacked) & set(predictions)
        if overlap:
            raise ContractError(
                "duplicate stacked component rank prediction")
        stacked.update(predictions)
        details.append({
            "fold": fold,
            "fit_rows": len(fit_rows),
            "hold_rows": len(hold_rows),
            "hold_components": len(predictions),
            "fit_ranker": fit_detail,
        })
    expected = {
        _record_key(graph, component)
        for graph in graphs
        for component in graph["_source"]["relational_graph"]["components"]
    }
    if set(stacked) != expected:
        raise ContractError(
            "stacked component rank prediction coverage failure")
    final, final_detail = fit_pairwise_component_ranker(
        graphs, records, gold_by)
    return stacked, final, {
        "subject_grouped": True,
        "each_training_score_excludes_subject": True,
        "components": len(stacked),
        "folds": details,
        "final_ranker": final_detail,
    }


def _ranked_top_k(
    graph: Mapping[str, Any],
    scores: Mapping[tuple[str, str, str], float],
    cardinality: int,
) -> list[str]:
    components = graph["_source"]["relational_graph"]["components"]
    ranked = sorted(
        components,
        key=lambda component: (
            float(scores[_record_key(graph, component)]),
            str(component["representative"]),
        ),
        reverse=True,
    )
    if cardinality <= 0 or len(ranked) < cardinality:
        return []
    return [
        str(component["representative"])
        for component in ranked[:cardinality]
    ]


def counterfactual_incumbents(
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect unique label-free states actually emitted by graph routes."""
    source = graph["_source"]
    candidates: list[tuple[str, Sequence[Any]]] = [
        ("accepted", graph["incumbent_objects"]),
        ("baseline", source.get("baseline_objects", [])),
    ]
    for agent, objects in sorted(source.get("agent_outputs", {}).items()):
        candidates.append((f"agent:{agent}", objects))
    for route, metadata in sorted(source.get("proposal_routes", {}).items()):
        objects = metadata.get("objects")
        if isinstance(objects, list):
            candidates.append((f"route:{route}", objects))
        selected = [
            str(component["representative"])
            for component in source["relational_graph"]["components"]
            if component.get("routes", {}).get(route, {}).get("selected")
        ]
        if selected:
            candidates.append((f"route-selected:{route}", selected))
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for origin, raw_objects in candidates:
        if not isinstance(raw_objects, Sequence) or isinstance(
                raw_objects, (str, bytes)):
            continue
        objects = [str(item) for item in raw_objects]
        key = state_key(graph, objects)
        if key not in unique:
            unique[key] = {
                "objects": objects,
                "origins": [origin],
            }
        else:
            unique[key]["origins"].append(origin)
    return list(unique.values())


def rank_gate_features(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    proposal: Sequence[str],
    calibrated: Mapping[tuple[str, str, str], float],
    rank_scores: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[float]:
    view = fixed_cardinality_state_view(graph, incumbent)
    keep = next(
        action for action in view["actions"]
        if action["action_type"] == "KEEP")
    action = {
        "action_type": "RANK_TOP_K",
        "objects": [str(item) for item in proposal],
    }
    source = graph["_source"]
    proposal_components = _component_values(source, proposal)
    incumbent_components = _component_values(source, incumbent)

    def values(
        components: Sequence[Mapping[str, Any]],
    ) -> list[float]:
        return [
            float(rank_scores[_record_key(graph, component)])
            for component in components
        ]

    proposal_values = values(proposal_components)
    incumbent_values = values(incumbent_components)
    proposal_summary = _summary(proposal_values)
    incumbent_summary = _summary(incumbent_values)
    incumbent_min = incumbent_summary["min"]
    proposal_keys = set(state_key(graph, proposal))
    incumbent_keys = set(state_key(graph, incumbent))
    overlap_denominator = max(len(proposal_keys | incumbent_keys), 1)
    features = [
        *pairwise_action_features(
            view, action, keep, calibrated, records),
        proposal_summary["max"],
        proposal_summary["mean"],
        proposal_summary["min"],
        incumbent_summary["max"],
        incumbent_summary["mean"],
        incumbent_summary["min"],
        proposal_summary["max"] - incumbent_summary["max"],
        proposal_summary["mean"] - incumbent_summary["mean"],
        (
            sum(value > incumbent_min for value in proposal_values)
            / len(proposal_values)
            if proposal_values else 0.0
        ),
        len(proposal_keys & incumbent_keys) / overlap_denominator,
        *(float(str(graph["Relation"]) == relation)
          for relation in RELATIONS),
    ]
    if len(features) != len(RANK_GATE_FEATURE_NAMES):
        raise AssertionError("rank gate feature schema drift")
    if not all(math.isfinite(value) for value in features):
        raise ContractError("non-finite counterfactual rank gate feature")
    return features


class CounterfactualRankGate:
    """Estimate the signed F1 utility of replacing an incumbent by top-k."""

    def __init__(self, expected_delta_model: StandardizedLinear):
        self.expected_delta_model = expected_delta_model

    @property
    def parameter_count(self) -> int:
        return self.expected_delta_model.parameter_count

    def predict(
        self, features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        return self.expected_delta_model.predict(features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "counterfactual-rank-edit-gate-v1",
            "decision_rule": "E[row_f1(proposal)-row_f1(incumbent)]>0",
            "expected_delta_model": self.expected_delta_model.to_dict(),
            "parameter_count": self.parameter_count,
        }


def fit_counterfactual_rank_gate(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
    rank_scores: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[CounterfactualRankGate, dict[str, Any]]:
    x, y, weights = [], [], []
    states_per_row, changed_per_row = [], []
    origin_counts: Counter[str] = Counter()
    for graph in graphs:
        relation = str(graph["Relation"])
        gold = gold_by[_key(graph)]
        examples = []
        states = counterfactual_incumbents(graph)
        states_per_row.append(len(states))
        for state in states:
            incumbent = state["objects"]
            proposal = _ranked_top_k(
                graph, rank_scores, len(state_key(graph, incumbent)))
            if state_key(graph, proposal) == state_key(graph, incumbent):
                continue
            delta = (
                _row_f1(proposal, gold, relation)
                - _row_f1(incumbent, gold, relation))
            examples.append((
                rank_gate_features(
                    graph, incumbent, proposal,
                    calibrated, rank_scores, records),
                delta,
            ))
            origin_counts.update(state["origins"])
        changed_per_row.append(len(examples))
        if not examples:
            continue
        row_weight = 1.0 / len(examples)
        for features, delta in examples:
            x.append(features)
            y.append(float(delta))
            weights.append(row_weight)
    if len(x) < 2:
        raise ContractError("insufficient counterfactual rank-gate examples")
    model = StandardizedLinear(
        RANK_GATE_FEATURE_NAMES,
        COUNTERFACTUAL_GATE_L2,
        logistic=False,
    ).fit(x, y, weights)
    return CounterfactualRankGate(model), {
        "rows": len(graphs),
        "counterfactual_states": sum(states_per_row),
        "changed_proposals": len(x),
        "positive_deltas": sum(delta > 1e-12 for delta in y),
        "negative_deltas": sum(delta < -1e-12 for delta in y),
        "neutral_deltas": sum(abs(delta) <= 1e-12 for delta in y),
        "mean_states_per_row": statistics.mean(states_per_row),
        "mean_changed_proposals_per_row": statistics.mean(changed_per_row),
        "row_balanced": True,
        "label_free_state_origins": dict(origin_counts),
    }


def gated_rank_predictions(
    gate: CounterfactualRankGate,
    graphs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
    rank_scores: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for graph in graphs:
        incumbent = list(graph["incumbent_objects"])
        proposal = _ranked_top_k(
            graph, rank_scores, len(state_key(graph, incumbent)))
        changed = state_key(graph, proposal) != state_key(graph, incumbent)
        expected_delta = (
            float(gate.predict([rank_gate_features(
                graph, incumbent, proposal,
                calibrated, rank_scores, records)])[0])
            if changed else 0.0
        )
        accepted = changed and expected_delta > 0.0
        selected = proposal if accepted else incumbent
        predictions.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": selected,
        })
        diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "initial_objects": incumbent,
            "selected_objects": selected,
            "changed": accepted,
            "steps": int(accepted),
            "trace": [{
                "step": 0,
                "selected_action": (
                    "RANK_TOP_K" if accepted else "KEEP"),
                "proposed_objects": proposal,
                "selected_objects": selected,
                "incumbent_objects": incumbent,
                "predicted_expected_f1_delta": expected_delta,
                "gate_open": accepted,
            }],
        })
    return predictions, diagnostics


def crossfit_calibrator(
    graphs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]], *, seed: int,
) -> tuple[
    dict[tuple[str, str, str], float], StandardizedLinear, dict[str, Any],
]:
    folds = grouped_relation_folds(
        graphs, INNER_CALIBRATION_FOLDS, seed=seed)
    oof: dict[tuple[str, str, str], float] = {}
    fold_records = []
    for fold in range(INNER_CALIBRATION_FOLDS):
        fit_rows = [row for row in graphs if folds[_key(row)] != fold]
        hold_rows = [row for row in graphs if folds[_key(row)] == fold]
        model = fit_calibrator(fit_rows, records, gold_by)
        predicted = predict_calibrated(model, hold_rows, records)
        if set(oof) & set(predicted):
            raise ContractError("duplicate stacked calibration component")
        oof.update(predicted)
        fold_records.append({
            "fold": fold,
            "fit_rows": len(fit_rows),
            "hold_rows": len(hold_rows),
            "hold_components": len(predicted),
        })
    expected = {
        _record_key(graph, component)
        for graph in graphs
        for component in graph["_source"]["relational_graph"]["components"]
    }
    if set(oof) != expected:
        raise ContractError("stacked calibration coverage failure")
    final = fit_calibrator(graphs, records, gold_by)
    return oof, final, {
        "folds": fold_records,
        "components": len(oof),
        "subject_grouped": True,
        "each_training_probability_excludes_subject": True,
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "max": max(values, default=0.0),
        "mean": statistics.mean(values) if values else 0.0,
        "min": min(values, default=0.0),
        "expected_true": sum(values) / 10.0,
        "expected_false": sum(1.0 - value for value in values) / 10.0,
    }


def _truth_values(
    graph: Mapping[str, Any], components: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
) -> list[float]:
    return [
        float(calibrated[_record_key(graph, component)])
        for component in components]


def truth_action_features(
    graph: Mapping[str, Any], action: Mapping[str, Any],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[float]:
    source = graph["_source"]
    incumbent_components = _component_values(
        source, graph["incumbent_objects"])
    output_components = _component_values(source, action["objects"])
    incumbent_ids = {str(item["id"]) for item in incumbent_components}
    output_ids = {str(item["id"]) for item in output_components}
    added = [
        item for item in output_components
        if str(item["id"]) not in incumbent_ids]
    removed = [
        item for item in incumbent_components
        if str(item["id"]) not in output_ids]
    out = _summary(_truth_values(graph, output_components, calibrated))
    inc = _summary(_truth_values(graph, incumbent_components, calibrated))
    add_values = _truth_values(graph, added, calibrated)
    remove_values = _truth_values(graph, removed, calibrated)
    add = _summary(add_values)
    remove = _summary(remove_values)

    def raw_max(components: Sequence[Mapping[str, Any]], agent: str) -> float:
        return max((
            float(records[_record_key(graph, item)]["raw"][agent])
            for item in components
        ), default=0.0)

    def disagreement(components: Sequence[Mapping[str, Any]]) -> float:
        return max((
            abs(
                float(records[_record_key(graph, item)]["raw"][QWEN])
                - float(records[_record_key(graph, item)]["raw"][GEMMA])
            )
            for item in components
        ), default=0.0)

    action_type = str(action["action_type"])
    advantage = add["max"] - inc["max"]
    values = [
        out["max"], out["mean"], out["min"],
        inc["max"], inc["mean"], inc["min"],
        add["max"], add["mean"], add["min"],
        remove["max"], remove["mean"], remove["min"],
        out["expected_true"], out["expected_false"],
        add["expected_true"], add["expected_false"],
        remove["expected_true"], remove["expected_false"],
        advantage,
        float(action_type == "REPLACE") * advantage,
        float(action_type == "ADD") * (2.0 * add["mean"] - 1.0),
        float(action_type == "DROP") * (1.0 - 2.0 * remove["mean"]),
        float(action_type == "EMPTY") * (1.0 - inc["mean"]),
        raw_max(added, QWEN), raw_max(added, GEMMA),
        raw_max(removed, QWEN), raw_max(removed, GEMMA),
        disagreement(added), disagreement(removed),
    ]
    if len(values) != len(TRUTH_ACTION_FEATURE_NAMES):
        raise AssertionError("truth action feature schema drift")
    return values


def augmented_action_features(
    graph: Mapping[str, Any], action: Mapping[str, Any],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[float]:
    values = [
        *action_features(graph, action),
        *truth_action_features(graph, action, calibrated, records),
    ]
    if len(values) != len(ACTION_FEATURE_NAMES):
        raise AssertionError("augmented action feature schema drift")
    return values


def truth_gate_features(
    graph: Mapping[str, Any],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[float]:
    source = graph["_source"]
    all_components = source["relational_graph"]["components"]
    incumbent = _component_values(source, graph["incumbent_objects"])
    incumbent_ids = {str(item["id"]) for item in incumbent}
    challengers = [
        item for item in all_components
        if str(item["id"]) not in incumbent_ids]
    all_values = _truth_values(graph, all_components, calibrated)
    inc_values = _truth_values(graph, incumbent, calibrated)
    challenger_values = _truth_values(graph, challengers, calibrated)
    all_summary = _summary(all_values)
    inc_summary = _summary(inc_values)
    challenger_summary = _summary(challenger_values)

    def raw_max(agent: str) -> float:
        return max((
            float(records[_record_key(graph, item)]["raw"][agent])
            for item in all_components
        ), default=0.0)

    disagreements = [
        abs(
            float(records[_record_key(graph, item)]["raw"][QWEN])
            - float(records[_record_key(graph, item)]["raw"][GEMMA])
        )
        for item in all_components
    ]
    values = [
        all_summary["max"], all_summary["mean"], all_summary["min"],
        inc_summary["max"], inc_summary["mean"], inc_summary["min"],
        challenger_summary["max"],
        challenger_summary["max"] - inc_summary["max"],
        (
            sum(value > 0.5 for value in all_values) / len(all_values)
            if all_values else 0.0
        ),
        raw_max(QWEN), raw_max(GEMMA),
        statistics.mean(disagreements) if disagreements else 0.0,
    ]
    if len(values) != len(TRUTH_GATE_FEATURE_NAMES):
        raise AssertionError("truth gate feature schema drift")
    return values


def augmented_gate_features(
    graph: Mapping[str, Any],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[float]:
    values = [
        *row_gate_features(graph),
        *truth_gate_features(graph, calibrated, records),
    ]
    if len(values) != len(GATE_FEATURE_NAMES):
        raise AssertionError("augmented gate feature schema drift")
    return values


class TruthActionSelector:
    def __init__(
        self, pairwise_win_model: StandardizedLinear,
        conditional_gain_model: StandardizedLinear,
        conditional_loss_model: StandardizedLinear,
    ):
        self.pairwise_win_model = pairwise_win_model
        self.conditional_gain_model = conditional_gain_model
        self.conditional_loss_model = conditional_loss_model

    @property
    def parameter_count(self) -> int:
        return (
            self.pairwise_win_model.parameter_count
            + self.conditional_gain_model.parameter_count
            + self.conditional_loss_model.parameter_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "truth-calibrated-pairwise-action-selector-v3",
            "pairwise_win_model": self.pairwise_win_model.to_dict(),
            "conditional_gain_model": self.conditional_gain_model.to_dict(),
            "conditional_loss_model": self.conditional_loss_model.to_dict(),
            "decision_rule": (
                "P(win)*E[gain|win] - "
                "P(not-win)*E[loss|not-win] > 0"),
            "comparison": "exact_legal_action_vs_keep_at_current_state",
            "parameter_count": self.parameter_count,
        }


class DirectUtilitySelector:
    """Predict official row-F1 delta for every legal incumbent-relative edit."""

    def __init__(self, expected_delta_model: StandardizedLinear):
        self.expected_delta_model = expected_delta_model

    @property
    def parameter_count(self) -> int:
        return self.expected_delta_model.parameter_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "direct-incumbent-relative-action-utility-v1",
            "decision_rule": (
                "argmax legal action E[row_f1(action)-row_f1(incumbent)]; "
                "edit iff maximum is positive"),
            "training_population": (
                "all fixed-cardinality legal edits with row-balanced weights"),
            "expected_delta_model": self.expected_delta_model.to_dict(),
            "parameter_count": self.parameter_count,
        }


def pairwise_action_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    keep: Mapping[str, Any],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[float]:
    """Represent an exact edit and its signed contrast with KEEP."""
    action_values = augmented_action_features(
        graph, action, calibrated, records)
    keep_values = augmented_action_features(
        graph, keep, calibrated, records)
    values = [
        *action_values,
        *(left - right for left, right in zip(
            action_values, keep_values)),
    ]
    if len(values) != len(PAIRWISE_FEATURE_NAMES):
        raise AssertionError("pairwise action feature schema drift")
    return values


def training_arrays(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    prepared: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[
    list[list[float]], list[float], list[float], list[float],
    dict[str, Any],
]:
    pairwise_x, pairwise_win_y, pairwise_delta_y = [], [], []
    pairwise_weights = []
    state_counts, action_counts = [], []
    positive_actions = 0
    for graph in graphs:
        item = (
            prepared[_key(graph)] if prepared is not None
            else prepare_supervision(
                [graph], gold_by,
                decision_mode="fixed-cardinality")[_key(graph)]
        )
        states = item["states"]
        state_counts.append(len(states))
        state_weight = 1.0 / len(states)
        for state_item in states:
            state = state_item["state"]
            deltas = state_item["deltas"]
            keep_index = next(
                index for index, action in enumerate(state["actions"])
                if action["action_type"] == "KEEP")
            keep = state["actions"][keep_index]
            alternatives = [
                (action, delta)
                for index, (action, delta) in enumerate(zip(
                    state["actions"], deltas))
                if index != keep_index
            ]
            action_counts.append(len(alternatives))
            if not alternatives:
                continue
            action_weight = state_weight / len(alternatives)
            for action, delta in alternatives:
                pairwise_x.append(pairwise_action_features(
                    state, action, keep, calibrated, records))
                won = float(delta > 1e-12)
                pairwise_win_y.append(won)
                pairwise_delta_y.append(float(delta))
                pairwise_weights.append(action_weight)
                positive_actions += int(won)
    return (
        pairwise_x, pairwise_win_y, pairwise_delta_y,
        pairwise_weights,
        {
            "rows": len(graphs),
            "states": sum(state_counts),
            "actions": sum(action_counts),
            "positive_actions": positive_actions,
            "positive_action_rate": (
                positive_actions / len(pairwise_win_y)
                if pairwise_win_y else 0.0),
            "mean_states_per_row": statistics.mean(state_counts),
            "max_states_per_row": max(state_counts, default=0),
            "mean_actions_per_state": (
                statistics.mean(action_counts) if action_counts else 0.0),
        },
    )


def prepare_supervision(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    *, decision_mode: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Cache gold-derived action labels once; never serialize this mapping."""
    if decision_mode not in DECISION_MODES:
        raise ValueError(f"unknown decision mode: {decision_mode}")
    prepared = {}
    for graph in graphs:
        relation = str(graph["Relation"])
        gold = gold_by[_key(graph)]
        states = []
        expanded = (
            expand_fixed_cardinality_states(graph)
            if decision_mode == "fixed-cardinality"
            else expand_states(graph)
        )
        for state in expanded:
            baseline = _row_f1(
                state["incumbent_objects"], gold, relation)
            states.append({
                "state": state,
                "deltas": [
                    _row_f1(action["objects"], gold, relation) - baseline
                    for action in state["actions"]
                ],
            })
        prepared[_key(graph)] = {"states": states}
    return prepared


def fit_selector(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    prepared: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[TruthActionSelector, dict[str, Any]]:
    arrays = training_arrays(
        graphs, gold_by, calibrated, records, prepared)
    pairwise_win_model = StandardizedLinear(
        PAIRWISE_FEATURE_NAMES, ACTION_L2, logistic=True,
    ).fit(arrays[0], arrays[1], arrays[3])
    positive = [
        index for index, delta in enumerate(arrays[2])
        if delta > 1e-12]
    nonpositive = [
        index for index, delta in enumerate(arrays[2])
        if delta <= 1e-12]
    conditional_gain_model = StandardizedLinear(
        PAIRWISE_FEATURE_NAMES, ACTION_L2, logistic=False,
    ).fit(
        [arrays[0][index] for index in positive],
        [arrays[2][index] for index in positive],
        [arrays[3][index] for index in positive],
    )
    conditional_loss_model = StandardizedLinear(
        PAIRWISE_FEATURE_NAMES, ACTION_L2, logistic=False,
    ).fit(
        [arrays[0][index] for index in nonpositive],
        [-arrays[2][index] for index in nonpositive],
        [arrays[3][index] for index in nonpositive],
    )
    return (
        TruthActionSelector(
            pairwise_win_model,
            conditional_gain_model,
            conditional_loss_model),
        arrays[4],
    )


def fit_direct_utility_selector(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    prepared: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[DirectUtilitySelector, dict[str, Any]]:
    """Fit one dense metric-aligned action model, without a sparse win gate."""
    arrays = training_arrays(
        graphs, gold_by, calibrated, records, prepared)
    model = StandardizedLinear(
        PAIRWISE_FEATURE_NAMES, DIRECT_UTILITY_L2, logistic=False,
    ).fit(arrays[0], arrays[2], arrays[3])
    return DirectUtilitySelector(model), {
        **arrays[4],
        "target": "row_f1_action_minus_incumbent",
        "hurdle_model": False,
        "row_balanced": True,
        "l2": DIRECT_UTILITY_L2,
    }


def direct_utility_predictions(
    model: DirectUtilitySelector,
    graphs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose the highest expected-utility identity edit or keep incumbent."""
    predictions, diagnostics = [], []
    for source in graphs:
        graph = fixed_cardinality_state_view(
            source, source["incumbent_objects"])
        keep = next(
            action for action in graph["actions"]
            if action["action_type"] == "KEEP")
        alternatives = [
            action for action in graph["actions"]
            if action["action_type"] != "KEEP"]
        features = [
            pairwise_action_features(
                graph, action, keep, calibrated, records)
            for action in alternatives
        ]
        utilities = (
            model.expected_delta_model.predict(features)
            if features else np.asarray([], dtype=np.float64))
        best_index = max(
            range(len(alternatives)),
            key=lambda index: (
                float(utilities[index]),
                str(alternatives[index]["objects"]),
            ),
            default=None,
        )
        expected_delta = (
            float(utilities[best_index]) if best_index is not None else 0.0)
        accepted = best_index is not None and expected_delta > 0.0
        selected = (
            list(alternatives[best_index]["objects"])
            if accepted else list(keep["objects"]))
        proposed = (
            list(alternatives[best_index]["objects"])
            if best_index is not None else list(keep["objects"]))
        predictions.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": selected,
        })
        diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "initial_objects": list(keep["objects"]),
            "selected_objects": selected,
            "changed": accepted,
            "steps": int(accepted),
            "trace": [{
                "step": 0,
                "selected_action": (
                    "DIRECT_UTILITY_EDIT" if accepted else "KEEP"),
                "proposed_objects": proposed,
                "selected_objects": selected,
                "incumbent_objects": list(keep["objects"]),
                "predicted_expected_f1_delta": expected_delta,
                "gate_open": accepted,
            }],
        })
    return predictions, diagnostics


def decode_one(
    model: TruthActionSelector, graph: Mapping[str, Any],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    keep_index = next(
        index for index, action in enumerate(graph["actions"])
        if action["action_type"] == "KEEP")
    keep = graph["actions"][keep_index]
    alternatives = [
        (index, action)
        for index, action in enumerate(graph["actions"])
        if index != keep_index
    ]
    probabilities = (
        model.pairwise_win_model.predict([
            pairwise_action_features(
                graph, action, keep, calibrated, records)
            for _, action in alternatives
        ])
        if alternatives else np.asarray([], dtype=np.float64)
    )
    pairwise_features = [
        pairwise_action_features(
            graph, action, keep, calibrated, records)
        for _, action in alternatives
    ]
    expected_gains = (
        model.conditional_gain_model.predict(pairwise_features)
        if alternatives else np.asarray([], dtype=np.float64)
    )
    expected_losses = (
        model.conditional_loss_model.predict(pairwise_features)
        if alternatives else np.asarray([], dtype=np.float64)
    )
    expected_deltas = (
        np.asarray([
            float(probability) * max(float(gain), 0.0)
            - (1.0 - float(probability)) * max(float(loss), 0.0)
            for probability, gain, loss in zip(
                probabilities, expected_gains, expected_losses)
        ], dtype=np.float64)
        if alternatives else np.asarray([], dtype=np.float64)
    )
    best_alternative = max(
        range(len(alternatives)),
        key=lambda index: (
            float(expected_deltas[index]),
            float(probabilities[index]),
            -len(alternatives[index][1]["objects"]),
            -index,
        ),
        default=None,
    )
    best_probability = (
        float(probabilities[best_alternative])
        if best_alternative is not None else 0.0)
    best_expected_delta = (
        float(expected_deltas[best_alternative])
        if best_alternative is not None else 0.0)
    gate_open = best_expected_delta > 0.0
    best = (
        alternatives[best_alternative][0]
        if gate_open and best_alternative is not None
        else keep_index)
    action = graph["actions"][best]
    return list(action["objects"]), {
        "selected_action": action["action_type"],
        "selected_objects": list(action["objects"]),
        "incumbent_objects": list(graph["incumbent_objects"]),
        "predicted_utility": (
            best_expected_delta if gate_open else 0.0),
        "predicted_keep_utility": 0.0,
        "predicted_advantage": (
            best_expected_delta if gate_open else 0.0),
        "predicted_expected_f1_delta": best_expected_delta,
        "predicted_conditional_gain": (
            float(expected_gains[best_alternative])
            if best_alternative is not None else 0.0),
        "predicted_conditional_loss": (
            float(expected_losses[best_alternative])
            if best_alternative is not None else 0.0),
        "edit_probability": best_probability,
        "edit_gate_open": gate_open,
        "action_improvement_probability": best_probability,
        "action_safety_passed": gate_open,
        "nonkeep_actions": len(alternatives),
        "nonkeep_probability_passed": int(np.sum(probabilities > 0.5)),
        "nonkeep_delta_passed": int(np.sum(expected_deltas > 0.0)),
        "nonkeep_both_passed": int(np.sum(
            (probabilities > 0.5) & (expected_deltas > 0.0))),
        "maximum_nonkeep_probability": (
            float(np.max(probabilities)) if len(probabilities) else 0.0),
        "maximum_nonkeep_expected_delta": (
            float(np.max(expected_deltas)) if len(expected_deltas) else 0.0),
    }


def walk_predictions(
    model: TruthActionSelector,
    graphs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *, decision_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for graph in graphs:
        if decision_mode == "free":
            objects, trace = walk_with_chooser(
                graph,
                lambda view: decode_one(model, view, calibrated, records),
                max_steps=MAX_WALK_STEPS,
            )
        else:
            objects = list(graph["incumbent_objects"])
            trace = []
            for step in range(MAX_WALK_STEPS):
                view = fixed_cardinality_state_view(graph, objects)
                next_objects, detail = decode_one(
                    model, view, calibrated, records)
                trace.append({**detail, "step": step})
                if state_key(graph, next_objects) == state_key(
                        graph, objects):
                    break
                objects = next_objects
        predictions.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": objects,
        })
        diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "initial_objects": graph["incumbent_objects"],
            "selected_objects": objects,
            "changed": state_key(graph, objects) != state_key(
                graph, graph["incumbent_objects"]),
            "steps": len(trace),
            "trace": trace,
        })
    return predictions, diagnostics


def calibrated_rank_predictions(
    graphs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, str], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select top-k component identities at upstream-selected cardinality."""
    predictions, diagnostics = [], []
    for graph in graphs:
        incumbent = list(graph["incumbent_objects"])
        cardinality = len(state_key(graph, incumbent))
        components = graph["_source"]["relational_graph"]["components"]
        ranked = sorted(
            components,
            key=lambda component: (
                float(calibrated[_record_key(graph, component)]),
                str(component["representative"]),
            ),
            reverse=True,
        )
        selected = (
            [str(item["representative"])
             for item in ranked[:cardinality]]
            if cardinality and len(ranked) >= cardinality
            else incumbent
        )
        changed = state_key(graph, selected) != state_key(graph, incumbent)
        predictions.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": selected,
        })
        diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "initial_objects": incumbent,
            "selected_objects": selected,
            "changed": changed,
            "steps": int(changed),
            "trace": [{
                "step": 0,
                "selected_action": "RANK_TOP_K" if changed else "KEEP",
                "selected_objects": selected,
                "incumbent_objects": incumbent,
                "selected_cardinality": cardinality,
                "maximum_calibrated_truth": (
                    float(calibrated[_record_key(graph, ranked[0])])
                    if ranked else 0.0),
            }],
        })
    return predictions, diagnostics


def _control_predictions(
    graphs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": row["incumbent_objects"],
    } for row in graphs]


def fixed_cardinality_atomic_oracle(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Gold-aware train diagnostic for one cardinality-preserving edit."""
    predictions, rows_with_headroom = [], 0
    for graph in graphs:
        relation = str(graph["Relation"])
        gold = gold_by[_key(graph)]
        actions = fixed_cardinality_actions(
            graph, graph["incumbent_objects"])
        best = max(
            actions,
            key=lambda action: (
                _row_f1(action["objects"], gold, relation),
                action["action_type"] == "KEEP",
            ),
        )
        keep_score = _row_f1(
            graph["incumbent_objects"], gold, relation)
        rows_with_headroom += (
            _row_f1(best["objects"], gold, relation)
            > keep_score + 1e-12
        )
        predictions.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": list(best["objects"]),
        })
    return predictions, rows_with_headroom


def apply_starting_predictions(
    graphs: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    """Root every graph at a label-free upstream prediction artifact."""
    by_key = {_key(row): row for row in predictions}
    if len(by_key) != len(predictions) or set(by_key) != {
            _key(row) for row in graphs}:
        raise ContractError("starting prediction coverage mismatch")
    for graph in graphs:
        objects = [
            str(item) for item in by_key[_key(graph)]["ObjectEntities"]]
        graph["incumbent_objects"] = objects
        graph["actions"] = _legal_actions(graph["_source"], objects)
        if isinstance(graph, dict):
            graph.pop("_state_view_cache", None)


def fixed_cardinality_actions(
    graph: Mapping[str, Any], objects: Sequence[str],
) -> list[dict[str, Any]]:
    """Build identity substitutions without changing ZERO/ONE/MANY.

    The accepted upstream selector remains solely responsible for abstention
    and output cardinality. Each non-KEEP action substitutes one selected
    surface with one candidate-component representative.
    """
    current = [str(item) for item in objects]
    current_key = state_key(graph, current)
    raw: list[tuple[str, list[str]]] = [("KEEP", current)]
    if current:
        representatives = [
            str(item["representative"])
            for item in graph["_source"]["relational_graph"]["components"]
        ]
        for index in range(len(current)):
            for representative in representatives:
                candidate = list(current)
                candidate[index] = representative
                candidate_key = state_key(graph, candidate)
                if (
                    len(candidate_key) == len(current_key)
                    and candidate_key != current_key
                ):
                    raw.append(("REPLACE", candidate))
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for action_type, candidate in raw:
        key = state_key(graph, candidate)
        if key not in unique or action_type == "KEEP":
            unique[key] = {
                "id": f"action:{len(unique)}",
                "node_type": "counterfactual_action",
                "action_type": action_type,
                "objects": list(dict.fromkeys(candidate)),
            }
    actions = list(unique.values())
    if (
        sum(item["action_type"] == "KEEP" for item in actions) != 1
        or any(
            len(state_key(graph, item["objects"])) != len(current_key)
            for item in actions
        )
    ):
        raise AssertionError("fixed-cardinality action contract failed")
    return actions


def fixed_cardinality_state_view(
    graph: Mapping[str, Any], objects: Sequence[str],
) -> dict[str, Any]:
    """Return a counterfactual state with only cardinality-preserving edits."""
    view = {
        key: value for key, value in graph.items()
        if key not in {"actions", "incumbent_objects", "_state_view_cache"}
    }
    view["incumbent_objects"] = [str(item) for item in objects]
    view["actions"] = fixed_cardinality_actions(
        graph, view["incumbent_objects"])
    return view


def expand_fixed_cardinality_states(
    graph: Mapping[str, Any], *, depth: int = 0, max_states: int = 1,
) -> list[dict[str, Any]]:
    """Training states for the conditional candidate-identity decoder.

    Every candidate substitution is already reachable from the accepted root.
    Root-only supervision avoids quadratically repeating the same swaps on
    long list relations. Inference may still walk for three substitutions.
    """
    root = fixed_cardinality_state_view(graph, graph["incumbent_objects"])
    output: list[dict[str, Any]] = []
    queue: list[tuple[dict[str, Any], int]] = [(root, 0)]
    seen: set[tuple[str, ...]] = set()
    while queue and len(output) < max_states:
        view, level = queue.pop(0)
        key = state_key(view, view["incumbent_objects"])
        if key in seen:
            continue
        seen.add(key)
        output.append(view)
        if level >= depth:
            continue
        for action in view["actions"]:
            if action["action_type"] == "KEEP":
                continue
            next_key = state_key(view, action["objects"])
            if next_key not in seen:
                queue.append((
                    fixed_cardinality_state_view(
                        graph, action["objects"]),
                    level + 1,
                ))
    return output


def _subset_gold(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [gold_by[_key(row)] for row in graphs]


def run_train_audit(args: argparse.Namespace) -> int:
    graph_path = Path(args.train_graph).resolve()
    gold_path = Path(args.train_gold).resolve()
    truth_plan_path = Path(args.truth_plan).resolve()
    likelihood_run = Path(args.likelihood_run).resolve()
    control_path = Path(args.accepted_control).resolve()
    agents_path = Path(args.agents).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_graphs = read_jsonl(graph_path)
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    if (
        len(raw_graphs) != len(gold_rows)
        or {_key(row) for row in raw_graphs} != set(gold_by)
    ):
        raise ContractError("truth decoder graph/gold mismatch")
    graphs = [build_hierarchical_row(row) for row in raw_graphs]
    incumbent_predictions = _control_predictions(graphs)
    control_manifest = validate_registered_predictions(
        control_path, pipeline_id=COMPETITION_PIPELINE_ID, split="train")
    accepted = read_jsonl(control_path)
    if {_key(row) for row in accepted} != {_key(row) for row in graphs}:
        raise ContractError("accepted selector control coverage mismatch")
    if args.starting_state == "accepted":
        apply_starting_predictions(graphs, accepted)
    records, truth_plan = load_truth_records(truth_plan_path)
    if Path(truth_plan["train_graph"]).resolve() != graph_path:
        raise ContractError("truth plan was built from a different graph")
    validate_graph_truth_coverage(graphs, records)
    likelihood_detail = attach_likelihood_records(
        records, graphs, likelihood_run, graph_path)
    print("preparing reusable state/action supervision", flush=True)
    prepared = prepare_supervision(
        graphs, gold_by, decision_mode=args.decision_mode)
    print(
        "prepared "
        f"{sum(len(item['states']) for item in prepared.values())} states",
        flush=True,
    )

    outer = grouped_relation_folds(graphs, OUTER_FOLDS, seed=args.seed)
    accepted_fold_path = Path(control_manifest["folds"]).resolve()
    if control_manifest.get("folds_sha256") != sha256(accepted_fold_path):
        raise ContractError("accepted selector fold artifact is stale")
    accepted_folds = {
        _key(row): int(row["fold"]) for row in read_jsonl(accepted_fold_path)}
    if (
        set(accepted_folds) != set(outer)
        or sorted(set(accepted_folds.values())) != list(range(OUTER_FOLDS))
    ):
        raise ContractError(
            "accepted selector OOF fold coverage is invalid")
    oof_by: dict[tuple[str, str], dict[str, Any]] = {}
    diagnostics, fold_records = [], []
    for fold in range(OUTER_FOLDS):
        print(f"outer fold {fold + 1}/{OUTER_FOLDS}", flush=True)
        fit_rows = [row for row in graphs if outer[_key(row)] != fold]
        hold_rows = [row for row in graphs if outer[_key(row)] == fold]
        stacked, calibrator, calibration_detail = crossfit_calibrator(
            fit_rows, records, gold_by,
            seed=args.seed + 1009 * (fold + 1))
        selector, expansion = fit_selector(
            fit_rows, gold_by, stacked, records, prepared)
        hold_calibrated = predict_calibrated(
            calibrator, hold_rows, records)
        ranker_detail = gate_detail = direct_detail = None
        if args.decoder_mode == "direct-utility-action":
            direct, direct_detail = fit_direct_utility_selector(
                fit_rows, gold_by, stacked, records, prepared)
            predictions, detail = direct_utility_predictions(
                direct, hold_rows, hold_calibrated, records)
        elif args.decoder_mode in {
            "counterfactual-gated-rank", "set-aware-gated-rank",
        }:
            ranker_crossfit = (
                crossfit_set_aware_component_ranker
                if args.decoder_mode == "set-aware-gated-rank"
                else crossfit_component_ranker)
            stacked_ranked, ranker, ranker_detail = ranker_crossfit(
                fit_rows, records, gold_by,
                seed=args.seed + 2003 * (fold + 1))
            gate, gate_detail = fit_counterfactual_rank_gate(
                fit_rows, gold_by, stacked, stacked_ranked, records)
            hold_ranked = predict_component_ranks(
                ranker, hold_rows, records)
            predictions, detail = gated_rank_predictions(
                gate, hold_rows, hold_calibrated,
                hold_ranked, records)
        elif args.decoder_mode == "pairwise-component-rank":
            ranker, ranker_detail = fit_pairwise_component_ranker(
                fit_rows, records, gold_by)
            hold_ranked = predict_component_ranks(
                ranker, hold_rows, records)
            predictions, detail = calibrated_rank_predictions(
                hold_rows, hold_ranked)
        elif args.decoder_mode == "calibrated-rank":
            predictions, detail = calibrated_rank_predictions(
                hold_rows, hold_calibrated)
        else:
            predictions, detail = walk_predictions(
                selector, hold_rows, hold_calibrated, records,
                decision_mode=args.decision_mode)
        for row in predictions:
            if _key(row) in oof_by:
                raise ContractError("duplicate outer OOF prediction")
            oof_by[_key(row)] = row
        diagnostics.extend([
            {**item, "outer_fold": fold} for item in detail])
        hold_gold = _subset_gold(hold_rows, gold_by)
        selected_score = score(
            predictions, hold_gold)["*** All Relations ***"]
        starting_score = score(
            _control_predictions(hold_rows),
            hold_gold)["*** All Relations ***"]
        incumbent_score = score(
            [
                next(
                    row for row in incumbent_predictions
                    if _key(row) == _key(graph))
                for graph in hold_rows
            ],
            hold_gold,
        )["*** All Relations ***"]
        fold_records.append({
            "fold": fold,
            "fit_rows": len(fit_rows),
            "hold_rows": len(hold_rows),
            "calibration": calibration_detail,
            "component_ranker": ranker_detail,
            "counterfactual_gate": gate_detail,
            "direct_utility_selector": direct_detail,
            "training_expansion": expansion,
            "incumbent_score": incumbent_score,
            "starting_score": starting_score,
            "selected_score": selected_score,
            "delta": selected_score - incumbent_score,
            "incremental_delta": selected_score - starting_score,
        })
    if set(oof_by) != {_key(row) for row in graphs}:
        raise ContractError("truth decoder outer OOF coverage failure")

    predictions = [oof_by[_key(row)] for row in graphs]
    starting_predictions = _control_predictions(graphs)
    selected_scores = score(predictions, gold_rows)
    control_scores = score(incumbent_predictions, gold_rows)
    starting_scores = score(starting_predictions, gold_rows)
    conditional_oracle, conditional_oracle_rows = (
        fixed_cardinality_atomic_oracle(graphs, gold_by))
    conditional_oracle_scores = score(conditional_oracle, gold_rows)
    pooled_delta = (
        selected_scores["*** All Relations ***"]
        - control_scores["*** All Relations ***"])
    relation_deltas = _relation_deltas(selected_scores, control_scores)

    accepted_scores = score(accepted, gold_rows)
    incremental_delta = (
        selected_scores["*** All Relations ***"]
        - accepted_scores["*** All Relations ***"])
    broad_gate = _deployment_gate(
        pooled_delta,
        [item["delta"] for item in fold_records],
        relation_deltas,
    )
    deployment_gate = {
        **broad_gate,
        "passed": (
            broad_gate["passed"]
            and incremental_delta >= MIN_INCREMENTAL_DELTA),
        "incremental_check": (
            incremental_delta >= MIN_INCREMENTAL_DELTA),
        "minimum_incremental_delta": MIN_INCREMENTAL_DELTA,
    }

    final_stacked, final_calibrator, final_calibration = (
        crossfit_calibrator(
            graphs, records, gold_by, seed=args.seed + 9001))
    final_selector, final_expansion = fit_selector(
        graphs, gold_by, final_stacked, records, prepared)
    final_ranker = final_ranker_detail = None
    final_gate = final_gate_detail = None
    final_direct = final_direct_detail = None
    if args.decoder_mode == "direct-utility-action":
        final_direct, final_direct_detail = fit_direct_utility_selector(
            graphs, gold_by, final_stacked, records, prepared)
    elif args.decoder_mode in {
        "counterfactual-gated-rank", "set-aware-gated-rank",
    }:
        ranker_crossfit = (
            crossfit_set_aware_component_ranker
            if args.decoder_mode == "set-aware-gated-rank"
            else crossfit_component_ranker)
        final_rank_scores, final_ranker, final_ranker_detail = (
            ranker_crossfit(
                graphs, records, gold_by, seed=args.seed + 19001))
        final_gate, final_gate_detail = fit_counterfactual_rank_gate(
            graphs, gold_by, final_stacked,
            final_rank_scores, records)
    elif args.decoder_mode == "pairwise-component-rank":
        final_ranker, final_ranker_detail = (
            fit_pairwise_component_ranker(graphs, records, gold_by))
    selector_parameters = (
        final_calibrator.parameter_count + final_selector.parameter_count)
    if final_ranker is not None:
        selector_parameters += final_ranker.parameter_count
    if final_gate is not None:
        selector_parameters += final_gate.parameter_count
    if final_direct is not None:
        selector_parameters += final_direct.parameter_count
    agent_parameters = _agent_parameter_total(agents_path)
    combined_parameters = agent_parameters + selector_parameters
    if combined_parameters > PARAMETER_CAP:
        raise ContractError("truth decoder exceeds parameter cap")

    write_jsonl_atomic(output / "FOLDS.jsonl", [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "fold": outer[_key(row)],
    } for row in graphs])
    write_jsonl_atomic(
        output / "TRAIN_OOF_PREDICTIONS.jsonl", predictions)
    write_jsonl_atomic(
        output / "TRAIN_OOF_DIAGNOSTICS.jsonl", diagnostics)
    prediction_path = output / "TRAIN_OOF_PREDICTIONS.jsonl"
    prediction_manifest = {
        "schema": "truth-calibrated-action-oof-predictions-manifest-v1",
        "development_only": True,
        "deployable": False,
        "contains_labels": False,
        "gold_aware": False,
        "train_labels_used": True,
        "validation_labels_used": False,
        "oof_model_excludes_subject": True,
        "rows": len(predictions),
        "output_sha256": sha256(prediction_path),
    }
    prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json").write_text(
            json.dumps(prediction_manifest, indent=2, sort_keys=True) + "\n")
    model_artifact = {
        "schema": "truth-calibrated-action-decoder-model-v3",
        "development_only": True,
        "deployable": False,
        "contains_labels": True,
        "gold_aware": False,
        "train_labels_used": True,
        "validation_labels_used": False,
        "relation_specific_models": False,
        "relation_specific_thresholds": False,
        "starting_state": args.starting_state,
        "decision_mode": args.decision_mode,
        "decoder_mode": args.decoder_mode,
        "cardinality_owner": (
            "accepted_upstream_selector"
            if args.decision_mode == "fixed-cardinality" else "joint"),
        "candidate_truth_role": (
            "conditional_identity_only"
            if args.decision_mode == "fixed-cardinality"
            else "joint_identity_and_cardinality"),
        "outer_subject_grouped": True,
        "stacked_calibration": True,
        "action_specific_safety": True,
        "row_existence_gate": False,
        "calibrator_l2": CALIBRATOR_L2,
        "action_l2": ACTION_L2,
        "calibrator": final_calibrator.to_dict(),
        "selector": final_selector.to_dict(),
        "component_ranker": (
            final_ranker.to_dict() if final_ranker is not None else None),
        "component_ranker_training": final_ranker_detail,
        "counterfactual_rank_gate": (
            final_gate.to_dict() if final_gate is not None else None),
        "counterfactual_rank_gate_training": final_gate_detail,
        "direct_utility_selector": (
            final_direct.to_dict() if final_direct is not None else None),
        "direct_utility_selector_training": final_direct_detail,
        "final_calibration": final_calibration,
        "final_training_expansion": final_expansion,
        "selector_parameter_count": selector_parameters,
        "agent_parameter_upper_bound": agent_parameters,
        "combined_parameter_upper_bound": combined_parameters,
        "parameter_cap": PARAMETER_CAP,
        "truth_plan": str(truth_plan_path),
        "truth_plan_sha256": sha256(truth_plan_path),
        "likelihood_evidence": likelihood_detail,
        "train_graph": str(graph_path),
        "train_graph_sha256": sha256(graph_path),
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "starting_train_predictions": str(control_path),
        "starting_train_predictions_sha256": sha256(control_path),
        "starting_train_folds": str(accepted_fold_path),
        "starting_train_folds_sha256": sha256(accepted_fold_path),
        "downstream_folds_may_differ": True,
    }
    model_path = output / "MODEL.json"
    model_path.write_text(
        json.dumps(model_artifact, indent=2, sort_keys=True) + "\n")

    helped = harmed = 0
    for graph in graphs:
        relation = str(graph["Relation"])
        gold = gold_by[_key(graph)]
        delta = (
            _row_f1(oof_by[_key(graph)]["ObjectEntities"], gold, relation)
            - _row_f1(graph["incumbent_objects"], gold, relation))
        helped += delta > 1e-12
        harmed += delta < -1e-12
    result = {
        "schema": "truth-calibrated-action-decoder-train-audit-v2",
        "development_only": True,
        "deployable": False,
        "contains_labels": True,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "rows": len(graphs),
        "components": len(records),
        "likelihood_evidence": likelihood_detail,
        "decision_mode": args.decision_mode,
        "decoder_mode": args.decoder_mode,
        "control_scores": control_scores,
        "starting_scores": starting_scores,
        "accepted_selector_scores": accepted_scores,
        "selected_scores": selected_scores,
        "fixed_cardinality_atomic_oracle_scores": (
            conditional_oracle_scores),
        "fixed_cardinality_rows_with_headroom": conditional_oracle_rows,
        "fixed_cardinality_atomic_oracle_delta": (
            conditional_oracle_scores["*** All Relations ***"]
            - starting_scores["*** All Relations ***"]),
        "pooled_delta": pooled_delta,
        "incremental_delta_over_accepted_selector": incremental_delta,
        "relation_deltas": relation_deltas,
        "folds": fold_records,
        "deployment_gate": deployment_gate,
        "changed_rows": sum(item["changed"] for item in diagnostics),
        "helped_rows": helped,
        "harmed_rows": harmed,
        "action_counts": dict(Counter(
            step["selected_action"]
            for item in diagnostics for step in item["trace"])),
        "selector_parameter_count": selector_parameters,
        "combined_parameter_upper_bound": combined_parameters,
        "artifacts": {
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "oof_predictions": str(
                output / "TRAIN_OOF_PREDICTIONS.jsonl"),
            "oof_predictions_sha256": sha256(
                output / "TRAIN_OOF_PREDICTIONS.jsonl"),
        },
    }
    result_path = output / "RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Truth-calibrated heterogeneous-memory action decoder", "",
        "Train-only stacked subject-grouped OOF audit. Validation was not read.",
        "",
        f"- Incumbent F1: **{control_scores['*** All Relations ***']:.9f}**",
        f"- Decoder starting-state F1: "
        f"**{starting_scores['*** All Relations ***']:.9f}**",
        f"- Registered SOTA starting pipeline F1: "
        f"**{accepted_scores['*** All Relations ***']:.9f}**",
        f"- Truth-calibrated selector F1: "
        f"**{selected_scores['*** All Relations ***']:.9f}**",
        f"- Fixed-cardinality one-edit oracle F1: "
        f"**{conditional_oracle_scores['*** All Relations ***']:.9f}**",
        f"- Rows with fixed-cardinality headroom: "
        f"**{conditional_oracle_rows}**",
        f"- Delta over incumbent: **{pooled_delta:+.9f}**",
        f"- Delta over accepted selector: **{incremental_delta:+.9f}**",
        f"- Changed/helped/harmed: "
        f"**{result['changed_rows']} / {helped} / {harmed}**",
        f"- Deployment gate: **{deployment_gate['passed']}**", "",
        "## Relation deltas", "",
        "| relation | incumbent | calibrated selector | delta |",
        "|---|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        lines.append(
            f"| {relation} | {control_scores[relation]:.6f} | "
            f"{selected_scores[relation]:.6f} | "
            f"{relation_deltas[relation]:+.6f} |")
    lines.extend(["", "## Outer folds", "",
                  "| fold | rows | delta |",
                  "|---:|---:|---:|"])
    for item in fold_records:
        lines.append(
            f"| {item['fold']} | {item['hold_rows']} | "
            f"{item['delta']:+.6f} |")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "control": control_scores["*** All Relations ***"],
        "accepted_selector": accepted_scores["*** All Relations ***"],
        "truth_calibrated": selected_scores["*** All Relations ***"],
        "pooled_delta": pooled_delta,
        "incremental_delta": incremental_delta,
        "gate_passed": deployment_gate["passed"],
        "helped": helped,
        "harmed": harmed,
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


def _validated_frozen_model(
    model_dir: Path,
) -> tuple[
    StandardizedLinear, StandardizedLinear, CounterfactualRankGate,
    dict[str, Any], Path,
]:
    model_dir = Path(model_dir).resolve()
    result_path = model_dir / "RESULT.json"
    model_path = model_dir / "MODEL.json"
    if not result_path.is_file() or not model_path.is_file():
        raise ContractError("missing frozen truth-selector audit")
    result = _json(result_path)
    artifact = _json(model_path)
    if (
        result.get("schema")
        != "truth-calibrated-action-decoder-train-audit-v2"
        or not result.get("deployment_gate", {}).get("passed")
        or result.get("validation_labels_used") is not False
        or artifact.get("schema")
        != "truth-calibrated-action-decoder-model-v3"
        or artifact.get("decoder_mode") != "counterfactual-gated-rank"
        or artifact.get("starting_state") != "accepted"
        or artifact.get("decision_mode") != "fixed-cardinality"
        or artifact.get("validation_labels_used") is not False
        or artifact.get("relation_specific_models") is not False
        or artifact.get("relation_specific_thresholds") is not False
        or result.get("artifacts", {}).get("model_sha256")
        != sha256(model_path)
    ):
        raise ContractError("truth selector is not a frozen passing model")
    calibrator = StandardizedLinear.from_dict(artifact["calibrator"])
    ranker = StandardizedLinear.from_dict(artifact["component_ranker"])
    gate_artifact = artifact.get("counterfactual_rank_gate")
    if (
        not isinstance(gate_artifact, Mapping)
        or gate_artifact.get("schema")
        != "counterfactual-rank-edit-gate-v1"
        or gate_artifact.get("decision_rule")
        != "E[row_f1(proposal)-row_f1(incumbent)]>0"
    ):
        raise ContractError("missing frozen counterfactual rank gate")
    gate = CounterfactualRankGate(StandardizedLinear.from_dict(
        gate_artifact["expected_delta_model"]))
    return calibrator, ranker, gate, artifact, model_path


def _validation_graphs(path: Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ContractError("missing validation graph manifest")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != "heterogeneous-memory-graph-manifest-v1"
        or manifest.get("split") != "validation"
        or manifest.get("contains_labels")
        or manifest.get("gold_aware")
        or manifest.get("output_sha256") != sha256(path)
    ):
        raise ContractError("validation graph is not certified label-free")
    return [build_hierarchical_row(row) for row in read_jsonl(path)]


def _starting_validation_predictions(
    graphs: Sequence[Mapping[str, Any]], prediction_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the exact registered validation side of the train OOF pipeline."""
    prediction_path = Path(prediction_path).resolve()
    detail = validate_registered_predictions(
        prediction_path,
        pipeline_id=COMPETITION_PIPELINE_ID,
        split="validation")
    predictions = read_jsonl(prediction_path)
    if {_key(row) for row in predictions} != {_key(row) for row in graphs}:
        raise ContractError("registered SOTA validation coverage mismatch")
    return predictions, {
        **detail,
        "rows": len(predictions),
    }


def run_validation_decode(args: argparse.Namespace) -> int:
    """Apply the frozen passing chain without reading validation labels."""
    graph_path = Path(args.validation_graph).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "VALIDATION_PREDICTIONS.jsonl"
    if prediction_path.is_file():
        raise ContractError(
            "validation predictions already exist; use a new output directory")
    calibrator, ranker, gate, model_artifact, model_path = (
        _validated_frozen_model(Path(args.model_dir)))
    graphs = _validation_graphs(graph_path)
    if model_artifact.get("starting_pipeline_id") != COMPETITION_PIPELINE_ID:
        raise ContractError(
            "frozen truth model was trained against a different pipeline")
    starting, starting_detail = _starting_validation_predictions(
        graphs, Path(args.starting_predictions))
    apply_starting_predictions(graphs, starting)

    truth_plan_path = (
        Path(args.validation_truth_run).resolve() / "plan/PLAN.json")
    records, truth_plan = load_truth_records(
        truth_plan_path, expected_split="validation")
    truth_graph = Path(
        truth_plan.get("graph", truth_plan.get("train_graph"))).resolve()
    if truth_graph != graph_path:
        raise ContractError(
            "validation truth evidence was built from a different graph")
    validate_graph_truth_coverage(graphs, records)
    likelihood_detail = attach_likelihood_records(
        records, graphs, Path(args.validation_likelihood_run),
        graph_path, expected_split="validation")
    calibrated = predict_calibrated(calibrator, graphs, records)
    ranked = predict_component_ranks(ranker, graphs, records)
    predictions, diagnostics = gated_rank_predictions(
        gate, graphs, calibrated, ranked, records)
    if len(predictions) != len(graphs):
        raise ContractError("validation prediction row-count mismatch")

    write_jsonl_atomic(
        output_dir / "STARTING_PREDICTIONS.jsonl", starting)
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(
        output_dir / "VALIDATION_DIAGNOSTICS.jsonl", diagnostics)
    manifest = {
        "schema": "truth-calibrated-validation-predictions-manifest-v1",
        "development_only": True,
        "deployable": False,
        "contains_labels": False,
        "gold_aware": False,
        "validation_labels_used": False,
        "selection_uses_validation_labels": False,
        "rows": len(predictions),
        "output_sha256": sha256(prediction_path),
        "diagnostics_sha256": sha256(
            output_dir / "VALIDATION_DIAGNOSTICS.jsonl"),
        "starting_predictions_sha256": sha256(
            output_dir / "STARTING_PREDICTIONS.jsonl"),
        "frozen_truth_model": str(model_path),
        "frozen_truth_model_sha256": sha256(model_path),
        "frozen_truth_model_training_graph_sha256": (
            model_artifact["train_graph_sha256"]),
        "starting_stage": starting_detail,
        "validation_graph": str(graph_path),
        "validation_graph_sha256": sha256(graph_path),
        "truth_plan": str(truth_plan_path),
        "truth_plan_sha256": sha256(truth_plan_path),
        "likelihood_evidence": likelihood_detail,
    }
    manifest_path = prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "rows": len(predictions),
        "changed_rows": sum(item["changed"] for item in diagnostics),
        "validation_labels_used": False,
    }, indent=2, sort_keys=True))
    return 0


def run_validation_evaluate(args: argparse.Namespace) -> int:
    """Open validation labels only after the prediction manifest is frozen."""
    output_dir = Path(args.output_dir).resolve()
    prediction_path = output_dir / "VALIDATION_PREDICTIONS.jsonl"
    manifest_path = prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json")
    result_path = output_dir / "VALIDATION_RESULT.json"
    if result_path.is_file():
        raise ContractError(
            "validation result already exists; refusing repeated evaluation")
    if not prediction_path.is_file() or not manifest_path.is_file():
        raise ContractError("missing frozen validation predictions")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema")
        != "truth-calibrated-validation-predictions-manifest-v1"
        or manifest.get("contains_labels")
        or manifest.get("validation_labels_used") is not False
        or manifest.get("selection_uses_validation_labels") is not False
        or manifest.get("output_sha256") != sha256(prediction_path)
    ):
        raise ContractError("invalid frozen validation prediction manifest")
    gold_path = Path(args.validation_gold).resolve()
    gold = read_jsonl(gold_path)
    predictions = read_jsonl(prediction_path)
    if {_key(row) for row in gold} != {_key(row) for row in predictions}:
        raise ContractError("validation prediction/gold coverage mismatch")
    scores = score(predictions, gold)
    result = {
        "schema": "truth-calibrated-validation-result-v1",
        "development_only": True,
        "deployable": False,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": True,
        "validation_used_for_selection": False,
        "scores": scores,
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "validation_gold": str(gold_path),
        "validation_gold_sha256": sha256(gold_path),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command",
        choices=("train-audit", "decode-validation", "evaluate-validation"))
    value.add_argument("--train-graph", default=str(DEFAULT_GRAPH))
    value.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    value.add_argument("--truth-plan", default=str(DEFAULT_TRUTH_PLAN))
    value.add_argument(
        "--likelihood-run", default=str(DEFAULT_LIKELIHOOD_RUN))
    value.add_argument("--accepted-control", default=str(DEFAULT_CONTROL))
    value.add_argument("--agents", default=str(DEFAULT_AGENTS))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    value.add_argument("--model-dir", default=str(DEFAULT_OUTPUT))
    value.add_argument(
        "--starting-predictions", default=str(DEFAULT_STARTING_PREDICTIONS))
    value.add_argument(
        "--validation-graph", default=str(DEFAULT_VALIDATION_GRAPH))
    value.add_argument(
        "--validation-truth-run", default=str(DEFAULT_VALIDATION_TRUTH_RUN))
    value.add_argument(
        "--validation-likelihood-run",
        default=str(DEFAULT_VALIDATION_LIKELIHOOD_RUN))
    value.add_argument(
        "--validation-gold",
        default=str(Path(DEFAULT_GOLD).with_name("val.jsonl")))
    value.add_argument("--seed", type=int, default=20260727)
    value.add_argument(
        "--starting-state", choices=("incumbent", "accepted"),
        default="accepted",
    )
    value.add_argument(
        "--decision-mode", choices=DECISION_MODES,
        default="fixed-cardinality",
    )
    value.add_argument(
        "--decoder-mode", choices=DECODER_MODES,
        default="counterfactual-gated-rank",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "train-audit":
        return run_train_audit(args)
    if args.command == "decode-validation":
        return run_validation_decode(args)
    if args.command == "evaluate-validation":
        return run_validation_evaluate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
