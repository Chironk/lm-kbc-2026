"""Deserialize the compact frozen decoder models used by the final pipeline."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from experiments.heterogeneous_agents.baseline_relative_route_decoder import ResidualRidge
from experiments.heterogeneous_agents.core import ContractError
from experiments.heterogeneous_agents.explicit_cardinality_ablation import ExplicitCardinalityModel
from experiments.heterogeneous_agents.heterogeneous_memory_selector import LogisticCalibrator
from experiments.heterogeneous_agents.relation_specific_numeric_decoder import RelationSpecificNumericModel


def calibrator(value: Mapping[str, Any]) -> LogisticCalibrator:
    return LogisticCalibrator.from_dict(value)


def cardinality_model(value: Mapping[str, Any]) -> ExplicitCardinalityModel:
    if value.get("schema") != "explicit-cardinality-ovr-v1":
        raise ContractError("foreign explicit-cardinality model")
    model = ExplicitCardinalityModel(float(value["l2"]))
    model.many_mean = {
        str(relation): float(mean)
        for relation, mean in value["many_mean"].items()
    }
    model.models = {
        str(relation): {
            str(label): calibrator(current)
            for label, current in labels.items()
        }
        for relation, labels in value["models"].items()
    }
    return model


def numeric_model(value: Mapping[str, Any]) -> RelationSpecificNumericModel:
    if value.get("schema") != "relation-specific-numeric-model-v1":
        raise ContractError("foreign relation-specific numeric model")
    model = RelationSpecificNumericModel(float(value["l2"]))
    model.models = {
        str(relation): calibrator(current)
        for relation, current in value["models"].items()
    }
    return model


def residual_model(value: Mapping[str, Any]) -> ResidualRidge:
    if value.get("schema") != "signed-standardized-residual-ridge-v1":
        raise ContractError("foreign route-residual model")
    model = ResidualRidge(value["feature_names"], float(value["l2"]))
    model.mean = np.asarray(value["mean"], dtype=np.float64)
    model.scale = np.asarray(value["scale"], dtype=np.float64)
    model.coefficients = np.asarray(value["coefficients"], dtype=np.float64)
    if (
        model.mean.shape != (len(model.names),)
        or model.scale.shape != (len(model.names),)
        or model.coefficients.shape != (len(model.names) + 1,)
    ):
        raise ContractError("route-residual model shape mismatch")
    return model
