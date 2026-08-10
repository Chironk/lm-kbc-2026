import copy

import pytest

from experiments.heterogeneous_agents.core import ContractError
from experiments.heterogeneous_agents.components.ministral_cot40_training import (
    MINISTRAL,
    N_MAX,
    REASONING_WORDS,
    SYNTHETIC_SHOTS,
    _cot40_config,
    _task_parity,
)


def _config() -> dict:
    return {
        "agents": [{
            "id": MINISTRAL,
            "model": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            "revision": "f6fae9795746f63c9be8344932f01275f3c63734",
            "role": "independent_direct_recall",
            "synthetic_shots": 0,
            "proposal_output": "bounded_reasoning",
            "proposal_reasoning_words": REASONING_WORDS,
            "proposal_max_new_tokens": {
                "default": 112,
                "awardWonBy": 256,
            },
            "demonstration_reasoning": False,
        }],
    }


def _task(kind: str) -> dict:
    is_cot = kind in {"cot20", "cot40"}
    cap = 20 if kind == "cot20" else 40
    n = 3 if kind == "zero40" else N_MAX
    max_tokens = 80 if kind == "cot20" else 112
    prompt = f"direct recall; at most {cap} words"
    shots = []
    if is_cot:
        prompt = (
            "PRIVATE RECALL DEMONSTRATIONS:\n"
            "QUESTION: A\nQUESTION: B\nQUESTION: C\n"
            "QUESTION: D\nQUESTION: E\n"
            f"at most {cap} words"
        )
        shots = ["A", "B", "C", "D", "E"]
    return {
        "task_id": f"{MINISTRAL}::0::proposal",
        "agent_id": MINISTRAL,
        "subject": "Target",
        "relation": "hasArea",
        "input_index": 0,
        "phase": "propose",
        "mode": "generate",
        "n_samples": n,
        "temperature": 0.8,
        "max_new_tokens": max_tokens,
        "proposal_output": "bounded_reasoning",
        "shot_subjects": shots,
        "prompt": prompt,
    }


def test_cot40_config_changes_only_demonstration_policy() -> None:
    base = _config()
    frozen = copy.deepcopy(base)
    value = _cot40_config(base)
    assert base == frozen
    agent = value["agents"][0]
    assert agent["role"] == "synthetic_cot_recall"
    assert agent["synthetic_shots"] == SYNTHETIC_SHOTS
    assert agent["demonstration_reasoning"] is True
    assert agent["proposal_reasoning_words"] == REASONING_WORDS
    assert agent["proposal_max_new_tokens"]["default"] == 112


def test_task_parity_accepts_matched_factorial() -> None:
    summary = _task_parity(
        [_task("zero40")],
        [_task("cot20")],
        [_task("cot40")],
    )
    assert summary["paired_tasks"] == 1
    assert summary["nested_prefixes"] == [1, 3, 5, 10]


def test_task_parity_rejects_shot_or_sampling_drift() -> None:
    cot40 = _task("cot40")
    cot40["shot_subjects"][0] = "Target"
    with pytest.raises(ContractError, match="synthetic-shot"):
        _task_parity(
            [_task("zero40")], [_task("cot20")], [cot40])
    cot40 = _task("cot40")
    cot40["temperature"] = 1.0
    with pytest.raises(ContractError, match="temperature"):
        _task_parity(
            [_task("zero40")], [_task("cot20")], [cot40])
