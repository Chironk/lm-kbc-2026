import copy

import pytest

from experiments.heterogeneous_agents.core import ContractError
from experiments.heterogeneous_agents.ministral_cot_ablation import (
    MINISTRAL,
    N_MAX,
    REASONING_WORDS,
    SYNTHETIC_SHOTS,
    _cot_config,
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
            "demonstration_reasoning": False,
        }],
    }


def _task(*, cot: bool) -> dict:
    value = {
        "task_id": f"{MINISTRAL}::0::proposal",
        "agent_id": MINISTRAL,
        "subject": "Example",
        "relation": "companyTradesAtStockExchange",
        "input_index": 0,
        "phase": "propose",
        "mode": "generate",
        "n_samples": N_MAX,
        "temperature": 0.8,
        "max_new_tokens": 80,
        "proposal_output": "bounded_reasoning",
        "demonstration_reasoning": cot,
        "shot_subjects": [],
        "prompt": (
            "zero-shot prompt with at most 20 words"
            if not cot else
            "PRIVATE RECALL DEMONSTRATIONS:\n"
            "QUESTION: one\nQUESTION: two\nQUESTION: three\n"
            "QUESTION: four\nQUESTION: five\nat most 20 words"
        ),
    }
    if cot:
        value["shot_subjects"] = ["A", "B", "C", "D", "E"]
    return value


def test_cot_config_changes_only_ministral_prompt_policy() -> None:
    base = _config()
    original = copy.deepcopy(base)
    value = _cot_config(base)
    assert base == original
    agent = value["agents"][0]
    assert agent["role"] == "synthetic_cot_recall"
    assert agent["synthetic_shots"] == SYNTHETIC_SHOTS
    assert agent["demonstration_reasoning"] is True
    assert agent["proposal_output"] == "bounded_reasoning"
    assert agent["proposal_reasoning_words"] == REASONING_WORDS


def test_task_parity_accepts_only_demonstration_changes() -> None:
    summary = _task_parity([_task(cot=False)], [_task(cot=True)])
    assert summary["prompt_changes"] == 1
    assert summary["shots_per_task"] == SYNTHETIC_SHOTS


def test_task_parity_rejects_sampling_or_target_leakage() -> None:
    control, cot = _task(cot=False), _task(cot=True)
    cot["temperature"] = 1.0
    with pytest.raises(ContractError, match="temperature"):
        _task_parity([control], [cot])
    cot = _task(cot=True)
    cot["shot_subjects"][0] = cot["subject"]
    with pytest.raises(ContractError, match="shot assignment"):
        _task_parity([control], [cot])
