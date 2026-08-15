#!/usr/bin/env python3
"""Model identifiers and graph-free proposal decoding used by the release."""
from __future__ import annotations
import math
import statistics
from typing import Any, Mapping, Sequence
from lm_kbc.core import ContractError, NUMERIC_RELATIONS, SINGLE_RELATIONS, build_agent_tasks, canonical_key, load_agent_config, load_synthetic_by_relation, proposal_prompt, proposal_candidates, proposal_parse_status, read_jsonl, select_synthetic_shots, sha256, validate_inputs, validate_task_response, write_jsonl_atomic
GEMMA = 'gemma_independent'
QWEN = 'qwen_recall'

def proposal_only_prediction(response: Mapping[str, Any]) -> list[str]:
    """Plain self-consistency control with no heterogeneous-router signals."""
    relation = response['relation']
    candidates = proposal_candidates(response)
    generations = response.get('generations', [])
    if relation in NUMERIC_RELATIONS:
        values = []
        for candidate in candidates:
            try:
                value = float(str(candidate['item']).replace(',', ''))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                values.extend([value] * int(candidate['support']))
        return [format(statistics.median(values), '.12g')] if values else []
    if relation in SINGLE_RELATIONS:
        none_support = sum((proposal_parse_status(str(generation), relation)[0] == 'explicit_none' for generation in generations))
        if not candidates or none_support >= int(candidates[0]['support']):
            return []
        return [candidates[0]['item']]
    threshold = max(1, math.ceil(len(generations) / 2))
    return [candidate['item'] for candidate in candidates if int(candidate['support']) >= threshold]
