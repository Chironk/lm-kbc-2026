#!/usr/bin/env python3
"""Pure contracts for heterogeneous-agent LM-KBC experiments.

The design intentionally separates candidate-blind commitments from candidate
generation and cross-agent review.  Agent prose is never passed to another
agent.  Only normalized claims are placed on the shared blackboard.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from evaluate import RELATION_TYPE, normalize_string


RELATION_QUESTIONS = {
    "hasArea": "What is the area of {subject} in square kilometers?",
    "hasCapacity": "What is the total spectator capacity of {subject}?",
    "countryLandBordersCountry": "Which countries share a land border with {subject}?",
    "personHasCityOfDeath": "In which city did {subject} die?",
    "awardWonBy": "Who won the {subject}?",
    "companyTradesAtStockExchange": "On which stock exchange or exchanges do shares of {subject} trade?",
}

# Versioned task-definition-aligned questions.  Keep ``RELATION_QUESTIONS``
# unchanged because historical task manifests hash prompts rendered with those
# strings.  New experiments can opt into this contract without invalidating
# frozen reproduction artifacts.
OFFICIAL_RELATION_QUESTIONS_V1 = {
    **RELATION_QUESTIONS,
    "hasCapacity": (
        "What is the highest published maximum spectator capacity of "
        "{subject}, as an integer number of people?"
    ),
}

NULLABLE_RELATIONS = {
    "countryLandBordersCountry",
    "personHasCityOfDeath",
    "companyTradesAtStockExchange",
}
NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}
SINGLE_RELATIONS = {"personHasCityOfDeath", "hasArea", "hasCapacity"}
# Single-character answer codes.  Most tasks use the A-D prefix; listwise
# complete-action selectors may need a wider menu.  The inference contract
# verifies that every used code is exactly one token for the loaded tokenizer
# before scoring.
CHOICE_CODES = ("A", "B", "C", "D", "E", "F", "G", "H")


class ContractError(RuntimeError):
    """An agent artifact violates a fail-closed structural contract."""


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ContractError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    payload = "\x1f".join(str(value) for value in values).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def balanced_choice_codes(choices: Sequence[str], *salt: object) -> Dict[str, str]:
    """Return the first deterministic member of a balanced codebook cycle."""
    return balanced_choice_codebooks(choices, *salt)[0]


def balanced_choice_codebooks(choices: Sequence[str],
                              *salt: object) -> List[Dict[str, str]]:
    """Assign every semantic label to every code exactly once.

    The variants are scored together in one batched forward pass. Averaging a
    semantic label across the full cycle removes row-level A/B/C/D token priors
    instead of relying only on dataset-level randomization.
    """
    if len(choices) > len(CHOICE_CODES) or len(set(choices)) != len(choices):
        raise ContractError(f"cannot code choices: {list(choices)!r}")
    codes = list(CHOICE_CODES[:len(choices)])
    random.Random(stable_seed("choice-codes", *salt)).shuffle(codes)
    return [
        {choice: codes[(choice_index + shift) % len(codes)]
         for choice_index, choice in enumerate(choices)}
        for shift in range(len(codes))
    ]


def validate_agent_config(config: Mapping[str, Any]) -> dict:
    cap = config.get("parameter_cap")
    agents = config.get("agents")
    if not isinstance(cap, int) or cap <= 0:
        raise ContractError("parameter_cap must be a positive integer")
    if not isinstance(agents, list) or len(agents) < 2:
        raise ContractError("at least two heterogeneous agents are required")
    required = {"id", "model", "verified_parameter_count", "parameter_upper_bound",
                "active_text_inference_parameter_count",
                "unused_text_only_parameter_count", "unused_text_only_parameter_prefixes",
                "strip_unused_vision", "model_class", "role", "synthetic_shots",
                "runtime_dtype", "allow_fp32_cpu_offload", "device_map_strategy",
                "precision"}
    ids = set()
    models = set()
    total = 0
    verified_total = 0
    active_total = 0
    for index, agent in enumerate(agents):
        missing = required - set(agent)
        if missing:
            raise ContractError(f"agent {index} missing fields: {sorted(missing)}")
        if agent["id"] in ids:
            raise ContractError(f"duplicate agent id: {agent['id']}")
        ids.add(agent["id"])
        models.add(agent["model"])
        count = agent["parameter_upper_bound"]
        if not isinstance(count, int) or count <= 0:
            raise ContractError(f"{agent['id']}: invalid parameter upper bound")
        total += count
        verified = agent["verified_parameter_count"]
        if not isinstance(verified, int) or verified <= 0:
            raise ContractError(f"{agent['id']}: invalid verified parameter count")
        if verified > count:
            raise ContractError(
                f"{agent['id']}: verified parameter count {verified:,} exceeds "
                f"declared upper bound {count:,}")
        verified_total += verified
        unused = agent["unused_text_only_parameter_count"]
        active = agent["active_text_inference_parameter_count"]
        prefixes = agent["unused_text_only_parameter_prefixes"]
        if (not isinstance(unused, int) or unused < 0 or not isinstance(active, int)
                or active <= 0 or active + unused != verified):
            raise ContractError(
                f"{agent['id']}: active + unused parameters must equal verified count")
        if (not isinstance(prefixes, list)
                or any(not isinstance(prefix, str) or not prefix for prefix in prefixes)):
            raise ContractError(f"{agent['id']}: invalid unused parameter prefixes")
        if bool(prefixes) != bool(agent["strip_unused_vision"]):
            raise ContractError(f"{agent['id']}: inconsistent vision stripping contract")
        active_total += active
        if agent["model_class"] not in {"causal", "multimodal"}:
            raise ContractError(f"{agent['id']}: invalid model_class")
        if agent["precision"] not in {"4bit", "fp16"}:
            raise ContractError(f"{agent['id']}: invalid precision")
        if agent["runtime_dtype"] not in {"float16", "float32", "bfloat16"}:
            raise ContractError(f"{agent['id']}: invalid runtime_dtype")
        quant_compute_dtype = agent.get(
            "quant_compute_dtype", agent["runtime_dtype"])
        if quant_compute_dtype not in {"float16", "float32", "bfloat16"}:
            raise ContractError(f"{agent['id']}: invalid quant_compute_dtype")
        if not isinstance(agent["allow_fp32_cpu_offload"], bool):
            raise ContractError(f"{agent['id']}: invalid allow_fp32_cpu_offload")
        if agent["allow_fp32_cpu_offload"] and agent["runtime_dtype"] != "float32":
            raise ContractError(
                f"{agent['id']}: fp32 CPU offload requires runtime_dtype=float32")
        if agent["device_map_strategy"] not in {
                "auto", "gemma3_four_gpu", "gemma3_two_or_four_gpu",
                "gemma3_text_single_gpu",
                "qwen35_four_gpu_or_single",
                "llama_four_gpu_or_single"}:
            raise ContractError(f"{agent['id']}: invalid device_map_strategy")
        if (agent["device_map_strategy"] in {
                    "gemma3_four_gpu", "gemma3_two_or_four_gpu"}
                and (agent["model"] != "google/gemma-3-12b-it"
                     or agent["runtime_dtype"] != "float32")):
            raise ContractError(
                f"{agent['id']}: Gemma manual mapping requires float32 runtime modules")
        text_only_runtime = agent.get("text_only_runtime", False)
        if not isinstance(text_only_runtime, bool):
            raise ContractError(f"{agent['id']}: text_only_runtime must be boolean")
        if text_only_runtime and (
                agent["model"] != "google/gemma-3-12b-it"
                or agent["device_map_strategy"] != "gemma3_text_single_gpu"
                or not agent["strip_unused_vision"]):
            raise ContractError(
                f"{agent['id']}: text-only runtime is restricted to the audited "
                "single-GPU Gemma diagnostic with legally counted stripped vision")
        fp32_residual_stream = agent.get("fp32_residual_stream", False)
        if not isinstance(fp32_residual_stream, bool):
            raise ContractError(
                f"{agent['id']}: fp32_residual_stream must be boolean")
        if fp32_residual_stream and not text_only_runtime:
            raise ContractError(
                f"{agent['id']}: fp32_residual_stream requires text_only_runtime")
        head_scale = agent.get("fp16_lm_head_input_scale", 1.0)
        if (not isinstance(head_scale, (int, float)) or isinstance(head_scale, bool)
                or not math.isfinite(float(head_scale)) or float(head_scale) < 1.0):
            raise ContractError(
                f"{agent['id']}: fp16_lm_head_input_scale must be finite and >= 1")
    if len(models) != len(agents):
        raise ContractError("agents must use distinct model checkpoints")
    if total > cap:
        raise ContractError(f"declared parameter upper bound {total:,} exceeds cap {cap:,}")
    result = dict(config)
    result["declared_parameter_total"] = total
    result["declared_parameter_headroom"] = cap - total
    result["verified_parameter_total"] = verified_total
    result["active_text_inference_parameter_total"] = active_total
    return result


def load_agent_config(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read agent config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("agent config must be a JSON object")
    return validate_agent_config(value)


def validate_inputs(rows: Sequence[Mapping[str, Any]]) -> None:
    seen = set()
    for index, row in enumerate(rows):
        subject = row.get("SubjectEntity")
        relation = row.get("Relation")
        if not isinstance(subject, str) or not subject.strip():
            raise ContractError(f"input row {index}: invalid SubjectEntity")
        if relation not in RELATION_QUESTIONS:
            raise ContractError(f"input row {index}: unknown Relation {relation!r}")
        key = subject, relation
        if key in seen:
            raise ContractError(f"duplicate input key: {key}")
        seen.add(key)


def load_synthetic_by_relation(path: Optional[Path]) -> Dict[str, List[dict]]:
    pools: Dict[str, List[dict]] = defaultdict(list)
    if path is None:
        return pools
    for row in read_jsonl(path):
        relation = row.get("Relation")
        if relation in RELATION_QUESTIONS:
            pools[relation].append(row)
    return pools


def select_synthetic_shots(pool: Sequence[dict], *, subject: str, relation: str,
                           count: int, seed: int) -> List[dict]:
    """Deterministic target-excluded, subject-distinct demonstration draw."""
    if count <= 0:
        return []
    candidates = [row for row in pool if row.get("SubjectEntity") != subject]
    rng = random.Random(stable_seed("shots", seed, subject, relation))
    rng.shuffle(candidates)
    selected = []
    used_subjects = set()
    for row in candidates:
        shot_subject = row.get("SubjectEntity")
        if not isinstance(shot_subject, str) or shot_subject in used_subjects:
            continue
        selected.append(row)
        used_subjects.add(shot_subject)
        if len(selected) == count:
            break
    return selected


def _question(subject: str, relation: str) -> str:
    return RELATION_QUESTIONS[relation].format(subject=subject)


def relation_question(subject: str, relation: str, *,
                      contract: str = "legacy") -> str:
    """Render a versioned relation question without changing frozen prompts."""
    if contract == "legacy":
        questions = RELATION_QUESTIONS
    elif contract == "official-v1":
        questions = OFFICIAL_RELATION_QUESTIONS_V1
    else:
        raise ContractError(f"unknown question contract {contract!r}")
    return questions[relation].format(subject=subject)


def _format_shots(shots: Sequence[dict], *, include_reasoning: bool = True) -> str:
    blocks = []
    for row in shots:
        question = row.get("Question") or _question(row["SubjectEntity"], row["Relation"])
        reasoning = str(row.get("think", "")).strip()
        answer = str(row.get("Answer", "None")).strip()
        if include_reasoning:
            blocks.append(
                f"QUESTION: {question}\nREASONING: {reasoning}\nANSWER: {answer}")
        else:
            blocks.append(f"QUESTION: {question}\nANSWER: {answer}")
    return "\n\n".join(blocks)


def proposal_prompt(agent: Mapping[str, Any], subject: str, relation: str,
                    shots: Sequence[dict], *,
                    question_contract: str = "legacy") -> str:
    role = agent["role"]
    proposal_output = agent.get("proposal_output", "reasoning_then_answer")
    if proposal_output not in {
            "reasoning_then_answer", "bounded_reasoning", "answer_only"}:
        raise ContractError(f"unknown proposal_output {proposal_output!r}")
    include_demo_reasoning = bool(agent.get(
        "demonstration_reasoning", proposal_output != "answer_only"))
    common = (
        "You are one independent closed-book parametric-memory agent. Do not use "
        "retrieval, tools, or facts supplied by another agent. The benchmark asks "
        "for the complete object set for one subject-relation pair. "
    )
    if relation in NUMERIC_RELATIONS:
        contract = "Return exactly one line: ANSWER: <single number>. Never return None."
    else:
        contract = (
            "Return exactly one final line beginning ANSWER:. Put a comma-separated "
            "complete object list after it, or None only when the relation is genuinely empty."
        )
    if proposal_output == "answer_only":
        style = (
            "Use the demonstrations only as private recall cues. Do not output analysis, "
            "reasoning, explanations, or <think> tags. Emit the required ANSWER line "
            "immediately."
        )
    elif proposal_output == "bounded_reasoning":
        reasoning_words = int(agent.get("proposal_reasoning_words", 20))
        if reasoning_words < 1:
            raise ContractError("proposal_reasoning_words must be positive")
        style = (
            "Use the demonstrations as private recall cues. Before the answer, emit "
            f"exactly one REASONING: line of at most {reasoning_words} words. Then emit "
            "exactly one ANSWER: line. Do not use <think> tags, add another sentence, "
            "or write anything after the answer."
        )
    elif role == "synthetic_cot_recall":
        style = (
            "Recall the specific fact rather than estimating from a generic class. "
            "Reason briefly inside <think></think>, then obey the answer contract."
        )
    elif role == "independent_direct_recall":
        style = (
            "Answer independently and directly. Do not imitate chain-of-thought or infer "
            "from what a different model would probably say."
        )
    else:
        style = (
            "Be skeptical of common default answers and distinguish remembered facts from "
            "plausible guesses. Still provide your best independent answer."
        )
    demonstrations = ""
    if shots:
        demonstrations = (
            "\n\nPRIVATE RECALL DEMONSTRATIONS:\n"
            + _format_shots(shots, include_reasoning=include_demo_reasoning))
    return (
        f"{common}{style}{demonstrations}\n\nTARGET QUESTION: "
        f"{relation_question(subject, relation, contract=question_contract)}\n"
        f"{contract}"
    )


def existence_prompt(subject: str, relation: str,
                     choice_codes: Optional[Mapping[str, str]] = None) -> str:
    null_note = (
        "The relation can genuinely be empty." if relation in NULLABLE_RELATIONS
        else "The task definition guarantees at least one real object/value."
    )
    codes = choice_codes or {"YES": "A", "NO": "B", "UNKNOWN": "C"}
    codebook = ", ".join(f"{codes[label]} = {label}"
                         for label in ("YES", "NO", "UNKNOWN"))
    return (
        "Make a candidate-blind commitment from your own closed-book memory. No candidate "
        "answers, peer outputs, vote counts, or rationales are visible. Decide whether this "
        "subject has at least one object for the relation. Do not invent an object.\n"
        f"SUBJECT: {subject}\nRELATION: {relation}\nQUESTION: {_question(subject, relation)}\n"
        f"SCHEMA: {null_note}\nChoose exactly one code: {codebook}. "
        "Return only the code.\nCODE:"
    )


def cardinality_prompt(subject: str, relation: str,
                       choice_codes: Optional[Mapping[str, str]] = None) -> str:
    choices = cardinality_choices(relation)
    codes = choice_codes or dict(zip(choices, CHOICE_CODES))
    codebook = ", ".join(
        f"{codes[choice]} = {choice}" for choice in choices)
    return (
        "Make a candidate-blind cardinality commitment from your own closed-book memory. "
        "No candidate identities or peer outputs are visible.\n"
        f"SUBJECT: {subject}\nRELATION: {relation}\nQUESTION: {_question(subject, relation)}\n"
        f"Choose exactly one code: {codebook}. Return only the code.\nCODE:"
    )


def cardinality_choices(relation: str) -> List[str]:
    if relation in NUMERIC_RELATIONS:
        return ["ONE"]
    if relation == "personHasCityOfDeath":
        return ["ZERO", "ONE", "UNKNOWN"]
    if relation == "awardWonBy":
        return ["ONE", "MANY", "UNKNOWN"]
    return ["ZERO", "ONE", "MANY", "UNKNOWN"]


def build_agent_tasks(rows: Sequence[Mapping[str, Any]], agent: Mapping[str, Any],
                      synthetic_by_relation: Mapping[str, Sequence[dict]], *,
                      seed: int, n_proposals: int,
                      question_contract: str = "legacy") -> List[dict]:
    validate_inputs(rows)
    if n_proposals < 1:
        raise ContractError("n_proposals must be >= 1")
    tasks = []
    for index, row in enumerate(rows):
        subject, relation = row["SubjectEntity"], row["Relation"]
        base = {
            "agent_id": agent["id"], "subject": subject, "relation": relation,
            "input_index": index,
        }
        existence_task = {
            **base,
            "task_id": f"{agent['id']}::{index}::existence",
            "phase": "commit_existence",
            "prompt": "",
            "choices": ["YES", "NO", "UNKNOWN"],
        }
        if relation in NULLABLE_RELATIONS:
            existence_task["mode"] = "choice"
            codebooks = balanced_choice_codebooks(
                existence_task["choices"], agent["id"], subject, relation, "existence")
            existence_task["choice_codes"] = codebooks[0]
            existence_task["choice_variants"] = [
                {"prompt": existence_prompt(subject, relation, codebook),
                 "choice_codes": codebook} for codebook in codebooks]
            existence_task["prompt"] = existence_task["choice_variants"][0]["prompt"]
        else:
            existence_task.update({"mode": "constant", "constant_choice": "YES"})
            existence_task["prompt"] = existence_prompt(subject, relation)
        tasks.append(existence_task)
        cardinality_task = {
            **base,
            "task_id": f"{agent['id']}::{index}::cardinality",
            "phase": "commit_cardinality",
            "prompt": "",
            "choices": cardinality_choices(relation),
        }
        if len(cardinality_task["choices"]) == 1:
            cardinality_task.update({"mode": "constant",
                                     "constant_choice": cardinality_task["choices"][0]})
            cardinality_task["prompt"] = cardinality_prompt(subject, relation)
        else:
            cardinality_task["mode"] = "choice"
            codebooks = balanced_choice_codebooks(
                cardinality_task["choices"], agent["id"], subject, relation,
                "cardinality")
            cardinality_task["choice_codes"] = codebooks[0]
            cardinality_task["choice_variants"] = [
                {"prompt": cardinality_prompt(subject, relation, codebook),
                 "choice_codes": codebook} for codebook in codebooks]
            cardinality_task["prompt"] = cardinality_task["choice_variants"][0]["prompt"]
        tasks.append(cardinality_task)
        shots = select_synthetic_shots(
            synthetic_by_relation.get(relation, []), subject=subject,
            relation=relation, count=int(agent.get("synthetic_shots", 0)), seed=seed)
        default_max_tokens = 192 if relation != "awardWonBy" else 384
        configured_max_tokens = agent.get("proposal_max_new_tokens", {})
        if configured_max_tokens:
            if not isinstance(configured_max_tokens, Mapping):
                raise ContractError("proposal_max_new_tokens must be a mapping")
            max_new_tokens = int(configured_max_tokens.get(
                relation, configured_max_tokens.get("default", default_max_tokens)))
        else:
            max_new_tokens = default_max_tokens
        if max_new_tokens < 1:
            raise ContractError("proposal max_new_tokens must be positive")
        tasks.append({
            **base,
            "task_id": f"{agent['id']}::{index}::proposal",
            "phase": "propose", "mode": "generate",
            "prompt": proposal_prompt(
                agent, subject, relation, shots,
                question_contract=question_contract),
            "n_samples": n_proposals,
            "temperature": 0.8,
            "max_new_tokens": max_new_tokens,
            "proposal_output": agent.get(
                "proposal_output", "reasoning_then_answer"),
            "demonstration_reasoning": bool(agent.get(
                "demonstration_reasoning",
                agent.get("proposal_output") != "answer_only")),
            "shot_subjects": [shot["SubjectEntity"] for shot in shots],
        })
    return tasks


_ANSWER_FIELD = re.compile(
    r"^[ \t]*ANSWER[ \t]*:[ \t]*([^\r\n]*)$", re.IGNORECASE | re.MULTILINE)
_ANSWER_PREFIX = re.compile(r"^[ \t]*ANSWER[ \t]*:[ \t]*", re.IGNORECASE)
_NULL_ANSWER_KEYS = {"none", "null", "no answer", "n/a", "na", "unknown",
                     "empty"}


def _strip_repeated_answer_prefix(value: str) -> str:
    """Remove model-echoed ``ANSWER:`` labels from an isolated answer value.

    Proposal prompts already end in an ANSWER field, but some checkpoints
    repeat that field in their continuation (occasionally more than once).
    Treating the echoed label as entity text creates a false candidate node.
    The loop is deliberately anchored at the start: an entity containing the
    word "answer" elsewhere is unchanged.
    """
    result = str(value).strip()
    while True:
        stripped = _ANSWER_PREFIX.sub("", result, count=1).strip()
        if stripped == result:
            return result
        result = stripped


def answer_field_status(text: str) -> tuple[str, Optional[str]]:
    """Extract explicit answer lines without requiring them to end the reply.

    Later prose is ignored, but conflicting explicit ANSWER fields fail closed.
    This keeps formatting mistakes separate from absent factual candidates.
    """
    matches = [value.strip() for value in _ANSWER_FIELD.findall(str(text))
               if value.strip()]
    if not matches:
        return "missing_answer_field", None
    normalized = {re.sub(r"[ \t]+", " ", value).strip().casefold()
                  for value in matches}
    if len(normalized) != 1:
        return "conflicting_answer_fields", None
    return "answer_field", matches[-1]


def answer_field(text: str) -> Optional[str]:
    _, answer = answer_field_status(text)
    return answer


def proposal_parse_status(text: str, relation: str) -> tuple[str, List[str]]:
    status, answer = answer_field_status(text)
    if answer is None:
        return status, []
    answer = _strip_repeated_answer_prefix(answer)
    if normalize_string(answer) in _NULL_ANSWER_KEYS:
        return "explicit_none", []
    # Lazy import is load-bearing: run_inference imports torch. Importing it at
    # module load would initialize CUDA before spawned workers establish their
    # per-worker CUDA_VISIBLE_DEVICES isolation.
    from run_inference import parse_answer_items
    items = [
        cleaned
        for item in parse_answer_items(
            answer, relation, response_protocol="legacy-cot")
        if (
            (cleaned := _strip_repeated_answer_prefix(str(item)))
            and normalize_string(cleaned) not in _NULL_ANSWER_KEYS
        )
    ]
    if not items:
        # A list containing only null markers is a valid abstention, not a
        # parser failure.  This also handles values such as
        # ``ANSWER: ANSWER: None``.
        raw_items = parse_answer_items(
            answer, relation, response_protocol="legacy-cot")
        if raw_items and all(
            normalize_string(_strip_repeated_answer_prefix(str(item)))
            in _NULL_ANSWER_KEYS
            for item in raw_items
        ):
            return "explicit_none", []
        return "unparseable_answer_field", []
    return "parsed_nonempty", items


def parse_proposal_generation(text: str, relation: str) -> List[str]:
    return proposal_parse_status(text, relation)[1]


def canonical_key(item: str, relation: Optional[str] = None) -> str:
    if relation in NUMERIC_RELATIONS:
        try:
            value = float(str(item).replace(",", ""))
        except ValueError:
            return ""
        if not math.isfinite(value) or value <= 0:
            return ""
        # String normalization removes decimal punctuation (72.9 -> 729), so
        # numerics require a typed key or materially different values collide.
        return f"numeric:{format(value, '.15g')}"
    from run_inference import canonicalize
    return canonicalize(normalize_string(item))


def proposal_candidates(response: Mapping[str, Any]) -> List[dict]:
    relation = response["relation"]
    counts: Counter = Counter()
    displays: Dict[str, str] = {}
    first: Dict[str, int] = {}
    for generation in response.get("generations", []):
        seen = set()
        for item in parse_proposal_generation(str(generation), relation):
            key = canonical_key(item, relation)
            if not key or key in seen:
                continue
            seen.add(key)
            if key not in displays:
                displays[key] = item
                first[key] = len(first)
            counts[key] += 1
    rows = [{"key": key, "item": displays[key], "support": count,
             "first_seen": first[key]}
            for key, count in counts.items()]
    return sorted(rows, key=lambda row: (-row["support"], row["first_seen"]))


def validate_task_response(task: Mapping[str, Any], response: Mapping[str, Any]) -> None:
    for field in ("task_id", "agent_id", "subject", "relation", "phase", "mode"):
        if response.get(field) != task.get(field):
            raise ContractError(
                f"{task.get('task_id')}: response {field} mismatch: "
                f"{response.get(field)!r} != {task.get(field)!r}")
    if task["mode"] == "generate":
        generations = response.get("generations")
        if not isinstance(generations, list) or len(generations) != task["n_samples"]:
            raise ContractError(f"{task['task_id']}: wrong generation count")
        if any(not isinstance(value, str) for value in generations):
            raise ContractError(f"{task['task_id']}: non-string generation")
    elif task["mode"] == "choice":
        scores = response.get("choice_scores")
        if not isinstance(scores, dict) or set(scores) != set(task["choices"]):
            raise ContractError(f"{task['task_id']}: choice score keys mismatch")
        if response.get("selected_choice") not in task["choices"]:
            raise ContractError(f"{task['task_id']}: invalid selected choice")
    elif task["mode"] == "representation":
        representation = response.get("representation")
        fractions = {
            f"{float(value):.6g}"
            for value in task["representation_layer_fractions"]}
        if (
            not isinstance(representation, dict)
            or representation.get("schema")
            != "frozen-hidden-representation-v1"
            or representation.get("projection_namespace")
            != task["representation_projection_namespace"]
            or representation.get("projection_dim")
            != task["representation_projection_dim"]
            or not isinstance(representation.get("decoder_layer_count"), int)
            or representation["decoder_layer_count"] < 1
            or not isinstance(representation.get("layers"), dict)
            or set(representation["layers"]) != fractions
        ):
            raise ContractError(
                f"{task['task_id']}: invalid representation response")
        layer_count = int(representation["decoder_layer_count"])
        for fraction, layer in representation["layers"].items():
            projection = layer.get("projection")
            expected_index = max(
                1, min(layer_count, int(round(float(fraction) * layer_count))))
            if (
                not isinstance(layer.get("layer_index"), int)
                or layer["layer_index"] != expected_index
                or not isinstance(layer.get("hidden_size"), int)
                or layer["hidden_size"] < 1
                or not isinstance(layer.get("rms"), (int, float))
                or not math.isfinite(float(layer["rms"]))
                or float(layer["rms"]) <= 0.0
                or not isinstance(projection, list)
                or len(projection)
                != task["representation_projection_dim"]
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in projection)
            ):
                raise ContractError(
                    f"{task['task_id']}: malformed representation layer")
    else:
        if response.get("selected_choice") != task.get("constant_choice"):
            raise ContractError(f"{task['task_id']}: constant choice mismatch")


def softmax(values: Mapping[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    maximum = max(values.values())
    exponentials = {key: math.exp(value - maximum) for key, value in values.items()}
    denominator = sum(exponentials.values())
    return {key: value / denominator for key, value in exponentials.items()}


def review_prompt(subject: str, relation: str, candidate: str,
                  choice_codes: Optional[Mapping[str, str]] = None) -> str:
    """Source-blind review prompt: deliberately contains no proposer metadata."""
    labels = ("SUPPORTED", "CONTRADICTED", "UNKNOWN")
    codes = choice_codes or dict(zip(labels, CHOICE_CODES))
    codebook = ", ".join(f"{codes[label]} = {label}" for label in labels)
    return (
        "Evaluate one factual claim from your own closed-book memory. The claim source, "
        "number of votes, and any reasoning trace are intentionally hidden. Do not endorse "
        "a claim merely because it is plausible. SUPPORTED means you independently remember "
        "the exact fact; CONTRADICTED means you independently remember it is false; UNKNOWN "
        "means you cannot reliably decide.\n"
        f"SUBJECT: {subject}\nRELATION: {relation}\nCANDIDATE OBJECT: {candidate}\n"
        f"Choose exactly one code: {codebook}. Return only the code.\nCODE:"
    )


def build_review_tasks(claim_graphs: Sequence[Mapping[str, Any]],
                       agents: Sequence[Mapping[str, Any]]) -> List[dict]:
    tasks = []
    for graph_index, graph in enumerate(claim_graphs):
        if graph["Relation"] in NUMERIC_RELATIONS:
            # Numeric review is not consumed by the current decoder; do not
            # spend two additional model calls per numeric candidate.
            continue
        for candidate_index, candidate in enumerate(graph.get("candidates", [])):
            proposers = set(candidate.get("proposer_agents", []))
            for agent in agents:
                if agent["id"] in proposers:
                    continue
                choices = ["SUPPORTED", "CONTRADICTED", "UNKNOWN"]
                codebooks = balanced_choice_codebooks(
                    choices, agent["id"], graph["SubjectEntity"], graph["Relation"],
                    candidate["key"], "review")
                choice_codes = codebooks[0]
                choice_variants = [
                    {"prompt": review_prompt(
                        graph["SubjectEntity"], graph["Relation"], candidate["item"],
                        codebook), "choice_codes": codebook}
                    for codebook in codebooks]
                tasks.append({
                    "task_id": (f"{agent['id']}::{graph_index}::review::"
                                f"{candidate_index}::{candidate['key']}"),
                    "agent_id": agent["id"], "subject": graph["SubjectEntity"],
                    "relation": graph["Relation"], "input_index": graph_index,
                    "phase": "blind_review", "mode": "choice",
                    "prompt": choice_variants[0]["prompt"],
                    "choices": choices,
                    "choice_codes": choice_codes,
                    "choice_variants": choice_variants,
                    # Provenance is retained for audit but never interpolated into prompt.
                    "candidate_key": candidate["key"],
                    "candidate_item": candidate["item"],
                    "excluded_proposer_agents": sorted(proposers),
                })
    return tasks
