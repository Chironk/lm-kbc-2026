#!/usr/bin/env python3
"""Dynamic 5-shot, self-consistency (n_c=10) inference over JSONL inputs.

New runs retain the factual-recall-positive legacy CoT prompt while using the
measured robust aggregation profile: malformed generations are not semantic
None votes, lists use valid-sample denominators, city uses the K=5 support
gate, and numerics use relative-mode clustering. More invasive prompt and
demonstration changes remain explicit ablation flags until they win paired
held-out evaluation.

Runs one quantized (4-bit) model replica per available GPU -- the same
multi-worker pattern validated in run_baseline.py tonight (CUDA_VISIBLE_DEVICES
pinning, a memory-fraction cap with CPU overflow, and the
PYTORCH_CUDA_ALLOC_CONF fragmentation fix), reused here deliberately rather
than re-derived, since building a fresh multi-GPU dispatcher risks
reintroducing the exact OOM/fragmentation bugs that took real time to fix.

Usage:
    python3 run_inference.py
    python3 run_inference.py --input data/val.jsonl --output data/predictions.jsonl
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Dict, List, Optional

# Must be set before torch initializes CUDA -- see run_baseline.py for why
# (long runs mix short and long generations; the default allocator can
# fragment enough to fail a later large allocation even with aggregate free
# memory available).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from evaluate import RELATION_TYPE, normalize_string, read_jsonl_file
from models.baseline_qwen import load_prompt_templates, parse_answer

MODEL = "Qwen/Qwen3.5-9B"
# Current local Hugging Face snapshot. Pinning prevents an upstream model or
# tokenizer update from silently changing a future artifact.
DEFAULT_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
N_SHOTS = 5
# Self-consistency samples per subject. Measured (post-hoc subsample of a full
# 20-sample run): pooled macro-F1 is 0.317@N=1, 0.405@N=5, 0.437@N=10,
# 0.430@N=20 -- voting saturates by 10, so 20 was double the cost for no gain.
# 10 keeps the full voting benefit at half the generation time.
N_CONSISTENCY = 10
TEMPERATURE = 1.0
MIXED_TEMPERATURES = [0.25] * 3 + [0.55] * 3 + [0.80] * 2 + [1.0] * 2
# Budget per generation. With the v2 exemplars demonstrating 3-6 sentence
# walkthroughs, the student imitates reasoning of ~100-160 tokens + the answer,
# so 256 is the ceiling (generation still stops early at <|im_end|> via
# stop_strings, so unused budget costs nothing). awardWonBy keeps its larger
# budget below because its answer lists are genuinely long.
MAX_NEW_TOKENS = 256
# awardWonBy needs far more budget than the spec's stated minimum -- we have
# direct evidence from tonight's runs that 128 tokens truncates its long
# recipient lists (some subjects have 40-70+ names). Deviating from the
# literal spec here because we've already measured the failure mode.
AWARD_MAX_NEW_TOKENS = 384

# One-to-many list relations get entity-level threshold voting: keep an
# entity only if it appears in at least this fraction of the n_c samples.
# awardWonBy's low threshold reflects its huge, sparse answer sets (a real
# winner may only surface in 2-3 of 20 samples); the tighter relations have
# small answer sets where a genuine fact should recur often.
LIST_VOTE_THRESHOLDS = {
    "countryLandBordersCountry": 0.5,
    "companyTradesAtStockExchange": 0.35,
    "awardWonBy": 0.10,
}

# The frozen 0.468 artifact is rebuilt by reproduce_hybrid.py, which calls the
# legacy aggregation functions directly. New inference runs use this explicit
# robust profile by default; --aggregation-profile legacy is retained for clean
# A/B comparisons against the historical behavior.
ROBUST_LIST_VOTE_THRESHOLDS = {
    # Cross-cache sweep with the denominator restricted to valid generations:
    # mean/min F1 = .955/.952 at .25, vs .944/.939 for the legacy denominator.
    "countryLandBordersCountry": 0.25,
    # .35 was the most stable valid-denominator company setting across A/B/C/v2.
    "companyTradesAtStockExchange": 0.35,
    "awardWonBy": 0.10,
}
ROBUST_CITY_MIN_VOTES = 5
ROBUST_NUMERIC_CLUSTER_WIDTH = 0.20
SINGLE_VOTE_RELATIONS = {"personHasCityOfDeath"}  # majority vote, "None" is a valid candidate
NUMERIC_RELATIONS = {"hasCapacity", "hasArea"}

# Two instructions, one per relation TYPE. The key correctness fix: numeric
# relations must NEVER be invited to answer "None" -- every stadium has a
# capacity and every place has an area, so "None" is always wrong there, yet
# the old single one-size instruction explicitly offered it (measured: 37% of
# hasCapacity samples abstained). List/single relations DO have legitimate
# nulls in the data, so "None" stays a valid option only for them.
STRING_SYSTEM_INSTRUCTION = (
    "You answer factual questions. First reason in TWO to FOUR focused sentences "
    "inside <think></think> tags: state what the subject is and the specific "
    "fact(s) that determine the answer. Then, on the next line, give ONLY the "
    "final answer -- the object(s) as a comma-separated list, or \"None\" if "
    "there genuinely is no answer. Write nothing after the answer."
)
NUMERIC_SYSTEM_INSTRUCTION = (
    "You answer questions that require a single number. First reason in TWO to "
    "FOUR focused sentences inside <think></think> tags: state what the subject is "
    "and the specific value you recall. Then, on the next line, give ONLY the "
    "number -- no units, no commas, no words. Every subject HAS a real value, so "
    "never answer \"None\" or \"unknown\"; always give your single best estimate."
)

# Robust response contract.  The answer is emitted BEFORE any optional
# evidence, so an early end-of-turn can no longer destroy an otherwise usable
# fact.  " || " avoids the ambiguity of comma-separated entities whose real
# names themselves contain commas.
ANSWER_FIRST_STRING_INSTRUCTION = (
    "Answer factual questions from your own knowledge. Your first line MUST be "
    "`ANSWER: ` followed by the object names separated by ` || `, or `None` "
    "when the true answer set is empty. On the second line you may write "
    "`EVIDENCE: ` followed by at most ONE short sentence. Never put reasoning "
    "before the ANSWER line and never invent an item merely to fill the list."
)
ANSWER_FIRST_NUMERIC_INSTRUCTION = (
    "Answer with a single factual number from your own knowledge. Your first "
    "line MUST be `ANSWER: ` followed only by the number, with no units or "
    "commas. Every subject has a real value, so never answer None or unknown. "
    "On the second line you may write `EVIDENCE: ` followed by at most ONE "
    "short sentence. Never put reasoning before the ANSWER line."
)

CLOZE_PROMPTS = {
    "personHasCityOfDeath": "{subject_entity} died in the city of ____.",
    "companyTradesAtStockExchange": "Shares of {subject_entity} trade on ____.",
    "countryLandBordersCountry": "{subject_entity} shares land borders with ____.",
    "hasArea": "The area of {subject_entity} is ____ square kilometers.",
    "hasCapacity": "The maximum spectator capacity of {subject_entity} is ____ people.",
    "awardWonBy": "Recipients of the {subject_entity} include ____.",
}

# --- Per-relation instruction overrides (each behind its own CLI flag, so the
# ablation can attribute per-relation deltas to exactly one change) ---

# v2-run diagnosis: 36/100 companies with a real listing got "None", and in
# those the student abstained in 6.8/10 samples on average -- the exemplars'
# (epistemically correct) "can't verify -> None" pattern over-fires on
# obscure-but-listed companies. This softens the abstention criterion only.
COMPANY_SOFT_ABSTAIN_INSTRUCTION = (
    "You answer factual questions about where companies' shares trade. First "
    "reason in TWO to FOUR focused sentences inside <think></think> tags: state "
    "what the company is and its listing status. If the company is real and "
    "plausibly publicly traded, name the most likely exchange(s) -- typically "
    "its home country's main exchange -- even if you are not fully certain. "
    "Answer \"None\" ONLY when you have a concrete reason to believe it is "
    "private, defunct, acquired, or a subsidiary. Then, on the next line, give "
    "ONLY the final answer -- the exchange(s) as a comma-separated list, or "
    "\"None\". Write nothing after the answer."
)

# v2-run diagnosis: hasCapacity predictions skew high (median pred/gt 1.16,
# 57 over vs 33 under) -- when the exact number isn't recalled, the exemplars'
# "expected scale for this venue class" step fills the gap with a
# class-typical value, inflating small venues. This says recall-beats-class.
CAPACITY_RECALL_FIRST_INSTRUCTION = (
    "You answer questions about a venue's spectator capacity. First reason in "
    "TWO to FOUR focused sentences inside <think></think> tags: identify THIS "
    "exact venue and state the specific official capacity you recall for it. "
    "Do NOT estimate from what venues of its type typically hold -- a specific "
    "remembered figure always beats a class-typical guess, and small venues "
    "are easily overestimated. Then, on the next line, give ONLY the number -- "
    "no units, no commas, no words. Never answer \"None\"; always give your "
    "single best figure."
)

# Filled per-worker from CLI flags; consulted by build_prompt before the
# generic numeric/string instructions.
INSTRUCTION_OVERRIDES: Dict[str, str] = {}

# --- Tie-router for personHasCityOfDeath ---
# A perfect vote tie is a signal the subject is especially hard; breaking it
# by sample order is an accident. When --city-tie-judge is on, tied cases are
# routed to a judge pass on the SAME loaded model (one extra greedy
# generation) that re-reads the reasoning behind each tied candidate and
# ALWAYS picks one (forced choice: exact candidate match in the reply, else
# "None" mention, else the candidate with the largest character overlap).
TIE_JUDGE_INSTRUCTION = (
    "You are judging answers to a question about where a person died. Several "
    "candidate answers received equal support from independent reasoning "
    "attempts. Read the reasoning behind each candidate. Pick the candidate "
    "whose reasoning is the most specific and factually grounded (real dates, "
    "real biographical facts). Prefer \"None\" only if the doubt about the "
    "person having died at all is better grounded than any named city. Reply "
    "with EXACTLY the winning candidate answer and nothing else."
)


def find_vote_tie(raw_samples: List[str]):
    """Return [(candidate_display, [reasoning traces])] for tied top
    candidates, or None when there is a clear winner.

    MUST count votes exactly like aggregate_single_vote does -- including
    empty/unclosed samples, which parse to no items and therefore vote
    "none" -- otherwise this detects "ties" the live vote never sees (and
    misses real ones)."""
    counts: Dict[str, int] = {}
    support: Dict[str, List[str]] = {}
    display: Dict[str, str] = {}
    for s in raw_samples:
        ans = extract_after_think(s)
        items = parse_answer(ans, is_numeric=False)
        key = "none" if not items else canonicalize(normalize_string(items[0]))
        counts[key] = counts.get(key, 0) + 1
        display.setdefault(key, "None" if key == "none" else items[0])
        if ans:  # only closed samples contribute readable reasoning traces
            think = s.split("<think>", 1)[-1].split("</think>", 1)[0].strip() if "<think>" in s else ""
            support.setdefault(key, []).append(think)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) < 2 or ranked[0][1] != ranked[1][1]:
        return None
    top = ranked[0][1]
    return [(display[k], support.get(k, [])[:2]) for k, v in ranked if v == top]


def judge_tie(model, tok, question: str, tied) -> str:
    """One greedy judge generation; ALWAYS returns one of the tied candidates."""
    body = [f"Question: {question}", "", "Tied candidates and their reasoning:"]
    for i, (cand, thinks) in enumerate(tied, 1):
        body.append(f"\nCandidate {i}: {cand}")
        for t in thinks:
            body.append(f"  reasoning: {t[:400]}")
    body.append("\nWhich candidate is best supported? Reply with exactly that answer.")
    # Empty-think prefill: without it the format-primed student opens a fresh
    # <think> block and never emits a bare answer within the token budget.
    prompt = (f"{IM_START}system\n{TIE_JUDGE_INSTRUCTION}{IM_END}\n"
              f"{IM_START}user\n" + "\n".join(body) + f"{IM_END}\n"
              f"{IM_START}assistant\n<think></think>\nANSWER:")
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    tnorm = normalize_string(text)
    candidates = [c for c, _ in tied]
    for c in candidates:
        if normalize_string(c) and normalize_string(c) in tnorm:
            return c
    if re.search(r"\bnone\b", text, re.IGNORECASE):
        for c in candidates:
            if c == "None":
                return c
    # forced choice: largest character overlap with the judge's reply
    def overlap(c):
        cn = normalize_string(c)
        return sum(1 for ch in set(cn) if ch in tnorm) / max(len(set(cn)), 1)
    return max(candidates, key=overlap)


def load_synthetic_examples(path: str) -> Dict[str, List[Dict]]:
    by_relation: Dict[str, List[Dict]] = defaultdict(list)
    for row in read_jsonl_file(path):
        by_relation[row["Relation"]].append(row)
    return by_relation


NULL_SHOTS_PER_FIVE = {
    "personHasCityOfDeath": 2,
    "companyTradesAtStockExchange": 2,
    "countryLandBordersCountry": 1,
}


def _example_is_null(example: Dict) -> bool:
    if "ObjectEntities" in example:
        return not bool(example["ObjectEntities"])
    return not bool(parse_answer(example.get("Answer", ""), is_numeric=False))


def parse_null_shots(spec: str) -> Dict[str, int]:
    """Parse RELATION=COUNT pairs for five-shot null-calibration ablations."""
    result: Dict[str, int] = {}
    if not spec:
        return result
    for entry in spec.split(","):
        if "=" not in entry:
            raise ValueError(f"Invalid null-shot entry {entry!r}; expected RELATION=COUNT")
        relation, raw_count = (part.strip() for part in entry.split("=", 1))
        if relation not in RELATION_TYPE:
            raise ValueError(f"Unknown null-shot relation: {relation}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"Invalid null-shot count for {relation}: {raw_count}") from exc
        if not 0 <= count <= 5:
            raise ValueError(f"Null-shot count must be in [0, 5], got {relation}={count}")
        result[relation] = count
    return result


def sample_null_stratified_shots(pool: List[Dict], relation: str, n_shots: int,
                                 rng: random.Random,
                                 null_per_five: Dict[str, int]) -> List[Dict]:
    """Legacy row sampling with an exact null/non-null prompt composition.

    Unlike duplicating JSONL rows, this samples without replacement and cannot
    put the same exemplar in a prompt twice.  Unlike subject-balanced sampling,
    it changes only the answer-class composition and otherwise retains legacy
    path-level sampling, making the None-rate hypothesis directly testable.
    """
    # An omitted relation is outside the intervention.  Falling back to the
    # legacy row sampler is essential for clean relation-specific ablations;
    # treating omission as zero would silently remove every null demonstration
    # from unrelated relations (especially city-of-death).
    if relation not in null_per_five:
        return rng.sample(pool, min(n_shots, len(pool))) if pool else []
    wanted_null = round(null_per_five[relation] * n_shots / 5)
    null_rows = [example for example in pool if _example_is_null(example)]
    nonnull_rows = [example for example in pool if not _example_is_null(example)]
    wanted_null = min(wanted_null, len(null_rows), n_shots)
    wanted_nonnull = min(n_shots - wanted_null, len(nonnull_rows))
    chosen = rng.sample(null_rows, wanted_null)
    chosen += rng.sample(nonnull_rows, wanted_nonnull)
    if len(chosen) < n_shots:
        remaining = [example for example in pool if example not in chosen]
        chosen += rng.sample(remaining, min(n_shots - len(chosen), len(remaining)))
    rng.shuffle(chosen)
    return chosen


def sample_subject_balanced_shots(pool: List[Dict], relation: str, n_shots: int,
                                  rng: random.Random) -> List[Dict]:
    """Choose distinct subjects, then one path per subject.

    Sampling the flat exemplar pool gave easy subjects up to six times the
    probability of hard subjects and occasionally placed two paths for the
    same subject in one prompt.  This removes both biases and explicitly
    controls null demonstrations for relations where abstention is legitimate.
    Numeric shots are spread across value quantiles to reduce round-number
    anchoring.
    """
    by_subject: Dict[str, List[Dict]] = defaultdict(list)
    for example in pool:
        by_subject[example["SubjectEntity"]].append(example)
    subjects = list(by_subject)
    if not subjects:
        return []

    chosen: List[str] = []
    if relation in NUMERIC_RELATIONS:
        valued = []
        for subject in subjects:
            parsed = parse_answer(by_subject[subject][0].get("Answer", ""), True)
            try:
                value = float(parsed[0].replace(",", "")) if parsed else None
            except ValueError:
                value = None
            if value is not None and value > 0:
                valued.append((value, subject))
        valued.sort()
        if valued:
            for i in range(min(n_shots, len(valued))):
                lo = int(i * len(valued) / min(n_shots, len(valued)))
                hi = max(lo + 1, int((i + 1) * len(valued) /
                                     min(n_shots, len(valued))))
                chosen.append(rng.choice([subject for _, subject in valued[lo:hi]]))
    else:
        null_subjects = [s for s in subjects if _example_is_null(by_subject[s][0])]
        nonnull_subjects = [s for s in subjects if s not in set(null_subjects)]
        wanted_null = round(NULL_SHOTS_PER_FIVE.get(relation, 0) * n_shots / 5)
        wanted_null = min(wanted_null, len(null_subjects), n_shots)
        chosen.extend(rng.sample(null_subjects, wanted_null))
        remaining = n_shots - len(chosen)
        chosen.extend(rng.sample(nonnull_subjects, min(remaining, len(nonnull_subjects))))

    if len(chosen) < n_shots:
        remaining_subjects = [s for s in subjects if s not in set(chosen)]
        chosen.extend(rng.sample(remaining_subjects,
                                 min(n_shots - len(chosen), len(remaining_subjects))))
    rng.shuffle(chosen)
    return [rng.choice(by_subject[subject]) for subject in chosen[:n_shots]]


def sample_diverse_shot_sets(pool: List[Dict], relation: str, n_shots: int,
                             n_samples: int, base_seed: int) -> List[List[Dict]]:
    """Build a different subject-balanced demonstration set per completion.

    Historical self-consistency sampled one five-shot prompt and cloned it ten
    times. That makes all completions inherit the same anchors and wastes the
    multiple paths in the synthetic pool. This preserves five-shot prompts but
    diversifies the demonstrated subjects/path for every completion.
    """
    sets: List[List[Dict]] = []
    signatures = set()
    for sample_idx in range(n_samples):
        chosen = None
        # Different deterministic seeds normally suffice; bounded retries avoid
        # an accidental identical set without risking an infinite loop on a
        # tiny diagnostic pool.
        for retry in range(8):
            rng = random.Random(base_seed + sample_idx * 1009 + retry * 104729)
            candidate = sample_subject_balanced_shots(pool, relation, n_shots, rng)
            signature = tuple((x["SubjectEntity"], x.get("path")) for x in candidate)
            chosen = candidate
            if signature not in signatures or len(pool) <= n_shots:
                signatures.add(signature)
                break
        sets.append(chosen or [])
    return sets


def select_curated_shots(curated: Dict[str, List[Dict]], relation: str,
                         subject: str, n_shots: int,
                         exclude_target: bool) -> List[Dict]:
    """Select fixed shots without leaking the evaluated subject's exemplar."""
    pool = curated.get(relation, [])
    if exclude_target:
        pool = [example for example in pool
                if example["SubjectEntity"] != subject]
    return pool[:n_shots]


# Defensive cap on how much of each exemplar's <think> we put in the prompt.
# Human-written demonstrations span 2-4 sentences / 200-428 characters, and
# synthetic v3 explicitly matches that distribution. 700 lets those through
# intact -- it is insurance against a stray runaway
# exemplar blowing up the KV-cache (the failure mode that OOM'd the run when
# thinks were 2000+ chars), not a style constraint. Truncates at a sentence
# boundary when possible so the demonstrated reasoning stays coherent.
MAX_SHOT_THINK_CHARS = 700


def _cap_think(think: str) -> str:
    think = think.strip()
    if len(think) <= MAX_SHOT_THINK_CHARS:
        return think
    cut = think[:MAX_SHOT_THINK_CHARS]
    last_period = cut.rfind(". ")
    return cut[: last_period + 1] if last_period > 0 else cut


def _one_sentence(text: str, max_chars: int = 240) -> str:
    """Compact teacher reasoning for the answer-first protocol.

    Historical prompts demonstrated 3-6 sentence rationales even while the
    system instruction requested 1-2 sentences.  Qwen often copied the longer
    format and ended its turn before emitting an answer.  Keep only the first
    complete sentence (or a bounded prefix) so demonstrations and instructions
    agree.
    """
    compact = " ".join((text or "").split()).strip()
    if not compact:
        return ""
    match = re.match(r"(.{1,%d}?[.!?])(?:\s|$)" % max_chars, compact)
    if match:
        return match.group(1)
    return compact[:max_chars].rstrip()


def _shot_answer_first(shot: Dict, relation: str) -> str:
    """Render an exemplar without relying on ambiguous comma parsing."""
    objects = shot.get("ObjectEntities")
    if objects is not None:
        items = [aliases[0] for aliases in objects if aliases]
    else:
        items = parse_answer(shot.get("Answer", ""), relation in NUMERIC_RELATIONS)
    answer = "None" if not items else " || ".join(items)
    evidence = _one_sentence(shot.get("think", ""))
    return f"ANSWER: {answer}" + (f"\nEVIDENCE: {evidence}" if evidence else "")


def _shot_answer_only(shot: Dict, relation: str) -> str:
    objects = shot.get("ObjectEntities")
    if objects is not None:
        items = [aliases[0] for aliases in objects if aliases]
    else:
        items = parse_answer(shot.get("Answer", ""), relation in NUMERIC_RELATIONS)
    return "ANSWER: " + ("None" if not items else " || ".join(items))


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt(tok, template: str, relation: str, subject: str, shots: List[Dict],
                 use_cot: bool = True, response_protocol: str = "legacy-cot",
                 prompt_variant: str = "direct",
                 legacy_think_prefill: bool = False) -> str:
    """Build the ChatML prompt as RAW TEXT rather than via
    tok.apply_chat_template(). This is deliberate: Qwen3.5's chat template
    STRIPS <think>...</think> blocks out of every prior assistant turn (it drops
    reasoning from history to save context) and, with enable_thinking=False,
    injects an empty <think></think> before generation that tells the model
    "thinking is done, just answer". Both behaviours destroy our few-shot CoT
    demonstrations -- which is exactly why the student reproduced the <think>
    format only ~2% of the time. Assembling the ChatML tokens by hand keeps the
    demonstrated <think> blocks verbatim, so the student can imitate the
    human-written reasoning style.

    use_cot=False drops the <think> block (assistant turn = bare answer) -- the
    ablation that isolates whether the chain-of-thought does any work beyond the
    answer-format demonstration + self-consistency voting."""
    is_numeric = relation in NUMERIC_RELATIONS
    if response_protocol == "answer-first":
        system = (ANSWER_FIRST_NUMERIC_INSTRUCTION if is_numeric
                  else ANSWER_FIRST_STRING_INSTRUCTION)
        # Preserve the empirically useful company willingness-to-answer rule,
        # but keep the answer-first output contract load-bearing.
        if relation == "companyTradesAtStockExchange" and relation in INSTRUCTION_OVERRIDES:
            system += (
                " For company listings, answer None only with concrete evidence "
                "that the company is private, defunct, acquired, or an unlisted "
                "subsidiary; otherwise name its most likely public exchange."
            )
        if relation == "hasCapacity" and relation in INSTRUCTION_OVERRIDES:
            system += (
                " Recall this exact venue's published capacity; do not substitute "
                "a class-typical estimate for a specific remembered figure."
            )
    else:
        system = INSTRUCTION_OVERRIDES.get(relation) or (
            NUMERIC_SYSTEM_INSTRUCTION if is_numeric else STRING_SYSTEM_INSTRUCTION)

    parts = [f"{IM_START}system\n{system}{IM_END}\n"]
    for shot in shots:
        shot_question = (CLOZE_PROMPTS[relation].format(
            subject_entity=shot["SubjectEntity"])
            if prompt_variant == "cloze" else shot["Question"])
        parts.append(f"{IM_START}user\n{shot_question}{IM_END}\n")
        if response_protocol == "answer-first":
            answer = (_shot_answer_first(shot, relation) if use_cot
                      else _shot_answer_only(shot, relation))
        elif use_cot:
            answer = f"<think>{_cap_think(shot['think'])}</think>\n{shot['Answer']}"
        else:
            answer = shot["Answer"]
        parts.append(f"{IM_START}assistant\n{answer}{IM_END}\n")
    question = ((CLOZE_PROMPTS[relation] if prompt_variant == "cloze" else template)
                .format(subject_entity=subject))
    parts.append(f"{IM_START}user\n{question}{IM_END}\n")
    if response_protocol == "answer-first":
        # Qwen3.5's reasoning mode can reopen <think> even after a bare ANSWER
        # prefill. Explicitly closing an empty thinking block is its native
        # control for direct answering; this makes the next generated tokens
        # the answer instead of another potentially truncated monologue. The
        # decoded raw sample excludes this prompt prefix, so the parser accepts
        # a bare first line or a repeated ANSWER line.
        parts.append(f"{IM_START}assistant\n<think></think>\nANSWER: ")
    elif legacy_think_prefill:
        # Match Qwen's native thinking-mode continuation contract while still
        # assembling raw ChatML so prior demonstrations retain their think
        # blocks.  The generated continuation does not contain this prompt
        # prefix; run_worker reattaches it before parsing/caching.
        parts.append(f"{IM_START}assistant\n<think>\n")
    else:
        parts.append(f"{IM_START}assistant\n")  # bare -- no forced empty <think>
    return "".join(parts)


def extract_after_think(text: str) -> str:
    """Isolate the final answer that follows the </think> tag.

    Three cases:
    - closed <think>...</think>: return what comes after (the answer).
    - an OPENED but unclosed <think> (the model hit a limit or emitted an early
      EOS/stop before the answer): return "" so this malformed sample is dropped from voting,
      rather than leaking the raw reasoning in as a fake answer (which produced
      garbage predictions like ['<think>United Wire Factories Co']).
    - no <think> at all (model answered directly): return the whole text."""
    idx = text.find("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    if "<think>" in text:
        return ""
    return text.strip()


NULL_TEXT = {"none", "n/a", "na", "unknown", "no answer", "empty", "null"}


def extract_answer_with_status(text: str, response_protocol: str = "legacy-cot"):
    """Return (answer_text, status) without conflating format failure and None.

    Status is one of valid, explicit-none, unclosed-think, or empty.  Frozen
    reproduction continues to use extract_after_think directly; this richer
    contract is for new inference caches and robust aggregation.
    """
    raw = (text or "").strip()
    if response_protocol == "legacy-cot":
        if "<think>" in raw and "</think>" not in raw:
            return "", "unclosed-think"
        answer = extract_after_think(raw)
    else:
        if not raw:
            return "", "empty"
        # Prefer an explicit ANSWER line if the model repeated the prefixed
        # label; otherwise the first generated line is the continuation after
        # the prompt's `ANSWER: ` prefill.
        explicit = re.search(r"(?:^|\n)\s*ANSWER\s*:\s*(.*)", raw, re.IGNORECASE)
        answer = explicit.group(1).strip() if explicit else raw.splitlines()[0].strip()
        if answer.startswith("<think>"):
            return "", "unclosed-think" if "</think>" not in raw else "empty"

    answer = answer.strip().strip(" .")
    if not answer:
        return "", "empty"
    if answer.lower() in NULL_TEXT:
        return answer, "explicit-none"
    return answer, "valid"


def parse_answer_items(answer: str, relation: str,
                       response_protocol: str = "legacy-cot") -> List[str]:
    """Relation-aware parser for an already isolated answer field."""
    if not answer or answer.strip().lower() in NULL_TEXT:
        return []
    is_numeric = relation in NUMERIC_RELATIONS
    if response_protocol == "answer-first" and not is_numeric and "||" in answer:
        return [p.strip(" .") for p in answer.split("||") if p.strip(" .")]
    return parse_answer(answer, is_numeric=is_numeric)


def drop_self_reference(subject: str, items: List[str]) -> List[str]:
    subj_norm = subject.strip().lower()
    return [item for item in items if item.strip().lower() != subj_norm]


# Small, bounded canonicalization for known abbreviation variants -- applied
# only to the vote-counting KEY, never to the stored display text. This is
# not an attempt at general synonym resolution (that's unbounded and
# unvalidatable); it targets specific cases we've actually observed
# fragmenting votes in our own predictions tonight (e.g. 'NYSE' and 'New
# York Stock Exchange' both appearing as separate common answers for the
# same company, each individually under-counted relative to their combined
# support). evaluate.py's own alias matching already handles this at the
# final scoring stage -- this only prevents our OWN threshold vote from
# under-counting a well-supported entity due to surface-form fragmentation
# before it ever reaches that scoring stage.
SYNONYM_CANONICAL_FORM = {
    "nyse": "new york stock exchange",
    "lse": "london stock exchange",
    "nasdaq": "nasdaq",  # already its own canonical form, listed for clarity
    "usa": "united states",
    "us": "united states",
    "u s a": "united states",
    "uk": "united kingdom",
    "u k": "united kingdom",
    "uae": "united arab emirates",
    "prc": "china",
}


def canonicalize(normalized_item: str) -> str:
    return SYNONYM_CANONICAL_FORM.get(normalized_item, normalized_item)


def aggregate_single_vote(answers: List[str]) -> List[str]:
    """personHasCityOfDeath: majority vote over atomic answers, "None" is a
    valid candidate that can win."""
    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for ans in answers:
        items = parse_answer(ans, is_numeric=False)
        key = "none" if not items else canonicalize(normalize_string(items[0]))
        counts[key] = counts.get(key, 0) + 1
        if key != "none":
            originals.setdefault(key, items[0])
    if not counts:
        return []
    best_key = max(counts, key=counts.get)
    return [] if best_key == "none" else [originals[best_key]]


def aggregate_city_support_gate(answers: List[str], min_votes: int,
                                response_protocol: str = "legacy-cot") -> List[str]:
    """Commit only when an actual city reaches absolute support.

    Explicit None and malformed generations are deliberately different states,
    but neither contributes evidence for a named city.  This is the K=5 rule
    measured positive on B/C/v2.
    """
    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for answer in answers:
        items = parse_answer_items(answer, "personHasCityOfDeath", response_protocol)
        if not items:
            continue
        key = canonicalize(normalize_string(items[0]))
        counts[key] = counts.get(key, 0) + 1
        originals.setdefault(key, items[0])
    if not counts:
        return []
    best = max(counts, key=counts.get)
    return [originals[best]] if counts[best] >= min_votes else []


def aggregate_single_vote_parsed(answers: List[str], response_protocol: str) -> List[str]:
    """Legacy plurality semantics with protocol-aware entity parsing."""
    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for answer in answers:
        items = parse_answer_items(answer, "personHasCityOfDeath", response_protocol)
        key = "none" if not items else canonicalize(normalize_string(items[0]))
        counts[key] = counts.get(key, 0) + 1
        if items:
            originals.setdefault(key, items[0])
    if not counts:
        return []
    best = max(counts, key=counts.get)
    return [] if best == "none" else [originals[best]]


def aggregate_threshold_vote(answers: List[str], n_samples: int, threshold: float) -> List[str]:
    """One-to-many list relations: flatten every sample's items, keep an
    entity only if it recurs in at least `threshold` fraction of samples."""
    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for ans in answers:
        items = parse_answer(ans, is_numeric=False)
        seen_this_sample = set()
        for item in items:
            key = canonicalize(normalize_string(item))
            if not key or key in seen_this_sample:
                continue  # count each entity at most once per sample
            seen_this_sample.add(key)
            counts[key] = counts.get(key, 0) + 1
            originals.setdefault(key, item)
    return [originals[k] for k, c in counts.items() if c / n_samples >= threshold]


def aggregate_item_vote(answer_items: List[List[str]], denominator: int,
                        threshold: float) -> List[str]:
    """Threshold vote over pre-parsed items (preserves commas inside names)."""
    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for items in answer_items:
        seen_this_sample = set()
        for item in items:
            key = canonicalize(normalize_string(item))
            if not key or key in seen_this_sample:
                continue
            seen_this_sample.add(key)
            counts[key] = counts.get(key, 0) + 1
            originals.setdefault(key, item)
    return [originals[k] for k, count in counts.items()
            if count / max(denominator, 1) >= threshold]


def is_pure_numeric_candidate(answer_text: str) -> bool:
    """Type-constraint gate: reject a sample outright if it contains any
    alphabetic character, rather than relying solely on regex-extraction.
    Catches a hedge like "not certain, but perhaps around 2023" where a
    stray number would otherwise get pulled out and treated as a confident
    answer -- parse_answer's regex would find "2023" there even though the
    sample is really a refusal."""
    return not any(c.isalpha() for c in answer_text)


def _median_str(values: List[float]) -> List[str]:
    values.sort()
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 == 1 else (values[mid - 1] + values[mid]) / 2
    return [str(int(median))] if median == int(median) else [str(median)]


def aggregate_median(answers: List[str]) -> List[str]:
    """Numeric relations NEVER have an empty ground truth (every stadium has a
    capacity, every place an area), so this must not return []. Primary pass:
    median over samples that are cleanly numeric (the purity gate rejects hedges
    like "around 2023"). Fallback: if that leaves nothing (e.g. every sample
    wrapped its number in words), relax the gate and take the median of ANY
    number found -- a best-effort guess always beats a guaranteed-wrong empty."""
    strict = []
    for ans in answers:
        if not is_pure_numeric_candidate(ans):
            continue
        parsed = parse_answer(ans, is_numeric=True)
        if parsed:
            try:
                strict.append(float(parsed[0].replace(",", "")))
            except ValueError:
                pass
    if strict:
        return _median_str(strict)

    # Fallback: never abstain on a numeric relation -- pull any number out.
    loose = []
    for ans in answers:
        parsed = parse_answer(ans, is_numeric=True)
        if parsed:
            try:
                loose.append(float(parsed[0].replace(",", "")))
            except ValueError:
                pass
    return _median_str(loose) if loose else []


def aggregate_numeric_cluster(answers: List[str], relative_width: float) -> List[str]:
    """Median of the densest positive relative-value cluster.

    Numeric generations are frequently multimodal (a recalled value plus one
    or more class-typical round-number guesses).  A global median can land
    between modes.  This keeps the most corroborated mode and then takes its
    median.  Falls back to the legacy never-empty median when no strict values
    survive.
    """
    values: List[float] = []
    for answer in answers:
        if not is_pure_numeric_candidate(answer):
            continue
        parsed = parse_answer(answer, is_numeric=True)
        if not parsed:
            continue
        try:
            value = float(parsed[0].replace(",", ""))
        except ValueError:
            continue
        if value > 0 and math.isfinite(value):
            values.append(value)
    if not values:
        return aggregate_median(answers)

    best_cluster: List[float] = []
    best_key = None
    # Audit P1-6 (hygiene): iterate centers in sorted order so identical
    # sample MULTISETS give identical output regardless of arrival order.
    # (No production prediction changed under 10-seed reshuffles, but the
    # first-seen tie-break was order-dependent in principle.) The center value
    # itself is the final deterministic tie-break.
    for center in sorted(values):
        cluster = [v for v in values
                   if abs(v - center) / max(abs(center), 1e-12) <= relative_width]
        # First maximize support, then minimize log-distance from the center.
        # final tie-break: smaller center wins deterministically (matches the
        # historical first-seen-in-sorted-order behavior made explicit).
        key = (len(cluster), -sum(abs(math.log(v / center)) for v in cluster),
               -center)
        if best_key is None or key > best_key:
            best_key, best_cluster = key, cluster
    return _median_str(best_cluster)


def aggregate(relation: str, subject: str, raw_generations: List[str],
              response_protocol: str = "legacy-cot",
              aggregation_profile: str = "legacy") -> List[str]:
    if aggregation_profile == "relation-v1":
        # Relation-specific production candidate.  The old global "robust"
        # switch coupled changes that transfer differently: valid-denominator
        # voting is consistently positive for borders but unstable for company,
        # while city K=5 is stable and the numeric cluster is not.  Keep each
        # relation on the independently supported rule.
        extracted = [extract_answer_with_status(text, response_protocol)
                     for text in raw_generations]
        answers = [answer for answer, status in extracted
                   if status in {"valid", "explicit-none"}]
        if relation == "countryLandBordersCountry":
            result = aggregate_item_vote(
                [parse_answer_items(answer, relation, response_protocol)
                 for answer in answers], len(answers), 0.25)
        elif relation == "companyTradesAtStockExchange":
            legacy_answers = [extract_after_think(text) for text in raw_generations]
            result = aggregate_threshold_vote(
                legacy_answers, len(legacy_answers), 0.35)
        elif relation == "personHasCityOfDeath":
            result = aggregate_city_support_gate(
                answers, ROBUST_CITY_MIN_VOTES,
                response_protocol=response_protocol)
        elif relation in NUMERIC_RELATIONS:
            legacy_answers = [extract_after_think(text) for text in raw_generations]
            result = aggregate_median(legacy_answers)
        else:
            legacy_answers = [extract_after_think(text) for text in raw_generations]
            result = aggregate_threshold_vote(
                legacy_answers, len(legacy_answers),
                LIST_VOTE_THRESHOLDS.get(relation, 0.5))
        return drop_self_reference(subject, result)

    if aggregation_profile == "legacy":
        if response_protocol == "legacy-cot":
            answers = [extract_after_think(text) for text in raw_generations]
        else:
            answers = [extract_answer_with_status(text, response_protocol)[0]
                       for text in raw_generations]
        if relation in NUMERIC_RELATIONS:
            result = aggregate_median(answers)
        elif relation in SINGLE_VOTE_RELATIONS:
            result = (aggregate_single_vote(answers) if response_protocol == "legacy-cot"
                      else aggregate_single_vote_parsed(answers, response_protocol))
        else:
            threshold = LIST_VOTE_THRESHOLDS.get(relation, 0.5)
            if response_protocol == "legacy-cot":
                result = aggregate_threshold_vote(answers, len(answers), threshold)
            else:
                result = aggregate_item_vote(
                    [parse_answer_items(answer, relation, response_protocol)
                     for answer in answers], len(answers), threshold)
        return drop_self_reference(subject, result)

    extracted = [extract_answer_with_status(text, response_protocol)
                 for text in raw_generations]
    answers = [answer for answer, status in extracted
               if status in {"valid", "explicit-none"}]
    valid_denominator = len(answers)
    if relation in NUMERIC_RELATIONS:
        result = aggregate_numeric_cluster(answers, ROBUST_NUMERIC_CLUSTER_WIDTH)
    elif relation in SINGLE_VOTE_RELATIONS:
        result = aggregate_city_support_gate(
            answers, ROBUST_CITY_MIN_VOTES, response_protocol=response_protocol)
    else:
        threshold = ROBUST_LIST_VOTE_THRESHOLDS.get(relation, 0.5)
        answer_items = [parse_answer_items(answer, relation, response_protocol)
                        for answer in answers]
        result = aggregate_item_vote(answer_items, valid_denominator, threshold)
    return drop_self_reference(subject, result)


def build_max_memory(reserve_fraction: float = 0.9) -> Optional[Dict]:
    if not torch.cuda.is_available():
        return None
    max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * reserve_fraction)
                  for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "64GiB"
    return max_memory


def resolve_effective_revision(model_name: str,
                               requested: Optional[str]) -> str:
    """Return the commit hash that a run will actually use, fail-closed.

    The DEFAULT_MODEL_REVISION pin belongs to the default 9B only. For any
    other model an un-overridden default means "unpinned", which previously
    loaded latest silently while the manifest still recorded the 9B pin
    (audit P1-2: the Qwen2.5-14B artifacts falsely claim the 9B revision).
    Resolution order: explicit pin -> single local cache snapshot -> HF hub
    main; raise rather than record an unknown revision."""
    if model_name == MODEL:
        return requested or DEFAULT_MODEL_REVISION
    if requested and requested != DEFAULT_MODEL_REVISION:
        return requested
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_name:
                revs = {r.commit_hash for r in repo.revisions}
                if len(revs) == 1:
                    return next(iter(revs))
                raise RuntimeError(
                    f"{model_name}: {len(revs)} local snapshots; pass "
                    f"--model-revision explicitly: {sorted(revs)}")
    except RuntimeError:
        raise
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_name)
        return info.sha
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resolve an effective revision for {model_name}; pass "
            f"--model-revision explicitly ({exc})") from exc


def load_model_and_tokenizer(revision: Optional[str] = DEFAULT_MODEL_REVISION,
                             precision: str = "4bit",
                             model_name: str = MODEL):
    """Load any ChatML-compatible HF model into the System-1 pipeline.

    model_name defaults to the pinned Qwen3.5-9B student. The DEFAULT_MODEL
    revision pin only applies to that default; other models resolve their own
    latest revision unless one is passed explicitly. Qwen3.5 uses the
    multimodal ImageTextToText class; plain causal LMs (e.g. Qwen2.5) fall
    back to AutoModelForCausalLM."""
    if model_name != MODEL and revision == DEFAULT_MODEL_REVISION:
        revision = None  # the pin belongs to the 9B; don't apply it elsewhere
    revision_kwargs = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(model_name, **revision_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if precision not in {"4bit", "fp16"}:
        raise ValueError(f"Unknown precision {precision!r}")
    model_kwargs = dict(
        dtype=torch.float16,
        device_map="auto",
        max_memory=build_max_memory(),
    )
    if precision == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, **revision_kwargs, **model_kwargs)
    except ValueError:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **revision_kwargs, **model_kwargs)
    model.eval()
    return model, tok


# Tile the N_CONSISTENCY samples into batched generate() calls of this size.
# benchmark_self_consistency.py measured batched sampling at ~1.5x the cost of
# a single sample (not Nx), because decoding is memory-bandwidth-bound -- so a
# bigger tile is much faster wall-clock. The only limit is KV-cache memory. Now
# that the few-shot CoT is concise (~150 chars/shot, was ~2.2KB), a whole 2080
# Ti holds the 4-bit model plus a batch of 8 short sequences comfortably, so
# this is 8 (was dropped to 4 as an OOM band-aid when prompts were huge).
# One generate() call per subject: N_CONSISTENCY=10 must fit in a single tile.
# The old value (8) split every subject into a batch-8 + batch-2 call pair;
# decode is weight-read-bound, so the batch-2 call cost almost as much
# wall-clock as the batch-8 one -- ~2x per-subject cost for nothing.
MAX_TILE_SUB_BATCH = 10
AWARD_TILE_SUB_BATCH = 2


def sampled_generate(model, tok, prompt: str, max_new_tokens: int, n_samples: int,
                     temperature_profile: str = "uniform",
                     max_sub_batch: int = MAX_TILE_SUB_BATCH,
                     fixed_temperature: Optional[float] = None) -> List[str]:
    """n_samples independent stochastic completions of the SAME prompt,
    tiled in chunks of up to MAX_TILE_SUB_BATCH per generate() call."""
    results: List[str] = []
    if fixed_temperature is not None:
        temperatures = [fixed_temperature] * n_samples
    else:
        temperatures = ([TEMPERATURE] * n_samples if temperature_profile == "uniform"
                        else (MIXED_TEMPERATURES *
                              ((n_samples + len(MIXED_TEMPERATURES) - 1)
                               // len(MIXED_TEMPERATURES)))[:n_samples])
    # model.generate accepts one scalar temperature per call. Group adjacent
    # equal-temperature samples; mixed mode therefore costs four decode calls
    # for N=10 and is deliberately opt-in for a controlled quality/cost ablation.
    start = 0
    while start < n_samples:
        temperature = temperatures[start]
        same_end = start
        while same_end < n_samples and temperatures[same_end] == temperature:
            same_end += 1
        same_end = min(same_end, start + max_sub_batch)
        chunk_n = same_end - start
        tiled = [prompt] * chunk_n
        encoded = tok(tiled, return_tensors="pt", padding=True).to(model.device)
        try:
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tok.pad_token_id,
                    # Stop as soon as the model finishes its turn. <|im_end|> is the
                    # trained end-of-turn token, but a 4-bit model at temp 1.0
                    # sometimes drifts into a hallucinated next turn ("\nuser\n...")
                    # instead -- catch that too, so we don't waste the token budget.
                    stop_strings=["<|im_end|>", "\nuser\n", "\n<|im_start|>"],
                    tokenizer=tok,
                )
        except torch.OutOfMemoryError:
            del encoded
            torch.cuda.empty_cache()
            if chunk_n <= 1:
                raise
            # Retry just this group in smaller tiles. This preserves N and
            # refuses silent empty rows while adapting to prompt-length spikes.
            smaller = max(1, chunk_n // 2)
            results.extend(sampled_generate(
                model, tok, prompt, max_new_tokens, chunk_n,
                temperature_profile="uniform", max_sub_batch=smaller,
                fixed_temperature=temperature))
            start = same_end
            continue
        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        results.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
        start = same_end
    return results


def sampled_generate_prompts(model, tok, prompts: List[str], max_new_tokens: int,
                             temperature_profile: str = "uniform",
                             max_sub_batch: int = MAX_TILE_SUB_BATCH) -> List[str]:
    """Sample once from each possibly different prompt, preserving order."""
    temperatures = ([TEMPERATURE] * len(prompts) if temperature_profile == "uniform"
                    else (MIXED_TEMPERATURES *
                          ((len(prompts) + len(MIXED_TEMPERATURES) - 1)
                           // len(MIXED_TEMPERATURES)))[:len(prompts)])
    results: List[Optional[str]] = [None] * len(prompts)

    def generate_indices(indices: List[int], temperature: float, limit: int) -> None:
        start = 0
        while start < len(indices):
            chunk = indices[start:start + limit]
            encoded = tok([prompts[i] for i in chunk], return_tensors="pt",
                          padding=True).to(model.device)
            try:
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tok.pad_token_id,
                        stop_strings=["<|im_end|>", "\nuser\n", "\n<|im_start|>"],
                        tokenizer=tok,
                    )
            except torch.OutOfMemoryError:
                del encoded
                torch.cuda.empty_cache()
                if len(chunk) <= 1:
                    raise
                midpoint = len(chunk) // 2
                generate_indices(chunk[:midpoint], temperature, max(1, midpoint))
                generate_indices(chunk[midpoint:], temperature,
                                 max(1, len(chunk) - midpoint))
                start += len(chunk)
                continue
            new_tokens = generated[:, encoded["input_ids"].shape[1]:]
            decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for index, text in zip(chunk, decoded):
                results[index] = text
            start += len(chunk)

    by_temperature: Dict[float, List[int]] = defaultdict(list)
    for index, temperature in enumerate(temperatures):
        by_temperature[temperature].append(index)
    for temperature, indices in by_temperature.items():
        generate_indices(indices, temperature, max_sub_batch)
    if any(result is None for result in results):
        raise RuntimeError("varied-prompt generation failed to fill every sample")
    return [result for result in results if result is not None]


def recover_unclosed_samples(model, tok, sample_prompts: List[str], samples: List[str],
                             max_cont_tokens: int = 48,
                             max_sub_batch: int = MAX_TILE_SUB_BATCH,
                             prompt_prefix: str = "") -> List[str]:
    """Salvage samples that ran out before closing <think>: the model reasoned
    to an answer INSIDE the think and stopped without emitting </think>+answer,
    so extract_after_think() discards them (~25-34% of samples on val). Instead
    of dropping them, force-close the think and let the MODEL commit the answer
    it already reasoned to (one short greedy continuation per unclosed sample).

    This is deliberately model-driven, not regex-driven: an earlier experiment
    grabbing the last number / last clause out of the raw think measured NEGATIVE
    (the buried text is noisy). Here the model itself states its committed answer
    given the reasoning, which is the clean signal. Only meaningful for
    legacy-cot; a no-op when nothing is unclosed."""
    idx = [i for i, s in enumerate(samples)
           if "<think>" in s and "</think>" not in s]
    if not idx:
        return samples
    recovered = list(samples)
    for start in range(0, len(idx), max_sub_batch):
        chunk = idx[start:start + max_sub_batch]
        continuations = []
        for i in chunk:
            sample = samples[i]
            if prompt_prefix and sample.startswith(prompt_prefix):
                sample = sample[len(prompt_prefix):]
            continuations.append(sample_prompts[i] + sample + "</think>\n")
        cont_prompts = continuations
        encoded = tok(cont_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_cont_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                stop_strings=["<|im_end|>", "\nuser\n", "\n<|im_start|>"],
                tokenizer=tok,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        conts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for j, i in enumerate(chunk):
            recovered[i] = samples[i] + "</think>\n" + conts[j].strip()
    return recovered


def subject_seed_base(base_seed: int, idx: int, relation: str, subject: str,
                      seed_scheme: str = "legacy") -> int:
    """Root seed for one subject's shot draw and generation stream.

    legacy: base_seed + idx (global input-row index) -- preserved verbatim so
    every frozen artifact stays reproducible, but relation subsets or reordered
    inputs change every draw (FABLE_HANDOFF caveat #1: global-index seeding).
    stable-key: derived from (base_seed, relation, subject), so the draw is
    invariant to input order and subsetting -- use this for new controlled
    comparisons where arms slice or reorder the input."""
    if seed_scheme == "legacy":
        return base_seed + idx
    if seed_scheme != "stable-key":
        raise ValueError(f"Unknown seed scheme {seed_scheme!r}")
    digest = hashlib.sha256(
        f"{base_seed}|{relation}|{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def generation_seed_for(base_seed: int, idx: int, relation: str, subject: str,
                        attempt: int, seed_scheme: str = "legacy") -> int:
    """Torch seed per subject/attempt. Legacy formula preserved exactly."""
    if seed_scheme == "legacy":
        return base_seed * 1_000_003 + idx * 1009 + attempt * 104729
    root = subject_seed_base(base_seed, idx, relation, subject, seed_scheme)
    return (root * 1_000_003 + attempt * 104729) % (2**63 - 1)


def run_worker(worker_idx: int, gpu_ids: List[int], task_queue: "mp.Queue",
                result_queue: "mp.Queue", synthetic_path: str, templates_path: str,
                base_seed: int, n_shots: int, use_cot: bool, exemplars_path: Optional[str],
                instruction_overrides: Optional[Dict[str, str]] = None,
                city_tie_judge: bool = False,
                response_protocol: str = "legacy-cot",
                aggregation_profile: str = "legacy",
                shot_sampling: str = "legacy",
                subject_retries: int = 1,
                model_revision: Optional[str] = None,
                temperature_profile: str = "uniform",
                prompt_profile: str = "single",
                exclude_target_from_shots: bool = False,
                recover_unclosed_relations: Optional[set] = None,
                null_shots_per_five: Optional[Dict[str, int]] = None,
                precision: str = "4bit",
                max_tile_sub_batch: int = MAX_TILE_SUB_BATCH,
                legacy_think_prefill: bool = False,
                max_new_tokens_default: int = MAX_NEW_TOKENS,
                award_max_new_tokens: int = AWARD_MAX_NEW_TOKENS,
                seed_scheme: str = "legacy",
                model_name: str = MODEL):
    if instruction_overrides:
        # worker is a fresh process -- install the overrides in ITS module state
        INSTRUCTION_OVERRIDES.update(instruction_overrides)
    """Pull subjects off a SHARED task queue until it's drained, instead of
    processing a fixed pre-assigned chunk. This is the load-balancing fix:
    generation length varies a lot per subject (a short "None" vs an awardWonBy
    with 384 tokens x 20 samples), so static 1/N chunks left fast workers idle
    for many minutes while slow ones finished. With a shared queue every GPU
    stays busy right up to the last subject.

    Each subject is seeded by its global index (base_seed + idx), so which
    exemplars it draws is deterministic regardless of which worker happens to
    grab it -- the run stays reproducible despite the nondeterministic dispatch.

    gpu_ids is a list so a replica can shard across >1 GPU (used only when
    --num-workers < num_gpus, e.g. to fit a bigger model)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    try:
        model, tok = load_model_and_tokenizer(model_revision, precision=precision,
                                              model_name=model_name)
        templates = load_prompt_templates(templates_path)
        examples_by_relation = load_synthetic_examples(synthetic_path)
        # Curated mode: one fixed, hand-picked set per relation used for EVERY
        # subject (from curate_exemplars.py), instead of a per-subject random
        # draw. The ablation testing whether deliberate exemplar choice lifts
        # answer support (and thus recall) over the arbitrary random draw.
        curated = None
        if exemplars_path:
            with open(exemplars_path) as f:
                curated = json.load(f)
    except Exception as exc:
        # Fatal (model load) -- signal the dispatcher so it doesn't hang.
        result_queue.put((-1, None, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        return

    done = 0
    while True:
        item = task_queue.get()
        if item is None:  # sentinel: no more work
            break
        idx, row = item
        relation, subject = row["Relation"], row["SubjectEntity"]
        last_error = None
        for attempt in range(subject_retries + 1):
            try:
                # Reset RNG state per subject/attempt so dynamic worker dispatch
                # no longer determines the sampling stream. CUDA kernels can
                # still be nondeterministic, but this removes the largest source
                # of avoidable A/B noise and makes the recorded seed meaningful.
                generation_seed = generation_seed_for(
                    base_seed, idx, relation, subject, attempt, seed_scheme)
                torch.manual_seed(generation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(generation_seed)

                pool = examples_by_relation.get(relation, [])
                if exclude_target_from_shots:
                    pool = [example for example in pool
                            if example["SubjectEntity"] != subject]
                if shot_sampling == "per-sample-diverse":
                    if curated is not None:
                        raise ValueError("per-sample-diverse is incompatible with fixed --exemplars")
                    diverse_seed = (base_seed + idx * 1009 if seed_scheme == "legacy"
                                    else subject_seed_base(base_seed, idx, relation,
                                                           subject, seed_scheme) * 1009)
                    shot_sets = sample_diverse_shot_sets(
                        pool, relation, n_shots, N_CONSISTENCY, diverse_seed)
                    shots = shot_sets[0]
                elif curated is not None:
                    shots = select_curated_shots(
                        curated, relation, subject, n_shots,
                        exclude_target_from_shots)
                    shot_sets = [shots] * N_CONSISTENCY
                else:
                    rng = random.Random(subject_seed_base(
                        base_seed, idx, relation, subject, seed_scheme))
                    if shot_sampling == "subject-balanced":
                        shots = sample_subject_balanced_shots(pool, relation, n_shots, rng)
                    elif shot_sampling == "null-stratified":
                        shots = sample_null_stratified_shots(
                            pool, relation, n_shots, rng, null_shots_per_five or {})
                    else:
                        shots = rng.sample(pool, min(n_shots, len(pool))) if pool else []
                    shot_sets = [shots] * N_CONSISTENCY
                max_new_tokens = (award_max_new_tokens if relation == "awardWonBy"
                                  else max_new_tokens_default)
                max_sub_batch = (AWARD_TILE_SUB_BATCH if relation == "awardWonBy"
                                 else max_tile_sub_batch)
                if shot_sampling == "per-sample-diverse":
                    prompt_variants = (["direct"] * N_CONSISTENCY if prompt_profile == "single"
                                       else (["direct"] * ((N_CONSISTENCY + 1) // 2)
                                             + ["cloze"] * (N_CONSISTENCY // 2)))
                    prompts = [build_prompt(
                        tok, templates[relation], relation, subject, shot_sets[i],
                        use_cot=use_cot, response_protocol=response_protocol,
                        prompt_variant=prompt_variants[i],
                        legacy_think_prefill=legacy_think_prefill)
                        for i in range(N_CONSISTENCY)]
                    raw = sampled_generate_prompts(
                        model, tok, prompts, max_new_tokens,
                        temperature_profile=temperature_profile,
                        max_sub_batch=max_sub_batch)
                    sample_prompts = prompts
                elif prompt_profile == "diverse":
                    n_direct = (N_CONSISTENCY + 1) // 2
                    n_cloze = N_CONSISTENCY - n_direct
                    direct_prompt = build_prompt(
                        tok, templates[relation], relation, subject, shots,
                        use_cot=use_cot, response_protocol=response_protocol,
                        prompt_variant="direct",
                        legacy_think_prefill=legacy_think_prefill)
                    cloze_prompt = build_prompt(
                        tok, templates[relation], relation, subject, shots,
                        use_cot=use_cot, response_protocol=response_protocol,
                        prompt_variant="cloze",
                        legacy_think_prefill=legacy_think_prefill)
                    raw = sampled_generate(
                        model, tok, direct_prompt, max_new_tokens, n_direct,
                        temperature_profile=temperature_profile,
                        max_sub_batch=max_sub_batch)
                    raw += sampled_generate(
                        model, tok, cloze_prompt, max_new_tokens, n_cloze,
                        temperature_profile=temperature_profile,
                        max_sub_batch=max_sub_batch)
                    prompt_variants = ["direct"] * n_direct + ["cloze"] * n_cloze
                    sample_prompts = [direct_prompt] * n_direct + [cloze_prompt] * n_cloze
                else:
                    prompt = build_prompt(
                        tok, templates[relation], relation, subject, shots,
                        use_cot=use_cot, response_protocol=response_protocol,
                        prompt_variant="direct",
                        legacy_think_prefill=legacy_think_prefill)
                    raw = sampled_generate(
                        model, tok, prompt, max_new_tokens, N_CONSISTENCY,
                        temperature_profile=temperature_profile,
                        max_sub_batch=max_sub_batch)
                    prompt_variants = ["direct"] * N_CONSISTENCY
                    sample_prompts = [prompt] * N_CONSISTENCY
                prefill_prefix = "<think>\n" if legacy_think_prefill else ""
                if prefill_prefix:
                    raw = [prefill_prefix + sample for sample in raw]
                # Optional: rescue unclosed-<think> samples (legacy-cot only) by
                # letting the model commit the answer it already reasoned to,
                # instead of discarding ~25-34% of generations.
                if (recover_unclosed_relations and relation in recover_unclosed_relations
                        and response_protocol == "legacy-cot"):
                    raw = recover_unclosed_samples(
                        model, tok, sample_prompts, raw,
                        max_sub_batch=max_sub_batch,
                        prompt_prefix=prefill_prefix)
                objects = aggregate(
                    relation, subject, raw, response_protocol=response_protocol,
                    aggregation_profile=aggregation_profile)
                if city_tie_judge and relation == "personHasCityOfDeath":
                    if response_protocol != "legacy-cot" or aggregation_profile != "legacy":
                        raise ValueError("--city-tie-judge is only defined for the legacy pipeline")
                    tied = find_vote_tie(raw)
                    if tied:
                        question = templates[relation].format(subject_entity=subject)
                        choice = judge_tie(model, tok, question, tied)
                        objects = drop_self_reference(subject, [] if choice == "None" else [choice])
                        logger.info(f"[worker{worker_idx}] tie-judged {subject!r}: "
                                    f"{[c for c, _ in tied]} -> {choice}")
                statuses = [extract_answer_with_status(s, response_protocol)[1] for s in raw]
                result_queue.put((
                    idx,
                    {"SubjectEntity": subject, "Relation": relation, "ObjectEntities": objects},
                    {"SubjectEntity": subject, "Relation": relation,
                     "raw_samples": raw,
                     "sample_statuses": statuses,
                     "prompt_variants": prompt_variants,
                     "shot_subjects": [s["SubjectEntity"] for s in shots],
                     "shot_subjects_by_sample": [
                         [s["SubjectEntity"] for s in sample_shots]
                         for sample_shots in shot_sets],
                     "generation_seed": generation_seed},
                    None,
                ))
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                logger.error(f"[worker{worker_idx}] {relation} {subject!r} attempt "
                             f"{attempt + 1}/{subject_retries + 1} failed: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if last_error is not None:
            # Never turn an infrastructure failure into a semantic empty answer.
            result_queue.put((idx, None, None,
                              f"{type(last_error).__name__}: {last_error}\n"
                              f"subject={subject!r} relation={relation!r}"))
            return
        done += 1
        if done % 20 == 0:
            logger.info(f"[worker{worker_idx}, gpu{gpu_ids}] {done} subjects done")


def build_parser() -> argparse.ArgumentParser:
    """Extracted so production commands can be parse-validated in tests
    (audit P0-3: invalid flag combos must not live only in Markdown)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/val.jsonl")
    parser.add_argument("--output", default="data/predictions.jsonl")
    parser.add_argument("--raw-cache", default="data/raw_generations.jsonl",
                         help="Cache of the 10 raw samples per subject, so tune_thresholds.py "
                              "can sweep theta without re-running GPU inference")
    parser.add_argument("--synthetic-cot", default="data/synthetic_cot_faithful.jsonl",
                        help="Strategy-faithfulness-filtered demonstration pool used by the "
                             "best System-1 artifact.")
    parser.add_argument("--prompt-templates", default="prompt_templates/question_prompts.csv")
    parser.add_argument("--num-workers", type=int, default=None,
                         help="Default: one per GPU (a whole 4-bit replica fits on one 2080 "
                              "Ti now that the few-shot CoT is concise). Pass a smaller number "
                              "to shard each replica across several GPUs if you ever OOM.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-scheme", choices=["legacy", "stable-key"],
                        default="legacy",
                        help="legacy: subject seed = base_seed + global row index "
                             "(preserves every frozen artifact, but relation subsets "
                             "or reordered inputs change all draws). stable-key: seed "
                             "derived from (base_seed, relation, subject) -- input-order "
                             "and subset invariant; use for new controlled comparisons.")
    parser.add_argument("--n-shots", type=int, default=N_SHOTS,
                         help=f"Few-shot exemplars per prompt (default {N_SHOTS}). For factual "
                              "recall the exemplars teach format, not knowledge, so fewer may "
                              "do as well -- this arg exists to ablate that.")
    parser.add_argument("--no-cot", action="store_true",
                         help="Ablation: omit exemplar reasoning/evidence. Under legacy-cot "
                              "this strips <think>; under answer-first it leaves ANSWER only.")
    parser.add_argument("--exemplars", default=None,
                         help="Path to a curated {relation: [exemplar,...]} JSON "
                              "(from curate_exemplars.py). When set, uses that fixed hand-picked "
                              "set for every subject instead of a per-subject random draw.")
    parser.add_argument("--company-soft-abstain", action="store_true",
                         help="companyTradesAtStockExchange: replace the system instruction with "
                              "one that reserves \"None\" for concrete private/defunct evidence "
                              "(counters the over-abstention the v2 exemplars taught).")
    parser.add_argument("--capacity-recall-first", action="store_true",
                         help="hasCapacity: replace the system instruction with one that forbids "
                              "class-typical estimation (counters the upward scale-anchor bias).")
    parser.add_argument("--city-tie-judge", action="store_true",
                         help="personHasCityOfDeath: when the majority vote TIES, route the case "
                              "to a judge pass on the same model that re-reads each tied "
                              "candidate's reasoning and always picks one (instead of the "
                              "arbitrary sample-order tie-break).")
    parser.add_argument("--response-protocol", choices=["answer-first", "legacy-cot"],
                        default="legacy-cot",
                        help="Output contract. legacy-cot is the factual-recall-positive "
                             "default. answer-first is retained as a rejected/diagnostic "
                             "ablation because it improved format validity but hurt F1.")
    parser.add_argument("--aggregation-profile",
                        choices=["robust", "legacy", "relation-v1"],
                        default="robust",
                        help="robust uses validity-aware denominators, city K=5, and numeric "
                             "relative clustering. relation-v1 keeps only independently "
                             "supported per-relation changes. legacy preserves history.")
    parser.add_argument("--shot-sampling",
                        choices=["subject-balanced", "legacy", "per-sample-diverse",
                                 "null-stratified"],
                        default="legacy",
                        help="legacy is the production default. subject-balanced selects "
                             "distinct demonstration subjects and controls null/magnitude "
                             "coverage. per-sample-diverse draws a different balanced set for "
                             "each of the ten completions. null-stratified changes only the "
                             "null/non-null class count in the five-shot prompt.")
    parser.add_argument("--null-shots-per-five", default="",
                        help="For --shot-sampling null-stratified, comma-separated exact "
                             "counts such as companyTradesAtStockExchange=2,"
                             "personHasCityOfDeath=2,countryLandBordersCountry=1.")
    parser.add_argument("--subject-retries", type=int, default=1,
                        help="Retries per subject before failing the whole run (default 1).")
    parser.add_argument("--model-name", default=MODEL,
                        help="HF model id to run through the System-1 pipeline. Must be "
                             "ChatML-compatible (raw <|im_start|> prompts are built verbatim). "
                             "The DEFAULT_MODEL_REVISION pin applies only to the default 9B.")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                        help="Optional pinned Hugging Face model/tokenizer revision. Recorded "
                             "in the run manifest; strongly recommended for artifact runs.")
    parser.add_argument("--precision", choices=["4bit", "fp16"], default="4bit",
                        help="Inference weight precision. fp16 is the same 9B model sharded "
                             "across multiple GPUs; use --num-workers 2 on four 11GB cards.")
    parser.add_argument("--max-tile-sub-batch", type=int,
                        default=MAX_TILE_SUB_BATCH,
                        help="Maximum parallel self-consistency sequences per decode call. "
                             "Lower this for fp16/KV-cache memory pressure without changing N.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help="Maximum generated tokens for every non-award relation. The historical "
             f"default is {MAX_NEW_TOKENS}; record any larger truncation-ablation "
             "ceiling explicitly in the manifest.")
    parser.add_argument(
        "--award-max-new-tokens", type=int, default=AWARD_MAX_NEW_TOKENS,
        help="Maximum generated tokens for awardWonBy only (default "
             f"{AWARD_MAX_NEW_TOKENS}). Kept separate because award lists are long.")
    parser.add_argument("--legacy-think-prefill", action="store_true",
                        help="Ablation: end the raw ChatML assistant prefix with Qwen's native "
                             "<think> opening and reattach it to cached continuations. This "
                             "tests whether the high unclosed-think rate is a prompt-contract "
                             "problem without changing answer order.")
    parser.add_argument("--temperature-profile", choices=["uniform", "mixed"],
                        default="uniform",
                        help="uniform reproduces ten temperature-1.0 samples. mixed uses "
                             "3x.25/3x.55/2x.8/2x1.0 and is a slower opt-in ablation.")
    parser.add_argument("--prompt-profile", choices=["single", "diverse"],
                        default="single",
                        help="diverse splits the ten-sample budget evenly across direct and "
                             "cloze prompts. It changes framing, not model size.")
    parser.add_argument("--exclude-target-from-shots", action="store_true",
                        help="Required for train-fold evaluation: never demonstrate the "
                             "target subject's gold-derived exemplar in its own prompt.")
    parser.add_argument("--recover-unclosed", action="store_true",
                        help="Rescue unclosed-<think> samples (legacy-cot only): force-close "
                             "the think and let the model commit the answer it already reasoned "
                             "to, instead of discarding ~25-34%% of generations. An unclosed tag "
                             "may reflect a token limit or an early EOS/stop. One short greedy "
                             "continuation per unclosed sample; the recovered text is stored in "
                             "the raw cache so aggregation and status counting see it as valid.")
    parser.add_argument("--recover-unclosed-relations", default=None,
                        help="Comma-separated relation gate for unclosed-think recovery. "
                             "Prefer this over global --recover-unclosed after the paired "
                             "validation run showed recovery helps company/borders but harms "
                             "city. The two flags are mutually exclusive.")
    parser.add_argument("--manifest", default=None,
                        help="Run-manifest path (default: <raw-cache>.manifest.json).")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Debug only: write rows with missing samples or guaranteed-wrong "
                             "empty outputs instead of failing validation.")
    return parser


def main():
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    parser = build_parser()
    args = parser.parse_args()
    use_cot = not args.no_cot

    if args.city_tie_judge and (args.response_protocol != "legacy-cot"
                                or args.aggregation_profile != "legacy"):
        raise SystemExit("--city-tie-judge requires --response-protocol legacy-cot "
                         "--aggregation-profile legacy")
    if args.subject_retries < 0:
        raise SystemExit("--subject-retries must be >= 0")
    if args.max_tile_sub_batch < 1:
        raise SystemExit("--max-tile-sub-batch must be >= 1")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")
    if args.award_max_new_tokens < 1:
        raise SystemExit("--award-max-new-tokens must be >= 1")
    if args.legacy_think_prefill and args.response_protocol != "legacy-cot":
        raise SystemExit("--legacy-think-prefill requires --response-protocol legacy-cot")
    if args.prompt_profile == "diverse" and args.temperature_profile == "mixed":
        raise SystemExit("Run prompt diversity and mixed temperature as separate ablations; "
                         "they are intentionally not combined until each is validated.")
    try:
        null_shots_per_five = parse_null_shots(args.null_shots_per_five)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.shot_sampling == "null-stratified" and not null_shots_per_five:
        raise SystemExit("--shot-sampling null-stratified requires --null-shots-per-five")
    if args.shot_sampling != "null-stratified" and null_shots_per_five:
        raise SystemExit("--null-shots-per-five requires --shot-sampling null-stratified")
    if args.recover_unclosed and args.recover_unclosed_relations:
        raise SystemExit("Use either --recover-unclosed or "
                         "--recover-unclosed-relations, not both")
    if args.recover_unclosed:
        recover_unclosed_relations = set(RELATION_TYPE)
    elif args.recover_unclosed_relations:
        recover_unclosed_relations = {
            value.strip() for value in args.recover_unclosed_relations.split(",")
            if value.strip()
        }
        unknown_recovery = recover_unclosed_relations - set(RELATION_TYPE)
        if unknown_recovery:
            raise SystemExit(f"Unknown recovery relations: {sorted(unknown_recovery)}")
    else:
        recover_unclosed_relations = set()

    instruction_overrides: Dict[str, str] = {}
    if args.company_soft_abstain:
        instruction_overrides["companyTradesAtStockExchange"] = COMPANY_SOFT_ABSTAIN_INSTRUCTION
    if args.capacity_recall_first:
        instruction_overrides["hasCapacity"] = CAPACITY_RECALL_FIRST_INSTRUCTION

    requested_revision = args.model_revision
    args.model_revision = resolve_effective_revision(
        args.model_name, args.model_revision)
    if args.model_revision != requested_revision:
        logger.info(f"Resolved {args.model_name} revision: "
                    f"{requested_revision!r} -> {args.model_revision}")

    if not Path(args.synthetic_cot).exists():
        raise SystemExit(f"{args.synthetic_cot} not found -- run generate_synthetic_data.py first.")

    rows = read_jsonl_file(args.input)
    total_gpus = torch.cuda.device_count()
    if total_gpus < 1:
        raise SystemExit(
            "No CUDA GPU detected. Refusing silent CPU fallback for model inference; "
            "run from a GPU-enabled shell or scheduler allocation.")
    num_workers = args.num_workers or max(1, total_gpus)
    num_workers = max(1, min(num_workers, total_gpus or 1))

    gpus_per_worker = total_gpus // num_workers if total_gpus else 0
    gpu_groups: List[List[int]] = [
        list(range(w * gpus_per_worker, (w + 1) * gpus_per_worker)) for w in range(num_workers)
    ]
    leftover = list(range(num_workers * gpus_per_worker, total_gpus))
    if leftover:
        gpu_groups[-1].extend(leftover)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    # Seed the shared queue with every subject, then one sentinel per worker.
    for idx, row in enumerate(rows):
        task_queue.put((idx, row))
    for _ in range(num_workers):
        task_queue.put(None)

    processes = []
    for w in range(num_workers):
        p = ctx.Process(
            target=run_worker,
            args=(w, gpu_groups[w], task_queue, result_queue, args.synthetic_cot,
                  args.prompt_templates, args.seed, args.n_shots, use_cot, args.exemplars,
                  instruction_overrides, args.city_tie_judge,
                  args.response_protocol, args.aggregation_profile,
                  args.shot_sampling, args.subject_retries, args.model_revision,
                  args.temperature_profile, args.prompt_profile,
                  args.exclude_target_from_shots, recover_unclosed_relations,
                  null_shots_per_five, args.precision,
                  args.max_tile_sub_batch, args.legacy_think_prefill,
                  args.max_new_tokens, args.award_max_new_tokens,
                  args.seed_scheme, args.model_name),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {w} on GPU(s) {gpu_groups[w]}")

    final: List[Optional[Dict]] = [None] * len(rows)
    final_raw: List[Optional[Dict]] = [None] * len(rows)
    collected = 0
    while collected < len(rows):
        try:
            idx, pred, raw_cache, error = result_queue.get(timeout=900)
        except Exception:  # queue.Empty -- no result for 15 min
            if not any(p.is_alive() for p in processes):
                raise RuntimeError(
                    f"All workers died with only {collected}/{len(rows)} results collected."
                )
            continue  # workers still alive, just slow -- keep waiting
        if error is not None:  # fatal worker error (e.g. model load)
            for p in processes:
                p.terminate()
            raise RuntimeError(f"Worker failed:\n{error}")
        final[idx] = pred
        final_raw[idx] = raw_cache
        collected += 1
        if collected % 25 == 0 or collected == len(rows):
            logger.info(f"{collected}/{len(rows)} subjects done")

    for p in processes:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    validation_errors = []
    if any(row is None for row in final) or any(row is None for row in final_raw):
        validation_errors.append("one or more input rows have no result")
    else:
        expected_keys = [(r["SubjectEntity"], r["Relation"]) for r in rows]
        actual_keys = [(r["SubjectEntity"], r["Relation"]) for r in final]
        if actual_keys != expected_keys:
            validation_errors.append("prediction key/order mismatch against input")
        bad_sample_rows = [
            (r["SubjectEntity"], r["Relation"], len(r.get("raw_samples", [])))
            for r in final_raw if len(r.get("raw_samples", [])) != N_CONSISTENCY
        ]
        if bad_sample_rows:
            validation_errors.append(
                f"{len(bad_sample_rows)} rows do not contain N={N_CONSISTENCY} samples; "
                f"examples={bad_sample_rows[:3]}")
        never_empty = NUMERIC_RELATIONS | {"awardWonBy"}
        guaranteed_empty = [
            (r["SubjectEntity"], r["Relation"]) for r in final
            if r["Relation"] in never_empty and not r["ObjectEntities"]
        ]
        if guaranteed_empty:
            validation_errors.append(
                f"{len(guaranteed_empty)} guaranteed-nonempty rows predicted empty; "
                f"examples={guaranteed_empty[:3]}")
    if validation_errors and not args.allow_incomplete:
        raise RuntimeError("REFUSING to write incomplete inference artifact:\n  - "
                           + "\n  - ".join(validation_errors))
    for error in validation_errors:
        logger.warning(f"Artifact validation: {error}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Audit P2: temp+rename so a crash mid-write cannot leave a partial
    # final-path artifact that a resume check would later trust.
    with open(f"{args.output}.tmp", "w") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")
    os.replace(f"{args.output}.tmp", args.output)
    logger.info(f"Wrote {len(final)} predictions to {args.output}")

    with open(f"{args.raw_cache}.tmp", "w") as f:
        for row in final_raw:
            f.write(json.dumps(row) + "\n")
    os.replace(f"{args.raw_cache}.tmp", args.raw_cache)
    logger.info(f"Wrote raw generation cache to {args.raw_cache} (for tune_thresholds.py)")

    def sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    status_counts = Counter(
        status for row in final_raw for status in row.get("sample_statuses", []))
    package_names = ["torch", "transformers", "accelerate", "bitsandbytes",
                     "numpy", "pandas", "PyYAML"]
    package_versions = {}
    for name in package_names:
        try:
            package_versions[name] = package_version(name)
        except PackageNotFoundError:
            package_versions[name] = None
    manifest_path = args.manifest or f"{args.raw_cache}.manifest.json"
    manifest = {
        "started_utc": started_at.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "model": args.model_name,
        "requested_model_revision": requested_revision,
        "model_revision": args.model_revision,
        "precision": args.precision,
        "max_tile_sub_batch": args.max_tile_sub_batch,
        "max_new_tokens": args.max_new_tokens,
        "award_max_new_tokens": args.award_max_new_tokens,
        "python": sys.version,
        "packages": package_versions,
        "input": args.input,
        "input_sha256": sha256_file(args.input),
        "synthetic_cot": args.synthetic_cot,
        "synthetic_cot_sha256": sha256_file(args.synthetic_cot),
        "prompt_templates": args.prompt_templates,
        "prompt_templates_sha256": sha256_file(args.prompt_templates),
        "output": args.output,
        "output_sha256": sha256_file(args.output),
        "raw_cache": args.raw_cache,
        "raw_cache_sha256": sha256_file(args.raw_cache),
        "n_rows": len(rows),
        "n_completed": len(final),
        "n_consistency": N_CONSISTENCY,
        "n_shots": args.n_shots,
        "seed": args.seed,
        "seed_scheme": args.seed_scheme,
        "temperature": TEMPERATURE,
        "temperature_profile": args.temperature_profile,
        "temperature_schedule": (MIXED_TEMPERATURES if args.temperature_profile == "mixed"
                                 else [TEMPERATURE] * N_CONSISTENCY),
        "prompt_profile": args.prompt_profile,
        "legacy_think_prefill": args.legacy_think_prefill,
        "exclude_target_from_shots": args.exclude_target_from_shots,
        "recover_unclosed": bool(recover_unclosed_relations),
        "recover_unclosed_relations": sorted(recover_unclosed_relations),
        "response_protocol": args.response_protocol,
        "aggregation_profile": args.aggregation_profile,
        "shot_sampling": args.shot_sampling,
        "null_shots_per_five": null_shots_per_five,
        "use_cot_or_evidence": use_cot,
        "instruction_overrides": sorted(instruction_overrides),
        "sample_status_counts": dict(status_counts),
        "argv": sys.argv,
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{manifest_path}.tmp", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(f"{manifest_path}.tmp", manifest_path)
    logger.info(f"Wrote run manifest to {manifest_path}")


if __name__ == "__main__":
    main()
