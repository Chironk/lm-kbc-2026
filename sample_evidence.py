#!/usr/bin/env python3
"""Structured, relation-aware evidence extracted from System-1 samples.

The frozen aggregators historically collapse several states to an empty list.
That behavior is preserved for frozen policies, but new selectors must be able
to distinguish an explicit semantic abstention from an infrastructure/format
failure.  This module is the single contract for that distinction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Dict, Iterable, List, Optional

from evaluate import normalize_string
from run_inference import (canonicalize, extract_answer_with_status,
                           parse_answer_items)


ENTITY = "entity"
EXPLICIT_ABSTENTION = "explicit-abstention"
INVALID = "invalid"


def _company_items_from_final_prose(text: str) -> tuple[str, ...]:
    """Extract explicitly asserted exchanges from a prose *final answer*.

    This never scans the hidden reasoning block.  It only repairs recovery
    continuations that ignored the requested answer-only format.  The bounded
    pattern requires an explicit listing/trading predicate and stops at the
    sentence boundary; it is not an open-ended entity extractor.
    """
    pattern = re.compile(
        r"\b(?:listed|traded|trades|trade)\s+(?:primarily\s+)?on\s+"
        r"(?:the\s+)?([^.;\n]+)", re.IGNORECASE)
    output = []
    for match in pattern.finditer(text):
        item = match.group(1).strip(" ,")
        item = re.split(r"\s+(?:under|with|via|as)\s+", item,
                        maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,")
        if item:
            output.append(item)
    return tuple(output)


@dataclass(frozen=True)
class SampleEvidence:
    """Semantic outcome of one raw generation."""

    kind: str
    parser_status: str
    answer: str
    items: tuple[str, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["items"] = list(self.items)
        return value


def classify_sample(raw_sample: str, relation: str,
                    response_protocol: str = "legacy-cot") -> SampleEvidence:
    """Classify one sample without treating malformed text as ``None``."""
    answer, status = extract_answer_with_status(raw_sample, response_protocol)
    if status == "explicit-none":
        return SampleEvidence(EXPLICIT_ABSTENTION, status, answer, ())
    if status != "valid":
        return SampleEvidence(INVALID, status, answer, ())
    parse_text = answer
    if response_protocol == "legacy-cot":
        # Greedy unclosed-think recovery occasionally emits one explanatory
        # sentence and then the actual bare answer on the last line.  The
        # legacy parser reads only line one, turning the sentence into a fake
        # entity.  Prefer an explicit ANSWER field; otherwise use the last line
        # only when line one is clearly prose.  This affects new evidence
        # consumers, not any frozen aggregation rule.
        explicit = re.findall(r"(?:^|\n)\s*ANSWER\s*:\s*(.+)", answer,
                              flags=re.IGNORECASE)
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if explicit:
            parse_text = explicit[-1].strip()
        elif len(lines) > 1:
            first = lines[0]
            sentence_like = (len(first) > 80 or len(first.split()) > 12
                             or ". " in first or "! " in first or "? " in first)
            if sentence_like:
                parse_text = lines[-1]
    prose_items = (_company_items_from_final_prose(parse_text)
                   if relation == "companyTradesAtStockExchange" else ())
    items = (prose_items or
             tuple(parse_answer_items(parse_text, relation, response_protocol)))
    if not items:
        # A nominally valid answer that yields no relation-compatible item is a
        # parser/format failure, not evidence that the fact is absent.
        return SampleEvidence(INVALID, "valid-but-unparseable", answer, ())
    return SampleEvidence(ENTITY, status, parse_text, items)


def classify_samples(raw_samples: Iterable[str], relation: str,
                     response_protocol: str = "legacy-cot") -> List[SampleEvidence]:
    return [classify_sample(sample, relation, response_protocol)
            for sample in raw_samples]


def validate_recorded_statuses(raw_samples: List[str], statuses: List[str],
                               response_protocol: str = "legacy-cot") -> List[str]:
    """Return mismatch descriptions for cached parser statuses."""
    errors = []
    if len(raw_samples) != len(statuses):
        return [f"status/sample length mismatch: {len(statuses)} != {len(raw_samples)}"]
    for index, (sample, recorded) in enumerate(zip(raw_samples, statuses)):
        actual = extract_answer_with_status(sample, response_protocol)[1]
        if recorded != actual:
            errors.append(
                f"sample {index} parser status mismatch: recorded={recorded!r}, "
                f"recomputed={actual!r}")
    return errors


def candidate_table(evidence: Iterable[SampleEvidence]) -> List[dict]:
    """Count each normalized entity at most once per generation.

    Rows are sorted by descending votes and then first-seen order, matching the
    deterministic convention used by the existing architecture.
    """
    counts: Counter = Counter()
    display: Dict[str, str] = {}
    first_seen: Dict[str, int] = {}
    for sample in evidence:
        if sample.kind != ENTITY:
            continue
        seen = set()
        for item in sample.items:
            key = canonicalize(normalize_string(item))
            if not key or key in seen:
                continue
            seen.add(key)
            if key not in first_seen:
                first_seen[key] = len(first_seen)
                display[key] = item
            counts[key] += 1
    rows = [{"key": key, "item": display[key], "votes": votes,
             "first_seen": first_seen[key]}
            for key, votes in counts.items()]
    rows.sort(key=lambda row: (-row["votes"], row["first_seen"]))
    return rows


def evidence_summary(raw_samples: Iterable[str], relation: str,
                     response_protocol: str = "legacy-cot",
                     max_candidates: Optional[int] = None) -> dict:
    evidence = classify_samples(raw_samples, relation, response_protocol)
    kinds = Counter(sample.kind for sample in evidence)
    parser_statuses = Counter(sample.parser_status for sample in evidence)
    candidates = candidate_table(evidence)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return {
        "n_samples": len(evidence),
        "entity_samples": kinds[ENTITY],
        "explicit_abstentions": kinds[EXPLICIT_ABSTENTION],
        "invalid_samples": kinds[INVALID],
        "parser_status_counts": dict(sorted(parser_statuses.items())),
        "candidates": candidates,
    }
