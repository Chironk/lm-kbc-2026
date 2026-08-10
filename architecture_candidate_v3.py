#!/usr/bin/env python3
"""Structured evidence helpers for post-v0501 architecture experiments.

Nothing in this module changes the frozen v0501 composer.  It removes two
incidental implementation choices from future experiments:

* company candidates are obtained from relation-aware parsed sample evidence,
  rather than treating arbitrary final prose as an exchange name;
* System-2 city corroboration is allowed to select a co-top candidate, rather
  than whichever tied candidate happened to be generated first.
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from evaluate import normalize_string
from run_inference import canonicalize, drop_self_reference
from sample_evidence import evidence_summary


COMPANY_MIN_VOTES = 4
COMPANY_SYSTEM2_VOTES = 3
CITY_MIN_VOTES = 6
CITY_SYSTEM2_VOTES = 2


def normalized_key(item: str) -> str:
    return canonicalize(normalize_string(item))


def plausible_exchange_candidate(candidate: Mapping) -> bool:
    """Bound obviously malformed/prose candidates without naming exchanges."""
    item = str(candidate.get("item", "")).strip()
    normalized = normalize_string(item)
    words = normalized.split()
    if not normalized or len(item) > 80 or len(words) > 12:
        return False
    if normalized in {"think", "answer", "unknown", "not sure"}:
        return False
    prose_markers = (
        " shares ", " publicly traded ", " company ", " its stock ",
        " trade on the ", " trades on the ", " listed on the ",
        " i believe ", " i think ",
    )
    padded = f" {normalized} "
    return not any(marker in padded for marker in prose_markers)


def company_evidence(raw_samples: Iterable[str],
                     system2_items: Sequence[str]) -> dict:
    """Return structured company candidates and the frozen-rule analogue."""
    summary = evidence_summary(
        raw_samples, "companyTradesAtStockExchange",
        response_protocol="legacy-cot")
    candidates = [dict(row) for row in summary["candidates"]
                  if plausible_exchange_candidate(row)]
    system2_keys = {normalized_key(item) for item in system2_items}
    for candidate in candidates:
        candidate["system2_support"] = int(candidate["key"] in system2_keys)
    chosen = [candidate for candidate in candidates
              if candidate["votes"] >= COMPANY_MIN_VOTES]
    chosen_keys = {candidate["key"] for candidate in chosen}
    for candidate in candidates:
        if (candidate["key"] not in chosen_keys
                and candidate["votes"] >= COMPANY_SYSTEM2_VOTES
                and candidate["system2_support"]):
            chosen.append(candidate)
            chosen_keys.add(candidate["key"])
    return {
        **summary,
        "candidates": candidates,
        "system2_items": list(system2_items),
        "baseline_objects": [candidate["item"] for candidate in chosen],
    }


def company_prediction(subject: str, raw_samples: Iterable[str],
                       system2_items: Sequence[str]) -> List[str]:
    evidence = company_evidence(raw_samples, system2_items)
    return drop_self_reference(subject, evidence["baseline_objects"])


def city_prediction(subject: str, raw_samples: Iterable[str],
                    system2_items: Sequence[str], *, corroborate: bool = True,
                    min_votes: int = CITY_MIN_VOTES) -> List[str]:
    """Use vote confidence, with order-invariant System-2 co-top support."""
    summary = evidence_summary(
        raw_samples, "personHasCityOfDeath", response_protocol="legacy-cot")
    candidates = summary["candidates"]
    if not candidates:
        return []
    max_votes = candidates[0]["votes"]
    if max_votes >= min_votes:
        return drop_self_reference(subject, [candidates[0]["item"]])
    if corroborate and system2_items:
        system2_key = normalized_key(system2_items[0])
        match = next((candidate for candidate in candidates
                      if candidate["key"] == system2_key), None)
        if (match is not None and match["votes"] == max_votes
                and match["votes"] >= CITY_SYSTEM2_VOTES):
            return drop_self_reference(subject, [system2_items[0]])
    return []
