#!/usr/bin/env python3
"""Deterministic relation-specific routing and cross-system corroboration.

This is intentionally a small, predeclared policy rather than an unconstrained
learned meta-model.  System 1 remains the generator for borders/company/city;
System 2 supplies numeric relations.  For company and city, System 2 may only
promote an answer already independently present in System 1's raw samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from evaluate import (RELATION_TYPE, evaluate_per_sr_pair,
                      macro_average_per_relation, normalize_string,
                      read_jsonl_file)
from models.baseline_qwen import parse_answer
from run_inference import (aggregate, canonicalize, drop_self_reference,
                           extract_after_think)


COMPANY_SYSTEM1_MIN_VOTES = 4
COMPANY_SYSTEM2_CORROBORATION_VOTES = 3
CITY_SYSTEM1_MIN_VOTES = 5
CITY_SYSTEM2_CORROBORATION_VOTES = 2
SYSTEM2_ROUTED = {"hasArea", "hasCapacity", "awardWonBy"}


def key(row: Mapping) -> Tuple[str, str]:
    return row["SubjectEntity"], row["Relation"]


def item_votes(raw_samples: Iterable[str]) -> Tuple[Counter, Dict[str, str]]:
    counts: Counter = Counter()
    display: Dict[str, str] = {}
    for sample in raw_samples:
        seen = set()
        for item in parse_answer(extract_after_think(sample), is_numeric=False):
            normalized = canonicalize(normalize_string(item))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            counts[normalized] += 1
            display.setdefault(normalized, item)
    return counts, display


def company_prediction(subject: str, raw_samples: List[str],
                       system2_items: List[str], corroborate: bool) -> List[str]:
    counts, display = item_votes(raw_samples)
    chosen = [candidate for candidate, votes in counts.items()
              if votes >= COMPANY_SYSTEM1_MIN_VOTES]
    if corroborate:
        for item in system2_items:
            candidate = canonicalize(normalize_string(item))
            if (counts[candidate] >= COMPANY_SYSTEM2_CORROBORATION_VOTES
                    and candidate not in chosen):
                chosen.append(candidate)
                display.setdefault(candidate, item)
    return drop_self_reference(subject, [display[candidate] for candidate in chosen])


def city_prediction(subject: str, raw_samples: List[str],
                    system2_items: List[str], corroborate: bool) -> List[str]:
    counts, display = item_votes(raw_samples)
    if counts:
        top, votes = counts.most_common(1)[0]
        if votes >= CITY_SYSTEM1_MIN_VOTES:
            return drop_self_reference(subject, [display[top]])
    if corroborate and system2_items and counts:
        item = system2_items[0]
        candidate = canonicalize(normalize_string(item))
        top = counts.most_common(1)[0][0]
        if (candidate == top
                and counts[candidate] >= CITY_SYSTEM2_CORROBORATION_VOTES):
            return drop_self_reference(subject, [item])
    return []


def build_rows(reference: List[Dict], system1_raw: List[Dict],
               system2_predictions: List[Dict],
               corroborate: bool = True) -> List[Dict]:
    raw_by_key = {key(row): row for row in system1_raw}
    system2_by_key = {key(row): row for row in system2_predictions}
    output = []
    missing_raw = []
    for row in reference:
        subject, relation = key(row)
        pair = (subject, relation)
        raw = raw_by_key.get(pair)
        system2 = system2_by_key.get(pair, {"ObjectEntities": []})
        if relation in SYSTEM2_ROUTED and pair in system2_by_key:
            objects = system2["ObjectEntities"]
        elif raw is None:
            missing_raw.append(pair)
            continue
        elif relation == "companyTradesAtStockExchange":
            objects = company_prediction(
                subject, raw["raw_samples"], system2["ObjectEntities"], corroborate)
        elif relation == "personHasCityOfDeath":
            objects = city_prediction(
                subject, raw["raw_samples"], system2["ObjectEntities"], corroborate)
        else:
            objects = aggregate(
                relation, subject, raw["raw_samples"],
                response_protocol="legacy-cot", aggregation_profile="relation-v1")
        output.append({"SubjectEntity": subject, "Relation": relation,
                       "ObjectEntities": objects})
    if missing_raw:
        raise ValueError(f"Missing {len(missing_raw)} System-1 raw rows: {missing_raw[:3]}")
    return output


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--system1-raw", required=True)
    parser.add_argument("--system2-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-output", default=None,
                        help="Also write the same routed architecture without corroboration.")
    parser.add_argument("--ground-truth", default="",
                        help="Optional labeled file for reporting only.")
    args = parser.parse_args()

    reference = read_jsonl_file(args.reference)
    raw = read_jsonl_file(args.system1_raw)
    system2 = read_jsonl_file(args.system2_predictions)
    candidate = build_rows(reference, raw, system2, corroborate=True)
    write_jsonl(Path(args.output), candidate)
    if args.baseline_output:
        write_jsonl(Path(args.baseline_output),
                    build_rows(reference, raw, system2, corroborate=False))

    manifest = {
        "reference": args.reference,
        "reference_sha256": sha256(args.reference),
        "system1_raw": args.system1_raw,
        "system1_raw_sha256": sha256(args.system1_raw),
        "system2_predictions": args.system2_predictions,
        "system2_predictions_sha256": sha256(args.system2_predictions),
        "output": args.output,
        "output_sha256": sha256(args.output),
        "policy": {
            "company_system1_min_votes": COMPANY_SYSTEM1_MIN_VOTES,
            "company_system2_corroboration_votes": COMPANY_SYSTEM2_CORROBORATION_VOTES,
            "city_system1_min_votes": CITY_SYSTEM1_MIN_VOTES,
            "city_system2_corroboration_votes": CITY_SYSTEM2_CORROBORATION_VOTES,
            "city_system2_must_equal_system1_top": True,
        },
    }
    Path(f"{args.output}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.ground_truth:
        scores = evaluate_per_sr_pair(
            candidate, read_jsonl_file(args.ground_truth),
            RELATION_TYPE, tolerance=0.05)
        macro = macro_average_per_relation(scores)
        for relation, values in macro.items():
            print(f"{relation}: {values['macro-f1']:.6f}")


if __name__ == "__main__":
    main()
