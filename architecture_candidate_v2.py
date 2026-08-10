#!/usr/bin/env python3
"""Frozen relation-specific architecture selected on held-out train folds.

This module contains no validation-dependent choices.  It composes two
precision-specific System-1 raw caches, a fresh System-2 prediction file, and
the frozen historical award route into one prediction file:

* borders: 4-bit + unclosed-think recovery + relation-v1 aggregation;
* company: fp16 + recovery, four-vote base with three-vote System-2 promotion;
* city: fp16, six-vote commitment with two-vote System-2 corroboration;
* area: fp16 strict numeric median;
* capacity: fp16 densest relative cluster, fixed width 0.30;
* award: frozen System-2 iterative prediction.

The optional switches exist only to produce declared diagnostic
counterfactuals from the same paid generations.  They must not be used to
choose a different production policy after validation is revealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from architecture_ensemble import (company_prediction, item_votes)
from evaluate import read_jsonl_file
from run_inference import (aggregate, aggregate_median,
                           aggregate_numeric_cluster, drop_self_reference,
                           extract_after_think)


CITY_MIN_VOTES = 6
CITY_SYSTEM2_CORROBORATION_VOTES = 2
CAPACITY_CLUSTER_WIDTH = 0.30


def key(row: Mapping) -> Tuple[str, str]:
    return row["SubjectEntity"], row["Relation"]


def city_prediction(subject: str, raw_samples: List[str],
                    system2_items: List[str], corroborate: bool = True,
                    min_votes: int = CITY_MIN_VOTES) -> List[str]:
    """Commit at K votes, or promote only an independently corroborated top city."""
    counts, display = item_votes(raw_samples)
    if counts:
        top, votes = counts.most_common(1)[0]
        if votes >= min_votes:
            return drop_self_reference(subject, [display[top]])
    if corroborate and system2_items and counts:
        from evaluate import normalize_string
        from run_inference import canonicalize

        item = system2_items[0]
        candidate = canonicalize(normalize_string(item))
        top = counts.most_common(1)[0][0]
        if (candidate == top
                and counts[candidate] >= CITY_SYSTEM2_CORROBORATION_VOTES):
            return drop_self_reference(subject, [item])
    return []


def build_rows(reference: List[Dict], border_raw: List[Dict],
               fp16_raw: List[Dict], system2_predictions: List[Dict],
               award_predictions: List[Dict], *, corroborate: bool = True,
               city_min_votes: int = CITY_MIN_VOTES,
               capacity_cluster: bool = True) -> List[Dict]:
    def indexed(rows: List[Dict], label: str) -> Dict:
        by_key: Dict = {}
        for row in rows:
            k = key(row)
            if k in by_key:
                raise ValueError(f"duplicate {label} row for {k}; refusing to "
                                 "silently overwrite (dict comprehensions used "
                                 "to keep only the last row)")
            by_key[k] = row
        return by_key

    border_by_key = indexed(border_raw, "border raw")
    fp16_by_key = indexed(fp16_raw, "fp16 raw")
    system2_by_key = indexed(system2_predictions, "system2 prediction")
    award_by_key = indexed(award_predictions, "award prediction")
    output: List[Dict] = []
    missing = []

    for row in reference:
        subject, relation = key(row)
        pair = subject, relation
        if relation == "countryLandBordersCountry":
            raw = border_by_key.get(pair)
            if raw is None:
                missing.append((pair, "4-bit border raw"))
                continue
            objects = aggregate(
                relation, subject, raw["raw_samples"],
                response_protocol="legacy-cot", aggregation_profile="relation-v1")
        elif relation == "companyTradesAtStockExchange":
            raw = fp16_by_key.get(pair)
            if raw is None:
                missing.append((pair, "fp16 raw"))
                continue
            system2 = system2_by_key.get(pair)
            if system2 is None:
                if corroborate:
                    missing.append((pair, "system2 company prediction (required "
                                          "while corroboration is enabled)"))
                    continue
                system2 = {"ObjectEntities": []}
            objects = company_prediction(
                subject, raw["raw_samples"], system2["ObjectEntities"], corroborate)
        elif relation == "personHasCityOfDeath":
            raw = fp16_by_key.get(pair)
            if raw is None:
                missing.append((pair, "fp16 raw"))
                continue
            system2 = system2_by_key.get(pair)
            if system2 is None:
                if corroborate:
                    missing.append((pair, "system2 city prediction (required "
                                          "while corroboration is enabled)"))
                    continue
                system2 = {"ObjectEntities": []}
            objects = city_prediction(
                subject, raw["raw_samples"], system2["ObjectEntities"],
                corroborate=corroborate, min_votes=city_min_votes)
        elif relation == "hasArea":
            raw = fp16_by_key.get(pair)
            if raw is None:
                missing.append((pair, "fp16 raw"))
                continue
            answers = [extract_after_think(sample) for sample in raw["raw_samples"]]
            objects = aggregate_median(answers)
        elif relation == "hasCapacity":
            raw = fp16_by_key.get(pair)
            if raw is None:
                missing.append((pair, "fp16 raw"))
                continue
            answers = [extract_after_think(sample) for sample in raw["raw_samples"]]
            objects = (aggregate_numeric_cluster(answers, CAPACITY_CLUSTER_WIDTH)
                       if capacity_cluster else aggregate_median(answers))
        elif relation == "awardWonBy":
            award = award_by_key.get(pair)
            if award is None:
                missing.append((pair, "frozen award prediction"))
                continue
            objects = award["ObjectEntities"]
        else:
            raise ValueError(f"Unknown relation {relation!r}")
        output.append({"SubjectEntity": subject, "Relation": relation,
                       "ObjectEntities": objects})

    if missing:
        raise ValueError(f"Missing {len(missing)} required rows: {missing[:5]}")
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--border-raw", required=True)
    parser.add_argument("--fp16-raw", required=True)
    parser.add_argument("--system2-predictions", required=True)
    parser.add_argument("--award-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-corroboration", action="store_true")
    parser.add_argument("--city-min-votes", type=int, default=CITY_MIN_VOTES)
    parser.add_argument("--capacity-median", action="store_true")
    args = parser.parse_args()

    inputs = {
        "reference": Path(args.reference),
        "border_raw": Path(args.border_raw),
        "fp16_raw": Path(args.fp16_raw),
        "system2_predictions": Path(args.system2_predictions),
        "award_predictions": Path(args.award_predictions),
    }
    predictions = build_rows(
        read_jsonl_file(inputs["reference"]),
        read_jsonl_file(inputs["border_raw"]),
        read_jsonl_file(inputs["fp16_raw"]),
        read_jsonl_file(inputs["system2_predictions"]),
        read_jsonl_file(inputs["award_predictions"]),
        corroborate=not args.no_corroboration,
        city_min_votes=args.city_min_votes,
        capacity_cluster=not args.capacity_median,
    )
    output = Path(args.output)
    write_jsonl(output, predictions)
    manifest = {
        "inputs": {name: {"path": str(path), "sha256": sha256(path)}
                   for name, path in inputs.items()},
        "output": str(output),
        "output_sha256": sha256(output),
        "policy": {
            "border": "4bit-recovery/relation-v1",
            "company": "fp16-recovery/k4-plus-system2-at-k3",
            "city": f"fp16/k{args.city_min_votes}-plus-system2-at-k2",
            "area": "fp16/strict-median",
            "capacity": ("fp16/relative-cluster-0.30"
                         if not args.capacity_median else "fp16/strict-median"),
            "award": "frozen-system2-iterative",
            "corroboration": not args.no_corroboration,
        },
    }
    Path(f"{output}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
