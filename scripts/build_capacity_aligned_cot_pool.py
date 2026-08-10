#!/usr/bin/env python3
"""Surgically align legacy capacity CoTs with the official train contract.

This builder deliberately preserves the historical pool and its Gemini-written
rationale prose.  It copies non-capacity rows byte-for-byte at the parsed JSON
level and changes capacity rows only where required:

* render the official maximum-spectator-capacity question;
* make the displayed ``Answer`` agree with the official training label;
* replace that stale answer in the rationale when necessary; and
* normalize the capacity phrase without inventing new venue facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RELATION = "hasCapacity"
CONTRACT = "highest-published-maximum-spectator-capacity-v2"
QUESTION = (
    "What is the highest published maximum spectator capacity of {subject}, "
    "as an integer number of people?"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_integer(value: Any) -> str:
    normalized = str(value).strip().replace(",", "")
    if not normalized.isdigit() or int(normalized) <= 0:
        raise ValueError(f"invalid positive integer {value!r}")
    return normalized


def _gold_capacity(row: Mapping[str, Any]) -> str:
    objects = row.get("ObjectEntities")
    if not isinstance(objects, list) or len(objects) != 1:
        raise ValueError(f"{row.get('SubjectEntity')}: expected one capacity")
    aliases = objects[0]
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"{row.get('SubjectEntity')}: missing capacity alias")
    return _normalize_integer(aliases[0])


def _number_pattern(value: str) -> re.Pattern[str]:
    plain = _normalize_integer(value)
    comma = f"{int(plain):,}"
    alternatives = sorted({plain, comma}, key=len, reverse=True)
    return re.compile(
        r"(?<!\d)(?:" + "|".join(re.escape(v) for v in alternatives) + r")(?!\d)")


def _align_reasoning(reasoning: str, old_answer: str, gold: str) -> tuple[str, bool]:
    text = str(reasoning).strip()
    corrected_number = False
    if old_answer != gold:
        pattern = _number_pattern(old_answer)
        text, replacements = pattern.subn(f"{int(gold):,}", text)
        corrected_number = replacements > 0

    # Retain the surrounding Gemini rationale, but make its decisive factual
    # sentence use the benchmark relation rather than the ambiguous "official
    # seating/total capacity" wording.
    replacements = (
        (r"\bthe officially listed seating capacity\b",
         "the highest published maximum spectator capacity"),
        (r"\bthe official historical seating capacity\b",
         "the highest published maximum spectator capacity"),
        (r"\bthe official seating capacity\b",
         "the highest published maximum spectator capacity"),
        (r"\bits officially listed seating capacity\b",
         "its highest published maximum spectator capacity"),
        (r"\bits official historical seating capacity\b",
         "its highest published maximum spectator capacity"),
        (r"\bits official seating capacity\b",
         "its highest published maximum spectator capacity"),
        (r"\bthe official capacity\b",
         "the highest published maximum spectator capacity"),
        (r"\bits official capacity\b",
         "its highest published maximum spectator capacity"),
    )
    for pattern, replacement in replacements:
        def preserve_initial_case(match: re.Match[str], value: str = replacement) -> str:
            return value[:1].upper() + value[1:] if match.group(0)[:1].isupper() else value
        text = re.sub(
            pattern, preserve_initial_case, text, flags=re.IGNORECASE)

    # One legacy rationale spells its answer in words and a few may omit the
    # displayed number.  Append a factual sentence from the official label
    # instead of regenerating or discarding the Gemini rationale.
    numeric_tokens = {
        token.replace(",", "")
        for token in re.findall(r"(?<!\w)\d[\d,]*(?!\w)", text)
    }
    if gold not in numeric_tokens:
        text = text.rstrip()
        if text and text[-1] not in ".!?":
            text += "."
        text += (
            f" The highest published maximum spectator capacity is "
            f"{int(gold):,} people."
        )
    return text, corrected_number


def build_pool(
    train_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    train_capacity = {
        str(row["SubjectEntity"]): _gold_capacity(row)
        for row in train_rows if row.get("Relation") == RELATION
    }
    if len(train_capacity) != 100:
        raise ValueError(
            f"expected 100 unique training capacity subjects, found {len(train_capacity)}")

    output = []
    canonicalized_answers = canonicalized_reasoning_numbers = capacity_rows = 0
    capacity_subjects = set()
    for source in legacy_rows:
        if source.get("Relation") != RELATION:
            output.append(dict(source))
            continue
        capacity_rows += 1
        row = dict(source)
        subject = str(row["SubjectEntity"])
        if subject not in train_capacity:
            raise ValueError(f"capacity demonstration absent from train: {subject!r}")
        capacity_subjects.add(subject)
        gold = train_capacity[subject]
        old_answer = _normalize_integer(row.get("Answer", ""))
        old_question = str(row.get("Question", ""))
        aligned_reasoning, number_corrected = _align_reasoning(
            str(row.get("think", "")), old_answer, gold)
        canonicalized_answers += old_answer != gold
        canonicalized_reasoning_numbers += number_corrected
        row.update({
            "Question": QUESTION.format(subject=subject),
            "think": aligned_reasoning,
            "Answer": gold,
            "ObjectEntities": [[gold]],
            "question_contract": "official-v1",
            "capacity_contract": CONTRACT,
            "capacity_alignment": {
                "legacy_question": old_question,
                "legacy_answer": old_answer,
                "answer_canonicalized": old_answer != gold,
                "reasoning_number_canonicalized": number_corrected,
            },
        })
        output.append(row)

    diagnostics = {
        "capacity_rows": capacity_rows,
        "capacity_unique_subjects": len(capacity_subjects),
        "displayed_answers_canonicalized": canonicalized_answers,
        "reasoning_numbers_canonicalized": canonicalized_reasoning_numbers,
        "non_capacity_rows_copied": len(output) - capacity_rows,
    }
    return output, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.jsonl")
    parser.add_argument(
        "--legacy-pool", type=Path,
        default=ROOT / "data/synthetic_cot_faithful.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/synthetic_cot_capacity_aligned_v2.jsonl",
    )
    args = parser.parse_args()
    rows, diagnostics = build_pool(
        read_jsonl(args.train), read_jsonl(args.legacy_pool))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "schema": "capacity-aligned-cot-pool-v2",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "official_train": str(args.train.resolve()),
        "official_train_sha256": sha256(args.train),
        "legacy_pool": str(args.legacy_pool.resolve()),
        "legacy_pool_sha256": sha256(args.legacy_pool),
        "capacity_contract": CONTRACT,
        "non_capacity_rows_copied_unchanged": True,
        "rows": len(rows),
        **diagnostics,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**manifest, "manifest": str(manifest_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
