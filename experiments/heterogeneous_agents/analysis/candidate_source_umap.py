#!/usr/bin/env python3
"""Label-free UMAP audit of heterogeneous candidate components.

The projection is fitted exclusively from semantic embeddings of
``subject + relation gloss + candidate representative``. Gold correctness,
model provenance, split identity, and decoder selection are attached only
after the coordinates have been computed. This makes the plot a post-hoc
diagnostic rather than a learned selector or a gold-informed representation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Both caches must be writable before matplotlib/numba are imported.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lm-kbc-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/lm-kbc-numba")

from evaluate import RELATION_TYPE, normalize_string, try_parse_number
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)


RELATION_GLOSSES = {
    "awardWonBy": "award recipients",
    "companyTradesAtStockExchange": "stock exchanges where the company trades",
    "countryLandBordersCountry": "countries sharing a land border",
    "hasArea": "geographic area in square kilometers",
    "hasCapacity": "venue spectator capacity",
    "personHasCityOfDeath": "city of death",
}

ROUTE_FAMILIES = {
    "qwen": "qwen",
    "gemma": "gemma",
    "ministral": "ministral",
}

SOURCE_ORDER = ("qwen_only", "gemma_only", "ministral_only", "cross_model")
SOURCE_LABELS = {
    "qwen_only": "Qwen only",
    "gemma_only": "Gemma only",
    "ministral_only": "Ministral only",
    "cross_model": "Cross-model",
}
SOURCE_COLORS = {
    "qwen_only": "#2F6BFF",
    "gemma_only": "#E88922",
    "ministral_only": "#8A46D5",
    "cross_model": "#138A72",
}


def _json_dump_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _text_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def route_family(route: str, metadata: Mapping[str, Any]) -> str:
    """Resolve a family across old and new graph schemas, failing closed."""
    explicit = str(metadata.get("model_family", "")).casefold()
    combined = f"{route} {explicit}"
    matches = {family for token, family in ROUTE_FAMILIES.items() if token in combined}
    if len(matches) != 1:
        raise ContractError(f"cannot resolve one model family for route {route!r}: {metadata!r}")
    return next(iter(matches))


def component_families(component: Mapping[str, Any]) -> tuple[str, ...]:
    routes = component.get("routes")
    if not isinstance(routes, Mapping) or not routes:
        raise ContractError(f"component has no route provenance: {component!r}")
    families = {
        route_family(str(route), metadata)
        for route, metadata in routes.items()
        if isinstance(metadata, Mapping)
    }
    if not families:
        raise ContractError(f"component has no resolvable family: {component!r}")
    return tuple(sorted(families))


def source_category(families: Sequence[str]) -> str:
    family_set = set(families)
    if len(family_set) > 1:
        return "cross_model"
    if family_set == {"qwen"}:
        return "qwen_only"
    if family_set == {"gemma"}:
        return "gemma_only"
    if family_set == {"ministral"}:
        return "ministral_only"
    raise ContractError(f"unsupported family set: {sorted(family_set)!r}")


def semantic_text(subject: str, relation: str, representative: str) -> str:
    try:
        gloss = RELATION_GLOSSES[relation]
    except KeyError as exc:
        raise ContractError(f"missing relation gloss for {relation!r}") from exc
    return f"subject: {subject}. relation: {gloss}. candidate: {representative}."


def _numeric_equivalent(left: str, right: str, tolerance: float = 0.05) -> bool:
    left_number = try_parse_number(left)
    right_number = try_parse_number(right)
    if left_number is None or right_number is None:
        return False
    if right_number == 0:
        return left_number == 0
    return abs(left_number - right_number) / abs(right_number) <= tolerance


def component_gold_matches(
    relation: str,
    member_items: Sequence[str],
    gold_entities: Sequence[Sequence[str]],
) -> list[int]:
    """Return official gold-entity indices matched by a component."""
    if RELATION_TYPE[relation] == "numeric":
        return [
            index for index, aliases in enumerate(gold_entities)
            if aliases and any(
                _numeric_equivalent(str(item), str(aliases[0]))
                for item in member_items
            )
        ]
    member_keys = {normalize_string(str(item)) for item in member_items}
    return [
        index for index, aliases in enumerate(gold_entities)
        if member_keys & {normalize_string(str(alias)) for alias in aliases}
    ]


def candidate_matches_gold(
    relation: str,
    member_items: Sequence[str],
    gold_entities: Sequence[Sequence[str]],
) -> bool:
    """Whether any component surface matches one official gold entity."""
    return bool(component_gold_matches(relation, member_items, gold_entities))


def component_selected(
    relation: str,
    member_items: Sequence[str],
    prediction: Sequence[str],
) -> bool:
    """Whether the frozen decoder emitted a surface represented by a component."""
    if RELATION_TYPE[relation] == "numeric":
        return any(
            _numeric_equivalent(str(predicted), str(item))
            for predicted in prediction
            for item in member_items
        )
    member_keys = {normalize_string(str(item)) for item in member_items}
    return any(normalize_string(str(predicted)) in member_keys for predicted in prediction)


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def build_candidate_records(
    split: str,
    graph_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gold = {_key(row): row["ObjectEntities"] for row in gold_rows}
    predictions = {_key(row): row["ObjectEntities"] for row in prediction_rows}
    graph_keys = [_key(row) for row in graph_rows]
    if len(set(graph_keys)) != len(graph_keys):
        raise ContractError(f"{split}: duplicate subject-relation graph rows")
    if set(graph_keys) != set(gold) or set(graph_keys) != set(predictions):
        raise ContractError(
            f"{split}: graph/gold/prediction key mismatch: "
            f"graph={len(set(graph_keys))} gold={len(gold)} predictions={len(predictions)}"
        )

    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(graph_rows):
        subject, relation = _key(row)
        relational = row.get("relational_graph")
        if not isinstance(relational, Mapping):
            raise ContractError(f"{split} {subject!r}/{relation}: missing relational_graph")
        components = relational.get("components")
        if not isinstance(components, list):
            raise ContractError(f"{split} {subject!r}/{relation}: missing components")
        for component in components:
            member_items = [str(value) for value in component.get("member_items", [])]
            representative = str(component.get("representative", "")).strip()
            if not member_items or not representative:
                raise ContractError(f"{split} {subject!r}/{relation}: malformed component")
            families = component_families(component)
            gold_matches = component_gold_matches(
                relation, member_items, gold[(subject, relation)])
            records.append({
                "candidate_id": f"{split}:{row.get('input_index', row_index)}:{component['id']}",
                "split": split,
                "subject": subject,
                "relation": relation,
                "component_id": str(component["id"]),
                "representative": representative,
                "member_items": member_items,
                "families": list(families),
                "routes": sorted(str(route) for route in component["routes"]),
                "source_category": source_category(families),
                "contains_ministral": "ministral" in families,
                "route_count": len(component["routes"]),
                "alias_collapsed": bool(component.get("alias_collapsed", False)),
                "semantic_text": semantic_text(subject, relation, representative),
                "gold_entity_indices": gold_matches,
                "correct": bool(gold_matches),
                "selected_by_decoder": component_selected(
                    relation, member_items, predictions[(subject, relation)]),
            })
    return records


def _source_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in SOURCE_ORDER:
        subset = [row for row in records if row["source_category"] == source]
        correct = sum(bool(row["correct"]) for row in subset)
        selected = sum(bool(row["selected_by_decoder"]) for row in subset)
        selected_correct = sum(
            bool(row["correct"]) and bool(row["selected_by_decoder"])
            for row in subset
        )
        result[source] = {
            "label": SOURCE_LABELS[source],
            "candidates": len(subset),
            "correct_candidates": correct,
            "candidate_correctness_rate": correct / len(subset) if subset else None,
            "selected_candidates": selected,
            "selected_correct_candidates": selected_correct,
            "selected_precision": selected_correct / selected if selected else None,
            "correct_candidate_capture_rate": selected_correct / correct if correct else None,
        }
    return result


def _relation_source_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for relation in RELATION_GLOSSES:
        output[relation] = _source_statistics(
            [row for row in records if row["relation"] == relation]
        )
    return output


def _gold_object_supply(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Attribute each covered gold object to the families that supplied it."""
    supply: dict[tuple[str, str, str, int], dict[str, Any]] = defaultdict(
        lambda: {"families": set(), "selected": False})
    for row in records:
        for gold_index in row["gold_entity_indices"]:
            key = (row["split"], row["subject"], row["relation"], gold_index)
            supply[key]["families"].update(row["families"])
            supply[key]["selected"] = (
                supply[key]["selected"] or bool(row["selected_by_decoder"])
            )

    overall: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    for key, value in supply.items():
        category = source_category(tuple(value["families"]))
        overall[category] += 1
        by_split[key[0]][category] += 1
        by_relation[key[2]][category] += 1
        if value["selected"]:
            selected[category] += 1
    return {
        "covered_gold_objects": sum(overall.values()),
        "by_supply": {source: overall[source] for source in SOURCE_ORDER},
        "selected_by_supply": {source: selected[source] for source in SOURCE_ORDER},
        "by_split": {
            split: {source: values[source] for source in SOURCE_ORDER}
            for split, values in sorted(by_split.items())
        },
        "by_relation": {
            relation: {source: values[source] for source in SOURCE_ORDER}
            for relation, values in sorted(by_relation.items())
        },
    }


def _embedding_neighborhood_diagnostics(
    embeddings: Any,
    records: Sequence[Mapping[str, Any]],
    k: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if len(records) <= k:
        raise ContractError(f"need more than k={k} records for neighborhood audit")
    sources = np.asarray([row["source_category"] for row in records], dtype=object)
    relations = np.asarray([row["relation"] for row in records], dtype=object)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=1)
    neighbors.fit(embeddings)
    distances, indices = neighbors.kneighbors(embeddings)
    distances, indices = distances[:, 1:], indices[:, 1:]

    same_source = sources[indices] == sources[:, None]
    same_relation = relations[indices] == relations[:, None]
    relation_neighbor_counts = same_relation.sum(axis=1)
    relation_conditioned_same = (same_source & same_relation).sum(axis=1)
    conditioned = np.divide(
        relation_conditioned_same,
        relation_neighbor_counts,
        out=np.full(len(records), np.nan),
        where=relation_neighbor_counts > 0,
    )

    by_source: dict[str, Any] = {}
    for source in SOURCE_ORDER:
        mask = sources == source
        by_source[source] = {
            "source_frequency": float(mask.mean()),
            "mean_same_source_fraction": float(same_source[mask].mean()) if mask.any() else None,
            "same_source_excess_over_frequency": (
                float(same_source[mask].mean() - mask.mean()) if mask.any() else None
            ),
            "mean_relation_conditioned_same_source_fraction": float(np.nanmean(conditioned[mask])) if mask.any() else None,
            "mean_neighbor_cosine_distance": float(distances[mask].mean()) if mask.any() else None,
        }

    counts = Counter(sources.tolist())
    total = len(records)
    permutation_baseline = sum((count / total) ** 2 for count in counts.values())
    return {
        "k": k,
        "mean_same_source_fraction": float(same_source.mean()),
        "source_frequency_permutation_baseline": permutation_baseline,
        "mean_relation_conditioned_same_source_fraction": float(np.nanmean(conditioned)),
        "by_source": by_source,
    }


def _plot(
    coordinates: Any,
    records: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Mapping[str, Any]],
    output_png: Path,
    output_pdf: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 9,
    })
    sources = np.asarray([row["source_category"] for row in records], dtype=object)
    correct = np.asarray([row["correct"] for row in records], dtype=bool)
    selected = np.asarray([row["selected_by_decoder"] for row in records], dtype=bool)

    figure = plt.figure(figsize=(13.2, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.45, 1.45, 1.0))
    axis = figure.add_subplot(grid[:, :2])
    correctness_axis = figure.add_subplot(grid[0, 2])
    capture_axis = figure.add_subplot(grid[1, 2])

    # Rejected/incorrect candidates form the quiet background; correct stars and
    # selected rings are added afterward so neither status changes coordinates.
    for source in SOURCE_ORDER:
        mask = (sources == source) & ~correct
        axis.scatter(
            coordinates[mask, 0], coordinates[mask, 1],
            s=7, marker="o", c=SOURCE_COLORS[source], alpha=0.24,
            linewidths=0, rasterized=True,
        )
    for source in SOURCE_ORDER:
        mask = (sources == source) & correct
        axis.scatter(
            coordinates[mask, 0], coordinates[mask, 1],
            s=31, marker="*", c=SOURCE_COLORS[source], alpha=0.92,
            linewidths=0.25, edgecolors="white", rasterized=True,
        )
    axis.scatter(
        coordinates[selected, 0], coordinates[selected, 1],
        s=28, marker="o", facecolors="none", edgecolors="#111111",
        linewidths=0.65, alpha=0.88, rasterized=True,
    )
    axis.set_title("Candidate components in label-free semantic space", loc="left", fontweight="bold")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#EAEAEA", linewidth=0.5, alpha=0.65)

    source_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SOURCE_COLORS[source],
               markeredgecolor="none", markersize=7, label=SOURCE_LABELS[source])
        for source in SOURCE_ORDER
    ]
    state_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#555555",
               markeredgecolor="none", markersize=9, label="Correct (post-hoc)"),
        Line2D([0], [0], marker="o", color="#111111", markerfacecolor="none",
               markeredgewidth=0.8, markersize=7, label="Selected by decoder"),
    ]
    first_legend = axis.legend(
        handles=source_handles, title="Candidate provenance", loc="upper left",
        frameon=True, framealpha=0.95, ncol=2,
    )
    axis.add_artist(first_legend)
    axis.legend(handles=state_handles, loc="lower left", frameon=True, framealpha=0.95)

    y = np.arange(len(SOURCE_ORDER))
    labels = [SOURCE_LABELS[source] for source in SOURCE_ORDER]
    candidate_rates = [statistics[source]["candidate_correctness_rate"] or 0.0 for source in SOURCE_ORDER]
    capture_rates = [statistics[source]["correct_candidate_capture_rate"] or 0.0 for source in SOURCE_ORDER]
    colors = [SOURCE_COLORS[source] for source in SOURCE_ORDER]

    correctness_axis.barh(y, candidate_rates, color=colors, alpha=0.9)
    correctness_axis.set_yticks(y, labels)
    correctness_axis.invert_yaxis()
    correctness_axis.set_xlim(0, max(0.01, max(candidate_rates) * 1.18))
    correctness_axis.set_title("Correct candidates / candidates", loc="left", fontweight="bold")
    correctness_axis.set_xlabel("Post-hoc rate")

    capture_axis.barh(y, capture_rates, color=colors, alpha=0.9)
    capture_axis.set_yticks(y, labels)
    capture_axis.invert_yaxis()
    capture_axis.set_xlim(0, max(0.01, max(capture_rates) * 1.18))
    capture_axis.set_title("Correct candidates selected", loc="left", fontweight="bold")
    capture_axis.set_xlabel("Decoder capture rate")

    for bar_axis, values in ((correctness_axis, candidate_rates), (capture_axis, capture_rates)):
        bar_axis.spines[["top", "right", "left"]].set_visible(False)
        bar_axis.grid(axis="x", color="#E4E4E4", linewidth=0.6)
        bar_axis.set_axisbelow(True)
        for index, value in enumerate(values):
            bar_axis.text(value, index, f" {value:.1%}", va="center", ha="left", fontsize=8)

    figure.suptitle(
        "Heterogeneous candidate supply and decoder utilization",
        fontsize=14, fontweight="bold", x=0.02, ha="left",
    )
    figure.text(
        0.02, 0.005,
        "Projection uses only subject, relation, and candidate text. Provenance, correctness, and selection are overlays.",
        fontsize=8, color="#444444",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def _write_coordinate_csv(path: Path, coordinates: Any, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "candidate_id", "split", "subject", "relation", "component_id",
            "representative", "families", "source_category", "correct",
            "selected_by_decoder", "umap_1", "umap_2",
        ))
        writer.writeheader()
        for row, coordinate in zip(records, coordinates):
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "split": row["split"],
                "subject": row["subject"],
                "relation": row["relation"],
                "component_id": row["component_id"],
                "representative": row["representative"],
                "families": "+".join(row["families"]),
                "source_category": row["source_category"],
                "correct": int(row["correct"]),
                "selected_by_decoder": int(row["selected_by_decoder"]),
                "umap_1": float(coordinate[0]),
                "umap_2": float(coordinate[1]),
            })


def _result_markdown(
    records: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    figure_name: str,
) -> str:
    lines = [
        "# Candidate-source UMAP audit",
        "",
        "This is a post-hoc development diagnostic, not a deployable selector. The",
        "semantic embeddings and UMAP coordinates use only subject text, a fixed",
        "relation gloss, and the normalized candidate representative. Gold",
        "correctness, model provenance, and decoder selection are visualization",
        "overlays added after the projection is fixed.",
        "",
        f"![Candidate-source UMAP]({figure_name})",
        "",
        "## Source diagnostics",
        "",
        "| source | candidates | correct | correctness | selected | selected precision | correct capture |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCE_ORDER:
        value = diagnostics["source_statistics"][source]
        selected_precision = value["selected_precision"]
        capture = value["correct_candidate_capture_rate"]
        lines.append(
            f"| {value['label']} | {value['candidates']} | {value['correct_candidates']} | "
            f"{value['candidate_correctness_rate']:.3f} | {value['selected_candidates']} | "
            f"{selected_precision:.3f} | {capture:.3f} |"
        )
    lines.extend([
        "",
        "A correct component is counted when any of its alias/numeric member surfaces",
        "matches an official gold entity under the official normalization or 5% numeric",
        "tolerance. A selected component is one represented in the frozen decoder output.",
        "Empty-answer events are not candidate components and therefore are not dots.",
        "",
        "## Interpretation guardrails",
        "",
        "UMAP preserves local neighborhoods imperfectly and its axes have no semantic",
        "meaning. Apparent clusters are descriptive rather than causal. The original",
        "384-dimensional embedding-space neighborhood diagnostics are stored in",
        "`DIAGNOSTICS.json`; source labels never enter either representation.",
        "",
        f"Total components: **{len(records)}**.",
    ])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import sklearn
    import sentence_transformers
    import umap
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import trustworthiness

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "train_graph": args.train_graph.resolve(),
        "train_gold": args.train_gold.resolve(),
        "train_predictions": args.train_predictions.resolve(),
        "dev_graph": args.dev_graph.resolve(),
        "dev_gold": args.dev_gold.resolve(),
        "dev_predictions": args.dev_predictions.resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise ContractError(f"missing {name}: {path}")

    records = build_candidate_records(
        "train", read_jsonl(inputs["train_graph"]), read_jsonl(inputs["train_gold"]),
        read_jsonl(inputs["train_predictions"]),
    ) + build_candidate_records(
        "dev", read_jsonl(inputs["dev_graph"]), read_jsonl(inputs["dev_gold"]),
        read_jsonl(inputs["dev_predictions"]),
    )
    texts = [row["semantic_text"] for row in records]
    texts_sha256 = _text_sha256(texts)

    model = SentenceTransformer(
        args.embedding_model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    if embeddings.shape[0] != len(records):
        raise ContractError("embedding row count mismatch")
    np.save(output_dir / "EMBEDDINGS.npy", embeddings, allow_pickle=False)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
        transform_seed=args.seed,
        n_jobs=1,
    )
    coordinates = reducer.fit_transform(embeddings).astype("float32")
    np.save(output_dir / "UMAP_COORDINATES.npy", coordinates, allow_pickle=False)

    source_statistics = _source_statistics(records)
    rng = np.random.default_rng(args.seed)
    sample_n = min(args.trustworthiness_sample, len(records))
    sample_indices = np.sort(rng.choice(len(records), size=sample_n, replace=False))
    diagnostics = {
        "schema": "candidate-source-umap-diagnostics-v1",
        "development_only": True,
        "gold_aware_overlays": True,
        "labels_used_in_embeddings": False,
        "labels_used_in_projection": False,
        "projection_features": ["subject", "fixed_relation_gloss", "candidate_representative"],
        "components": len(records),
        "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "relation_counts": dict(sorted(Counter(row["relation"] for row in records).items())),
        "route_component_counts_by_split": {
            split: dict(sorted(Counter(
                route
                for row in records if row["split"] == split
                for route in row["routes"]
            ).items()))
            for split in ("train", "dev")
        },
        "source_statistics": source_statistics,
        "split_source_statistics": {
            split: _source_statistics([row for row in records if row["split"] == split])
            for split in ("train", "dev")
        },
        "relation_source_statistics": _relation_source_statistics(records),
        "gold_object_supply": _gold_object_supply(records),
        "embedding_neighborhoods": _embedding_neighborhood_diagnostics(
            embeddings, records, args.neighbor_audit_k,
        ),
        "umap_trustworthiness": {
            "sample_n": sample_n,
            "n_neighbors": args.neighbor_audit_k,
            "value": float(trustworthiness(
                embeddings[sample_indices], coordinates[sample_indices],
                n_neighbors=args.neighbor_audit_k, metric="cosine",
            )),
        },
    }
    _json_dump_atomic(output_dir / "DIAGNOSTICS.json", diagnostics)
    write_jsonl_atomic(output_dir / "CANDIDATE_METADATA.jsonl", records)
    _write_coordinate_csv(output_dir / "UMAP_COORDINATES.csv", coordinates, records)
    _plot(
        coordinates, records, source_statistics,
        output_dir / "candidate_source_umap.png",
        output_dir / "candidate_source_umap.pdf",
    )
    (output_dir / "RESULT.md").write_text(
        _result_markdown(records, diagnostics, "candidate_source_umap.png")
    )

    manifest = {
        "schema": "candidate-source-umap-manifest-v1",
        "development_only": True,
        "gold_aware_overlays": True,
        "labels_used_in_embeddings": False,
        "labels_used_in_projection": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "embedding_model": args.embedding_model,
        "embedding_dimension": int(embeddings.shape[1]),
        "semantic_texts_sha256": texts_sha256,
        "umap": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "seed": args.seed,
        },
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "umap_learn": umap.__version__,
        },
        "outputs": {
            name: {"path": str(output_dir / name), "sha256": sha256(output_dir / name)}
            for name in (
                "CANDIDATE_METADATA.jsonl", "DIAGNOSTICS.json", "EMBEDDINGS.npy",
                "RESULT.md", "UMAP_COORDINATES.csv", "UMAP_COORDINATES.npy",
                "candidate_source_umap.pdf", "candidate_source_umap.png",
            )
        },
    }
    _json_dump_atomic(output_dir / "MANIFEST.json", manifest)
    print(json.dumps({
        "components": len(records),
        "figure": str(output_dir / "candidate_source_umap.png"),
        "result": str(output_dir / "RESULT.md"),
        "source_statistics": source_statistics,
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    experiments = root / "experiments" / "heterogeneous_agents" / "runs"
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--train-graph", type=Path, default=(
        experiments / "minimal_structural_commitment_dev_20260804_v1" /
        "base/train/graph/FINAL_EXACT_EVIDENCE_GRAPH.jsonl"
    ))
    value.add_argument("--train-gold", type=Path, default=root / "data/train.jsonl")
    value.add_argument("--train-predictions", type=Path, default=(
        experiments / "minimal_structural_commitment_dev_20260804_v1" /
        "base/train/FINAL_PREDICTIONS.jsonl"
    ))
    value.add_argument("--dev-graph", type=Path, default=(
        experiments / "cot40_cardinality_validation_confirmation_20260730_v1" /
        "graph/VALIDATION_GRAPH.jsonl"
    ))
    value.add_argument("--dev-gold", type=Path, default=root / "data/val.jsonl")
    value.add_argument("--dev-predictions", type=Path, default=(
        root / "results/heterogeneous/candidates/frozen_20260803" /
        "strict_proof_0_520729_validation.jsonl"
    ))
    value.add_argument("--output-dir", type=Path, default=(
        root / "results/heterogeneous/analysis/candidate_source_umap_20260808"
    ))
    value.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    value.add_argument(
        "--allow-model-download", dest="local_files_only", action="store_false",
        help="permit an embedding-model download instead of requiring the local cache",
    )
    value.set_defaults(local_files_only=True)
    value.add_argument("--device", default="cpu")
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--n-neighbors", type=int, default=30)
    value.add_argument("--min-dist", type=float, default=0.15)
    value.add_argument("--neighbor-audit-k", type=int, default=15)
    value.add_argument("--trustworthiness-sample", type=int, default=2000)
    value.add_argument("--seed", type=int, default=20260808)
    value.set_defaults(function=run)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.n_neighbors < 2 or args.neighbor_audit_k < 1:
        raise ContractError("neighbor counts are out of range")
    if not 0 <= args.min_dist <= 1:
        raise ContractError("min_dist must be in [0, 1]")
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
