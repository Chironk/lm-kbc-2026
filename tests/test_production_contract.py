import json
import tempfile
import types
import unittest
from pathlib import Path

from artifact_contract import (ArtifactContractError, sha256,
                               validate_system1_bundle)
from numeric_aggregation import aggregate_quantile
from run_submission import Submission


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class NumericAggregationTests(unittest.TestCase):
    def test_frozen_quantile_semantics(self):
        self.assertEqual(aggregate_quantile([str(i) for i in range(1, 11)], 0.55),
                         ["6"])

    def test_quantile_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            aggregate_quantile(["1"], 1.1)


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="artifact_contract_"))
        self.reference = self.root / "input.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        self.raw = self.root / "raw.jsonl"
        self.manifest = self.root / "manifest.json"
        write_jsonl(self.reference, [{"SubjectEntity": "Acme", "Relation":
                                      "companyTradesAtStockExchange",
                                      "ObjectEntities": []}])
        write_jsonl(self.predictions, [{"SubjectEntity": "Acme", "Relation":
                                        "companyTradesAtStockExchange",
                                        "ObjectEntities": ["NYSE"]}])
        samples = ["<think>x</think>\nNYSE"] * 10
        write_jsonl(self.raw, [{
            "SubjectEntity": "Acme",
            "Relation": "companyTradesAtStockExchange",
            "raw_samples": samples,
            "sample_statuses": ["valid"] * 10,
            "prompt_variants": ["direct"] * 10,
            "shot_subjects": ["Other"],
            "shot_subjects_by_sample": [["Other"]] * 10,
            "generation_seed": 1,
        }])
        self.expected = {"seed": 45, "model": "model", "argv": ["run.py"]}
        payload = {
            **self.expected,
            "input_sha256": sha256(self.reference),
            "output_sha256": sha256(self.predictions),
            "raw_cache_sha256": sha256(self.raw),
        }
        self.manifest.write_text(json.dumps(payload))

    def validate(self):
        return validate_system1_bundle(
            label="test", reference_path=self.reference,
            predictions_path=self.predictions, raw_path=self.raw,
            manifest_path=self.manifest, expected_manifest=self.expected)

    def test_valid_bundle_passes(self):
        self.assertEqual(self.validate()["seed"], 45)

    def test_stale_manifest_is_rejected(self):
        payload = json.loads(self.manifest.read_text())
        payload["seed"] = 44
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ArtifactContractError, "expected 45"):
            self.validate()

    def test_target_leak_in_any_sample_is_rejected(self):
        rows = [json.loads(line) for line in self.raw.read_text().splitlines()]
        rows[0]["shot_subjects_by_sample"][7] = ["Acme"]
        write_jsonl(self.raw, rows)
        payload = json.loads(self.manifest.read_text())
        payload["raw_cache_sha256"] = sha256(self.raw)
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ArtifactContractError, "sample 7"):
            self.validate()


class CompositionSmokeTests(unittest.TestCase):
    def test_v0501_composes_all_relations_without_experiment_imports(self):
        root = Path(tempfile.mkdtemp(prefix="compose_v0501_"))
        reference = root / "input.jsonl"
        rows = [
            {"SubjectEntity": "A", "Relation": "countryLandBordersCountry", "ObjectEntities": []},
            {"SubjectEntity": "B", "Relation": "companyTradesAtStockExchange", "ObjectEntities": []},
            {"SubjectEntity": "C", "Relation": "personHasCityOfDeath", "ObjectEntities": []},
            {"SubjectEntity": "D", "Relation": "hasArea", "ObjectEntities": []},
            {"SubjectEntity": "E", "Relation": "hasCapacity", "ObjectEntities": []},
            {"SubjectEntity": "F", "Relation": "awardWonBy", "ObjectEntities": []},
        ]
        write_jsonl(reference, rows)
        args = types.SimpleNamespace(policy="v0501", input=str(reference),
                                     output_dir=str(root / "out"), dry_run=True,
                                     skip_inference=False)
        sub = Submission(args)
        sub.out.mkdir()
        raw = lambda subject, relation, answer: {
            "SubjectEntity": subject, "Relation": relation,
            "raw_samples": [f"<think>x</think>\n{answer}"] * 10,
        }
        write_jsonl(sub.p("raw_borders.jsonl"),
                    [raw("A", "countryLandBordersCountry", "G")])
        write_jsonl(sub.p("raw_fp16.jsonl"), [
            raw("B", "companyTradesAtStockExchange", "NYSE"),
            raw("C", "personHasCityOfDeath", "Paris"),
            raw("E", "hasCapacity", "1000"),
        ])
        write_jsonl(sub.p("raw_area14b.jsonl"), [
            {"SubjectEntity": "D", "Relation": "hasArea",
             "raw_samples": [f"<think>x</think>\n{i}" for i in range(1, 11)]},
        ])
        write_jsonl(sub.p("pred_system2.jsonl"), [
            {"SubjectEntity": "B", "Relation": "companyTradesAtStockExchange",
             "ObjectEntities": ["NYSE"]},
            {"SubjectEntity": "C", "Relation": "personHasCityOfDeath",
             "ObjectEntities": ["Paris"]},
            {"SubjectEntity": "F", "Relation": "awardWonBy",
             "ObjectEntities": ["Winner"]},
        ])
        composed = sub.compose()
        by_relation = {row["Relation"]: row["ObjectEntities"] for row in composed}
        self.assertEqual(len(composed), 6)
        self.assertEqual(by_relation["hasArea"], ["6"])
        self.assertEqual(by_relation["awardWonBy"], ["Winner"])

    def test_verified_skip_inference_path_composes_and_records_real_manifests(self):
        root = Path(tempfile.mkdtemp(prefix="verified_v0501_"))
        reference = root / "input.jsonl"
        rows = [
            {"SubjectEntity": "A", "Relation": "countryLandBordersCountry", "ObjectEntities": []},
            {"SubjectEntity": "B", "Relation": "companyTradesAtStockExchange", "ObjectEntities": []},
            {"SubjectEntity": "C", "Relation": "personHasCityOfDeath", "ObjectEntities": []},
            {"SubjectEntity": "D", "Relation": "hasArea", "ObjectEntities": []},
            {"SubjectEntity": "E", "Relation": "hasCapacity", "ObjectEntities": []},
            {"SubjectEntity": "F", "Relation": "awardWonBy", "ObjectEntities": []},
        ]
        write_jsonl(reference, rows)
        args = types.SimpleNamespace(policy="v0501", input=str(reference),
                                     output_dir=str(root / "out"), dry_run=False,
                                     skip_inference=True)
        sub = Submission(args)
        sub.out.mkdir()
        sub.area_revision = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
        sub.split_inputs()
        answer = {
            "countryLandBordersCountry": ["G"],
            "companyTradesAtStockExchange": ["NYSE"],
            "personHasCityOfDeath": ["Paris"],
            "hasArea": ["6"], "hasCapacity": ["1000"],
            "awardWonBy": ["Winner"],
        }
        for name, cmd in sub.commands().items():
            paths = sub._bundle_paths(name)
            input_rows = [json.loads(line) for line in paths["input"].read_text().splitlines()]
            predictions = [{"SubjectEntity": row["SubjectEntity"],
                            "Relation": row["Relation"],
                            "ObjectEntities": answer[row["Relation"]]}
                           for row in input_rows]
            write_jsonl(paths["predictions"], predictions)
            if name == "system2":
                raw_rows = [{"SubjectEntity": row["SubjectEntity"],
                             "Relation": row["Relation"], "strategy": "test"}
                            for row in input_rows]
            else:
                raw_rows = []
                for row in input_rows:
                    text = answer[row["Relation"]][0]
                    raw_rows.append({
                        "SubjectEntity": row["SubjectEntity"],
                        "Relation": row["Relation"],
                        "raw_samples": [f"<think>x</think>\n{text}"] * 10,
                        "sample_statuses": ["valid"] * 10,
                        "prompt_variants": ["direct"] * 10,
                        "shot_subjects": ["Other"],
                        "shot_subjects_by_sample": [["Other"]] * 10,
                        "generation_seed": 1,
                    })
            write_jsonl(paths["raw"], raw_rows)
            manifest = sub._expected_manifest(name, cmd)
            manifest.update({
                "input_sha256": sha256(paths["input"]),
                "output_sha256": sha256(paths["predictions"]),
                "raw_cache_sha256": sha256(paths["raw"]),
            })
            if name == "system2":
                manifest["config_sha256"] = sha256(Path(manifest["config"]))
            paths["manifest"].write_text(json.dumps(manifest))

        source_manifests = sub.validate_all_bundles()
        composed = sub.compose()
        sub.validate(composed)
        final = sub.finalize(composed, source_manifests)
        self.assertTrue(final.is_file())
        final_manifest = json.loads(sub.p("MANIFEST.json").read_text())
        self.assertEqual(set(final_manifest["source_manifests"]),
                         {"borders", "fp16", "system2", "area14b"})
        self.assertEqual(len(read_jsonl(final)), 6)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    unittest.main()
