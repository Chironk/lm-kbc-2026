import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from experiments.heterogeneous_agents import end_to_end_pipeline as e2e
from experiments.heterogeneous_agents.single_ministral_validation import (
    _assert_no_n3,
    _component_area,
    _write_deterministic_zip,
)
from experiments.heterogeneous_agents.core import ContractError


class SingleMinistralValidationTest(unittest.TestCase):
    def test_component_area_requires_unique_seven_of_ten(self):
        graph = {
            "SubjectEntity": "x",
            "Relation": "hasArea",
            "relational_graph": {"nodes": [
                {
                    "node_type": "candidate_component",
                    "representative": "100",
                    "routes": {e2e.MINISTRAL_COT40: {
                        "distinct_generation_support": 7,
                    }},
                },
                {
                    "node_type": "candidate_component",
                    "representative": "200",
                    "routes": {e2e.MINISTRAL_COT40: {
                        "distinct_generation_support": 2,
                    }},
                },
            ]},
        }
        selected, detail = _component_area(graph, ["200"])
        self.assertEqual(selected, ["100"])
        self.assertTrue(detail["applied"])

    def test_n3_provenance_fails_closed(self):
        graph = {
            "SubjectEntity": "x",
            "Relation": "hasArea",
            "proposal_routes": {e2e.MINISTRAL_N3: {}},
        }
        with self.assertRaises(ContractError):
            _assert_no_n3(graph)

    def test_zip_is_byte_deterministic(self):
        payload = b'{"SubjectEntity":"x","Relation":"hasArea","ObjectEntities":[]}\n'
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.zip"
            second = Path(directory) / "second.zip"
            _write_deterministic_zip(first, payload)
            _write_deterministic_zip(second, payload)
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).digest(),
                hashlib.sha256(second.read_bytes()).digest(),
            )
            with zipfile.ZipFile(first) as handle:
                self.assertEqual(handle.namelist(), ["predictions.jsonl"])
                self.assertEqual(handle.read("predictions.jsonl"), payload)


if __name__ == "__main__":
    unittest.main()
