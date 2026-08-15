from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from lm_kbc.core import sha256
from lm_kbc.sota_reproduction import (
    EXPECTED_LEGACY_GRAPH_SHA256,
    EXPECTED_ROWS,
    EXPECTED_SCORE,
    PIPELINE_ID,
    all_stages,
    verify_snapshot,
)


class SnapshotContractTest(unittest.TestCase):
    def test_portable_snapshot_is_complete_and_hash_valid(self):
        verified = verify_snapshot()
        self.assertEqual(verified["rows"], EXPECTED_ROWS)
        self.assertEqual(
            verified["manifest"]["pipeline_id"], PIPELINE_ID)
        self.assertEqual(
            verified["manifest"]["reported_pooled_macro_f1"],
            EXPECTED_SCORE,
        )
        for path in verified["artifacts"].values():
            self.assertTrue(Path(path).is_file())


class FullReproductionTest(unittest.TestCase):
    def test_legacy_mode_reproduces_original_prediction_bytes(self):
        verified = verify_snapshot()
        expected_hash = verified["manifest"]["artifacts"][
            "deployed_predictions"]["sha256"]
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                output_dir=directory,
                parser_mode="legacy-20260729",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                status = all_stages(args)
            self.assertEqual(status, 0)
            output = Path(directory)
            self.assertEqual(
                sha256(output / "VALIDATION_PREDICTIONS.jsonl"),
                expected_hash,
            )
            self.assertEqual(
                sha256(output / "graph/UNIFIED_VALIDATION_GRAPH.jsonl"),
                EXPECTED_LEGACY_GRAPH_SHA256,
            )
            result = json.loads((output / "REPRODUCTION.json").read_text())
            self.assertTrue(result["byte_identical_to_original"])
            self.assertEqual(result["divergent_rows"], 0)
            self.assertEqual(result["official_pooled_macro_f1"], EXPECTED_SCORE)
            self.assertFalse(result["blind_safe"])
            incumbent = json.loads(
                (output / "incumbent/RECONSTRUCTION.json").read_text())
            expected = verified["manifest"]["artifacts"]
            self.assertEqual(
                incumbent["stages"]["L1_cardinality"]["sha256"],
                expected["l1_cardinality_target"]["sha256"],
            )
            self.assertEqual(
                incumbent["stages"]["L2_numeric"]["sha256"],
                expected["l2_numeric_target"]["sha256"],
            )
            self.assertEqual(
                incumbent["stages"]["L3_route_residual"]["sha256"],
                expected["chain_incumbent"]["sha256"],
            )
            self.assertTrue(all(
                stage.get("byte_identical", True)
                for stage in incumbent["stages"].values()
            ))


if __name__ == "__main__":
    unittest.main()
