import tempfile
import types
import unittest
from pathlib import Path

from run_inference import (
    build_parser,
    generation_seed_for,
    subject_seed_base,
)
from run_submission import POLICIES, Submission, build_submission_parser


def make_submission(policy: str, seed_scheme: str = "legacy") -> Submission:
    args = types.SimpleNamespace(
        policy=policy, input="data/val.jsonl",
        output_dir=tempfile.mkdtemp(prefix=f"cmdtest_{policy}_"),
        dry_run=True, skip_inference=False, seed_scheme=seed_scheme)
    sub = Submission(args)
    # avoid network in tests: the area revision is only needed for the command
    sub.area_revision = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    return sub


class FrozenCommandParseTests(unittest.TestCase):
    """Audit P0-3: an invalid flag combination lived only in Markdown and
    aborted a real run at argparse. Every production command must parse
    against the REAL parser, in CI, for every policy."""

    def test_every_system1_command_parses(self):
        for policy in POLICIES:
            sub = make_submission(policy)
            for name, cmd in sub.commands().items():
                if name == "system2":
                    continue  # run_baseline has its own parser (below)
                argv = cmd[cmd.index("run_inference.py") + 1:]
                args = build_parser().parse_args(argv)  # raises on invalid
                # the historical failure: both recovery flags together
                self.assertFalse(
                    args.recover_unclosed and args.recover_unclosed_relations,
                    f"{policy}/{name}: mutually exclusive recovery flags")

    def test_production_invariants(self):
        for policy in POLICIES:
            sub = make_submission(policy)
            for name, cmd in sub.commands().items():
                if name == "system2":
                    continue
                argv = cmd[cmd.index("run_inference.py") + 1:]
                args = build_parser().parse_args(argv)
                self.assertEqual(args.seed, 45, f"{policy}/{name}")
                self.assertEqual(args.seed_scheme, "legacy", f"{policy}/{name}")
                self.assertTrue(args.exclude_target_from_shots, f"{policy}/{name}")
                self.assertEqual(args.aggregation_profile, "relation-v1")
                self.assertIn("synthetic_cot_faithful.jsonl", args.synthetic_cot)
                self.assertTrue(args.model_revision,
                                f"{policy}/{name}: revision must be pinned")

    def test_stable_key_commands_select_stable_seed_scheme(self):
        sub = make_submission("v0495", seed_scheme="stable-key")
        for name, cmd in sub.commands().items():
            if name == "system2":
                continue
            argv = cmd[cmd.index("run_inference.py") + 1:]
            args = build_parser().parse_args(argv)
            self.assertEqual(args.seed_scheme, "stable-key", name)

    def test_stable_key_seed_is_invariant_to_row_position(self):
        values = {
            subject_seed_base(45, idx, "hasCapacity", "Example Arena",
                              "stable-key")
            for idx in (0, 17, 999)
        }
        self.assertEqual(len(values), 1)
        generation_values = {
            generation_seed_for(45, idx, "hasCapacity", "Example Arena", 0,
                                "stable-key")
            for idx in (0, 17, 999)
        }
        self.assertEqual(len(generation_values), 1)

    def test_legacy_seed_remains_position_dependent(self):
        self.assertNotEqual(
            generation_seed_for(45, 0, "hasCapacity", "Example Arena", 0),
            generation_seed_for(45, 1, "hasCapacity", "Example Arena", 0),
        )

    def test_system2_input_includes_award(self):
        """Audit P0-4: test awards must be freshly generated."""
        sub = make_submission("v0501")
        sub.split_inputs()
        import json
        with (Path(sub.out) / "in_system2.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        relations = {r["Relation"] for r in rows}
        self.assertIn("awardWonBy", relations)
        self.assertEqual(relations, {"companyTradesAtStockExchange",
                                     "personHasCityOfDeath", "awardWonBy"})

    def test_area_arm_only_for_v0501(self):
        self.assertIn("area14b", make_submission("v0501").commands())
        self.assertNotIn("area14b", make_submission("v0495").commands())
        self.assertNotIn("area14b", make_submission("v0491").commands())

    def test_v0501_does_not_duplicate_area_in_fp16_arm(self):
        sub = make_submission("v0501")
        sub.split_inputs()
        import json
        with (Path(sub.out) / "in_fp16.jsonl").open() as handle:
            relations = {json.loads(line)["Relation"] for line in handle}
        self.assertNotIn("hasArea", relations)
        self.assertEqual(relations, {"hasCapacity", "companyTradesAtStockExchange",
                                     "personHasCityOfDeath"})

    def test_area14b_pins_resolved_revision(self):
        sub = make_submission("v0501")
        cmd = sub.commands()["area14b"]
        idx = cmd.index("--model-revision")
        self.assertEqual(cmd[idx + 1],
                         "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")

    def test_run_baseline_flags_exist(self):
        """run_baseline's parser must accept every flag the submission uses."""
        import run_baseline
        import inspect
        source = inspect.getsource(run_baseline)
        sub = make_submission("v0501")
        cmd = sub.commands()["system2"]
        for flag in ("-c", "-i", "-o", "-w", "--raw-cache", "--seed"):
            self.assertIn(flag, cmd)
            lookup = {"-c": "--config", "-i": "--input", "-o": "--output",
                      "-w": "--num-workers"}.get(flag, flag)
            self.assertIn(f'"{lookup}"', source,
                          f"run_baseline no longer defines {lookup}")

    def test_stage_cli_accepts_safe_stage_by_stage_execution(self):
        parser = build_submission_parser()
        for stage in ("all", "borders", "fp16", "system2", "area14b",
                      "compose"):
            args = parser.parse_args([
                "--policy", "v0501", "--input", "data/test.jsonl",
                "--output-dir", "/tmp/submission-stage-test",
                "--stage", stage,
            ])
            self.assertEqual(args.stage, stage)

    def test_single_stage_selects_only_that_frozen_command(self):
        sub = make_submission("v0501")
        sub.stage = "fp16"
        self.assertEqual(list(sub.selected_inference_commands()), ["fp16"])

    def test_unavailable_area_stage_fails_closed(self):
        sub = make_submission("v0495")
        sub.stage = "area14b"
        with self.assertRaises(SystemExit):
            sub.selected_inference_commands()


if __name__ == "__main__":
    unittest.main()
