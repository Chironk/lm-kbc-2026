import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from experiments.heterogeneous_agents.core import ContractError
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    GEMMA,
    QWEN,
    LogisticCalibrator,
    _make_graph,
    _materialize_prompt_routes,
    _objects_utility,
    _candidate_training_weights,
    _fixed_prompt_routes,
    _select_guard_margin_one_se,
    _select_prompt_routes,
    _training_source_diagnostics,
    candidate_features,
    decode_graph,
    feature_names,
    null_feature_names,
    null_features,
    prepare_commitments,
)


def proposal(agent, subject, relation, generations):
    return {
        "agent_id": agent, "subject": subject, "relation": relation,
        "phase": "propose", "mode": "generate", "generations": generations,
    }


def commitment(selected, probabilities=None):
    return {"available": True, "selected": selected,
            "probabilities": probabilities or {selected: 1.0}}


def graph(relation="personHasCityOfDeath", q=None, g=None,
          qfinal=None, gfinal=None):
    subject = "Target"
    q = q or ["ANSWER: None", "ANSWER: None", "ANSWER: Paris"]
    g = g or ["ANSWER: Paris", "ANSWER: Paris", "ANSWER: Rome"]
    commits = {
        QWEN: {"existence": commitment("NO", {"YES": .2, "NO": .8}),
               "cardinality": commitment("ZERO", {"ZERO": .8, "ONE": .2})},
        GEMMA: {"existence": commitment("YES", {"YES": .8, "NO": .2}),
                "cardinality": commitment("ONE", {"ZERO": .2, "ONE": .8})},
    }
    return _make_graph(
        0, subject, relation,
        {QWEN: proposal(QWEN, subject, relation, q),
         GEMMA: proposal(GEMMA, subject, relation, g)},
        {QWEN: qfinal or [], GEMMA: gfinal or ["Paris"]}, commits, "shared_cot5")


class FixedCandidateModel:
    def __init__(self, qwen=.2, gemma=.8, cross=.95):
        self.qwen, self.gemma, self.cross = qwen, gemma, cross

    def predict(self, rows):
        names = feature_names()
        qi, gi, ci = map(names.index, ("qwen_source", "gemma_source", "cross_model"))
        values = []
        for row in rows:
            values.append(self.cross if row[ci] else self.gemma if row[gi] else self.qwen)
        return np.asarray(values)


class FixedNullModel:
    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        return np.asarray([self.value] * len(rows))


class GraphSchemaTests(unittest.TestCase):
    def test_graph_tracks_normalized_per_model_support_and_selection(self):
        value = graph()
        paris = next(node for node in value["candidates"] if node["item"] == "Paris")
        self.assertEqual(paris["sources"][QWEN]["support_rate"], 1 / 3)
        self.assertEqual(paris["sources"][GEMMA]["support_rate"], 2 / 3)
        self.assertTrue(paris["selected_by"][GEMMA])
        self.assertFalse(paris["selected_by"][QWEN])
        self.assertNotIn("ObjectEntities", value)

    def test_feature_schemas_are_fixed_and_finite(self):
        value = graph()
        for node in value["candidates"]:
            features = candidate_features(value, node)
            self.assertEqual(len(features), len(feature_names()))
            self.assertTrue(np.all(np.isfinite(features)))
        features = null_features(value)
        self.assertEqual(len(features), len(null_feature_names()))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_missing_commitments_are_explicit_not_synthesized(self):
        value = graph()
        value["agents"][GEMMA]["existence"] = {
            "available": False, "selected": None, "probabilities": {}}
        _, missing = divmod(null_features(value)[null_feature_names().index(
            "gemma_exist_missing")], 1)
        self.assertEqual(missing, 0.0)
        self.assertEqual(null_features(value)[null_feature_names().index(
            "gemma_exist_missing")], 1.0)


class CalibrationTests(unittest.TestCase):
    def test_logistic_calibrator_learns_simple_separation(self):
        model = LogisticCalibrator(["intercept", "signal"], l2=.1)
        model.fit([[1, 0], [1, 0], [1, 1], [1, 1]], [0, 0, 1, 1])
        low, high = model.predict([[1, 0], [1, 1]])
        self.assertLess(low, .5)
        self.assertGreater(high, .5)

    def test_bad_matrix_fails_closed(self):
        with self.assertRaises(ContractError):
            LogisticCalibrator(["intercept"]).fit([[1], [1]], [0, 2])

    def test_row_agent_balancing_prevents_qwen_candidate_multiplicity(self):
        value = graph(
            relation="companyTradesAtStockExchange",
            q=["ANSWER: A, B, C"], g=["ANSWER: D"],
            qfinal=["A"], gfinal=["D"])
        weights = _candidate_training_weights(value, "row-agent-balanced")
        by_item = {
            node["item"]: weight
            for node, weight in zip(value["candidates"], weights)
        }
        self.assertAlmostEqual(sum(by_item.values()), 1.0)
        self.assertAlmostEqual(by_item["D"], .5)
        self.assertAlmostEqual(
            by_item["A"] + by_item["B"] + by_item["C"], .5)

    def test_source_diagnostics_report_both_agents(self):
        value = graph(
            relation="companyTradesAtStockExchange",
            q=["ANSWER: NASDAQ, Wrong"], g=["ANSWER: NASDAQ"],
            qfinal=["NASDAQ"], gfinal=["NASDAQ"])
        gold = {
            ("Target", "companyTradesAtStockExchange"): {
                "SubjectEntity": "Target",
                "Relation": "companyTradesAtStockExchange",
                "ObjectEntities": [["NASDAQ"]],
            }
        }
        diagnostics = _training_source_diagnostics(
            [value], gold, "row-agent-balanced")
        self.assertEqual(diagnostics["rows"], 1)
        self.assertIn("shared", diagnostics["by_source_group"])
        self.assertIn("qwen_only", diagnostics["by_source_group"])
        for agent in (QWEN, GEMMA):
            self.assertEqual(
                diagnostics["gold_coverage_by_agent_relation"][agent][
                    "companyTradesAtStockExchange"]["micro_gold_coverage"],
                1.0)

    def test_guard_margin_uses_paired_one_standard_error_rule(self):
        baseline = {0: .40, 1: .50, 2: .60, 3: .30, 4: .45}
        scores = {
            0.0: {0: .45, 1: .65, 2: .70, 3: .50, 4: .45},
            0.1: {0: .48, 1: .59, 2: .69, 3: .39, 4: .54},
            0.2: {0: .40, 1: .50, 2: .60, 3: .30, 4: .45},
        }
        selected, diagnostic = _select_guard_margin_one_se(scores, baseline)
        # Margin .1 is slightly below the best mean but within its fold SE;
        # margin .2 has no improvement and is outside it.
        self.assertEqual(selected, .1)
        self.assertEqual(diagnostic["best_mean_margin"], 0.0)
        self.assertTrue(diagnostic["margins"]["0.1"]["within_one_se_of_best"])
        self.assertFalse(diagnostic["margins"]["0.2"]["within_one_se_of_best"])

    def test_guard_margin_requires_identical_fold_coverage(self):
        with self.assertRaisesRegex(ContractError, "cover every fold"):
            _select_guard_margin_one_se(
                {0.0: {0: .5, 1: .6}, 0.1: {0: .5}}, {0: .4, 1: .4})

    def test_prompt_route_selection_uses_supplied_training_partition(self):
        relation_answers = {
            "awardWonBy": "Winner", "companyTradesAtStockExchange": "NASDAQ",
            "countryLandBordersCountry": "France", "hasArea": "100",
            "hasCapacity": "1000", "personHasCityOfDeath": "Paris",
        }
        graphs, gold = [], {}
        for index, (relation, answer) in enumerate(relation_answers.items()):
            subject = f"Target {index}"
            variants = {}
            for policy in ("direct", "shared_cot5", "disjoint_cot5"):
                gemma_answer = answer if policy == "direct" else "Wrong"
                variants[policy] = _make_graph(
                    index, subject, relation,
                    {QWEN: proposal(QWEN, subject, relation, ["ANSWER: None"]),
                     GEMMA: proposal(GEMMA, subject, relation,
                                     [f"ANSWER: {gemma_answer}"])},
                    {QWEN: [], GEMMA: [gemma_answer]},
                    {QWEN: {"existence": commitment("YES"),
                            "cardinality": commitment("ONE")},
                     GEMMA: {"existence": commitment("YES"),
                             "cardinality": commitment("ONE")}}, policy)
            row = dict(variants["shared_cot5"])
            row["prompt_variants"] = variants
            graphs.append(row)
            gold[(subject, relation)] = {
                "SubjectEntity": subject, "Relation": relation,
                "ObjectEntities": [[answer]],
            }
        routes, _ = _select_prompt_routes(graphs, gold)
        self.assertEqual(set(routes.values()), {"direct"})
        selected = _materialize_prompt_routes(graphs, routes)
        self.assertTrue(all(row["prompt_policy"] == "direct" for row in selected))

    def test_fixed_prompt_routes_require_one_policy_per_relation(self):
        values = []
        for relation in (
                "awardWonBy", "companyTradesAtStockExchange",
                "countryLandBordersCountry", "hasArea", "hasCapacity",
                "personHasCityOfDeath"):
            row = graph(relation=relation)
            row["prompt_policy"] = "direct"
            values.append(row)
        self.assertEqual(set(_fixed_prompt_routes(values).values()), {"direct"})
        values[0]["prompt_policy"] = "invalid"
        with self.assertRaisesRegex(ContractError, "exactly one valid policy"):
            _fixed_prompt_routes(values)


class DecoderTests(unittest.TestCase):
    @staticmethod
    def scored(value, probabilities):
        return [(node, probabilities[node["item"]]) for node in value["candidates"]]

    def test_city_proposal_and_baseline_share_probability_scale(self):
        value = graph(qfinal=["Paris"])
        probabilities = {node["item"]: .2 for node in value["candidates"]}
        probabilities["Paris"] = .8
        scored = self.scored(value, probabilities)
        self.assertAlmostEqual(_objects_utility(value, ["Paris"], scored, .1), .8)
        objects, detail = decode_graph(
            value, FixedCandidateModel(qwen=.2, gemma=.8, cross=.8),
            FixedNullModel(.1), guard_margin=0.0, require_commitments=True)
        self.assertEqual(objects, ["Paris"])
        self.assertAlmostEqual(detail["proposed_utility"], detail["baseline_utility"])
        self.assertTrue(detail["used_baseline"])

    def test_numeric_proposal_and_baseline_use_same_tolerance_utility(self):
        value = graph(
            relation="hasArea",
            q=["ANSWER: 100", "ANSWER: 101", "ANSWER: 500"],
            g=["ANSWER: 99", "ANSWER: 100", "ANSWER: 1000"],
            qfinal=["101"], gfinal=["100"])
        objects, detail = decode_graph(
            value, FixedCandidateModel(qwen=.7, gemma=.7, cross=.9),
            FixedNullModel(0.0), guard_margin=0.0, require_commitments=True)
        self.assertEqual(objects, ["101"])
        self.assertAlmostEqual(detail["proposed_utility"], detail["baseline_utility"])
        self.assertTrue(detail["used_baseline"])
        self.assertGreater(
            detail["decoder_selection_utility"], detail["proposed_utility"])

    def test_single_relation_compares_candidate_against_null(self):
        value = graph(qfinal=[])
        objects, detail = decode_graph(
            value, FixedCandidateModel(), FixedNullModel(.4),
            guard_margin=0.0, require_commitments=True)
        self.assertEqual(objects, ["Paris"])
        self.assertFalse(detail["used_baseline"])
        objects, _ = decode_graph(
            value, FixedCandidateModel(), FixedNullModel(.99),
            guard_margin=0.0, require_commitments=True)
        self.assertEqual(objects, [])

    def test_gemma_only_list_candidate_can_survive(self):
        value = graph(
            relation="companyTradesAtStockExchange",
            q=["ANSWER: None"] * 3,
            g=["ANSWER: NASDAQ"] * 3,
            qfinal=[], gfinal=["NASDAQ"])
        objects, _ = decode_graph(
            value, FixedCandidateModel(qwen=.1, gemma=.95), FixedNullModel(.05),
            guard_margin=0.0, require_commitments=True)
        self.assertEqual(objects, ["NASDAQ"])

    def test_numeric_uses_tolerance_cluster_not_arithmetic_mean(self):
        value = graph(
            relation="hasArea",
            q=["ANSWER: 100", "ANSWER: 101", "ANSWER: 500"],
            g=["ANSWER: 99", "ANSWER: 100", "ANSWER: 1000"],
            qfinal=["101"], gfinal=["100"])
        objects, _ = decode_graph(
            value, FixedCandidateModel(qwen=.7, gemma=.7, cross=.9),
            FixedNullModel(0.0), guard_margin=0.0, require_commitments=True)
        self.assertLessEqual(abs(float(objects[0]) - 100) / 100, .05)

    def test_production_decode_rejects_missing_commitments(self):
        value = graph()
        value["agents"][QWEN]["existence"]["available"] = False
        with self.assertRaisesRegex(ContractError, "missing blind commitments"):
            decode_graph(value, FixedCandidateModel(), FixedNullModel(.2),
                         guard_margin=0.0, require_commitments=True)


class CommitmentPlanTests(unittest.TestCase):
    def test_commitment_plan_is_label_free_and_contains_no_proposals(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            input_path = temp / "input.jsonl"
            input_path.write_text(json.dumps({
                "SubjectEntity": "Target", "Relation": "personHasCityOfDeath",
                "ObjectEntities": [["SECRET"]],
            }) + "\n")
            # The production input validator accepts labels, but task creation
            # must never copy them into model-facing artifacts.
            args = Namespace(
                input=str(input_path),
                agents=str(root / "configs/final/portfolio_cot.json"),
                synthetic_cot=str(root / "data/synthetic_cot_faithful.jsonl"),
                output_dir=str(temp / "plan"), seed=7)
            prepare_commitments(args)
            for agent in (QWEN, GEMMA):
                rows = [json.loads(line) for line in
                        (temp / f"plan/tasks/{agent}.jsonl").read_text().splitlines()]
                self.assertEqual({row["phase"] for row in rows}, {
                    "commit_existence", "commit_cardinality"})
                self.assertNotIn("SECRET", json.dumps(rows))
                self.assertNotIn("ObjectEntities", json.dumps(rows))


if __name__ == "__main__":
    unittest.main()
