import unittest
from unittest import mock
import baseline
from report import group_results


class GroupReportTests(unittest.TestCase):
    def test_groups_use_training_activity_and_same_users(self):
        model = [{"user_id": 1, "ndcg": .4}, {"user_id": 2, "ndcg": .2}]
        baseline = [{"user_id": 2, "ndcg": .3}, {"user_id": 1, "ndcg": .1}]
        positives = {1: set(range(5)), 2: set(range(50))}
        groups = group_results(model, baseline, positives)
        self.assertEqual([group["group"] for group in groups], ["5-19", "50+"])
        self.assertEqual(groups[0]["users"], 1)
        self.assertEqual(groups[0]["popular_ndcg"], .1)
        self.assertEqual(groups[1]["mf_ndcg"], .2)

    def test_different_users_cannot_be_compared(self):
        with self.assertRaises(ValueError):
            group_results([{"user_id": 1, "ndcg": .4}], [{"user_id": 2, "ndcg": .2}], {1: {1, 2}})


class BaselineWorkflowTests(unittest.TestCase):
    def test_validation_only_does_not_evaluate_test_window(self):
        data = baseline.synthetic_ratings()
        _, valid, _, _ = baseline.temporal_split(data)
        arguments = ["baseline.py", "--demo", "--validation-only", "--out", "runs/test_validation"]
        with mock.patch("sys.argv", arguments), mock.patch("builtins.print"), \
             mock.patch.object(baseline, "save_report") as save_report, \
             mock.patch.object(baseline, "evaluate", wraps=baseline.evaluate) as evaluate:
            baseline.main()
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(evaluate.call_args.args[1], valid)
        summary = save_report.call_args_list[0].args[1]
        self.assertEqual(summary["report_split"], "validation")


if __name__ == "__main__":
    unittest.main()
