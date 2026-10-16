"""Tests of important evaluation semantics; no third-party dependencies."""
import math
import unittest
from baseline import (Rating, evaluate, paired_bootstrap, popularity_recommender,
                      ranking_metrics, temporal_split)


class ProtocolTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        result = ranking_metrics([9, 2, 3], {2, 3, 4}, 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
        self.assertAlmostEqual(result["ndcg"], expected)

    def test_timestamp_groups_never_cross_boundaries(self):
        rows = [Rating(i + 1, 1, 4, i // 2) for i in range(40)]
        train, valid, test, _ = temporal_split(rows)
        self.assertLess(max(r.timestamp for r in train), min(r.timestamp for r in valid))
        self.assertLess(max(r.timestamp for r in valid), min(r.timestamp for r in test))

    def test_cold_labels_excluded_and_seen_low_rating_masked(self):
        train = [Rating(1, 1, 5, 1), Rating(1, 2, 1, 2), Rating(2, 3, 5, 3)]
        holdout = [Rating(1, 3, 5, 4), Rating(1, 99, 5, 5), Rating(99, 3, 5, 6)]
        summary, rows = evaluate(train, holdout, popularity_recommender(train), k=10, min_positives=1)
        self.assertEqual(summary["Recall@10"], 1)
        self.assertEqual(rows[0]["relevant_count"], 1)
        self.assertEqual(summary["filter_counts_sequential"]["excluded_new_item_rows"], 1)
        self.assertEqual(summary["filter_counts_sequential"]["excluded_new_user_rows"], 1)

    def test_invalid_seen_recommendation_fails(self):
        train = [Rating(1, 1, 5, 1), Rating(2, 2, 5, 2)]
        with self.assertRaises(ValueError):
            evaluate(train, [Rating(1, 2, 5, 3)], lambda u, c, k: [1], min_positives=1)

    def test_paired_bootstrap_requires_same_users(self):
        a = [{"user_id": 1, "ndcg": .2}, {"user_id": 2, "ndcg": .4}]
        b = [{"user_id": 1, "ndcg": .3}, {"user_id": 2, "ndcg": .5}]
        result = paired_bootstrap(a, b, repeats=100)
        self.assertAlmostEqual(result["mean_difference_B_minus_A"], .1)
        with self.assertRaises(ValueError):
            paired_bootstrap(a, b[:1])


if __name__ == "__main__":
    unittest.main()
