import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from baseline import synthetic_ratings, temporal_split, train_state
from inference import RecommendationService
from train_torch import Recommender


class LearningTests(unittest.TestCase):
    def test_hand_computed_user_item_scores(self):
        model = Recommender(2, 2, dim=2, architecture="mf")
        with torch.no_grad():
            model.user_embedding.weight.copy_(torch.tensor([[1., 0.], [0., 1.]]))
            model.item_embedding.weight.copy_(torch.tensor([[.8, .2], [.1, .9]]))
        scores = model(torch.tensor([0, 1]), torch.tensor([0, 1]))
        torch.testing.assert_close(scores, torch.tensor([.8, .9]))

    def test_bpr_training_learns_relative_preference(self):
        torch.manual_seed(7)
        model = Recommender(1, 2, dim=4, architecture="mf")
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        user, positive, negative = torch.tensor([0]), torch.tensor([0]), torch.tensor([1])
        before = (model(user, positive) - model(user, negative)).item()
        for _ in range(40):
            loss = F.softplus(model(user, negative) - model(user, positive)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        after = (model(user, positive) - model(user, negative)).item()
        self.assertGreater(after, before + .5)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        project = Path(__file__).resolve().parent
        self.temp = tempfile.TemporaryDirectory(prefix="_model_test_", dir=project)
        self.root = Path(self.temp.name).resolve()
        if not self.root.is_relative_to(project):
            raise RuntimeError("Test directory escaped the project")
        self.ratings = self.root / "ratings.dat"
        rows = synthetic_ratings()
        self.ratings.write_text("\n".join(f"{r.user}::{r.item}::{r.rating}::{r.timestamp}" for r in rows), encoding="ascii")
        train, _, _, cutoffs = temporal_split(rows)
        seen, _, _, catalog = train_state(train)
        self.movies = self.root / "movies.dat"
        self.movies.write_text("\n".join(f"{item}::Movie {item}::Drama" for item in catalog), encoding="ascii")
        torch.manual_seed(42)
        model = Recommender(len(seen), len(catalog), dim=4, architecture="mf")
        config = {"dim": 4, "architecture": "mf", "genres": False}
        torch.save({"state_dict": model.state_dict(), "users": sorted(seen), "items": catalog, "config": config}, self.root / "model.pt")
        metrics = {"data_kind": "MovieLens_local", "source_sha256": hashlib.sha256(self.ratings.read_bytes()).hexdigest(), "cutoffs": cutoffs}
        (self.root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        self.service = RecommendationService(self.root, self.ratings, self.movies)

    def tearDown(self):
        # This is a test-generated directory whose absolute boundary was checked above.
        self.temp.cleanup()

    def test_saved_model_filters_seen_items_and_is_repeatable(self):
        user = self.service.default_user
        result = self.service.recommend(user)
        self.assertEqual(result, self.service.recommend(user))
        items = [row["movie_id"] for row in result["recommendations"]]
        self.assertEqual(len(items), 10)
        self.assertEqual(len(set(items)), 10)
        self.assertFalse(set(items) & self.service.seen[user])
        self.assertEqual(result["method"], "bpr_matrix_factorization")

    def test_new_user_gets_popular_fallback(self):
        result = self.service.recommend(999999)
        expected = sorted(self.service.catalog, key=lambda item: (-self.service.popularity[item], item))[:10]
        self.assertEqual([row["movie_id"] for row in result["recommendations"]], expected)
        self.assertEqual(result["method"], "popular_fallback_for_new_user")

    def test_changed_dataset_is_rejected(self):
        self.ratings.write_text(self.ratings.read_text(encoding="ascii") + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "hash"):
            RecommendationService(self.root, self.ratings, self.movies)


if __name__ == "__main__":
    unittest.main()
