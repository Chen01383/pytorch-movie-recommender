"""Load a trained MF model once, cache item vectors and serve ranked recommendations."""
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

import torch

from baseline import load_ratings, temporal_split, train_state
from train_torch import Recommender


class RecommendationService:
    def __init__(self, run, ratings, movies):
        self.run = Path(run)
        ratings = Path(ratings)
        self.metrics = json.loads((self.run / "metrics.json").read_text(encoding="utf-8"))
        if self.metrics.get("data_kind") != "MovieLens_local":
            raise ValueError("Inference requires a real MovieLens run")
        actual_hash = hashlib.sha256(ratings.read_bytes()).hexdigest()
        if actual_hash != self.metrics.get("source_sha256"):
            raise ValueError("Ratings file hash differs from the training run")
        checkpoint = torch.load(self.run / "model.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if config["architecture"] != "mf" or config.get("genres"):
            raise ValueError("This inference service expects MF without genre features")
        train, _, _, cutoffs = temporal_split(load_ratings(ratings))
        self.seen, self.positives, self.popularity, self.catalog = train_state(train)
        if cutoffs != self.metrics["cutoffs"]:
            raise ValueError("Training time boundaries differ")
        if checkpoint["items"] != self.catalog or checkpoint["users"] != sorted(self.seen):
            raise ValueError("Saved ID mappings differ from the dataset")
        self.user_index = {user: index for index, user in enumerate(checkpoint["users"])}
        self.item_index = {item: index for index, item in enumerate(self.catalog)}
        self.model = Recommender(len(self.user_index), len(self.catalog), dim=config["dim"], architecture="mf")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        with torch.no_grad():
            self.item_vectors = self.model.encode_items(torch.arange(len(self.catalog), dtype=torch.long))
        self.titles = {}
        with Path(movies).open(encoding="latin-1") as stream:
            for line in stream:
                item, title, _ = line.rstrip("\n").split("::")
                self.titles[int(item)] = title
        self.history = defaultdict(lambda: deque(maxlen=8))
        for row in train:
            self.history[row.user].append({"movie_id": row.item, "rating": row.rating,
                                            "title": self.titles.get(row.item, str(row.item))})
        self.default_user = min(user for user in self.positives if len(self.positives[user]) >= 5)

    def recommend(self, user_id, k=10):
        if user_id < 1 or not 1 <= k <= 100:
            raise ValueError("user_id must be positive; k must be between 1 and 100")
        seen = self.seen.get(user_id, set())
        candidates = [item for item in self.catalog if item not in seen]
        if user_id not in self.user_index:
            method = "popular_fallback_for_new_user"
            items = sorted(candidates, key=lambda item: (-self.popularity[item], item))[:k]
            ranked = [(item, None) for item in items]
        else:
            method = "bpr_matrix_factorization"
            with torch.no_grad():
                vector = self.model.encode_users(torch.tensor([self.user_index[user_id]], dtype=torch.long))
                scores = (vector * self.item_vectors).sum(dim=-1).tolist()
            ranked = sorted(((item, scores[self.item_index[item]]) for item in candidates),
                            key=lambda entry: (-entry[1], entry[0]))[:k]
        return {"user_id": user_id, "method": method, "history": "frozen_at_training_cutoff",
                "score_note": "Ranking score, not calibrated probability",
                "recent_training_history": list(self.history.get(user_id, [])),
                "recommendations": [{"movie_id": item, "title": self.titles.get(item, str(item)), "score": score}
                                    for item, score in ranked]}
