"""Learning implementation: ID matrix factorization or a small neural two-tower.

Dependencies: PyTorch. CPU default. See README for installation.
Example: python train_torch.py --ratings data/ml-1m/ratings.dat --movies data/ml-1m/movies.dat --genres
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

# Required by deterministic CUDA matrix multiplication; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

from baseline import (evaluate, load_ratings, paired_bootstrap, popularity_recommender,
                      save_report, synthetic_ratings, temporal_split, train_state)


def genre_matrix(catalog, movies_path):
    metadata = {}
    with Path(movies_path).open(encoding="latin-1") as stream:
        for line in stream:
            item, title, genres = line.rstrip("\n").split("::")
            metadata[int(item)] = genres.split("|")
    missing = set(catalog) - metadata.keys()
    if missing:
        raise ValueError(f"Missing metadata for {len(missing)} train catalog items")
    # Fit the genre vocabulary on the train catalog only.
    vocab = sorted({genre for item in catalog for genre in metadata[item]})
    matrix = torch.zeros(len(catalog), len(vocab))
    lookup = {genre: index for index, genre in enumerate(vocab)}
    for index, item in enumerate(catalog):
        for genre in metadata[item]:
            matrix[index, lookup[genre]] = 1.0
    return matrix, vocab


class Recommender(nn.Module):
    def __init__(self, n_users, n_items, dim=32, architecture="twotower", genres=None):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_embedding.weight, std=.1)
        nn.init.normal_(self.item_embedding.weight, std=.1)
        self.user_tower = (nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                           if architecture == "twotower" else nn.Identity())
        self.item_tower = (nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                           if architecture == "twotower" else nn.Identity())
        self.register_buffer("genre_features", genres)
        self.genre_projection = (nn.Linear(genres.shape[1], dim, bias=False)
                                 if genres is not None else None)

    def encode_users(self, users):
        return self.user_tower(self.user_embedding(users))

    def encode_items(self, items):
        embedding = self.item_embedding(items)
        if self.genre_projection is not None:
            embedding = embedding + self.genre_projection(self.genre_features[items])
        return self.item_tower(embedding)

    def forward(self, users, items):
        return (self.encode_users(users) * self.encode_items(items)).sum(dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--movies", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--genres", action="store_true")
    parser.add_argument("--architecture", choices=["mf", "twotower"], default="twotower")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.00001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-positives", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validation-only", action="store_true",
                        help="Use during tuning: never evaluate the test window")
    parser.add_argument("--out", type=Path, default=Path("runs/twotower"))
    args = parser.parse_args()
    if bool(args.ratings) == args.demo:
        parser.error("Choose exactly one of --ratings or --demo")
    if args.genres and not args.movies:
        parser.error("--genres requires --movies")
    if min(args.dim, args.epochs, args.batch_size, args.k, args.min_positives, args.patience, args.threads) < 1:
        parser.error("Dimensions, counts and k must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid learning rate or regularization")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: verify your driver and the installed PyTorch wheel")
    device = torch.device(device_name)
    rng = random.Random(args.seed)
    rows = synthetic_ratings() if args.demo else load_ratings(args.ratings)
    train, valid, test, cutoffs = temporal_split(rows)
    seen, _, _, catalog = train_state(train)
    user_ids = sorted(seen)
    user_index = {user: index for index, user in enumerate(user_ids)}
    item_index = {item: index for index, item in enumerate(catalog)}
    seen_indices = {user: {item_index[item] for item in items} for user, items in seen.items()}
    pairs = [(r.user, r.item) for r in train if r.rating >= 4 and len(seen[r.user]) < len(catalog)]
    if not pairs:
        raise ValueError("No training pairs with eligible negative candidates")
    features, vocabulary = genre_matrix(catalog, args.movies) if args.genres else (None, [])
    model = Recommender(len(user_ids), len(catalog), args.dim, args.architecture, features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    item_vectors = None

    def refresh_item_vectors():
        with torch.no_grad():
            return model.encode_items(torch.arange(len(catalog), dtype=torch.long, device=device))

    def recommend(user, candidates, k):
        indices = torch.tensor([item_index[item] for item in candidates], dtype=torch.long, device=device)
        user = torch.tensor([user_index[user]], dtype=torch.long, device=device)
        with torch.no_grad():
            scores = (model.encode_users(user) * item_vectors[indices]).sum(dim=-1).tolist()
        ordered = sorted(zip(candidates, scores), key=lambda entry: (-entry[1], entry[0]))
        return [item for item, _ in ordered[:k]]

    best_value = float("-inf")
    best_weights = None
    history = []
    stale_epochs = 0
    best_epoch = None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(pairs)
        total_loss = 0.0
        for offset in range(0, len(pairs), args.batch_size):
            batch = pairs[offset:offset + args.batch_size]
            negatives = []
            for user, _ in batch:
                negative = rng.randrange(len(catalog))
                while negative in seen_indices[user]:
                    negative = rng.randrange(len(catalog))
                negatives.append(negative)
            users = torch.tensor([user_index[user] for user, _ in batch], dtype=torch.long, device=device)
            positive = torch.tensor([item_index[item] for _, item in batch], dtype=torch.long, device=device)
            negative = torch.tensor(negatives, dtype=torch.long, device=device)
            # -log(sigmoid(pos_score - neg_score)), using stable softplus.
            loss = F.softplus(model(users, negative) - model(users, positive)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        model.eval()
        item_vectors = refresh_item_vectors()
        validation, _ = evaluate(train, valid, recommend, args.k, args.min_positives)
        value = validation[f"NDCG@{args.k}"]
        history.append({"epoch": epoch, "mean_bpr_loss": total_loss / len(pairs), "validation_ndcg": value})
        print(json.dumps(history[-1]), flush=True)
        if value > best_value:
            best_value = value
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    model.load_state_dict(best_weights)
    model.eval()
    item_vectors = refresh_item_vectors()
    # Tune with --validation-only, then freeze configurations before final test runs.
    report_window = valid if args.validation_only else test
    summary, per_user = evaluate(train, report_window, recommend, args.k, args.min_positives)
    popular, popular_per_user = evaluate(train, report_window, popularity_recommender(train), args.k, args.min_positives)
    summary.update({
        "data_kind": "SYNTHETIC_SMOKE_ONLY" if args.demo else "MovieLens_local",
        "model": args.architecture, "genres": args.genres, "device": device_name,
        "report_split": "validation" if args.validation_only else "test",
        "best_epoch": best_epoch, "best_validation_ndcg": best_value,
        "history": history, "cutoffs": cutoffs,
        "split_rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch_version": torch.__version__, "python_version": sys.version,
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device_name == "cuda" else None,
        "source_sha256": None if args.demo else hashlib.sha256(args.ratings.read_bytes()).hexdigest(),
        "movie_metadata_sha256": hashlib.sha256(args.movies.read_bytes()).hexdigest() if args.genres else None,
        "train_and_evaluation_seconds": time.perf_counter() - start,
        "popular_metrics_same_report_split": popular,
        "paired_user_bootstrap_vs_popular": paired_bootstrap(popular_per_user, per_user, seed=args.seed),
    })
    save_report(args.out, summary, per_user)
    torch.save({"state_dict": model.state_dict(), "users": user_ids, "items": catalog,
                "genre_vocabulary": vocabulary, "config": summary["config"]}, args.out / "model.pt")
    print(json.dumps({key: summary[key] for key in [f"Recall@{args.k}", f"NDCG@{args.k}", "best_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
