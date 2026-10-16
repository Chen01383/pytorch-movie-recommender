"""A dependency-free, global-time-split MovieLens baseline and evaluation harness.

Real data: python baseline.py --ratings data/ml-1m/ratings.dat --out runs/popular
Synthetic smoke example: python baseline.py --demo --out runs/demo
The demo is synthetic and must not be used as a resume result.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys


@dataclass(frozen=True)
class Rating:
    user: int
    item: int
    rating: int
    timestamp: int


def load_ratings(path):
    rows = []
    pairs = set()
    with Path(path).open(encoding="latin-1") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.strip().split("::")
            if len(fields) != 4:
                raise ValueError(f"Invalid row {number}: expected 4 fields")
            user, item, rating, timestamp = map(int, fields)
            if not 1 <= rating <= 5 or user <= 0 or item <= 0 or timestamp < 0:
                raise ValueError(f"Invalid values on row {number}")
            if (user, item) in pairs:
                raise ValueError(f"Duplicate user-item pair on row {number}; decide a policy first")
            pairs.add((user, item))
            rows.append(Rating(user, item, rating, timestamp))
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def temporal_split(rows):
    """Approximate 80/10/10 by global timestamp; never split a timestamp group."""
    ordered = sorted(rows, key=lambda r: (r.timestamp, r.user, r.item))
    t1 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.8))].timestamp
    t2 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))].timestamp
    train = [r for r in ordered if r.timestamp < t1]
    valid = [r for r in ordered if t1 <= r.timestamp < t2]
    test = [r for r in ordered if r.timestamp >= t2]
    if not train or not valid or not test:
        raise ValueError("Not enough distinct timestamp groups for three nonempty splits")
    return train, valid, test, {"valid_start_timestamp": t1, "test_start_timestamp": t2}


def train_state(train):
    seen = defaultdict(set)
    positives = defaultdict(set)
    popularity = Counter()
    catalog = set()
    for row in train:
        catalog.add(row.item)
        seen[row.user].add(row.item)
        if row.rating >= 4:
            positives[row.user].add(row.item)
            popularity[row.item] += 1
    return seen, positives, popularity, sorted(catalog)


def ranking_metrics(recommended, relevant, k=10):
    if not relevant:
        raise ValueError("Users without eligible positive labels have no defined Recall")
    if len(set(recommended)) != len(recommended):
        raise ValueError("Duplicate recommendations")
    hits = [int(item in relevant) for item in recommended[:k]]
    recall = sum(hits) / len(relevant)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {"recall": recall, "ndcg": dcg / ideal}


def evaluate(train, holdout, recommend, k=10, min_positives=5):
    """Frozen training history; warm-user/warm-item, full train catalog evaluation.

    Holdout windows are evaluated independently with the same frozen train state.
    No validation or test observations are added to histories or model parameters.
    """
    seen, positives, _, catalog = train_state(train)
    catalog_set = set(catalog)
    relevant = defaultdict(set)
    counts = Counter()
    holdout_positive_users = set()
    eligible_users = {user for user in positives if len(positives[user]) >= min_positives}
    for row in holdout:
        if row.rating < 4:
            continue
        counts["positive_rows_total"] += 1
        holdout_positive_users.add(row.user)
        if row.user not in seen:
            counts["excluded_new_user_rows"] += 1
        elif row.user not in eligible_users:
            counts["excluded_insufficient_history_rows"] += 1
        elif row.item not in catalog_set:
            counts["excluded_new_item_rows"] += 1
        elif row.item in seen[row.user]:
            counts["excluded_seen_item_rows"] += 1
        else:
            relevant[row.user].add(row.item)
            counts["eligible_positive_rows"] += 1
    per_user = []
    recommended_items = set()
    for user in sorted(relevant):
        candidates = [item for item in catalog if item not in seen[user]]
        candidate_set = set(candidates)
        recommended = list(recommend(user, candidates, k))
        if len(recommended) != min(k, len(candidates)):
            raise ValueError("Recommender must return min(k, eligible catalog size) items")
        if any(item not in candidate_set for item in recommended):
            raise ValueError("Recommender returned an ineligible or seen item")
        metrics = ranking_metrics(recommended, relevant[user], k)
        per_user.append({"user_id": user, **metrics, "relevant_count": len(relevant[user])})
        recommended_items.update(recommended)
    if not per_user:
        raise ValueError("No evaluable users under this protocol")
    summary = {
        f"Recall@{k}": sum(row["recall"] for row in per_user) / len(per_user),
        f"NDCG@{k}": sum(row["ndcg"] for row in per_user) / len(per_user),
        f"CatalogCoverage@{k}": len(recommended_items) / len(catalog),
        "evaluated_users": len(per_user),
        "train_catalog_items": len(catalog),
        "holdout_positive_users": len(holdout_positive_users),
        "excluded_users_without_eligible_labels": len(holdout_positive_users) - len(per_user),
        "filter_counts_sequential": dict(counts),
        "protocol": "global_time_split; frozen_train_history; full_train_catalog; warm_only",
        "minimum_train_positives": min_positives,
        "positive_rating_threshold": 4,
    }
    return summary, per_user


def popularity_recommender(train):
    _, _, counts, _ = train_state(train)
    def recommend(user, candidates, k):
        return sorted(candidates, key=lambda item: (-counts[item], item))[:k]
    return recommend


def paired_bootstrap(a, b, key="ndcg", repeats=2000, seed=42):
    """B minus A, paired resampling of evaluated users, not individual rating rows."""
    a_map = {row["user_id"]: row[key] for row in a}
    b_map = {row["user_id"]: row[key] for row in b}
    if a_map.keys() != b_map.keys() or not a_map:
        raise ValueError("Paired comparison requires the same nonempty user set")
    differences = [b_map[user] - a_map[user] for user in sorted(a_map)]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                     for _ in range(repeats))
    return {"mean_difference_B_minus_A": sum(differences) / len(differences),
            "approx_95pct_user_bootstrap_interval": [samples[int(repeats * .025)],
                                                    samples[int(repeats * .975)]],
            "repeats": repeats, "seed": seed,
            "scope": "user_sampling_uncertainty_for_fixed_trained_models"}


def save_report(out, summary, per_user):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_user.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["user_id", "recall", "ndcg", "relevant_count"])
        writer.writeheader()
        writer.writerows(per_user)


def synthetic_ratings():
    rows = []
    timestamp = 1
    for item_round in range(12):
        for user in range(1, 21):
            # Each user has unique pairs; items are reused by other users at earlier times.
            rows.append(Rating(user, (user + item_round * 3) % 40 + 1, 4 + user % 2, timestamp))
            timestamp += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/popular"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-positives", type=int, default=5)
    parser.add_argument("--validation-only", action="store_true",
                        help="During development, evaluate validation only and leave test unused")
    args = parser.parse_args()
    if bool(args.ratings) == args.demo:
        parser.error("Choose exactly one of --ratings or --demo")
    if args.k < 1 or args.min_positives < 1:
        parser.error("k and min-positives must be positive")
    rows = synthetic_ratings() if args.demo else load_ratings(args.ratings)
    train, valid, test, cutoffs = temporal_split(rows)
    recommend = popularity_recommender(train)
    validation, validation_rows = evaluate(train, valid, recommend, args.k, args.min_positives)
    if args.validation_only:
        summary, per_user = validation, validation_rows
    else:
        summary, per_user = evaluate(train, test, recommend, args.k, args.min_positives)
    summary.update({"data_kind": "SYNTHETIC_SMOKE_ONLY" if args.demo else "MovieLens_local",
                    "model": "train_positive_count_popularity",
                    "report_split": "validation" if args.validation_only else "test",
                    "split_rows": {"train": len(train), "valid": len(valid), "test": len(test)},
                    "cutoffs": cutoffs, "python_version": sys.version,
                    "source_sha256": None if args.demo else hashlib.sha256(args.ratings.read_bytes()).hexdigest()})
    save_report(args.out, summary, per_user)
    save_report(args.out / "validation", validation, validation_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
