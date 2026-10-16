"""Day 2: inspect real MovieLens ratings without evaluating any holdout model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from baseline import load_ratings, temporal_split, train_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/data_audit.json"))
    args = parser.parse_args()
    rows = load_ratings(args.ratings)
    train, valid, test, cutoffs = temporal_split(rows)
    _, positives, _, catalog = train_state(train)
    info = {
        "ratings": len(rows),
        "unique_users": len({row.user for row in rows}),
        "unique_rated_movies": len({row.item for row in rows}),
        "rating_distribution": dict(sorted(Counter(row.rating for row in rows).items())),
        "duplicate_policy": "loader_rejects_duplicate_user_item_pairs",
        "first_timestamp_utc": datetime.fromtimestamp(min(row.timestamp for row in rows), timezone.utc).isoformat(),
        "last_timestamp_utc": datetime.fromtimestamp(max(row.timestamp for row in rows), timezone.utc).isoformat(),
        "split_rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "cutoffs": cutoffs,
        "train_catalog_items": len(catalog),
        "train_users_with_at_least_5_positive_items": sum(len(items) >= 5 for items in positives.values()),
        "source_sha256": hashlib.sha256(args.ratings.read_bytes()).hexdigest(),
        "note": "Descriptive data audit. This does not evaluate or select a model on test labels.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
