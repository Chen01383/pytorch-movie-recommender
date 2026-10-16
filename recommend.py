"""Print named movie recommendations from a saved real MF run."""
import argparse
import json
from pathlib import Path

from inference import RecommendationService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/final"))
    parser.add_argument("--ratings", type=Path, default=Path("data/ml-1m/ratings.dat"))
    parser.add_argument("--movies", type=Path, default=Path("data/ml-1m/movies.dat"))
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    service = RecommendationService(args.run, args.ratings, args.movies)
    user = service.default_user if args.user_id is None else args.user_id
    print(json.dumps(service.recommend(user, args.k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
