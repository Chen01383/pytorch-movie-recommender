"""Interactive local recommendation demonstration, without a web server."""
from pathlib import Path

from inference import RecommendationService


def main():
    root = Path(__file__).resolve().parent
    if not (root / "data/ml-1m/ratings.dat").is_file():
        print("First run: python download_data.py")
        return
    print("Loading trained model and training histories...")
    service = RecommendationService(root / "runs/final", root / "data/ml-1m/ratings.dat", root / "data/ml-1m/movies.dat")
    print(f"Ready. Example known user: {service.default_user}; example new user: 999999.")
    print("Press Enter for the example user, enter an ID, or q to quit.")
    while True:
        try:
            value = input("\nUser ID: ").strip()
        except EOFError:
            break
        if value.lower() == "q":
            break
        try:
            user = service.default_user if not value else int(value)
            result = service.recommend(user)
        except ValueError as error:
            print(error)
            continue
        print("Method:", result["method"])
        for rank, movie in enumerate(result["recommendations"], 1):
            score = "popular" if movie["score"] is None else f"{movie['score']:.4f}"
            print(f"{rank:2d}. {movie['title']} | ID={movie['movie_id']} | score={score}")


if __name__ == "__main__":
    main()
