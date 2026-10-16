"""First lesson: Python containers, functions and a popularity baseline. No dependencies."""
from collections import Counter


def recommend_popular(ratings, user_id, k=3):
    # All rated items are seen, including low ratings.
    seen = {movie for user, movie, score in ratings if user == user_id}
    positive_counts = Counter(movie for user, movie, score in ratings if score >= 4)
    ordered = sorted(positive_counts, key=lambda movie: (-positive_counts[movie], movie))
    return [movie for movie in ordered if movie not in seen][:k]


if __name__ == "__main__":
    ratings = [
        (1, "A", 5), (1, "B", 2),
        (2, "A", 4), (2, "C", 5),
        (3, "C", 4), (3, "D", 5),
        (4, "E", 4),
    ]
    print("Recommendations for user 1:", recommend_popular(ratings, 1, 3))
    print("Recommendations for a new user:", recommend_popular(ratings, 99, 3))
    print("Exercises: explain why B is excluded; change k; add two positive ratings for E.")
