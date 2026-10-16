"""Verify saved real results, inference behavior and all unit tests."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import torch

from inference import RecommendationService


ROOT = Path(__file__).resolve().parent


def main():
    tests = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_protocol.py", "test_five_day.py", "test_model.py"],
                           cwd=ROOT, text=True, capture_output=True)
    print(tests.stdout + tests.stderr)
    if tests.returncode:
        raise SystemExit(tests.returncode)
    count = int(re.search(r"Ran (\d+) tests", tests.stderr).group(1))
    locked = json.loads((ROOT / "runs/selection_locked.json").read_text(encoding="utf-8"))
    source_sha = hashlib.sha256((ROOT / "data/ml-1m/ratings.dat").read_bytes()).hexdigest()
    for seed in locked["final_seeds"]:
        directory = ROOT / ("runs/final" if seed == locked["deployment_seed"] else f"runs/final_seed{seed}")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["report_split"] == "test"
        assert metrics["source_sha256"] == source_sha
        assert metrics["config"]["dim"] == locked["selected_dim"]
        assert metrics["config"]["seed"] == seed
        assert (directory / "model.pt").is_file()
    torch.set_num_threads(1)
    service = RecommendationService(ROOT / "runs/final", ROOT / "data/ml-1m/ratings.dat", ROOT / "data/ml-1m/movies.dat")
    known = service.recommend(service.default_user)
    unknown = service.recommend(999999)
    assert known["method"] == "bpr_matrix_factorization"
    assert unknown["method"] == "popular_fallback_for_new_user"
    assert known == service.recommend(service.default_user)
    items = [row["movie_id"] for row in known["recommendations"]]
    assert len(items) == len(set(items)) == 10
    assert not set(items) & service.seen[service.default_user]
    assert all(math.isfinite(row["score"]) for row in known["recommendations"])
    assert all(row["title"] for row in known["recommendations"])
    result = {"status": "passed", "unit_tests_passed": count, "source_sha256": source_sha,
              "final_seeds_verified": locked["final_seeds"], "deployment_seed": locked["deployment_seed"],
              "known_demo_user": service.default_user, "known_user_recommendations": len(items),
              "new_user_fallback": "passed", "seen_filtering": "passed", "repeatability": "passed",
              "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    (ROOT / "report/verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
