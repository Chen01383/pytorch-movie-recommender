"""Measure warm single-request CPU inference with cached item vectors, excluding loading."""
import argparse
import json
import math
from pathlib import Path
import platform
from statistics import median
import time

import torch

from inference import RecommendationService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/final"))
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("report/latency.json"))
    args = parser.parse_args()
    if args.requests < 20 or args.warmup < 1:
        parser.error("Use at least 20 measured requests and one warmup")
    torch.set_num_threads(1)
    service = RecommendationService(args.run, "data/ml-1m/ratings.dat", "data/ml-1m/movies.dat")
    users = sorted(user for user in service.positives if len(service.positives[user]) >= 5)[:100]
    for index in range(args.warmup):
        service.recommend(users[index % len(users)])
    elapsed = []
    for index in range(args.requests):
        start = time.perf_counter_ns()
        service.recommend(users[index % len(users)])
        elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(elapsed)
    result = {"device": "cpu", "torch_threads": 1, "processor": platform.processor(),
              "requests": args.requests, "warmup": args.warmup, "concurrency": 1,
              "k": 10, "unique_users": min(len(users), args.requests), "p50_ms": median(elapsed),
              "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
              "min_ms": min(elapsed), "max_ms": max(elapsed),
              "includes": "user encoding, seen filtering, full-catalog scores, stable ranking, named result creation",
              "excludes": "model/dataset loading, HTTP, disk IO, network, GUI",
              "item_vectors": "cached", "source_sha256": service.metrics["source_sha256"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
