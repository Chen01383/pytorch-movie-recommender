"""Reproduce the predeclared validation selection, then evaluate all fixed final seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_logged(name, command):
    logs = ROOT / "runs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"Starting {name}", flush=True)
    with (logs / f"{name}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print((logs / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-6000:])
        raise SystemExit(f"Failed: {name}; see runs/logs/{name}.log")
    print(f"Completed {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    plan_path = ROOT / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not (ROOT / "data/ml-1m/ratings.dat").is_file():
        run_logged("download", ["download_data.py"])
    run_logged("data_audit", ["inspect_data.py", "--ratings", "data/ml-1m/ratings.dat"])
    run_logged("popular_validation", ["baseline.py", "--ratings", "data/ml-1m/ratings.dat",
                                      "--validation-only", "--out", "runs/popular"])

    def train(dim, seed, destination, validation):
        command = ["train_torch.py", "--ratings", "data/ml-1m/ratings.dat", "--architecture", plan["architecture"],
                   "--dim", str(dim), "--seed", str(seed), "--epochs", str(plan["epochs"]),
                   "--patience", str(plan["patience"]), "--batch-size", str(plan["batch_size"]),
                   "--learning-rate", str(plan["learning_rate"]), "--weight-decay", str(plan["weight_decay"]),
                   "--k", str(plan["k"]), "--min-positives", str(plan["min_train_positives"]),
                   "--device", args.device, "--out", destination]
        if validation:
            command.append("--validation-only")
        run_logged(destination.replace("/", "_"), command)
        return json.loads((ROOT / destination / "metrics.json").read_text(encoding="utf-8"))

    comparisons = []
    for dim in plan["validation_dimensions"]:
        result = train(dim, plan["selection_seed"], f"runs/mf{dim}_valid", True)
        comparisons.append({"dim": dim, "seed": plan["selection_seed"], "best_epoch": result["best_epoch"],
                            "validation_ndcg": result["best_validation_ndcg"],
                            "validation_recall": result[f"Recall@{plan['k']}"]})
    selected = sorted(comparisons, key=lambda row: (-row["validation_ndcg"], row["dim"]))[0]
    locked = {"plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "validation_comparison": comparisons, "selected_dim": selected["dim"],
              "deployment_seed": plan["deployment_seed_fixed_before_test"], "final_seeds": plan["final_seeds"],
              "selection": "validation_only_before_any_final_test_run"}
    lock_path = ROOT / "runs/selection_locked.json"
    lock_path.write_text(json.dumps(locked, indent=2), encoding="utf-8")
    print("Locked selection: " + json.dumps(locked), flush=True)
    for seed in plan["final_seeds"]:
        destination = "runs/final" if seed == plan["deployment_seed_fixed_before_test"] else f"runs/final_seed{seed}"
        train(selected["dim"], seed, destination, False)
    run_logged("report", ["report.py", "--run", "runs/final", "--ratings", "data/ml-1m/ratings.dat", "--out", "report", "--plots"])
    run_logged("benchmark", ["benchmark.py"])
    run_logged("results", ["build_results.py"])
    run_logged("verification", ["verify_project.py"])
    print("Experiment complete. Deployment model: runs/final/model.pt", flush=True)


if __name__ == "__main__":
    main()
