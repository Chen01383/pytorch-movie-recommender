"""Generate a report from a real, final MF run. Standard library; plots optional."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from baseline import (evaluate, load_ratings, popularity_recommender,
                      synthetic_ratings, temporal_split, train_state)


def group_results(model_rows, baseline_rows, positives):
    baseline_by_user = {row["user_id"]: row for row in baseline_rows}
    if {row["user_id"] for row in model_rows} != set(baseline_by_user):
        raise ValueError("Model and baseline must use the same user set")
    groups = [(0, 4, "<5"), (5, 19, "5-19"), (20, 49, "20-49"), (50, math.inf, "50+")]
    result = []
    for low, high, label in groups:
        rows = [row for row in model_rows if low <= len(positives[row["user_id"]]) <= high]
        if rows:
            result.append({"group": label, "users": len(rows),
                           "mf_ndcg": mean(row["ndcg"] for row in rows),
                           "popular_ndcg": mean(baseline_by_user[row["user_id"]]["ndcg"] for row in rows)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--out", type=Path, default=Path("report"))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--allow-demo", action="store_true", help="Smoke testing only; labels the report SYNTHETIC")
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("report_split") != "test":
        raise ValueError("Use a final test run, not validation-only output")
    synthetic = metrics.get("data_kind") == "SYNTHETIC_SMOKE_ONLY"
    if synthetic:
        if not args.allow_demo:
            raise ValueError("Demo metrics cannot be presented as real results")
        rows = synthetic_ratings()
    else:
        if metrics.get("data_kind") != "MovieLens_local" or args.ratings is None:
            raise ValueError("A MovieLens run requires --ratings")
        actual_hash = hashlib.sha256(args.ratings.read_bytes()).hexdigest()
        if metrics.get("source_sha256") != actual_hash:
            raise ValueError("Ratings file hash differs from the training run")
        rows = load_ratings(args.ratings)

    train, _, test, cutoffs = temporal_split(rows)
    if cutoffs != metrics.get("cutoffs"):
        raise ValueError("Time cutoffs differ from the training run")
    config = metrics["config"]
    k = config["k"]
    min_positives = config["min_positives"]
    popular, popular_rows = evaluate(train, test, popularity_recommender(train), k, min_positives)
    with (args.run / "per_user.csv").open(encoding="utf-8", newline="") as stream:
        model_rows = [{"user_id": int(row["user_id"]), "recall": float(row["recall"]),
                       "ndcg": float(row["ndcg"]), "relevant_count": int(row["relevant_count"])}
                      for row in csv.DictReader(stream)]
    if len(model_rows) != metrics.get("evaluated_users"):
        raise ValueError("User row count differs from metrics.json")
    if len({row["user_id"] for row in model_rows}) != len(model_rows):
        raise ValueError("Duplicate user rows")
    for row in model_rows:
        if any(not math.isfinite(row[key]) or not 0 <= row[key] <= 1 for key in ["recall", "ndcg"]):
            raise ValueError("Invalid per-user metric")
    for key, column in [(f"Recall@{k}", "recall"), (f"NDCG@{k}", "ndcg")]:
        if not math.isclose(mean(row[column] for row in model_rows), metrics[key], abs_tol=1e-10):
            raise ValueError("Per-user results differ from summary metrics")
    _, positives, _, _ = train_state(train)
    groups = group_results(model_rows, popular_rows, positives)
    interval = metrics["paired_user_bootstrap_vs_popular"]

    lines = ["# MF 推荐系统实验报告", ""]
    if synthetic:
        lines += ["**SYNTHETIC SMOKE ONLY：合成调试文件，不是真实项目成绩。**", ""]
    lines += ["本文由运行文件生成。解释与个人贡献需要你补充。", "",
              "| 指标 | 热门基线 | MF | 绝对变化 |", "|---|---:|---:|---:|"]
    for key in [f"Recall@{k}", f"NDCG@{k}", f"CatalogCoverage@{k}"]:
        a, b = popular[key], metrics[key]
        lines.append(f"| {key} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    lines += ["", f"评估用户数：{metrics['evaluated_users']}；训练目录：{metrics['train_catalog_items']} 部电影。",
              f"时间边界：{json.dumps(cutoffs, ensure_ascii=False)}。",
              "", "## 固定模型的逐用户配对 bootstrap", "",
              f"NDCG 平均差值（MF 减热门）：{interval['mean_difference_B_minus_A']:.6f}。",
              f"近似 95% 区间：{interval['approx_95pct_user_bootstrap_interval']}。",
              "仅衡量固定模型的用户抽样不确定性；首版单种子，未完整衡量训练波动。", "",
              "## 按训练期正向评分数分组", "",
              "| 训练正向项目数 | 用户数 | 热门 NDCG | MF NDCG |", "|---|---:|---:|---:|"]
    for group in groups:
        lines.append(f"| {group['group']} | {group['users']} | {group['popular_ndcg']:.6f} | {group['mf_ndcg']:.6f} |")
    lines += ["", "## 范围与局限", "",
              "全局时间切分、训练历史冻结、训练目录全候选评价，主指标限制为合格用户与电影。",
              f"过滤情况：{json.dumps(metrics['filter_counts_sequential'], ensure_ascii=False)}。",
              "评分偏好不是点击日志；离线差异不能直接推断业务 CTR 或因果效果。", "",
              "## 复现信息", "", f"配置：`{json.dumps(config, ensure_ascii=False)}`",
              f"PyTorch：{metrics.get('torch_version')}；设备：{metrics.get('device')}。",
              f"数据 SHA256：{metrics.get('source_sha256')}。", "",
              "## 需要自己补充", "", "- 验证集为什么选这个维度？",
              "- 哪类用户更难推荐？列两个实际案例。", "- 模型改善或未改善的可能原因。",
              "- 你独立重写、修改、检查或分析了哪些部分？", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    # Preserve the text report even if the optional plotting dependency is absent.
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if args.plots:
        try:
            import matplotlib
        except ImportError:
            print("Text report saved. Install matplotlib to generate optional plots.")
            return
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        history = metrics["history"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].plot([row["epoch"] for row in history], [row["mean_bpr_loss"] for row in history], marker="o")
        axes[1].plot([row["epoch"] for row in history], [row["validation_ndcg"] for row in history], marker="o")
        for axis, title in zip(axes, ["Training BPR loss", f"Validation NDCG@{k}"]):
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(args.out / "training_curves.png", dpi=160)
        plt.close(fig)
        labels = [f"Recall@{k}", f"NDCG@{k}"]
        fig, axis = plt.subplots(figsize=(6, 3.5))
        axis.bar([-.18, .82], [popular[label] for label in labels], width=.36, label="Popular")
        axis.bar([.18, 1.18], [metrics[label] for label in labels], width=.36, label="MF")
        axis.set_xticks([0, 1], labels)
        axis.set_ylabel("Test metric")
        axis.legend()
        axis.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(args.out / "model_comparison.png", dpi=160)
        plt.close(fig)
        lines += ["## 图表", "", "![Training curves](training_curves.png)", "",
                  "![Model comparison](model_comparison.png)", ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {(args.out / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
