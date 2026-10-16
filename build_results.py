"""Aggregate all preregistered final seeds and create a factual Chinese results report."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    plan = read("experiment_plan.json")
    locked = read("runs/selection_locked.json")
    if locked["plan_sha256"] != hashlib.sha256((ROOT / "experiment_plan.json").read_bytes()).hexdigest():
        raise ValueError("Experiment plan changed after selection was locked")
    audit = read("runs/data_audit.json")
    finals = []
    for seed in plan["final_seeds"]:
        directory = "runs/final" if seed == locked["deployment_seed"] else f"runs/final_seed{seed}"
        result = read(directory + "/metrics.json")
        if result["report_split"] != "test" or result["config"]["seed"] != seed:
            raise ValueError("Wrong final seed or data window")
        if result["config"]["dim"] != locked["selected_dim"] or result["source_sha256"] != audit["source_sha256"]:
            raise ValueError("Data or selected dimension differs")
        finals.append(result)
    primary = finals[plan["final_seeds"].index(locked["deployment_seed"])]
    baseline = primary["popular_metrics_same_report_split"]
    if any(result["cutoffs"] != primary["cutoffs"] or result["evaluated_users"] != primary["evaluated_users"] for result in finals):
        raise ValueError("Final runs used different evaluation populations")
    k = plan["k"]
    metrics = [f"Recall@{k}", f"NDCG@{k}", f"CatalogCoverage@{k}"]
    aggregate = {key: {"mean": mean(result[key] for result in finals),
                       "sample_std": stdev(result[key] for result in finals), "popular": baseline[key]}
                 for key in metrics}
    bootstrap = primary["paired_user_bootstrap_vs_popular"]
    latency = read("report/latency.json")
    summary = {"dataset": audit, "locked_selection": locked, "aggregate": aggregate,
               "final_seeds": [{"seed": result["config"]["seed"], "best_epoch": result["best_epoch"],
                                **{key: result[key] for key in metrics}} for result in finals],
               "deployment_seed": locked["deployment_seed"], "deployment_bootstrap": bootstrap,
               "latency": latency, "trained_on": {"torch": primary["torch_version"], "gpu": primary["gpu_name"],
                                                    "cuda": primary["cuda_build"]},
               "evaluation_scope": "warm users with >=5 train positive items, warm train-catalog items, frozen train history"}
    report = ROOT / "report"
    report.mkdir(exist_ok=True)
    (report / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# PyTorch 电影推荐系统：实际实验结果", "",
             "本报告来自已执行的真实 MovieLens 1M 实验。模型和实验由 AI 辅助实现；个人简历应准确说明自己的学习、修改与分析贡献。", "",
             "## 数据与任务", "",
             f"读取 {audit['ratings']:,} 条评分、{audit['unique_users']:,} 位用户、{audit['unique_rated_movies']:,} 部实际被评分电影。",
             f"按全局时间划分：训练 {audit['split_rows']['train']:,}、验证 {audit['split_rows']['valid']:,}、测试 {audit['split_rows']['test']:,} 条；同时间戳不跨窗口。",
             f"训练目录 {primary['train_catalog_items']:,} 部电影；最终合格评估用户 {primary['evaluated_users']:,} 位。",
             "评分 >=4 为正向标签，主评估要求用户至少五个训练正向项目；过滤全部训练期已评分电影。用户画像冻结在训练截止状态，验证/测试分别评价。",
             "推荐候选为训练目录中的全部未评分电影。新用户、新电影不进入主排名指标，需要结合过滤人数理解结论范围。", "",
             "## 模型与选型", "", "模型是用户/电影 ID embedding 的点积，使用 BPR 排序损失与 Adam。",
             "先比较两个维度的验证结果，锁定选择之后才执行最终测试；部署种子事先固定为 42。", "",
             "| 维度 | 验证 NDCG@10 | 验证 Recall@10 | 最佳 epoch |", "|---:|---:|---:|---:|"]
    for row in locked["validation_comparison"]:
        lines.append(f"| {row['dim']} | {row['validation_ndcg']:.6f} | {row['validation_recall']:.6f} | {row['best_epoch']} |")
    lines += ["", f"选中维度：**{locked['selected_dim']}**。其余参数见 experiment_plan.json，未按测试表现挑选种子。", "",
              "## 测试结果", "", "标准差使用三种子的样本标准差；热门基线为同一测试协议。", "",
              "| 指标 | 热门基线 | MF 三种子均值 ± 标准差 | 均值绝对变化 |", "|---|---:|---:|---:|"]
    for key in metrics:
        row = aggregate[key]
        lines.append(f"| {key} | {row['popular']:.6f} | {row['mean']:.6f} ± {row['sample_std']:.6f} | {row['mean']-row['popular']:+.6f} |")
    lines += ["", "| 随机种子 | 最佳 epoch | Recall@10 | NDCG@10 |", "|---:|---:|---:|---:|"]
    for result in finals:
        lines.append(f"| {result['config']['seed']} | {result['best_epoch']} | {result[f'Recall@{k}']:.6f} | {result[f'NDCG@{k}']:.6f} |")
    improvement = aggregate[f"NDCG@{k}"]["mean"] - baseline[f"NDCG@{k}"]
    if baseline[f"NDCG@{k}"]:
        relative = improvement / baseline[f"NDCG@{k}"] * 100
        lines += ["", f"三种子平均 NDCG 的相对变化为 {relative:+.2f}%；绝对变化为 {improvement:+.6f}。"]
    low, high = bootstrap["approx_95pct_user_bootstrap_interval"]
    interpretation = "区间整体大于零，支持这次合格评估人群上的平均改善。" if low > 0 else ("区间整体小于零，这次固定模型低于基线。" if high < 0 else "区间包含零，这次固定模型的改善尚不稳定。")
    lines += ["", "## 不确定性与分组", "",
              f"部署模型（种子 42）相对热门的 NDCG 配对差值为 {bootstrap['mean_difference_B_minus_A']:+.6f}，逐用户 bootstrap 的近似 95% 区间为 [{low:+.6f}, {high:+.6f}]。",
              interpretation, "该区间衡量固定模型的用户抽样不确定性；上表的三种子标准差另行描述训练波动，两者不能混为同一个区间。",
              "用户活跃度分组和种子 42 的详细报告见 [report.md](report.md)。", "",
              "## 失败案例观察", ""]
    with (ROOT / "runs/final/per_user.csv").open(encoding="utf-8", newline="") as stream:
        users = [{"user_id": int(row["user_id"]), "ndcg": float(row["ndcg"]),
                  "recall": float(row["recall"]), "relevant_count": int(row["relevant_count"])} for row in csv.DictReader(stream)]
    worst = sorted(users, key=lambda row: (row["ndcg"], -row["relevant_count"], row["user_id"]))[:2]
    for row in worst:
        lines.append(f"- 用户 {row['user_id']}：合格未来正向标签 {row['relevant_count']} 个，Recall@10={row['recall']:.6f}，NDCG@10={row['ndcg']:.6f}。该例说明平均改善仍可能伴随个体推荐失败。")
    lines += ["", "## 推理性能", "",
              f"CPU、单线程、并发 1、预热 {latency['warmup']} 次后测量 {latency['requests']} 次请求：p50 **{latency['p50_ms']:.3f} ms**，p95 **{latency['p95_ms']:.3f} ms**。",
              "物品向量缓存；包含用户编码、过滤、全目录打分、排序与结果构造；不含模型/数据加载、HTTP 和磁盘读写。该数字是本机热推理测量。", "",
              "## 实验限制", "", "- 主指标只代表满足训练历史和目录条件的人群；新用户回退热门，新电影需要额外内容模型。",
              "- 数据为历史电影评分，存在曝光与选择偏差，无法从这次离线实验推断业务 CTR、收益或因果提升。",
              "- 三种子与单个时间切分仍有限，后续可使用更多时间窗口和模型进行验证。", "",
              "## 环境与复现", "", f"PyTorch {primary['torch_version']}；CUDA {primary['cuda_build']}；训练显卡 {primary['gpu_name']}。",
              f"评分文件 SHA256：`{audit['source_sha256']}`。", "",
              "模型权重与运行文件保存在 runs/。按 README 的下载和复现命令执行即可重新训练；原始数据按 GroupLens 条款从官网下载。", "",
              "数据：[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)；方法：[BPR 原论文](https://arxiv.org/abs/1205.2618)。", "",
              "## 图表", "", "![Three seed comparison](seed_comparison.png)", "",
              "![Training curve](training_curves.png)", ""]
    (report / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, key in zip(axes, metrics[:2]):
        axis.bar([str(result["config"]["seed"]) for result in finals], [result[key] for result in finals], color="#2563eb")
        axis.axhline(baseline[key], color="#f97316", linestyle="--", label="Popular baseline")
        axis.set_title(key)
        axis.set_xlabel("Training seed")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report / "seed_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"selected_dim": locked["selected_dim"], "aggregate": aggregate, "latency": latency}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
