# PyTorch Movie Recommender · 电影推荐与统计评估

An end-to-end **BPR matrix factorization** project with temporal evaluation, validation-only model selection, three training seeds, paired bootstrap analysis, and cached inference.

基于 MovieLens 1M 的可复现推荐系统。项目在 AI 辅助下实现，重点展示训练、评估、统计分析与推理流程。

## Results

| Metric | Popular baseline | BPR-MF, mean ± sample std across 3 seeds |
|---|---:|---:|
| Recall@10 | 0.053273 | 0.065752 ± 0.001389 |
| NDCG@10 | 0.195653 | 0.228696 ± 0.003995 |
| Catalog coverage@10 | 0.035227 | 0.106772 ± 0.008206 |

**NDCG@10 improved by 16.89% relative to the popularity baseline.** These are offline results for 955 eligible users and a training catalog of 3,662 movies under the protocol below.

![Three-seed test results](report/seed_comparison.png)

See the [full experiment report](report/RESULTS.md), [per-user analysis](report/report.md), and [machine-readable summary](report/experiment_summary.json).

## Quick start — Windows

Prepare Python 3.12 or 3.13, then run these commands in the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe recommend.py
```

The repository includes the trained checkpoint. The original MovieLens files are downloaded from GroupLens; no retraining is needed for this demo. Inference runs on CPU, so a CUDA GPU is optional for using the saved model.

```powershell
# Known user: personalized recommendations
.\.venv\Scripts\python.exe recommend.py --user-id 635

# New user: popularity fallback
.\.venv\Scripts\python.exe recommend.py --user-id 999999

# Interactive demonstration
.\.venv\Scripts\python.exe demo.py
```

After environment setup, you can also double-click `start_demo.cmd`.

## Reproduce the experiment

```powershell
.\.venv\Scripts\python.exe run_experiment.py --device cpu
```

For a compatible NVIDIA GPU, install the build used in the original experiment and verify an actual CUDA operation:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe check_setup.py
.\.venv\Scripts\python.exe run_experiment.py --device cuda
```

Actual training used PyTorch 2.14.1+cu130 on an NVIDIA GeForce RTX 5070 Ti Laptop GPU. Hardware and library changes may affect numerical results. See [experiment_plan.json](experiment_plan.json) for the fixed settings and [requirements-lock.txt](requirements-lock.txt) for the recorded environment.

## Evaluation protocol

- Load 1,000,209 ratings from 6,040 users; 3,706 movies have recorded ratings.
- Split globally by time into 800,164 training, 100,024 validation, and 100,021 test rows. Equal timestamps stay together.
- Treat ratings >=4 as positive preference. Learn ID mappings and popularity only from the training window.
- Rank all unrated movies in the training catalog; filter every training-rated movie, including low ratings.
- Main ranking evaluation requires at least five training positives per user and eligible future positive labels in the training catalog. Report exclusion counts.
- Freeze training histories. Evaluate validation and test as separate future windows without updating user histories.
- Compare 16/32 dimensions using validation only, lock the selected dimension, then evaluate the predeclared seeds 42/43/44.
- Fix deployment to seed 42 before testing. Report seed variability separately from the paired user-bootstrap interval.

![Training curves](report/training_curves.png)

## Inference performance

Cached, single-thread CPU inference over 200 measured requests after 20 warmups: **p50 1.194 ms, p95 1.304 ms**. This includes scoring, filtering, ranking, and result construction; excludes model/data loading, HTTP, disk I/O, and networking. See [latency.json](report/latency.json).

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_protocol.py test_five_day.py test_model.py
.\.venv\Scripts\python.exe verify_project.py
```

13 tests passed, with additional checks of the saved real model, known-user recommendations, unseen-item filtering, and new-user fallback. `verify_project.py` requires the downloaded original data. See [verification.json](report/verification.json).

## Project map

| File | Purpose |
|---|---|
| `baseline.py` | Data loading, time split, popularity baseline, ranking metrics, bootstrap |
| `train_torch.py` | PyTorch BPR-MF training and validation early stopping |
| `inference.py` | Checkpoint loading, cached item vectors, ranking and cold-user fallback |
| `recommend.py`, `demo.py` | Command-line and interactive recommendations |
| `run_experiment.py` | Reproducible validation selection, final runs, reports and verification |
| `report.py`, `build_results.py` | Per-user/group analysis, seed aggregation and charts |
| `benchmark.py` | Precisely scoped warm-inference latency measurements |
| `runs/` | Saved checkpoints and real experiment outputs |

For the mathematics and code walkthrough, see [核心代码与面试讲解](核心代码与面试讲解.md).

## Data and references

Original data: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). Read the original README downloaded with the data for its terms. Raw ratings and the local Python environment are excluded from this repository; source checksums are recorded in [DATA_SOURCES.json](DATA_SOURCES.json).

Dataset citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872).

Method: Rendle et al., [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618).

This historical-rating benchmark has exposure and selection bias. Offline ranking improvements do not establish online CTR, revenue, or causal effects. Main metrics cover eligible warm users/items; new users use popularity fallback and new movies need additional content modeling.
