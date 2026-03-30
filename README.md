# 🦾 The Apex Data Cleaner

### Meta PyTorch Hackathon 2026 | OpenEnv Track

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Framework-6E40C9?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

---

## 🎬 2-Minute Demo

**[▶ Watch the 2-Minute Architecture & UI Demo Here](LINK)**

---

## 🎯 The Mission

### Project Description

We are building the **Apex Data Cleaner** — an interactive RL environment that simulates a Junior Data Engineer's workflow. It trains AI agents to autonomously ingest corrupted, messy datasets and execute precise data wrangling operations until the data is machine-learning ready.

The agent observes a **profile** of the dataset (row counts, missing values, data types) and takes deterministic, Pydantic-enforced actions — such as `impute_mean`, `drop_null_rows`, and `one_hot_encode` — on specific columns. The environment's grader physically trains a baseline Machine Learning model on the agent's output; the agent's final score (0.0 to 1.0) is **directly tied to the predictive accuracy of the resulting ML pipeline**.

### Why We Selected This Problem

In any rigorous data science environment — whether an enterprise startup or an advanced academic curriculum — **80% of a practitioner's time is wasted wrangling messy data**. We chose this problem because autonomous data cleaning is the ultimate bottleneck in modern AI development.

Existing RL environments mostly train agents to browse the web or play games. By forcing an agent to understand column types, handle extreme outliers, and format data for scikit-learn compatibility, we are providing **immediate, massive value to the agentic AI community**. This environment tests an LLM's true logical reasoning and its ability to write safe, non-destructive data pipelines.

---

## 🏗️ Architecture

> **The pipeline:** `Dirty CSV → JSON Profile → RL Agent → Pydantic Actions → Clean CSV → scikit-learn Grader → Reward`

<!-- Embed your architecture diagram below once exported -->
<!-- ![Architecture Diagram](./assets/architecture_diagram.png) -->

### 1. 🔭 Observation — What the Agent Sees

At each step, the agent receives a **structured JSON dataset profile** as its observation — never the raw CSV directly. This profile includes:

```json
{
  "row_count": 100,
  "columns": {
    "Revenue_M": { "dtype": "float", "null_rate": 0.20, "mean": 124.3, "min": 0.0001, "max": 99999.99 },
    "Industry":  { "dtype": "string", "null_rate": 0.0, "unique_values": 14, "sample": ["tech", "SaaS ", "Fintceh"] },
    "JUNK_COL":  { "dtype": "string", "null_rate": 0.0, "unique_values": 100, "entropy": 4.99 }
  }
}
```

This forces the agent to reason about the data the same way a real data engineer would when handed an unfamiliar file.

### 2. ⚙️ Action Space — What the Agent Can Do

The agent selects from a **discrete set of Pydantic-enforced Pandas operations**. Strict Pydantic validation ensures every action is well-formed before it touches the dataframe — eliminating hallucinations, silent failures, and partial writes.

| Action | Parameters | Effect |
|---|---|---|
| `impute_mean` | `column: str` | Fill nulls with column mean |
| `impute_median` | `column: str` | Fill nulls with column median |
| `drop_null_rows` | `column: str` | Drop rows where column is null |
| `normalize_text` | `column: str, strategy: str` | Standardize casing / strip whitespace |
| `clip_outliers` | `column: str, lower: float, upper: float` | Cap extreme values |
| `drop_column` | `column: str` | Remove irrelevant feature |
| `one_hot_encode` | `column: str` | Encode categorical column |

### 3. 🏆 The Grader — How Reward is Computed

After the agent applies its cleaning operations, the cleaned dataset is passed to a **deterministic scikit-learn grading model** — a Random Forest classifier that attempts to predict the target label (`Is_Profitable`) using the agent-cleaned features.

The reward signal is unambiguous and 100% reproducible:

| Agent Cleaning Quality | ML Accuracy | Reward |
|---|---|---|
| Perfect clean | ~90%+ accuracy | **1.0** |
| Partially clean | ~60–80% accuracy | **0.3 – 0.79** |
| Made it worse | < 50% accuracy | **0.0** |

> **The key insight:** because the grader is deterministic, the reward cannot be gamed. The agent can only earn a high reward by **genuinely improving data quality**. An 88% ML accuracy yields exactly a `0.88` reward score.

---

## 📂 The Datasets

All task files live in the `/tasks` folder. They are derived from **real-world Kaggle startup funding and SaaS churn data** (100 rows, 13 columns), with **synthetic corruption injected at three severity levels** to benchmark the agent across a meaningful difficulty curve.

- **`task_easy.csv`** — The warm-up. The dataset is nearly pristine. `Revenue_M` values are deleted for 5 randomly chosen startups. The agent must detect and impute 5 missing numerical values without disturbing anything else. A strong baseline agent should score near-perfect here.

- **`task_medium.csv`** — A realistic mess. Roughly 8 values are nulled out across five numerical columns (`Total_Funding_M`, `Revenue_M`, `Employee_Count`, `ARPU`, `Monthly_Churn_Rate`). The `Industry` text column has been deliberately corrupted with inconsistent casing and typos — e.g., `"tech"`, `"Technolgy"`, `"Fin Tech"`, `"SaaS "` (trailing space). The agent must handle both numerical imputation and categorical normalization in a single episode.

- **`task_hard.csv`** — Controlled chaos. Four extreme statistical outliers have been injected (e.g., `Employee_Count = 999,999`; `Monthly_Churn_Rate = 999.9`). A fully useless `JUNK_COL` — packed with random alphanumeric noise — has been appended to test whether the agent can identify and prune irrelevant features. Finally, 20% of all numerical values across eight columns have been randomly deleted. This task demands the agent's full repertoire: outlier clipping, column pruning, and large-scale imputation — simultaneously.

---

## ✅ Hackathon Rubric — Our Verdict

| Criterion | Weight | Status | Rationale |
|---|---|---|---|
| **Real-world utility** | 30% | ✅ PASS | Models a genuine Junior Data Engineer workflow — a high-value, daily tech task |
| **Task & grader quality** | 25% | ✅ PASS | Three-tier difficulty curve; 100% deterministic reward tied to ML accuracy |
| **Environment design** | 20% | ✅ PASS | Pydantic action space eliminates hallucination; clean state transitions |
| **Code quality & spec compliance** | 15% | 🔧 ON US | FastAPI endpoints + Docker build = full points |
| **Creativity & novelty** | 10% | ✅ PASS | Tying RL reward to downstream ML performance (Auto-ML readiness) is novel |

---

## 💻 Local Installation

```bash
# Terminal commands to run the project locally will go here.
# This section will be completed once the backend is finished.
```

---

## 📁 Repository Structure

```
apex-data-cleaner/
├── tasks/
│   ├── task_easy.csv          ← 5 missing Revenue values
│   ├── task_medium.csv        ← missing numerics + misspelled Industry
│   └── task_hard.csv          ← outliers + junk column + 20% missing
├── assets/
│   └── architecture_diagram.png
├── env/                       ← OpenEnv environment definition
├── agent/                     ← PyTorch RL agent
├── grader/                    ← scikit-learn deterministic grader
├── api/                       ← FastAPI backend
├── Dockerfile
├── generate_datasets.py
└── README.md
```

---

## 🔖 Recommended Commit Style

Keep your Git history clean and professional using [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat: scaffold OpenEnv environment with Pydantic action space
feat: implement scikit-learn deterministic grader with 0.0-1.0 reward
feat: add three-tier synthetic dataset corruption pipeline
fix: handle edge case in outlier clipper for zero-variance columns
docs: draft README with architecture diagram and rubric breakdown
chore: add Dockerfile for containerised environment deployment
refactor: extract Pydantic action schemas to dedicated module
```

---

## 📜 License

MIT © 2026 Apex Predators
