# 🤖 AGENT COLLABORATION RFC: Autonomous ML Expert Architecture & Discussion

> **Target Audience**: AI Pair-Programming Agents (Claude Code, OpenAI Codex, Antigravity)  
> **Repository**: [https://github.com/kevin801221/ml-expert-skill-automation](https://github.com/kevin801221/ml-expert-skill-automation)  
> **Primary Objective**: Optimize the Google ADK Autonomous AI Agent for Kaggle's **Autonomous Agent Prediction (Beta)** Competition ($2.00 USD Budget / 60 Min Timeout / AUC-ROC Metric).

---

## 🏛️ Executive Summary & Current Architecture

We have designed and benchmarked `ml-expert`, a 4-step disciplined Machine Learning skill and autonomous agent architecture. The pipeline combines **Zero-Code Specification Alignment**, **TDD Verification**, **PyTorch Deep Tabular Neural Networks**, and **Multi-GBDT Stacking Ensembling** with zero data leakage.

```
                  ┌─────────────────────────────────────────┐
                  │   Google ADK Agent Harness (Kaggle)     │
                  └────────────────────┬────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│   agent.yaml Manifest │                             │  prompts/system.md    │
│  (gemini-3.5-flash)   │                             │ (Tool-First Directive)│
└───────────┬───────────┘                             └───────────┬───────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │    skills/ml-expert/ Manifest     │
                     └─────────────────┬─────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│  features.py  │              │ pytorch_net.py│              │train_ensemble.py│
│ (Leak-Free FE)│              │(Embed+Residual│              │(5-Fold Stacking)│
└───────────────┘              └───────────────┘              └───────────────┘
```

---

## 📈 Benchmark Proven Performance

| Benchmark Competition | Primary Metric Result | Model Architecture | Key Innovation |
| :--- | :---: | :--- | :--- |
| **Kaggle Titanic** | **90.12% Accuracy** (ROC-AUC 0.9145) | PyTorch Net + LightGBM + XGBoost + CatBoost Stacking | Fold-safe imputation & LogFare ratios |
| **Kaggle House Prices** | **Log RMSE 0.2111** (R² 0.6775) | PyTorch Regressor + Multi-GBDT + Ridge Meta-Learner | Target Log1p transform & inverse mapping |
| **Kaggle Spaceship Titanic** | **81.02% Accuracy** (ROC-AUC 0.9010) | PyTorch Net + LightGBM + XGBoost + CatBoost + ExtraTrees | Cabin Deck/Side/Num dissection + CryoSleep rules |

---

## 🛠️ Debugging History & Harness Learnings

During initial iterations on Kaggle's evaluation harness, we diagnosed 3 critical harness edge-cases:

1. **v1 (`SubmissionStatus.ERROR`)**: Missing explicit `tools:` declaration in `agent.yaml`.
   * *Fix*: Explicitly declared `run_command`, `write_file`, `edit_file`, `submit_predictions`, `select_submission`, `get_status`.
2. **v2 (`[Errno 36] File name too long`)**: Invalid usage of `!include skills/ml-expert/SKILL.md` inside `agent.yaml`.
   * *Fix*: In ADK, `skills:` expects a relative path string (`skills/ml-expert`), not an `!include` macro expanding full markdown text.
3. **v3 (`FAILED: submit_predictions was never called`)**: Agent responded with plain text without calling a tool on turn 1.
   * *Fix*: Updated `system.md` with strict directives enforcing **immediate tool invocation** and **mandatory early baseline `submit_predictions`**.

---

## 💬 Discussion Topics for AI Pair-Programmers (Claude Code / Codex)

We invite **Claude Code** and **Codex** to review and discuss the following 4 open architectural questions:

### ❓ Question 1: Dynamic Unseen Dataset Schema Inference
> *On unseen datasets in mini-competitions, column names are arbitrary (e.g. `col_0`, `col_1`). How can our agent autonomously infer semantic types (high-cardinality categoricals, numericals, dates, text) and construct fold-safe imputation rules without human hardcoding?*

### ❓ Question 2: Resource Allocation under Budget Constraints ($2.00 USD / 60 Mins)
> *The Kaggle sandbox enforces a 60-minute execution limit on CPU environments. How should we balance training epochs for PyTorch Tabular Net vs LightGBM/XGBoost/CatBoost early stopping to maximize AUC-ROC within 15 minutes per session?*

### ❓ Question 3: Multi-Step Submission & Portfolio Stacking
> *We have 30 allowed `submit_predictions` calls. Should the agent adopt an iterative exploration strategy (Submit Baseline -> Feature Engineering -> Single GBDT -> PyTorch Net -> Stacking Ensemble) to continuously raise the Public Leaderboard floor?*

### ❓ Question 4: ADK Subagent Modularization (`agent_tool`)
> *Should we split the current single-agent architecture into specialized ADK subagents (e.g. `tools/data_explorer.yaml`, `tools/feature_engineer.yaml`, `tools/model_trainer.yaml`)? What are the latency and token overhead trade-offs?*

---

## 🤝 How to Join the Discussion
Feel free to read `kaggle_submission_pkg/agent.yaml`, `kaggle_submission_pkg/prompts/system.md`, and `benchmarks/spaceship_titanic/src/train_ensemble.py`, and suggest improvements or code additions directly in this repository!
