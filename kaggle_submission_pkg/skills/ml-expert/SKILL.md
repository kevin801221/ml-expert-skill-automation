---
name: ml-expert
description: Autonomous tabular ML recipe for binary-classification AUC competitions - schema inference, leak-free CV, LightGBM-first, crash-safe submissions.
---

# ml-expert (autonomous competition mode)

- This is an unattended sandbox. Never wait for human approval. Every response must contain a tool call; plain text ends the session with zero score.
- The metric is AUC-ROC. Submit raw probabilities from predict_proba. Never tune thresholds, never output 0/1 labels.
- The sandbox is offline with preinstalled packages only. Never run pip, uv, conda, or any install command.
- Leakage rules: fit any imputer, encoder, or scaler inside each training fold only. Evaluate with out-of-fold predictions. Never evaluate a meta-learner in-sample.
- Model priority on CPU: LightGBM 5-fold with early stopping first, XGBoost rank-average blend if time remains. Skip neural networks and skip CatBoost defaults (slow on CPU).
- Time discipline: a baseline submission must exist within the first few minutes. Checkpoint predictions after every stage. Freeze all training 10 minutes before the time limit and make sure select_submission has been called.
- scripts/run_pipeline.py implements this entire recipe end-to-end (data discovery, schema inference, staged training, atomic prediction writes, crash fallback). Prefer running it over writing new code.
