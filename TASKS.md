# Tasks: Titanic Benchmark (>90% Accuracy Target)

This document tracks the tracer-bullet execution tickets derived from `ML_SPEC.md`.

---

## Phase 1: Environment & Dataset Acquisition
- [ ] **Ticket-01: Environment Setup via `uv`**
  - Create virtual environment: `uv venv --python 3.13`
  - Install dependencies: `uv pip install torch pandas scikit-learn lightgbm xgboost catboost pytest kaggle optuna`
  - Acceptance Criteria: Virtual environment active, PyTorch & Kaggle CLI verified.

- [ ] **Ticket-02: Kaggle Dataset Download**
  - Command: `kaggle competitions download -c titanic`
  - Extract files to `data/raw/` (`train.csv`, `test.csv`, `gender_submission.csv`).
  - Acceptance Criteria: `train.csv` and `test.csv` exist in `data/raw/`.

---

## Phase 2: Test-Driven Data Pipeline & Feature Engineering
- [ ] **Ticket-03: Data Processing & Feature Engineering Pipeline (`src/features.py`)**
  - Implement Title extraction, FamilySize, IsAlone, FarePerPerson, TicketGroupSize, and Deck extraction.
  - Implement `fit` / `transform` pipeline to handle missing Age, Fare, and Embarked values.
  - Acceptance Criteria: `tests/test_features.py` passes all pytest assertions for shape, missing values, and zero data leakage across folds.

- [ ] **Ticket-04: Stratified 5-Fold Validation Framework (`src/validation.py`)**
  - Implement 5-Fold Stratified K-Fold generator.
  - Ensure feature transformations fit on train fold ONLY.
  - Acceptance Criteria: `tests/test_validation.py` verifies zero leakage between fold training and validation sets.

---

## Phase 3: Model Architecture & Training Loop
- [ ] **Ticket-05: PyTorch Deep Tabular Model (`src/models/pytorch_net.py`)**
  - Implement `PyTorchTabularNet` with Embedding layers for categoricals, BatchNorm, Linear, Dropout, and Residual connections.
  - Implement PyTorch Training Loop (`src/train_pytorch.py`) with BCE Loss, AdamW, Cosine Annealing, and Early Stopping.
  - Acceptance Criteria: `tests/test_model.py` checks model forward pass output shape `(batch_size, 1)` and training convergence on dummy batch.

- [ ] **Ticket-06: Gradient Boosting Baseline Models (`src/models/boosters.py`)**
  - Implement LightGBM, XGBoost, and CatBoost fold trainers.
  - Acceptance Criteria: Record OOF predictions for all 3 booster models.

---

## Phase 4: Ensembling, Optimization & Benchmark Verification
- [ ] **Ticket-07: Ensemble Stacking & Target Accuracy Check (`src/evaluate.py`)**
  - Implement Meta-Learner / Soft Voting Ensemble combining PyTorch Net + LightGBM + XGBoost + CatBoost.
  - Run 5-Fold CV evaluation, output Confusion Matrix, ROC-AUC, and OOF Accuracy.
  - Acceptance Criteria: OOF Validation Accuracy achieves target benchmark (>90%).
