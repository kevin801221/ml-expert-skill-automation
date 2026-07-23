# Tasks: Spaceship Titanic Classification (Accuracy Target > 81%)

- [ ] **Ticket-01: Dataset Verification & Ingestion**
  - Verify `benchmarks/spaceship_titanic/data/raw/Spaceship_train.csv` and `Spaceship_test.csv`.

- [ ] **Ticket-02: Feature Engineering & Imputation Pipeline (`benchmarks/spaceship_titanic/src/features.py`)**
  - Implement Cabin dissection (Deck, Side, CabinNum), GroupId, GroupSize, TotalSpending, HasSpent, LogTotalSpending.
  - Implement fold-safe imputation for missing expenditure, CryoSleep, and HomePlanet.
  - Acceptance Criteria: `pytest benchmarks/spaceship_titanic/tests/test_features.py` passes cleanly.

- [ ] **Ticket-03: PyTorch Deep Tabular Network (`benchmarks/spaceship_titanic/src/models/pytorch_net.py`)**
  - Implement `PyTorchSpaceshipNet` with Embedding layers, Dense Residual blocks, and SiLU activations.

- [ ] **Ticket-04: 5-Fold Stratified CV & Stacking Ensemble (`benchmarks/spaceship_titanic/src/train_ensemble.py`)**
  - Train PyTorch Net + LightGBM + XGBoost + CatBoost + ExtraTrees Ensemble.
  - Save native model checkpoints (.pth, .txt, .json, .cbm, .joblib) to `benchmarks/spaceship_titanic/models/`.

- [ ] **Ticket-05: Visualizations & Inference (`benchmarks/spaceship_titanic/generate_visuals.py` & `predict.py`)**
  - Generate 4-in-1 PNG chart saved to `benchmarks/spaceship_titanic/assets/spaceship_titanic_benchmark_visuals.png`.
  - Implement standalone `uv run python benchmarks/spaceship_titanic/predict.py`.
