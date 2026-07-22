# Tasks: House Prices Benchmark (Log RMSE < 0.120)

- [ ] **Ticket-01: Kaggle Dataset Download via API**
  - Command: `kaggle competitions download -c house-prices-advanced-regression-techniques -p benchmarks/house_prices/data/raw`
  - Unzip into `benchmarks/house_prices/data/raw/` (`train.csv`, `test.csv`).

- [ ] **Ticket-02: Feature Engineering & Imputation Pipeline (`benchmarks/house_prices/src/features.py`)**
  - Implement TotalSF, TotalBathrooms, HouseAge, RemodAge, IsRemodeled, and Log1p transforms.
  - Implement fit/transform pipeline to handle categorical 'None' values and numerical medians.
  - Acceptance Criteria: Pytest assertions in `benchmarks/house_prices/tests/test_features.py` pass cleanly.

- [ ] **Ticket-03: PyTorch Deep Regression Network (`benchmarks/house_prices/src/models/pytorch_regressor.py`)**
  - Implement PyTorch Tabular Regressor with Entity Embeddings, Dense Residual blocks, and MSE Loss.

- [ ] **Ticket-04: 5-Fold Cross-Validation & Stacking Ensemble (`benchmarks/house_prices/src/train_ensemble.py`)**
  - Execute 5-Fold CV training across PyTorch Net + LightGBM + XGBoost + CatBoost.
  - Train Ridge Meta-Learner on OOF log predictions.

- [ ] **Ticket-05: High-Res Evaluation Visualizations (`benchmarks/house_prices/generate_visuals.py`)**
  - Generate 4-in-1 chart: Log RMSE comparison, Predicted vs Actual Scatter Plot, Residual Plot, Feature Importance.
  - Acceptance Criteria: Output PNG image saved to `benchmarks/house_prices/assets/house_prices_benchmark_visuals.png`.
