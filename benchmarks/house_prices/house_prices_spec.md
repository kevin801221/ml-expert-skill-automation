# ML Specification: House Prices Advanced Regression Benchmark (Log RMSE < 0.120)

## 1. Problem Formulation & Benchmark Target
- **Task Type**: Tabular High-Dimensional Advanced Regression
- **Target Variable**: `SalePrice` (Continuous USD price)
- **Log Target Transformation**: `y_log = log1p(SalePrice)` to eliminate extreme right skewness.
- **Primary Metric Benchmark**: Log RMSE < 0.120 (Root Mean Squared Error on log-transformed prices) via 5-Fold Cross-Validation.

---

## 2. Environment & Technical Constitution Compliance
- **Python Runtime**: Python 3.13+ via `uv`
- **Data Source**: Kaggle API (`kaggle competitions download -c house-prices-advanced-regression-techniques`)
- **Primary Framework**: **PyTorch** (`PyTorchRegressionNet` with Dense Residual Blocks & Entity Embeddings) + Ensemble with LightGBM, XGBoost, and CatBoost Regressors.

---

## 3. Data Schema & Feature Engineering Contract
- **Raw Features**: 79 Features (36 Continuous/Discrete Numerical, 43 Nominal/Ordinal Categorical).
- **High-Signal Feature Engineering**:
  1. `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` (Total Living Area)
  2. `TotalBathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`
  3. `HouseAge = YrSold - YearBuilt`
  4. `RemodAge = YrSold - YearRemodAdd`
  5. `IsRemodeled = (YearBuilt != YearRemodAdd).astype(int)`
  6. `HasGarage`, `HasPool`, `HasBsmt`, `HasFireplace` binary flags.
  7. Log1p transformation on skewed continuous features (`GrLivArea`, `LotArea`, `1stFlrSF`).

---

## 4. Validation Strategy & Data Leakage Prevention
- **Split Mechanism**: **5-Fold K-Fold Cross-Validation** (random seed = 2026).
- **Leakage Prohibition Rule**:
  - All numerical scalers (`StandardScaler`), missing value imputers (`Median` for num, `Mode` / `'None'` for cat), and target transformers MUST fit **EXCLUSIVELY** on the training fold.

---

## 5. Model Architecture & Ensemble Stacking
1. **PyTorch Deep Tabular Regressor (`PyTorchRegressionNet`)**:
   - Entity Embeddings for categorical features (e.g., `Neighborhood`, `MSSubClass`, `ExterQual`, `KitchenQual`).
   - Dense layers with BatchNorm, SiLU activation, and Dropout (0.2).
   - MSE Loss on `log1p(SalePrice)`.
2. **Gradient Boosted Decision Trees**:
   - LightGBM Regressor (`objective='regression'`)
   - XGBoost Regressor (`eval_metric='rmse'`)
   - CatBoost Regressor (`loss_function='RMSE'`)
3. **Stacking Ensemble**:
   - Meta-learner (Ridge / ElasticNet Regressor) on Out-Of-Fold (OOF) log-price predictions.
