# ML Specification: Spaceship Titanic Classification Benchmark (Accuracy Target > 81%)

## 1. Problem Formulation & Benchmark Target
- **Task Type**: Tabular Binary Classification (Futuristic Sci-Fi Dataset)
- **Target Variable**: `Transported` (True / False)
- **Primary Metric Benchmark**: Accuracy > 0.810 (81%) on 5-Fold Stratified Cross-Validation
- **Secondary Evaluation Metrics**: ROC-AUC, F1-Score, Confusion Matrix Analysis

---

## 2. Environment & Technical Constitution Compliance
- **Python Runtime**: Python 3.13+ via `uv`
- **Data Source**: Kaggle Competition API (`kaggle datasets download -d jyothishri/spaceship-titanic`)
- **Primary Framework**: **PyTorch** (`PyTorchSpaceshipNet` with Embeddings & Residual Skip Connections) + Ensemble with LightGBM, XGBoost, CatBoost, and ExtraTrees.

---

## 3. Data Schema & Feature Engineering Strategy
- **Raw Features**:
  - `PassengerId` (Format `gggg_pp` -> Group ID `gggg` & Group Member `pp`)
  - `HomePlanet` (Categorical: Earth, Europa, Mars)
  - `CryoSleep` (Binary Boolean: True/False)
  - `Cabin` (Format `Deck/Num/Side` -> e.g. `B/0/P` or `F/0/S`)
  - `Destination` (Categorical: TRAPPIST-1e, 55 Cancri e, PSO J318.5-22)
  - `Age` (Continuous)
  - `VIP` (Binary Boolean: True/False)
  - Amenities Expenditure (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`)
  - `Name` (String -> Extract `Surname` for family grouping)

- **High-Signal Engineered Features**:
  1. **Group Dynamics**:
     - `GroupId = PassengerId.split('_')[0]`
     - `GroupSize = Count(GroupId)`
     - `IsAloneGroup = (GroupSize == 1).astype(int)`
  2. **Cabin Dissection**:
     - `Deck` (A, B, C, D, E, F, G, T, Unknown)
     - `Side` (P = Port, S = Starboard, Unknown)
     - `CabinNum` (Extracted integer ordinal)
  3. **Amenity Spending Signals**:
     - `TotalSpending = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck`
     - `HasSpent = (TotalSpending > 0).astype(int)`
     - `LogTotalSpending = log1p(TotalSpending)`
     - Expenditure Ratios (`RoomService / (TotalSpending + 1)`, etc.)
  4. **CryoSleep Interaction Rule**:
     - If `CryoSleep == True`, set expenditure features to 0.

---

## 4. Validation Strategy & Data Leakage Prevention
- **Split Mechanism**: **5-Fold Stratified K-Fold Cross-Validation** (random seed = 2026).
- **Leakage Prohibition Rule**:
  - Imputation medians (e.g., spending per CryoSleep/HomePlanet group) and scaling parameters MUST be fitted **exclusively within each training fold**.

---

## 5. Model Architecture & Ensemble Stacking
1. **PyTorch Deep Tabular Net (`PyTorchSpaceshipNet`)**:
   - Embeddings for Categoricals (`Deck`, `Side`, `HomePlanet`, `Destination`, `CryoSleep`, `VIP`, `IsAloneGroup`).
   - Dense Batch Normalization + SiLU + Dropout (0.25) + Residual Blocks.
   - Binary Cross-Entropy Loss with AdamW & Cosine Annealing.
2. **Gradient Boosted Decision Trees & Stacking**:
   - Ensemble PyTorch Net + LightGBM + XGBoost + CatBoost + ExtraTrees Classifier.
   - Meta-learner (Logistic Regression) on Out-Of-Fold (OOF) prediction probabilities.
