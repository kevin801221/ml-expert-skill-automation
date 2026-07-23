# ML Specification: Titanic Survival Prediction Benchmark (>90% Accuracy Target)

## 1. Problem Formulation & Benchmark Target
- **Task Type**: Tabular Binary Classification
- **Target Variable**: `Survived` (0 = Deceased, 1 = Survived)
- **Primary Metric Benchmark**: Accuracy > 0.900 (90%) on 5-Fold Stratified Cross-Validation
- **Secondary Evaluation Metrics**: ROC-AUC, F1-Score, Confusion Matrix Analysis

---

## 2. Environment & Technical Constitution Compliance
- **Python Runtime**: Python 3.13+
- **Environment & Package Manager**: `uv` (`uv venv` and `uv pip install`)
- **Data Source**: Official Kaggle Competition API (`kaggle competitions download -c titanic`)
- **Primary Framework**: **PyTorch** (PyTorch Deep Neural Network with Embeddings / Dense Blocks) + Ensemble with Gradient Boosters (LightGBM/XGBoost/CatBoost).

---

## 3. Data Schema & Input Contract
- **Raw Features**:
  - `PassengerId` (Identifier, drop from features)
  - `Pclass` (Categorical, Ordinal 1, 2, 3)
  - `Name` (String - Source for Title extraction)
  - `Sex` (Categorical - male, female)
  - `Age` (Continuous - Missing ~20%)
  - `SibSp` (Numerical - Number of siblings/spouses aboard)
  - `Parch` (Numerical - Number of parents/children aboard)
  - `Ticket` (String - Ticket group code / prefix)
  - `Fare` (Continuous - Missing in test set)
  - `Cabin` (String - Deck level, Missing ~77%)
  - `Embarked` (Categorical - Port of embarkation: C, Q, S, Missing 2 values)

---

## 4. Feature Engineering Strategy (High Signal for >90% Accuracy)
To breach the ~80% baseline barrier and target >90% Accuracy, the feature engineering pipeline must construct:

1. **Title Extraction & Social Grouping**:
   - Extract titles from `Name` (`Mr`, `Mrs`, `Miss`, `Master`, `Dr`, `Rev`, `Col`, `Major`, `Lady`, `Sir`, `Countess`, etc.).
   - Group rare titles into unified categories (`Officer`, `Royalty`, `Special`).
2. **Family Survival Signal & Size Dynamics**:
   - `FamilySize = SibSp + Parch + 1`
   - `IsAlone` binary boolean indicator.
   - Extract `Surname` from `Name` to identify family groups and compute `FamilySurvivalRate` within folds.
3. **Ticket Frequency & Ticket Prefix Analysis**:
   - Group passengers with identical `Ticket` numbers to infer shared travel parties.
   - `TicketGroupSize = Count(Ticket)`
   - `FarePerPerson = Fare / TicketGroupSize`
4. **Deck & Structural Compartment**:
   - Extract `Deck` from `Cabin` first letter (`A`, `B`, `C`, `D`, `E`, `F`, `G`, `T`, `Unknown`).
5. **Age & Fare Imputation & Binned Representations**:
   - Impute missing `Age` using median within `(Title, Pclass)` groups.
   - Impute missing `Fare` using median of `(Pclass, Embarked)`.
   - Log-transform `Fare` to handle heavy skewness (`log1p(Fare)`).

---

## 5. Validation Strategy & Data Leakage Prevention
- **Split Mechanism**: **5-Fold Stratified K-Fold Cross-Validation** (random seed = 42).
- **Leakage Prohibition Rule**:
  - All feature transformers (StandardScaler, OneHotEncoder, Target Encoding, Median Imputers) MUST be fit **EXCLUSIVELY** on the training fold.
  - Test and validation folds must ONLY use `.transform()` derived from the training fold parameters.

---

## 6. PyTorch & Model Ensemble Architecture
1. **PyTorch Deep Tabular Network (`PyTorchTabularNet`)**:
   - Embedding layers for categorical features (`Title`, `Deck`, `Sex`, `Embarked`, `Pclass`).
   - Dense Batch Normalization + Linear + SiLU/ReLU + Dropout layers for numerical features.
   - Residual Skip Connections across dense layers.
   - Binary Cross-Entropy Loss with Pos-Weight tuning / Focal Loss.
   - Optimizer: AdamW (lr=1e-3, weight_decay=1e-2) with Cosine Annealing Learning Rate Scheduler.
2. **Ensemble & Stacking Model**:
   - Ensemble PyTorch Neural Net with LightGBM, XGBoost, and CatBoost.
   - Meta-learner (Logistic Regression or Soft Voting) trained on Out-Of-Fold (OOF) predictions.

---

## 7. Acceptance Criteria for Step 2
- Specification document `ML_SPEC.md` created with ZERO Python code.
- User review and approval received before proceeding to Step 3 (`/ml-tickets`).
