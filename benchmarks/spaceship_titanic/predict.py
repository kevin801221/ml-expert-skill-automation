import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from benchmarks.spaceship_titanic.src.models.pytorch_net import PyTorchSpaceshipNet

def run_spaceship_inference():
    models_dir = 'benchmarks/spaceship_titanic/models'
    test_path = 'benchmarks/spaceship_titanic/data/raw/Spaceship_train.csv'

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Missing '{models_dir}' directory! Run train_ensemble.py first.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test dataset at '{test_path}'.")

    print("=========================================================================", flush=True)
    print("🚀 Running Spaceship Titanic Benchmark Inference Pipeline...", flush=True)
    print("=========================================================================", flush=True)

    # 1. Load Feature Pipeline & Test Data Sample
    print("\n1️⃣ Loading test dataset & fitted feature pipeline...", flush=True)
    pipeline = joblib.load(os.path.join(models_dir, 'feature_pipeline.joblib'))
    raw_test_df = pd.read_csv(test_path).tail(100).copy()  # Take held-out 100 sample passengers
    print(f"   Loaded sample test dataset shape: {raw_test_df.shape}", flush=True)

    test_feat = pipeline.transform(raw_test_df)

    cat_cols = ['HomePlanetCode', 'DestinationCode', 'DeckCode', 'SideCode', 'CryoSleepCode', 'VIPCode', 'IsAloneGroup', 'HasSpent']
    num_cols = ['Age', 'GroupSize', 'CabinNum', 'TotalSpending', 'LogTotalSpending', 'Log_RoomService', 'Log_FoodCourt', 'Log_ShoppingMall', 'Log_Spa', 'Log_VRDeck']

    X_test_cat = test_feat[cat_cols]
    X_test_num = test_feat[num_cols]

    num_mean = X_test_num.mean()
    num_std = X_test_num.std().replace(0, 1.0)
    X_test_num_scaled = (X_test_num - num_mean) / num_std
    X_test_gb = pd.concat([X_test_cat, X_test_num_scaled], axis=1)

    # 2. PyTorch Model Predictions
    print("\n2️⃣ Loading PyTorch Deep Net checkpoints (.pth)...", flush=True)
    emb_dims = {
        'HomePlanetCode': (4, 2), 'DestinationCode': (4, 2), 'DeckCode': (9, 4),
        'SideCode': (3, 2), 'CryoSleepCode': (2, 2), 'VIPCode': (2, 2),
        'IsAloneGroup': (2, 2), 'HasSpent': (2, 2)
    }
    cat_dims = {col: emb_dims[col][0] for col in cat_cols}

    cat_t = {col: torch.tensor(X_test_cat[col].values, dtype=torch.long) for col in X_test_cat.columns}
    num_t = torch.tensor(X_test_num_scaled.values, dtype=torch.float32)

    pt_preds = np.zeros(len(raw_test_df))
    fold_count = 0
    for fold in range(1, 6):
        ckpt_path = os.path.join(models_dir, f'pytorch_fold_{fold}.pth')
        if os.path.exists(ckpt_path):
            model = PyTorchSpaceshipNet(cat_dims=cat_dims, emb_dims=emb_dims, num_dim=len(num_cols), hidden_dim=128)
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
            with torch.no_grad():
                logits = model(cat_t, num_t)
                pt_preds += torch.sigmoid(logits).numpy()
            fold_count += 1
            print(f"   Loaded PyTorch checkpoint: {ckpt_path}", flush=True)

    if fold_count > 0:
        pt_preds /= fold_count

    # 3. Native GBDT C++ Model Predictions
    print("\n3️⃣ Loading Native GBDT C++ Checkpoints...", flush=True)
    lgb_booster = lgb.Booster(model_file=os.path.join(models_dir, 'lightgbm_model.txt'))
    lgb_preds = lgb_booster.predict(X_test_gb)

    xgb_booster = xgb.Booster()
    xgb_booster.load_model(os.path.join(models_dir, 'xgboost_model.json'))
    dtest = xgb.DMatrix(X_test_gb)
    xgb_preds = xgb_booster.predict(dtest)

    cat_model = CatBoostClassifier()
    cat_model.load_model(os.path.join(models_dir, 'catboost_model.cbm'))
    cat_preds = cat_model.predict_proba(X_test_gb)[:, 1]

    et_model = joblib.load(os.path.join(models_dir, 'extratrees_model.joblib'))
    et_preds = et_model.predict_proba(X_test_gb)[:, 1]

    # 4. Meta Stacking Inference
    print("\n4️⃣ Performing Meta-Learner Stacking Inference...", flush=True)
    meta = joblib.load(os.path.join(models_dir, 'stacking_meta_learner.joblib'))
    X_test_meta = np.column_stack([pt_preds, lgb_preds, xgb_preds, cat_preds, et_preds])

    ensemble_probs = meta.predict_proba(X_test_meta)[:, 1]
    final_preds = (ensemble_probs >= 0.50).astype(bool)

    results_df = pd.DataFrame({
        'PassengerId': raw_test_df['PassengerId'],
        'HomePlanet': raw_test_df['HomePlanet'],
        'CryoSleep': raw_test_df['CryoSleep'],
        'Cabin': raw_test_df['Cabin'],
        'PyTorch_Prob': np.round(pt_preds, 4),
        'LightGBM_Prob': np.round(lgb_preds, 4),
        'XGBoost_Prob': np.round(xgb_preds, 4),
        'CatBoost_Prob': np.round(cat_preds, 4),
        'Ensemble_Prob': np.round(ensemble_probs, 4),
        'Predicted_Transported': final_preds
    })

    print("\n=========================================================================", flush=True)
    print("✨ SPACESHIP TITANIC INFERENCE COMPLETED SUCCESSFULLY!", flush=True)
    print("=========================================================================\n", flush=True)
    print("Sample Inference Predictions (First 10 Passengers):\n", flush=True)
    print(results_df.head(10).to_string(index=False), flush=True)

if __name__ == '__main__':
    run_spaceship_inference()
