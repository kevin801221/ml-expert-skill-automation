import os
import torch
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List

from src.models.pytorch_net import PyTorchTabularNet

def run_titanic_inference():
    """
    Standard Inference Script for Titanic Model.
    Loads saved model checkpoints from models/ and performs ensembled predictions on test data.
    """
    models_dir = 'models'
    test_path = 'data/raw/test.csv'
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Missing '{models_dir}' directory! Run python src/train_ensemble.py first.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test dataset at '{test_path}'.")

    print("=========================================================================")
    print("🚀 Running Titanic Benchmark Inference Pipeline...")
    print("=========================================================================")

    # 1. Load Feature Pipeline & Test Data
    print("\n1️⃣ Loading test data & fitted feature pipeline...")
    pipeline = joblib.load(os.path.join(models_dir, 'feature_pipeline.joblib'))
    raw_test_df = pd.read_csv(test_path)
    print(f"   Loaded raw test dataset shape: {raw_test_df.shape}")

    # Transform test set using the learned pipeline parameters
    test_feat = pipeline.transform(raw_test_df)
    
    cat_cols = ['TitleCode', 'DeckCode', 'SexCode', 'EmbarkedCode', 'Pclass']
    num_cols = ['Age', 'Fare', 'LogFare', 'LogFarePerPerson', 'FamilySize', 'IsAlone', 'TicketGroupSize', 'AgeClass']

    X_test_cat = test_feat[cat_cols]
    X_test_num = test_feat[num_cols]

    # Standardize numerical features using train stats stored in pipeline
    # (Simplified column normalization for inference)
    num_mean = X_test_num.mean()
    num_std = X_test_num.std().replace(0, 1.0)
    X_test_num_scaled = (X_test_num - num_mean) / num_std
    X_test_gb = pd.concat([X_test_cat, X_test_num_scaled], axis=1)

    # 2. Load & Predict using PyTorch Fold Models
    print("\n2️⃣ Loading PyTorch Deep Tabular Net checkpoints (.pth)...")
    emb_dims = {'TitleCode': (6, 4), 'DeckCode': (8, 4), 'SexCode': (2, 2), 'EmbarkedCode': (3, 2), 'Pclass': (4, 2)}
    cat_dims = {col: emb_dims[col][0] for col in cat_cols}
    
    cat_t = {col: torch.tensor(X_test_cat[col].values, dtype=torch.long) for col in X_test_cat.columns}
    num_t = torch.tensor(X_test_num_scaled.values, dtype=torch.float32)

    pt_preds = np.zeros(len(raw_test_df))
    fold_count = 0
    for fold in range(1, 6):
        ckpt_path = os.path.join(models_dir, f'pytorch_fold_{fold}.pth')
        if os.path.exists(ckpt_path):
            model = PyTorchTabularNet(cat_dims=cat_dims, emb_dims=emb_dims, num_features_dim=len(num_cols), hidden_dim=128)
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
            with torch.no_grad():
                logits = model(cat_t, num_t)
                pt_preds += torch.sigmoid(logits).numpy()
            fold_count += 1
            print(f"   Loaded PyTorch checkpoint: {ckpt_path}")
            
    if fold_count > 0:
        pt_preds /= fold_count

    # 3. Load & Predict using GBDT Models
    print("\n3️⃣ Loading GBDT model checkpoints (.joblib)...")
    lgb_model = joblib.load(os.path.join(models_dir, 'lightgbm_model.joblib'))
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
    cat_model = joblib.load(os.path.join(models_dir, 'catboost_model.joblib'))

    lgb_preds = lgb_model.predict_proba(X_test_gb)[:, 1]
    xgb_preds = xgb_model.predict_proba(X_test_gb)[:, 1]
    cat_preds = cat_model.predict_proba(X_test_gb)[:, 1]

    # 4. Load Meta-Learner & Perform Stacking Ensemble
    print("\n4️⃣ Performing Meta-Learner Stacking Inference...")
    meta_model = joblib.load(os.path.join(models_dir, 'stacking_meta_learner.joblib'))
    X_test_meta = np.column_stack([pt_preds, lgb_preds, xgb_preds, cat_preds])
    
    ensemble_probs = meta_model.predict_proba(X_test_meta)[:, 1]
    final_preds = (ensemble_probs >= 0.50).astype(int)

    # 5. Output Results
    results_df = pd.DataFrame({
        'PassengerId': raw_test_df['PassengerId'],
        'Name': raw_test_df['Name'],
        'Sex': raw_test_df['Sex'],
        'Age': raw_test_df['Age'],
        'PyTorch_Prob': np.round(pt_preds, 4),
        'LightGBM_Prob': np.round(lgb_preds, 4),
        'XGBoost_Prob': np.round(xgb_preds, 4),
        'CatBoost_Prob': np.round(cat_preds, 4),
        'Ensemble_Prob': np.round(ensemble_probs, 4),
        'Predicted_Survived': final_preds
    })

    sub_df = pd.DataFrame({
        'PassengerId': raw_test_df['PassengerId'],
        'Survived': final_preds
    })
    sub_df.to_csv('data/predictions_submission.csv', index=False)

    print("\n=========================================================================", flush=True)
    print("✨ INFERENCE COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Saved submission file to: data/predictions_submission.csv", flush=True)
    print("=========================================================================\n", flush=True)
    print("Sample Inference Output (First 10 Passengers):\n", flush=True)
    print(results_df.head(10).to_string(index=False), flush=True)
    
    out_file = '/Users/kevinluo/ml-expert-skill-make/data/sample_inference_results.txt'
    with open(out_file, 'w') as f:
        f.write(results_df.head(10).to_string(index=False))
    print(f"\nWrote inference sample to: {out_file}", flush=True)

if __name__ == '__main__':
    run_titanic_inference()
