import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from benchmarks.spaceship_titanic.src.features import SpaceshipFeaturePipeline
from benchmarks.spaceship_titanic.src.models.pytorch_net import PyTorchSpaceshipNet

def seed_everything(seed=2026):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_pytorch_spaceship_fold(train_x_cat, train_x_num, train_y, val_x_cat, val_x_num, val_y, cat_dims, emb_dims, num_dim, epochs=120, batch_size=64, lr=1e-3, fold_idx: int = None) -> np.ndarray:
    train_cat_t = {col: torch.tensor(train_x_cat[col].values, dtype=torch.long) for col in train_x_cat.columns}
    train_num_t = torch.tensor(train_x_num.values, dtype=torch.float32)
    train_y_t = torch.tensor(train_y.values, dtype=torch.float32)

    val_cat_t = {col: torch.tensor(val_x_cat[col].values, dtype=torch.long) for col in val_x_cat.columns}
    val_num_t = torch.tensor(val_x_num.values, dtype=torch.float32)

    model = PyTorchSpaceshipNet(cat_dims=cat_dims, emb_dims=emb_dims, num_dim=num_dim, hidden_dim=128, dropout_rate=0.25)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset_size = len(train_y)
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_cat = {col: train_cat_t[col][indices] for col in train_cat_t}
            batch_num = train_num_t[indices]
            batch_y = train_y_t[indices]

            optimizer.zero_grad()
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    models_dir = '/Users/kevinluo/ml-expert-skill-make/benchmarks/spaceship_titanic/models'
    os.makedirs(models_dir, exist_ok=True)
    if fold_idx is not None:
        torch.save(model.state_dict(), os.path.join(models_dir, f'pytorch_fold_{fold_idx+1}.pth'))

    with torch.no_grad():
        val_logits = model(val_cat_t, val_num_t)
        probs = torch.sigmoid(val_logits).numpy()
    return probs

def main():
    seed_everything(2026)
    data_path = 'benchmarks/spaceship_titanic/data/raw/Spaceship_train.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}!")

    df = pd.read_csv(data_path)
    print(f"Loaded Spaceship Titanic dataset with shape: {df.shape}")

    cat_cols = ['HomePlanetCode', 'DestinationCode', 'DeckCode', 'SideCode', 'CryoSleepCode', 'VIPCode', 'IsAloneGroup', 'HasSpent']
    num_cols = ['Age', 'GroupSize', 'CabinNum', 'TotalSpending', 'LogTotalSpending', 'Log_RoomService', 'Log_FoodCourt', 'Log_ShoppingMall', 'Log_Spa', 'Log_VRDeck']

    emb_dims = {
        'HomePlanetCode': (4, 2),
        'DestinationCode': (4, 2),
        'DeckCode': (9, 4),
        'SideCode': (3, 2),
        'CryoSleepCode': (2, 2),
        'VIPCode': (2, 2),
        'IsAloneGroup': (2, 2),
        'HasSpent': (2, 2)
    }
    cat_dims = {col: emb_dims[col][0] for col in cat_cols}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    
    pipeline_init = SpaceshipFeaturePipeline()
    pipeline_init.fit(df)
    transformed_init = pipeline_init.transform(df)
    target = transformed_init['Target'].values

    oof_pt = np.zeros(len(df))
    oof_lgb = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    oof_cat = np.zeros(len(df))
    oof_et = np.zeros(len(df))

    print("\n--- Starting 5-Fold Stratified Cross-Validation Training ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, target)):
        print(f"\n>>> Executing Fold {fold+1} / 5 <<<")
        train_raw = df.iloc[train_idx].copy()
        val_raw = df.iloc[val_idx].copy()

        pipeline = SpaceshipFeaturePipeline()
        pipeline.fit(train_raw)

        train_feat = pipeline.transform(train_raw)
        val_feat = pipeline.transform(val_raw)

        X_train_cat, X_train_num, y_train = train_feat[cat_cols], train_feat[num_cols], train_feat['Target']
        X_val_cat, X_val_num, y_val = val_feat[cat_cols], val_feat[num_cols], val_feat['Target']

        num_mean = X_train_num.mean()
        num_std = X_train_num.std().replace(0, 1.0)
        X_train_num_scaled = (X_train_num - num_mean) / num_std
        X_val_num_scaled = (X_val_num - num_mean) / num_std

        # 1. PyTorch Tabular Net
        print("Training PyTorch Spaceship Net...")
        probs_pt = train_pytorch_spaceship_fold(
            X_train_cat, X_train_num_scaled, y_train,
            X_val_cat, X_val_num_scaled, y_val,
            cat_dims, emb_dims, len(num_cols), fold_idx=fold
        )
        oof_pt[val_idx] = probs_pt

        X_tr_gb = pd.concat([X_train_cat, X_train_num_scaled], axis=1)
        X_va_gb = pd.concat([X_val_cat, X_val_num_scaled], axis=1)

        # 2. LightGBM
        lgb = LGBMClassifier(n_estimators=180, learning_rate=0.03, num_leaves=15, max_depth=4, subsample=0.8, random_state=2026, verbose=-1, n_jobs=1)
        lgb.fit(X_tr_gb, y_train)
        oof_lgb[val_idx] = lgb.predict_proba(X_va_gb)[:, 1]

        # 3. XGBoost
        xgb = XGBClassifier(n_estimators=180, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=2026, eval_metric='logloss', n_jobs=1)
        xgb.fit(X_tr_gb, y_train)
        oof_xgb[val_idx] = xgb.predict_proba(X_va_gb)[:, 1]

        # 4. CatBoost
        cat = CatBoostClassifier(iterations=220, learning_rate=0.03, depth=4, random_seed=2026, verbose=0, thread_count=1)
        cat.fit(X_tr_gb, y_train)
        oof_cat[val_idx] = cat.predict_proba(X_va_gb)[:, 1]

        # 5. ExtraTrees
        et = ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=2026, n_jobs=1)
        et.fit(X_tr_gb, y_train)
        oof_et[val_idx] = et.predict_proba(X_va_gb)[:, 1]

    # Model Stacking Ensemble
    X_meta = np.column_stack([oof_pt, oof_lgb, oof_xgb, oof_cat, oof_et])
    meta = LogisticRegression(random_state=2026)
    meta.fit(X_meta, target)

    ensemble_probs = meta.predict_proba(X_meta)[:, 1]

    best_thresh = 0.5
    best_acc = 0.0
    for thresh in np.arange(0.35, 0.65, 0.005):
        acc = accuracy_score(target, (ensemble_probs >= thresh).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    ensemble_preds = (ensemble_probs >= best_thresh).astype(int)
    final_acc = accuracy_score(target, ensemble_preds)
    final_auc = roc_auc_score(target, ensemble_probs)

    print("\n================== Out-of-Fold Model Performance ==================")
    print(f"PyTorch Net OOF Accuracy: {accuracy_score(target, (oof_pt >= 0.5).astype(int)):.4f}")
    print(f"LightGBM    OOF Accuracy: {accuracy_score(target, (oof_lgb >= 0.5).astype(int)):.4f}")
    print(f"XGBoost     OOF Accuracy: {accuracy_score(target, (oof_xgb >= 0.5).astype(int)):.4f}")
    print(f"CatBoost    OOF Accuracy: {accuracy_score(target, (oof_cat >= 0.5).astype(int)):.4f}")
    print(f"ExtraTrees  OOF Accuracy: {accuracy_score(target, (oof_et >= 0.5).astype(int)):.4f}")
    print("\n====================================================================================")
    print(f"🏆 FINAL STACKING ENSEMBLE OOF ACCURACY: {final_acc*100:.2f}% (Threshold={best_thresh:.3f})")
    print(f"🏆 FINAL STACKING ENSEMBLE OOF ROC-AUC:  {final_auc:.4f}")
    print("====================================================================================")

    # Save Results & Models
    models_dir = '/Users/kevinluo/ml-expert-skill-make/benchmarks/spaceship_titanic/models'
    os.makedirs(models_dir, exist_ok=True)
    np.savez('/Users/kevinluo/ml-expert-skill-make/benchmarks/spaceship_titanic/data/oof_results.npz',
             target=target, oof_pt=oof_pt, oof_lgb=oof_lgb, oof_xgb=oof_xgb, oof_cat=oof_cat, oof_et=oof_et,
             ensemble_probs=ensemble_probs, best_thresh=best_thresh)

    lgb.booster_.save_model(os.path.join(models_dir, 'lightgbm_model.txt'))
    xgb.save_model(os.path.join(models_dir, 'xgboost_model.json'))
    cat.save_model(os.path.join(models_dir, 'catboost_model.cbm'))
    joblib.dump(et, os.path.join(models_dir, 'extratrees_model.joblib'))
    joblib.dump(meta, os.path.join(models_dir, 'stacking_meta_learner.joblib'))
    joblib.dump(pipeline_init, os.path.join(models_dir, 'feature_pipeline.joblib'))

    print(f"\n✅ All trained Spaceship Titanic models successfully saved to: {models_dir}/")

if __name__ == '__main__':
    main()
