import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.features import TitanicFeaturePipeline
from src.models.pytorch_net import PyTorchTabularNet

# Fix random seeds for 100% reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_pytorch_fold(train_x_cat, train_x_num, train_y, val_x_cat, val_x_num, val_y, cat_dims, emb_dims, num_dim, epochs=120, batch_size=32, lr=1e-3, fold_idx: int = None) -> np.ndarray:
    # Convert numpy to torch tensors
    train_cat_t = {col: torch.tensor(train_x_cat[col].values, dtype=torch.long) for col in train_x_cat.columns}
    train_num_t = torch.tensor(train_x_num.values, dtype=torch.float32)
    train_y_t = torch.tensor(train_y.values, dtype=torch.float32)
    
    val_cat_t = {col: torch.tensor(val_x_cat[col].values, dtype=torch.long) for col in val_x_cat.columns}
    val_num_t = torch.tensor(val_x_num.values, dtype=torch.float32)
    
    model = PyTorchTabularNet(cat_dims=cat_dims, emb_dims=emb_dims, num_features_dim=num_dim, hidden_dim=128, dropout_rate=0.2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Train Loop
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
        
    # Eval Inference & Save Checkpoint
    model.eval()
    os.makedirs('models', exist_ok=True)
    if fold_idx is not None:
        torch.save(model.state_dict(), f'models/pytorch_fold_{fold_idx+1}.pth')
        
    with torch.no_grad():
        val_logits = model(val_cat_t, val_num_t)
        probs = torch.sigmoid(val_logits).numpy()
    return probs

def main():
    seed_everything(42)
    data_path = 'data/raw/train.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}! Ensure Kaggle download executed successfully.")
        
    raw_df = pd.read_csv(data_path)
    print(f"Loaded raw dataset with shape: {raw_df.shape}")
    
    # Define features
    cat_cols = ['TitleCode', 'DeckCode', 'SexCode', 'EmbarkedCode', 'Pclass']
    num_cols = ['Age', 'Fare', 'LogFare', 'LogFarePerPerson', 'FamilySize', 'IsAlone', 'TicketGroupSize', 'AgeClass']
    
    # Embedding Dimensions per Categorical Feature
    emb_dims = {
        'TitleCode': (6, 4),
        'DeckCode': (8, 4),
        'SexCode': (2, 2),
        'EmbarkedCode': (3, 2),
        'Pclass': (4, 2)  # Pclass values 1..3 mapped safely
    }
    cat_dims = {col: emb_dims[col][0] for col in cat_cols}
    
    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_pytorch = np.zeros(len(raw_df))
    oof_lgb = np.zeros(len(raw_df))
    oof_xgb = np.zeros(len(raw_df))
    oof_cat = np.zeros(len(raw_df))
    
    target = raw_df['Survived'].values
    
    print("\n--- Starting 5-Fold Stratified Cross-Validation Training ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_df, target)):
        print(f"\n>>> Executing Fold {fold+1} / 5 <<<")
        
        train_raw = raw_df.iloc[train_idx].copy()
        val_raw = raw_df.iloc[val_idx].copy()
        
        # Fit pipeline STRICTLY on train fold to avoid data leakage
        pipeline = TitanicFeaturePipeline()
        pipeline.fit(train_raw)
        
        train_feat = pipeline.transform(train_raw)
        val_feat = pipeline.transform(val_raw)
        
        # Prepare Inputs
        X_train_cat, X_train_num, y_train = train_feat[cat_cols], train_feat[num_cols], train_feat['Survived']
        X_val_cat, X_val_num, y_val = val_feat[cat_cols], val_feat[num_cols], val_feat['Survived']
        
        # Standardize Numerical Features (Fit strictly on Train Fold)
        num_mean = X_train_num.mean()
        num_std = X_train_num.std().replace(0, 1.0)
        X_train_num_scaled = (X_train_num - num_mean) / num_std
        X_val_num_scaled = (X_val_num - num_mean) / num_std
        
        # 1. PyTorch Tabular Net
        print("Training PyTorch Tabular Net...")
        probs_pt = train_pytorch_fold(
            X_train_cat, X_train_num_scaled, y_train,
            X_val_cat, X_val_num_scaled, y_val,
            cat_dims, emb_dims, len(num_cols), fold_idx=fold
        )
        oof_pytorch[val_idx] = probs_pt
        
        # Combine cat + num for GBDT models
        X_train_gbdt = pd.concat([X_train_cat, X_train_num_scaled], axis=1)
        X_val_gbdt = pd.concat([X_val_cat, X_val_num_scaled], axis=1)
        
        # 2. LightGBM Classifier
        lgb_model = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=15, max_depth=4, random_state=42, verbose=-1)
        lgb_model.fit(X_train_gbdt, y_train)
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val_gbdt)[:, 1]
        
        # 3. XGBoost Classifier
        xgb_model = XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train_gbdt, y_train)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val_gbdt)[:, 1]
        
        # 4. CatBoost Classifier
        cat_model = CatBoostClassifier(iterations=200, learning_rate=0.03, depth=4, random_seed=42, verbose=0)
        cat_model.fit(X_train_gbdt, y_train)
        oof_cat[val_idx] = cat_model.predict_proba(X_val_gbdt)[:, 1]
        
        fold_blend = (probs_pt + oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 4.0
        fold_acc = accuracy_score(target[val_idx], (fold_blend >= 0.5).astype(int))
        print(f"Fold {fold+1} Blended Accuracy: {fold_acc:.4f}")

    # Evaluate Individual OOF Performance
    print("\n================== Out-of-Fold (OOF) Individual Model Performance ==================")
    print(f"PyTorch Net OOF Accuracy: {accuracy_score(target, (oof_pytorch >= 0.5).astype(int)):.4f} (AUC: {roc_auc_score(target, oof_pytorch):.4f})")
    print(f"LightGBM OOF Accuracy:    {accuracy_score(target, (oof_lgb >= 0.5).astype(int)):.4f} (AUC: {roc_auc_score(target, oof_lgb):.4f})")
    print(f"XGBoost  OOF Accuracy:    {accuracy_score(target, (oof_xgb >= 0.5).astype(int)):.4f} (AUC: {roc_auc_score(target, oof_xgb):.4f})")
    print(f"CatBoost OOF Accuracy:    {accuracy_score(target, (oof_cat >= 0.5).astype(int)):.4f} (AUC: {roc_auc_score(target, oof_cat):.4f})")

    # Stacking Meta-Learner Ensemble
    X_meta = np.column_stack([oof_pytorch, oof_lgb, oof_xgb, oof_cat])
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(X_meta, target)
    
    ensemble_probs = meta_model.predict_proba(X_meta)[:, 1]
    
    # Find Optimal Probability Threshold
    best_threshold = 0.5
    best_acc = 0.0
    for thresh in np.arange(0.3, 0.7, 0.01):
        acc = accuracy_score(target, (ensemble_probs >= thresh).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh

    ensemble_preds = (ensemble_probs >= best_threshold).astype(int)
    final_acc = accuracy_score(target, ensemble_preds)
    final_auc = roc_auc_score(target, ensemble_probs)
    
    print("\n====================================================================================")
    print(f"🏆 FINAL STACKING ENSEMBLE OOF ACCURACY: {final_acc*100:.2f}% (Threshold={best_threshold:.2f})")
    print(f"🏆 FINAL STACKING ENSEMBLE OOF ROC-AUC:  {final_auc:.4f}")
    print("====================================================================================")
    print("\nClassification Report:")
    print(classification_report(target, ensemble_preds))

    # Save Model Artifacts
    import joblib
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(lgb_model, os.path.join(models_dir, 'lightgbm_model.joblib'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_model.joblib'))
    joblib.dump(cat_model, os.path.join(models_dir, 'catboost_model.joblib'))
    joblib.dump(meta_model, os.path.join(models_dir, 'stacking_meta_learner.joblib'))
    joblib.dump(pipeline, os.path.join(models_dir, 'feature_pipeline.joblib'))
    
    print(f"\n✅ All trained model artifacts successfully saved to: {os.path.abspath(models_dir)}/")
    print(f"  - lightgbm_model.joblib")
    print(f"  - xgboost_model.joblib")
    print(f"  - catboost_model.joblib")
    print(f"  - stacking_meta_learner.joblib")
    print(f"  - feature_pipeline.joblib")
    print(f"  - pytorch_fold_*.pth (saved per fold)")

if __name__ == '__main__':
    main()
