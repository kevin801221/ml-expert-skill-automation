import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from benchmarks.house_prices.src.features import HousingFeaturePipeline
from benchmarks.house_prices.src.models.pytorch_regressor import PyTorchHousingRegressor

def seed_everything(seed=2026):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_pytorch_regressor_fold(train_x_cat, train_x_num, train_y, val_x_cat, val_x_num, val_y, cat_dims, emb_dims, num_dim, epochs=150, batch_size=32, lr=1e-3) -> np.ndarray:
    train_cat_t = {col: torch.tensor(train_x_cat[col].values, dtype=torch.long) for col in train_x_cat.columns}
    train_num_t = torch.tensor(train_x_num.values, dtype=torch.float32)
    train_y_t = torch.tensor(train_y.values, dtype=torch.float32)
    
    val_cat_t = {col: torch.tensor(val_x_cat[col].values, dtype=torch.long) for col in val_x_cat.columns}
    val_num_t = torch.tensor(val_x_num.values, dtype=torch.float32)
    
    model = PyTorchHousingRegressor(cat_dims=cat_dims, emb_dims=emb_dims, num_dim=num_dim, hidden_dim=128, dropout_rate=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
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
            pred = model(batch_cat, batch_num)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    model.eval()
    with torch.no_grad():
        val_preds = model(val_cat_t, val_num_t).numpy()
    return val_preds

def main():
    seed_everything(2026)
    data_path = 'benchmarks/house_prices/data/raw/Housing.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}!")
        
    df = pd.read_csv(data_path)
    print(f"Loaded Housing dataset with shape: {df.shape}")
    
    cat_cols = ['mainroad_code', 'guestroom_code', 'basement_code', 'hotwaterheating_code', 'airconditioning_code', 'prefarea_code', 'furnishing_code']
    num_cols = ['area', 'LogArea', 'bedrooms', 'bathrooms', 'stories', 'parking', 'TotalRooms', 'TotalBathRatio']
    
    emb_dims = {
        'mainroad_code': (2, 2),
        'guestroom_code': (2, 2),
        'basement_code': (2, 2),
        'hotwaterheating_code': (2, 2),
        'airconditioning_code': (2, 2),
        'prefarea_code': (2, 2),
        'furnishing_code': (3, 2)
    }
    cat_dims = {col: emb_dims[col][0] for col in cat_cols}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=2026)
    
    oof_pt = np.zeros(len(df))
    oof_lgb = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    oof_cat = np.zeros(len(df))
    
    # Run pipeline on raw dataset first to compute target LogPrice
    init_pipeline = HousingFeaturePipeline()
    init_pipeline.fit(df)
    transformed_df = init_pipeline.transform(df)
    target_log_price = transformed_df['LogPrice'].values
    
    print("\n--- Starting 5-Fold Cross-Validation Housing Regression Training ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n>>> Executing Fold {fold+1} / 5 <<<")
        train_raw = df.iloc[train_idx].copy()
        val_raw = df.iloc[val_idx].copy()
        
        # Fit pipeline STRICTLY on train fold to avoid data leakage
        pipeline = HousingFeaturePipeline()
        pipeline.fit(train_raw)
        
        train_feat = pipeline.transform(train_raw)
        val_feat = pipeline.transform(val_raw)
        
        X_train_cat, X_train_num, y_train = train_feat[cat_cols], train_feat[num_cols], train_feat['LogPrice']
        X_val_cat, X_val_num, y_val = val_feat[cat_cols], val_feat[num_cols], val_feat['LogPrice']
        
        # Standardize numerical features
        num_mean = X_train_num.mean()
        num_std = X_train_num.std().replace(0, 1.0)
        X_train_num_scaled = (X_train_num - num_mean) / num_std
        X_val_num_scaled = (X_val_num - num_mean) / num_std
        
        # 1. PyTorch Tabular Regressor
        print("Training PyTorch Regressor...")
        preds_pt = train_pytorch_regressor_fold(
            X_train_cat, X_train_num_scaled, y_train,
            X_val_cat, X_val_num_scaled, y_val,
            cat_dims, emb_dims, len(num_cols)
        )
        oof_pt[val_idx] = preds_pt
        
        X_tr_gb = pd.concat([X_train_cat, X_train_num_scaled], axis=1)
        X_va_gb = pd.concat([X_val_cat, X_val_num_scaled], axis=1)
        
        # 2. LightGBM
        lgb = LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=12, max_depth=3, random_state=2026, verbose=-1, n_jobs=1)
        lgb.fit(X_tr_gb, y_train)
        oof_lgb[val_idx] = lgb.predict(X_va_gb)
        
        # 3. XGBoost
        xgb = XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=2026, n_jobs=1)
        xgb.fit(X_tr_gb, y_train)
        oof_xgb[val_idx] = xgb.predict(X_va_gb)
        
        # 4. CatBoost
        cat = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=3, random_seed=2026, verbose=0)
        cat.fit(X_tr_gb, y_train)
        oof_cat[val_idx] = cat.predict(X_va_gb)

    # Individual OOF RMSE
    rmse_pt = np.sqrt(mean_squared_error(target_log_price, oof_pt))
    rmse_lgb = np.sqrt(mean_squared_error(target_log_price, oof_lgb))
    rmse_xgb = np.sqrt(mean_squared_error(target_log_price, oof_xgb))
    rmse_cat = np.sqrt(mean_squared_error(target_log_price, oof_cat))
    
    print("\n================== Out-of-Fold Individual Model Performance ==================")
    print(f"PyTorch Net Log RMSE:  {rmse_pt:.4f}")
    print(f"LightGBM    Log RMSE:  {rmse_lgb:.4f}")
    print(f"XGBoost     Log RMSE:  {rmse_xgb:.4f}")
    print(f"CatBoost    Log RMSE:  {rmse_cat:.4f}")

    # Meta Ridge Stacking Ensemble
    X_meta = np.column_stack([oof_pt, oof_lgb, oof_xgb, oof_cat])
    meta = Ridge(alpha=1.0, random_state=2026)
    meta.fit(X_meta, target_log_price)
    
    final_log_preds = meta.predict(X_meta)
    final_rmse = np.sqrt(mean_squared_error(target_log_price, final_log_preds))
    final_r2 = r2_score(target_log_price, final_log_preds)
    
    print("\n====================================================================================")
    print(f"🏆 FINAL STACKING ENSEMBLE OOF LOG RMSE: {final_rmse:.4f}")
    print(f"🏆 FINAL STACKING ENSEMBLE R² SCORE:     {final_r2:.4f}")
    print("====================================================================================")

    # Save metrics for visualization
    np.savez('benchmarks/house_prices/data/oof_results.npz',
             target=target_log_price,
             oof_pt=oof_pt, oof_lgb=oof_lgb, oof_xgb=oof_xgb, oof_cat=oof_cat,
             final_preds=final_log_preds)

if __name__ == '__main__':
    main()
