import os
import torch
import joblib
import numpy as np
import pandas as pd

from benchmarks.house_prices.src.models.pytorch_regressor import PyTorchHousingRegressor

def run_house_prices_inference():
    """
    Standard Inference Script for House Prices Model.
    Loads saved model checkpoints from benchmarks/house_prices/models/ and predicts real USD prices.
    """
    models_dir = 'benchmarks/house_prices/models'
    data_path = 'benchmarks/house_prices/data/raw/Housing.csv'

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Missing '{models_dir}' directory! Run train_ensemble.py first.")
        
    print("=========================================================================")
    print("🚀 Running House Prices Regression Inference Pipeline...")
    print("=========================================================================")

    # 1. Load Feature Pipeline & Data
    print("\n1️⃣ Loading Housing dataset & feature pipeline...")
    pipeline = joblib.load(os.path.join(models_dir, 'feature_pipeline.joblib'))
    raw_df = pd.read_csv(data_path)
    
    # Take first 10 rows as sample inference test cases
    test_df = raw_df.head(10).copy()
    transformed_test = pipeline.transform(test_df)

    cat_cols = ['mainroad_code', 'guestroom_code', 'basement_code', 'hotwaterheating_code', 'airconditioning_code', 'prefarea_code', 'furnishing_code']
    num_cols = ['area', 'LogArea', 'bedrooms', 'bathrooms', 'stories', 'parking', 'TotalRooms', 'TotalBathRatio']

    X_test_cat = transformed_test[cat_cols]
    X_test_num = transformed_test[num_cols]

    num_mean = X_test_num.mean()
    num_std = X_test_num.std().replace(0, 1.0)
    X_test_num_scaled = (X_test_num - num_mean) / num_std
    X_test_gb = pd.concat([X_test_cat, X_test_num_scaled], axis=1)

    # 2. PyTorch Model Predictions
    print("\n2️⃣ Loading PyTorch Regressor checkpoints (.pth)...")
    emb_dims = {
        'mainroad_code': (2, 2), 'guestroom_code': (2, 2), 'basement_code': (2, 2),
        'hotwaterheating_code': (2, 2), 'airconditioning_code': (2, 2), 'prefarea_code': (2, 2),
        'furnishing_code': (3, 2)
    }
    cat_dims = {col: emb_dims[col][0] for col in cat_cols}

    cat_t = {col: torch.tensor(X_test_cat[col].values, dtype=torch.long) for col in X_test_cat.columns}
    num_t = torch.tensor(X_test_num_scaled.values, dtype=torch.float32)

    pt_preds = np.zeros(len(test_df))
    fold_count = 0
    for fold in range(1, 6):
        ckpt_path = os.path.join(models_dir, f'pytorch_fold_{fold}.pth')
        if os.path.exists(ckpt_path):
            model = PyTorchHousingRegressor(cat_dims=cat_dims, emb_dims=emb_dims, num_dim=len(num_cols), hidden_dim=128)
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
            with torch.no_grad():
                pt_preds += model(cat_t, num_t).numpy()
            fold_count += 1
            print(f"   Loaded PyTorch checkpoint: {ckpt_path}")
            
    if fold_count > 0:
        pt_preds /= fold_count

    # 3. GBDT Model Predictions
    print("\n3️⃣ Loading GBDT Regressor checkpoints (.joblib)...")
    lgb = joblib.load(os.path.join(models_dir, 'lightgbm_regressor.joblib'))
    xgb = joblib.load(os.path.join(models_dir, 'xgboost_regressor.joblib'))
    cat = joblib.load(os.path.join(models_dir, 'catboost_regressor.joblib'))

    lgb_preds = lgb.predict(X_test_gb)
    xgb_preds = xgb.predict(X_test_gb)
    cat_preds = cat.predict(X_test_gb)

    # 4. Meta Stacking Inference & Inverse Log Transform
    print("\n4️⃣ Performing Meta-Learner Stacking & Inverse Log1p Transformation...")
    meta = joblib.load(os.path.join(models_dir, 'stacking_ridge_meta.joblib'))
    X_test_meta = np.column_stack([pt_preds, lgb_preds, xgb_preds, cat_preds])
    
    final_log_preds = meta.predict(X_test_meta)
    
    # Convert log1p back to actual USD price values
    actual_prices = test_df['price'].values
    predicted_prices = np.expm1(final_log_preds)

    results_df = pd.DataFrame({
        'Area_sqft': test_df['area'],
        'Bedrooms': test_df['bedrooms'],
        'Bathrooms': test_df['bathrooms'],
        'Actual_Price_USD': [f"${x:,.0f}" for x in actual_prices],
        'Predicted_Price_USD': [f"${x:,.0f}" for x in predicted_prices],
        'Error_Pct': [f"{abs(p - a)/a*100:.2f}%" for a, p in zip(actual_prices, predicted_prices)]
    })

    print("\n=========================================================================")
    print("✨ HOUSE PRICES INFERENCE COMPLETED SUCCESSFULLY!")
    print("=========================================================================\n")
    print("Sample Inference Predictions vs Actual Prices:\n")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    run_house_prices_inference()
