import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

from benchmarks.house_prices.src.features import HousingFeaturePipeline

def save_house_prices_native():
    models_dir = '/Users/kevinluo/ml-expert-skill-make/benchmarks/house_prices/models'
    os.makedirs(models_dir, exist_ok=True)
    
    df = pd.read_csv('benchmarks/house_prices/data/raw/Housing.csv')
    pipeline = HousingFeaturePipeline()
    pipeline.fit(df)
    
    transformed_df = pipeline.transform(df)
    cat_cols = ['mainroad_code', 'guestroom_code', 'basement_code', 'hotwaterheating_code', 'airconditioning_code', 'prefarea_code', 'furnishing_code']
    num_cols = ['area', 'LogArea', 'bedrooms', 'bathrooms', 'stories', 'parking', 'TotalRooms', 'TotalBathRatio']
    
    X_cat, X_num, y = transformed_df[cat_cols], transformed_df[num_cols], transformed_df['LogPrice']
    num_mean = X_num.mean()
    num_std = X_num.std().replace(0, 1.0)
    X_num_scaled = (X_num - num_mean) / num_std
    X_gb = pd.concat([X_cat, X_num_scaled], axis=1)
    
    lgb = LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=12, max_depth=3, random_state=2026, verbose=-1, n_jobs=1)
    lgb.fit(X_gb, y)
    
    xgb = XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=2026, n_jobs=1)
    xgb.fit(X_gb, y)
    
    cat = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=3, random_seed=2026, verbose=0, thread_count=1)
    cat.fit(X_gb, y)
    
    p_lgb = lgb.predict(X_gb)
    p_xgb = xgb.predict(X_gb)
    p_cat = cat.predict(X_gb)
    
    meta = Ridge(alpha=1.0, random_state=2026)
    meta.fit(np.column_stack([p_lgb, p_xgb, p_cat]), y)
    
    lgb.booster_.save_model(os.path.join(models_dir, 'lightgbm_regressor.txt'))
    xgb.save_model(os.path.join(models_dir, 'xgboost_regressor.json'))
    cat.save_model(os.path.join(models_dir, 'catboost_regressor.cbm'))
    
    joblib.dump(pipeline, os.path.join(models_dir, 'feature_pipeline.joblib'))
    joblib.dump(meta, os.path.join(models_dir, 'stacking_ridge_meta.joblib'))
    print("✅ Successfully saved native House Prices models (.txt, .json, .cbm, .joblib)")

if __name__ == '__main__':
    save_house_prices_native()
