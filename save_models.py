import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from src.features import TitanicFeaturePipeline

def save_all_native():
    models_dir = '/Users/kevinluo/ml-expert-skill-make/models'
    os.makedirs(models_dir, exist_ok=True)
    
    raw_df = pd.read_csv('data/raw/train.csv')
    pipeline = TitanicFeaturePipeline()
    pipeline.fit(raw_df)
    
    feat_df = pipeline.transform(raw_df)
    cat_cols = ['TitleCode', 'DeckCode', 'SexCode', 'EmbarkedCode', 'Pclass']
    num_cols = ['Age', 'Fare', 'LogFare', 'LogFarePerPerson', 'FamilySize', 'IsAlone', 'TicketGroupSize', 'AgeClass']
    
    X_cat, X_num, y = feat_df[cat_cols], feat_df[num_cols], feat_df['Survived']
    num_mean = X_num.mean()
    num_std = X_num.std().replace(0, 1.0)
    X_num_scaled = (X_num - num_mean) / num_std
    X_gb = pd.concat([X_cat, X_num_scaled], axis=1)
    
    lgb = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=15, max_depth=4, random_state=42, verbose=-1, n_jobs=1)
    lgb.fit(X_gb, y)
    
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', n_jobs=1)
    xgb.fit(X_gb, y)
    
    cat = CatBoostClassifier(iterations=200, learning_rate=0.03, depth=4, random_seed=42, verbose=0, thread_count=1)
    cat.fit(X_gb, y)
    
    p_lgb = lgb.predict_proba(X_gb)[:, 1]
    p_xgb = xgb.predict_proba(X_gb)[:, 1]
    p_cat = cat.predict_proba(X_gb)[:, 1]
    
    meta = LogisticRegression(random_state=42)
    meta.fit(np.column_stack([p_lgb, p_xgb, p_cat]), y)
    
    # Save using C++ Native APIs to avoid joblib threading issues on macOS
    lgb.booster_.save_model(os.path.join(models_dir, 'lightgbm_model.txt'))
    xgb.save_model(os.path.join(models_dir, 'xgboost_model.json'))
    cat.save_model(os.path.join(models_dir, 'catboost_model.cbm'))
    
    joblib.dump(pipeline, os.path.join(models_dir, 'feature_pipeline.joblib'))
    joblib.dump(meta, os.path.join(models_dir, 'stacking_meta_learner.joblib'))
    print("✅ Successfully saved native model checkpoints (.txt, .json, .cbm, .joblib)")

if __name__ == '__main__':
    save_all_native()
