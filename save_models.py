import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from src.features import TitanicFeaturePipeline

def save_all():
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
    
    lgb = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=15, max_depth=4, random_state=42, verbose=-1)
    lgb.fit(X_gb, y)
    
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    xgb.fit(X_gb, y)
    
    cat = CatBoostClassifier(iterations=200, learning_rate=0.03, depth=4, random_seed=42, verbose=0)
    cat.fit(X_gb, y)
    
    meta = LogisticRegression(random_state=42)
    meta.fit(np.column_stack([lgb.predict_proba(X_gb)[:,1], xgb.predict_proba(X_gb)[:,1], cat.predict_proba(X_gb)[:,1]]), y)
    
    joblib.dump(pipeline, os.path.join(models_dir, 'feature_pipeline.joblib'))
    joblib.dump(lgb, os.path.join(models_dir, 'lightgbm_model.joblib'))
    joblib.dump(xgb, os.path.join(models_dir, 'xgboost_model.joblib'))
    joblib.dump(cat, os.path.join(models_dir, 'catboost_model.joblib'))
    joblib.dump(meta, os.path.join(models_dir, 'stacking_meta_learner.joblib'))
    print("✅ Successfully serialized all models to models/")

if __name__ == '__main__':
    save_all()
