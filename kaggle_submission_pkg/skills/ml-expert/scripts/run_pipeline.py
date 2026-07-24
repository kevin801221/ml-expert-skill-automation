import os
import sys
import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def find_files():
    search_dirs = ['.', '..', '/kaggle/input', '/kaggle/input/*']
    train_file, test_file, sample_sub_file = None, None, None
    
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, '*.csv')):
            fname = os.path.basename(f).lower()
            if 'train' in fname and not train_file:
                train_file = f
            elif 'test' in fname and not test_file:
                test_file = f
            elif ('sub' in fname or 'sample' in fname) and not sample_sub_file:
                sample_sub_file = f
                
    return train_file, test_file, sample_sub_file

def run_universal_pipeline():
    print("=========================================================================")
    print("🚀 Running Universal High-Performance Auto-ML Pipeline...")
    print("=========================================================================")

    train_file, test_file, sample_sub_file = find_files()
    
    if not train_file or not test_file:
        print(f"Error: Could not locate train or test CSV files. Searched current and parent directories.")
        sys.exit(1)

    print(f"Found Train File: {train_file}")
    print(f"Found Test File:  {test_file}")
    if sample_sub_file:
        print(f"Found Sample Sub: {sample_sub_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

    # Determine Target & ID columns
    target_col = None
    if sample_sub_file:
        sample_sub = pd.read_csv(sample_sub_file)
        possible_targets = [c for c in sample_sub.columns if c in train_df.columns and c not in test_df.columns]
        if possible_targets:
            target_col = possible_targets[0]
            
    if not target_col:
        candidates = [c for c in train_df.columns if c not in test_df.columns]
        if candidates:
            target_col = candidates[0]
        else:
            target_col = train_df.columns[-1]

    id_cols = [c for c in train_df.columns if c.lower() in ['id', 'passengerid', 'guid', 'index'] or c in test_df.columns and test_df[c].nunique() == len(test_df)]
    id_col = id_cols[0] if id_cols else None

    print(f"Identified Target Column: [ {target_col} ]")
    print(f"Identified ID Column:     [ {id_col} ]")

    y = train_df[target_col].values
    if y.dtype == object or isinstance(y[0], (str, bool)):
        y = pd.Series(y).astype('category').cat.codes.values

    # Feature Processing
    feature_cols = [c for c in train_df.columns if c != target_col and c != id_col]
    
    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    # Preprocessing numerical and categorical
    cat_cols, num_cols = [], []
    for c in feature_cols:
        if X_train_raw[c].dtype == object or X_train_raw[c].dtype.name == 'category' or X_train_raw[c].nunique() < 10:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    print(f"Categorical Features ({len(cat_cols)}): {cat_cols[:5]}...")
    print(f"Numerical Features ({len(num_cols)}): {num_cols[:5]}...")

    # Transform Categoricals to Ordinal Codes safely
    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].astype(str)
        X_test_raw[c] = X_test_raw[c].astype(str)
        
        categories = sorted(list(set(X_train_raw[c].unique()).union(set(X_test_raw[c].unique()))))
        cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
        
        X_train_raw[c] = X_train_raw[c].astype(cat_type).cat.codes
        X_test_raw[c] = X_test_raw[c].astype(cat_type).cat.codes

    # Handle missing numerical values with median
    for c in num_cols:
        med = X_train_raw[c].median()
        if pd.isna(med):
            med = 0.0
        X_train_raw[c] = X_train_raw[c].fillna(med)
        X_test_raw[c] = X_test_raw[c].fillna(med)

    # Standard Scaling
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(X_train_raw[num_cols]) if num_cols else np.zeros((len(X_train_raw), 0))
    X_te_num = scaler.transform(X_test_raw[num_cols]) if num_cols else np.zeros((len(X_test_raw), 0))

    X_tr_final = np.column_stack([X_train_raw[cat_cols].values, X_tr_num]) if cat_cols else X_tr_num
    X_te_final = np.column_stack([X_test_raw[cat_cols].values, X_te_num]) if cat_cols else X_te_num

    print(f"Final Preprocessed Feature Matrix Shape: Train {X_tr_final.shape}, Test {X_te_final.shape}")

    # 5-Fold Stratified K-Fold Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    
    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    oof_hgb = np.zeros(len(train_df))

    test_lgb = np.zeros(len(test_df))
    test_xgb = np.zeros(len(test_df))
    test_cat = np.zeros(len(test_df))
    test_hgb = np.zeros(len(test_df))

    print("\n--- Starting 5-Fold Stratified CV Model Zoo Training ---")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr_final, y)):
        X_tr, y_tr = X_tr_final[trn_idx], y[trn_idx]
        X_va, y_va = X_tr_final[val_idx], y[val_idx]

        # 1. LightGBM
        lgb = LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=31, max_depth=5, random_state=2026, verbose=-1, n_jobs=1)
        lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb.predict_proba(X_va)[:, 1]
        test_lgb += lgb.predict_proba(X_te_final)[:, 1] / 5.0

        # 2. XGBoost
        xgb = XGBClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=2026, eval_metric='logloss', n_jobs=1)
        xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb.predict_proba(X_va)[:, 1]
        test_xgb += xgb.predict_proba(X_te_final)[:, 1] / 5.0

        # 3. CatBoost
        cat = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, random_seed=2026, verbose=0, thread_count=1)
        cat.fit(X_tr, y_tr)
        oof_cat[val_idx] = cat.predict_proba(X_va)[:, 1]
        test_cat += cat.predict_proba(X_te_final)[:, 1] / 5.0

        # 4. HistGradientBoosting
        hgb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.03, max_depth=5, random_state=2026)
        hgb.fit(X_tr, y_tr)
        oof_hgb[val_idx] = hgb.predict_proba(X_va)[:, 1]
        test_hgb += hgb.predict_proba(X_te_final)[:, 1] / 5.0

    # Stacking Meta Learner
    X_meta = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hgb])
    X_test_meta = np.column_stack([test_lgb, test_xgb, test_cat, test_hgb])

    meta = LogisticRegression(random_state=2026)
    meta.fit(X_meta, y)

    oof_ensemble = meta.predict_proba(X_meta)[:, 1]
    final_test_probs = meta.predict_proba(X_test_meta)[:, 1]

    auc = roc_auc_score(y, oof_ensemble)
    acc = accuracy_score(y, (oof_ensemble >= 0.5).astype(int))

    print("\n=========================================================================")
    print(f"🏆 UNIVERSAL AUTO-ML STACKING ENSEMBLE OOF ROC-AUC: {auc:.4f}")
    print(f"🏆 UNIVERSAL AUTO-ML STACKING ENSEMBLE OOF ACCURACY: {acc*100:.2f}%")
    print("=========================================================================")

    # Output predictions matching test / sample submission
    if id_col and id_col in test_df:
        sub_df = pd.DataFrame({id_col: test_df[id_col], target_col: final_test_probs})
    else:
        sub_df = pd.DataFrame({target_col: final_test_probs})

    sub_path = 'submission.csv'
    sub_df.to_csv(sub_path, index=False)
    print(f"\n✅ Saved prediction submission to: {os.path.abspath(sub_path)}")
    print(f"Submitting predictions table head:\n{sub_df.head(5)}")

if __name__ == '__main__':
    run_universal_pipeline()
