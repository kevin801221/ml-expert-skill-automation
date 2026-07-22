import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

class HousingFeaturePipeline:
    """
    Leakage-free Feature Engineering & Preprocessing Pipeline for Housing Regression.
    Fitted ONLY on training fold split during Cross-Validation.
    """
    def __init__(self):
        self.num_means_: Dict[str, float] = {}
        self.num_stds_: Dict[str, float] = {}
        self.fitted_: bool = False

        self.binary_map = {'yes': 1, 'no': 0}
        self.furnish_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}

    def fit(self, df: pd.DataFrame) -> 'HousingFeaturePipeline':
        df_copy = df.copy()
        
        # High-signal feature engineering
        df_copy['LogArea'] = np.log1p(df_copy['area'])
        df_copy['TotalRooms'] = df_copy['bedrooms'] + df_copy['bathrooms']
        df_copy['TotalBathRatio'] = df_copy['bathrooms'] / (df_copy['bedrooms'] + 1e-5)
        
        num_cols = ['area', 'LogArea', 'bedrooms', 'bathrooms', 'stories', 'parking', 'TotalRooms', 'TotalBathRatio']
        for col in num_cols:
            self.num_means_[col] = float(df_copy[col].mean())
            std = float(df_copy[col].std())
            self.num_stds_[col] = std if std > 0 else 1.0
            
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Pipeline must be fitted before calling transform!")
            
        out = df.copy()
        
        # Target Log1p Transform
        if 'price' in out:
            out['LogPrice'] = np.log1p(out['price'])
            
        # Feature Engineering
        out['LogArea'] = np.log1p(out['area'])
        out['TotalRooms'] = out['bedrooms'] + out['bathrooms']
        out['TotalBathRatio'] = out['bathrooms'] / (out['bedrooms'] + 1e-5)
        
        # Binary Categorical Encoding
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        for col in binary_cols:
            if col in out:
                out[col + '_code'] = out[col].map(self.binary_map).fillna(0).astype(int)
                
        if 'furnishingstatus' in out:
            out['furnishing_code'] = out['furnishingstatus'].map(self.furnish_map).fillna(0).astype(int)
            
        return out
