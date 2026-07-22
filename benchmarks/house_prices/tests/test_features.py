import pytest
import pandas as pd
import numpy as np
from benchmarks.house_prices.src.features import HousingFeaturePipeline

def test_housing_feature_pipeline():
    raw_data = {
        'price': [13300000, 12250000, 12250000],
        'area': [7420, 8960, 9960],
        'bedrooms': [4, 4, 3],
        'bathrooms': [2, 4, 2],
        'stories': [3, 4, 2],
        'mainroad': ['yes', 'yes', 'yes'],
        'guestroom': ['no', 'no', 'no'],
        'basement': ['no', 'no', 'yes'],
        'hotwaterheating': ['no', 'no', 'no'],
        'airconditioning': ['yes', 'yes', 'no'],
        'parking': [2, 3, 2],
        'prefarea': ['yes', 'no', 'yes'],
        'furnishingstatus': ['furnished', 'furnished', 'semi-furnished']
    }
    df = pd.DataFrame(raw_data)
    
    pipeline = HousingFeaturePipeline()
    pipeline.fit(df)
    transformed = pipeline.transform(df)
    
    assert 'LogPrice' in transformed.columns
    assert 'LogArea' in transformed.columns
    assert 'TotalRooms' in transformed.columns
    assert 'mainroad_code' in transformed.columns
    assert transformed['mainroad_code'].iloc[0] == 1
    assert transformed['furnishing_code'].iloc[2] == 1
