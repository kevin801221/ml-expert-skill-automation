import pytest
import pandas as pd
import numpy as np
from src.features import TitanicFeaturePipeline

def test_feature_pipeline_fit_transform():
    raw_data = {
        'PassengerId': [1, 2, 3, 4],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath'],
        'Sex': ['male', 'female', 'female', 'female'],
        'Age': [22.0, np.nan, 26.0, 35.0],
        'SibSp': [1, 1, 0, 1],
        'Parch': [0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803'],
        'Fare': [7.25, 71.2833, np.nan, 53.1],
        'Cabin': [np.nan, 'C85', np.nan, 'C123'],
        'Embarked': ['S', 'C', np.nan, 'S'],
        'Pclass': [3, 1, 3, 1]
    }
    df = pd.DataFrame(raw_data)
    
    pipeline = TitanicFeaturePipeline()
    pipeline.fit(df)
    transformed = pipeline.transform(df)
    
    # Assert missing values imputed
    assert transformed['Age'].isna().sum() == 0
    assert transformed['Fare'].isna().sum() == 0
    assert transformed['Embarked'].isna().sum() == 0
    
    # Assert engineered features created
    assert 'Title' in transformed.columns
    assert 'FamilySize' in transformed.columns
    assert 'IsAlone' in transformed.columns
    assert 'Deck' in transformed.columns
    assert 'LogFarePerPerson' in transformed.columns
    
    # Verify titles
    assert list(transformed['Title']) == ['Mr', 'Mrs', 'Miss', 'Mrs']
    assert transformed['FamilySize'].iloc[0] == 2
    assert transformed['IsAlone'].iloc[2] == 1
