import pytest
import pandas as pd
import numpy as np
from benchmarks.spaceship_titanic.src.features import SpaceshipFeaturePipeline

def test_spaceship_feature_pipeline():
    raw_data = {
        'PassengerId': ['0001_01', '0002_01', '0003_01', '0003_02'],
        'HomePlanet': ['Europa', 'Earth', 'Europa', np.nan],
        'CryoSleep': [False, False, False, True],
        'Cabin': ['B/0/P', 'F/0/S', 'A/0/S', 'A/0/S'],
        'Destination': ['TRAPPIST-1e', 'TRAPPIST-1e', 'TRAPPIST-1e', '55 Cancri e'],
        'Age': [39.0, 24.0, 58.0, 33.0],
        'VIP': [False, False, True, False],
        'RoomService': [0.0, 109.0, 43.0, np.nan],
        'FoodCourt': [0.0, 9.0, 3576.0, np.nan],
        'ShoppingMall': [0.0, 25.0, 0.0, np.nan],
        'Spa': [0.0, 549.0, 6715.0, np.nan],
        'VRDeck': [0.0, 44.0, 49.0, np.nan],
        'Name': ['Maham Ofracculy', 'Juanna Vines', 'Altark Susent', 'Solam Susent'],
        'Transported': [False, True, False, False]
    }
    df = pd.DataFrame(raw_data)

    pipeline = SpaceshipFeaturePipeline()
    pipeline.fit(df)
    transformed = pipeline.transform(df)

    assert 'Deck' in transformed.columns
    assert 'Side' in transformed.columns
    assert 'GroupSize' in transformed.columns
    assert 'TotalSpending' in transformed.columns

    # CryoSleep passengers spend 0
    assert transformed.loc[transformed['CryoSleep'] == True, 'TotalSpending'].iloc[0] == 0.0
    assert transformed['GroupSize'].iloc[2] == 2
    assert transformed['Deck'].iloc[0] == 'B'
    assert transformed['Side'].iloc[0] == 'P'
