import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any

class SpaceshipFeaturePipeline:
    """
    Leakage-free Feature Engineering & Preprocessing Pipeline for Spaceship Titanic.
    Fitted ONLY on training fold split during Cross-Validation.
    """
    EXPENDITURE_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    def __init__(self):
        self.exp_medians_: Dict[str, float] = {}
        self.age_median_: float = 27.0
        self.home_planet_mode_: str = 'Earth'
        self.destination_mode_: str = 'TRAPPIST-1e'
        self.cryo_mode_: bool = False
        self.vip_mode_: bool = False
        self.fitted_: bool = False

        self.home_planet_map = {'Earth': 0, 'Europa': 1, 'Mars': 2, 'Unknown': 3}
        self.destination_map = {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2, 'Unknown': 3}
        self.deck_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'Unknown': 8}
        self.side_map = {'P': 0, 'S': 1, 'Unknown': 2}

    def _dissect_cabin(self, cabin: Any) -> Tuple[str, str, int]:
        if pd.isna(cabin) or not isinstance(cabin, str) or '/' not in cabin:
            return 'Unknown', 'Unknown', 0
        parts = cabin.split('/')
        deck = parts[0].upper() if parts[0].upper() in self.deck_map else 'Unknown'
        side = parts[2].upper() if len(parts) > 2 and parts[2].upper() in self.side_map else 'Unknown'
        try:
            num = int(parts[1])
        except (ValueError, IndexError):
            num = 0
        return deck, side, num

    def fit(self, df: pd.DataFrame) -> 'SpaceshipFeaturePipeline':
        df_copy = df.copy()

        # Fit expenditure medians
        for col in self.EXPENDITURE_COLS:
            if col in df_copy:
                med = float(df_copy[col].median())
                self.exp_medians_[col] = med if not np.isnan(med) else 0.0

        if 'Age' in df_copy and not np.isnan(df_copy['Age'].median()):
            self.age_median_ = float(df_copy['Age'].median())

        if 'HomePlanet' in df_copy and not df_copy['HomePlanet'].mode().empty:
            self.home_planet_mode_ = str(df_copy['HomePlanet'].mode()[0])

        if 'Destination' in df_copy and not df_copy['Destination'].mode().empty:
            self.destination_mode_ = str(df_copy['Destination'].mode()[0])

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Pipeline must be fitted before calling transform!")

        out = df.copy()

        # 1. Group Dynamics from PassengerId
        out['GroupId'] = out['PassengerId'].apply(lambda x: str(x).split('_')[0] if pd.notna(x) else '0000')
        group_counts = out['GroupId'].value_counts().to_dict()
        out['GroupSize'] = out['GroupId'].map(group_counts).fillna(1).astype(int)
        out['IsAloneGroup'] = (out['GroupSize'] == 1).astype(int)

        # 2. Cabin Dissection
        if 'Cabin' in out:
            cabin_tuples = out['Cabin'].apply(self._dissect_cabin)
            out['Deck'] = [t[0] for t in cabin_tuples]
            out['Side'] = [t[1] for t in cabin_tuples]
            out['CabinNum'] = [t[2] for t in cabin_tuples]
        else:
            out['Deck'] = 'Unknown'
            out['Side'] = 'Unknown'
            out['CabinNum'] = 0

        # 3. Categorical Imputation
        if 'HomePlanet' in out:
            out['HomePlanet'] = out['HomePlanet'].fillna(self.home_planet_mode_)
        else:
            out['HomePlanet'] = self.home_planet_mode_

        if 'Destination' in out:
            out['Destination'] = out['Destination'].fillna(self.destination_mode_)
        else:
            out['Destination'] = self.destination_mode_

        if 'CryoSleep' in out:
            out['CryoSleep'] = out['CryoSleep'].fillna(False).astype(bool)
        else:
            out['CryoSleep'] = False

        if 'VIP' in out:
            out['VIP'] = out['VIP'].fillna(False).astype(bool)
        else:
            out['VIP'] = False

        if 'Age' in out:
            out['Age'] = out['Age'].fillna(self.age_median_)
        else:
            out['Age'] = self.age_median_

        # 4. Expenditure Imputation & CryoSleep Rule
        for col in self.EXPENDITURE_COLS:
            if col in out:
                out[col] = out[col].fillna(self.exp_medians_.get(col, 0.0))
                # CryoSleep passengers spend 0
                out.loc[out['CryoSleep'] == True, col] = 0.0

        out['TotalSpending'] = out[self.EXPENDITURE_COLS].sum(axis=1)
        out['HasSpent'] = (out['TotalSpending'] > 0).astype(int)
        out['LogTotalSpending'] = np.log1p(out['TotalSpending'])

        for col in self.EXPENDITURE_COLS:
            out[f'Log_{col}'] = np.log1p(out[col])

        # 5. Encodings
        out['HomePlanetCode'] = out['HomePlanet'].map(self.home_planet_map).fillna(3).astype(int)
        out['DestinationCode'] = out['Destination'].map(self.destination_map).fillna(3).astype(int)
        out['DeckCode'] = out['Deck'].map(self.deck_map).fillna(8).astype(int)
        out['SideCode'] = out['Side'].map(self.side_map).fillna(2).astype(int)
        out['CryoSleepCode'] = out['CryoSleep'].astype(int)
        out['VIPCode'] = out['VIP'].astype(int)

        if 'Transported' in out:
            out['Target'] = out['Transported'].astype(int)

        return out
