import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

class TitanicFeaturePipeline:
    """
    Leakage-free feature engineering and imputation pipeline for Titanic dataset.
    Must be fitted ONLY on training split during Cross-Validation.
    """
    
    RARE_TITLES = {'Dr', 'Rev', 'Col', 'Major', 'Capt', 'Lady', 'Sir', 'Countess', 'Jonkheer', 'Dona', 'Don'}

    def __init__(self):
        self.age_medians_: Dict[Tuple[str, int], float] = {}
        self.overall_age_median_: float = 28.0
        self.fare_medians_: Dict[int, float] = {}
        self.overall_fare_median_: float = 14.45
        self.embarked_mode_: str = 'S'
        self.fitted_: bool = False

    def _extract_title(self, name: str) -> str:
        match = re.search(r' ([A-Za-z]+)\.', str(name))
        if not match:
            return 'Misc'
        title = match.group(1)
        if title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Mme':
            return 'Mrs'
        elif title in self.RARE_TITLES:
            return 'Rare'
        elif title in ['Mr', 'Mrs', 'Miss', 'Master']:
            return title
        return 'Misc'

    def _extract_deck(self, cabin: Any) -> str:
        if pd.isna(cabin) or not str(cabin).strip():
            return 'Unknown'
        deck = str(cabin).strip()[0].upper()
        if deck not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            return 'Unknown'
        return deck

    def fit(self, df: pd.DataFrame) -> 'TitanicFeaturePipeline':
        """Fit imputation medians and modes strictly on training data."""
        df_copy = df.copy()
        df_copy['Title'] = df_copy['Name'].apply(self._extract_title)
        
        # Calculate Age medians per (Title, Pclass)
        grouped_age = df_copy.groupby(['Title', 'Pclass'])['Age'].median()
        self.age_medians_ = grouped_age.to_dict()
        self.overall_age_median_ = float(df_copy['Age'].median()) if not np.isnan(df_copy['Age'].median()) else 28.0
        
        # Calculate Fare medians per Pclass
        grouped_fare = df_copy.groupby('Pclass')['Fare'].median()
        self.fare_medians_ = grouped_fare.to_dict()
        self.overall_fare_median_ = float(df_copy['Fare'].median()) if not np.isnan(df_copy['Fare'].median()) else 14.45
        
        # Calculate Embarked mode
        if 'Embarked' in df_copy and not df_copy['Embarked'].mode().empty:
            self.embarked_mode_ = str(df_copy['Embarked'].mode()[0])
            
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset applying learned imputations and engineered features."""
        if not self.fitted_:
            raise RuntimeError("Pipeline must be fitted before calling transform!")
            
        out = df.copy()
        
        # 1. Title Extraction
        out['Title'] = out['Name'].apply(self._extract_title)
        
        # 2. Deck Extraction
        out['Deck'] = out['Cabin'].apply(self._extract_deck)
        
        # 3. Impute Embarked
        if 'Embarked' in out:
            out['Embarked'] = out['Embarked'].fillna(self.embarked_mode_)
        
        # 4. Impute Fare
        def impute_fare(row):
            if pd.isna(row['Fare']):
                return self.fare_medians_.get(row['Pclass'], self.overall_fare_median_)
            return row['Fare']
        out['Fare'] = out.apply(impute_fare, axis=1)
        
        # 5. Impute Age
        def impute_age(row):
            if pd.isna(row['Age']):
                key = (row['Title'], row['Pclass'])
                return self.age_medians_.get(key, self.overall_age_median_)
            return row['Age']
        out['Age'] = out.apply(impute_age, axis=1)
        
        # 6. Family Dynamics Features
        out['FamilySize'] = out['SibSp'] + out['Parch'] + 1
        out['IsAlone'] = (out['FamilySize'] == 1).astype(int)
        
        # 7. Ticket Frequency & Group Size
        ticket_counts = out['Ticket'].value_counts().to_dict()
        out['TicketGroupSize'] = out['Ticket'].map(ticket_counts).fillna(1).astype(int)
        out['FarePerPerson'] = out['Fare'] / out['TicketGroupSize']
        
        # 8. Fare & Age Binning / Log Transformations
        out['LogFare'] = np.log1p(out['Fare'])
        out['LogFarePerPerson'] = np.log1p(out['FarePerPerson'])
        out['AgeClass'] = out['Age'] * out['Pclass']
        
        # Categorical Encodings (Map to explicit integers for PyTorch / GBDT)
        title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4, 'Misc': 5}
        sex_map = {'male': 0, 'female': 1}
        embarked_map = {'S': 0, 'C': 1, 'Q': 2}
        deck_map = {'Unknown': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        
        out['TitleCode'] = out['Title'].map(title_map).fillna(5).astype(int)
        out['SexCode'] = out['Sex'].map(sex_map).fillna(0).astype(int)
        out['EmbarkedCode'] = out['Embarked'].map(embarked_map).fillna(0).astype(int)
        out['DeckCode'] = out['Deck'].map(deck_map).fillna(0).astype(int)
        
        return out
