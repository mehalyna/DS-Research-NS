"""
Week 4: Preprocessing Utilities
Reusable functions for feature engineering and data preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering utilities."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.label_encoders = {}
        self.feature_names = []
        
    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[FeatureEngineer] {message}")
    
    def create_age_buckets(self, df: pd.DataFrame, bins: List[int], 
                          labels: List[str], col: str = 'Age') -> pd.DataFrame:
        """Create age bucket categories."""
        self.log(f"Creating age buckets: {labels}")
        df['Age_Bucket'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features (multiplication)."""
        self.log(f"Creating {len(pairs)} interaction features")
        for feat1, feat2 in pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                # Handle categorical features by encoding first if needed
                if df[feat1].dtype == 'object' or df[feat2].dtype == 'object':
                    self.log(f"  Skipping {interaction_name} (categorical feature)")
                    continue
                df[interaction_name] = df[feat1] * df[feat2]
                self.log(f"  Created: {interaction_name}")
        return df
    
    def create_ratio_features(self, df: pd.DataFrame, 
                            ratios: List[Tuple[str, str, str]]) -> pd.DataFrame:
        """Create ratio features (division with zero handling)."""
        self.log(f"Creating {len(ratios)} ratio features")
        for numerator, denominator, name in ratios:
            if numerator in df.columns and denominator in df.columns:
                # Avoid division by zero
                df[name] = df[numerator] / (df[denominator] + 1e-6)
                self.log(f"  Created: {name}")
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                  columns: List[str], 
                                  degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for specified columns."""
        self.log(f"Creating polynomial features (degree={degree})")
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                for d in range(2, degree + 1):
                    df[f"{col}_pow{d}"] = df[col] ** d
                    self.log(f"  Created: {col}_pow{d}")
        return df
    
    def encode_target_labels(self, df: pd.DataFrame, 
                           target_cols: Dict[str, str]) -> pd.DataFrame:
        """Encode target variables using LabelEncoder."""
        self.log(f"Encoding {len(target_cols)} target variables")
        
        for task_name, col_name in target_cols.items():
            if col_name in df.columns:
                # Handle missing values in target (fill with 'Unknown')
                df[col_name] = df[col_name].fillna('Unknown')
                
                # Encode
                le = LabelEncoder()
                df[f"{col_name}_encoded"] = le.fit_transform(df[col_name])
                self.label_encoders[col_name] = le
                
                # Log mapping
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                self.log(f"  {col_name}: {mapping}")
        
        return df
    
    def frequency_encoding(self, df: pd.DataFrame, 
                          columns: List[str]) -> pd.DataFrame:
        """Apply frequency encoding to high-cardinality categorical features."""
        self.log(f"Applying frequency encoding to {len(columns)} columns")
        for col in columns:
            if col in df.columns:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f"{col}_freq"] = df[col].map(freq_map)
                self.log(f"  Encoded: {col} ({len(freq_map)} unique values)")
        return df
    
    def onehot_encoding(self, df: pd.DataFrame, 
                       columns: List[str]) -> pd.DataFrame:
        """Apply one-hot encoding to low-cardinality features."""
        self.log(f"Applying one-hot encoding to {len(columns)} columns")
        for col in columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                self.log(f"  Encoded: {col} ({len(dummies.columns)} new columns)")
        return df


class DataSplitter:
    """Train/validation/test splitting utilities."""
    
    def __init__(self, train_size: float = 0.7, val_size: float = 0.15, 
                 test_size: float = 0.15, random_state: int = 42):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Validate sizes
        total = train_size + val_size + test_size
        assert abs(total - 1.0) < 0.01, f"Sizes must sum to 1.0 (got {total})"
    
    def split(self, df: pd.DataFrame, 
              stratify_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets."""
        print(f"Splitting data: {self.train_size:.0%} train / {self.val_size:.0%} val / {self.test_size:.0%} test")
        
        # Prepare stratification
        stratify_data = None
        if stratify_col and stratify_col in df.columns:
            # Remove rows with rare classes
            value_counts = df[stratify_col].value_counts()
            valid_values = value_counts[value_counts >= 2].index
            df_filtered = df[df[stratify_col].isin(valid_values)].copy()
            print(f"  Filtered to {len(df_filtered)} rows for stratification on '{stratify_col}'")
            stratify_data = df_filtered[stratify_col]
        else:
            df_filtered = df.copy()
            print(f"  No stratification applied")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df_filtered,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=stratify_data
        )
        
        # Second split: val vs test
        val_ratio = self.val_size / (self.val_size + self.test_size)
        
        if stratify_col and stratify_col in temp_df.columns:
            stratify_temp = temp_df[stratify_col]
        else:
            stratify_temp = None
        
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            random_state=self.random_state,
            stratify=stratify_temp
        )
        
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df


class PreprocessingPipeline:
    """Complete preprocessing pipeline."""
    
    def __init__(self, scaling_method: str = 'standard'):
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_engineer = FeatureEngineer()
        self.numeric_cols = []
        self.fitted = False
        
    def _get_scaler(self):
        """Get scaler based on method."""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def fit_transform(self, df: pd.DataFrame, 
                     numeric_cols: List[str]) -> pd.DataFrame:
        """Fit scaler on training data and transform."""
        print(f"[Pipeline] Fitting {self.scaling_method} scaler on {len(numeric_cols)} numeric features")
        
        self.numeric_cols = numeric_cols
        self.scaler = self._get_scaler()
        
        # Fit and transform
        df_scaled = df.copy()
        df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        self.fitted = True
        
        print(f"[Pipeline] Scaling complete")
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler."""
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        print(f"[Pipeline] Transforming data with fitted scaler")
        df_scaled = df.copy()
        df_scaled[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df_scaled


def handle_missing_values(df: pd.DataFrame, 
                         numeric_strategy: str = 'median',
                         categorical_strategy: str = 'mode',
                         constant: str = 'Unknown') -> pd.DataFrame:
    """Handle missing values in dataset."""
    print(f"[Missing Values] Handling missing data")
    
    df_clean = df.copy()
    missing_before = df.isnull().sum().sum()
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if numeric_strategy == 'mean':
                fill_value = df[col].mean()
            elif numeric_strategy == 'median':
                fill_value = df[col].median()
            else:
                fill_value = 0
            df_clean[col] = df[col].fillna(fill_value)
            print(f"  Filled {col} with {numeric_strategy}")
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            if categorical_strategy == 'mode':
                fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else constant
            else:
                fill_value = constant
            df_clean[col] = df[col].fillna(fill_value)
            print(f"  Filled {col} with '{fill_value}'")
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"  Missing values: {missing_before} → {missing_after}")
    
    return df_clean


def get_feature_importance_data(df: pd.DataFrame, 
                               target_cols: List[str]) -> pd.DataFrame:
    """Prepare basic feature statistics for later importance analysis."""
    print("[Feature Stats] Computing basic statistics")
    
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in target_cols]
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100
        }
    
    stats_df = pd.DataFrame(stats).T
    print(f"  Computed stats for {len(stats_df)} numeric features")
    
    return stats_df
