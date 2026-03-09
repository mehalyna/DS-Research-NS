"""
Week 4: Main Preprocessing Pipeline
Complete preprocessing pipeline execution script.

This script performs:
1. Feature engineering (age buckets, interactions, encoding)
2. Train/validation/test split
3. Data preprocessing and scaling
4. Save cleaned datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Import configuration and utilities
import preprocessing_config as config
from preprocessing_utils import (
    FeatureEngineer,
    DataSplitter,
    PreprocessingPipeline,
    handle_missing_values,
    get_feature_importance_data
)


def main():
    """Main preprocessing pipeline execution."""
    
    print("=" * 80)
    print("WEEK 4: PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"From: {config.RAW_DATA_PATH}")
    
    # Display basic info
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # ========================================================================
    # 2. HANDLE MISSING VALUES
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: HANDLING MISSING VALUES")
    print("=" * 80)
    
    df = handle_missing_values(
        df,
        numeric_strategy=config.NUMERIC_IMPUTATION,
        categorical_strategy=config.CATEGORICAL_IMPUTATION,
        constant=config.MISSING_CONSTANT
    )
    
    # ========================================================================
    # 3. FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    fe = FeatureEngineer(verbose=config.VERBOSE)
    
    # 3.1 Age buckets
    if config.CREATE_AGE_BUCKETS:
        df = fe.create_age_buckets(
            df,
            bins=config.AGE_BINS,
            labels=config.AGE_LABELS
        )
    
    # 3.2 Encode target variables
    df = fe.encode_target_labels(df, config.TARGET_COLUMNS)
    
    # 3.3 Frequency encoding for high-cardinality features
    df = fe.frequency_encoding(df, config.HIGH_CARDINALITY_FEATURES)
    
    # 3.4 One-hot encoding for low-cardinality features
    df = fe.onehot_encoding(df, config.LOW_CARDINALITY_FEATURES)
    
    # 3.5 Create ratio features
    if config.CREATE_RATIOS:
        df = fe.create_ratio_features(df, config.RATIO_FEATURES)
    
    # 3.6 Create interaction features (numeric only)
    if config.CREATE_INTERACTIONS:
        # Filter interaction pairs to only numeric features
        numeric_interactions = [
            (f1, f2) for f1, f2 in config.INTERACTION_PAIRS
            if f1 in config.NUMERIC_FEATURES and f2 in config.NUMERIC_FEATURES
        ]
        df = fe.create_interaction_features(df, numeric_interactions)
    
    # 3.7 Polynomial features (optional)
    if config.CREATE_POLYNOMIAL_FEATURES:
        df = fe.create_polynomial_features(
            df,
            config.NUMERIC_FEATURES,
            degree=config.POLYNOMIAL_DEGREE
        )
    
    print(f"\nFeature engineering complete!")
    print(f"New shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Save feature-engineered dataset
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.FEATURE_ENGINEERED_PATH, index=False)
    print(f"Saved feature-engineered data to: {config.FEATURE_ENGINEERED_PATH}")
    
    # ========================================================================
    # 4. TRAIN/VAL/TEST SPLIT
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: TRAIN/VALIDATION/TEST SPLIT")
    print("=" * 80)
    
    splitter = DataSplitter(
        train_size=config.TRAIN_SIZE,
        val_size=config.VAL_SIZE,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    train_df, val_df, test_df = splitter.split(
        df,
        stratify_col=config.STRATIFY_COLUMN
    )
    
    # ========================================================================
    # 5. PREPROCESSING & SCALING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: PREPROCESSING & SCALING")
    print("=" * 80)
    
    # Identify numeric columns for scaling (exclude targets and non-numeric)
    all_numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID and encoded targets
    exclude_patterns = ['_encoded', 'ID']
    numeric_cols_to_scale = [
        col for col in all_numeric_cols
        if not any(pattern in col for pattern in exclude_patterns)
    ]
    
    print(f"Scaling {len(numeric_cols_to_scale)} numeric features")
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(scaling_method=config.SCALING_METHOD)
    
    # Fit on train and transform all sets
    train_df_scaled = pipeline.fit_transform(train_df, numeric_cols_to_scale)
    val_df_scaled = pipeline.transform(val_df)
    test_df_scaled = pipeline.transform(test_df)
    
    # ========================================================================
    # 6. SAVE PROCESSED DATASETS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: SAVING PROCESSED DATASETS")
    print("=" * 80)
    
    train_df_scaled.to_csv(config.TRAIN_DATA_PATH, index=False)
    val_df_scaled.to_csv(config.VAL_DATA_PATH, index=False)
    test_df_scaled.to_csv(config.TEST_DATA_PATH, index=False)
    
    print(f"✓ Train set saved: {config.TRAIN_DATA_PATH} ({len(train_df_scaled)} rows)")
    print(f"✓ Val set saved:   {config.VAL_DATA_PATH} ({len(val_df_scaled)} rows)")
    print(f"✓ Test set saved:  {config.TEST_DATA_PATH} ({len(test_df_scaled)} rows)")
    
    # ========================================================================
    # 7. SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: SUMMARY STATISTICS")
    print("=" * 80)
    
    # Feature statistics
    stats_df = get_feature_importance_data(
        train_df_scaled,
        target_cols=[col + '_encoded' for col in config.TARGET_COLUMNS.values()]
    )
    
    stats_path = config.PROCESSED_DATA_DIR / "feature_statistics.csv"
    stats_df.to_csv(stats_path)
    print(f"✓ Feature statistics saved: {stats_path}")
    
    # Target distribution in each set
    print("\nTarget distributions:")
    for target_name, target_col in config.TARGET_COLUMNS.items():
        encoded_col = f"{target_col}_encoded"
        if encoded_col in train_df_scaled.columns:
            print(f"\n{target_name.upper()} ({target_col}):")
            print(f"  Train: {train_df_scaled[encoded_col].value_counts().to_dict()}")
            print(f"  Val:   {val_df_scaled[encoded_col].value_counts().to_dict()}")
            print(f"  Test:  {test_df_scaled[encoded_col].value_counts().to_dict()}")
    
    # Feature engineering summary
    print(f"\n" + "=" * 80)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 80)
    print(f"Original features: {len(config.NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES) + len(config.BINARY_FEATURES)}")
    print(f"Engineered features: {df.shape[1] - len(pd.read_csv(config.RAW_DATA_PATH).columns)}")
    print(f"Total features: {df.shape[1]}")
    
    new_features = set(df.columns) - set(pd.read_csv(config.RAW_DATA_PATH).columns)
    print(f"\nNew features created: {len(new_features)}")
    for feat in sorted(new_features):
        print(f"  - {feat}")
    
    # ========================================================================
    # 8. COMPLETION
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {config.PROCESSED_DATA_DIR}")
    print("\nFiles created:")
    print(f"  1. {config.FEATURE_ENGINEERED_PATH.name}")
    print(f"  2. {config.TRAIN_DATA_PATH.name}")
    print(f"  3. {config.VAL_DATA_PATH.name}")
    print(f"  4. {config.TEST_DATA_PATH.name}")
    print(f"  5. feature_statistics.csv")
    print("\n✓ Ready for Week 5: Baseline modeling!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
