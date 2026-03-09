# Week 4: Preprocessing Scripts - README

## Overview

This directory contains the complete preprocessing pipeline for Week 4 of the DS-Research-NS project. The scripts implement feature engineering, train/validation/test splitting, and preprocessing pipelines as specified in the roadmap.

## Files Created

### 1. **preprocessing_config.py**
Configuration file containing all parameters for the preprocessing pipeline:
- File paths and directory structure
- Target variable definitions
- Feature groups (numeric, categorical, binary)
- Age bucket definitions
- Interaction feature pairs
- Encoding strategies
- Data splitting ratios
- Scaling methods
- Feature engineering flags

### 2. **preprocessing_utils.py**
Utility module with reusable classes and functions:
- `FeatureEngineer`: Feature engineering operations
  - Age bucketing
  - Interaction features
  - Ratio features
  - Polynomial features
  - Target encoding
  - Frequency encoding
  - One-hot encoding
- `DataSplitter`: Train/val/test splitting with stratification
- `PreprocessingPipeline`: Scaling and preprocessing
- `handle_missing_values()`: Missing value imputation
- `get_feature_importance_data()`: Feature statistics computation

### 3. **preprocess_pipeline.py**
Main executable script that runs the complete pipeline:
1. Load raw data
2. Handle missing values
3. Feature engineering
4. Train/val/test split
5. Preprocessing & scaling
6. Save processed datasets
7. Generate summary statistics

### 4. **02_preprocessing.ipynb**
Interactive Jupyter notebook demonstrating the preprocessing pipeline:
- Step-by-step execution with visualizations
- Feature engineering examples
- Data split visualization
- Scaling comparisons
- Target distribution analysis

## Quick Start

### Option 1: Run the Python Script (Automated)

```powershell
# Navigate to analysis directory
cd analysis

# Run the preprocessing pipeline
python preprocess_pipeline.py
```

This will:
- Process the raw data
- Create all engineered features
- Split into train/val/test sets
- Apply scaling
- Save processed datasets to `data/processed/`

### Option 2: Use the Jupyter Notebook (Interactive)

```powershell
# Start Jupyter
jupyter notebook 02_preprocessing.ipynb
```

Run the cells sequentially to see the preprocessing steps with visualizations.

### Option 3: Import as a Module

```python
import preprocessing_config as config
from preprocessing_utils import FeatureEngineer, DataSplitter

# Load data
import pandas as pd
df = pd.read_csv(config.RAW_DATA_PATH)

# Apply feature engineering
fe = FeatureEngineer()
df = fe.create_age_buckets(df, config.AGE_BINS, config.AGE_LABELS)
df = fe.create_ratio_features(df, config.RATIO_FEATURES)

# Split data
splitter = DataSplitter()
train, val, test = splitter.split(df, stratify_col='Sleep_Quality')
```

## Output Files

After running the preprocessing pipeline, the following files are created in `data/processed/`:

1. **features_engineered.csv** - Full dataset with all engineered features (before splitting)
2. **train.csv** - Training set (70%, scaled)
3. **val.csv** - Validation set (15%, scaled)
4. **test.csv** - Test set (15%, scaled)
5. **feature_statistics.csv** - Statistical summary of all features

## Features Created

### Age Buckets
- **Age_Bucket**: Categorical age groups (18-25, 26-35, 36-45, 46-55, 56-65, 65+)

### Ratio Features
- **caffeine_per_cup**: Caffeine_mg / Coffee_Intake
- **sleep_per_cup**: Sleep_Hours / Coffee_Intake
- **activity_age_ratio**: Physical_Activity_Hours / Age
- **bmi_activity_ratio**: BMI / Physical_Activity_Hours

### Interaction Features
- **Coffee_Intake_x_Caffeine_mg**
- **Coffee_Intake_x_Sleep_Hours**
- **BMI_x_Physical_Activity_Hours**
- **Caffeine_mg_x_Heart_Rate**
- **Age_x_Coffee_Intake**

### Encoding Features
- **Country_freq**: Frequency encoding for country
- **Occupation_freq**: Frequency encoding for occupation
- **Gender_***: One-hot encoded gender columns
- **Sleep_Quality_encoded**: Label-encoded target
- **Stress_Level_encoded**: Label-encoded target
- **Health_Issues_encoded**: Label-encoded target

## Configuration Options

You can modify `preprocessing_config.py` to customize:

### Data Splitting
```python
TRAIN_SIZE = 0.7    # 70% for training
VAL_SIZE = 0.15     # 15% for validation
TEST_SIZE = 0.15    # 15% for testing
RANDOM_STATE = 42   # For reproducibility
```

### Scaling Method
```python
SCALING_METHOD = 'standard'  # Options: 'standard', 'minmax', 'robust'
```

### Feature Engineering Flags
```python
CREATE_AGE_BUCKETS = True
CREATE_INTERACTIONS = True
CREATE_RATIOS = True
CREATE_POLYNOMIAL_FEATURES = False  # Set to True to enable
POLYNOMIAL_DEGREE = 2
```

### Missing Value Strategy
```python
NUMERIC_IMPUTATION = 'median'       # 'mean', 'median', or value
CATEGORICAL_IMPUTATION = 'mode'     # 'mode' or 'constant'
MISSING_CONSTANT = 'Unknown'        # Used if strategy is 'constant'
```

## Target Variables

The pipeline handles three target variables for multi-task learning:

1. **Sleep_Quality**: Sleep quality classification (Poor, Fair, Good, Excellent)
2. **Stress_Level**: Stress level classification (Low, Medium, High)
3. **Health_Issues**: Health issues classification (None, Mild, Moderate, Severe)

All targets are label-encoded (ordinal encoding) and saved with `_encoded` suffix.

## Data Validation

The pipeline includes validation checks:
- Missing value detection and handling
- Stratification validation (minimum samples per class)
- Data leakage prevention (fit only on train, transform on val/test)
- Feature statistics computation

## Troubleshooting

### ImportError: No module named 'preprocessing_config'
Make sure you're running from the `analysis/` directory or add it to your Python path:
```python
import sys
sys.path.append('path/to/analysis')
```

### Missing data directory
The script will create the `data/processed/` directory automatically.

### Memory issues
If you encounter memory issues with large datasets:
- Reduce the number of polynomial features
- Disable interaction features
- Use chunking for very large datasets

## Next Steps (Week 5)

After completing preprocessing, you're ready for:
1. **Baseline model training** (Week 5)
2. **Model evaluation** and comparison
3. **Feature importance analysis**

The processed datasets are ready to be loaded directly into scikit-learn or LightGBM models:

```python
import pandas as pd

# Load processed data
train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')

# Separate features and targets
X_train = train.drop(columns=['Sleep_Quality_encoded', 'Stress_Level_encoded', 'Health_Issues_encoded'])
y_train = train[['Sleep_Quality_encoded', 'Stress_Level_encoded', 'Health_Issues_encoded']]
```


```
# Feature statistics
python view_csv_html.py ..\data\processed\feature_statistics.csv

# Training data (first 1000 rows recommended for large files)
python view_csv.py ..\data\processed\train.csv

# Validation data
python view_csv_html.py ..\data\processed\val.csv
```


## Contact & Support

For issues or questions about the preprocessing pipeline, refer to:
- Main project README: `../README.md`
- Project roadmap: `../ROADMAP.md`
- Week 3 EDA notebook: `01_EDA.ipynb`

---

**Deliverable Status**: ✓ Complete

- [x] Feature engineering (age buckets, interactions, encoding)
- [x] Train/validation/test split with stratification
- [x] Preprocessing pipelines with scaling
- [x] Cleaned datasets saved
- [x] Documentation and reproducibility
