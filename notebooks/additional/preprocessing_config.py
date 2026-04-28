"""
Week 4: Preprocessing Configuration
Configuration parameters for feature engineering and preprocessing pipelines.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "synthetic_coffee_health_10000.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output files
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
FEATURE_ENGINEERED_PATH = PROCESSED_DATA_DIR / "features_engineered.csv"

# ============================================================================
# TARGET VARIABLES
# ============================================================================
TARGET_COLUMNS = {
    'sleep_quality': 'Sleep_Quality',
    'stress_level': 'Stress_Level',
    'health_issues': 'Health_Issues'
}

# ============================================================================
# FEATURE GROUPS
# ============================================================================
NUMERIC_FEATURES = [
    'Age',
    'Coffee_Intake',
    'Caffeine_mg',
    'Sleep_Hours',
    'BMI',
    'Heart_Rate',
    'Physical_Activity_Hours'
]

CATEGORICAL_FEATURES = [
    'Gender',
    'Country',
    'Occupation'
]

BINARY_FEATURES = [
    'Smoking',
    'Alcohol_Consumption'
]

# Features to exclude from modeling (ID and targets)
EXCLUDE_FEATURES = ['ID'] + list(TARGET_COLUMNS.values())

# ============================================================================
# AGE BUCKETS
# ============================================================================
AGE_BINS = [0, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']

# ============================================================================
# INTERACTION FEATURES
# ============================================================================
INTERACTION_PAIRS = [
    ('Coffee_Intake', 'Caffeine_mg'),
    ('Coffee_Intake', 'Sleep_Hours'),
    ('BMI', 'Physical_Activity_Hours'),
    ('Caffeine_mg', 'Heart_Rate'),
    ('Age', 'Coffee_Intake'),
    ('Stress_Level', 'Sleep_Hours')
]

# ============================================================================
# ENCODING CONFIG
# ============================================================================
# High-cardinality features for target/frequency encoding
HIGH_CARDINALITY_FEATURES = ['Country', 'Occupation']

# Low-cardinality features for one-hot encoding
LOW_CARDINALITY_FEATURES = ['Gender']

# ============================================================================
# DATA SPLITTING
# ============================================================================
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Stratification column (use primary target)
STRATIFY_COLUMN = 'Sleep_Quality'

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
# Scaling method: 'standard', 'minmax', 'robust'
SCALING_METHOD = 'standard'

# Handle missing values
NUMERIC_IMPUTATION = 'median'  # 'mean', 'median', 'mode'
CATEGORICAL_IMPUTATION = 'mode'  # 'mode', 'constant'
MISSING_CONSTANT = 'Unknown'

# ============================================================================
# FEATURE ENGINEERING FLAGS
# ============================================================================
CREATE_AGE_BUCKETS = True
CREATE_INTERACTIONS = True
CREATE_POLYNOMIAL_FEATURES = False
POLYNOMIAL_DEGREE = 2
CREATE_RATIOS = True

# Ratio features to create
RATIO_FEATURES = [
    ('Caffeine_mg', 'Coffee_Intake', 'caffeine_per_cup'),
    ('Sleep_Hours', 'Coffee_Intake', 'sleep_per_cup'),
    ('Physical_Activity_Hours', 'Age', 'activity_age_ratio'),
    ('BMI', 'Physical_Activity_Hours', 'bmi_activity_ratio')
]

# ============================================================================
# VALIDATION
# ============================================================================
# Minimum samples per class for stratification
MIN_SAMPLES_PER_CLASS = 10

# Check for data leakage
CHECK_LEAKAGE = True

# ============================================================================
# LOGGING
# ============================================================================
VERBOSE = True
LOG_FILE = PROCESSED_DATA_DIR / "preprocessing.log"
