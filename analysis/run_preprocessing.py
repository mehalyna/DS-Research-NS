"""
Quick Start Script for Week 4 Preprocessing
============================================

This script provides a simple interface to run the preprocessing pipeline.

Usage:
    python run_preprocessing.py [options]

Options:
    --help          Show this help message
    --config        Show current configuration
    --notebook      Open the preprocessing notebook
    --clean         Remove existing processed data files
"""

import sys
from pathlib import Path


def print_help():
    """Print help message."""
    print(__doc__)


def show_config():
    """Display current configuration."""
    import preprocessing_config as config
    
    print("=" * 80)
    print("PREPROCESSING CONFIGURATION")
    print("=" * 80)
    print(f"\nData Paths:")
    print(f"  Raw data: {config.RAW_DATA_PATH}")
    print(f"  Output dir: {config.PROCESSED_DATA_DIR}")
    
    print(f"\nData Splitting:")
    print(f"  Train: {config.TRAIN_SIZE:.0%}")
    print(f"  Val:   {config.VAL_SIZE:.0%}")
    print(f"  Test:  {config.TEST_SIZE:.0%}")
    print(f"  Random state: {config.RANDOM_STATE}")
    print(f"  Stratify on: {config.STRATIFY_COLUMN}")
    
    print(f"\nFeature Engineering:")
    print(f"  Age buckets: {config.CREATE_AGE_BUCKETS}")
    print(f"  Interactions: {config.CREATE_INTERACTIONS}")
    print(f"  Ratios: {config.CREATE_RATIOS}")
    print(f"  Polynomial: {config.CREATE_POLYNOMIAL_FEATURES}")
    
    print(f"\nPreprocessing:")
    print(f"  Scaling method: {config.SCALING_METHOD}")
    print(f"  Numeric imputation: {config.NUMERIC_IMPUTATION}")
    print(f"  Categorical imputation: {config.CATEGORICAL_IMPUTATION}")
    
    print(f"\nTarget Variables:")
    for name, col in config.TARGET_COLUMNS.items():
        print(f"  {name}: {col}")
    
    print("\n" + "=" * 80)


def open_notebook():
    """Open the preprocessing notebook."""
    import subprocess
    
    notebook_path = Path(__file__).parent / "02_preprocessing.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    print(f"Opening notebook: {notebook_path}")
    
    try:
        subprocess.run(["jupyter", "notebook", str(notebook_path)])
        return True
    except FileNotFoundError:
        print("❌ Jupyter not found. Install with: pip install jupyter")
        return False


def clean_processed_data():
    """Remove existing processed data files."""
    import preprocessing_config as config
    import shutil
    
    if config.PROCESSED_DATA_DIR.exists():
        response = input(f"Delete all files in {config.PROCESSED_DATA_DIR}? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            shutil.rmtree(config.PROCESSED_DATA_DIR)
            print(f"✓ Removed {config.PROCESSED_DATA_DIR}")
            return True
        else:
            print("Cancelled.")
            return False
    else:
        print(f"Directory does not exist: {config.PROCESSED_DATA_DIR}")
        return False


def run_pipeline():
    """Run the main preprocessing pipeline."""
    print("\n" + "=" * 80)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 80 + "\n")
    
    try:
        # Import and run the main pipeline
        from preprocess_pipeline import main
        main()
        
        print("\n" + "=" * 80)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    if not args or args[0] == '--help':
        print_help()
        return
    
    if args[0] == '--config':
        show_config()
        return
    
    if args[0] == '--notebook':
        open_notebook()
        return
    
    if args[0] == '--clean':
        clean_processed_data()
        return
    
    # Default: run preprocessing
    success = run_pipeline()
    
    if success:
        print("\n📊 Next steps:")
        print("  1. Review processed data in data/processed/")
        print("  2. Check feature_statistics.csv for feature info")
        print("  3. Proceed to Week 5: Baseline model training")
        print("\nFor interactive exploration:")
        print("  python run_preprocessing.py --notebook")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
