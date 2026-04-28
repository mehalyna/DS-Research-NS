"""
Simple CSV Viewer
Usage: python view_csv.py [csv_file_path]
"""

import pandas as pd
import sys
from pathlib import Path


def view_csv(file_path, max_rows=None, max_cols=None):
    """Display CSV file in a readable format."""
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Set display options
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_cols)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.precision', 4)
    
    # Display file info
    print("=" * 80)
    print(f"FILE: {Path(file_path).name}")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Display the dataframe
    print(df.to_string())
    
    # Display summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Numeric columns: {len(df.select_dtypes(include='number').columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to feature statistics
        file_path = "../data/processed/feature_statistics.csv"
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    view_csv(file_path)
