# cross_platform_features_debug.py
# Debugging cross-platform features

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_cross_platform_data():
    """Check what data is available from cross-platform analysis"""
    
    data_dir = Path("data/cross_platform_features")
    
    print("Available files in cross_platform_features:")
    if data_dir.exists():
        for file in data_dir.glob("*.csv"):
            print(f"  - {file.name}")
    else:
        print("  Directory doesn't exist!")
        return
    
    # Check each file's columns
    files_to_check = [
        "unified_cross_platform_dataset.csv",
        "virality_scores.csv", 
        "comprehensive_cross_platform_features.csv",
        "demographic_alignment.csv",
        "platform_affinity_analysis.csv"
    ]
    
    for filename in files_to_check:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                print(f"\n{filename}:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                
                # Check for duplicate columns
                if df.columns.duplicated().any():
                    print(f"  WARNING: Has duplicate columns!")
                    duplicates = df.columns[df.columns.duplicated()].tolist()
                    print(f"  Duplicates: {duplicates}")
                
                # Show first few rows
                print(f"First 3 rows preview:")
                print(df.head(3).to_string())
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"\n{filename}: File not found")

def check_for_virality_score():
    """Specifically check for virality score availability"""
    comp_file = Path("data/cross_platform_features/comprehensive_cross_platform_features.csv")
    if comp_file.exists():
        df = pd.read_csv(comp_file)
        print(f"\nComprehensive features file check:")
        print(f"Has virality_score: {'virality_score' in df.columns}")
        print(f"Has virality_tier: {'virality_tier' in df.columns}")
        print(f"Has platform_diversity: {'platform_diversity' in df.columns}")
        
        if 'virality_score' in df.columns:
            print(f"Virality score stats:")
            print(f"  Non-null count: {df['virality_score'].count()}")
            print(f"  Mean: {df['virality_score'].mean():.2f}")
            print(f"  Range: {df['virality_score'].min():.2f} - {df['virality_score'].max():.2f}")
        
        # Check what columns are available that might be related
        score_cols = [col for col in df.columns if 'score' in col.lower()]
        print(f"Available score columns: {score_cols}")
        
        popularity_cols = [col for col in df.columns if 'popular' in col.lower()]
        print(f"Available popularity columns: {popularity_cols}")
    
    # Check virality scores file
    viral_file = Path("data/cross_platform_features/virality_scores.csv")
    if viral_file.exists():
        df_viral = pd.read_csv(viral_file)
        print(f"\nVirality scores file:")
        print(f"  Shape: {df_viral.shape}")
        print(f"  Columns: {list(df_viral.columns)}")
        print(f"  Sample data:")
        print(df_viral.head(3).to_string())

if __name__ == "__main__":
    print("Checking cross-platform analysis data...")
    check_cross_platform_data()
    print("\n" + "="*50)
    check_for_virality_score()