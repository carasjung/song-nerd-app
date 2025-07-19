import pandas as pd
import numpy as np
import os
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_preprocessing_pipeline(input_file='integration_cache/master_music_dataset_deduplicated.csv', 
                                output_dir='final_datasets'):
    """Debug version of preprocessing pipeline to find issues"""
    
    print("Debug preprocessing pipeline")
    
    # Check input file
    print(f"Checking input file...")
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    file_size = os.path.getsize(input_file) / (1024*1024)  # MB
    print(f"Input file found: {input_file} ({file_size:.1f} MB)")
    
    # Load dataset
    print(f"Loading dataset...")
    try:
        df = pd.read_csv(input_file)
        print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Check output directory
    print(f"Checking output directory...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory ready: {output_dir}")
    except Exception as e:
        print(f"Errorreating output directory: {e}")
        return False
    
    # Create simple test dataset
    print(f"Creating test dataset...")
    try:
        # Create a simple test dataset with just essential columns
        test_columns = []
        
        # Check for essential columns
        required_cols = ['track_name_clean', 'artist_name_clean', 'genre_clean']
        for col in required_cols:
            if col in df.columns:
                test_columns.append(col)
            elif col.replace('_clean', '') in df.columns:
                # Use original column if clean version doesn't exist
                orig_col = col.replace('_clean', '')
                df[col] = df[orig_col]  # Create clean version
                test_columns.append(col)
        
        # Add audio features
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'normalized_popularity']
        for col in audio_features:
            if col in df.columns:
                test_columns.append(col)
        
        # Add platform columns
        platform_cols = ['spotify', 'tiktok', 'youtube']
        for col in platform_cols:
            if col in df.columns:
                test_columns.append(col)
        
        print(f"   Available columns for test: {test_columns}")
        
        if test_columns:
            test_df = df[test_columns].copy()
            test_file = os.path.join(output_dir, 'test_dataset.csv')
            test_df.to_csv(test_file, index=False)
            print(f"Test dataset created: {test_file} ({len(test_df):,} rows)")
        else:
            print(f"No suitable columns found for test dataset")
            return False
            
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        return False
    
    # Create minimal master dataset
    print(f"Creating minimal master dataset...")
    try:
        # Create master dataset with available columns
        master_columns = test_columns.copy()
        
        # Add any additional columns that exist
        additional_cols = ['data_quality_score', 'source_count', 'primary_platform']
        for col in additional_cols:
            if col in df.columns:
                master_columns.append(col)
        
        master_df = df[master_columns].copy()
        master_file = os.path.join(output_dir, 'master_music_data.csv')
        master_df.to_csv(master_file, index=False)
        print(f"Master dataset created: {master_file}")
        print(f"   Rows: {len(master_df):,}, Columns: {len(master_df.columns)}")
        
    except Exception as e:
        print(f"Error creating master dataset: {e}")
        return False
    
    # Create simple platform performance dataset
    print(f"Creating platform performance dataset...")
    try:
        platform_data = []
        
        # Sample first 1000 rows for speed
        sample_df = df.head(1000)
        
        for _, track in sample_df.iterrows():
            track_name = track.get('track_name_clean', track.get('track_name', 'Unknown'))
            artist_name = track.get('artist_name_clean', track.get('artist_name', 'Unknown'))
            
            platforms = ['spotify', 'tiktok', 'youtube']
            for platform in platforms:
                if track.get(platform, False):
                    platform_data.append({
                        'track_name': track_name,
                        'artist_name': artist_name,
                        'platform': platform,
                        'popularity_score': track.get('normalized_popularity', 0)
                    })
        
        if platform_data:
            platform_df = pd.DataFrame(platform_data)
            platform_file = os.path.join(output_dir, 'platform_performance.csv')
            platform_df.to_csv(platform_file, index=False)
            print(f"Platform performance dataset created: {platform_file}")
            print(f"   Rows: {len(platform_df):,}")
        else:
            print(f"No platform data found")
            
    except Exception as e:
        print(f"Error creating platform dataset: {e}")
        return False
    
    # List all created files
    print(f"Listing created files...")
    try:
        created_files = []
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                filepath = os.path.join(output_dir, file)
                file_size = os.path.getsize(filepath) / 1024  # KB
                created_files.append((file, file_size))
        
        if created_files:
            print(f"Created {len(created_files)} files:")
            for filename, size in created_files:
                print(f"   • {filename} ({size:.1f} KB)")
        else:
            print(f"No files were created")
            return False
            
    except Exception as e:
        print(f"Error listing files: {e}")
        return False
    
    print(f"\nDebug preprocessing complete")
    print(f"All steps completed successfully")
    print(f"Check {output_dir}/ for created files")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Get arguments
    input_file = 'integration_cache/master_music_dataset_deduplicated.csv'
    output_dir = 'final_datasets'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Run debug pipeline
    success = debug_preprocessing_pipeline(input_file, output_dir)
    
    if success:
        print(f"\nIf this worked, the issue was in the complex preprocessing logic.")
        print(f"Try running the full pipeline again, or use these simpler datasets.")
    else:
        print(f"\nDebug pipeline failed. Check the error messages above.")