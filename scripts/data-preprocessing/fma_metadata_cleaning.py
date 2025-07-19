# fma_metadata_cleaning.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_music_metadata():
    """Clean and standardize music metadata with comprehensive data validation"""
    input_path = "data/raw/music/million-spotify-lastfm/music_info.csv"
    output_dir = Path("data/cleaned_music")
    output_path = output_dir / "music_info_clean.csv"
    
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded {len(df)} music records with {len(df.columns)} columns")
        
        initial_rows = len(df)
        
        # Remove duplicates based on track_id
        if 'track_id' in df.columns:
            df = df.drop_duplicates(subset=['track_id'], keep='first')
            logging.info(f"Removed {initial_rows - len(df)} duplicate tracks")
        
        # Clean basic metadata
        df = clean_basic_metadata(df)
        
        # Clean audio features
        df = clean_audio_features(df)
        
        # Clean categorical data
        df = clean_categorical_data(df)
        
        # Add derived features
        df = add_derived_features(df)
        
        # Final data validation
        df = validate_data(df)
        
        # Save cleaned data
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Cleaned metadata saved to {output_path}")
        
        generate_summary_report(df, output_dir)
        
        return df
        
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        raise
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise

def clean_basic_metadata(df):
    """Clean basic track metadata"""
    logging.info("Cleaning basic metadata...")
    
    # Clean track names
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str).str.strip()
        df['name'] = df['name'].replace('nan', np.nan)
        df['name_length'] = df['name'].str.len()
        logging.info("Cleaned track names")
    
    # Clean artist names
    if 'artist' in df.columns:
        df['artist'] = df['artist'].astype(str).str.strip()
        df['artist'] = df['artist'].replace('nan', np.nan)
        
        # Handle multi-artist tracks
        df['artist_count'] = df['artist'].apply(
            lambda x: len(re.split(r'[,&;]|\s+feat\.?\s+|\s+ft\.?\s+', str(x))) 
            if pd.notna(x) and x != 'nan' else 1
        )
        
        multi_artist = df[df['artist_count'] > 1]
        logging.info(f"Found {len(multi_artist)} tracks with multiple artists")
        
        # Compute artist popularity
        artist_popularity = df['artist'].value_counts().to_dict()
        df['artist_track_count'] = df['artist'].map(artist_popularity)
        logging.info("Computed artist popularity metrics")
    
    # Clean year data
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Set year bounds from 1900-2025
        df.loc[(df['year'] < 1900) | (df['year'] > 2025), 'year'] = np.nan
        
        # Create decade feature
        df['decade'] = (df['year'] // 10 * 10).astype('Int64')
        
        year_missing = df['year'].isna().sum()
        logging.info(f"Cleaned year data - {year_missing} missing values")
    
    # Clean duration
    if 'duration_ms' in df.columns:
        df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')
        
        # Remove unreasonable durations (< 10 seconds or > 30 minutes)
        df.loc[(df['duration_ms'] < 10000) | (df['duration_ms'] > 1800000), 'duration_ms'] = np.nan
        
        # Convert to minutes 
        df['duration_min'] = df['duration_ms'] / 60000
        
        duration_missing = df['duration_ms'].isna().sum()
        logging.info(f"Cleaned duration data - {duration_missing} missing/invalid values")
    
    return df

def clean_audio_features(df):
    """Clean Spotify audio features"""
    logging.info("Cleaning audio features...")
    
    # Features that should be between 0 and 1
    normalized_features = [
        'danceability', 'energy', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence'
    ]
    
    for feature in normalized_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            
            # Clip values to valid range [0, 1]
            df[feature] = df[feature].clip(0, 1)
            
            invalid_count = df[feature].isna().sum()
            if invalid_count > 0:
                logging.info(f"Cleaned {feature} - {invalid_count} invalid values")
    
    # Clean loudness (typically between -60 and 0 dB)
    if 'loudness' in df.columns:
        df['loudness'] = pd.to_numeric(df['loudness'], errors='coerce')
        df.loc[(df['loudness'] < -60) | (df['loudness'] > 5), 'loudness'] = np.nan
        loudness_missing = df['loudness'].isna().sum()
        logging.info(f"Cleaned loudness - {loudness_missing} invalid values")
    
    # Clean tempo (range: 50-250 BPM)
    if 'tempo' in df.columns:
        df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
        df.loc[(df['tempo'] < 50) | (df['tempo'] > 250), 'tempo'] = np.nan
        
        # Create tempo categories
        df['tempo_category'] = pd.cut(
            df['tempo'], 
            bins=[0, 90, 120, 140, 200, 300], 
            labels=['Slow', 'Moderate', 'Fast', 'Very Fast', 'Extreme'],
            include_lowest=True
        )
        
        tempo_missing = df['tempo'].isna().sum()
        logging.info(f"Cleaned tempo - {tempo_missing} invalid values")
    
    return df

def clean_categorical_data(df):
    """Clean categorical features"""
    logging.info("Cleaning categorical data...")
    
    # Clean genre data
    if 'genre' in df.columns:
        df['genre'] = df['genre'].astype(str).str.lower().str.strip()
        df['genre'] = df['genre'].replace('nan', np.nan)
        
        # Remove special characters and normalize
        df['genre'] = df['genre'].str.replace(r'[^\w\s,]', '', regex=True)
        df['genre'] = df['genre'].str.replace(r'\s+', ' ', regex=True)
        
        # Count genres per track
        df['genre_count'] = df['genre'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'nan' else 0
        )
        
        logging.info("Cleaned and parsed genre data")
    
    # Clean tags
    if 'tags' in df.columns:
        df['tags'] = df['tags'].astype(str).str.lower().str.strip()
        df['tags'] = df['tags'].replace('nan', np.nan)
        
        # Count tags per track
        df['tag_count'] = df['tags'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'nan' else 0
        )
        
        logging.info("Cleaned tag data")
    
    # Clean key and mode
    if 'key' in df.columns:
        df['key'] = pd.to_numeric(df['key'], errors='coerce')
        df['key'] = df['key'].astype('Int64')  # Nullable integer
    
    if 'mode' in df.columns:
        df['mode'] = pd.to_numeric(df['mode'], errors='coerce')
        df['mode'] = df['mode'].astype('Int64')  # 0 = minor, 1 = major
    
    if 'time_signature' in df.columns:
        df['time_signature'] = pd.to_numeric(df['time_signature'], errors='coerce')
        df['time_signature'] = df['time_signature'].astype('Int64')
    
    return df

def add_derived_features(df):
    """Add useful derived features"""
    logging.info("Adding derived features...")
    
    # Energy-valence quadrants for mood classification
    if 'energy' in df.columns and 'valence' in df.columns:
        conditions = [
            (df['energy'] >= 0.5) & (df['valence'] >= 0.5),
            (df['energy'] >= 0.5) & (df['valence'] < 0.5),
            (df['energy'] < 0.5) & (df['valence'] >= 0.5),
            (df['energy'] < 0.5) & (df['valence'] < 0.5)
        ]
        choices = ['Happy/Energetic', 'Angry/Intense', 'Peaceful/Calm', 'Sad/Melancholic']
        df['mood_category'] = np.select(conditions, choices, default='Unknown')
    
    # Danceability categories
    if 'danceability' in df.columns:
        df['danceability_level'] = pd.cut(
            df['danceability'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
    
    # Acoustic vs Electronic classification
    if 'acousticness' in df.columns and 'energy' in df.columns:
        df['acoustic_electronic'] = np.where(
            df['acousticness'] > 0.6, 'Acoustic',
            np.where(df['energy'] > 0.7, 'Electronic', 'Mixed')
        )
    
    # Data completeness score
    feature_columns = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    available_features = [col for col in feature_columns if col in df.columns]
    
    df['feature_completeness'] = df[available_features].notna().sum(axis=1) / len(available_features)
    
    logging.info(f"Added {len(['mood_category', 'danceability_level', 'acoustic_electronic', 'feature_completeness'])} derived features")
    
    return df

def validate_data(df):
    """Final data validation and quality checks"""
    logging.info("Performing final data validation...")
    
    # Remove rows with critical missing data
    critical_columns = ['track_id', 'name', 'artist']
    available_critical = [col for col in critical_columns if col in df.columns]
    
    initial_count = len(df)
    df = df.dropna(subset=available_critical)
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        logging.info(f"Removed {removed_count} rows with missing critical data")
    
    # Log data quality summary
    total_features = len(df.columns)
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    logging.info(f"Final dataset: {len(df)} tracks, {total_features} features")
    logging.info(f"Numeric features: {len(numeric_features)}, Categorical: {len(categorical_features)}")
    
    return df

def generate_summary_report(df, output_dir):
    """Create data quality summary report"""
    report_path = output_dir / "data_quality_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("MUSIC DATASET CLEANING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total tracks: {len(df)}\n")
        f.write(f"Total features: {len(df.columns)}\n\n")
        
        f.write("MISSING DATA SUMMARY:\n")
        f.write("-" * 20 + "\n")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        for col, missing_count in missing_data.items():
            percentage = (missing_count / len(df)) * 100
            f.write(f"{col}: {missing_count} ({percentage:.1f}%)\n")
        
        f.write("\nFEATURE STATISTICS:\n")
        f.write("-" * 20 + "\n")
        
        # Audio feature statistics
        audio_features = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'speechiness', 'liveness'
        ]
        
        for feature in audio_features:
            if feature in df.columns:
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                f.write(f"{feature}: mean={mean_val:.3f}, std={std_val:.3f}\n")
        
        if 'year' in df.columns:
            year_range = f"{df['year'].min():.0f} - {df['year'].max():.0f}"
            f.write(f"\nYear range: {year_range}\n")
        
        if 'artist_track_count' in df.columns:
            top_artists = df.nlargest(5, 'artist_track_count')[['artist', 'artist_track_count']]
            f.write(f"\nTop 5 artists by track count:\n")
            for _, row in top_artists.iterrows():
                f.write(f"  {row['artist']}: {row['artist_track_count']} tracks\n")
    
    logging.info(f"Data quality report saved to {report_path}")

if __name__ == "__main__":
    try:
        cleaned_df = clean_music_metadata()
        logging.info("Data cleaning completed successfully!")
    except Exception as e:
        logging.error(f"Data cleaning failed: {str(e)}")
        raise