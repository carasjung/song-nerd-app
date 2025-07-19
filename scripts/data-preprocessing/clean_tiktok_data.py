import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'mode', 'key',
                  'speechiness', 'acousticness', 'instrumentalness',
                  'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

# Expected ranges for validation
AUDIO_RANGES = {
    'danceability': (0, 1),
    'energy': (0, 1),
    'loudness': (-60, 0),
    'speechiness': (0, 1),
    'acousticness': (0, 1),
    'instrumentalness': (0, 1),
    'liveness': (0, 1),
    'valence': (0, 1),
    'tempo': (0, 300),  # Increased upper bound for extreme cases
    'duration_ms': (1000, 600000),  # 1 second to 10 minutes reasonable range
    'mode': (0, 1),
    'key': (0, 11),
    'time_signature': (3, 7)  # Common time signatures
}

def load_and_label(path: str, year: str) -> pd.DataFrame:
    """Load CSV and add source year label with error handling."""
    try:
        df = pd.read_csv(path)
        df['source_year'] = year
        logging.info(f"Loaded {len(df)} records from {year}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        raise

def detect_duplicates_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced duplicate detection with fuzzy matching for track names."""
    # Normalize text for better duplicate detection
    df['track_normalized'] = df['track_name'].str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)
    df['artist_normalized'] = df['artist_name'].str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)
    
    # Drop duplicates based on normalized names
    before = len(df)
    df_clean = df.drop_duplicates(subset=['track_normalized', 'artist_normalized'], keep='first')
    
    # Remove temporary columns
    df_clean = df_clean.drop(['track_normalized', 'artist_normalized'], axis=1)
    
    duplicates_removed = before - len(df_clean)
    logging.info(f"Removed {duplicates_removed} duplicate rows ({duplicates_removed/before*100:.2f}%)")
    
    return df_clean

def validate_audio_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean audio feature ranges with detailed reporting."""
    outliers_summary = {}
    
    for feature in AUDIO_FEATURES:
        if feature not in df.columns:
            logging.warning(f"Feature '{feature}' not found in dataset")
            continue
            
        if feature in AUDIO_RANGES:
            min_val, max_val = AUDIO_RANGES[feature]
            
            # Count outliers
            outliers = ((df[feature] < min_val) | (df[feature] > max_val)).sum()
            if outliers > 0:
                outliers_summary[feature] = outliers
                logging.info(f"Found {outliers} outliers in {feature}")
            
            # Clip values
            df[feature] = df[feature].clip(min_val, max_val)
    
    if outliers_summary:
        total_outliers = sum(outliers_summary.values())
        logging.info(f"Total outliers corrected: {total_outliers}")
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive missing value handling with reporting."""
    missing_report = {}
    
    # Handle popularity scores
    if 'track_pop' in df.columns:
        missing_pop = df['track_pop'].isna().sum()
        if missing_pop > 0:
            missing_report['track_pop'] = missing_pop
            df['track_pop'] = pd.to_numeric(df['track_pop'], errors='coerce')
            median_pop = df['track_pop'].median()
            df['track_pop'] = df['track_pop'].fillna(median_pop)
            logging.info(f"Filled {missing_pop} missing popularity scores with median ({median_pop})")
    
    # Handle missing audio features
    for feature in AUDIO_FEATURES:
        if feature in df.columns:
            missing_count = df[feature].isna().sum()
            if missing_count > 0:
                missing_report[feature] = missing_count
                
                # Use feature-specific strategies
                if feature in ['mode', 'key', 'time_signature']:
                    # Use mode for categorical features
                    fill_value = df[feature].mode().iloc[0] if not df[feature].mode().empty else 0
                else:
                    # Use median for continuous features
                    fill_value = df[feature].median()
                
                df[feature] = df[feature].fillna(fill_value)
                logging.info(f"Filled {missing_count} missing values in {feature} with {fill_value}")
    
    return df

def generate_yearly_comparison(df: pd.DataFrame) -> dict:
    """Generate year-over-year trend analysis."""
    if 'source_year' not in df.columns:
        logging.warning("No source_year column found for trend analysis")
        return {}
    
    trends = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in AUDIO_FEATURES + ['track_pop']:
            yearly_means = df.groupby('source_year')[col].mean()
            if len(yearly_means) >= 2:
                years = sorted(yearly_means.index)
                change = yearly_means[years[-1]] - yearly_means[years[0]]
                percent_change = (change / yearly_means[years[0]]) * 100 if yearly_means[years[0]] != 0 else 0
                trends[col] = {
                    'change': change,
                    'percent_change': percent_change,
                    'direction': 'increase' if change > 0 else 'decrease' if change < 0 else 'stable'
                }
    
    return trends

def quality_check(df: pd.DataFrame) -> dict:
    """Perform data quality checks."""
    quality_report = {
        'total_records': len(df),
        'duplicate_check': df.duplicated().sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'years_present': sorted(df['source_year'].unique()) if 'source_year' in df.columns else []
    }
    
    return quality_report

def clean_tiktok_combined():
    """Main cleaning function with comprehensive reporting."""
    try:
        df_2021 = load_and_label("data/raw/market/tiktok_songs_2021.csv", "2021")
        df_2022 = load_and_label("data/raw/market/tiktok_songs_2022.csv", "2022")
        
        combined = pd.concat([df_2021, df_2022], ignore_index=True)
        logging.info(f"Combined datasets: {len(combined)} total records")
        
        # Enhanced duplicate removal
        combined = detect_duplicates_advanced(combined)
        
        # Handle missing values
        combined = handle_missing_values(combined)
        
        # Validate audio feature ranges
        combined = validate_audio_ranges(combined)
        
        # Generate trend analysis
        trends = generate_yearly_comparison(combined)
        if trends:
            logging.info("Year-over-year trends:")
            for feature, trend in trends.items():
                logging.info(f"  {feature}: {trend['direction']} ({trend['percent_change']:.2f}%)")
        
        # Quality check
        quality_report = quality_check(combined)
        logging.info(f"Final dataset: {quality_report['total_records']} records across {len(quality_report['years_present'])} years")
        
        # Save cleaned data
        output_dir = Path("data/cleaned_market")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "tiktok_combined_clean.csv"
        combined.to_csv(output_path, index=False)
        logging.info(f"Saved cleaned data to {output_path}")
        
        # Save quality report
        quality_path = output_dir / "tiktok_cleaning_report.txt"
        with open(quality_path, 'w') as f:
            f.write("TikTok Data Cleaning Report\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total records: {quality_report['total_records']}\n")
            f.write(f"Years: {', '.join(map(str, quality_report['years_present']))}\n\n")
            
            if trends:
                f.write("Year-over-Year Trends:\n")
                for feature, trend in trends.items():
                    f.write(f"  {feature}: {trend['direction']} ({trend['percent_change']:.2f}%)\n")
        
        logging.info(f"Saved cleaning report to {quality_path}")
        return combined
        
    except Exception as e:
        logging.error(f"Error in cleaning process: {e}")
        raise

if __name__ == "__main__":
    cleaned_data = clean_tiktok_combined()