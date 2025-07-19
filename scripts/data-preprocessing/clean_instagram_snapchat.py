import pandas as pd
import numpy as np
from pathlib import Path
import logging
import chardet
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PLATFORM_COLUMNS = ['platform', 'trend_type', 'genre', 'trend_description', 'start_date', 'end_date']

# Standardized mappings for data consistency
PLATFORM_MAPPINGS = {
    'insta': 'instagram',
    'ig': 'instagram', 
    'snap': 'snapchat',
    'snapchat stories': 'snapchat',
    'instagram stories': 'instagram',
    'instagram reels': 'instagram',
    'reels': 'instagram'
}

TREND_TYPE_MAPPINGS = {
    'filter': 'visual_effect',
    'filters': 'visual_effect',
    'dance challenge': 'dance',
    'dance challenges': 'dance',
    'sound trend': 'audio',
    'audio trend': 'audio',
    'music trend': 'audio',
    'hashtag challenge': 'challenge',
    'hashtag': 'challenge',
    'meme': 'meme_trend',
    'viral video': 'video_trend'
}

GENRE_MAPPINGS = {
    'pop music': 'pop',
    'hip hop': 'hip_hop',
    'hip-hop': 'hip_hop',
    'r&b': 'rnb',
    'r and b': 'rnb',
    'edm': 'electronic',
    'electronic dance music': 'electronic'
}

def detect_encoding(file_path: str) -> str:
    """Detect file encoding with fallback options."""
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            logging.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        logging.warning(f"Encoding detection failed: {e}")
        return None

def load_data(path: str) -> pd.DataFrame:
    """Load CSV with automatic encoding detection and enhanced error handling."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    detected_encoding = detect_encoding(path)
    encodings_to_try = []
    
    if detected_encoding:
        encodings_to_try.append(detected_encoding)
    
    encodings_to_try.extend(['utf-8', 'latin-1', 'iso-8859-1', 'cp1252'])
    encodings_to_try = list(dict.fromkeys(encodings_to_try))  # Remove duplicates
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=encoding)
            logging.info(f"Loaded {len(df)} records from {path} using {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.warning(f"Error with {encoding}: {e}")
            continue
    
    raise ValueError(f"Could not read file {path} with any encoding")

def clean_platform_names(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced platform name cleaning with comprehensive mapping."""
    if 'platform' not in df.columns:
        logging.warning("'platform' column not found")
        return df
    
    df['platform'] = (df['platform']
                     .astype(str)
                     .str.lower()
                     .str.strip()
                     .str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters
                     .str.replace(r'\s+', ' ', regex=True))    # Normalize whitespace
    
    # Apply mappings
    df['platform'] = df['platform'].replace(PLATFORM_MAPPINGS)
    
    # Count platform distribution
    platform_counts = df['platform'].value_counts()
    logging.info(f"Platform distribution: {platform_counts.to_dict()}")
    
    # Flag unknown platforms
    known_platforms = set(PLATFORM_MAPPINGS.values()) | {'instagram', 'snapchat', 'tiktok', 'youtube'}
    unknown_platforms = set(df['platform'].unique()) - known_platforms
    if unknown_platforms:
        logging.warning(f"Unknown platforms found: {unknown_platforms}")
    
    return df

def clean_genre_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced genre cleaning with standardization."""
    if 'genre' not in df.columns:
        logging.warning("'genre' column not found")
        return df
    
    # Clean genres
    df['genre'] = (df['genre']
                   .astype(str)
                   .str.lower()
                   .str.strip()
                   .str.replace(r'[^\w\s]', ' ', regex=True)
                   .str.replace(r'\s+', ' ', regex=True))
    
    df['genre'] = df['genre'].replace(GENRE_MAPPINGS)
    
    # Handle missing/invalid genres
    invalid_genres = ['nan', 'null', 'none', '', ' ']
    df.loc[df['genre'].isin(invalid_genres), 'genre'] = 'unknown'
    
    genre_counts = df['genre'].value_counts()
    logging.info(f"Top 5 genres: {genre_counts.head().to_dict()}")
    
    return df

def categorize_trend_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced trend type categorization with validation."""
    if 'trend_type' not in df.columns:
        logging.warning("'trend_type' column not found")
        return df
    
    # Clean trend types
    df['trend_type'] = (df['trend_type']
                       .astype(str)
                       .str.lower()
                       .str.strip()
                       .str.replace(r'[^\w\s]', ' ', regex=True)
                       .str.replace(r'\s+', ' ', regex=True))
    
    # Apply mappings
    df['trend_type'] = df['trend_type'].replace(TREND_TYPE_MAPPINGS)
    
    # Handle missing trend types
    invalid_types = ['nan', 'null', 'none', '', ' ']
    df.loc[df['trend_type'].isin(invalid_types), 'trend_type'] = 'other'
    
    trend_counts = df['trend_type'].value_counts()
    logging.info(f"Trend type distribution: {trend_counts.to_dict()}")
    
    return df

def analyze_trend_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced date handling with validation and feature engineering."""
    date_columns = ['start_date', 'end_date']
    
    for col in date_columns:
        if col not in df.columns:
            logging.warning(f"'{col}' column not found")
            continue
            
        # Convert to datetime with multiple format attempts
        original_missing = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
        conversion_failed = df[col].isna().sum() - original_missing
        
        if conversion_failed > 0:
            logging.warning(f"{conversion_failed} date conversions failed for {col}")
    
    # Calculate duration and validate date logic
    if all(col in df.columns for col in date_columns):
        df['duration_days'] = (df['end_date'] - df['start_date']).dt.days
        
        # Flag suspicious durations
        negative_duration = (df['duration_days'] < 0).sum()
        long_duration = (df['duration_days'] > 365).sum()
        
        if negative_duration > 0:
            logging.warning(f"{negative_duration} trends have negative duration (end before start)")
        if long_duration > 0:
            logging.warning(f"{long_duration} trends last longer than a year")
        
        # Add temporal features
        if 'start_date' in df.columns:
            df['start_year'] = df['start_date'].dt.year
            df['start_month'] = df['start_date'].dt.month
            df['start_quarter'] = df['start_date'].dt.quarter
            df['start_weekday'] = df['start_date'].dt.day_name()
        
        # Duration categories
        df['duration_category'] = pd.cut(
            df['duration_days'],
            bins=[-np.inf, 7, 30, 90, 365, np.inf],
            labels=['short_term', 'weekly', 'monthly', 'seasonal', 'long_term'],
            include_lowest=True
        )
        
        logging.info(f"Duration stats - Mean: {df['duration_days'].mean():.1f} days, "
                    f"Median: {df['duration_days'].median():.1f} days")
    
    return df

def detect_and_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Intelligent duplicate detection."""
    initial_count = len(df)
    
    # Check for exact duplicates
    exact_dupes = df.duplicated().sum()
    if exact_dupes > 0:
        df = df.drop_duplicates()
        logging.info(f"Removed {exact_dupes} exact duplicate rows")
    
    # Check for potential content duplicates (same platform + similar description)
    if 'trend_description' in df.columns and 'platform' in df.columns:
        # Normalize descriptions for comparison
        df['desc_normalized'] = (df['trend_description']
                                .astype(str)
                                .str.lower()
                                .str.strip()
                                .str.replace(r'[^\w\s]', '', regex=True))
        
        # Find potential duplicates
        potential_dupes = df.duplicated(subset=['platform', 'desc_normalized']).sum()
        if potential_dupes > 0:
            logging.info(f"Found {potential_dupes} potential content duplicates (same platform + description)")
            # Remove exact content duplicates
            df = df.drop_duplicates(subset=['platform', 'desc_normalized'])
        
        # Clean up temporary column
        df = df.drop('desc_normalized', axis=1)
    
    final_count = len(df)
    if initial_count != final_count:
        logging.info(f"Total duplicates removed: {initial_count - final_count}")
    
    return df

def validate_data_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data consistency and fix common issues."""
    validation_issues = []
    
    # Check for reasonable date ranges
    if 'start_date' in df.columns:
        current_year = datetime.now().year
        future_dates = (df['start_date'].dt.year > current_year + 1).sum()
        old_dates = (df['start_date'].dt.year < 2010).sum()
        
        if future_dates > 0:
            validation_issues.append(f"{future_dates} trends have start dates far in the future")
        if old_dates > 0:
            validation_issues.append(f"{old_dates} trends have very old start dates (pre-2010)")
    
    # Check for missing critical information
    if 'trend_description' in df.columns:
        empty_descriptions = (df['trend_description'].astype(str).str.strip() == '').sum()
        if empty_descriptions > 0:
            validation_issues.append(f"{empty_descriptions} trends have empty descriptions")
    
    if validation_issues:
        logging.warning("Data validation issues found:")
        for issue in validation_issues:
            logging.warning(f"  - {issue}")
    
    return df

def quality_check(df: pd.DataFrame) -> dict:
    """Comprehensive quality assessment."""
    quality_report = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Platform-specific metrics
    if 'platform' in df.columns:
        quality_report['unique_platforms'] = df['platform'].nunique()
        quality_report['platform_distribution'] = df['platform'].value_counts().to_dict()
    
    # Date range analysis
    if all(col in df.columns for col in ['start_date', 'end_date']):
        quality_report['date_range'] = {
            'earliest_start': df['start_date'].min(),
            'latest_end': df['end_date'].max(),
            'date_span_days': (df['end_date'].max() - df['start_date'].min()).days
        }
    
    # Duration analysis
    if 'duration_days' in df.columns:
        quality_report['duration_stats'] = {
            'mean_duration': df['duration_days'].mean(),
            'median_duration': df['duration_days'].median(),
            'max_duration': df['duration_days'].max(),
            'min_duration': df['duration_days'].min()
        }
    
    # Trend type analysis
    if 'trend_type' in df.columns:
        quality_report['trend_types'] = df['trend_type'].value_counts().to_dict()
    
    return quality_report

def clean_instagram_snapchat():
    """Main cleaning function with comprehensive processing."""
    try:
        input_path = "data/raw/market/ig_snap.csv"
        output_dir = Path("data/cleaned_market")
        output_path = output_dir / "insta_snap_clean.csv"
        report_path = output_dir / "insta_snap_report.txt"
        
        logging.info("Starting Instagram/Snapchat trends data cleaning...")
        
        # Load and initial info
        df = load_data(input_path)
        logging.info(f"Initial data shape: {df.shape}")
        logging.info(f"Columns: {list(df.columns)}")
        
        # Cleaning pipeline
        df = detect_and_remove_duplicates(df)
        df = clean_platform_names(df)
        df = clean_genre_classifications(df)
        df = categorize_trend_types(df)
        df = analyze_trend_timing(df)
        df = validate_data_consistency(df)
        
        # Quality assessment
        quality = quality_check(df)
        logging.info(f"Final dataset: {quality['total_records']} records, {quality['total_columns']} columns")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Generate report
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("Instagram/Snapchat Trends Cleaning Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write(f"Total records: {quality['total_records']}\n")
            f.write(f"Total columns: {quality['total_columns']}\n")
            
            # Platform info
            if 'unique_platforms' in quality:
                f.write(f"Unique platforms: {quality['unique_platforms']}\n")
                f.write("Platform distribution:\n")
                for platform, count in quality['platform_distribution'].items():
                    percentage = (count / quality['total_records']) * 100
                    f.write(f"  {platform}: {count} ({percentage:.1f}%)\n")
            
            # Date range
            if 'date_range' in quality:
                date_info = quality['date_range']
                f.write(f"\nDate range: {date_info['earliest_start']} to {date_info['latest_end']}\n")
                f.write(f"Total span: {date_info['date_span_days']} days\n")
            
            # Duration stats
            if 'duration_stats' in quality:
                duration = quality['duration_stats']
                f.write(f"\nTrend Duration Statistics:\n")
                f.write(f"  Mean: {duration['mean_duration']:.1f} days\n")
                f.write(f"  Median: {duration['median_duration']:.1f} days\n")
                f.write(f"  Range: {duration['min_duration']} - {duration['max_duration']} days\n")
            
            # Trend types
            if 'trend_types' in quality:
                f.write(f"\nTrend Types:\n")
                for trend_type, count in quality['trend_types'].items():
                    percentage = (count / quality['total_records']) * 100
                    f.write(f"  {trend_type}: {count} ({percentage:.1f}%)\n")
            
            # Missing values
            f.write(f"\nMissing Values:\n")
            missing_cols = {col: count for col, count in quality['missing_values'].items() if count > 0}
            if missing_cols:
                for col, count in missing_cols.items():
                    percentage = (count / quality['total_records']) * 100
                    f.write(f"  {col}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("  No missing values found!\n")
        
        logging.info(f"Cleaned data saved to {output_path}")
        logging.info(f"Cleaning report saved to {report_path}")
        logging.info("Instagram/Snapchat trends cleaning completed successfully!")
        
        return df
        
    except Exception as e:
        logging.error(f"Error during cleaning: {e}")
        raise

if __name__ == "__main__":
    cleaned_df = clean_instagram_snapchat()