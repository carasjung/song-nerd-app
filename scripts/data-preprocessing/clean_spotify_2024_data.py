import pandas as pd
import numpy as np
from pathlib import Path
import logging
import chardet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUMERIC_COLUMNS = [
    'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach',
    'Spotify Popularity', 'YouTube Views', 'YouTube Likes', 'TikTok Posts',
    'TikTok Likes', 'TikTok Views', 'YouTube Playlist Reach',
    'Apple Music Playlist Count', 'AirPlay Spins', 'SiriusXM Spins',
    'Deezer Playlist Count', 'Deezer Playlist Reach', 'Amazon Playlist Count',
    'Pandora Streams', 'Pandora Track Stations', 'Soundcloud Streams',
    'Shazam Counts'
]

PLATFORM_COLS = {
    'spotify': ['Spotify Streams', 'Spotify Playlist Reach'],
    'youtube': ['YouTube Views', 'YouTube Likes'],
    'tiktok': ['TikTok Posts', 'TikTok Views']
}

def detect_encoding(file_path: str) -> str:
    """Detect the encoding of a file."""
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)  # Read first 10KB to detect encoding
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            logging.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        logging.warning(f"Encoding detection failed: {e}. Trying common encodings...")
        return None

def load_spotify_file(path: str) -> pd.DataFrame:
    """Load CSV file with automatic encoding detection."""
    try:
        # First, try to detect encoding
        detected_encoding = detect_encoding(path)
        
        # List of encodings to try in order
        encodings_to_try = []
        if detected_encoding:
            encodings_to_try.append(detected_encoding)
        
        # Add common encodings
        encodings_to_try.extend([
            'utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 
            'utf-16', 'utf-8-sig', 'ascii'
        ])
        
        # Remove duplicates while preserving order
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
        
        # Try each encoding
        for encoding in encodings_to_try:
            try:
                logging.info(f"Attempting to load with encoding: {encoding}")
                df = pd.read_csv(path, encoding=encoding)
                logging.info(f"Successfully loaded {len(df)} records using {encoding} encoding")
                return df
            except UnicodeDecodeError:
                logging.warning(f"Failed to decode with {encoding}")
                continue
            except Exception as e:
                logging.warning(f"Error with {encoding}: {e}")
                continue
        
        raise ValueError("Could not determine file encoding. Please check the file format.")
        
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        raise

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert numeric columns with enhanced error handling."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            original_missing = df[col].isna().sum()
            
            # Convert to string and clean
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace(' ', '', regex=False)  # Remove spaces
                .replace(['N/A', 'n/a', 'NA', 'null', 'NULL', ''], np.nan)
            )
            
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values
            missing_after_conversion = df[col].isna().sum()
            total_missing = missing_after_conversion
            
            if total_missing > 0:
                # Use median for most columns, but handle special cases
                if col in ['Spotify Popularity']:
                    fill_value = df[col].median()
                else:
                    fill_value = df[col].median()
                
                df[col] = df[col].fillna(fill_value)
                logging.info(f"'{col}': {original_missing} originally missing, "
                           f"{missing_after_conversion - original_missing} failed conversion, "
                           f"filled {total_missing} total with median {fill_value:.2f}")
            else:
                logging.info(f"'{col}': No missing values")
    
    return df

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute meaningful ratios with zero-division protection."""
    ratios_computed = []
    
    # Spotify engagement ratio
    if 'Spotify Streams' in df.columns and 'Spotify Playlist Reach' in df.columns:
        df['spotify_streams_per_reach'] = np.where(
            df['Spotify Playlist Reach'] > 0,
            df['Spotify Streams'] / df['Spotify Playlist Reach'],
            0
        )
        ratios_computed.append('spotify_streams_per_reach')
    
    # TikTok engagement ratio
    if 'TikTok Views' in df.columns and 'TikTok Posts' in df.columns:
        df['tiktok_views_per_post'] = np.where(
            df['TikTok Posts'] > 0,
            df['TikTok Views'] / df['TikTok Posts'],
            0
        )
        ratios_computed.append('tiktok_views_per_post')
    
    # YouTube engagement ratio
    if 'YouTube Views' in df.columns and 'YouTube Likes' in df.columns:
        df['youtube_like_ratio'] = np.where(
            df['YouTube Views'] > 0,
            df['YouTube Likes'] / df['YouTube Views'],
            0
        )
        ratios_computed.append('youtube_like_ratio')
    
    # Cross-platform comparison ratios
    if 'Spotify Streams' in df.columns and 'YouTube Views' in df.columns:
        df['spotify_youtube_ratio'] = np.where(
            df['YouTube Views'] > 0,
            df['Spotify Streams'] / df['YouTube Views'],
            0
        )
        ratios_computed.append('spotify_youtube_ratio')
    
    if ratios_computed:
        logging.info(f"Computed {len(ratios_computed)} engagement ratios: {', '.join(ratios_computed)}")
    
    return df

def validate_data_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean unrealistic values."""
    validation_rules = {
        'Spotify Popularity': (0, 100),  # Spotify popularity is 0-100
        'YouTube Views': (0, float('inf')),  # Can't be negative
        'Spotify Streams': (0, float('inf')),  # Can't be negative
    }
    
    for col, (min_val, max_val) in validation_rules.items():
        if col in df.columns:
            outliers = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if outliers > 0:
                logging.info(f"Found {outliers} outliers in {col}, clipping to range [{min_val}, {max_val}]")
                df[col] = df[col].clip(min_val, max_val)
    
    return df

def quality_check(df: pd.DataFrame) -> dict:
    """Enhanced quality check with more metrics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    quality_report = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Add summary statistics for key columns
    if 'Spotify Streams' in df.columns:
        quality_report['spotify_streams_stats'] = {
            'min': df['Spotify Streams'].min(),
            'max': df['Spotify Streams'].max(),
            'median': df['Spotify Streams'].median(),
            'mean': df['Spotify Streams'].mean()
        }
    
    return quality_report

def clean_spotify_2024():
    """Main cleaning function with comprehensive error handling."""
    try:
        input_path = "data/raw/market/most_streamed_spotify_2024.csv"
        output_dir = Path("data/cleaned_market")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "spotify_2024_clean.csv"
        report_path = output_dir / "spotify_cleaning_report.txt"
        
        logging.info("Starting Spotify 2024 data cleaning...")
        df = load_spotify_file(input_path)
        
        logging.info(f"Initial data shape: {df.shape}")
        logging.info(f"Columns: {list(df.columns)}")
        
        df = clean_numeric_columns(df)
        
        df = validate_data_ranges(df)
        
        df = compute_ratios(df)

        # Quality report
        quality = quality_check(df)
        logging.info(f"Final dataset: {quality['total_records']} records, {quality['total_columns']} columns")
        
        # Save cleaned data
        df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Saved cleaned data to {output_path}")
        
        # Save comprehensive report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Spotify 2024 Cleaning Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total records: {quality['total_records']}\n")
            f.write(f"Total columns: {quality['total_columns']}\n")
            f.write(f"Numeric columns: {quality['numeric_columns']}\n")
            f.write(f"Duplicate rows found: {quality['duplicate_rows']}\n\n")
            
            # Missing values summary
            f.write("Missing Values Summary:\n")
            f.write("-" * 25 + "\n")
            missing_cols = {col: count for col, count in quality['missing_values'].items() if count > 0}
            if missing_cols:
                for col, count in missing_cols.items():
                    percentage = (count / quality['total_records']) * 100
                    f.write(f"  {col}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("  No missing values found!\n")
            
            # Spotify streams statistics
            if 'spotify_streams_stats' in quality:
                f.write(f"\nSpotify Streams Statistics:\n")
                f.write("-" * 30 + "\n")
                stats = quality['spotify_streams_stats']
                f.write(f"  Min: {stats['min']:,.0f}\n")
                f.write(f"  Max: {stats['max']:,.0f}\n")
                f.write(f"  Median: {stats['median']:,.0f}\n")
                f.write(f"  Mean: {stats['mean']:,.0f}\n")
        
        logging.info(f"Saved cleaning report to {report_path}")
        logging.info("Spotify 2024 data cleaning completed successfully.")
        
        return df
        
    except Exception as e:
        logging.error(f"Error during Spotify 2024 cleaning: {e}")
        raise

if __name__ == "__main__":
    cleaned_df = clean_spotify_2024()