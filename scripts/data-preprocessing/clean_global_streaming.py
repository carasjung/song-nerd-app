# clean_global_streaming.py
# This script cleans the global streaming data from Spotify

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    input_path: str = "data/raw/market/global_streaming.csv"
    output_dir: str = "data/cleaned_market"
    output_filename: str = "global_streaming_clean.csv"
    report_filename: str = "global_streaming_report.json"
    
    country_mappings: Dict[str, str] = None
    subscription_mappings: Dict[str, str] = None
    platform_mappings: Dict[str, str] = None
    listening_time_mappings: Dict[str, str] = None
    
    def __post_init__(self):
        if self.country_mappings is None:
            self.country_mappings = {
                'usa': 'United States',
                'us': 'United States',
                'uk': 'United Kingdom',
                'uae': 'United Arab Emirates',
                'south korea': 'Korea',
                'russia': 'Russian Federation'
            }
        
        if self.subscription_mappings is None:
            self.subscription_mappings = {
                'premium': 'Premium',
                'free': 'Free',
                'basic': 'Basic',
                'student': 'Premium',
                'family': 'Premium',
                'trial': 'Free'
            }
        
        if self.platform_mappings is None:
            self.platform_mappings = {
                'spotify': 'Spotify',
                'apple music': 'Apple Music',
                'youtube music': 'YouTube Music',
                'amazon music': 'Amazon Music',
                'pandora': 'Pandora',
                'tidal': 'Tidal',
                'deezer': 'Deezer'
            }
        
        if self.listening_time_mappings is None:
            self.listening_time_mappings = {
                'morning': 'Morning',
                'afternoon': 'Afternoon',
                'evening': 'Night',
                'night': 'Night',
                'am': 'Morning',
                'pm': 'Afternoon'
            }

class StreamingDataCleaner:
    """Main class for cleaning streaming data with validation and reporting."""
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.cleaning_stats = {
            'records_processed': 0,
            'cleaning_steps': {},
            'validation_errors': [],
            'warnings': []
        }
        
        self.expected_columns = [
            'User_ID', 'Age', 'Country', 'Streaming Platform', 'Top Genre',
            'Minutes Streamed Per Day', 'Number of Songs Liked', 'Most Played Artist',
            'Subscription Type', 'Listening Time (Morning/Afternoon/Night)',
            'Discover Weekly Engagement (%)', 'Repeat Song Rate (%)'
        ]
    
    def load_data(self, path: str) -> pd.DataFrame:
        """Load CSV data with comprehensive error handling."""
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            
            # Try different encodings if needed
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    logger.info(f"Successfully loaded {len(df)} rows from {path} using {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read file with any supported encoding")
            
            self.cleaning_stats['records_processed'] = len(df)
            return df
            
        except Exception as e:
            logger.error(f"Failed to load file {path}: {e}")
            raise
    
    def validate_data_structure(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame has expected structure."""
        missing_columns = [col for col in self.expected_columns if col not in df.columns]
        
        if missing_columns:
            self.cleaning_stats['warnings'].append(f"Missing expected columns: {missing_columns}")
            logger.warning(f"Missing expected columns: {missing_columns}")
        
        # Check for completely empty DataFrame
        if df.empty:
            self.cleaning_stats['validation_errors'].append("DataFrame is empty")
            return False
        
        # Log actual columns found
        logger.info(f"Columns found: {list(df.columns)}")
        
        return True
    
    def clean_user_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate User IDs."""
        if 'User_ID' not in df.columns:
            return df
        
        original_count = len(df)
        
        df = df.dropna(subset=['User_ID'])
        
        df['User_ID'] = df['User_ID'].astype(str).str.strip()
        
        df = df[df['User_ID'] != '']
        
        cleaned_count = len(df)
        removed_count = original_count - cleaned_count
        
        self.cleaning_stats['cleaning_steps']['user_ids'] = {
            'invalid_ids_removed': removed_count,
            'final_count': cleaned_count
        }
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with invalid User_IDs")
        
        return df
    
    def clean_age_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate age data."""
        if 'Age' not in df.columns:
            return df
        
        original_values = df['Age'].value_counts()
        
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        # Keep ages between 13 and 100
        age_mask = (df['Age'] >= 13) & (df['Age'] <= 100)
        invalid_ages = (~age_mask).sum()
        
        if invalid_ages > 0:
            logger.warning(f"Found {invalid_ages} invalid age values")
            self.cleaning_stats['warnings'].append(f"Invalid ages found: {invalid_ages}")
        
        # Replace invalid ages with median age
        median_age = df.loc[age_mask, 'Age'].median()
        df.loc[~age_mask, 'Age'] = median_age
        
        self.cleaning_stats['cleaning_steps']['age'] = {
            'invalid_ages_replaced': invalid_ages,
            'replacement_value': median_age,
            'age_range': f"{df['Age'].min():.0f}-{df['Age'].max():.0f}"
        }
        
        return df
    
    def standardize_countries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize country names with comprehensive mapping."""
        if 'Country' not in df.columns:
            return df
        
        original_values = df['Country'].value_counts()
        
        df['Country'] = (df['Country']
                        .astype(str)
                        .str.strip()
                        .str.title())
        
        country_map_lower = {k.lower(): v for k, v in self.config.country_mappings.items()}
        df['Country'] = df['Country'].apply(
            lambda x: country_map_lower.get(x.lower(), x) if pd.notna(x) else x
        )
        
        cleaned_values = df['Country'].value_counts()
        self.cleaning_stats['cleaning_steps']['countries'] = {
            'original_unique_values': len(original_values),
            'cleaned_unique_values': len(cleaned_values),
            'mappings_applied': len(self.config.country_mappings)
        }
        
        return df
    
    def standardize_streaming_platforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize streaming platform names."""
        if 'Streaming Platform' not in df.columns:
            return df
        
        original_values = df['Streaming Platform'].value_counts()
        
        df['Streaming Platform'] = (df['Streaming Platform']
                                   .astype(str)
                                   .str.strip()
                                   .str.lower())
        
        # Apply mappings
        df['Streaming Platform'] = df['Streaming Platform'].replace(self.config.platform_mappings)
        
        cleaned_values = df['Streaming Platform'].value_counts()
        self.cleaning_stats['cleaning_steps']['platforms'] = {
            'original_unique_values': len(original_values),
            'cleaned_unique_values': len(cleaned_values),
            'mappings_applied': len(self.config.platform_mappings)
        }
        
        return df
    
    def clean_genre_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean top genre data."""
        if 'Top Genre' not in df.columns:
            return df
        
        original_values = df['Top Genre'].value_counts()
        
        # Standardize genre names
        df['Top Genre'] = (df['Top Genre']
                          .astype(str)
                          .str.strip()
                          .str.title()
                          .str.replace('_', ' ', regex=False)
                          .str.replace('-', ' ', regex=False))
        
        df['Top Genre'] = df['Top Genre'].str.replace(r'\s+', ' ', regex=True)
        
        cleaned_values = df['Top Genre'].value_counts()
        self.cleaning_stats['cleaning_steps']['genres'] = {
            'original_unique_values': len(original_values),
            'cleaned_unique_values': len(cleaned_values)
        }
        
        return df
    
    def clean_numeric_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric streaming metrics."""
        numeric_columns = [
            'Minutes Streamed Per Day',
            'Number of Songs Liked',
            'Discover Weekly Engagement (%)',
            'Repeat Song Rate (%)'
        ]
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
            
            # Convert to numeric, handling any string representations
            if col.endswith('(%)'):
                # Handle percentage columns
                df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle outliers and invalid values
            if col == 'Minutes Streamed Per Day':
                # Cap at realistic daily streaming. 24 hours = 1440 minutes
                df[col] = df[col].clip(0, 1440)
            elif col == 'Number of Songs Liked':
                # Remove negative values
                df[col] = df[col].clip(0, None)
            elif col.endswith('(%)'):
                # Cap percentages at 0-100
                df[col] = df[col].clip(0, 100)
            
            # Fill missing values with median
            median_val = df[col].median()
            missing_count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            
            self.cleaning_stats['cleaning_steps'][col] = {
                'missing_values_filled': missing_count,
                'fill_value': median_val,
                'range': f"{df[col].min():.1f}-{df[col].max():.1f}"
            }
        
        return df
    
    def clean_artist_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean most played artist names."""
        if 'Most Played Artist' not in df.columns:
            return df
        
        original_values = df['Most Played Artist'].value_counts()
        
        # Standardize artist names
        df['Most Played Artist'] = (df['Most Played Artist']
                                   .astype(str)
                                   .str.strip()
                                   .str.title())
        
        df['Most Played Artist'] = df['Most Played Artist'].str.replace(r'\s+', ' ', regex=True)
        
        cleaned_values = df['Most Played Artist'].value_counts()
        self.cleaning_stats['cleaning_steps']['artists'] = {
            'original_unique_values': len(original_values),
            'cleaned_unique_values': len(cleaned_values)
        }
        
        return df
    
    def normalize_subscription_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize subscription type categories."""
        if 'Subscription Type' not in df.columns:
            return df
        
        original_values = df['Subscription Type'].value_counts()
        
        df['Subscription Type'] = (df['Subscription Type']
                                  .astype(str)
                                  .str.lower()
                                  .str.strip())
        
        # Apply mappings
        df['Subscription Type'] = df['Subscription Type'].replace(self.config.subscription_mappings)
        
        cleaned_values = df['Subscription Type'].value_counts()
        self.cleaning_stats['cleaning_steps']['subscriptions'] = {
            'original_unique_values': len(original_values),
            'cleaned_unique_values': len(cleaned_values),
            'mappings_applied': len(self.config.subscription_mappings)
        }
        
        return df
    
    def clean_listening_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean listening time preferences."""
        col_name = 'Listening Time (Morning/Afternoon/Night)'
        if col_name not in df.columns:
            return df
        
        original_values = df[col_name].value_counts()
        
        df[col_name] = (df[col_name]
                       .astype(str)
                       .str.lower()
                       .str.strip())
        
        # Apply mappings
        df[col_name] = df[col_name].replace(self.config.listening_time_mappings)
        
        cleaned_values = df[col_name].value_counts()
        self.cleaning_stats['cleaning_steps']['listening_time'] = {
            'original_unique_values': len(original_values),
            'cleaned_unique_values': len(cleaned_values),
            'mappings_applied': len(self.config.listening_time_mappings)
        }
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records and track the operation."""
        original_count = len(df)
        
        # Remove duplicates based on User_ID, otherwise all columns
        if 'User_ID' in df.columns:
            df_clean = df.drop_duplicates(subset=['User_ID'])
        else:
            df_clean = df.drop_duplicates()
        
        duplicates_removed = original_count - len(df_clean)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate records")
            self.cleaning_stats['cleaning_steps']['duplicates'] = {
                'duplicates_removed': duplicates_removed,
                'final_record_count': len(df_clean)
            }
        
        return df_clean
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report."""
        report = {
            'summary': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'missing_data': {
                'columns_with_missing': df.isnull().sum().to_dict(),
                'missing_data_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            },
            'data_distribution': {},
            'data_quality_metrics': {},
            'cleaning_statistics': self.cleaning_stats
        }
        
        # Add distribution stats for categorical columns
        categorical_columns = ['Country', 'Streaming Platform', 'Top Genre', 
                             'Subscription Type', 'Listening Time (Morning/Afternoon/Night)']
        for col in categorical_columns:
            if col in df.columns:
                report['data_distribution'][col] = {
                    'unique_values': df[col].nunique(),
                    'top_5_values': df[col].value_counts().head().to_dict()
                }
        
        # Add numeric column statistics
        numeric_columns = ['Age', 'Minutes Streamed Per Day', 'Number of Songs Liked',
                          'Discover Weekly Engagement (%)', 'Repeat Song Rate (%)']
        for col in numeric_columns:
            if col in df.columns:
                report['data_quality_metrics'][col] = {
                    'mean': round(df[col].mean(), 2),
                    'median': round(df[col].median(), 2),
                    'std': round(df[col].std(), 2),
                    'min': round(df[col].min(), 2),
                    'max': round(df[col].max(), 2)
                }
        
        return report
    
    def save_results(self, df: pd.DataFrame, report: Dict) -> Tuple[Path, Path]:
        """Save cleaned data and quality report."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        output_path = output_dir / self.config.output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")
        
        # Save quality report as JSON
        report_path = output_dir / self.config.report_filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Quality report saved to {report_path}")
        
        return output_path, report_path
    
    def clean_pipeline(self) -> Tuple[pd.DataFrame, Dict]:
        """Execute the complete data cleaning pipeline."""
        logger.info("Starting streaming data cleaning pipeline")
        
        try:
            df = self.load_data(self.config.input_path)
            
            # Validate structure
            if not self.validate_data_structure(df):
                raise ValueError("Data validation failed")
            
            # Apply cleaning steps in logical order
            df = self.clean_user_ids(df)
            df = self.clean_age_data(df)
            df = self.standardize_countries(df)
            df = self.standardize_streaming_platforms(df)
            df = self.clean_genre_data(df)
            df = self.clean_numeric_metrics(df)
            df = self.clean_artist_names(df)
            df = self.normalize_subscription_types(df)
            df = self.clean_listening_time(df)
            df = self.remove_duplicates(df)
            
            # Generate report
            quality_report = self.generate_quality_report(df)
            
            output_path, report_path = self.save_results(df, quality_report)
            
            logger.info("Data cleaning pipeline completed successfully")
            return df, quality_report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main execution function."""
    config = CleaningConfig()
    cleaner = StreamingDataCleaner(config)
    
    try:
        cleaned_data, report = cleaner.clean_pipeline()
        
        print("STREAMING DATA CLEANING SUMMARY")
        print(f"Records processed: {report['summary']['total_records']:,}")
        print(f"Columns: {report['summary']['total_columns']}")
        print(f"Memory usage: {report['summary']['memory_usage_mb']} MB")
        
        print(f"\nCountries: {report['data_distribution'].get('Country', {}).get('unique_values', 'N/A')}")
        print(f"Streaming platforms: {report['data_distribution'].get('Streaming Platform', {}).get('unique_values', 'N/A')}")
        print(f"Genres: {report['data_distribution'].get('Top Genre', {}).get('unique_values', 'N/A')}")
        
        if 'Age' in report['data_quality_metrics']:
            age_stats = report['data_quality_metrics']['Age']
            print(f"Age range: {age_stats['min']}-{age_stats['max']} (avg: {age_stats['mean']})")
        
        if report['cleaning_statistics']['warnings']:
            print(f"\nWarnings: {len(report['cleaning_statistics']['warnings'])}")
            for warning in report['cleaning_statistics']['warnings']:
                print(f"  - {warning}")
        
        print("\nCleaning completed successfully!")
        
    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())