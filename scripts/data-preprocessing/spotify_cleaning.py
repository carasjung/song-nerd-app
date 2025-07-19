# spotify_cleaning.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spotify_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SpotifyCleaningConfig:
    """Configuration for Spotify data cleaning operations."""
    input_path: str = "data/raw/music/spotify-dataset.csv"
    output_dir: str = "data/cleaned_music"
    output_filename: str = "spotify_tracks_clean.csv"
    report_filename: str = "spotify_cleaning_report.json"
    
    # Audio feature validation ranges
    audio_features: Dict[str, Tuple[float, float]] = None
    
    # Genre grouping mappings
    genre_mappings: Dict[str, str] = None
    
    # Data quality thresholds
    min_track_name_length: int = 1
    max_artist_count: int = 10
    
    def __post_init__(self):
        if self.audio_features is None:
            self.audio_features = {
                'danceability': (0.0, 1.0),
                'energy': (0.0, 1.0),
                'loudness': (-60.0, 5.0),  # Extended upper range for edge cases
                'speechiness': (0.0, 1.0),
                'acousticness': (0.0, 1.0),
                'instrumentalness': (0.0, 1.0),
                'liveness': (0.0, 1.0),
                'valence': (0.0, 1.0),
                'tempo': (30.0, 300.0),  # Extended range for extreme genres
                'duration_ms': (5000, 1800000),  # 5 sec to 30 min
                'popularity': (0, 100),
                'key': (0, 11),  # Musical keys (C=0 to B=11)
                'mode': (0, 1),  # Major=1, Minor=0
                'time_signature': (1, 7)  # Common time signatures
            }
        
        if self.genre_mappings is None:
            self.genre_mappings = {
                # Pop and mainstream
                'pop': 'pop',
                'k-pop': 'pop',
                'dance pop': 'pop',
                'electropop': 'pop',
                'synth-pop': 'pop',
                
                # Rock and metal
                'rock': 'rock',
                'indie rock': 'rock',
                'alternative rock': 'rock',
                'hard rock': 'rock',
                'metal': 'rock',
                'heavy metal': 'rock',
                'death metal': 'rock',
                'black metal': 'rock',
                
                # Electronic and EDM
                'edm': 'electronic',
                'electronic': 'electronic',
                'house': 'electronic',
                'techno': 'electronic',
                'trance': 'electronic',
                'dubstep': 'electronic',
                'ambient': 'electronic',
                'drum and bass': 'electronic',
                
                # Hip hop and urban
                'hip hop': 'hip-hop',
                'rap': 'hip-hop',
                'trap': 'hip-hop',
                'r&b': 'r&b',
                'soul': 'r&b',
                'funk': 'r&b',
                
                # Jazz and blues
                'jazz': 'jazz',
                'blues': 'jazz',
                'swing': 'jazz',
                'bebop': 'jazz',
                
                # Classical and instrumental
                'classical': 'classical',
                'opera': 'classical',
                'instrumental': 'instrumental',
                'soundtrack': 'instrumental',
                'film score': 'instrumental',
                
                # Country and folk
                'country': 'country',
                'folk': 'folk',
                'bluegrass': 'country',
                'americana': 'folk',
                
                # World music
                'reggae': 'world',
                'latin': 'world',
                'afrobeat': 'world',
                'world': 'world',
                'ethnic': 'world'
            }


class SpotifyDataCleaner:
    """Main class for cleaning Spotify track data"""
    
    def __init__(self, config: SpotifyCleaningConfig):
        self.config = config
        self.cleaning_stats = {
            'records_processed': 0,
            'cleaning_steps': {},
            'validation_errors': [],
            'warnings': [],
            'feature_engineering': {}
        }
        
        self.expected_columns = [
            'track_id', 'artists', 'album_name', 'track_name', 'popularity',
            'duration_ms', 'explicit', 'danceability', 'energy', 'key',
            'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'
        ]
    
    def load_data(self, path: str) -> pd.DataFrame:
        """Load CSV data with comprehensive error handling."""
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    logger.info(f"Successfully loaded {len(df)} tracks using {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read file with any supported encoding")
            
            # Remove unnamed index column if present
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
                logger.info("Removed unnamed index column")
            
            self.cleaning_stats['records_processed'] = len(df)
            return df
            
        except Exception as e:
            logger.error(f"Failed to load file {path}: {e}")
            raise
    
    def validate_data_structure(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame structure and log findings."""
        missing_columns = [col for col in self.expected_columns if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in self.expected_columns]
        
        if missing_columns:
            self.cleaning_stats['warnings'].append(f"Missing expected columns: {missing_columns}")
            logger.warning(f"Missing expected columns: {missing_columns}")
        
        if extra_columns:
            logger.info(f"Found extra columns: {extra_columns}")
        
        if df.empty:
            self.cleaning_stats['validation_errors'].append("DataFrame is empty")
            return False
        
        logger.info(f"Dataset shape: {df.shape}")
        return True
    
    def clean_track_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean track IDs and basic identifiers."""
        original_count = len(df)
        
        # Clean track_id
        if 'track_id' in df.columns:
            # Remove rows with missing track_id
            df = df.dropna(subset=['track_id'])
            df['track_id'] = df['track_id'].astype(str).str.strip()
            df = df[df['track_id'] != '']
            
            # Remove duplicates based on track_id
            duplicates = df.duplicated(subset=['track_id']).sum()
            df = df.drop_duplicates(subset=['track_id'])
            
            self.cleaning_stats['cleaning_steps']['track_ids'] = {
                'duplicates_removed': duplicates,
                'invalid_ids_removed': original_count - len(df) - duplicates
            }
            
            if duplicates > 0:
                logger.info(f"Removed {duplicates} duplicate tracks")
        
        return df
    
    def clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields."""
        text_fields = ['artists', 'album_name', 'track_name']
        
        for field in text_fields:
            if field not in df.columns:
                continue
            
            original_missing = df[field].isnull().sum()
            
            # Convert to string and clean
            df[field] = df[field].astype(str).str.strip()
            
            # Replace 'nan' strings with actual NaN
            df[field] = df[field].replace(['nan', 'NaN', 'None', ''], np.nan)
            
            # For track names, remove rows with missing names
            if field == 'track_name':
                df = df.dropna(subset=[field])
                # Check min length
                short_names = (df[field].str.len() < self.config.min_track_name_length).sum()
                if short_names > 0:
                    df = df[df[field].str.len() >= self.config.min_track_name_length]
                    logger.info(f"Removed {short_names} tracks with very short names")
            
            # For artists, handle multiple artists
            if field == 'artists':
                # Count artists (split by common separators)
                df['artist_count'] = df[field].str.count('[,;&]') + 1
                df['artist_count'] = df['artist_count'].fillna(1)
                
                # Cap excessive artist counts, which are possible data errors
                excessive_artists = (df['artist_count'] > self.config.max_artist_count).sum()
                if excessive_artists > 0:
                    logger.warning(f"Found {excessive_artists} tracks with >10 artists (possible data errors)")
            
            final_missing = df[field].isnull().sum()
            
            self.cleaning_stats['cleaning_steps'][field] = {
                'original_missing': original_missing,
                'final_missing': final_missing,
                'unique_values': df[field].nunique()
            }
        
        return df
    
    def validate_audio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean audio features."""
        for feature, (min_val, max_val) in self.config.audio_features.items():
            if feature not in df.columns:
                continue
            
            # Convert to numeric
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            
            # Count missing values
            missing_count = df[feature].isnull().sum()
            
            # Count outliers before clipping
            if missing_count < len(df):  
                outliers_low = (df[feature] < min_val).sum()
                outliers_high = (df[feature] > max_val).sum()
                total_outliers = outliers_low + outliers_high
                
                # Clip outliers
                df[feature] = df[feature].clip(min_val, max_val)
                
                # Fill missing values with median
                if missing_count > 0:
                    median_val = df[feature].median()
                    df[feature] = df[feature].fillna(median_val)
                
                self.cleaning_stats['cleaning_steps'][f'{feature}_validation'] = {
                    'missing_filled': missing_count,
                    'outliers_clipped': total_outliers,
                    'outliers_low': outliers_low,
                    'outliers_high': outliers_high,
                    'valid_range': f"{min_val}-{max_val}"
                }
                
                if total_outliers > 0:
                    logger.info(f"Clipped {total_outliers} outliers in {feature}")
        
        return df
    
    def clean_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean categorical features like explicit flag."""
        if 'explicit' in df.columns:
            original_values = df['explicit'].value_counts()
            
            # Convert to boolean
            df['explicit'] = df['explicit'].astype(str).str.lower()
            explicit_map = {
                'true': True, '1': True, 'yes': True,
                'false': False, '0': False, 'no': False,
                'nan': False, 'none': False
            }
            df['explicit'] = df['explicit'].map(explicit_map).fillna(False)
            
            self.cleaning_stats['cleaning_steps']['explicit'] = {
                'original_unique_values': len(original_values),
                'true_count': df['explicit'].sum(),
                'false_count': (~df['explicit']).sum()
            }
        
        return df
    
    def process_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and categorize genres."""
        if 'track_genre' not in df.columns:
            return df
        
        original_genres = df['track_genre'].value_counts()
        
        # Clean genre names
        df['track_genre'] = df['track_genre'].astype(str).str.lower().str.strip()
        df['track_genre'] = df['track_genre'].replace(['nan', 'none', ''], 'unknown')
        
        # Apply genre mapping
        def map_genre(genre):
            if pd.isna(genre) or genre == 'unknown':
                return 'other'
            
            genre = str(genre).lower()
            
            # Direct mapping first
            if genre in self.config.genre_mappings:
                return self.config.genre_mappings[genre]
            
            # Partial matching
            for key, group in self.config.genre_mappings.items():
                if key in genre:
                    return group
            
            return 'other'
        
        df['genre_group'] = df['track_genre'].apply(map_genre)
        
        # Create additional genre features
        df['is_mainstream'] = df['genre_group'].isin(['pop', 'rock', 'hip-hop'])
        df['is_electronic'] = df['genre_group'] == 'electronic'
        df['is_instrumental'] = df['genre_group'].isin(['classical', 'instrumental'])
        
        cleaned_genres = df['track_genre'].value_counts()
        genre_groups = df['genre_group'].value_counts()
        
        self.cleaning_stats['cleaning_steps']['genres'] = {
            'original_unique_genres': len(original_genres),
            'cleaned_unique_genres': len(cleaned_genres),
            'genre_groups_created': len(genre_groups),
            'unknown_genres': (df['track_genre'] == 'unknown').sum()
        }
        
        self.cleaning_stats['feature_engineering']['genre_features'] = {
            'mainstream_tracks': df['is_mainstream'].sum(),
            'electronic_tracks': df['is_electronic'].sum(),
            'instrumental_tracks': df['is_instrumental'].sum()
        }
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing data."""
        feature_count = 0
        
        # Duration-based features
        if 'duration_ms' in df.columns:
            df['duration_minutes'] = df['duration_ms'] / 60000
            df['is_short_track'] = df['duration_minutes'] < 2.5
            df['is_long_track'] = df['duration_minutes'] > 6.0
            feature_count += 3
        
        # Energy and danceability combinations
        if 'energy' in df.columns and 'danceability' in df.columns:
            df['energy_danceability'] = df['energy'] * df['danceability']
            df['is_high_energy_danceable'] = (df['energy'] > 0.7) & (df['danceability'] > 0.7)
            feature_count += 2
        
        # Mood indicators
        if 'valence' in df.columns and 'energy' in df.columns:
            df['mood_score'] = (df['valence'] + df['energy']) / 2
            df['is_upbeat'] = (df['valence'] > 0.6) & (df['energy'] > 0.6)
            df['is_melancholy'] = (df['valence'] < 0.4) & (df['energy'] < 0.4)
            feature_count += 3
        
        # Acoustic vs electronic
        if 'acousticness' in df.columns and 'instrumentalness' in df.columns:
            df['acoustic_instrumental'] = df['acousticness'] * df['instrumentalness']
            df['is_acoustic'] = df['acousticness'] > 0.7
            feature_count += 2
        
        # Popularity categories
        if 'popularity' in df.columns:
            df['popularity_category'] = pd.cut(df['popularity'], 
                                             bins=[0, 20, 50, 80, 100],
                                             labels=['low', 'medium', 'high', 'viral'])
            df['is_popular'] = df['popularity'] > 70
            feature_count += 2
        
        # Tempo categories
        if 'tempo' in df.columns:
            df['tempo_category'] = pd.cut(df['tempo'],
                                        bins=[0, 90, 120, 140, 300],
                                        labels=['slow', 'moderate', 'fast', 'very_fast'])
            feature_count += 1
        
        self.cleaning_stats['feature_engineering']['derived_features'] = {
            'features_created': feature_count,
            'total_features': len(df.columns)
        }
        
        logger.info(f"Created {feature_count} derived features")
        return df
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report."""
        report = {
            'summary': {
                'total_tracks': len(df),
                'total_features': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'data_quality': {
                'missing_data': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            },
            'audio_features_stats': {},
            'genre_distribution': {},
            'popularity_stats': {},
            'cleaning_statistics': self.cleaning_stats
        }
        
        # Audio features statistics
        audio_features = [col for col in df.columns if col in self.config.audio_features.keys()]
        for feature in audio_features:
            if feature in df.columns:
                report['audio_features_stats'][feature] = {
                    'mean': round(df[feature].mean(), 3),
                    'median': round(df[feature].median(), 3),
                    'std': round(df[feature].std(), 3),
                    'min': round(df[feature].min(), 3),
                    'max': round(df[feature].max(), 3)
                }
        
        # Genre distribution
        if 'genre_group' in df.columns:
            report['genre_distribution'] = df['genre_group'].value_counts().to_dict()
        
        # Popularity statistics
        if 'popularity' in df.columns:
            report['popularity_stats'] = {
                'average_popularity': round(df['popularity'].mean(), 2),
                'popular_tracks_percent': round((df['popularity'] > 70).mean() * 100, 2),
                'zero_popularity_count': (df['popularity'] == 0).sum()
            }
        
        return report
    
    def save_results(self, df: pd.DataFrame, report: Dict) -> Tuple[Path, Path]:
        """Save cleaned data and report."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / self.config.output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned Spotify data saved to {output_path}")
        
        report_path = output_dir / self.config.report_filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Cleaning report saved to {report_path}")
        
        return output_path, report_path
    
    def clean_pipeline(self) -> Tuple[pd.DataFrame, Dict]:
        """Execute the complete cleaning pipeline."""
        logger.info("Starting Spotify data cleaning pipeline")
        
        try:
            df = self.load_data(self.config.input_path)
            if not self.validate_data_structure(df):
                raise ValueError("Data validation failed")
            
            # Start cleaning pipline
            df = self.clean_track_identifiers(df)
            df = self.clean_text_fields(df)
            df = self.validate_audio_features(df)
            df = self.clean_categorical_features(df)
            df = self.process_genres(df)
            df = self.create_derived_features(df)
            
            # Create report and save
            quality_report = self.generate_quality_report(df)
            output_path, report_path = self.save_results(df, quality_report)
            
            logger.info("Spotify data cleaning pipeline completed successfully")
            return df, quality_report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main execution function."""
    config = SpotifyCleaningConfig()
    cleaner = SpotifyDataCleaner(config)
    
    try:
        cleaned_data, report = cleaner.clean_pipeline()
        
        print("SPOTIFY DATA CLEANING SUMMARY")
        print(f"Total tracks processed: {report['summary']['total_tracks']:,}")
        print(f"Total features: {report['summary']['total_features']}")
        print(f"Memory usage: {report['summary']['memory_usage_mb']} MB")
        
        if 'genres' in report['cleaning_statistics']['cleaning_steps']:
            genre_stats = report['cleaning_statistics']['cleaning_steps']['genres']
            print(f"\nGenre processing:")
            print(f"  Original genres: {genre_stats['original_unique_genres']}")
            print(f"  Genre groups: {genre_stats['genre_groups_created']}")
        
        if 'derived_features' in report['cleaning_statistics']['feature_engineering']:
            eng_stats = report['cleaning_statistics']['feature_engineering']['derived_features']
            print(f"\nFeature engineering:")
            print(f"  New features created: {eng_stats['features_created']}")
        
        if report['popularity_stats']:
            pop_stats = report['popularity_stats']
            print(f"\nPopularity insights:")
            print(f"  Average popularity: {pop_stats['average_popularity']}")
            print(f"  Popular tracks: {pop_stats['popular_tracks_percent']}%")
        
        print("\nTop 5 genres:")
        for genre, count in list(report['genre_distribution'].items())[:5]:
            print(f"  {genre}: {count:,} tracks")
        
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