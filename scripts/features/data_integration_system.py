# data_integration_system.py
'''
Master Data Integration System for combining multiple music datasets
Master Dataset Creation
- Fuzzy matching to match songs across datasets using artist and track combinations
- Composite keys with automatic deduplication
- Higher quality datasets are prioritized in conflicts
Missing Data Handling
- Apply similar song imputation for audio features
- Flag missing data instead of fake imputation
- Genre-bsed demographic defaults
- Cross-platform normalization and averaging
Validation Framework
- Cross-validate for consistency across platforms
- Detect anomalies and outliers in audio feature validation
- Validate genre consistency in genre mapping
- Calculate overall data quality metrics 
'''

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import re
import json
import pickle
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MasterDataIntegrator:
    """
    Master Data Integration System for combining multiple music datasets
    with fuzzy matching, missing data handling, and validation
    """
    
    def __init__(self, data_path='data/cleaned_music', cache_dir='integration_cache'):
        self.data_path = data_path
        self.cache_dir = cache_dir
        
        # Audio features that are critical for analysis
        self.critical_audio_features = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness'
        ]
        
        # Standardized column mappings for track, artist, popularity, and genre variations
        self.column_mappings = {
            'track': 'track_name_clean',
            'title': 'track_name_clean', 
            'name': 'track_name_clean',
            'song': 'track_name_clean',
            
            'artist': 'artist_name_clean',
            'artist_name': 'artist_name_clean',
            'artists': 'artist_name_clean',
            'performer': 'artist_name_clean',
            
            'popularity': 'popularity_score',
            'track_pop': 'popularity_score',
            'views': 'view_count',
            'streams': 'stream_count',
            
            'genre': 'genre_clean',
            'track_genre': 'genre_clean',
            'category': 'genre_clean'
        }
        
        # Dataset source configurations
        self.dataset_configs = {
            'music_info_clean.csv': {
                'source': 'music_info',
                'priority': 1,  # Highest priority for audio features
                'key_columns': ['name', 'artist'],
                'popularity_column': None,
                'has_audio_features': True
            },
            'spotify_tracks_clean.csv': {
                'source': 'spotify_main',
                'priority': 2,
                'key_columns': ['track_name', 'artists'],
                'popularity_column': 'popularity',
                'has_audio_features': True
            },
            'spotify_ds_clean.csv': {
                'source': 'spotify_secondary',
                'priority': 3,
                'key_columns': ['track_name', 'artists'],
                'popularity_column': 'popularity',
                'has_audio_features': True
            },
            'spotify_yt_clean.csv': {
                'source': 'youtube',
                'priority': 4,
                'key_columns': ['track', 'artist'],
                'popularity_column': 'views',
                'has_audio_features': True
            },
            'tiktok_2021_clean.csv': {
                'source': 'tiktok_2021',
                'priority': 5,
                'key_columns': ['track_name', 'artist_name'],
                'popularity_column': 'track_pop',
                'has_audio_features': True
            },
            'tiktok_2022_clean.csv': {
                'source': 'tiktok_2022',
                'priority': 6,
                'key_columns': ['track_name', 'artist_name'],
                'popularity_column': 'track_pop',
                'has_audio_features': True
            }
        }
        
        # Initialize storage
        self.raw_datasets = {}
        self.master_dataset = None
        self.id_mappings = {}
        self.validation_results = {}
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def clean_text_for_matching(self, text: str) -> str:
        """Clean text for fuzzy matching"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Remove common variations and noise
        replacements = {
            r'\(.*?\)': '',  # Remove parentheses content
            r'\[.*?\]': '',  # Remove bracket content
            r'feat\.?\s*': ' ',  # Remove feat/featuring
            r'ft\.?\s*': ' ',
            r'featuring\s*': ' ',
            r'&': 'and',
            r'[\'""]': '',  # Remove quotes
            r'[^\w\s]': ' ',  # Remove special characters except spaces
            r'\s+': ' '  # Multiple spaces to single space
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def create_composite_key(self, track_name: str, artist_name: str) -> str:
        """Create a standardized composite key for matching"""
        clean_track = self.clean_text_for_matching(track_name)
        clean_artist = self.clean_text_for_matching(artist_name)
        
        return f"{clean_artist}||{clean_track}"
    
    def load_and_standardize_datasets(self):
        """Load all datasets and standardize column names"""
        logger.info("Loading and standardizing datasets...")
        
        for filename, config in self.dataset_configs.items():
            filepath = os.path.join(self.data_path, filename)
            
            # Try different possible paths based on your directory structure
            possible_paths = [
                filepath,  # data/cleaned_music/filename
                os.path.join('data/cleaned_normalized', filename.replace('_clean.csv', '_normalized.csv')),  # normalized versions
                os.path.join('data/cleaned_market', filename),  # market data
                os.path.join('data/raw', filename)  # fallback to raw
            ]
            
            file_found = False
            for test_path in possible_paths:
                if os.path.exists(test_path):
                    filepath = test_path
                    file_found = True
                    break
            
            if file_found:
                try:
                    logger.info(f"Loading {filename} from {filepath}...")
                    df = pd.read_csv(filepath)
                    
                    # Standardize column names
                    df_std = self._standardize_columns(df, config)
                    
                    # Add source information
                    df_std['source_dataset'] = config['source']
                    df_std['source_priority'] = config['priority']
                    df_std['source_file'] = filename
                    
                    # Create composite keys
                    df_std['composite_key'] = df_std.apply(
                        lambda row: self.create_composite_key(
                            row.get('track_name_clean', ''),
                            row.get('artist_name_clean', '')
                        ), axis=1
                    )
                    
                    # Add original IDs for mapping
                    df_std['original_index'] = df_std.index
                    df_std['dataset_id'] = f"{config['source']}_{df_std.index}"
                    
                    self.raw_datasets[config['source']] = df_std
                    logger.info(f"Loaded {len(df_std)} records from {filename}")
                    
                except Exception as e:
                    logger.warning(f"Could not load {filename} from {filepath}: {e}")
            else:
                logger.warning(f"File not found: {filename} (searched in multiple locations)")
        
        logger.info(f"Successfully loaded {len(self.raw_datasets)} datasets")
        return self.raw_datasets
    
    def _standardize_columns(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Standardize column names according to mapping"""
        df_std = df.copy()
        
        # Extract key columns for track and artist names
        key_cols = config['key_columns']
        if len(key_cols) >= 2:
            track_col = key_cols[0]
            artist_col = key_cols[1]
            
            if track_col in df_std.columns:
                df_std['track_name_clean'] = df_std[track_col].astype(str)
            if artist_col in df_std.columns:
                df_std['artist_name_clean'] = df_std[artist_col].astype(str)
        
        # Map popularity column
        pop_col = config.get('popularity_column')
        if pop_col and pop_col in df_std.columns:
            df_std['popularity_score'] = pd.to_numeric(df_std[pop_col], errors='coerce')
        
        # Apply general column mappings
        for old_col, new_col in self.column_mappings.items():
            if old_col in df_std.columns and new_col not in df_std.columns:
                df_std[new_col] = df_std[old_col]
        
        # Ensure critical audio features are present
        for feature in self.critical_audio_features:
            if feature in df_std.columns:
                df_std[feature] = pd.to_numeric(df_std[feature], errors='coerce')
            else:
                df_std[feature] = np.nan
        
        return df_std
    
    def perform_fuzzy_matching(self, threshold: int = 85) -> Dict[str, List[str]]:
        """Perform fuzzy matching to find duplicate tracks across datasets"""
        logger.info(f"Performing fuzzy matching with threshold {threshold}...")
        
        if not self.raw_datasets:
            self.load_and_standardize_datasets()
        
        # Collect all composite keys from all datasets
        all_keys = {}
        for source, df in self.raw_datasets.items():
            for idx, row in df.iterrows():
                key = row['composite_key']
                if key and key.strip():
                    if key not in all_keys:
                        all_keys[key] = []
                    all_keys[key].append({
                        'source': source,
                        'idx': idx,
                        'track': row.get('track_name_clean', ''),
                        'artist': row.get('artist_name_clean', ''),
                        'priority': row.get('source_priority', 999)
                    })
        
        # Find fuzzy matches
        match_groups = {}
        processed_keys = set()
        
        for key1 in all_keys.keys():
            if key1 in processed_keys:
                continue
            
            # Find all keys similar to this one
            similar_keys = [key1]
            processed_keys.add(key1)
            
            for key2 in all_keys.keys():
                if key2 in processed_keys:
                    continue
                
                # Calculate similarity
                similarity = fuzz.ratio(key1, key2)
                
                if similarity >= threshold:
                    similar_keys.append(key2)
                    processed_keys.add(key2)
            
            # Group all records with similar keys
            if len(similar_keys) > 1 or len(all_keys[key1]) > 1:
                group_id = f"group_{len(match_groups)}"
                group_records = []
                
                for key in similar_keys:
                    group_records.extend(all_keys[key])
                
                # Sort by priority (lower number = higher priority)
                group_records.sort(key=lambda x: x['priority'])
                match_groups[group_id] = group_records
        
        self.match_groups = match_groups
        logger.info(f"Found {len(match_groups)} potential duplicate groups")
        
        return match_groups
    
    def create_master_dataset(self) -> pd.DataFrame:
        """Create the master dataset by merging and deduplicating"""
        logger.info("Creating master dataset...")
        
        if not hasattr(self, 'match_groups'):
            self.perform_fuzzy_matching()
        
        master_records = []
        id_mappings = {}
        
        # Process each match group
        for group_id, records in self.match_groups.items():
            # Create master record from highest priority source
            master_record = self._create_master_record(records)
            master_records.append(master_record)
            
            # Store ID mappings
            master_id = master_record['master_id']
            id_mappings[master_id] = [
                {'source': r['source'], 'original_idx': r['idx']} for r in records
            ]
        
        # Add records that didn't match anything (singletons)
        singleton_count = 0
        for source, df in self.raw_datasets.items():
            for idx, row in df.iterrows():
                # Check if this record is already in a group
                in_group = False
                for group_records in self.match_groups.values():
                    if any(r['source'] == source and r['idx'] == idx for r in group_records):
                        in_group = True
                        break
                
                if not in_group:
                    # Create singleton master record
                    master_record = self._create_master_record([{
                        'source': source,
                        'idx': idx,
                        'track': row.get('track_name_clean', ''),
                        'artist': row.get('artist_name_clean', ''),
                        'priority': row.get('source_priority', 999)
                    }])
                    master_records.append(master_record)
                    singleton_count += 1
        
        # Create master dataframe
        self.master_dataset = pd.DataFrame(master_records)
        self.id_mappings = id_mappings
        
        logger.info(f"Created master dataset with {len(master_records)} unique tracks")
        logger.info(f"Including {singleton_count} singleton records")
        
        return self.master_dataset
    
    def _create_master_record(self, records: List[Dict]) -> Dict:
        """Create a master record from a group of matching records"""
        # Use highest priority record as base
        primary_record = records[0]
        primary_source = primary_record['source']
        primary_idx = primary_record['idx']
        primary_df = self.raw_datasets[primary_source]
        primary_row = primary_df.iloc[primary_idx]
        
        master_record = {
            'master_id': f"master_{len(self.id_mappings)}",
            'track_name': primary_row.get('track_name_clean', ''),
            'artist_name': primary_row.get('artist_name_clean', ''),
            'primary_source': primary_source,
            'source_count': len(records),
            'all_sources': [r['source'] for r in records]
        }
        
        # Aggregate audio features (prioritize non-null values from higher priority sources)
        for feature in self.critical_audio_features:
            feature_values = []
            for record in records:
                source_df = self.raw_datasets[record['source']]
                source_row = source_df.iloc[record['idx']]
                value = source_row.get(feature)
                if pd.notna(value):
                    feature_values.append(float(value))
            
            if feature_values:
                # Use median for robustness
                master_record[feature] = np.median(feature_values)
            else:
                master_record[feature] = np.nan
        
        # Aggregate popularity scores
        popularity_values = []
        view_counts = []
        stream_counts = []
        
        for record in records:
            source_df = self.raw_datasets[record['source']]
            source_row = source_df.iloc[record['idx']]
            
            # Collect different types of popularity metrics
            pop_score = source_row.get('popularity_score')
            if pd.notna(pop_score):
                popularity_values.append(float(pop_score))
            
            view_count = source_row.get('view_count')
            if pd.notna(view_count):
                view_counts.append(float(view_count))
            
            stream_count = source_row.get('stream_count')
            if pd.notna(stream_count):
                stream_counts.append(float(stream_count))
        
        # Calculate aggregated popularity metrics
        master_record['avg_popularity'] = np.mean(popularity_values) if popularity_values else np.nan
        master_record['max_popularity'] = np.max(popularity_values) if popularity_values else np.nan
        master_record['total_views'] = np.sum(view_counts) if view_counts else np.nan
        master_record['total_streams'] = np.sum(stream_counts) if stream_counts else np.nan
        
        # Handle genre information
        genres = []
        for record in records:
            source_df = self.raw_datasets[record['source']]
            source_row = source_df.iloc[record['idx']]
            genre = source_row.get('genre_clean')
            if pd.notna(genre) and str(genre).lower() not in ['unknown', 'nan', '']:
                genres.append(str(genre))
        
        if genres:
            # Use most common genre
            genre_counts = pd.Series(genres).value_counts()
            master_record['genre'] = genre_counts.index[0]
            master_record['all_genres'] = list(genre_counts.index)
        else:
            master_record['genre'] = 'unknown'
            master_record['all_genres'] = []
        
        # Platform presence indicators
        platform_presence = {
            'spotify': any('spotify' in r['source'] for r in records),
            'tiktok': any('tiktok' in r['source'] for r in records),
            'youtube': any('youtube' in r['source'] for r in records)
        }
        master_record.update(platform_presence)
        
        return master_record
    
    def handle_missing_data(self):
        """Handle missing data using prioritized approach"""
        logger.info("Handling missing data...")
        
        if self.master_dataset is None:
            self.create_master_dataset()
        
        # Critical features: Use similar song imputation
        self._impute_audio_features()
        
        # Platform metrics: Keep as null, add flags
        self._flag_missing_platform_metrics()
        
        # Demographics: Use genre-based defaults
        self._impute_demographics()
        
        # Popularity scores: Use cross-platform averaging
        self._normalize_popularity_scores()
        
        logger.info("Missing data handling complete")
    
    def _impute_audio_features(self):
        """Impute missing audio features using similar songs"""
        logger.info("Imputing missing audio features...")
        
        for feature in self.critical_audio_features:
            missing_mask = self.master_dataset[feature].isna()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                continue
            
            logger.info(f"Imputing {missing_count} missing values for {feature}")
            
            # For songs with missing features, find similar songs by genre and other features
            for idx in self.master_dataset[missing_mask].index:
                song = self.master_dataset.loc[idx]
                
                # Find similar songs by genre
                genre_mask = self.master_dataset['genre'] == song['genre']
                similar_songs = self.master_dataset[genre_mask & ~self.master_dataset[feature].isna()]
                
                if len(similar_songs) == 0:
                    # Fall back to overall median
                    imputed_value = self.master_dataset[feature].median()
                else:
                    # Use median of similar songs
                    imputed_value = similar_songs[feature].median()
                
                self.master_dataset.loc[idx, feature] = imputed_value
        
        # Validate that all critical features are now filled
        for feature in self.critical_audio_features:
            remaining_missing = self.master_dataset[feature].isna().sum()
            if remaining_missing > 0:
                logger.warning(f"{feature} still has {remaining_missing} missing values")
    
    def _flag_missing_platform_metrics(self):
        """Add flags for missing platform metrics instead of imputing"""
        platform_metrics = ['total_views', 'total_streams', 'avg_popularity']
        
        for metric in platform_metrics:
            flag_col = f'{metric}_missing'
            self.master_dataset[flag_col] = self.master_dataset[metric].isna()
    
    def _impute_demographics(self):
        """Use genre-based defaults for demographic information"""
        # Genre-demographic mappings
        genre_demographics = {
            'pop': {'primary_age': '18-34', 'secondary_age': '35-49'},
            'rock': {'primary_age': '25-49', 'secondary_age': '50+'},
            'hip hop': {'primary_age': '16-34', 'secondary_age': '35-49'},
            'rap': {'primary_age': '16-34', 'secondary_age': '35-49'},
            'electronic': {'primary_age': '18-34', 'secondary_age': '16-24'},
            'country': {'primary_age': '35-65', 'secondary_age': '25-49'},
            'jazz': {'primary_age': '35+', 'secondary_age': '25-49'},
            'classical': {'primary_age': '50+', 'secondary_age': '35-64'},
            'r&b': {'primary_age': '25-49', 'secondary_age': '18-34'},
            'soul': {'primary_age': '35-64', 'secondary_age': '25-49'}
        }
        
        # Apply demographic defaults
        for idx, row in self.master_dataset.iterrows():
            genre = str(row['genre']).lower()
            
            # Find best matching genre
            matched_genre = None
            for known_genre in genre_demographics.keys():
                if known_genre in genre:
                    matched_genre = known_genre
                    break
            
            if matched_genre:
                demo_info = genre_demographics[matched_genre]
                self.master_dataset.loc[idx, 'predicted_primary_demographic'] = demo_info['primary_age']
                self.master_dataset.loc[idx, 'predicted_secondary_demographic'] = demo_info['secondary_age']
            else:
                # Default for unknown genres
                self.master_dataset.loc[idx, 'predicted_primary_demographic'] = '18-49'
                self.master_dataset.loc[idx, 'predicted_secondary_demographic'] = 'all_ages'
    
    def _normalize_popularity_scores(self):
        """Normalize popularity scores across platforms"""
        logger.info("Normalizing popularity scores...")
        
        # Create normalized popularity score (0-100 scale)
        popularity_components = []
        
        # Normalize different popularity metrics to 0-100 scale
        if 'avg_popularity' in self.master_dataset.columns:
            avg_pop = self.master_dataset['avg_popularity'].copy()
            # Assume avg_popularity is already 0-100
            avg_pop_norm = avg_pop.fillna(50)  # Default to middle value
            popularity_components.append(avg_pop_norm)
        
        if 'total_views' in self.master_dataset.columns:
            views = self.master_dataset['total_views'].copy()
            views_log = np.log10(views.replace(0, 1).fillna(1))
            views_norm = ((views_log - views_log.min()) / (views_log.max() - views_log.min())) * 100
            popularity_components.append(views_norm)
        
        if 'total_streams' in self.master_dataset.columns:
            streams = self.master_dataset['total_streams'].copy()
            streams_log = np.log10(streams.replace(0, 1).fillna(1))
            streams_norm = ((streams_log - streams_log.min()) / (streams_log.max() - streams_log.min())) * 100
            popularity_components.append(streams_norm)
        
        # Calculate composite popularity score
        if popularity_components:
            self.master_dataset['normalized_popularity'] = np.mean(popularity_components, axis=0)
        else:
            self.master_dataset['normalized_popularity'] = 50  # Default middle value
        
        # Ensure it's in 0-100 range
        self.master_dataset['normalized_popularity'] = np.clip(
            self.master_dataset['normalized_popularity'], 0, 100
        )
    
    def create_validation_framework(self):
        """Create comprehensive validation framework"""
        logger.info("Creating validation framework...")
        
        if self.master_dataset is None:
            self.create_master_dataset()
        
        validation_results = {
            'popularity_validation': self._validate_popularity_scores(),
            'audio_feature_validation': self._validate_audio_features(),
            'genre_validation': self._validate_genre_mappings(),
            'demographic_validation': self._validate_demographic_data(),
            'data_quality_summary': self._generate_data_quality_summary()
        }
        
        self.validation_results = validation_results
        
        # Save validation results
        validation_file = os.path.join(self.cache_dir, 'validation_results.json')
        with open(validation_file, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(validation_results, f, indent=2, default=convert_types)
        
        logger.info("Validation framework created")
        return validation_results
    
    def _validate_popularity_scores(self):
        """Validate consistency of popularity scores across datasets"""
        validation = {
            'cross_platform_correlation': {},
            'outlier_detection': {},
            'consistency_scores': {}
        }
        
        # For songs present on multiple platforms, check correlation
        multi_platform_songs = self.master_dataset[self.master_dataset['source_count'] > 1]
        
        if len(multi_platform_songs) > 10:
            # Calculate consistency metrics
            popularity_variance = multi_platform_songs['avg_popularity'].var()
            validation['consistency_scores']['popularity_variance'] = float(popularity_variance)
            
            # Identify outliers (songs with very different popularity across platforms)
            outliers = []
            for idx, song in multi_platform_songs.iterrows():
                if song['max_popularity'] - song['avg_popularity'] > 30:  # Threshold for outlier
                    outliers.append({
                        'master_id': song['master_id'],
                        'track': song['track_name'],
                        'artist': song['artist_name'],
                        'avg_pop': float(song['avg_popularity']),
                        'max_pop': float(song['max_popularity'])
                    })
            
            validation['outlier_detection']['popularity_outliers'] = outliers
        
        return validation
    
    def _validate_audio_features(self):
        """Validate audio feature consistency and detect anomalies"""
        validation = {
            'feature_ranges': {},
            'correlation_matrix': {},
            'anomaly_detection': {}
        }
        
        # Check feature ranges
        for feature in self.critical_audio_features:
            if feature in self.master_dataset.columns:
                values = self.master_dataset[feature].dropna()
                validation['feature_ranges'][feature] = {
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'missing_count': int(self.master_dataset[feature].isna().sum())
                }
        
        # Calculate feature correlations
        feature_cols = [f for f in self.critical_audio_features if f in self.master_dataset.columns]
        if len(feature_cols) > 1:
            corr_matrix = self.master_dataset[feature_cols].corr()
            validation['correlation_matrix'] = corr_matrix.to_dict()
        
        # Detect songs with anomalous feature combinations
        anomalies = []
        for feature in ['danceability', 'energy', 'valence']:
            if feature in self.master_dataset.columns:
                # Find songs with extreme values (beyond 3 standard deviations)
                mean_val = self.master_dataset[feature].mean()
                std_val = self.master_dataset[feature].std()
                
                extreme_mask = (
                    (self.master_dataset[feature] > mean_val + 3 * std_val) |
                    (self.master_dataset[feature] < mean_val - 3 * std_val)
                )
                
                extreme_songs = self.master_dataset[extreme_mask]
                for idx, song in extreme_songs.iterrows():
                    anomalies.append({
                        'master_id': song['master_id'],
                        'track': song['track_name'],
                        'feature': feature,
                        'value': float(song[feature]),
                        'z_score': float((song[feature] - mean_val) / std_val)
                    })
        
        validation['anomaly_detection']['feature_anomalies'] = anomalies
        
        return validation
    
    def _validate_genre_mappings(self):
        """Validate genre consistency and mappings"""
        validation = {
            'genre_distribution': {},
            'genre_consistency': {},
            'genre_mappings': {}
        }
        
        # Genre distribution
        genre_counts = self.master_dataset['genre'].value_counts()
        validation['genre_distribution'] = genre_counts.to_dict()
        
        # Check consistency for songs with multiple genre sources
        multi_genre_songs = self.master_dataset[
            self.master_dataset['all_genres'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False)
        ]
        
        inconsistent_genres = []
        for idx, song in multi_genre_songs.iterrows():
            genres = song['all_genres']
            if len(set(genres)) > 1:  # Different genres assigned
                inconsistent_genres.append({
                    'master_id': song['master_id'],
                    'track': song['track_name'],
                    'assigned_genre': song['genre'],
                    'all_genres': genres
                })
        
        validation['genre_consistency']['inconsistent_count'] = len(inconsistent_genres)
        validation['genre_consistency']['inconsistent_examples'] = inconsistent_genres[:10]
        
        return validation
    
    def _validate_demographic_data(self):
        """Validate demographic predictions and mappings"""
        validation = {
            'demographic_distribution': {},
            'genre_demographic_mapping': {},
            'confidence_scores': {}
        }
        
        # Demographic distribution
        if 'predicted_primary_demographic' in self.master_dataset.columns:
            demo_dist = self.master_dataset['predicted_primary_demographic'].value_counts()
            validation['demographic_distribution'] = demo_dist.to_dict()
        
        # Genre-demographic mapping validation
        genre_demo_map = {}
        for genre in self.master_dataset['genre'].unique():
            if genre != 'unknown':
                genre_songs = self.master_dataset[self.master_dataset['genre'] == genre]
                if len(genre_songs) > 0:
                    primary_demos = genre_songs['predicted_primary_demographic'].value_counts()
                    if len(primary_demos) > 0:
                        genre_demo_map[genre] = {
                            'most_common_demo': primary_demos.index[0],
                            'demo_distribution': primary_demos.to_dict(),
                            'song_count': len(genre_songs)
                        }
        
        validation['genre_demographic_mapping'] = genre_demo_map
        
        return validation
    
    def _generate_data_quality_summary(self):
        """Generate overall data quality summary"""
        summary = {
            'total_records': len(self.master_dataset),
            'duplicate_groups': len(self.match_groups) if hasattr(self, 'match_groups') else 0,
            'data_completeness': {},
            'source_distribution': {},
            'quality_scores': {}
        }
        
        # Data completeness
        for col in ['track_name', 'artist_name'] + self.critical_audio_features:
            if col in self.master_dataset.columns:
                completeness = (1 - self.master_dataset[col].isna().sum() / len(self.master_dataset)) * 100
                summary['data_completeness'][col] = round(float(completeness), 2)
        
        # Source distribution
        source_dist = self.master_dataset['primary_source'].value_counts()
        summary['source_distribution'] = source_dist.to_dict()
        
        # Quality scores
        avg_completeness = np.mean(list(summary['data_completeness'].values()))
        source_diversity = len(summary['source_distribution'])
        
        # Overall quality score (0-100)
        quality_score = (avg_completeness * 0.7) + (min(source_diversity * 10, 30) * 0.3)
        summary['quality_scores']['overall_quality'] = round(float(quality_score), 2)
        summary['quality_scores']['completeness_score'] = round(float(avg_completeness), 2)
        summary['quality_scores']['diversity_score'] = source_diversity
        
        return summary
    
    def save_master_dataset(self, output_file='master_music_dataset.csv'):
        """Save the master dataset and all associated files"""
        if self.master_dataset is None:
            logger.error("No master dataset to save. Run create_master_dataset() first.")
            return
        
        # Save master dataset
        master_file = os.path.join(self.cache_dir, output_file)
        self.master_dataset.to_csv(master_file, index=False)
        logger.info(f"Master dataset saved to {master_file}")
        
        # Save ID mappings
        id_mapping_file = os.path.join(self.cache_dir, 'id_mappings.json')
        with open(id_mapping_file, 'w') as f:
            json.dump(self.id_mappings, f, indent=2)
        
        # Save match groups
        if hasattr(self, 'match_groups'):
            match_groups_file = os.path.join(self.cache_dir, 'match_groups.json')
            with open(match_groups_file, 'w') as f:
                json.dump(self.match_groups, f, indent=2)
        
        # Save processing metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_records': len(self.master_dataset),
            'source_datasets': list(self.dataset_configs.keys()),
            'processing_steps': [
                'dataset_loading',
                'column_standardization', 
                'fuzzy_matching',
                'record_merging',
                'missing_data_handling',
                'validation'
            ],
            'data_quality': self.validation_results.get('data_quality_summary', {}) if hasattr(self, 'validation_results') else {}
        }
        
        metadata_file = os.path.join(self.cache_dir, 'master_dataset_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"All integration files saved to {self.cache_dir}")
    
    def run_complete_integration(self, fuzzy_threshold=85, save_output=True):
        """Run the complete data integration pipeline in 6 phases"""
        logger.info("Starting complete data integration pipeline...")
        
        # Load and standardize datasets
        logger.info("Phase 1: Loading and standardizing datasets")
        self.load_and_standardize_datasets()
        
        if not self.raw_datasets:
            logger.error("No datasets loaded. Check your file paths.")
            return None
        
        # Perform fuzzy matching
        logger.info("Phase 2: Performing fuzzy matching")
        self.perform_fuzzy_matching(threshold=fuzzy_threshold)
        
        # Create master dataset
        logger.info("Phase 3: Creating master dataset")
        self.create_master_dataset()
        
        # Handle missing data
        logger.info("Phase 4: Handling missing data")
        self.handle_missing_data()
        
        # Create validation framework
        logger.info("Phase 5: Creating validation framework")
        self.create_validation_framework()
        
        # Save results
        if save_output:
            logger.info("Phase 6: Saving results")
            self.save_master_dataset()
        
        logger.info("Data integration pipeline complete!")
        
        # Return summary
        return {
            'master_dataset': self.master_dataset,
            'validation_results': self.validation_results,
            'integration_summary': {
                'total_master_records': len(self.master_dataset),
                'source_datasets': len(self.raw_datasets),
                'duplicate_groups': len(self.match_groups),
                'data_quality_score': self.validation_results['data_quality_summary']['quality_scores']['overall_quality']
            }
        }
    
    def get_song_by_id(self, master_id: str):
        """Retrieve a song and all its source records by master ID"""
        if self.master_dataset is None:
            logger.error("No master dataset loaded")
            return None
        
        # Get master record
        master_record = self.master_dataset[self.master_dataset['master_id'] == master_id]
        if len(master_record) == 0:
            logger.error(f"Master ID {master_id} not found")
            return None
        
        master_record = master_record.iloc[0]
        
        # Get all source records
        source_records = []
        if master_id in self.id_mappings:
            for mapping in self.id_mappings[master_id]:
                source = mapping['source']
                original_idx = mapping['original_idx']
                
                if source in self.raw_datasets:
                    source_df = self.raw_datasets[source]
                    if original_idx < len(source_df):
                        source_record = source_df.iloc[original_idx].to_dict()
                        source_record['source_dataset'] = source
                        source_records.append(source_record)
        
        return {
            'master_record': master_record.to_dict(),
            'source_records': source_records,
            'record_count': len(source_records)
        }
    
    def search_songs(self, query: str, limit: int = 10, search_fields=['track_name', 'artist_name']):
        """Search for songs in the master dataset"""
        if self.master_dataset is None:
            logger.error("No master dataset loaded")
            return []
        
        query_clean = self.clean_text_for_matching(query)
        results = []
        
        for idx, row in self.master_dataset.iterrows():
            max_similarity = 0
            
            for field in search_fields:
                if field in row:
                    field_value = self.clean_text_for_matching(str(row[field]))
                    similarity = fuzz.partial_ratio(query_clean, field_value)
                    max_similarity = max(max_similarity, similarity)
            
            if max_similarity > 60:  # Threshold for search results
                results.append({
                    'master_id': row['master_id'],
                    'track_name': row['track_name'],
                    'artist_name': row['artist_name'],
                    'similarity_score': max_similarity,
                    'source_count': row['source_count'],
                    'normalized_popularity': row.get('normalized_popularity', 0)
                })
        
        # Sort by similarity and popularity
        results.sort(key=lambda x: (x['similarity_score'], x['normalized_popularity']), reverse=True)
        
        return results[:limit]
    
    def generate_integration_report(self):
        """Generate a comprehensive integration report"""
        if not hasattr(self, 'validation_results'):
            logger.error("No validation results available. Run complete integration first.")
            return None
        
        report = {
            'executive_summary': self._generate_executive_summary(),
            'data_sources': self._analyze_data_sources(),
            'quality_metrics': self.validation_results['data_quality_summary'],
            'recommendations': self._generate_recommendations(),
            'detailed_findings': self.validation_results
        }
        
        # Save report
        report_file = os.path.join(self.cache_dir, 'integration_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_executive_summary(self):
        """Generate executive summary of the integration"""
        if self.master_dataset is None:
            return {}
        
        summary = {
            'total_unique_tracks': len(self.master_dataset),
            'source_datasets_processed': len(self.raw_datasets),
            'duplicate_pairs_found': len(self.match_groups) if hasattr(self, 'match_groups') else 0,
            'data_quality_score': self.validation_results['data_quality_summary']['quality_scores']['overall_quality'],
            'completeness_by_feature': self.validation_results['data_quality_summary']['data_completeness'],
            'top_platforms': dict(list(self.validation_results['data_quality_summary']['source_distribution'].items())[:3])
        }
        
        return summary
    
    def _analyze_data_sources(self):
        """Analyze contribution of each data source"""
        analysis = {}
        
        for source, config in self.dataset_configs.items():
            if config['source'] in self.raw_datasets:
                df = self.raw_datasets[config['source']]
                
                analysis[source] = {
                    'total_records': len(df),
                    'priority': config['priority'],
                    'has_audio_features': config['has_audio_features'],
                    'contribution_to_master': sum(1 for _, row in self.master_dataset.iterrows() 
                                                if row['primary_source'] == config['source']),
                    'data_completeness': {
                        feature: round((1 - df[feature].isna().sum() / len(df)) * 100, 2)
                        for feature in self.critical_audio_features
                        if feature in df.columns
                    }
                }
        
        return analysis
    
    def _generate_recommendations(self):
        """Generate recommendations for data quality improvement"""
        recommendations = []
        
        quality_score = self.validation_results['data_quality_summary']['quality_scores']['overall_quality']
        
        # Quality-based recommendations
        if quality_score < 70:
            recommendations.append({
                'priority': 'High',
                'category': 'Data Quality',
                'recommendation': 'Overall data quality is below 70%. Consider improving data collection processes.',
                'impact': 'High'
            })
        
        # Completeness recommendations
        completeness = self.validation_results['data_quality_summary']['data_completeness']
        for feature, score in completeness.items():
            if score < 80:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Data Completeness',
                    'recommendation': f'{feature} is only {score}% complete. Consider additional data sources or imputation strategies.',
                    'impact': 'Medium'
                })
        
        # Popularity validation recommendations
        popularity_validation = self.validation_results.get('popularity_validation', {})
        outliers = popularity_validation.get('outlier_detection', {}).get('popularity_outliers', [])
        
        if len(outliers) > 10:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Data Consistency',
                'recommendation': f'Found {len(outliers)} songs with inconsistent popularity across platforms. Review normalization methods.',
                'impact': 'Medium'
            })
        
        # Genre consistency recommendations
        genre_validation = self.validation_results.get('genre_validation', {})
        inconsistent_count = genre_validation.get('genre_consistency', {}).get('inconsistent_count', 0)
        
        if inconsistent_count > 50:
            recommendations.append({
                'priority': 'Low',
                'category': 'Genre Mapping',
                'recommendation': f'{inconsistent_count} songs have inconsistent genre assignments. Consider genre standardization.',
                'impact': 'Low'
            })
        
        return recommendations


# CLI interface and example use
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Master Data Integration System')
    parser.add_argument('command', choices=['integrate', 'validate', 'search', 'report'], 
                       help='Command to execute')
    parser.add_argument('--data-dir', default='data/cleaned_music', help='Data directory path')
    parser.add_argument('--cache-dir', default='integration_cache', help='Cache directory path')
    parser.add_argument('--fuzzy-threshold', type=int, default=85, help='Fuzzy matching threshold')
    parser.add_argument('--query', help='Search query (for search command)')
    parser.add_argument('--limit', type=int, default=10, help='Search result limit')
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = MasterDataIntegrator(
        data_path=args.data_dir,
        cache_dir=args.cache_dir
    )
    
    if args.command == 'integrate':
        print("Starting complete data integration...")
        results = integrator.run_complete_integration(fuzzy_threshold=args.fuzzy_threshold)
        
        if results:
            print("\n=== INTEGRATION COMPLETE ===")
            summary = results['integration_summary']
            print(f"Master dataset created: {summary['total_master_records']} unique tracks")
            print(f"Source datasets processed: {summary['source_datasets']}")
            print(f"Duplicate groups found: {summary['duplicate_groups']}")
            print(f"Data quality score: {summary['data_quality_score']:.1f}/100")
            
            # Show sample of results
            print(f"\n=== SAMPLE MASTER RECORDS ===")
            sample = results['master_dataset'].head()
            for idx, row in sample.iterrows():
                print(f"{row['track_name']} by {row['artist_name']}")
                print(f"  Sources: {row['source_count']} datasets")
                print(f"  Popularity: {row.get('normalized_popularity', 'N/A')}")
                print(f"  Genre: {row.get('genre', 'unknown')}")
                print()
    
    elif args.command == 'validate':
        print("Running validation framework...")
        
        # Load existing master dataset
        master_file = os.path.join(args.cache_dir, 'master_music_dataset.csv')
        if os.path.exists(master_file):
            integrator.master_dataset = pd.read_csv(master_file)
            validation_results = integrator.create_validation_framework()
            
            print("\n=== VALIDATION RESULTS ===")
            quality_summary = validation_results['data_quality_summary']
            print(f"Overall quality score: {quality_summary['quality_scores']['overall_quality']:.1f}/100")
            print(f"Data completeness score: {quality_summary['quality_scores']['completeness_score']:.1f}%")
            print(f"Source diversity: {quality_summary['quality_scores']['diversity_score']} datasets")
            
            print(f"\n=== FEATURE COMPLETENESS ===")
            for feature, completeness in quality_summary['data_completeness'].items():
                print(f"{feature}: {completeness:.1f}%")
            
        else:
            print("No master dataset found. Run integration first.")
    
    elif args.command == 'search':
        if not args.query:
            print("Please provide a search query with --query")
            sys.exit(1)
        
        # Load existing master dataset
        master_file = os.path.join(args.cache_dir, 'master_music_dataset.csv')
        if os.path.exists(master_file):
            integrator.master_dataset = pd.read_csv(master_file)
            
            print(f"Searching for: '{args.query}'")
            results = integrator.search_songs(args.query, limit=args.limit)
            
            print(f"\n=== SEARCH RESULTS ({len(results)} found) ===")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['track_name']} by {result['artist_name']}")
                print(f"   Match: {result['similarity_score']}% | Sources: {result['source_count']} | Popularity: {result['normalized_popularity']:.1f}")
                print()
        else:
            print("No master dataset found. Run integration first.")
    
    elif args.command == 'report':
        print("Generating integration report...")
        
        # Load existing validation results
        validation_file = os.path.join(args.cache_dir, 'validation_results.json')
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                integrator.validation_results = json.load(f)
            
            master_file = os.path.join(args.cache_dir, 'master_music_dataset.csv')
            if os.path.exists(master_file):
                integrator.master_dataset = pd.read_csv(master_file)
            
            report = integrator.generate_integration_report()
            
            if report:
                print("\n=== INTEGRATION REPORT ===")
                summary = report['executive_summary']
                print(f"Total unique tracks: {summary['total_unique_tracks']}")
                print(f"Source datasets: {summary['source_datasets_processed']}")
                print(f"Data quality: {summary['data_quality_score']:.1f}/100")
                
                print(f"\n=== RECOMMENDATIONS ===")
                for rec in report['recommendations'][:5]:  # Top 5 recommendations
                    print(f"[{rec['priority']}] {rec['category']}: {rec['recommendation']}")
                
                print(f"\nFull report saved to: {args.cache_dir}/integration_report.json")
        else:
            print("No validation results found. Run integration first.")


# Utility functions for working with the master dataset
class MasterDatasetUtils:
    """Utility functions for working with the integrated master dataset"""
    
    def __init__(self, master_dataset_path):
        self.df = pd.read_csv(master_dataset_path)
    
    def get_platform_statistics(self):
        """Get statistics about platform coverage"""
        stats = {
            'spotify_coverage': self.df['spotify'].sum(),
            'tiktok_coverage': self.df['tiktok'].sum(),
            'youtube_coverage': self.df['youtube'].sum(),
            'multi_platform_songs': self.df[self.df['source_count'] > 1].shape[0],
            'single_platform_songs': self.df[self.df['source_count'] == 1].shape[0]
        }
        return stats
    
    def get_genre_distribution(self):
        """Get genre distribution statistics"""
        return self.df['genre'].value_counts().to_dict()
    
    def get_audio_feature_ranges(self):
        """Get ranges for all audio features"""
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness']
        
        ranges = {}
        for feature in audio_features:
            if feature in self.df.columns:
                ranges[feature] = {
                    'min': float(self.df[feature].min()),
                    'max': float(self.df[feature].max()),
                    'mean': float(self.df[feature].mean()),
                    'median': float(self.df[feature].median())
                }
        return ranges
    
    def find_similar_songs(self, master_id, top_k=5):
        """Find songs similar to a given song based on audio features"""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler
        
        target_song = self.df[self.df['master_id'] == master_id]
        if len(target_song) == 0:
            return []
        
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
        
        # Prepare feature matrix
        feature_matrix = self.df[audio_features].fillna(0)
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Calculate similarities
        target_idx = target_song.index[0]
        target_features = feature_matrix_scaled[target_idx].reshape(1, -1)
        similarities = cosine_similarity(target_features, feature_matrix_scaled)[0]
        
        # Get top similar songs (excluding the target song itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        similar_songs = []
        for idx in similar_indices:
            song = self.df.iloc[idx]
            similar_songs.append({
                'master_id': song['master_id'],
                'track_name': song['track_name'],
                'artist_name': song['artist_name'],
                'similarity_score': float(similarities[idx]),
                'genre': song['genre']
            })
        
        return similar_songs