import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re
import json
import pickle
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import psutil

logger = logging.getLogger(__name__)

class OptimizedMasterDataIntegrator:
    """
    Optimized Master Data Integration System with progress tracking,
    sampling options, and performance monitoring
    """
    
    def __init__(self, data_path='data/cleaned_music', cache_dir='integration_cache', 
                 sample_size=None, single_file=None, progress_bar=True):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.sample_size = sample_size  # Limit rows per file
        self.single_file = single_file  # Process only one file
        self.progress_bar = progress_bar
        
        # Performance tracking
        self.start_time = None
        self.performance_stats = {}
        
        # Audio features that are critical for analysis
        self.critical_audio_features = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness'
        ]
        
        # Standardized column mappings
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
        
        # Dataset configurations - prioritized for performance
        self.dataset_configs = {
            'music_info_clean.csv': {
                'source': 'music_info',
                'priority': 1,
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
        
        # If single file specified, filter configs
        if single_file:
            self.dataset_configs = {single_file: self.dataset_configs.get(single_file, {})}
        
        # Initialize storage
        self.raw_datasets = {}
        self.master_dataset = None
        self.id_mappings = {}
        self.validation_results = {}
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _log_performance(self, step_name, start_time):
        """Log performance metrics for each step"""
        duration = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.performance_stats[step_name] = {
            'duration_seconds': round(duration, 2),
            'memory_mb': round(memory_usage, 2)
        }
        
        logger.info(f"{step_name} completed in {duration:.2f}s (Memory: {memory_usage:.1f}MB)")
    
    def clean_text_for_matching(self, text: str) -> str:
        """Clean text for fuzzy matching - optimized version"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Faster regex compilation and replacement
        text = re.sub(r'\(.*?\)|\[.*?\]', '', text)  # Remove parentheses/brackets
        text = re.sub(r'feat\.?\s*|ft\.?\s*|featuring\s*', ' ', text)  # Remove feat
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Clean spaces
        
        return text
    
    def create_composite_key(self, track_name: str, artist_name: str) -> str:
        """Create a standardized composite key - cached for performance"""
        clean_track = self.clean_text_for_matching(track_name)
        clean_artist = self.clean_text_for_matching(artist_name)
        return f"{clean_artist}||{clean_track}"
    
    def load_and_standardize_datasets(self):
        """Load datasets with progress tracking and sampling"""
        step_start = time.time()
        logger.info("Loading and standardizing datasets...")
        
        if self.progress_bar:
            file_iterator = tqdm(self.dataset_configs.items(), desc="Loading files")
        else:
            file_iterator = self.dataset_configs.items()
        
        for filename, config in file_iterator:
            file_start = time.time()
            
            # Try different possible paths
            possible_paths = [
                os.path.join(self.data_path, filename),
                os.path.join('data/cleaned_normalized', filename.replace('_clean.csv', '_normalized.csv')),
                os.path.join('data/cleaned_market', filename),
                os.path.join('data/raw', filename)
            ]
            
            file_found = False
            for test_path in possible_paths:
                if os.path.exists(test_path):
                    filepath = test_path
                    file_found = True
                    break
            
            if file_found:
                try:
                    if self.progress_bar:
                        file_iterator.set_description(f"Loading {filename}")
                    
                    # Check file size first
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    logger.info(f"Loading {filename} ({file_size_mb:.1f} MB)...")
                    
                    # Load with sampling if specified
                    if self.sample_size:
                        # Read in chunks to sample efficiently
                        chunk_size = min(self.sample_size, 10000)
                        chunks = []
                        
                        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                            chunks.append(chunk)
                            if len(pd.concat(chunks)) >= self.sample_size:
                                break
                        
                        df = pd.concat(chunks).head(self.sample_size)
                        logger.info(f"Sampled {len(df)} rows from {filename}")
                    else:
                        df = pd.read_csv(filepath)
                    
                    # Standardize columns
                    df_std = self._standardize_columns(df, config)
                    
                    # Add metadata
                    df_std['source_dataset'] = config['source']
                    df_std['source_priority'] = config['priority']
                    df_std['source_file'] = filename
                    
                    # Create composite keys with progress
                    if self.progress_bar:
                        tqdm.pandas(desc="Creating keys")
                        df_std['composite_key'] = df_std.progress_apply(
                            lambda row: self.create_composite_key(
                                row.get('track_name_clean', ''),
                                row.get('artist_name_clean', '')
                            ), axis=1
                        )
                    else:
                        df_std['composite_key'] = df_std.apply(
                            lambda row: self.create_composite_key(
                                row.get('track_name_clean', ''),
                                row.get('artist_name_clean', '')
                            ), axis=1
                        )
                    
                    # Add IDs
                    df_std['original_index'] = df_std.index
                    df_std['dataset_id'] = f"{config['source']}_{df_std.index}"
                    
                    self.raw_datasets[config['source']] = df_std
                    
                    file_duration = time.time() - file_start
                    logger.info(f"Loaded {len(df_std)} records from {filename} in {file_duration:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"Could not load {filename} from {filepath}: {e}")
            else:
                logger.warning(f"File not found: {filename}")
        
        self._log_performance("Data Loading", step_start)
        logger.info(f"Successfully loaded {len(self.raw_datasets)} datasets")
        return self.raw_datasets
    
    def _standardize_columns(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Standardize column names - optimized version"""
        df_std = df.copy()
        
        # Extract key columns
        key_cols = config['key_columns']
        if len(key_cols) >= 2:
            track_col, artist_col = key_cols[0], key_cols[1]
            
            if track_col in df_std.columns:
                df_std['track_name_clean'] = df_std[track_col].astype(str)
            if artist_col in df_std.columns:
                df_std['artist_name_clean'] = df_std[artist_col].astype(str)
        
        # Map popularity column
        pop_col = config.get('popularity_column')
        if pop_col and pop_col in df_std.columns:
            df_std['popularity_score'] = pd.to_numeric(df_std[pop_col], errors='coerce')
        
        # Vectorized audio feature processing
        for feature in self.critical_audio_features:
            if feature in df_std.columns:
                df_std[feature] = pd.to_numeric(df_std[feature], errors='coerce')
            else:
                df_std[feature] = np.nan
        
        return df_std
    
    def perform_optimized_fuzzy_matching(self, threshold: int = 85, max_comparisons: int = 100000) -> Dict[str, List[str]]:
        """Optimized fuzzy matching with early stopping and progress tracking"""
        step_start = time.time()
        logger.info(f"Performing optimized fuzzy matching (threshold: {threshold})...")
        
        if not self.raw_datasets:
            self.load_and_standardize_datasets()
        
        # Collect all composite keys
        all_keys = {}
        total_keys = 0
        
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
                    total_keys += 1
        
        logger.info(f"Processing {len(all_keys)} unique keys from {total_keys} total records")
        
        # Optimized matching with early stopping
        match_groups = {}
        processed_keys = set()
        comparison_count = 0
        
        keys_list = list(all_keys.keys())
        
        if self.progress_bar:
            key_iterator = tqdm(keys_list, desc="Fuzzy matching")
        else:
            key_iterator = keys_list
        
        for i, key1 in enumerate(key_iterator):
            if key1 in processed_keys:
                continue
            
            # Early stopping for performance
            if comparison_count > max_comparisons:
                logger.warning(f"Stopping fuzzy matching at {max_comparisons} comparisons for performance")
                break
            
            similar_keys = [key1]
            processed_keys.add(key1)
            
            # Only compare with remaining keys
            for key2 in keys_list[i+1:]:
                if key2 in processed_keys:
                    continue
                
                comparison_count += 1
                
                # Quick length-based filtering
                if abs(len(key1) - len(key2)) > len(key1) * 0.5:
                    continue
                
                similarity = fuzz.ratio(key1, key2)
                
                if similarity >= threshold:
                    similar_keys.append(key2)
                    processed_keys.add(key2)
            
            # Create group if multiple similar keys or multiple records
            if len(similar_keys) > 1 or len(all_keys[key1]) > 1:
                group_id = f"group_{len(match_groups)}"
                group_records = []
                
                for key in similar_keys:
                    group_records.extend(all_keys[key])
                
                # Sort by priority
                group_records.sort(key=lambda x: x['priority'])
                match_groups[group_id] = group_records
        
        self.match_groups = match_groups
        self._log_performance("Fuzzy Matching", step_start)
        logger.info(f"Found {len(match_groups)} potential duplicate groups with {comparison_count} comparisons")
        
        return match_groups
    
    def create_master_dataset_optimized(self) -> pd.DataFrame:
        """Create master dataset with optimized processing"""
        step_start = time.time()
        logger.info("Creating optimized master dataset...")
        
        if not hasattr(self, 'match_groups'):
            self.perform_optimized_fuzzy_matching()
        
        master_records = []
        id_mappings = {}
        
        # Process match groups with progress
        if self.progress_bar:
            group_iterator = tqdm(self.match_groups.items(), desc="Creating master records")
        else:
            group_iterator = self.match_groups.items()
        
        for group_id, records in group_iterator:
            master_record = self._create_master_record_optimized(records)
            master_records.append(master_record)
            
            # Store ID mappings
            master_id = master_record['master_id']
            id_mappings[master_id] = [
                {'source': r['source'], 'original_idx': r['idx']} for r in records
            ]
        
        # Add singletons more efficiently
        singleton_count = 0
        all_processed_records = set()
        
        # Track which records are already in groups
        for group_records in self.match_groups.values():
            for record in group_records:
                all_processed_records.add((record['source'], record['idx']))
        
        # Add unprocessed records
        for source, df in self.raw_datasets.items():
            for idx in df.index:
                if (source, idx) not in all_processed_records:
                    row = df.iloc[idx]
                    master_record = self._create_master_record_optimized([{
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
        
        self._log_performance("Master Dataset Creation", step_start)
        logger.info(f"Created master dataset: {len(master_records)} unique tracks ({singleton_count} singletons)")
        
        return self.master_dataset
    
    def _create_master_record_optimized(self, records: List[Dict]) -> Dict:
        """Create master record with optimized processing"""
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
        
        # Vectorized audio feature aggregation
        for feature in self.critical_audio_features:
            feature_values = []
            for record in records:
                source_df = self.raw_datasets[record['source']]
                source_row = source_df.iloc[record['idx']]
                value = source_row.get(feature)
                if pd.notna(value):
                    feature_values.append(float(value))
            
            if feature_values:
                master_record[feature] = np.median(feature_values)
            else:
                master_record[feature] = np.nan
        
        # Quick popularity aggregation
        popularity_values = []
        for record in records:
            source_df = self.raw_datasets[record['source']]
            source_row = source_df.iloc[record['idx']]
            pop_score = source_row.get('popularity_score')
            if pd.notna(pop_score):
                popularity_values.append(float(pop_score))
        
        master_record['avg_popularity'] = np.mean(popularity_values) if popularity_values else np.nan
        master_record['max_popularity'] = np.max(popularity_values) if popularity_values else np.nan
        
        # Simple genre handling
        genres = []
        for record in records:
            source_df = self.raw_datasets[record['source']]
            source_row = source_df.iloc[record['idx']]
            genre = source_row.get('genre_clean')
            if pd.notna(genre) and str(genre).lower() not in ['unknown', 'nan', '']:
                genres.append(str(genre))
        
        if genres:
            genre_counts = pd.Series(genres).value_counts()
            master_record['genre'] = genre_counts.index[0]
        else:
            master_record['genre'] = 'unknown'
        
        # Platform presence
        master_record['spotify'] = any('spotify' in r['source'] for r in records)
        master_record['tiktok'] = any('tiktok' in r['source'] for r in records)
        master_record['youtube'] = any('youtube' in r['source'] for r in records)
        
        return master_record
    
    def run_fast_integration(self, fuzzy_threshold=85, skip_validation=False):
        """Run fast integration with performance monitoring"""
        self.start_time = time.time()
        logger.info("Starting fast data integration pipeline...")
        
        # Load data
        self.load_and_standardize_datasets()
        
        if not self.raw_datasets:
            logger.error("No datasets loaded. Check your file paths.")
            return None
        
        # Fuzzy matching
        self.perform_optimized_fuzzy_matching(threshold=fuzzy_threshold)
        
        # Create master dataset
        self.create_master_dataset_optimized()
        
        # Basic missing data handling (skip complex imputation for speed)
        if not skip_validation:
            self._handle_missing_data_fast()
        
        # Save results
        self._save_fast_results()
        
        total_time = time.time() - self.start_time
        logger.info(f"Fast integration complete in {total_time:.2f}s!")
        
        return {
            'master_dataset': self.master_dataset,
            'performance_stats': self.performance_stats,
            'integration_summary': {
                'total_master_records': len(self.master_dataset),
                'source_datasets': len(self.raw_datasets),
                'duplicate_groups': len(self.match_groups),
                'total_time_seconds': round(total_time, 2)
            }
        }
    
    def _handle_missing_data_fast(self):
        """Fast missing data handling without complex imputation"""
        step_start = time.time()
        
        # Simple median imputation for critical features
        for feature in self.critical_audio_features:
            if feature in self.master_dataset.columns:
                median_val = self.master_dataset[feature].median()
                if pd.notna(median_val):
                    self.master_dataset[feature].fillna(median_val, inplace=True)
        
        # Simple popularity normalization
        if 'avg_popularity' in self.master_dataset.columns:
            self.master_dataset['normalized_popularity'] = self.master_dataset['avg_popularity'].fillna(50)
        
        self._log_performance("Fast Missing Data Handling", step_start)
    
    def _save_fast_results(self):
        """Save results with performance stats"""
        # Save master dataset
        master_file = os.path.join(self.cache_dir, 'master_music_dataset_fast.csv')
        self.master_dataset.to_csv(master_file, index=False)
        
        # Save performance stats
        stats_file = os.path.join(self.cache_dir, 'performance_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.performance_stats, f, indent=2)
        
    def validate_master_dataset(self):
        """Quick validation of the master dataset"""
        master_file = os.path.join(self.cache_dir, 'master_music_dataset_fast.csv')
        
        if not os.path.exists(master_file):
            logger.error("No master dataset found. Run integration first.")
            return None
        
        logger.info("Loading master dataset for validation...")
        self.master_dataset = pd.read_csv(master_file)
        
        validation = {
            'total_records': len(self.master_dataset),
            'data_completeness': {},
            'source_distribution': {},
            'platform_coverage': {},
            'audio_feature_stats': {}
        }
        
        # Data completeness
        for col in ['track_name', 'artist_name'] + self.critical_audio_features:
            if col in self.master_dataset.columns:
                completeness = (1 - self.master_dataset[col].isna().sum() / len(self.master_dataset)) * 100
                validation['data_completeness'][col] = round(float(completeness), 2)
        
        # Source distribution
        if 'primary_source' in self.master_dataset.columns:
            source_dist = self.master_dataset['primary_source'].value_counts()
            validation['source_distribution'] = source_dist.to_dict()
        
        # Platform coverage
        for platform in ['spotify', 'tiktok', 'youtube']:
            if platform in self.master_dataset.columns:
                coverage = self.master_dataset[platform].sum()
                validation['platform_coverage'][platform] = int(coverage)
        
        # Audio feature statistics
        for feature in self.critical_audio_features:
            if feature in self.master_dataset.columns:
                values = self.master_dataset[feature].dropna()
                if len(values) > 0:
                    validation['audio_feature_stats'][feature] = {
                        'mean': round(float(values.mean()), 3),
                        'std': round(float(values.std()), 3),
                        'min': round(float(values.min()), 3),
                        'max': round(float(values.max()), 3),
                        'missing_count': int(self.master_dataset[feature].isna().sum())
                    }
        
        # Save validation results
        validation_file = os.path.join(self.cache_dir, 'validation_results_fast.json')
        with open(validation_file, 'w') as f:
            json.dump(validation, f, indent=2)
        
        logger.info(f"Validation results saved to {validation_file}")
        return validation
    
    def search_master_dataset(self, query: str, limit: int = 10, master_file: str = None):
        """Search for songs in the master dataset"""
        if master_file:
            master_dataset_path = master_file
        else:
            # Priority order: deduplicated > fixed > fast
            deduplicated_path = os.path.join(self.cache_dir, 'master_music_dataset_deduplicated.csv')
            fixed_path = os.path.join(self.cache_dir, 'master_music_dataset_fixed.csv')
            fast_path = os.path.join(self.cache_dir, 'master_music_dataset_fast.csv')
            
            if os.path.exists(deduplicated_path):
                master_dataset_path = deduplicated_path
                logger.info("Using deduplicated dataset")
            elif os.path.exists(fixed_path):
                master_dataset_path = fixed_path
                logger.info("Using fixed dataset")
            elif os.path.exists(fast_path):
                master_dataset_path = fast_path
                logger.info("Using fast dataset")
            else:
                logger.error("No master dataset found. Run integration first.")
                return []
        
        if not os.path.exists(master_dataset_path):
            logger.error(f"Master dataset not found: {master_dataset_path}")
            return []
        
        logger.info(f"Loading master dataset from {master_dataset_path}...")
        self.master_dataset = pd.read_csv(master_dataset_path)
        
        query_clean = self.clean_text_for_matching(query)
        results = []
        
        logger.info(f"Searching for: '{query}'")
        
        for idx, row in self.master_dataset.iterrows():
            max_similarity = 0
            
            # Search in track name and artist name
            for field in ['track_name', 'artist_name']:
                if field in row and pd.notna(row[field]):
                    field_value = self.clean_text_for_matching(str(row[field]))
                    similarity = fuzz.partial_ratio(query_clean, field_value)
                    max_similarity = max(max_similarity, similarity)
            
            if max_similarity > 60:  # Threshold for search results
                # Use normalized_popularity if available, otherwise avg_popularity
                popularity = row.get('normalized_popularity', row.get('avg_popularity', 0))
                if pd.isna(popularity):
                    popularity = 0
                
                results.append({
                    'master_id': row.get('master_id', ''),
                    'track_name': row.get('track_name', ''),
                    'artist_name': row.get('artist_name', ''),
                    'similarity_score': max_similarity,
                    'source_count': row.get('source_count', 1),
                    'popularity': float(popularity),
                    'genre': row.get('genre', 'unknown'),
                    'spotify': bool(row.get('spotify', False)),
                    'tiktok': bool(row.get('tiktok', False)),
                    'youtube': bool(row.get('youtube', False)),
                    'quality_score': row.get('data_quality_score', 0)
                })
        
        # Sort by similarity and popularity
        results.sort(key=lambda x: (x['similarity_score'], x['popularity']), reverse=True)
        
        return results[:limit]

# CLI with optimization options
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Optimized Master Data Integration System')
    parser.add_argument('command', choices=['integrate', 'fast', 'validate', 'search'], 
                       help='Command to execute')
    parser.add_argument('--data-dir', default='data/cleaned_music', help='Data directory path')
    parser.add_argument('--cache-dir', default='integration_cache', help='Cache directory path')
    parser.add_argument('--fuzzy-threshold', type=int, default=85, help='Fuzzy matching threshold')
    parser.add_argument('--sample-size', type=int, help='Sample size per file (for testing)')
    parser.add_argument('--single-file', help='Process only one file')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation for speed')
    parser.add_argument('--query', help='Search query (for search command)')
    parser.add_argument('--limit', type=int, default=10, help='Search result limit')
    
    args = parser.parse_args()
    
    # Initialize optimized integrator
    integrator = OptimizedMasterDataIntegrator(
        data_path=args.data_dir,
        cache_dir=args.cache_dir,
        sample_size=args.sample_size,
        single_file=args.single_file,
        progress_bar=not args.no_progress
    )
    
    if args.command in ['integrate', 'fast']:
        print("Starting optimized data integration...")
        
        if args.sample_size:
            print(f"Using sample size: {args.sample_size} rows per file")
        if args.single_file:
            print(f"Processing single file: {args.single_file}")
        
        results = integrator.run_fast_integration(
            fuzzy_threshold=args.fuzzy_threshold,
            skip_validation=args.skip_validation
        )
        
        if results:
            print("\nIntegration complete")
            summary = results['integration_summary']
            print(f"Master dataset: {summary['total_master_records']} unique tracks")
            print(f"Total time: {summary['total_time_seconds']}s")
            print(f"Duplicate groups: {summary['duplicate_groups']}")
            
            print(f"\nPerformance breakdown")
            for step, stats in results['performance_stats'].items():
                print(f"  {step}: {stats['duration_seconds']}s ({stats['memory_mb']:.1f}MB)")
    
    elif args.command == 'validate':
        print("Validating master dataset...")
        validation_results = integrator.validate_master_dataset()
        
        if validation_results:
            print("\nValidation results")
            print(f"Total records: {validation_results['total_records']:,}")
            
            print(f"\nData completeness")
            for feature, completeness in validation_results['data_completeness'].items():
                status = "OK" if completeness > 90 else "Warning" if completeness > 70 else "Error"
                print(f"  {status} {feature}: {completeness}%")
            
            print(f"\nPlatform coverage")
            for platform, count in validation_results['platform_coverage'].items():
                percentage = (count / validation_results['total_records']) * 100
                print(f"  {platform.title()}: {count:,} tracks ({percentage:.1f}%)")
            
            print(f"\nSource distribution")
            for source, count in validation_results['source_distribution'].items():
                percentage = (count / validation_results['total_records']) * 100
                print(f"  {source}: {count:,} tracks ({percentage:.1f}%)")
        else:
            print("Validation failed. Make sure you've run integration first.")
    
    elif args.command == 'search':
        if not args.query:
            print("Please provide a search query with --query")
            sys.exit(1)
        
        print(f"Searching for: '{args.query}'")
        results = integrator.search_master_dataset(args.query, limit=args.limit)
        
        if results:
            print(f"\nSearch results ({len(results)} found)")
            print("="*60)
            
            for i, result in enumerate(results, 1):
                platforms = []
                if result['spotify']: platforms.append('Spotify')
                if result['tiktok']: platforms.append('TikTok')  
                if result['youtube']: platforms.append('YouTube')
                
                print(f"{i:2d}. {result['track_name']} by {result['artist_name']}")
                print(f"     Match: {result['similarity_score']}% | Genre: {result['genre']}")
                print(f"     Sources: {result['source_count']} | Platforms: {', '.join(platforms) if platforms else 'Unknown'}")
                
                # Show popularity and quality if available
                if result['popularity'] > 0:
                    print(f"     Popularity: {result['popularity']:.1f}")
                if 'quality_score' in result and result['quality_score'] > 0:
                    print(f"     Quality: {result['quality_score']:.1f}/100")
                print()
        else:
            print(f"No results found for '{args.query}'")
            print("Try a different search term or check spelling")