# preprocessing_pipeline.py
# This script is used to preprocess the master dataset.
# It uses the audio features of the song to generate insights on the song's target demographic,
# platform recommendations, and marketing suggestions.
# It also uses the audio features of the song to generate insights on the song's trend alignment, 
# similar artists, viral potential and overall trend alignment.

import pandas as pd
import numpy as np
import re
import json
import os
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MusicDataPreprocessingPipeline:
    """
    Complete preprocessing pipeline for music data
    Creates production-ready, ML-ready datasets
    """
    
    def __init__(self, input_dataset='integration_cache/master_music_dataset_deduplicated.csv',
                 output_dir='final_datasets'):
        self.input_dataset = input_dataset
        self.output_dir = output_dir
        self.df = None
        
        # Audio features that should be in 0-1 range
        self.normalized_features = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness'
        ]
        
        # Audio features with special ranges
        self.special_ranges = {
            'tempo': (50, 250),    # BPM
            'loudness': (-60, 0),  # dB
            'key': (0, 11),        # Musical keys
            'mode': (0, 1)         # Major/Minor
        }
        
        # Platform mappings
        self.platform_mapping = {
            'spotify': 1,
            'tiktok': 2,
            'youtube': 3
        }
        
        # Genre categories for encoding
        self.genre_categories = {
            'mainstream': ['pop', 'hip hop', 'rock', 'electronic', 'dance'],
            'alternative': ['indie', 'alternative', 'experimental', 'ambient'],
            'traditional': ['country', 'folk', 'blues', 'jazz', 'classical'],
            'niche': ['metal', 'punk', 'reggae', 'world', 'soundtrack']
        }
        
        # Validation results
        self.validation_results = {}
        self.preprocessing_stats = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_dataset(self):
        """Load the master dataset"""
        if not os.path.exists(self.input_dataset):
            logger.error(f"Dataset not found: {self.input_dataset}")
            return False
        
        logger.info(f"Loading dataset: {self.input_dataset}")
        self.df = pd.read_csv(self.input_dataset)
        logger.info(f"Loaded {len(self.df):,} tracks")
        return True
    
    def validate_data_quality(self):
        """Comprehensive data quality validation"""
        logger.info("Running comprehensive data quality checks...")
        
        validation = {
            'audio_feature_validation': {},
            'data_type_validation': {},
            'consistency_checks': {},
            'outlier_detection': {},
            'recommendations': []
        }
        
        # Audio Feature Range Validation
        print("Validating audio feature ranges...")
        
        for feature in self.normalized_features:
            if feature in self.df.columns:
                values = self.df[feature].dropna()
                
                # Check 0-1 range
                out_of_range = ((values < 0) | (values > 1)).sum()
                out_of_range_pct = (out_of_range / len(values)) * 100 if len(values) > 0 else 0
                
                validation['audio_feature_validation'][feature] = {
                    'total_values': len(values),
                    'out_of_range_count': int(out_of_range),
                    'out_of_range_percentage': round(out_of_range_pct, 2),
                    'min_value': float(values.min()),
                    'max_value': float(values.max()),
                    'status': 'PASS' if out_of_range_pct < 1 else 'FAIL'
                }
                
                if out_of_range_pct > 1:
                    validation['recommendations'].append({
                        'feature': feature,
                        'issue': f'{out_of_range_pct:.1f}% values outside 0-1 range',
                        'action': 'Clip values to 0-1 range'
                    })
        
        # Special range features
        for feature, (min_val, max_val) in self.special_ranges.items():
            if feature in self.df.columns:
                values = self.df[feature].dropna()
                
                out_of_range = ((values < min_val) | (values > max_val)).sum()
                out_of_range_pct = (out_of_range / len(values)) * 100 if len(values) > 0 else 0
                
                validation['audio_feature_validation'][feature] = {
                    'expected_range': f'{min_val}-{max_val}',
                    'actual_range': f'{float(values.min()):.1f}-{float(values.max()):.1f}',
                    'out_of_range_percentage': round(out_of_range_pct, 2),
                    'status': 'PASS' if out_of_range_pct < 5 else 'FAIL'
                }
        
        # Data Type Validation
        print("Validating data types...")
        
        expected_types = {
            'track_name': 'string',
            'artist_name': 'string',
            'genre': 'string',
            'normalized_popularity': 'numeric',
            'source_count': 'integer',
            'spotify': 'boolean',
            'tiktok': 'boolean',
            'youtube': 'boolean'
        }
        
        for column, expected_type in expected_types.items():
            if column in self.df.columns:
                actual_type = str(self.df[column].dtype)
                
                if expected_type == 'string' and 'object' not in actual_type:
                    validation['data_type_validation'][column] = 'FAIL - Should be string'
                elif expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(self.df[column]):
                    validation['data_type_validation'][column] = 'FAIL - Should be numeric'
                elif expected_type == 'integer' and 'int' not in actual_type:
                    validation['data_type_validation'][column] = 'FAIL - Should be integer'
                elif expected_type == 'boolean' and 'bool' not in actual_type:
                    validation['data_type_validation'][column] = 'FAIL - Should be boolean'
                else:
                    validation['data_type_validation'][column] = 'PASS'
        
        # Consistency Checks
        print("Running consistency checks...")
        
        # Check for impossible combinations
        if 'normalized_popularity' in self.df.columns:
            negative_popularity = (self.df['normalized_popularity'] < 0).sum()
            high_popularity = (self.df['normalized_popularity'] > 100).sum()
            
            validation['consistency_checks']['popularity'] = {
                'negative_values': int(negative_popularity),
                'values_over_100': int(high_popularity),
                'status': 'PASS' if negative_popularity == 0 and high_popularity == 0 else 'FAIL'
            }
        
        # Check platform consistency
        platform_cols = ['spotify', 'tiktok', 'youtube']
        if all(col in self.df.columns for col in platform_cols):
            no_platform = (self.df[platform_cols].sum(axis=1) == 0).sum()
            validation['consistency_checks']['platform_presence'] = {
                'tracks_with_no_platform': int(no_platform),
                'percentage': round((no_platform / len(self.df)) * 100, 2),
                'status': 'PASS' if no_platform < len(self.df) * 0.01 else 'WARN'
            }
        
        # Outlier Detection
        print("Detecting outliers...")
        
        numeric_features = self.normalized_features + ['tempo', 'loudness', 'normalized_popularity']
        
        for feature in numeric_features:
            if feature in self.df.columns:
                values = self.df[feature].dropna()
                
                if len(values) > 0:
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    outliers = values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)]
                    outlier_pct = (len(outliers) / len(values)) * 100
                    
                    validation['outlier_detection'][feature] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': round(outlier_pct, 2),
                        'status': 'PASS' if outlier_pct < 5 else 'WARN' if outlier_pct < 10 else 'FAIL'
                    }
        
        self.validation_results = validation
        return validation
    
    def standardize_text_fields(self):
        """Standardize text fields (artist names, track names)"""
        logger.info("Standardizing text fields...")
        
        def clean_text(text):
            """Clean and standardize text"""
            if pd.isna(text):
                return ''
            
            text = str(text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove leading/trailing quotes
            text = text.strip('\'"')
            
            # Standardize common abbreviations
            replacements = {
                r'\bfeat\.?\s*': 'feat. ',
                r'\bft\.?\s*': 'ft. ',
                r'\b&\b': 'and',
                r'\s*\(\s*': ' (',
                r'\s*\)\s*': ') ',
                r'\s*\[\s*': ' [',
                r'\s*\]\s*': '] '
            }
            
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            # Final cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # Clean track names
        if 'track_name' in self.df.columns:
            original_tracks = self.df['track_name'].copy()
            self.df['track_name_clean'] = self.df['track_name'].apply(clean_text)
            
            # Keep original for reference
            self.df['track_name_original'] = original_tracks
            
            changed_count = (self.df['track_name_clean'] != original_tracks).sum()
            print(f"Cleaned {changed_count:,} track names")
        
        # Clean artist names
        if 'artist_name' in self.df.columns:
            original_artists = self.df['artist_name'].copy()
            self.df['artist_name_clean'] = self.df['artist_name'].apply(clean_text)
            
            # Keep original for reference
            self.df['artist_name_original'] = original_artists
            
            changed_count = (self.df['artist_name_clean'] != original_artists).sum()
            print(f"Cleaned {changed_count:,} artist names")
        
        # Clean genres
        if 'genre' in self.df.columns:
            self.df['genre_clean'] = self.df['genre'].apply(
                lambda x: clean_text(x).lower() if pd.notna(x) and x != 'unknown' else 'unknown'
            )
    
    def scale_audio_features(self):
        """Scale and normalize audio features"""
        logger.info("Scaling audio features...")
        
        scaling_stats = {}
        
        # Clip normalized features to 0-1 range
        for feature in self.normalized_features:
            if feature in self.df.columns:
                original_values = self.df[feature].copy()
                
                # Clip to 0-1 range
                self.df[feature] = np.clip(self.df[feature], 0, 1)
                
                clipped_count = (self.df[feature] != original_values).sum()
                scaling_stats[feature] = {
                    'clipped_values': int(clipped_count),
                    'final_range': f'{self.df[feature].min():.3f}-{self.df[feature].max():.3f}'
                }
                
                if clipped_count > 0:
                    print(f"Clipped {clipped_count:,} values for {feature}")
        
        # Normalize tempo to 0-1 scale
        if 'tempo' in self.df.columns:
            original_tempo = self.df['tempo'].copy()
            
            # Clip to reasonable range first
            self.df['tempo'] = np.clip(self.df['tempo'], 50, 250)
            
            # Normalize to 0-1
            self.df['tempo_normalized'] = (self.df['tempo'] - 50) / (250 - 50)
            
            scaling_stats['tempo'] = {
                'original_range': f'{original_tempo.min():.1f}-{original_tempo.max():.1f}',
                'normalized_range': f'{self.df["tempo_normalized"].min():.3f}-{self.df["tempo_normalized"].max():.3f}'
            }
        
        # Normalize loudness to 0-1 scale
        if 'loudness' in self.df.columns:
            original_loudness = self.df['loudness'].copy()
            
            # Clip to reasonable range
            self.df['loudness'] = np.clip(self.df['loudness'], -60, 0)
            
            # Normalize to 0-1 (higher values = louder)
            self.df['loudness_normalized'] = (self.df['loudness'] + 60) / 60
            
            scaling_stats['loudness'] = {
                'original_range': f'{original_loudness.min():.1f}-{original_loudness.max():.1f}',
                'normalized_range': f'{self.df["loudness_normalized"].min():.3f}-{self.df["loudness_normalized"].max():.3f}'
            }
        
        # Create standardized feature set for ML
        ml_features = self.normalized_features.copy()
        if 'tempo_normalized' in self.df.columns:
            ml_features.append('tempo_normalized')
        if 'loudness_normalized' in self.df.columns:
            ml_features.append('loudness_normalized')
        
        # StandardScaler for ML models (mean=0, std=1)
        scaler = StandardScaler()
        
        available_features = [f for f in ml_features if f in self.df.columns]
        if available_features:
            feature_matrix = self.df[available_features].fillna(0)
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Create scaled feature columns
            for i, feature in enumerate(available_features):
                self.df[f'{feature}_scaled'] = scaled_features[:, i]
        
        self.preprocessing_stats['scaling'] = scaling_stats
        print(f"Scaled {len(available_features)} audio features")
    
    def encode_categorical_features(self):
        """Encode categorical features for ML models"""
        logger.info("Encoding categorical features...")
        
        encoding_stats = {}
        
        # Genre encoding
        if 'genre_clean' in self.df.columns:
            # Create genre category
            def categorize_genre(genre):
                genre = str(genre).lower()
                for category, genres in self.genre_categories.items():
                    if any(g in genre for g in genres):
                        return category
                return 'other'
            
            self.df['genre_category'] = self.df['genre_clean'].apply(categorize_genre)
            
            # Label encoding for genres
            le_genre = LabelEncoder()
            self.df['genre_encoded'] = le_genre.fit_transform(self.df['genre_clean'].fillna('unknown'))
            
            # One-hot encoding for genre categories
            genre_dummies = pd.get_dummies(self.df['genre_category'], prefix='genre')
            self.df = pd.concat([self.df, genre_dummies], axis=1)
            
            encoding_stats['genres'] = {
                'unique_genres': int(self.df['genre_clean'].nunique()),
                'genre_categories': list(self.df['genre_category'].unique()),
                'most_common_genre': self.df['genre_clean'].mode().iloc[0] if len(self.df['genre_clean'].mode()) > 0 else 'unknown'
            }
        
        # Platform encoding
        platform_cols = ['spotify', 'tiktok', 'youtube']
        
        if all(col in self.df.columns for col in platform_cols):
            # Create platform combination features
            self.df['platform_count'] = self.df[platform_cols].sum(axis=1)
            self.df['is_multi_platform'] = (self.df['platform_count'] > 1).astype(int)
            
            # Create primary platform
            def get_primary_platform(row):
                if row['spotify']:
                    return 'spotify'
                elif row['youtube']:
                    return 'youtube'
                elif row['tiktok']:
                    return 'tiktok'
                else:
                    return 'unknown'
            
            self.df['primary_platform'] = self.df.apply(get_primary_platform, axis=1)
            
            # Encode primary platform
            le_platform = LabelEncoder()
            self.df['primary_platform_encoded'] = le_platform.fit_transform(self.df['primary_platform'])
            
            encoding_stats['platforms'] = {
                'platform_distribution': self.df['primary_platform'].value_counts().to_dict(),
                'multi_platform_tracks': int(self.df['is_multi_platform'].sum()),
                'multi_platform_percentage': round((self.df['is_multi_platform'].sum() / len(self.df)) * 100, 2)
            }
        
        # Popularity tiers
        if 'normalized_popularity' in self.df.columns:
            def popularity_tier(score):
                if pd.isna(score):
                    return 'unknown'
                elif score >= 80:
                    return 'viral'
                elif score >= 60:
                    return 'popular'
                elif score >= 40:
                    return 'moderate'
                elif score >= 20:
                    return 'niche'
                else:
                    return 'underground'
            
            self.df['popularity_tier'] = self.df['normalized_popularity'].apply(popularity_tier)
            
            # One-hot encode popularity tiers
            popularity_dummies = pd.get_dummies(self.df['popularity_tier'], prefix='pop')
            self.df = pd.concat([self.df, popularity_dummies], axis=1)
            
            encoding_stats['popularity'] = {
                'tier_distribution': self.df['popularity_tier'].value_counts().to_dict()
            }
        
        # Quality tiers
        if 'data_quality_score' in self.df.columns:
            def quality_tier(score):
                if pd.isna(score):
                    return 'unknown'
                elif score >= 90:
                    return 'excellent'
                elif score >= 80:
                    return 'good'
                elif score >= 70:
                    return 'fair'
                else:
                    return 'poor'
            
            self.df['quality_tier'] = self.df['data_quality_score'].apply(quality_tier)
            
            # One-hot encode quality tiers
            quality_dummies = pd.get_dummies(self.df['quality_tier'], prefix='quality')
            self.df = pd.concat([self.df, quality_dummies], axis=1)
            
            encoding_stats['quality'] = {
                'tier_distribution': self.df['quality_tier'].value_counts().to_dict()
            }
        
        self.preprocessing_stats['encoding'] = encoding_stats
        print(f"Encoded categorical features")
    
    def handle_missing_values(self):
        """Advanced missing value imputation"""
        logger.info("Handling missing values...")
        
        imputation_stats = {}
        
        # Audio features - use median imputation by genre
        audio_features = self.normalized_features + ['tempo_normalized', 'loudness_normalized']
        available_audio = [f for f in audio_features if f in self.df.columns]
        
        for feature in available_audio:
            missing_count = self.df[feature].isna().sum()
            if missing_count > 0:
                # Impute by genre median, fallback to overall median
                def impute_by_genre(row):
                    if pd.notna(row[feature]):
                        return row[feature]
                    
                    genre = row.get('genre_clean', 'unknown')
                    genre_median = self.df[self.df['genre_clean'] == genre][feature].median()
                    
                    if pd.notna(genre_median):
                        return genre_median
                    else:
                        return self.df[feature].median()
                
                self.df[feature] = self.df.apply(impute_by_genre, axis=1)
                
                imputation_stats[feature] = {
                    'missing_count': int(missing_count),
                    'imputation_method': 'genre_median_fallback_overall'
                }
        
        # Popularity - use platform and genre averages
        if 'normalized_popularity' in self.df.columns:
            missing_pop = self.df['normalized_popularity'].isna().sum()
            if missing_pop > 0:
                def impute_popularity(row):
                    if pd.notna(row['normalized_popularity']):
                        return row['normalized_popularity']
                    
                    # Try genre average first
                    genre = row.get('genre_clean', 'unknown')
                    platform = row.get('primary_platform', 'unknown')
                    
                    genre_avg = self.df[self.df['genre_clean'] == genre]['normalized_popularity'].mean()
                    if pd.notna(genre_avg):
                        return genre_avg
                    
                    # Fallback to overall average
                    return self.df['normalized_popularity'].mean()
                
                self.df['normalized_popularity'] = self.df.apply(impute_popularity, axis=1)
                
                imputation_stats['normalized_popularity'] = {
                    'missing_count': int(missing_pop),
                    'imputation_method': 'genre_platform_average'
                }
        
        # Source count - fill with 1 (minimum possible)
        if 'source_count' in self.df.columns:
            missing_sources = self.df['source_count'].isna().sum()
            if missing_sources > 0:
                self.df['source_count'] = self.df['source_count'].fillna(1)
                imputation_stats['source_count'] = {
                    'missing_count': int(missing_sources),
                    'imputation_method': 'constant_fill_1'
                }
        
        # Platform booleans - fill with False
        platform_cols = ['spotify', 'tiktok', 'youtube']
        for col in platform_cols:
            if col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                if missing_count > 0:
                    self.df[col] = self.df[col].fillna(False)
                    imputation_stats[col] = {
                        'missing_count': int(missing_count),
                        'imputation_method': 'constant_fill_false'
                    }
        
        self.preprocessing_stats['imputation'] = imputation_stats
        print(f"Handled missing values for {len(imputation_stats)} features")
    
    def create_output_datasets(self):
        """Create the final output datasets"""
        logger.info("Creating output datasets...")
        
        datasets_created = []
        
        # Master Music Data - Complete dataset
        print("Creating master_music_data.csv...")
        
        master_columns = [
            # Identifiers
            'track_name_clean', 'artist_name_clean', 'genre_clean',
            
            # Original audio features
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness',
            'tempo', 'loudness',
            
            # Normalized features
            'tempo_normalized', 'loudness_normalized',
            
            # Platform and popularity
            'normalized_popularity', 'popularity_tier',
            'spotify', 'tiktok', 'youtube', 'primary_platform',
            'platform_count', 'is_multi_platform',
            
            # Quality and metadata
            'data_quality_score', 'quality_tier',
            'source_count',
            
            # Encoded features
            'genre_encoded', 'genre_category',
            'primary_platform_encoded'
        ]
        
        # Add scaled features if they exist
        scaled_features = [col for col in self.df.columns if col.endswith('_scaled')]
        master_columns.extend(scaled_features)
        
        # Add one-hot encoded columns
        encoded_cols = [col for col in self.df.columns if any(col.startswith(prefix) for prefix in ['genre_', 'pop_', 'quality_'])]
        master_columns.extend(encoded_cols)
        
        # Filter to existing columns
        available_master_cols = [col for col in master_columns if col in self.df.columns]
        
        master_df = self.df[available_master_cols].copy()
        master_file = os.path.join(self.output_dir, 'master_music_data.csv')
        master_df.to_csv(master_file, index=False)
        
        datasets_created.append({
            'name': 'master_music_data.csv',
            'rows': len(master_df),
            'columns': len(master_df.columns),
            'description': 'Complete standardized music dataset'
        })
        
        # Platform Performance - Cross-platform metrics
        print("Creating platform_performance.csv...")
        
        platform_performance = []
        
        for _, track in self.df.iterrows():
            base_record = {
                'track_name': track.get('track_name_clean', ''),
                'artist_name': track.get('artist_name_clean', ''),
                'genre': track.get('genre_clean', '')
            }
            
            platforms = ['spotify', 'tiktok', 'youtube']
            for platform in platforms:
                if track.get(platform, False):
                    platform_record = base_record.copy()
                    platform_record.update({
                        'platform': platform,
                        'popularity_score': track.get('normalized_popularity', 0),
                        'audio_appeal': np.mean([
                            track.get('danceability', 0),
                            track.get('energy', 0),
                            track.get('valence', 0)
                        ]),
                        'production_quality': track.get('data_quality_score', 0),
                        'is_primary_platform': (track.get('primary_platform') == platform)
                    })
                    platform_performance.append(platform_record)
        
        if platform_performance:
            platform_df = pd.DataFrame(platform_performance)
            platform_file = os.path.join(self.output_dir, 'platform_performance.csv')
            platform_df.to_csv(platform_file, index=False)
            
            datasets_created.append({
                'name': 'platform_performance.csv',
                'rows': len(platform_df),
                'columns': len(platform_df.columns),
                'description': 'Cross-platform performance metrics'
            })
        
        # Demographic Preferences - User behavior insights
        print("Creating demographic_preferences.csv...")
        
        demographic_preferences = []
        
        # Create demographic preference profiles
        demographic_profiles = {
            'Gen Z (16-24)': {
                'preferred_features': {'danceability': 0.8, 'energy': 0.75, 'valence': 0.7},
                'preferred_genres': ['pop', 'hip hop', 'electronic'],
                'preferred_platforms': ['tiktok', 'spotify']
            },
            'Millennials (25-40)': {
                'preferred_features': {'danceability': 0.65, 'energy': 0.6, 'valence': 0.6},
                'preferred_genres': ['pop', 'rock', 'hip hop'],
                'preferred_platforms': ['spotify', 'youtube']
            },
            'Gen X (41-56)': {
                'preferred_features': {'danceability': 0.5, 'energy': 0.55, 'valence': 0.5},
                'preferred_genres': ['rock', 'pop', 'country'],
                'preferred_platforms': ['spotify', 'youtube']
            },
            'Boomers (57+)': {
                'preferred_features': {'danceability': 0.4, 'energy': 0.4, 'valence': 0.45},
                'preferred_genres': ['rock', 'country', 'jazz'],
                'preferred_platforms': ['spotify', 'youtube']
            }
        }
        
        for demo, profile in demographic_profiles.items():
            # Find tracks that match this demographic
            matching_tracks = []
            
            for _, track in self.df.iterrows():
                match_score = 0
                
                # Check feature alignment
                for feature, ideal_value in profile['preferred_features'].items():
                    if feature in track and pd.notna(track[feature]):
                        alignment = 1 - abs(track[feature] - ideal_value)
                        match_score += alignment
                
                # Bonus for preferred genres
                if track.get('genre_clean', '').lower() in profile['preferred_genres']:
                    match_score += 0.5
                
                # Average match score
                avg_match = match_score / (len(profile['preferred_features']) + 0.5)
                
                if avg_match > 0.7:  # Good match threshold
                    matching_tracks.append({
                        'demographic': demo,
                        'track_name': track['track_name_clean'],
                        'artist_name': track['artist_name_clean'],
                        'genre': track['genre_clean'],
                        'match_score': round(avg_match * 100, 2),
                        'popularity': track.get('normalized_popularity', 0),
                        'preferred_platform': profile['preferred_platforms'][0]
                    })
            
            demographic_preferences.extend(matching_tracks)
        
        if demographic_preferences:
            demo_df = pd.DataFrame(demographic_preferences)
            demo_file = os.path.join(self.output_dir, 'demographic_preferences.csv')
            demo_df.to_csv(demo_file, index=False)
            
            datasets_created.append({
                'name': 'demographic_preferences.csv',
                'rows': len(demo_df),
                'columns': len(demo_df.columns),
                'description': 'Demographic preference analysis'
            })
        
        # Trend Analysis - Time-based and feature trends
        print("Creating trend_analysis.csv...")
        
        trend_data = []
        
        # Genre trend analysis
        for genre in self.df['genre_clean'].unique():
            if genre == 'unknown':
                continue
            
            genre_tracks = self.df[self.df['genre_clean'] == genre]
            if len(genre_tracks) < 5:
                continue
            
            # Calculate trend metrics
            trend_record = {
                'category': 'genre',
                'item': genre,
                'track_count': len(genre_tracks),
                'avg_popularity': genre_tracks['normalized_popularity'].mean(),
                'avg_quality': genre_tracks['data_quality_score'].mean(),
                'platform_diversity': genre_tracks['platform_count'].mean(),
                
                # Audio feature averages
                'avg_danceability': genre_tracks['danceability'].mean(),
                'avg_energy': genre_tracks['energy'].mean(),
                'avg_valence': genre_tracks['valence'].mean(),
                'avg_acousticness': genre_tracks['acousticness'].mean(),
                
                # Trend indicators
                'growth_potential': self._calculate_growth_potential(genre_tracks),
                'market_saturation': len(genre_tracks) / len(self.df) * 100
            }
            
            trend_data.append(trend_record)
        
        # Platform trend analysis
        platforms = ['spotify', 'tiktok', 'youtube']
        for platform in platforms:
            platform_tracks = self.df[self.df[platform] == True]
            
            if len(platform_tracks) > 0:
                trend_record = {
                    'category': 'platform',
                    'item': platform,
                    'track_count': len(platform_tracks),
                    'avg_popularity': platform_tracks['normalized_popularity'].mean(),
                    'avg_quality': platform_tracks['data_quality_score'].mean(),
                    'platform_diversity': platform_tracks['platform_count'].mean(),
                    
                    # Platform-specific audio preferences
                    'avg_danceability': platform_tracks['danceability'].mean(),
                    'avg_energy': platform_tracks['energy'].mean(),
                    'avg_valence': platform_tracks['valence'].mean(),
                    'avg_acousticness': platform_tracks['acousticness'].mean(),
                    
                    'growth_potential': self._calculate_growth_potential(platform_tracks),
                    'market_saturation': len(platform_tracks) / len(self.df) * 100
                }
                
                trend_data.append(trend_record)
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            trend_file = os.path.join(self.output_dir, 'trend_analysis.csv')
            trend_df.to_csv(trend_file, index=False)
            
            datasets_created.append({
                'name': 'trend_analysis.csv',
                'rows': len(trend_df),
                'columns': len(trend_df.columns),
                'description': 'Genre and platform trend analysis'
            })
        
    def _calculate_growth_potential(self, tracks_subset):
        """Calculate growth potential score for a subset of tracks"""
        if len(tracks_subset) == 0:
            return 0
        
        # Factors: quality vs popularity gap, platform diversity
        avg_quality = tracks_subset['data_quality_score'].mean()
        avg_popularity = tracks_subset['normalized_popularity'].mean()
        
        # Gap between quality and popularity (undervalued = high potential)
        quality_gap = max(0, avg_quality - avg_popularity) / 100
        
        # Platform diversity bonus
        platform_diversity = tracks_subset['platform_count'].mean() / 3
        
        # Combined growth potential (0-100 scale)
        growth_potential = (quality_gap * 0.7 + platform_diversity * 0.3) * 100
        
        return round(growth_potential, 2)
        """Create the final output datasets"""
        logger.info("Creating output datasets...")
        
        datasets_created = []
        
        # Master Music Data - Complete dataset
        print("Creating master_music_data.csv...")
        
        master_columns = [
            # Identifiers
            'track_name_clean', 'artist_name_clean', 'genre_clean',
            
            # Original audio features
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness',
            'tempo', 'loudness',
            
            # Normalized features
            'tempo_normalized', 'loudness_normalized',
            
            # Scaled features (for ML)
            'danceability_scaled', 'energy_scaled', 'valence_scaled',
            'acousticness_scaled', 'instrumentalness_scaled',
            'liveness_scaled', 'speechiness_scaled',
            'tempo_normalized_scaled', 'loudness_normalized_scaled',
            
            # Platform and popularity
            'normalized_popularity', 'popularity_tier',
            'spotify', 'tiktok', 'youtube', 'primary_platform',
            'platform_count', 'is_multi_platform',
            
            # Quality and metadata
            'data_quality_score', 'quality_tier',
            'source_count', 'ar_score',
            
            # Encoded features
            'genre_encoded', 'genre_category',
            'primary_platform_encoded'
        ]
        
        # Add one-hot encoded columns
        encoded_cols = [col for col in self.df.columns if any(col.startswith(prefix) for prefix in ['genre_', 'pop_', 'quality_'])]
        master_columns.extend(encoded_cols)
        
        # Filter to existing columns
        available_master_cols = [col for col in master_columns if col in self.df.columns]
        
        master_df = self.df[available_master_cols].copy()
        master_file = os.path.join(self.output_dir, 'master_music_data.csv')
        master_df.to_csv(master_file, index=False)
        
        datasets_created.append({
            'name': 'master_music_data.csv',
            'rows': len(master_df),
            'columns': len(master_df.columns),
            'description': 'Complete standardized music dataset'
        })
        
        # Platform Performance - Cross-platform metrics
        print("Creating platform_performance.csv...")
        
        platform_performance = []
        
        for _, track in self.df.iterrows():
            base_record = {
                'track_name': track['track_name_clean'],
                'artist_name': track['artist_name_clean'],
                'genre': track['genre_clean']
            }
            
            platforms = ['spotify', 'tiktok', 'youtube']
            for platform in platforms:
                if track.get(platform, False):
                    platform_record = base_record.copy()
                    platform_record.update({
                        'platform': platform,
                        'popularity_score': track.get('normalized_popularity', 0),
                        'audio_appeal': np.mean([
                            track.get('danceability', 0),
                            track.get('energy', 0),
                            track.get('valence', 0)
                        ]),
                        'production_quality': track.get('data_quality_score', 0),
                        'commercial_potential': track.get('commercial_potential', 0) if 'commercial_potential' in track else 0,
                        'is_primary_platform': (track.get('primary_platform') == platform)
                    })
                    platform_performance.append(platform_record)
        
        if platform_performance:
            platform_df = pd.DataFrame(platform_performance)
            platform_file = os.path.join(self.output_dir, 'platform_performance.csv')
            platform_df.to_csv(platform_file, index=False)
            
            datasets_created.append({
                'name': 'platform_performance.csv',
                'rows': len(platform_df),
                'columns': len(platform_df.columns),
                'description': 'Cross-platform performance metrics'
            })
        
        # Demographic Preferences - User behavior insights
        print("Creating demographic_preferences.csv...")
        
        demographic_preferences = []
        
        # Create demographic preference profiles
        demographics = ['Gen Z (16-24)', 'Millennials (25-40)', 'Gen X (41-56)', 'Boomers (57+)']
        
        demographic_profiles = {
            'Gen Z (16-24)': {
                'preferred_features': {'danceability': 0.8, 'energy': 0.75, 'valence': 0.7},
                'preferred_genres': ['pop', 'hip hop', 'electronic'],
                'preferred_platforms': ['tiktok', 'spotify']
            },
            'Millennials (25-40)': {
                'preferred_features': {'danceability': 0.65, 'energy': 0.6, 'valence': 0.6},
                'preferred_genres': ['pop', 'rock', 'hip hop'],
                'preferred_platforms': ['spotify', 'youtube']
            },
            'Gen X (41-56)': {
                'preferred_features': {'danceability': 0.5, 'energy': 0.55, 'valence': 0.5},
                'preferred_genres': ['rock', 'pop', 'country'],
                'preferred_platforms': ['spotify', 'youtube']
            },
            'Boomers (57+)': {
                'preferred_features': {'danceability': 0.4, 'energy': 0.4, 'valence': 0.45},
                'preferred_genres': ['rock', 'country', 'jazz'],
                'preferred_platforms': ['spotify', 'youtube']
            }
        }
        
        for demo, profile in demographic_profiles.items():
            # Find tracks that match this demographic
            matching_tracks = []
            
            for _, track in self.df.iterrows():
                match_score = 0
                
                # Check feature alignment
                for feature, ideal_value in profile['preferred_features'].items():
                    if feature in track and pd.notna(track[feature]):
                        alignment = 1 - abs(track[feature] - ideal_value)
                        match_score += alignment
                
                # Bonus for preferred genres
                if track.get('genre_clean', '').lower() in profile['preferred_genres']:
                    match_score += 0.5
                
                # Average match score
                avg_match = match_score / (len(profile['preferred_features']) + 0.5)
                
                if avg_match > 0.7:  # Good match threshold
                    matching_tracks.append({
                        'demographic': demo,
                        'track_name': track['track_name_clean'],
                        'artist_name': track['artist_name_clean'],
                        'genre': track['genre_clean'],
                        'match_score': round(avg_match * 100, 2),
                        'popularity': track.get('normalized_popularity', 0),
                        'preferred_platform': profile['preferred_platforms'][0]
                    })
            
            demographic_preferences.extend(matching_tracks)