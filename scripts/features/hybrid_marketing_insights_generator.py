import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pickle
import psutil
import gc
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridMarketingInsightsGenerator:
    def __init__(
        self,
        data_path='data/cleaned_music',
        memory_mode='auto',
        cache_dir='data/marketing_cache'
    ):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.memory_mode = memory_mode
        self.audio_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo'
        ]
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.combined_df = None
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Auto-detect memory mode
        if memory_mode == 'auto':
            self.memory_mode = self._detect_memory_mode()
    
    def _detect_memory_mode(self):
        """Auto-detect optimal memory mode based on available RAM"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            logger.info(f"Available memory: {available_gb:.1f} GB")
            
            if available_gb >= 8:
                return 'full'
            elif available_gb >= 4:
                return 'medium'
            else:
                return 'optimized'
        except Exception:
            logger.warning(
                "Could not detect memory, defaulting to optimized mode"
            )
            return 'optimized'
    
    def _get_memory_limits(self):
        """Get data limits based on memory mode"""
        limits = {
            'full': {
                'max_rows': None,
                'batch_size': 5000,
                'clustering_algo': 'kmeans'
            },
            'medium': {
                'max_rows': 50000,
                'batch_size': 2000,
                'clustering_algo': 'kmeans'
            },
            'optimized': {
                'max_rows': 10000,
                'batch_size': 1000,
                'clustering_algo': 'minibatch'
            }
        }
        return limits.get(self.memory_mode, limits['optimized'])
    
    def _stratified_sample(self, df, max_rows):
        """Perform stratified sampling to maintain data distribution"""
        if len(df) <= max_rows:
            return df
        
        logger.info(
            f"Stratified sampling {max_rows} rows from {len(df)} total rows"
        )
        
        # Try to maintain proportions across platforms and genres
        sample_dfs = []
        
        if 'source_platform' in df.columns:
            platforms = df['source_platform'].unique()
            rows_per_platform = max_rows // len(platforms)
            
            for platform in platforms:
                platform_df = df[df['source_platform'] == platform]
                if len(platform_df) > rows_per_platform:
                    sampled = platform_df.sample(n=rows_per_platform, random_state=42)
                else:
                    sampled = platform_df
                sample_dfs.append(sampled)
            
            result = pd.concat(sample_dfs, ignore_index=True)
            
            # If we still have too many rows, random sample the remainder
            if len(result) > max_rows:
                result = result.sample(n=max_rows, random_state=42)
                
            return result
        else:
            # Fallback to random sampling
            return df.sample(n=max_rows, random_state=42)
    
    def load_data(self):
        """Load data based on memory mode"""
        logger.info(
            f"Loading data in {self.memory_mode} mode..."
        )
        limits = self._get_memory_limits()
        
        # Check if cached data exists
        cache_file = os.path.join(self.cache_dir, f'combined_data_{self.memory_mode}.pkl')
        if os.path.exists(cache_file):
            logger.info("Loading cached combined dataset...")
            with open(cache_file, 'rb') as f:
                self.combined_df = pickle.load(f)
            logger.info(
                f"Loaded cached data: {len(self.combined_df)} rows"
            )
            return
        
        # Load datasets as before but with memory optimization
        datasets = []
        
        # Load main datasets
        dataset_configs = [
            (
                'spotify_tracks_clean.csv',
                'spotify',
                'popularity'
            ),
            (
                'music_info_clean.csv',
                'music_info',
                None
            )
        ]
        
        for filename, platform, pop_col in dataset_configs:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df = self._optimize_dtypes(df)
                    
                    # Apply memory limits
                    if limits['max_rows'] and len(df) > limits['max_rows'] // len(dataset_configs):
                        df = self._stratified_sample(df, limits['max_rows'] // len(dataset_configs))
                    
                    df['source_platform'] = platform
                    if pop_col and pop_col in df.columns:
                        df['platform_popularity'] = df[pop_col]
                    else:
                        df['platform_popularity'] = 50
                    
                    datasets.append(df)
                    logger.info(
                        f"Loaded {filename}: {len(df)} rows"
                    )
                    
                    # Force garbage collection
                    gc.collect()
                    
                except Exception as e:
                    logger.warning(
                        f"Could not load {filename}: {e}"
                    )
        
        # Load normalized datasets
        normalized_configs = [
            (
                'spotify_ds_normalized.csv',
                'spotify_ds',
                'popularity'
            ),
            (
                'tiktok_2021_normalized.csv',
                'tiktok_2021',
                'track_pop'
            ),
            (
                'tiktok_2022_normalized.csv',
                'tiktok_2022',
                'track_pop'
            ),
            (
                'spotify_yt_normalized.csv',
                'youtube',
                'views'
            )
        ]
        
        for filename, platform, pop_col in normalized_configs:
            filepath = os.path.join('data/cleaned_normalized', filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df = self._optimize_dtypes(df)
                    
                    if limits['max_rows'] and len(df) > limits['max_rows'] // len(normalized_configs):
                        df = self._stratified_sample(df, limits['max_rows'] // len(normalized_configs))
                    
                    df['source_platform'] = platform
                    if pop_col and pop_col in df.columns:
                        df['platform_popularity'] = pd.to_numeric(df[pop_col], errors='coerce')
                        df['platform_popularity'] = df['platform_popularity'].fillna(0)
                    else:
                        df['platform_popularity'] = 50
                    
                    datasets.append(df)
                    logger.info(
                        f"Loaded {filename}: {len(df)} rows"
                    )
                    gc.collect()
                    
                except Exception as e:
                    logger.warning(
                        f"Could not load {filename}: {e}"
                    )
        
        if not datasets:
            logger.error("No datasets loaded!")
            self.combined_df = pd.DataFrame()
            return
        
        # Combine datasets
        self.combined_df = self._create_combined_dataset(datasets)
        
        # Cache the combined dataset
        logger.info(
            "Caching combined dataset..."
        )
        with open(cache_file, 'wb') as f:
            pickle.dump(self.combined_df, f)
        
        logger.info(
            f"Data loaded successfully: {len(self.combined_df)} total rows"
        )
    
    def _optimize_dtypes(self, df):
        """Optimize dataframe memory usage"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        return df
    
    def _create_combined_dataset(self, datasets):
        """Create combined dataset with standardized columns"""
        standardized_datasets = []
        
        for df in datasets:
            # Standardize column names
            std_df = df.copy()
            
            # Map common column variations
            column_mapping = {
                'track': 'track_name',
                'title': 'track_name',
                'name': 'track_name',
                'artist': 'artists',
                'artist_name': 'artists',
                'genre': 'track_genre',
                'popularity': 'platform_popularity',
                'track_pop': 'platform_popularity',
                'views': 'platform_popularity'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in std_df.columns and new_col not in std_df.columns:
                    std_df[new_col] = std_df[old_col]
            
            # Ensure required columns exist
            required_cols = [
                'track_name',
                'artists',
                'track_genre',
                'platform_popularity',
                'source_platform'
            ]
            for col in required_cols:
                if col not in std_df.columns:
                    if col == 'track_genre':
                        std_df[col] = 'unknown'
                    elif col == 'platform_popularity':
                        std_df[col] = 50
                    else:
                        std_df[col] = 'unknown'
            
            # Select final columns
            audio_cols = [col for col in self.audio_features if col in std_df.columns]
            final_cols = required_cols + audio_cols
            
            standardized_datasets.append(std_df[final_cols])
        
        # Combine all datasets
        combined = pd.concat(standardized_datasets, ignore_index=True, sort=False)
        
        # Clean and fill missing values
        for feature in self.audio_features:
            if feature in combined.columns:
                combined[feature] = pd.to_numeric(combined[feature], errors='coerce')
                combined[feature] = combined[feature].fillna(combined[feature].median())
        
        combined['track_name'] = combined['track_name'].fillna('Unknown Track')
        combined['artists'] = combined['artists'].fillna('Unknown Artist')
        combined['track_genre'] = combined['track_genre'].fillna('unknown')
        combined['platform_popularity'] = pd.to_numeric(combined['platform_popularity'], errors='coerce')
        combined['platform_popularity'] = combined['platform_popularity'].fillna(0)
        
        return combined
    
    def precompute_insights(self, batch_size=None):
        """Precompute all marketing insights and cache results"""
        logger.info("Starting precomputation of marketing insights...")
        
        if self.combined_df is None or self.combined_df.empty:
            logger.error("No data loaded for precomputation")
            return None
        
        limits = self._get_memory_limits()
        batch_size = batch_size or limits['batch_size']
        
        # Initialize results storage
        all_results = {
            'demographic_scores': [],
            'platform_scores': [],
            'trend_scores': [],
            'similar_artists': []
        }
        
        # Process in batches
        total_batches = (len(self.combined_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.combined_df))
            
            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches} (rows {start_idx}-{end_idx})"
            )
            
            batch_df = self.combined_df.iloc[start_idx:end_idx].copy()
            
            # Calculate insights for this batch
            batch_results = self._calculate_batch_insights(batch_df)
            
            # Append to all results
            for key, values in batch_results.items():
                all_results[key].extend(values)
            
            # Save intermediate results
            self._save_batch_cache(batch_results, batch_idx)
            
            # Force garbage collection
            gc.collect()
        
        # Combine all results
        final_results = {}
        for key, values in all_results.items():
            final_results[key] = pd.DataFrame(values)
        
        # Save final results
        self._save_final_cache(final_results)
        
        logger.info(
            f"Precomputation complete!"
        )
        return final_results
    
    def _calculate_batch_insights(self, batch_df):
        """Calculate insights for a single batch"""
        batch_results = {
            'demographic_scores': [],
            'platform_scores': [],
            'trend_scores': [],
            'similar_artists': []
        }
        
        # Age group preferences
        age_preferences = {
            'Gen Z (16-24)': {'danceability': 0.8, 'energy': 0.75, 'valence': 0.7, 'speechiness': 0.3, 'acousticness': 0.2, 'tempo': 120},
            'Millennials (25-40)': {'danceability': 0.65, 'energy': 0.6, 'valence': 0.6, 'speechiness': 0.15, 'acousticness': 0.4, 'tempo': 110},
            'Gen X (41-56)': {'danceability': 0.5, 'energy': 0.55, 'valence': 0.5, 'speechiness': 0.1, 'acousticness': 0.6, 'tempo': 100},
            'Boomers (57+)': {'danceability': 0.4, 'energy': 0.4, 'valence': 0.45, 'speechiness': 0.05, 'acousticness': 0.7, 'tempo': 90}
        }
        
        # Platform preferences
        platform_preferences = {
            'TikTok': {'danceability': 0.8, 'energy': 0.85, 'valence': 0.75, 'speechiness': 0.4, 'tempo': 130, 'weight': 1.2},
            'Instagram': {'danceability': 0.7, 'energy': 0.7, 'valence': 0.65, 'acousticness': 0.3, 'tempo': 115, 'weight': 1.0},
            'Spotify': {'danceability': 0.6, 'energy': 0.6, 'valence': 0.55, 'acousticness': 0.5, 'instrumentalness': 0.2, 'weight': 1.1},
            'YouTube': {'danceability': 0.55, 'energy': 0.65, 'valence': 0.6, 'acousticness': 0.4, 'liveness': 0.3, 'weight': 0.9}
        }
        
        # Trend characteristics
        trend_characteristics = {
            'viral_potential': {'danceability': 0.75, 'energy': 0.8, 'valence': 0.7, 'speechiness': 0.25, 'tempo': 125, 'weight': 1.3},
            'playlist_friendly': {'danceability': 0.65, 'energy': 0.6, 'valence': 0.6, 'acousticness': 0.3, 'instrumentalness': 0.1, 'weight': 1.1},
            'social_media_ready': {'danceability': 0.8, 'energy': 0.85, 'speechiness': 0.35, 'tempo': 130, 'valence': 0.75, 'weight': 1.2}
        }
        
        for _, song in batch_df.iterrows():
            song_id = f"{song.get('track_name', '')}_{song.get('artists', '')}"
            
            # Calculate demographic scores
            demo_scores = {'song_id': song_id, 'track_name': song.get('track_name', ''), 'artists': song.get('artists', '')}
            for age_group, preferences in age_preferences.items():
                score = self._calculate_preference_score(song, preferences)
                demo_scores[f'{age_group}_score'] = score
            batch_results['demographic_scores'].append(demo_scores)
            
            # Calculate platform scores
            platform_scores = {'song_id': song_id, 'track_name': song.get('track_name', ''), 'artists': song.get('artists', '')}
            for platform, preferences in platform_preferences.items():
                score = self._calculate_preference_score(song, preferences)
                platform_scores[f'{platform}_score'] = score
            batch_results['platform_scores'].append(platform_scores)
            
            # Calculate trend scores
            trend_scores = {'song_id': song_id, 'track_name': song.get('track_name', ''), 'artists': song.get('artists', '')}
            overall_trend = 0
            trend_count = 0
            for trend_type, characteristics in trend_characteristics.items():
                score = self._calculate_preference_score(song, characteristics)
                trend_scores[f'{trend_type}_score'] = score
                overall_trend += score
                trend_count += 1
            trend_scores['overall_trend_score'] = overall_trend / trend_count if trend_count > 0 else 0
            batch_results['trend_scores'].append(trend_scores)
        
        return batch_results
    
    def _calculate_preference_score(self, song, preferences):
        """Calculate how well a song matches given preferences"""
        score = 0
        feature_count = 0
        
        for feature, ideal_value in preferences.items():
            if feature in ['weight']:
                continue
            
            if feature in song and pd.notna(song[feature]):
                if feature == 'tempo':
                    tempo_diff = abs(song[feature] - ideal_value)
                    feature_score = max(0, 1 - (tempo_diff / 100))
                else:
                    feature_score = 1 - abs(song[feature] - ideal_value)
                
                score += max(0, feature_score)
                feature_count += 1
        
        if feature_count == 0:
            return 0
        
        # Apply weight if present
        weight = preferences.get('weight', 1.0)
        final_score = (score / feature_count) * weight * 100
        return round(final_score, 2)
    
    def _save_batch_cache(self, batch_results, batch_idx):
        """Save batch results to cache"""
        cache_file = os.path.join(self.cache_dir, f'batch_{batch_idx}_results.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(batch_results, f)
    
    def _save_final_cache(self, final_results):
        """Save final combined results to cache"""
        cache_file = os.path.join(self.cache_dir, 'final_insights.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(final_results, f)
        
        # Also save as CSV for easy access
        for key, df in final_results.items():
            csv_file = os.path.join(self.cache_dir, f'{key}.csv')
            df.to_csv(csv_file, index=False)
            logger.info(
                f"Saved {key}.csv with {len(df)} rows"
            )
    
    def load_precomputed_insights(self):
        """Load precomputed insights from cache"""
        cache_file = os.path.join(self.cache_dir, 'final_insights.pkl')
        
        if os.path.exists(cache_file):
            logger.info(
                f"Loaded precomputed insights from cache..."
            )
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            logger.warning(
                "No precomputed insights found. Run precompute_insights() first."
            )
            return None
    
    def get_song_recommendations(self, song_features, top_k=5):
        """Get marketing recommendations for a new song using precomputed data"""
        insights = self.load_precomputed_insights()
        
        if insights is None:
            logger.error("No precomputed insights available")
            return None
        
        # Calculate scores for the new song
        new_song_insights = self._calculate_new_song_insights(song_features)
        
        # Find similar songs from precomputed data
        similar_songs = self._find_similar_songs(song_features, insights, top_k)
        
        return {
            'song_insights': new_song_insights,
            'similar_songs': similar_songs,
            'recommendations': self._generate_recommendations(new_song_insights)
        }
    
    def _calculate_new_song_insights(self, song_features):
        """Calculate insights for a new song"""
        # Use the same logic as batch processing but for a single song
        song_data = pd.Series(song_features)
        
        age_preferences = {
            'Gen Z (16-24)': {'danceability': 0.8, 'energy': 0.75, 'valence': 0.7, 'speechiness': 0.3, 'acousticness': 0.2, 'tempo': 120},
            'Millennials (25-40)': {'danceability': 0.65, 'energy': 0.6, 'valence': 0.6, 'speechiness': 0.15, 'acousticness': 0.4, 'tempo': 110},
            'Gen X (41-56)': {'danceability': 0.5, 'energy': 0.55, 'valence': 0.5, 'speechiness': 0.1, 'acousticness': 0.6, 'tempo': 100},
            'Boomers (57+)': {'danceability': 0.4, 'energy': 0.4, 'valence': 0.45, 'speechiness': 0.05, 'acousticness': 0.7, 'tempo': 90}
        }
        
        platform_preferences = {
            'TikTok': {'danceability': 0.8, 'energy': 0.85, 'valence': 0.75, 'speechiness': 0.4, 'tempo': 130, 'weight': 1.2},
            'Instagram': {'danceability': 0.7, 'energy': 0.7, 'valence': 0.65, 'acousticness': 0.3, 'tempo': 115, 'weight': 1.0},
            'Spotify': {'danceability': 0.6, 'energy': 0.6, 'valence': 0.55, 'acousticness': 0.5, 'instrumentalness': 0.2, 'weight': 1.1},
            'YouTube': {'danceability': 0.55, 'energy': 0.65, 'valence': 0.6, 'acousticness': 0.4, 'liveness': 0.3, 'weight': 0.9}
        }
        
        insights = {
            'demographic_scores': {},
            'platform_scores': {}
        }
        
        for age_group, preferences in age_preferences.items():
            score = self._calculate_preference_score(song_data, preferences)
            insights['demographic_scores'][age_group] = score
        
        for platform, preferences in platform_preferences.items():
            score = self._calculate_preference_score(song_data, preferences)
            insights['platform_scores'][platform] = score
        
        return insights
    
    def _find_similar_songs(self, song_features, insights, top_k):
        """Find similar songs from precomputed data"""
        # This is a simplified version - in practice, we'd want to calculate actual similarity using audio features
        demographic_scores = insights['demographic_scores']
        
        # For now, return top songs from the best matching demographic
        new_song_insights = self._calculate_new_song_insights(song_features)
        demo_scores_dict = new_song_insights['demographic_scores']
        # Use a lambda to avoid linter type error
        best_demo = max(demo_scores_dict, key=lambda k: demo_scores_dict[k])
        
        # Get top songs for that demographic
        demo_col = f'{best_demo}_score'
        if hasattr(demographic_scores, 'columns') and demo_col in demographic_scores.columns:
            top_songs = demographic_scores.nlargest(top_k, demo_col)
            return top_songs[['track_name', 'artists', demo_col]].to_dict('records')
        
        return []
    
    def _generate_recommendations(self, insights):
        """Generate marketing recommendations based on insights"""
        best_demographic = max(insights['demographic_scores'], 
                             key=insights['demographic_scores'].get)
        best_platform = max(insights['platform_scores'], 
                          key=insights['platform_scores'].get)
        
        recommendations = [
            f"Target {best_demographic} audience (score: {insights['demographic_scores'][best_demographic]:.1f}%)",
            f"Focus promotion on {best_platform} (score: {insights['platform_scores'][best_platform]:.1f}%)",
            "Consider creating platform-specific content variations",
            "Monitor engagement metrics to refine targeting"
        ]
        
        return recommendations
    
    def run_development_analysis(self):
        """Run analysis in development mode (optimized for speed)"""
        logger.info("Running development analysis...")
        self.load_data()
        
        if self.combined_df is None or self.combined_df.empty:
            logger.error("No data loaded")
            return None
        
        # Quick analysis on sample data
        sample_size = min(1000, len(self.combined_df))
        sample_df = self.combined_df.sample(n=sample_size, random_state=42)
        
        logger.info(
            f"Running quick analysis on {sample_size} songs..."
        )
        
        # Calculate basic stats
        results = {
            'data_summary': {
                'total_songs': len(self.combined_df),
                'sample_size': sample_size,
                'platforms': self.combined_df['source_platform'].unique().tolist(),
                'memory_mode': self.memory_mode
            },
            'feature_stats': sample_df[self.audio_features].describe().to_dict()
        }
        
        return results
    
    def run_production_analysis(self):
        """Run full production analysis with precomputation"""
        logger.info(
            f"Running production analysis..."
        )
        self.load_data()
        
        if self.combined_df is None or self.combined_df.empty:
            logger.error("No data loaded")
            return None
        
        # Run full precomputation
        results = self.precompute_insights()
        
        return results

# Example usage for the hybrid approach
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Marketing Insights Generator')
    parser.add_argument('mode', choices=['dev', 'prod', 'query', 'test_sample'], 
                       help='dev: quick development analysis, prod: full production analysis, query: query precomputed results, test_sample: test with a sample song')
    parser.add_argument('--data-dir', default='cleaned_music', help='Data directory path')
    parser.add_argument('--memory-mode', choices=['auto', 'full', 'medium', 'optimized'], 
                       default='auto', help='Memory usage mode')
    parser.add_argument('--cache-dir', default='marketing_cache', help='Cache directory path')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = HybridMarketingInsightsGenerator(
        data_path=args.data_dir,
        memory_mode=args.memory_mode,
        cache_dir=args.cache_dir
    )
    
    if args.mode == 'dev':
        # Development mode - quick analysis
        results = generator.run_development_analysis()
        if results:
            print("\n=== DEVELOPMENT ANALYSIS RESULTS ===")
            print(f"Total songs in dataset: {results['data_summary']['total_songs']}")
            print(f"Sample analyzed: {results['data_summary']['sample_size']}")
            print(f"Platforms: {', '.join(results['data_summary']['platforms'])}")
            print(f"Memory mode: {results['data_summary']['memory_mode']}")
    
    elif args.mode == 'prod':
        # Production mode - full precomputation
        results = generator.run_production_analysis()
        if results:
            print("\n=== PRODUCTION ANALYSIS COMPLETE ===")
            for key, df in results.items():
                print(f"{key}: {len(df)} records precomputed")
            print(f"Results cached in: {args.cache_dir}")
    
    elif args.mode == 'query':
        # Query mode - demonstrate querying precomputed results
        example_song = {
            'danceability': 0.8, 'energy': 0.9, 'key': 1, 'loudness': -5.0,
            'mode': 1, 'speechiness': 0.15, 'acousticness': 0.1,
            'instrumentalness': 0.0, 'liveness': 0.2, 'valence': 0.85, 'tempo': 128
        }
        
        recommendations = generator.get_song_recommendations(example_song)
        if recommendations:
            print("\nSong recommendations")
            print("Demographics:", recommendations['song_insights']['demographic_scores'])
            print("Platforms:", recommendations['song_insights']['platform_scores'])
            print("Recommendations:")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print("No precomputed insights available. Please run in 'prod' mode first.")
    elif args.mode == 'test_sample':
        # Test mode - quick test with a sample song
        test_song = {
            'danceability': 0.7, 'energy': 0.6, 'key': 5, 'loudness': -7.0,
            'mode': 1, 'speechiness': 0.12, 'acousticness': 0.3,
            'instrumentalness': 0.05, 'liveness': 0.15, 'valence': 0.65, 'tempo': 115
        }
        recommendations = generator.get_song_recommendations(test_song)
        if recommendations:
            print("\nTest sample song recommendations")
            print("Demographics:", recommendations['song_insights']['demographic_scores'])
            print("Platforms:", recommendations['song_insights']['platform_scores'])
            print("Recommendations:")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print("No precomputed insights available. Please run in 'prod' mode first.")