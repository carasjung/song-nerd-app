import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CrossPlatformMetrics:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.normalized_dir = self.data_dir / "cleaned_normalized"
        self.output_dir = self.data_dir / "cross_platform_features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio features for analysis
        self.audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo'
        ]
        
        self.datasets = {}
        
    def load_datasets(self):
        """Load all normalized datasets"""
        logging.info("Loading normalized datasets...")
        
        try:
            # Load TikTok datasets
            self.datasets['tiktok_2021'] = pd.read_csv(
                self.normalized_dir / "tiktok_2021_normalized.csv"
            )
            self.datasets['tiktok_2022'] = pd.read_csv(
                self.normalized_dir / "tiktok_2022_normalized.csv"
            )
            
            # Load Spotify datasets
            self.datasets['spotify_ds'] = pd.read_csv(
                self.normalized_dir / "spotify_ds_normalized.csv"
            )
            self.datasets['spotify_yt'] = pd.read_csv(
                self.normalized_dir / "spotify_yt_normalized.csv"
            )
            
            # Load music info
            self.datasets['music_info'] = pd.read_csv(
                self.normalized_dir / "music_info_normalized.csv"
            )
            
            # Load fuzzy matches
            self.fuzzy_matches = pd.read_csv(
                self.normalized_dir / "fuzzy_matches_spotifyds_vs_spotifyyt.csv"
            )
            
            for name, df in self.datasets.items():
                logging.info(f"Loaded {name}: {len(df)} tracks")
                
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
            
    def create_unified_dataset(self):
        """Create unified dataset with cross-platform matches"""
        logging.info("Creating unified cross-platform dataset...")
        
        datasets_with_platform = []
        
        # Process TikTok data
        tiktok_combined = pd.concat([
            self.datasets['tiktok_2021'].assign(platform='tiktok', year=2021),
            self.datasets['tiktok_2022'].assign(platform='tiktok', year=2022)
        ], ignore_index=True)
        
        # Remove duplicate columns
        tiktok_combined = tiktok_combined.loc[:, ~tiktok_combined.columns.duplicated()]
        
        # Standardize column names
        tiktok_combined = tiktok_combined.rename(columns={
            'track_name': 'track_clean',
            'artist_name': 'artist_clean',
            'track_pop': 'popularity_score',
            'artist_pop': 'artist_popularity'
        })
        
        # Process Spotify DS data
        spotify_ds = self.datasets['spotify_ds'].copy()

        spotify_ds = spotify_ds.loc[:, ~spotify_ds.columns.duplicated()]
        
        spotify_ds['platform'] = 'spotify'
        spotify_ds = spotify_ds.rename(columns={
            'track_name': 'track_clean',
            'artists': 'artist_clean',
            'popularity': 'popularity_score'
        })
        
        # Process Spotify YT data
        spotify_yt = self.datasets['spotify_yt'].copy()

        spotify_yt = spotify_yt.loc[:, ~spotify_yt.columns.duplicated()]
        
        spotify_yt['platform'] = 'youtube'
        spotify_yt = spotify_yt.rename(columns={
            'track': 'track_clean',
            'artist': 'artist_clean'
        })
        
        # Create popularity score for YT based on engagement
        if all(col in spotify_yt.columns for col in ['views', 'likes', 'comments']):
            spotify_yt['popularity_score'] = self._calculate_youtube_popularity(spotify_yt)
        
        # Select common columns
        common_cols = ['track_clean', 'artist_clean', 'platform'] + \
                    [col for col in self.audio_features if col in tiktok_combined.columns]
        
        if 'popularity_score' in tiktok_combined.columns:
            common_cols.append('popularity_score')
        if 'artist_popularity' in tiktok_combined.columns:
            common_cols.append('artist_popularity')
        if 'duration_ms' in tiktok_combined.columns:
            common_cols.append('duration_ms')
            
        # Filter datasets to common columns and ensure clean indices
        datasets_processed = []
        
        for df, name in [(tiktok_combined, 'tiktok'), (spotify_ds, 'spotify'), (spotify_yt, 'youtube')]:
            available_cols = [col for col in common_cols if col in df.columns]
            
            # Ensure no duplicate columns before filtering
            df_clean = df.loc[:, ~df.columns.duplicated()]
            
            # Filter to available columns
            df_filtered = df_clean[available_cols].copy()
            
            # Reset index and check for duplicates
            df_filtered = df_filtered.reset_index(drop=True)
            
            # Debug: Check for any remaining index issues
            if df_filtered.index.duplicated().any():
                logging.warning(f"{name} has duplicate indices after reset - fixing...")
                df_filtered = df_filtered.reset_index(drop=True)
            
            # Verify index is unique
            if not df_filtered.index.is_unique:
                logging.error(f"{name} still has non-unique index after reset!")
                # Force unique index
                df_filtered.index = range(len(df_filtered))
            
            datasets_processed.append(df_filtered)
            logging.info(f"Processed {name}: {len(df_filtered)} tracks, {len(available_cols)} features")
            
            # Debug info
            logging.debug(f"{name} columns after dedup: {list(df_filtered.columns)}")
        
        # Additional safety checks before concatenation
        for i, df in enumerate(datasets_processed):
            if not df.index.is_unique:
                logging.error(f"Dataset {i} has non-unique index before concat!")
                df.index = range(len(df))
            
            # Check for duplicate columns one final time
            if df.columns.duplicated().any():
                logging.error(f"Dataset {i} still has duplicate columns!")
                df = df.loc[:, ~df.columns.duplicated()]
                datasets_processed[i] = df
        
        try:
            # Combine all datasets with clean unique indices and no duplicate columns
            self.unified_dataset = pd.concat(datasets_processed, ignore_index=True, sort=False)
            
            # Final safety check and cleanup
            self.unified_dataset = self.unified_dataset.reset_index(drop=True)
            
            logging.info(f"Created unified dataset: {len(self.unified_dataset)} total tracks")
            logging.info(f"Final dataset columns: {list(self.unified_dataset.columns)}")
            
            # Verify no issues with the final dataset
            if not self.unified_dataset.index.is_unique:
                logging.warning("Final dataset has non-unique index - fixing...")
                self.unified_dataset = self.unified_dataset.reset_index(drop=True)
            
            return self.unified_dataset
            
        except Exception as e:
            logging.error(f"Error during concatenation: {str(e)}")
            
            # Additional debugging information
            for i, df in enumerate(datasets_processed):
                logging.error(f"Dataset {i} info:")
                logging.error(f"  Shape: {df.shape}")
                logging.error(f"  Index unique: {df.index.is_unique}")
                logging.error(f"  Index duplicated count: {df.index.duplicated().sum()}")
                logging.error(f"  Columns: {list(df.columns)}")
                logging.error(f"  Column duplicates: {df.columns.duplicated().sum()}")
            
            raise
    
        except pd.errors.InvalidIndexError as e:
            logging.error(f"Index error during concatenation: {str(e)}")
        
            # Try alternative approach: concatenate with explicit index handling
            logging.info("Attempting alternative concatenation method...")
        
            # Reset all indices and create new range indices
            for i, df in enumerate(datasets_processed):
                df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
                logging.debug(f"Dataset {i} new index: {df.index}")
            
            # Try concatenation again
            try:
                self.unified_dataset = pd.concat(datasets_processed, ignore_index=True, sort=False)
                self.unified_dataset = self.unified_dataset.reset_index(drop=True)
                logging.info(f"Alternative method successful: {len(self.unified_dataset)} total tracks")
                return self.unified_dataset
            except Exception as e2:
                logging.error(f"Alternative method also failed: {str(e2)}")
                raise e2
    
    def _calculate_youtube_popularity(self, df):
        """Calculate YouTube popularity score from engagement metrics"""
        # Normalize views, likes, comments
        scaler = MinMaxScaler()
        
        engagement_cols = ['views', 'likes', 'comments']
        available_cols = [col for col in engagement_cols if col in df.columns]
        
        if len(available_cols) == 0:
            return pd.Series([0] * len(df))
            
        # Fill missing values and convert to numeric
        df_copy = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
        for col in available_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        
        engagement_data = df_copy[available_cols].values
        engagement_normalized = scaler.fit_transform(engagement_data)
        
        # Weighted combination (views 50%, likes 30%, comments 20%)
        weights = [0.5, 0.3, 0.2][:len(available_cols)]
        weights = np.array(weights) / sum(weights)  # Normalize weights
        
        popularity = np.average(engagement_normalized, axis=1, weights=weights)
        return pd.Series(popularity * 100)  # Scale to 0-100
    
    def calculate_virality_score(self):
        """Calculate comprehensive virality score"""
        logging.info("Calculating virality scores...")
        
        # Group by track and artist to get cross-platform metrics
        grouped = self.unified_dataset.groupby(['track_clean', 'artist_clean']).agg({
            'popularity_score': ['mean', 'max', 'count'],
            'platform': lambda x: list(x.unique())
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['track_clean', 'artist_clean', 'popularity_mean', 
                          'popularity_max', 'platform_count', 'platforms']
        
        # Calculate virality components
        grouped['platform_diversity'] = grouped['platform_count'] / 3  # Max 3 platforms
        grouped['peak_popularity'] = grouped['popularity_max'] / 100  # Normalize to 0-1
        grouped['consistent_popularity'] = grouped['popularity_mean'] / 100
        
        # Virality score formula
        grouped['virality_score'] = (
            0.4 * grouped['peak_popularity'] +
            0.3 * grouped['consistent_popularity'] +
            0.3 * grouped['platform_diversity']
        ) * 100
        
        # Add virality categories
        grouped['virality_tier'] = pd.cut(
            grouped['virality_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        )
        
        self.virality_scores = grouped
        logging.info(f"Calculated virality scores for {len(grouped)} unique tracks")
        
        return grouped
    
    def analyze_platform_affinity(self):
        """Analyze which audio features correlate with success on each platform"""
        logging.info("Analyzing platform affinity...")
        
        platform_affinity = {}
        
        for platform in self.unified_dataset['platform'].unique():
            platform_data = self.unified_dataset[
                self.unified_dataset['platform'] == platform
            ].copy()
            
            if 'popularity_score' not in platform_data.columns:
                continue
                
            # Calculate correlations between audio features and popularity
            correlations = {}
            available_features = [f for f in self.audio_features if f in platform_data.columns]
            
            for feature in available_features:
                if platform_data[feature].notna().sum() > 10:  # Minimum data points
                    corr = platform_data[feature].corr(platform_data['popularity_score'])
                    if not np.isnan(corr):
                        correlations[feature] = corr
            
            # Find top positive and negative correlations
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            platform_affinity[platform] = {
                'correlations': correlations,
                'top_positive': [item for item in sorted_corr if item[1] > 0][:3],
                'top_negative': [item for item in sorted_corr if item[1] < 0][:3],
                'sample_size': len(platform_data)
            }
            
            logging.info(f"Platform {platform}: {len(correlations)} feature correlations calculated")
        
        self.platform_affinity = platform_affinity
        return platform_affinity
    
    def create_demographic_alignment(self):
        """Create demographic alignment based on audio features and platform preferences"""
        logging.info("Creating demographic alignment profiles...")
        
        # Define demographic profiles based on platform characteristics
        demographic_profiles = {
            'Gen Z (TikTok-focused)': {
                'platforms': ['tiktok'],
                'preferred_features': {
                    'danceability': (0.6, 1.0),
                    'energy': (0.5, 1.0),
                    'valence': (0.4, 1.0),
                    'tempo': (100, 180)
                }
            },
            'Millennials (Spotify-focused)': {
                'platforms': ['spotify'],
                'preferred_features': {
                    'danceability': (0.3, 0.8),
                    'energy': (0.3, 0.8),
                    'acousticness': (0.0, 0.6),
                    'valence': (0.2, 0.8)
                }
            },
            'Cross-Platform Appeal': {
                'platforms': ['tiktok', 'spotify', 'youtube'],
                'preferred_features': {
                    'danceability': (0.5, 0.8),
                    'energy': (0.4, 0.8),
                    'valence': (0.4, 0.8),
                    'speechiness': (0.0, 0.3)
                }
            }
        }
        
        # Score tracks against each demographic profile
        demographic_scores = []
        
        for _, track in self.unified_dataset.iterrows():
            track_scores = {'track_clean': track['track_clean'], 
                           'artist_clean': track['artist_clean'],
                           'platform': track['platform']}
            
            for demo_name, demo_profile in demographic_profiles.items():
                score = self._calculate_demographic_score(track, demo_profile)
                track_scores[f'{demo_name}_alignment'] = score
            
            demographic_scores.append(track_scores)
        
        self.demographic_alignment = pd.DataFrame(demographic_scores)
        logging.info(f"Created demographic alignment for {len(demographic_scores)} tracks")
        
        return self.demographic_alignment
    
    def _calculate_demographic_score(self, track, demo_profile):
        """Calculate how well a track aligns with a demographic profile"""
        scores = []
        
        for feature, (min_val, max_val) in demo_profile['preferred_features'].items():
            if feature in track and pd.notna(track[feature]):
                feature_val = track[feature]
                
                # Score based on how well the feature fits the range
                if min_val <= feature_val <= max_val:
                    score = 1.0 # Perfect fit
                else:
                    # Calculate distance from preferred range
                    if feature_val < min_val:
                        distance = min_val - feature_val
                    else:
                        distance = feature_val - max_val
                    
                    # Convert distance to score (max distance of 1.0 gives score of 0)
                    score = max(0, 1 - distance)
                
                scores.append(score)
        
        return np.mean(scores) if scores else 0
    
    def create_comprehensive_features(self):
        """Create comprehensive cross-platform features"""
        logging.info("Creating comprehensive cross-platform features...")
        
        # Merge all computed metrics
        comprehensive_features = self.unified_dataset.copy()
        
        # Add virality scores
        if hasattr(self, 'virality_scores'):
            comprehensive_features = comprehensive_features.merge(
                self.virality_scores[['track_clean', 'artist_clean', 'virality_score', 
                                    'virality_tier', 'platform_diversity']],
                on=['track_clean', 'artist_clean'],
                how='left'
            )
        
        # Add demographic alignment
        if hasattr(self, 'demographic_alignment'):
            demographic_cols = [col for col in self.demographic_alignment.columns 
                              if col.endswith('_alignment')]
            merge_cols = ['track_clean', 'artist_clean', 'platform'] + demographic_cols
            
            comprehensive_features = comprehensive_features.merge(
                self.demographic_alignment[merge_cols],
                on=['track_clean', 'artist_clean', 'platform'],
                how='left'
            )
        
        # Add platform success indicators
        comprehensive_features['is_cross_platform'] = comprehensive_features.groupby(
            ['track_clean', 'artist_clean']
        )['platform'].transform('nunique') > 1
        
        # Add popularity categories
        if 'popularity_score' in comprehensive_features.columns:
            comprehensive_features['popularity_tier'] = pd.cut(
                comprehensive_features['popularity_score'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        self.comprehensive_features = comprehensive_features
        logging.info(f"Created comprehensive features dataset: {len(comprehensive_features)} tracks")
        
        return comprehensive_features
    
    def generate_insights_report(self):
        """Generate insights report"""
        logging.info("Generating cross-platform insights report...")
        
        report_path = self.output_dir / "cross_platform_insights.txt"
        
        with open(report_path, 'w') as f:
            f.write("CROSS-PLATFORM MUSIC ANALYTICS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total tracks analyzed: {len(self.unified_dataset)}\n")
            f.write(f"Platforms: {', '.join(self.unified_dataset['platform'].unique())}\n")
            f.write(f"Platform distribution:\n")
            platform_counts = self.unified_dataset['platform'].value_counts()
            for platform, count in platform_counts.items():
                f.write(f"  {platform}: {count} tracks\n")
            
            # Virality insights
            if hasattr(self, 'virality_scores'):
                f.write(f"\nVIRALITY ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Unique tracks with virality scores: {len(self.virality_scores)}\n")
                
                virality_dist = self.virality_scores['virality_tier'].value_counts()
                f.write("Virality distribution:\n")
                for tier, count in virality_dist.items():
                    f.write(f"  {tier}: {count} tracks\n")
                
                # Top viral tracks
                top_viral = self.virality_scores.nlargest(5, 'virality_score')
                f.write(f"\nTop 5 viral tracks:\n")
                for _, track in top_viral.iterrows():
                    f.write(f"  {track['artist_clean']} - {track['track_clean']}: "
                           f"{track['virality_score']:.1f}\n")
            
            # Platform affinity insights
            if hasattr(self, 'platform_affinity'):
                f.write(f"\nPLATFORM AFFINITY ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                for platform, data in self.platform_affinity.items():
                    f.write(f"\n{platform.upper()} Platform:\n")
                    f.write(f"Sample size: {data['sample_size']} tracks\n")
                    
                    if data['top_positive']:
                        f.write("Top success factors:\n")
                        for feature, corr in data['top_positive']:
                            f.write(f"  {feature}: {corr:.3f}\n")
                    
                    if data['top_negative']:
                        f.write("Negative factors:\n")
                        for feature, corr in data['top_negative']:
                            f.write(f"  {feature}: {corr:.3f}\n")
            
            # Demographic insights
            if hasattr(self, 'demographic_alignment'):
                f.write(f"\nDEMOGRAPHIC ALIGNMENT:\n")
                f.write("-" * 25 + "\n")
                
                demo_cols = [col for col in self.demographic_alignment.columns 
                           if col.endswith('_alignment')]
                
                for col in demo_cols:
                    demo_name = col.replace('_alignment', '')
                    mean_score = self.demographic_alignment[col].mean()
                    f.write(f"{demo_name}: {mean_score:.3f} average alignment\n")
        
        logging.info(f"Insights report saved to {report_path}")
    
    def save_results(self):
        """Save all results to files"""
        logging.info("Saving cross-platform analysis results...")
        
        # Save unified dataset
        self.unified_dataset.to_csv(
            self.output_dir / "unified_cross_platform_dataset.csv", 
            index=False
        )
        
        # Save virality scores
        if hasattr(self, 'virality_scores'):
            self.virality_scores.to_csv(
                self.output_dir / "virality_scores.csv", 
                index=False
            )
        
        # Save platform affinity analysis
        if hasattr(self, 'platform_affinity'):
            affinity_data = []
            for platform, data in self.platform_affinity.items():
                for feature, correlation in data['correlations'].items():
                    affinity_data.append({
                        'platform': platform,
                        'feature': feature,
                        'correlation': correlation,
                        'sample_size': data['sample_size']
                    })
            
            pd.DataFrame(affinity_data).to_csv(
                self.output_dir / "platform_affinity_analysis.csv", 
                index=False
            )
        
        # Save demographic alignment
        if hasattr(self, 'demographic_alignment'):
            self.demographic_alignment.to_csv(
                self.output_dir / "demographic_alignment.csv", 
                index=False
            )
        
        # Save comprehensive features
        if hasattr(self, 'comprehensive_features'):
            self.comprehensive_features.to_csv(
                self.output_dir / "comprehensive_cross_platform_features.csv", 
                index=False
            )
        
        logging.info("All results saved successfully!")
    
    def run_complete_analysis(self):
        """Run the complete cross-platform analysis pipeline"""
        logging.info("Starting complete cross-platform analysis...")
        
        try:
            self.load_datasets()
            
            self.create_unified_dataset()
            
            # Calculate metrics
            self.calculate_virality_score()
            self.analyze_platform_affinity()
            self.create_demographic_alignment()
            
            # Create comprehensive features
            self.create_comprehensive_features()
            
            # Generate insights
            self.generate_insights_report()
            
            # Save results
            self.save_results()
            
            logging.info("Cross-platform analysis completed successfully!")
            
            return {
                'unified_dataset': self.unified_dataset,
                'virality_scores': self.virality_scores,
                'platform_affinity': self.platform_affinity,
                'demographic_alignment': self.demographic_alignment,
                'comprehensive_features': self.comprehensive_features
            }
            
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            raise

# Use example
if __name__ == "__main__":
    analyzer = CrossPlatformMetrics(data_dir="data")
    results = analyzer.run_complete_analysis()
    
    print("Cross-platform analysis completed!")
    print(f"Unified dataset shape: {results['unified_dataset'].shape}")
    print(f"Virality scores for {len(results['virality_scores'])} unique tracks")
    print(f"Platform affinity analysis for {len(results['platform_affinity'])} platforms")