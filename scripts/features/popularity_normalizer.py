# popularity_normalizer.py
# This script is used to normalize the popularity scores of the songs in the master dataset.
# It uses the audio features of the song to generate insights on the song's target demographic,
# platform recommendations, and marketing suggestions.
# It also uses the audio features of the song to generate insights on the song's trend alignment, 
# similar artists, viral potential and overall trend alignment.

import pandas as pd
import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)

class PopularityNormalizer:
    """
    Fix popularity scores and missing data in the master dataset
    """
    
    def __init__(self, master_dataset_path='integration_cache/master_music_dataset_fast.csv'):
        self.master_dataset_path = master_dataset_path
        self.df = None
        
    def load_dataset(self):
        """Load the master dataset"""
        if not os.path.exists(self.master_dataset_path):
            logger.error(f"Master dataset not found: {self.master_dataset_path}")
            return False
        
        logger.info("Loading master dataset...")
        self.df = pd.read_csv(self.master_dataset_path)
        logger.info(f"Loaded {len(self.df)} tracks")
        return True
    
    def analyze_popularity_issues(self):
        """Analyze the popularity data issues"""
        if self.df is None:
            self.load_dataset()
        
        analysis = {
            'total_tracks': len(self.df),
            'popularity_stats': {},
            'platform_stats': {},
            'missing_data': {}
        }
        
        # Analyze different popularity columns
        popularity_columns = ['avg_popularity', 'max_popularity', 'popularity_score']
        
        for col in popularity_columns:
            if col in self.df.columns:
                values = self.df[col].dropna()
                analysis['popularity_stats'][col] = {
                    'count': len(values),
                    'missing': self.df[col].isna().sum(),
                    'min': float(values.min()) if len(values) > 0 else None,
                    'max': float(values.max()) if len(values) > 0 else None,
                    'mean': float(values.mean()) if len(values) > 0 else None,
                    'median': float(values.median()) if len(values) > 0 else None
                }
        
        # Analyze by platform
        platforms = ['spotify', 'tiktok', 'youtube']
        for platform in platforms:
            if platform in self.df.columns:
                platform_tracks = self.df[self.df[platform] == True]
                analysis['platform_stats'][platform] = {
                    'total_tracks': len(platform_tracks),
                    'with_popularity': platform_tracks['avg_popularity'].notna().sum(),
                    'avg_popularity_range': {
                        'min': float(platform_tracks['avg_popularity'].min()) if len(platform_tracks) > 0 else None,
                        'max': float(platform_tracks['avg_popularity'].max()) if len(platform_tracks) > 0 else None
                    }
                }
        
        # Missing data analysis
        analysis['missing_data'] = {
            'no_popularity': self.df['avg_popularity'].isna().sum(),
            'no_platform': self.df[['spotify', 'tiktok', 'youtube']].sum(axis=1).eq(0).sum(),
            'no_genre': (self.df['genre'] == 'unknown').sum(),
            'no_audio_features': self.df[['danceability', 'energy', 'valence']].isna().all(axis=1).sum()
        }
        
        return analysis
    
    def normalize_popularity_scores(self):
        """Normalize popularity scores to a consistent 0-100 scale"""
        if self.df is None:
            self.load_dataset()
        
        logger.info("Normalizing popularity scores...")
        
        # Create a new normalized popularity column
        self.df['normalized_popularity'] = np.nan
        
        # Handle different platforms separately
        self._normalize_spotify_popularity()
        self._normalize_youtube_popularity()
        self._normalize_tiktok_popularity()
        
        # Fill missing values with platform-based defaults
        self._fill_missing_popularity()
        
        # Ensure all values are in 0-100 range
        self.df['normalized_popularity'] = np.clip(self.df['normalized_popularity'], 0, 100)
        
        logger.info("Popularity normalization complete")
    
    def _normalize_spotify_popularity(self):
        """Normalize Spotify popularity (already 0-100)"""
        spotify_mask = (self.df['spotify'] == True) & (self.df['avg_popularity'].notna())
        
        # Spotify popularity is already 0-100, but let's clean it
        spotify_pop = self.df.loc[spotify_mask, 'avg_popularity']
        
        # Clip to 0-100 range (in case of outliers)
        normalized_spotify = np.clip(spotify_pop, 0, 100)
        
        self.df.loc[spotify_mask, 'normalized_popularity'] = normalized_spotify
        
        logger.info(f"Normalized {len(spotify_pop)} Spotify tracks")
    
    def _normalize_youtube_popularity(self):
        """Normalize YouTube view counts to 0-100 scale"""
        youtube_mask = (self.df['youtube'] == True) & (self.df['avg_popularity'].notna())
        
        if youtube_mask.sum() == 0:
            return
        
        youtube_views = self.df.loc[youtube_mask, 'avg_popularity']
        
        # Log transform for better distribution
        log_views = np.log10(youtube_views.replace(0, 1))
        
        # Normalize to 0-100 scale
        # Use percentile-based normalization
        min_log = log_views.quantile(0.01)  # 1st percentile as min
        max_log = log_views.quantile(0.99)  # 99th percentile as max
        
        normalized_youtube = 100 * (log_views - min_log) / (max_log - min_log)
        normalized_youtube = np.clip(normalized_youtube, 0, 100)
        
        self.df.loc[youtube_mask, 'normalized_popularity'] = normalized_youtube
        
        logger.info(f"Normalized {len(youtube_views)} YouTube tracks")
    
    def _normalize_tiktok_popularity(self):
        """Normalize TikTok popularity (usually 0-100 already)"""
        tiktok_mask = (self.df['tiktok'] == True) & (self.df['avg_popularity'].notna())
        
        if tiktok_mask.sum() == 0:
            return
        
        tiktok_pop = self.df.loc[tiktok_mask, 'avg_popularity']
        
        # TikTok is usually 0-100, but double-check
        if tiktok_pop.max() <= 100:
            # Already normalized
            self.df.loc[tiktok_mask, 'normalized_popularity'] = tiktok_pop
        else:
            # Scale down to 0-100
            normalized_tiktok = 100 * (tiktok_pop - tiktok_pop.min()) / (tiktok_pop.max() - tiktok_pop.min())
            self.df.loc[tiktok_mask, 'normalized_popularity'] = normalized_tiktok
        
        logger.info(f"Normalized {len(tiktok_pop)} TikTok tracks")
    
    def _fill_missing_popularity(self):
        """Fill missing popularity values with intelligent defaults"""
        missing_mask = self.df['normalized_popularity'].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            return
        
        logger.info(f"Filling {missing_count} missing popularity values...")
        
        # Strategy 1: Use genre-based averages
        for genre in self.df['genre'].unique():
            if genre == 'unknown':
                continue
                
            genre_mask = (self.df['genre'] == genre) & missing_mask
            if genre_mask.sum() == 0:
                continue
            
            # Get average popularity for this genre
            genre_avg = self.df[
                (self.df['genre'] == genre) & 
                (self.df['normalized_popularity'].notna())
            ]['normalized_popularity'].mean()
            
            if pd.notna(genre_avg):
                self.df.loc[genre_mask, 'normalized_popularity'] = genre_avg
                missing_mask = self.df['normalized_popularity'].isna()  # Update mask
        
        # Strategy 2: Use platform-based defaults for remaining missing values
        still_missing = self.df['normalized_popularity'].isna()
        
        platform_defaults = {
            'spotify': 45,  # Slightly below average
            'youtube': 35,  # Lower default for YouTube
            'tiktok': 55    # Higher for TikTok (viral platform)
        }
        
        for platform, default_score in platform_defaults.items():
            if platform in self.df.columns:
                platform_missing = still_missing & (self.df[platform] == True)
                self.df.loc[platform_missing, 'normalized_popularity'] = default_score
                still_missing = self.df['normalized_popularity'].isna()  # Update
        
        # Strategy 3: Overall default for any remaining missing values
        final_missing = self.df['normalized_popularity'].isna()
        if final_missing.sum() > 0:
            self.df.loc[final_missing, 'normalized_popularity'] = 40  # Conservative default
    
    def fix_platform_indicators(self):
        """Fix missing platform indicators"""
        if self.df is None:
            self.load_dataset()
        
        logger.info("Fixing platform indicators...")
        
        # Fix platform columns based on primary_source
        if 'primary_source' in self.df.columns:
            # Set platform flags based on source
            self.df['spotify'] = self.df['primary_source'].str.contains('spotify', na=False)
            self.df['youtube'] = self.df['primary_source'].str.contains('youtube', na=False)
            self.df['tiktok'] = self.df['primary_source'].str.contains('tiktok', na=False)
            
        # Ensure at least one platform is marked for each track
        no_platform = self.df[['spotify', 'tiktok', 'youtube']].sum(axis=1) == 0
        
        if no_platform.sum() > 0:
            logger.info(f"Fixing {no_platform.sum()} tracks with no platform indicators")
            # Default to spotify for tracks with no platform
            self.df.loc[no_platform, 'spotify'] = True
    
    def fix_genres(self):
        """Improve genre assignments"""
        if self.df is None:
            self.load_dataset()
        
        logger.info("Improving genre assignments...")
        
        # Simple genre inference based on audio features
        unknown_genre = self.df['genre'] == 'unknown'
        
        if unknown_genre.sum() > 0:
            logger.info(f"Inferring genres for {unknown_genre.sum()} tracks")
            
            # Use audio features to guess genres
            for idx in self.df[unknown_genre].index:
                inferred_genre = self._infer_genre_from_audio_features(self.df.loc[idx])
                if inferred_genre != 'unknown':
                    self.df.loc[idx, 'genre'] = inferred_genre
    
    def _infer_genre_from_audio_features(self, track):
        """Infer genre based on audio features"""
        # Simple rule-based genre inference
        danceability = track.get('danceability', 0.5)
        energy = track.get('energy', 0.5)
        valence = track.get('valence', 0.5)
        acousticness = track.get('acousticness', 0.5)
        instrumentalness = track.get('instrumentalness', 0.5)
        
        # Skip if no audio features
        if pd.isna(danceability) or pd.isna(energy):
            return 'unknown'
        
        # Electronic/Dance
        if danceability > 0.7 and energy > 0.7 and acousticness < 0.3:
            return 'electronic'
        
        # Pop
        elif danceability > 0.6 and energy > 0.5 and valence > 0.5:
            return 'pop'
        
        # Acoustic/Folk
        elif acousticness > 0.7 and energy < 0.5:
            return 'acoustic'
        
        # Hip-hop/Rap (high speechiness would be better, but we'll use energy + low acousticness)
        elif energy > 0.6 and acousticness < 0.2 and danceability > 0.6:
            return 'hip hop'
        
        # Classical/Instrumental
        elif instrumentalness > 0.7:
            return 'classical'
        
        # Default
        else:
            return 'pop'  # Most common default
    
    def create_fixed_dataset(self, output_path='integration_cache/master_music_dataset_fixed.csv'):
        """Create the fixed version of the dataset"""
        if self.df is None:
            self.load_dataset()
        
        logger.info("Creating fixed dataset...")
        
        # Apply all fixes
        self.normalize_popularity_scores()
        self.fix_platform_indicators() 
        self.fix_genres()
        
        # Add a quality score for each track
        self.df['data_quality_score'] = self._calculate_quality_scores()
        
        # Save the fixed dataset
        self.df.to_csv(output_path, index=False)
        logger.info(f"Fixed dataset saved to: {output_path}")
        
        # Generate summary report
        self._generate_fix_report(output_path.replace('.csv', '_report.json'))
        
        return output_path
    
    def _calculate_quality_scores(self):
        """Calculate a data quality score (0-100) for each track"""
        quality_scores = np.zeros(len(self.df))
        
        # Audio features present (40 points max)
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
        audio_completeness = self.df[audio_features].notna().sum(axis=1) / len(audio_features)
        quality_scores += audio_completeness * 40
        
        # Has popularity score (20 points)
        has_popularity = self.df['normalized_popularity'].notna()
        quality_scores += has_popularity * 20
        
        # Has proper genre (15 points)
        has_genre = self.df['genre'] != 'unknown'
        quality_scores += has_genre * 15
        
        # Multi-platform presence (15 points)
        platform_count = self.df[['spotify', 'tiktok', 'youtube']].sum(axis=1)
        quality_scores += np.minimum(platform_count * 5, 15)
        
        # Multiple sources (10 points)
        multi_source = self.df['source_count'] > 1
        quality_scores += multi_source * 10
        
        return np.round(quality_scores, 1)
    
    def _generate_fix_report(self, report_path):
        """Generate a report on the fixes applied"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_tracks': int(len(self.df)),
            'fixes_applied': {
                'popularity_normalization': True,
                'platform_indicators': True,
                'genre_inference': True,
                'quality_scoring': True
            },
            'final_stats': {
                'tracks_with_popularity': int(self.df['normalized_popularity'].notna().sum()),
                'tracks_with_known_genre': int((self.df['genre'] != 'unknown').sum()),
                'avg_quality_score': float(self.df['data_quality_score'].mean()),
                'high_quality_tracks': int((self.df['data_quality_score'] >= 80).sum())
            },
            'platform_distribution': {
                'spotify': int(self.df['spotify'].sum()),
                'youtube': int(self.df['youtube'].sum()),
                'tiktok': int(self.df['tiktok'].sum())
            },
            'genre_distribution': {k: int(v) for k, v in self.df['genre'].value_counts().head(10).to_dict().items()},
            'popularity_stats': {
                'min': float(self.df['normalized_popularity'].min()),
                'max': float(self.df['normalized_popularity'].max()),
                'mean': float(self.df['normalized_popularity'].mean()),
                'median': float(self.df['normalized_popularity'].median())
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Fix report saved to: {report_path}")
        return report

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix popularity scores and missing data')
    parser.add_argument('command', choices=['analyze', 'fix'], 
                       help='analyze: show issues, fix: create fixed dataset')
    parser.add_argument('--input', default='integration_cache/master_music_dataset_fast.csv',
                       help='Input master dataset path')
    parser.add_argument('--output', default='integration_cache/master_music_dataset_fixed.csv',
                       help='Output fixed dataset path')
    
    args = parser.parse_args()
    
    normalizer = PopularityNormalizer(args.input)
    
    if args.command == 'analyze':
        print("Analyzing popularity and data quality issues...")
        analysis = normalizer.analyze_popularity_issues()
        
        print(f"\nDataset Overview")
        print(f"Total tracks: {analysis['total_tracks']:,}")
        
        print(f"\nMissing Data Issues")
        missing = analysis['missing_data']
        print(f"  No popularity: {missing['no_popularity']:,}")
        print(f"  No platform: {missing['no_platform']:,}")
        print(f"  Unknown genre: {missing['no_genre']:,}")
        print(f"  No audio features: {missing['no_audio_features']:,}")
        
        print(f"\nPopularity Ranges by Platform")
        for platform, stats in analysis['platform_stats'].items():
            if stats['total_tracks'] > 0:
                range_info = stats['avg_popularity_range']
                print(f"  {platform.title()}: {stats['total_tracks']:,} tracks")
                if range_info['min'] is not None:
                    print(f"    Range: {range_info['min']:.1f} - {range_info['max']:.1f}")
                print(f"    With popularity: {stats['with_popularity']:,}")
        
    elif args.command == 'fix':
        print("Creating fixed dataset with normalized popularity scores...")
        fixed_path = normalizer.create_fixed_dataset(args.output)
        
        print(f"\nFixes Applied")
        print(f"  Normalized popularity scores to 0-100 scale")
        print(f"  Fixed platform indicators")
        print(f"  Inferred missing genres")
        print(f"  Added data quality scores")
        
        print(f"\nFixed dataset saved to: {fixed_path}")
        print(f"Report saved to: {fixed_path.replace('.csv', '_report.json')}")
        
        print(f"\nNext step: Test search with the fixed dataset")
        print(f"   python3 scripts/features/optimized_integration_system.py search --query \"Bad Guy\" --master-file {fixed_path}")