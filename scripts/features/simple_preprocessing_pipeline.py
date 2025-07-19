import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SimplePreprocessingPipeline:
    """
    Simplified preprocessing pipeline that actually works
    Creates all 4 required datasets with essential features
    """
    
    def __init__(self, input_dataset='integration_cache/master_music_dataset_deduplicated.csv',
                 output_dir='final_datasets'):
        self.input_dataset = input_dataset
        self.output_dir = output_dir
        self.df = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_and_clean_data(self):
        """Load and perform basic cleaning"""
        print("Loading and cleaning data...")
        
        if not os.path.exists(self.input_dataset):
            print(f"Input file not found: {self.input_dataset}")
            return False
        
        # Load dataset
        self.df = pd.read_csv(self.input_dataset)
        print(f"Loaded {len(self.df):,} tracks")
        
        # Create clean text fields if they don't exist
        if 'track_name_clean' not in self.df.columns:
            self.df['track_name_clean'] = self.df['track_name'].astype(str).str.strip()
        
        if 'artist_name_clean' not in self.df.columns:
            self.df['artist_name_clean'] = self.df['artist_name'].astype(str).str.strip()
        
        if 'genre_clean' not in self.df.columns:
            self.df['genre_clean'] = self.df['genre'].fillna('unknown').astype(str).str.lower().str.strip()
        
        # Fill missing values
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
        
        for feature in audio_features:
            if feature in self.df.columns:
                self.df[feature] = self.df[feature].fillna(self.df[feature].median())
        
        # Fill other important columns
        if 'normalized_popularity' in self.df.columns:
            self.df['normalized_popularity'] = self.df['normalized_popularity'].fillna(50)
        
        if 'data_quality_score' in self.df.columns:
            self.df['data_quality_score'] = self.df['data_quality_score'].fillna(75)
        
        # Ensure platform columns are boolean
        platform_cols = ['spotify', 'tiktok', 'youtube']
        for col in platform_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(False).astype(bool)
        
        print(f"Data cleaned and filled")
        return True
    
    def create_derived_features(self):
        """Create derived features for analysis"""
        print("🔧 Creating derived features...")
        
        # Platform features
        platform_cols = ['spotify', 'tiktok', 'youtube']
        available_platforms = [col for col in platform_cols if col in self.df.columns]
        
        if available_platforms:
            self.df['platform_count'] = self.df[available_platforms].sum(axis=1)
            self.df['is_multi_platform'] = (self.df['platform_count'] > 1).astype(int)
            
            # Primary platform
            def get_primary_platform(row):
                if row.get('spotify', False):
                    return 'spotify'
                elif row.get('youtube', False):
                    return 'youtube'
                elif row.get('tiktok', False):
                    return 'tiktok'
                else:
                    return 'unknown'
            
            self.df['primary_platform'] = self.df.apply(get_primary_platform, axis=1)
        
        # Audio appeal score
        appeal_features = ['danceability', 'energy', 'valence']
        available_appeal = [f for f in appeal_features if f in self.df.columns]
        
        if available_appeal:
            self.df['audio_appeal'] = self.df[available_appeal].mean(axis=1)
        
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
                else:
                    return 'niche'
            
            self.df['popularity_tier'] = self.df['normalized_popularity'].apply(popularity_tier)
        
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
        
        # Genre categories
        def categorize_genre(genre):
            genre = str(genre).lower()
            if any(g in genre for g in ['pop', 'hip hop', 'rap', 'electronic', 'dance']):
                return 'mainstream'
            elif any(g in genre for g in ['rock', 'country', 'folk', 'blues', 'jazz']):
                return 'traditional'
            elif any(g in genre for g in ['indie', 'alternative', 'experimental']):
                return 'alternative'
            else:
                return 'other'
        
        if 'genre_clean' in self.df.columns:
            self.df['genre_category'] = self.df['genre_clean'].apply(categorize_genre)
        
        print(f"Derived features created")
    
    def create_master_dataset(self):
        """Create the master music dataset"""
        print("Creating master_music_data.csv...")
        
        # Select columns for master dataset
        master_columns = [
            # Core identifiers
            'track_name_clean', 'artist_name_clean', 'genre_clean', 'genre_category',
            
            # Audio features
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness',
            
            # Derived features
            'audio_appeal', 'normalized_popularity', 'popularity_tier',
            'data_quality_score', 'quality_tier',
            
            # Platform information
            'spotify', 'tiktok', 'youtube', 'primary_platform',
            'platform_count', 'is_multi_platform',
            
            # Metadata
            'source_count'
        ]
        
        # Filter to available columns
        available_columns = [col for col in master_columns if col in self.df.columns]
        
        master_df = self.df[available_columns].copy()
        
        # Save master dataset
        master_file = os.path.join(self.output_dir, 'master_music_data.csv')
        master_df.to_csv(master_file, index=False)
        
        print(f"Master dataset created: {len(master_df):,} rows, {len(master_df.columns)} columns")
        return len(master_df)
    
    def create_platform_performance(self):
        """Create platform performance dataset"""
        print("Creating platform_performance.csv...")
        
        platform_data = []
        platforms = ['spotify', 'tiktok', 'youtube']
        
        # Sample for performance (first 10,000 tracks)
        sample_df = self.df.head(10000)
        
        for _, track in sample_df.iterrows():
            base_record = {
                'track_name': track.get('track_name_clean', ''),
                'artist_name': track.get('artist_name_clean', ''),
                'genre': track.get('genre_clean', ''),
                'genre_category': track.get('genre_category', '')
            }
            
            for platform in platforms:
                if track.get(platform, False):
                    platform_record = base_record.copy()
                    platform_record.update({
                        'platform': platform,
                        'popularity_score': track.get('normalized_popularity', 0),
                        'audio_appeal': track.get('audio_appeal', 0),
                        'quality_score': track.get('data_quality_score', 0),
                        'is_primary_platform': (track.get('primary_platform') == platform)
                    })
                    platform_data.append(platform_record)
        
        if platform_data:
            platform_df = pd.DataFrame(platform_data)
            platform_file = os.path.join(self.output_dir, 'platform_performance.csv')
            platform_df.to_csv(platform_file, index=False)
            print(f"Platform performance dataset created: {len(platform_df):,} rows")
            return len(platform_df)
        
        return 0
    
    def create_demographic_preferences(self):
        """Create demographic preferences dataset"""
        print("Creating demographic_preferences.csv...")
        
        # Define demographic profiles
        demographic_profiles = {
            'Gen Z (16-24)': {
                'danceability': 0.8, 'energy': 0.75, 'valence': 0.7,
                'preferred_genres': ['pop', 'hip hop', 'electronic'],
                'preferred_platform': 'tiktok'
            },
            'Millennials (25-40)': {
                'danceability': 0.65, 'energy': 0.6, 'valence': 0.6,
                'preferred_genres': ['pop', 'rock', 'hip hop'],
                'preferred_platform': 'spotify'
            },
            'Gen X (41-56)': {
                'danceability': 0.5, 'energy': 0.55, 'valence': 0.5,
                'preferred_genres': ['rock', 'pop', 'country'],
                'preferred_platform': 'spotify'
            },
            'Boomers (57+)': {
                'danceability': 0.4, 'energy': 0.4, 'valence': 0.45,
                'preferred_genres': ['rock', 'country', 'jazz'],
                'preferred_platform': 'youtube'
            }
        }
        
        demographic_data = []
        
        # Sample tracks for analysis
        sample_df = self.df.sample(n=min(5000, len(self.df)), random_state=42)
        
        for demo, profile in demographic_profiles.items():
            for _, track in sample_df.iterrows():
                # Calculate match score
                match_score = 0
                feature_count = 0
                
                for feature in ['danceability', 'energy', 'valence']:
                    if feature in track and pd.notna(track[feature]):
                        ideal_value = profile[feature]
                        alignment = 1 - abs(track[feature] - ideal_value)
                        match_score += alignment
                        feature_count += 1
                
                # Average match score
                if feature_count > 0:
                    avg_match = match_score / feature_count
                    
                    # Genre bonus
                    genre = str(track.get('genre_clean', '')).lower()
                    if any(g in genre for g in profile['preferred_genres']):
                        avg_match += 0.2
                    
                    # Only include good matches
                    if avg_match > 0.6:
                        demographic_data.append({
                            'demographic': demo,
                            'track_name': track.get('track_name_clean', ''),
                            'artist_name': track.get('artist_name_clean', ''),
                            'genre': track.get('genre_clean', ''),
                            'match_score': round(avg_match * 100, 2),
                            'popularity': track.get('normalized_popularity', 0),
                            'preferred_platform': profile['preferred_platform']
                        })
        
        if demographic_data:
            demo_df = pd.DataFrame(demographic_data)
            demo_file = os.path.join(self.output_dir, 'demographic_preferences.csv')
            demo_df.to_csv(demo_file, index=False)
            print(f"Demographic preferences dataset created: {len(demo_df):,} rows")
            return len(demo_df)
        
        return 0
    
    def create_trend_analysis(self):
        """Create trend analysis dataset"""
        print("Creating trend_analysis.csv...")
        
        trend_data = []
        
        # Genre trends
        if 'genre_clean' in self.df.columns:
            genre_stats = self.df.groupby('genre_clean').agg({
                'normalized_popularity': 'mean',
                'data_quality_score': 'mean',
                'platform_count': 'mean',
                'danceability': 'mean',
                'energy': 'mean',
                'valence': 'mean',
                'acousticness': 'mean'
            }).reset_index()
            
            genre_stats['category'] = 'genre'
            genre_stats['item'] = genre_stats['genre_clean']
            genre_stats['track_count'] = self.df['genre_clean'].value_counts().values
            
            # Calculate growth potential (simple version)
            genre_stats['growth_potential'] = (
                (genre_stats['data_quality_score'] - genre_stats['normalized_popularity']) * 0.7 +
                genre_stats['platform_count'] * 10
            ).clip(0, 100)
            
            # Filter to genres with enough tracks
            genre_stats = genre_stats[genre_stats['track_count'] >= 10]
            
            trend_data.extend(genre_stats.to_dict('records'))
        
        # Platform trends
        platforms = ['spotify', 'tiktok', 'youtube']
        for platform in platforms:
            if platform in self.df.columns:
                platform_tracks = self.df[self.df[platform] == True]
                
                if len(platform_tracks) > 0:
                    trend_record = {
                        'category': 'platform',
                        'item': platform,
                        'track_count': len(platform_tracks),
                        'normalized_popularity': platform_tracks['normalized_popularity'].mean(),
                        'data_quality_score': platform_tracks['data_quality_score'].mean(),
                        'platform_count': platform_tracks['platform_count'].mean(),
                        'danceability': platform_tracks['danceability'].mean(),
                        'energy': platform_tracks['energy'].mean(),
                        'valence': platform_tracks['valence'].mean(),
                        'acousticness': platform_tracks['acousticness'].mean(),
                        'growth_potential': 50  # Default value
                    }
                    trend_data.append(trend_record)
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            trend_file = os.path.join(self.output_dir, 'trend_analysis.csv')
            trend_df.to_csv(trend_file, index=False)
            print(f"Trend analysis dataset created: {len(trend_df):,} rows")
            return len(trend_df)
        
        return 0
    
    def create_summary_report(self, dataset_counts):
        """Create a summary report"""
        print("Creating summary report...")
        
        report = {
            'processing_date': datetime.now().isoformat(),
            'input_dataset': self.input_dataset,
            'output_directory': self.output_dir,
            'total_tracks_processed': len(self.df),
            'datasets_created': dataset_counts,
            'data_summary': {
                'unique_artists': self.df['artist_name_clean'].nunique(),
                'unique_genres': self.df['genre_clean'].nunique(),
                'avg_popularity': self.df['normalized_popularity'].mean(),
                'avg_quality': self.df['data_quality_score'].mean(),
                'platform_coverage': {
                    'spotify': int(self.df['spotify'].sum()) if 'spotify' in self.df.columns else 0,
                    'youtube': int(self.df['youtube'].sum()) if 'youtube' in self.df.columns else 0,
                    'tiktok': int(self.df['tiktok'].sum()) if 'tiktok' in self.df.columns else 0
                }
            }
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, 'preprocessing_summary.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Summary report saved: preprocessing_summary.json")
        return report
    
    def run_pipeline(self):
        """Run the complete simplified pipeline"""
        print("🔧 SIMPLIFIED PREPROCESSING PIPELINE")
        print("=" * 50)
        
        # Step 1: Load and clean data
        if not self.load_and_clean_data():
            return None
        
        # Step 2: Create derived features
        self.create_derived_features()
        
        # Step 3: Create all datasets
        dataset_counts = {}
        
        dataset_counts['master_music_data'] = self.create_master_dataset()
        dataset_counts['platform_performance'] = self.create_platform_performance()
        dataset_counts['demographic_preferences'] = self.create_demographic_preferences()
        dataset_counts['trend_analysis'] = self.create_trend_analysis()
        
        # Step 4: Create summary report
        report = self.create_summary_report(dataset_counts)
        
        print("\nPreprocessing Pipeline Complete")
        print(f"Results:")
        print(f"   • Total tracks processed: {len(self.df):,}")
        print(f"   • Datasets created: {len([k for k, v in dataset_counts.items() if v > 0])}/4")
        
        print(f"\nDatasets Created:")
        for dataset_name, count in dataset_counts.items():
            if count > 0:
                status = "Pass"
                filename = dataset_name + '.csv'
                print(f"   {status} {filename}: {count:,} rows")
            else:
                print(f"   {dataset_name}.csv: No data generated")
        
        print(f"\nData Quality:")
        summary = report['data_summary']
        print(f"   • Unique artists: {summary['unique_artists']:,}")
        print(f"   • Unique genres: {summary['unique_genres']:,}")
        print(f"   • Average popularity: {summary['avg_popularity']:.1f}/100")
        print(f"   • Average quality: {summary['avg_quality']:.1f}/100")
        
        print(f"\nFiles saved to: {self.output_dir}/")
        print(f"Ready for Advanced Analytics!")
        
        return report

if __name__ == "__main__":
    import sys
    
    input_file = 'integration_cache/master_music_dataset_deduplicated.csv'
    output_dir = 'final_datasets'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Run simplified pipeline
    pipeline = SimplePreprocessingPipeline(input_file, output_dir)
    results = pipeline.run_pipeline()
    
    if results:
        print(f"\nSuccess, all datasets created successfully")
        print(f"Your music analytics platform is ready")
    else:
        print(f"\nPipeline failed. Check the error messages above.")