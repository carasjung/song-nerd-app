# marketing_insights_features.py

# This script is used to generate marketing insights for a given song.
# It uses the audio features of the song to generate insights on the song's target demographic,
# platform recommendations, and marketing suggestions.
# It also uses the audio features of the song to generate insights on the song's trend alignment.
# similar artists, viral potential and overall trend alignment.
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import json
import os
from datetime import datetime

class MarketingInsightsGenerator:
    def __init__(self, data_path='cleaned_music'):
        self.data_path = data_path
        self.audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                              'speechiness', 'acousticness', 'instrumentalness', 
                              'liveness', 'valence', 'tempo']
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.combined_df = None
        
    def load_data(self):
        """Load all necessary datasets"""
        print("Loading datasets...")
        
        try:
            self.spotify_tracks_df = pd.read_csv(
                os.path.join(self.data_path, 'spotify_tracks_clean.csv')
            )
            print(f"Loaded spotify_tracks_clean.csv: {len(self.spotify_tracks_df)} rows")
        except FileNotFoundError:
            print("spotify_tracks_clean.csv not found, skipping...")
            self.spotify_tracks_df = pd.DataFrame()
        
        try:
            self.music_info_df = pd.read_csv(
                os.path.join(self.data_path, 'music_info_clean.csv')
            )
            print(f"Loaded music_info_clean.csv: {len(self.music_info_df)} rows")
        except FileNotFoundError:
            print("music_info_clean.csv not found, skipping...")
            self.music_info_df = pd.DataFrame()
        
        # Load normalized datasets
        try:
            self.spotify_ds_df = pd.read_csv('cleaned_normalized/spotify_ds_normalized.csv')
            print(f"Loaded spotify_ds_normalized.csv: {len(self.spotify_ds_df)} rows")
        except FileNotFoundError:
            print("spotify_ds_normalized.csv not found, skipping...")
            self.spotify_ds_df = pd.DataFrame()
        
        try:
            self.tiktok_2021_df = pd.read_csv('cleaned_normalized/tiktok_2021_normalized.csv')
            print(f"Loaded tiktok_2021_normalized.csv: {len(self.tiktok_2021_df)} rows")
        except FileNotFoundError:
            print("tiktok_2021_normalized.csv not found, skipping...")
            self.tiktok_2021_df = pd.DataFrame()
        
        try:
            self.tiktok_2022_df = pd.read_csv('cleaned_normalized/tiktok_2022_normalized.csv')
            print(f"Loaded tiktok_2022_normalized.csv: {len(self.tiktok_2022_df)} rows")
        except FileNotFoundError:
            print("tiktok_2022_normalized.csv not found, skipping...")
            self.tiktok_2022_df = pd.DataFrame()
        
        try:
            self.spotify_yt_df = pd.read_csv('cleaned_normalized/spotify_yt_normalized.csv')
            print(f"Loaded spotify_yt_normalized.csv: {len(self.spotify_yt_df)} rows")
        except FileNotFoundError:
            print("spotify_yt_normalized.csv not found, skipping...")
            self.spotify_yt_df = pd.DataFrame()
        
        # Create a combined dataset for analysis
        self.combined_df = self._create_combined_dataset()
        
        print("Data loaded successfully!")
    
    def _create_combined_dataset(self):
        """Create a combined dataset from all sources for analysis"""
        print("Creating combined dataset...")
        
        datasets = []
        
        # Process Spotify tracks (main dataset with rich features)
        if not self.spotify_tracks_df.empty:
            spotify_std = self.spotify_tracks_df.copy()
            spotify_std['source_platform'] = 'spotify'
            spotify_std['platform_popularity'] = spotify_std.get('popularity', 0)
            spotify_std['artists'] = spotify_std.get('artists', spotify_std.get('artist', ''))
            
            # Select common columns
            common_cols = ['track_name', 'artists', 'platform_popularity', 'source_platform']
            audio_cols = [col for col in self.audio_features if col in spotify_std.columns]
            
            if 'track_genre' in spotify_std.columns:
                common_cols.append('track_genre')
            else:
                spotify_std['track_genre'] = 'unknown'
                common_cols.append('track_genre')
            
            datasets.append(spotify_std[common_cols + audio_cols])
        
        # Process Spotify DS dataset
        if not self.spotify_ds_df.empty:
            spotify_ds_std = self.spotify_ds_df.copy()
            spotify_ds_std['source_platform'] = 'spotify_ds'
            spotify_ds_std['platform_popularity'] = spotify_ds_std.get('popularity', 0)
            
            # Select common columns
            common_cols = ['track_name', 'artists', 'platform_popularity', 'source_platform']
            audio_cols = [col for col in self.audio_features if col in spotify_ds_std.columns]
            
            if 'track_genre' in spotify_ds_std.columns:
                common_cols.append('track_genre')
            else:
                spotify_ds_std['track_genre'] = 'unknown'
                common_cols.append('track_genre')
            
            datasets.append(spotify_ds_std[common_cols + audio_cols])
        
        # Process TikTok 2021
        if not self.tiktok_2021_df.empty:
            tiktok21_std = self.tiktok_2021_df.copy()
            tiktok21_std['source_platform'] = 'tiktok_2021'
            tiktok21_std['platform_popularity'] = tiktok21_std.get('track_pop', 0)
            tiktok21_std['track_genre'] = 'unknown'
            tiktok21_std['artists'] = tiktok21_std.get('artist_name', '')
            
            common_cols = ['track_name', 'artists', 'track_genre', 'platform_popularity', 'source_platform']
            audio_cols = [col for col in self.audio_features if col in tiktok21_std.columns]
            
            datasets.append(tiktok21_std[common_cols + audio_cols])
        
        # Process TikTok 2022
        if not self.tiktok_2022_df.empty:
            tiktok22_std = self.tiktok_2022_df.copy()
            tiktok22_std['source_platform'] = 'tiktok_2022'
            tiktok22_std['platform_popularity'] = tiktok22_std.get('track_pop', 0)
            tiktok22_std['track_genre'] = 'unknown'
            tiktok22_std['artists'] = tiktok22_std.get('artist_name', '')
            
            common_cols = ['track_name', 'artists', 'track_genre', 'platform_popularity', 'source_platform']
            audio_cols = [col for col in self.audio_features if col in tiktok22_std.columns]
            
            datasets.append(tiktok22_std[common_cols + audio_cols])
        
        # Process Spotify-YouTube dataset
        if not self.spotify_yt_df.empty:
            spotify_yt_std = self.spotify_yt_df.copy()
            spotify_yt_std['source_platform'] = 'youtube'
            spotify_yt_std['platform_popularity'] = pd.to_numeric(spotify_yt_std.get('views', 0), errors='coerce').fillna(0)
            spotify_yt_std['track_genre'] = 'unknown'
            spotify_yt_std['track_name'] = spotify_yt_std.get('track', spotify_yt_std.get('title', ''))
            spotify_yt_std['artists'] = spotify_yt_std.get('artist', '')
            
            common_cols = ['track_name', 'artists', 'track_genre', 'platform_popularity', 'source_platform']
            audio_cols = [col for col in self.audio_features if col in spotify_yt_std.columns]
            
            datasets.append(spotify_yt_std[common_cols + audio_cols])
        
        # Process Music Info dataset
        if not self.music_info_df.empty:
            music_info_std = self.music_info_df.copy()
            music_info_std['source_platform'] = 'music_info'
            music_info_std['platform_popularity'] = 50  # Default value
            music_info_std['track_name'] = music_info_std.get('name', '')
            music_info_std['artists'] = music_info_std.get('artist', '')
            music_info_std['track_genre'] = music_info_std.get('genre', 'unknown')
            
            common_cols = ['track_name', 'artists', 'track_genre', 'platform_popularity', 'source_platform']
            audio_cols = [col for col in self.audio_features if col in music_info_std.columns]
            
            datasets.append(music_info_std[common_cols + audio_cols])
        
        if not datasets:
            print("No datasets found! Please check your file paths.")
            return pd.DataFrame()
        
        # Combine all datasets
        combined = pd.concat(datasets, ignore_index=True, sort=False)
        
        # Fill missing values
        for feature in self.audio_features:
            if feature in combined.columns:
                combined[feature] = pd.to_numeric(combined[feature], errors='coerce')
                combined[feature] = combined[feature].fillna(combined[feature].median())
        
        combined['track_name'] = combined['track_name'].fillna('Unknown Track')
        combined['artists'] = combined['artists'].fillna('Unknown Artist')
        combined['track_genre'] = combined['track_genre'].fillna('unknown')
        combined['platform_popularity'] = pd.to_numeric(combined['platform_popularity'], errors='coerce').fillna(0)
        
        print(f"Combined dataset created with {len(combined)} tracks from {combined['source_platform'].nunique()} platforms")
        return combined
    
    def calculate_target_demographic_score(self):
        """Calculate likelihood scores for different age groups"""
        print("Calculating target demographic scores...")
        
        if self.combined_df is None or self.combined_df.empty:
            print("No data available for demographic scoring")
            return pd.DataFrame()
        
        # Define age group audio feature preferences based on research
        age_preferences = {
            'Gen Z (16-24)': {
                'danceability': 0.8, 'energy': 0.75, 'valence': 0.7,
                'speechiness': 0.3, 'acousticness': 0.2, 'tempo': 120
            },
            'Millennials (25-40)': {
                'danceability': 0.65, 'energy': 0.6, 'valence': 0.6,
                'speechiness': 0.15, 'acousticness': 0.4, 'tempo': 110
            },
            'Gen X (41-56)': {
                'danceability': 0.5, 'energy': 0.55, 'valence': 0.5,
                'speechiness': 0.1, 'acousticness': 0.6, 'tempo': 100
            },
            'Boomers (57+)': {
                'danceability': 0.4, 'energy': 0.4, 'valence': 0.45,
                'speechiness': 0.05, 'acousticness': 0.7, 'tempo': 90
            }
        }
        
        demographic_scores = []
        
        for _, song in self.combined_df.iterrows():
            song_scores = {
                'track_name': song.get('track_name', ''), 
                'artists': song.get('artists', ''), 
                'source_platform': song.get('source_platform', '')
            }
            
            for age_group, preferences in age_preferences.items():
                score = 0
                feature_count = 0
                
                for feature, ideal_value in preferences.items():
                    if feature in song and pd.notna(song[feature]):
                        if feature == 'tempo':
                            # Tempo similarity (closer to ideal = higher score)
                            tempo_diff = abs(song[feature] - ideal_value)
                            feature_score = max(0, 1 - (tempo_diff / 100))  # Normalize tempo difference
                        else:
                            # Other features: calculate similarity to ideal
                            feature_score = 1 - abs(song[feature] - ideal_value)
                        
                        score += max(0, feature_score)
                        feature_count += 1
                
                # Average score for this age group
                final_score = (score / feature_count) * 100 if feature_count > 0 else 0
                song_scores[f'{age_group}_score'] = round(final_score, 2)
            
            demographic_scores.append(song_scores)
        
        self.demographic_scores_df = pd.DataFrame(demographic_scores)
        return self.demographic_scores_df
    
    def calculate_platform_recommendation_score(self):
        """Calculate best platform scores for promotion"""
        print("Calculating platform recommendation scores...")
        
        if self.combined_df is None or self.combined_df.empty:
            print("No data available for platform scoring")
            return pd.DataFrame()
        
        # Platform-specific feature preferences
        platform_preferences = {
            'TikTok': {
                'danceability': 0.8, 'energy': 0.85, 'valence': 0.75,
                'speechiness': 0.4, 'tempo': 130, 'weight': 1.2
            },
            'Instagram': {
                'danceability': 0.7, 'energy': 0.7, 'valence': 0.65,
                'acousticness': 0.3, 'tempo': 115, 'weight': 1.0
            },
            'Spotify': {
                'danceability': 0.6, 'energy': 0.6, 'valence': 0.55,
                'acousticness': 0.5, 'instrumentalness': 0.2, 'weight': 1.1
            },
            'YouTube': {
                'danceability': 0.55, 'energy': 0.65, 'valence': 0.6,
                'acousticness': 0.4, 'liveness': 0.3, 'weight': 0.9
            }
        }
        
        platform_scores = []
        
        for _, song in self.combined_df.iterrows():
            song_scores = {
                'track_name': song.get('track_name', ''), 
                'artists': song.get('artists', ''),
                'source_platform': song.get('source_platform', '')
            }
            
            for platform, preferences in platform_preferences.items():
                score = 0
                feature_count = 0
                
                for feature, ideal_value in preferences.items():
                    if feature in ['weight']:
                        continue
                    
                    if feature in song and pd.notna(song[feature]):
                        if feature == 'tempo':
                            tempo_diff = abs(song[feature] - ideal_value)
                            feature_score = max(0, 1 - (tempo_diff / 120))
                        else:
                            feature_score = 1 - abs(song[feature] - ideal_value)
                        
                        score += max(0, feature_score)
                        feature_count += 1
                
                # Apply platform weight and normalize
                platform_weight = preferences.get('weight', 1.0)
                final_score = (score / feature_count) * platform_weight * 100 if feature_count > 0 else 0
                song_scores[f'{platform}_score'] = round(final_score, 2)
            
            platform_scores.append(song_scores)
        
        self.platform_scores_df = pd.DataFrame(platform_scores)
        return self.platform_scores_df
    
    def create_similar_artist_mapping(self, top_k=5):
        """Create similar artist mappings based on audio features and popularity"""
        print("Creating similar artist mappings...")
        
        if self.combined_df is None or self.combined_df.empty:
            print("No data available for similarity mapping")
            return pd.DataFrame()
        
        # Prepare feature matrix
        feature_columns = [col for col in self.audio_features if col in self.combined_df.columns]
        if not feature_columns:
            print("No audio features found for similarity calculation")
            return pd.DataFrame()
        
        feature_matrix = self.combined_df[feature_columns].fillna(0)
        
        # Normalize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(feature_matrix_scaled)
        
        similar_artists = []
        
        for i, song in self.combined_df.iterrows():
            song_similarities = similarity_matrix[i]
            
            # Get top similar songs (excluding the song itself)
            similar_indices = np.argsort(song_similarities)[::-1][1:top_k+1]
            
            similar_songs = []
            for idx in similar_indices:
                if idx < len(self.combined_df):
                    similar_song = self.combined_df.iloc[idx]
                    similar_songs.append({
                        'artist': similar_song.get('artists', ''),
                        'track': similar_song.get('track_name', ''),
                        'platform': similar_song.get('source_platform', ''),
                        'similarity_score': round(song_similarities[idx], 3)
                    })
            
            similar_artists.append({
                'track_name': song.get('track_name', ''),
                'artists': song.get('artists', ''),
                'source_platform': song.get('source_platform', ''),
                'similar_artists': similar_songs
            })
        
        self.similar_artists_df = pd.DataFrame(similar_artists)
        return self.similar_artists_df
    
    def calculate_trend_alignment_score(self):
        """Calculate how well songs match current trends"""
        print("Calculating trend alignment scores...")
        
        if self.combined_df is None or self.combined_df.empty:
            print("No data available for trend scoring")
            return pd.DataFrame()
        
        # Define current trend characteristics (based on 2024 music trends)
        current_trends = {
            'viral_potential': {
                'danceability': 0.75, 'energy': 0.8, 'valence': 0.7,
                'speechiness': 0.25, 'tempo': 125, 'weight': 1.3
            },
            'playlist_friendly': {
                'danceability': 0.65, 'energy': 0.6, 'valence': 0.6,
                'acousticness': 0.3, 'instrumentalness': 0.1, 'weight': 1.1
            },
            'social_media_ready': {
                'danceability': 0.8, 'energy': 0.85, 'speechiness': 0.35,
                'tempo': 130, 'valence': 0.75, 'weight': 1.2
            }
        }
        
        trend_scores = []
        
        for _, song in self.combined_df.iterrows():
            song_scores = {
                'track_name': song.get('track_name', ''), 
                'artists': song.get('artists', ''),
                'source_platform': song.get('source_platform', '')
            }
            
            overall_trend_score = 0
            trend_count = 0
            
            for trend_type, characteristics in current_trends.items():
                score = 0
                feature_count = 0
                
                for feature, ideal_value in characteristics.items():
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
                
                if feature_count > 0:
                    trend_weight = characteristics.get('weight', 1.0)
                    trend_score = (score / feature_count) * trend_weight
                    song_scores[f'{trend_type}_score'] = round(trend_score * 100, 2)
                    overall_trend_score += trend_score
                    trend_count += 1
            
            # Calculate overall trend alignment
            song_scores['overall_trend_score'] = round((overall_trend_score / trend_count) * 100, 2) if trend_count > 0 else 0
            trend_scores.append(song_scores)
        
        self.trend_scores_df = pd.DataFrame(trend_scores)
        return self.trend_scores_df
    
    def generate_marketing_insights(self, song_features):
        """Generate marketing insights for a new song"""
        print("Generating marketing insights for uploaded song...")
        
        # Ensure all required features are present
        for feature in self.audio_features:
            if feature not in song_features:
                song_features[feature] = 0
        
        insights = {
            'target_demographics': {},
            'platform_recommendations': {},
            'similar_artists': [],
            'trend_alignment': {},
            'marketing_suggestions': []
        }
        
        # Calculate demographic scores
        age_preferences = {
            'Gen Z (16-24)': {'danceability': 0.8, 'energy': 0.75, 'valence': 0.7, 'speechiness': 0.3, 'acousticness': 0.2, 'tempo': 120},
            'Millennials (25-40)': {'danceability': 0.65, 'energy': 0.6, 'valence': 0.6, 'speechiness': 0.15, 'acousticness': 0.4, 'tempo': 110},
            'Gen X (41-56)': {'danceability': 0.5, 'energy': 0.55, 'valence': 0.5, 'speechiness': 0.1, 'acousticness': 0.6, 'tempo': 100},
            'Boomers (57+)': {'danceability': 0.4, 'energy': 0.4, 'valence': 0.45, 'speechiness': 0.05, 'acousticness': 0.7, 'tempo': 90}
        }
        
        for age_group, preferences in age_preferences.items():
            score = 0
            feature_count = 0
            
            for feature, ideal_value in preferences.items():
                if feature in song_features:
                    if feature == 'tempo':
                        tempo_diff = abs(song_features[feature] - ideal_value)
                        feature_score = max(0, 1 - (tempo_diff / 100))
                    else:
                        feature_score = 1 - abs(song_features[feature] - ideal_value)
                    
                    score += max(0, feature_score)
                    feature_count += 1
            
            final_score = (score / feature_count) * 100 if feature_count > 0 else 0
            insights['target_demographics'][age_group] = round(final_score, 2)
        
        # Calculate platform scores
        platform_preferences = {
            'TikTok': {'danceability': 0.8, 'energy': 0.85, 'valence': 0.75, 'speechiness': 0.4, 'tempo': 130, 'weight': 1.2},
            'Instagram': {'danceability': 0.7, 'energy': 0.7, 'valence': 0.65, 'acousticness': 0.3, 'tempo': 115, 'weight': 1.0},
            'Spotify': {'danceability': 0.6, 'energy': 0.6, 'valence': 0.55, 'acousticness': 0.5, 'instrumentalness': 0.2, 'weight': 1.1},
            'YouTube': {'danceability': 0.55, 'energy': 0.65, 'valence': 0.6, 'acousticness': 0.4, 'liveness': 0.3, 'weight': 0.9}
        }
        
        for platform, preferences in platform_preferences.items():
            score = 0
            feature_count = 0
            
            for feature, ideal_value in preferences.items():
                if feature in ['weight']:
                    continue
                
                if feature in song_features:
                    if feature == 'tempo':
                        tempo_diff = abs(song_features[feature] - ideal_value)
                        feature_score = max(0, 1 - (tempo_diff / 120))
                    else:
                        feature_score = 1 - abs(song_features[feature] - ideal_value)
                    
                    score += max(0, feature_score)
                    feature_count += 1
            
            platform_weight = preferences.get('weight', 1.0)
            final_score = (score / feature_count) * platform_weight * 100 if feature_count > 0 else 0
            insights['platform_recommendations'][platform] = round(final_score, 2)
        
        # Generate marketing suggestions based on scores
        best_demographic = max(insights['target_demographics'], key=insights['target_demographics'].get)
        best_platform = max(insights['platform_recommendations'], key=insights['platform_recommendations'].get)
        
        insights['marketing_suggestions'] = [
            f"Target {best_demographic} audience for maximum engagement",
            f"Focus promotion efforts on {best_platform} for best results",
            f"Leverage the song's {self._get_strongest_features(song_features)} in marketing campaigns"
        ]
        
        return insights
    
    def _get_strongest_features(self, song_features):
        """Identify the strongest audio features for marketing messaging"""
        feature_descriptions = {
            'danceability': 'danceability and groove',
            'energy': 'high energy and intensity',
            'valence': 'positive and uplifting vibe',
            'acousticness': 'acoustic and organic sound',
            'instrumentalness': 'instrumental focus',
            'speechiness': 'lyrical content and vocals'
        }
        
        strong_features = []
        for feature, value in song_features.items():
            if feature in feature_descriptions and value > 0.7:
                strong_features.append(feature_descriptions[feature])
        
        return ', '.join(strong_features) if strong_features else 'unique sonic characteristics'
    
    def save_results(self, output_dir='marketing_insights'):
        """Save all marketing insights to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save demographic scores
        if hasattr(self, 'demographic_scores_df') and not self.demographic_scores_df.empty:
            self.demographic_scores_df.to_csv(
                os.path.join(output_dir, 'target_demographic_scores.csv'), 
                index=False
            )
            print(f"Saved target_demographic_scores.csv with {len(self.demographic_scores_df)} rows")
        
        # Save platform scores
        if hasattr(self, 'platform_scores_df') and not self.platform_scores_df.empty:
            self.platform_scores_df.to_csv(
                os.path.join(output_dir, 'platform_recommendation_scores.csv'), 
                index=False
            )
            print(f"Saved platform_recommendation_scores.csv with {len(self.platform_scores_df)} rows")
        
        # Save similar artists
        if hasattr(self, 'similar_artists_df') and not self.similar_artists_df.empty:
            self.similar_artists_df.to_csv(
                os.path.join(output_dir, 'similar_artist_mappings.csv'), 
                index=False
            )
            print(f"Saved similar_artist_mappings.csv with {len(self.similar_artists_df)} rows")
        
        # Save trend scores
        if hasattr(self, 'trend_scores_df') and not self.trend_scores_df.empty:
            self.trend_scores_df.to_csv(
                os.path.join(output_dir, 'trend_alignment_scores.csv'), 
                index=False
            )
            print(f"Saved trend_alignment_scores.csv with {len(self.trend_scores_df)} rows")
        
        print(f"Marketing insights saved to {output_dir}/")
    
    def run_full_analysis(self):
        """Run complete marketing insights analysis"""
        print("Starting Marketing Insights Analysis...")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        if self.combined_df is None or self.combined_df.empty:
            print("No data loaded. Please check your file paths and data.")
            return None
        
        # Calculate all insights
        self.calculate_target_demographic_score()
        self.calculate_platform_recommendation_score()
        self.create_similar_artist_mapping()
        self.calculate_trend_alignment_score()
        
        # Save results
        self.save_results()
        
        print("\nMarketing Insights Analysis Complete!")
        print("=" * 50)
        
        results = {}
        if hasattr(self, 'demographic_scores_df'):
            results['demographic_scores'] = self.demographic_scores_df
        if hasattr(self, 'platform_scores_df'):
            results['platform_scores'] = self.platform_scores_df
        if hasattr(self, 'similar_artists_df'):
            results['similar_artists'] = self.similar_artists_df
        if hasattr(self, 'trend_scores_df'):
            results['trend_scores'] = self.trend_scores_df
            
        return results

# Example use
if __name__ == "__main__":
    # Initialize the generator (pointing to cleaned_music folder)
    generator = MarketingInsightsGenerator('cleaned_music')
    
    # Run full analysis on existing dataset
    results = generator.run_full_analysis()
    
    if results:
        print("\nAnalysis Results Summary:")
        for key, df in results.items():
            if not df.empty:
                print(f"- {key}: {len(df)} records")
        
        # Example: Generate insights for a new song
        print("\n" + "="*50)
        print("EXAMPLE: Marketing Insights for New Song")
        print("="*50)
        
        new_song_features = {
            'danceability': 0.8,
            'energy': 0.9,
            'key': 1,
            'loudness': -5.0,
            'mode': 1,
            'speechiness': 0.15,
            'acousticness': 0.1,
            'instrumentalness': 0.0,
            'liveness': 0.2,
            'valence': 0.85,
            'tempo': 128
        }
        
        # Generate marketing insights for the new song
        insights = generator.generate_marketing_insights(new_song_features)
        
        print(f"\nNew Song Features:")
        for feature, value in new_song_features.items():
            print(f"  {feature}: {value}")
        
        print(f"\nTarget Demographics (Scores):")
        for demo, score in insights['target_demographics'].items():
            print(f"  {demo}: {score}%")
        
        print(f"\nPlatform Recommendations (Scores):")
        for platform, score in insights['platform_recommendations'].items():
            print(f"  {platform}: {score}%")
        
        print(f"\nMarketing Suggestions:")
        for i, suggestion in enumerate(insights['marketing_suggestions'], 1):
            print(f"  {i}. {suggestion}")
        
        # Display top performing songs from analysis
        if 'demographic_scores' in results and not results['demographic_scores'].empty:
            print(f"\n" + "="*50)
            print("TOP PERFORMING SONGS BY DEMOGRAPHIC")
            print("="*50)
            
            demo_df = results['demographic_scores']
            demographic_columns = [col for col in demo_df.columns if '_score' in col]
            
            for demo_col in demographic_columns:
                demo_name = demo_col.replace('_score', '')
                top_songs = demo_df.nlargest(3, demo_col)
                
                print(f"\nTop 3 songs for {demo_name}:")
                for idx, song in top_songs.iterrows():
                    print(f"  {song['track_name']} by {song['artists']} - {song[demo_col]}%")
        
        # Display platform performance insights
        if 'platform_scores' in results and not results['platform_scores'].empty:
            print(f"\n" + "="*50)
            print("PLATFORM PERFORMANCE INSIGHTS")
            print("="*50)
            
            platform_df = results['platform_scores']
            platform_columns = [col for col in platform_df.columns if '_score' in col]
            
            for platform_col in platform_columns:
                platform_name = platform_col.replace('_score', '')
                top_songs = platform_df.nlargest(3, platform_col)
                
                print(f"\nTop 3 songs for {platform_name}:")
                for idx, song in top_songs.iterrows():
                    print(f"  {song['track_name']} by {song['artists']} - {song[platform_col]}%")
        
        # Display trend analysis
        if 'trend_scores' in results and not results['trend_scores'].empty:
            print(f"\n" + "="*50)
            print("TREND ALIGNMENT ANALYSIS")
            print("="*50)
            
            trend_df = results['trend_scores']
            top_viral = trend_df.nlargest(5, 'viral_potential_score')
            
            print(f"\nTop 5 songs with highest viral potential:")
            for idx, song in top_viral.iterrows():
                print(f"  {song['track_name']} by {song['artists']} - {song['viral_potential_score']}%")
            
            # Overall trend leaders
            if 'overall_trend_score' in trend_df.columns:
                top_trending = trend_df.nlargest(5, 'overall_trend_score')
                print(f"\nTop 5 songs with highest overall trend alignment:")
                for idx, song in top_trending.iterrows():
                    print(f"  {song['track_name']} by {song['artists']} - {song['overall_trend_score']}%")
        
        # Generate business recommendations
        # This is for initial testing 
        print(f"\n" + "="*50)
        print("BUSINESS RECOMMENDATIONS")
        print("="*50)
        
        print("\n1. CONTENT STRATEGY:")
        print("   - Focus on high-energy, danceable tracks for maximum engagement")
        print("   - Develop content around top-performing songs for each demographic")
        print("   - Create platform-specific versions of content")
        
        print("\n2. ARTIST DEVELOPMENT:")
        print("   - Guide artists toward features that align with their target audience")
        print("   - Use similar artist mappings to identify collaboration opportunities")
        print("   - Develop sound profiles based on demographic preferences")
        
        print("\n3. MARKETING CAMPAIGNS:")
        print("   - Allocate budget based on platform recommendation scores")
        print("   - Time releases to align with trend patterns")
        print("   - Customize messaging for each demographic segment")
        
        print("\n4. DATA-DRIVEN DECISIONS:")
        print("   - Monitor trend alignment scores for release timing")
        print("   - Use demographic scores for playlist placement strategies")
        print("   - Leverage platform scores for social media advertising")
        
    else:
        print("\nNo results generated. Please check your data files and file paths.")
        print("Expected files:")
        print("- cleaned_music/spotify_tracks_clean.csv")
        print("- cleaned_music/music_info_clean.csv")
        print("- cleaned_normalized/spotify_ds_normalized.csv")
        print("- cleaned_normalized/tiktok_2021_normalized.csv")
        print("- cleaned_normalized/tiktok_2022_normalized.csv")
        print("- cleaned_normalized/spotify_yt_normalized.csv")
    
    print(f"\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)