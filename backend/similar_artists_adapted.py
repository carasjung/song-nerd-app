# similar_artists_adapted.py 

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

class SimilarArtistFinder:
    def __init__(self):
        self.artist_profiles = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
    def build_artist_database(self, master_data):
        """Build artist profile database from master dataset"""
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
        
        # Group by artist and calculate profiles
        artist_profiles = master_data.groupby('artist_name_clean').agg({
            **{feature: 'mean' for feature in audio_features},
            'genre_clean': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
            'genre_category': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
            'normalized_popularity': 'mean',
            'audio_appeal': 'mean',
            'popularity_tier': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
            'primary_platform': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
            'track_name_clean': 'count',  # Track count
            'platform_count': 'mean',
            'is_multi_platform': lambda x: x.any()
        }).reset_index()
        
        # Rename count column
        artist_profiles.rename(columns={'track_name_clean': 'track_count'}, inplace=True)
        
        # Handle missing values
        for feature in audio_features:
            artist_profiles[feature] = artist_profiles[feature].fillna(
                artist_profiles[feature].median()
            )
        
        # Scale audio features for similarity calculation
        feature_matrix = artist_profiles[audio_features].values
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Apply PCA for dimensionality reduction
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Store processed data
        self.artist_profiles = artist_profiles.copy()
        self.artist_profiles['feature_vector'] = list(pca_features)
        
        print(f"Built artist database with {len(self.artist_profiles)} artists")
        print(f"Feature dimensions after PCA: {pca_features.shape[1]}")
        
        return self
    
    def find_similar_artists(self, input_features, top_k=10, same_tier_only=False, exclude_self=True):
        """Find similar artists based on audio features"""
        
        # Prepare input features
        audio_feature_names = ['danceability', 'energy', 'valence', 'acousticness', 
                              'instrumentalness', 'liveness', 'speechiness']
        
        input_vector = np.array([[
            input_features.get(feature, 0) 
            for feature in audio_feature_names
        ]])
        
        # Scale and transform input
        input_scaled = self.scaler.transform(input_vector)
        input_pca = self.pca.transform(input_scaled)
        
        # Calculate similarities
        artist_vectors = np.vstack(self.artist_profiles['feature_vector'].values)
        similarities = cosine_similarity(input_pca, artist_vectors)[0]
        
        # Add similarity scores to dataframe
        results = self.artist_profiles.copy()
        results['similarity_score'] = similarities
        
        # Filter by popularity tier if requested
        if same_tier_only and 'normalized_popularity' in input_features:
            input_popularity = input_features['normalized_popularity']
            
            # Map popularity to tier
            if input_popularity < 0.3:
                target_tier = 'emerging'
            elif input_popularity < 0.6:
                target_tier = 'growing'
            elif input_popularity < 0.8:
                target_tier = 'established'
            else:
                target_tier = 'superstar'
            
            # Filter to same tier
            if 'popularity_tier' in results.columns:
                results = results[results['popularity_tier'] == target_tier]
        
        # Exclude self if artist name provided
        if exclude_self and 'artist_name' in input_features:
            input_artist = input_features['artist_name']
            results = results[results['artist_name_clean'] != input_artist]
        
        # Sort by similarity and return top k
        top_similar = results.nlargest(top_k, 'similarity_score')
        
        # Format results
        similar_artists = []
        for _, artist in top_similar.iterrows():
            similar_artists.append({
                'artist_name': artist['artist_name_clean'],
                'similarity_score': float(artist['similarity_score']),
                'genre': artist['genre_clean'],
                'genre_category': artist['genre_category'],
                'popularity': float(artist['normalized_popularity']),
                'popularity_tier': artist['popularity_tier'],
                'audio_appeal': float(artist['audio_appeal']),
                'track_count': int(artist['track_count']),
                'primary_platform': artist['primary_platform'],
                'is_multi_platform': bool(artist['is_multi_platform']),
                'key_similarities': self._identify_key_similarities(
                    input_features, artist, audio_feature_names
                )
            })
        
        return {
            'similar_artists': similar_artists,
            'total_found': len(similar_artists),
            'search_criteria': 'audio_features_similarity',
            'filters_applied': {
                'same_tier_only': same_tier_only,
                'exclude_self': exclude_self
            }
        }
    
    def _identify_key_similarities(self, input_features, artist_row, audio_features):
        """Identify which features make artists similar"""
        similarities = []
        
        for feature in audio_features:
            if feature in input_features:
                input_val = input_features[feature]
                artist_val = artist_row[feature]
                
                # Calculate similarity (closer to 1 means more similar)
                sim = 1 - abs(input_val - artist_val)
                
                if sim > 0.8:  # High similarity threshold
                    similarities.append({
                        'feature': feature,
                        'similarity': float(sim),
                        'input_value': float(input_val),
                        'artist_value': float(artist_val),
                        'description': self._feature_description(feature, artist_val)
                    })
        
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:3]
    
    def _feature_description(self, feature, value):
        """Generate human-readable feature descriptions"""
        descriptions = {
            'danceability': f"{'High' if value > 0.7 else 'Moderate' if value > 0.4 else 'Low'} danceability ({value:.2f})",
            'energy': f"{'High' if value > 0.7 else 'Moderate' if value > 0.4 else 'Low'} energy level ({value:.2f})",
            'valence': f"{'Upbeat' if value > 0.6 else 'Neutral' if value > 0.4 else 'Melancholic'} mood ({value:.2f})",
            'acousticness': f"{'Acoustic' if value > 0.6 else 'Semi-acoustic' if value > 0.3 else 'Electronic'} sound ({value:.2f})",
            'instrumentalness': f"{'Mostly instrumental' if value > 0.6 else 'Some vocals' if value > 0.3 else 'Vocal-focused'} ({value:.2f})",
            'liveness': f"{'Live recording feel' if value > 0.6 else 'Studio quality'} ({value:.2f})",
            'speechiness': f"{'Speech-like' if value > 0.6 else 'Sung vocals'} ({value:.2f})"
        }
        return descriptions.get(feature, f"{feature}: {value:.2f}")
    
    def get_artist_insights(self, artist_name):
        """Get detailed insights about a specific artist"""
        if self.artist_profiles is None:
            return None
        
        artist_data = self.artist_profiles[
            self.artist_profiles['artist_name_clean'] == artist_name
        ]
        
        if artist_data.empty:
            return None
        
        artist = artist_data.iloc[0]
        
        return {
            'artist_name': artist['artist_name_clean'],
            'genre': artist['genre_clean'],
            'genre_category': artist['genre_category'],
            'popularity': float(artist['normalized_popularity']),
            'popularity_tier': artist['popularity_tier'],
            'audio_profile': {
                'danceability': float(artist['danceability']),
                'energy': float(artist['energy']),
                'valence': float(artist['valence']),
                'acousticness': float(artist['acousticness']),
                'instrumentalness': float(artist['instrumentalness']),
                'liveness': float(artist['liveness']),
                'speechiness': float(artist['speechiness'])
            },
            'career_metrics': {
                'track_count': int(artist['track_count']),
                'audio_appeal': float(artist['audio_appeal']),
                'platform_count': float(artist['platform_count']),
                'is_multi_platform': bool(artist['is_multi_platform']),
                'primary_platform': artist['primary_platform']
            }
        }
    
    def save_model(self, filepath):
        """Save artist database and models"""
        model_data = {
            'artist_profiles': self.artist_profiles,
            'scaler': self.scaler,
            'pca': self.pca
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load artist database and models"""
        model_data = joblib.load(filepath)
        self.artist_profiles = model_data['artist_profiles']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        return self

# Build and save similar artists database
def build_similar_artists_database():
    master_data = pd.read_csv('final_datasets/master_music_data.csv')
    
    # Initialize and build database
    similar_artists = SimilarArtistFinder()
    similar_artists.build_artist_database(master_data)
    
    # Save database
    similar_artists.save_model('models/similar_artists.pkl')
    print("Similar artists database saved!")
    
    # Test with a sample
    if len(master_data) > 0:
        sample_song = master_data.iloc[0]
        test_features = {
            'danceability': sample_song['danceability'],
            'energy': sample_song['energy'],
            'valence': sample_song['valence'],
            'acousticness': sample_song['acousticness'],
            'instrumentalness': sample_song['instrumentalness'],
            'liveness': sample_song['liveness'],
            'speechiness': sample_song['speechiness'],
            'artist_name': sample_song['artist_name_clean']
        }
        
        results = similar_artists.find_similar_artists(test_features, top_k=5)
        print(f"\nTest: Similar artists to {sample_song['artist_name_clean']}:")
        for artist in results['similar_artists'][:3]:
            print(f"- {artist['artist_name']} (similarity: {artist['similarity_score']:.3f})")
    
    return similar_artists

if __name__ == "__main__":
    model = build_similar_artists_database()