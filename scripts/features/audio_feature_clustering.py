# audio_feature_clustering.py
''' 
- Group songs by audio feature similarity
- Create sound profiles for different demographics
- Identify platform-specfic audio preferences

Run clustering analysis: python3 scripts/features/hybrid_audio_feature_clustering.py analyze
Predict cluster for a song: python3 scripts/features/hybrid_audio_feature_clustering.py predict
Generate visualizations: python3 scripts/features/hybrid_audio_feature_clustering.py visualize
Run integrated analysis: python3 scripts/features/hybrid_audio_feature_clustering.py integrated analyze
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import os
import logging
import sys
import argparse

logger = logging.getLogger(__name__)

# Utility to convert numpy types to standard Python types for JSON serialization

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {
            convert_numpy_types(k): convert_numpy_types(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

class AudioFeatureClusteringEngine:
    def __init__(
        self,
        data_path='cleaned_music',
        cache_dir='clustering_cache',
        memory_mode='auto'
    ):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.memory_mode = memory_mode
        
        self.audio_features = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness'
        ]
        
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        
        # Clustering components
        self.kmeans_model = None
        self.pca_model = None
        self.cluster_labels = None
        self.feature_matrix = None
        
        # Results storage
        self.sound_profiles = {}
        self.demographic_clusters = {}
        self.platform_preferences = {}
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_combined_data(self):
        """Load the combined dataset from marketing insights cache"""
        marketing_cache = 'marketing_cache'
        
        # Try to load from marketing insights cache first
        cache_files = [
            os.path.join(marketing_cache, 'combined_data_full.pkl'),
            os.path.join(marketing_cache, 'combined_data_medium.pkl'),
            os.path.join(marketing_cache, 'combined_data_optimized.pkl')
        ]
        
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                logger.info(f"Loading combined data from {cache_file}")
                with open(cache_file, 'rb') as f:
                    self.combined_df = pickle.load(f)
                logger.info(f"Loaded {len(self.combined_df)} songs for clustering")
                return True
        
        logger.error("No combined data found. Run marketing insights analysis first.")
        return False
    
    def prepare_clustering_data(self):
        """Prepare data for clustering analysis"""
        if not hasattr(self, 'combined_df') or self.combined_df is None:
            if not self.load_combined_data():
                return False
        
        logger.info("Preparing data for clustering...")
        
        # Filter to only include songs with audio features
        available_features = [f for f in self.audio_features if f in self.combined_df.columns]
        
        if len(available_features) < 5:
            logger.error(f"Insufficient audio features. Found: {available_features}")
            return False
        
        # Create feature matrix
        self.feature_matrix = self.combined_df[available_features].copy()
        
        # Handle missing values
        self.feature_matrix = self.feature_matrix.fillna(self.feature_matrix.median())
        
        # Normalize tempo to 0-1 scale for better clustering
        if 'tempo' in self.feature_matrix.columns:
            self.feature_matrix['tempo'] = self.min_max_scaler.fit_transform(
                self.feature_matrix[['tempo']]
            ).astype(np.float64)
        
        # Normalize loudness (typically negative values)
        if 'loudness' in self.feature_matrix.columns:
            self.feature_matrix['loudness'] = self.min_max_scaler.fit_transform(
                self.feature_matrix[['loudness']]
            ).astype(np.float64)
        
        # Set feature matrix to float64 for sklearn compatibility (post normalization)
        self.feature_matrix = self.feature_matrix.astype(np.float64)
        
        # Standardize all features
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        logger.info(f"Prepared clustering data: {self.feature_matrix.shape}")
        return True
    
    def find_optimal_clusters(self, max_clusters=12, method='elbow'):
        """Find optimal number of clusters using elbow method or silhouette analysis"""
        logger.info(f"Finding optimal number of clusters using {method} method...")
        
        if self.feature_matrix_scaled is None:
            logger.error("No prepared data for clustering")
            return None
        
        # Memory-aware cluster range
        max_samples = len(self.feature_matrix_scaled)
        if max_samples > 50000:
            max_clusters = min(max_clusters, 8)  # Reduce for large datasets
        
        cluster_range = range(2, max_clusters + 1)
        scores = []
        
        for n_clusters in cluster_range:
            logger.info(f"Testing {n_clusters} clusters...")
            
            # Use MiniBatchKMeans for large datasets
            if max_samples > 10000:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    batch_size=1000,
                    n_init=3
                )
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            cluster_labels = kmeans.fit_predict(self.feature_matrix_scaled)
            
            if method == 'elbow':
                # Use inertia (within-cluster sum of squares)
                scores.append(kmeans.inertia_)
            elif method == 'silhouette':
                # Use silhouette score (can be memory intensive)
                if max_samples < 10000:  # Only for smaller datasets
                    score = silhouette_score(self.feature_matrix_scaled, cluster_labels)
                    scores.append(score)
                else:
                    scores.append(0)  # Skip for large datasets
        
        # Find optimal number of clusters
        if method == 'elbow':
            # Look for the "elbow" in the curve
            if len(scores) >= 3:
                # Simple elbow detection: find the point where improvement slows down
                improvements = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
                elbow_idx = np.argmax(np.diff(improvements)) + 2  # +2 because we start from 2 clusters
                optimal_clusters = min(elbow_idx + 2, max_clusters)
            else:
                optimal_clusters = 6  # Default fallback
        
        elif method == 'silhouette':
            optimal_clusters = cluster_range[np.argmax(scores)] if scores else 6
        
        logger.info(f"Optimal number of clusters: {optimal_clusters}")
        
        # Save analysis results
        analysis_results = {
            'method': method,
            'cluster_range': list(cluster_range),
            'scores': scores,
            'optimal_clusters': optimal_clusters
        }
        
        cache_file = os.path.join(self.cache_dir, 'cluster_analysis.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(analysis_results, f)
        
        return optimal_clusters
    
    def perform_clustering(self, n_clusters=None):
        """Perform the actual clustering"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        logger.info(f"Performing clustering with {n_clusters} clusters...")
        
        # Choose clustering algo based on data size
        max_samples = len(self.feature_matrix_scaled)
        
        if max_samples > 10000:
            self.kmeans_model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=min(1000, max_samples // 10),
                n_init=3
            )
        else:
            self.kmeans_model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
        
        # Fit the model
        self.cluster_labels = self.kmeans_model.fit_predict(self.feature_matrix_scaled)
        
        # Add cluster labels to original dataframe
        self.combined_df['audio_cluster'] = self.cluster_labels
        
        logger.info(f"Clustering complete. Silhouette score: {self._calculate_silhouette_score():.3f}")
        
        # Cache clustering results
        self._cache_clustering_results()
        
        return self.cluster_labels
    
    def _calculate_silhouette_score(self):
        """Calculate silhouette score for clustering quality"""
        if len(self.feature_matrix_scaled) > 10000:
            # Sample for large datasets to avoid memory issues
            sample_size = 5000
            indices = np.random.choice(len(self.feature_matrix_scaled), sample_size, replace=False)
            sample_features = self.feature_matrix_scaled[indices]
            sample_labels = self.cluster_labels[indices]
            return silhouette_score(sample_features, sample_labels)
        else:
            return silhouette_score(self.feature_matrix_scaled, self.cluster_labels)
    
    def _cache_clustering_results(self):
        """Cache clustering results"""
        results = {
            'kmeans_model': self.kmeans_model,
            'cluster_labels': self.cluster_labels,
            'scaler': self.scaler,
            'feature_columns': list(self.feature_matrix.columns),
            'n_clusters': len(np.unique(self.cluster_labels))
        }
        
        cache_file = os.path.join(self.cache_dir, 'clustering_model.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
    
    def create_sound_profiles(self):
        """Create sound profiles for each cluster"""
        logger.info("Creating sound profiles for clusters...")
        
        if self.cluster_labels is None:
            logger.error("No clustering results found. Run perform_clustering() first.")
            return None
        
        unique_clusters = np.unique(self.cluster_labels)
        self.sound_profiles = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_data = self.combined_df[cluster_mask]
            cluster_features = self.feature_matrix[cluster_mask]
            
            # Calculate cluster characteristics
            profile = {
                'cluster_id': int(cluster_id),
                'size': int(np.sum(cluster_mask)),
                'percentage': round(np.sum(cluster_mask) / len(self.combined_df) * 100, 2),
                'audio_features': {},
                'characteristics': {},
                'representative_songs': [],
                'platforms': {},
                'genres': {}
            }
            
            # Audio feature averages
            for feature in self.feature_matrix.columns:
                profile['audio_features'][feature] = round(cluster_features[feature].mean(), 3)
            
            # Cluster characteristics based on audio features
            profile['characteristics'] = self._interpret_cluster_characteristics(profile['audio_features'])
            
            # Representative songs (closest to cluster center)
            cluster_center = self.kmeans_model.cluster_centers_[cluster_id]
            cluster_scaled = self.scaler.transform(cluster_features)
            
            # Find songs closest to center
            distances = np.linalg.norm(cluster_scaled - cluster_center, axis=1)
            closest_indices = np.argsort(distances)[:5]  # Top 5 songs
            
            for idx in closest_indices:
                song_idx = cluster_data.iloc[idx]
                profile['representative_songs'].append({
                    'track_name': song_idx.get('track_name', 'Unknown'),
                    'artists': song_idx.get('artists', 'Unknown'),
                    'platform': song_idx.get('source_platform', 'Unknown'),
                    'popularity': song_idx.get('platform_popularity', 0)
                })
            
            # Platform distribution
            if 'source_platform' in cluster_data.columns:
                platform_counts = cluster_data['source_platform'].value_counts()
                total_in_cluster = len(cluster_data)
                
                for platform, count in platform_counts.items():
                    profile['platforms'][platform] = {
                        'count': int(count),
                        'percentage': round(count / total_in_cluster * 100, 2)
                    }
            
            # Genre distribution
            if 'track_genre' in cluster_data.columns:
                genre_counts = cluster_data['track_genre'].value_counts()
                total_in_cluster = len(cluster_data)
                
                for genre, count in genre_counts.head(5).items():  # Top 5 genres
                    if genre != 'unknown':
                        profile['genres'][genre] = {
                            'count': int(count),
                            'percentage': round(count / total_in_cluster * 100, 2)
                        }
            
            self.sound_profiles[cluster_id] = profile
        
        # Cache sound profiles
        cache_file = os.path.join(self.cache_dir, 'sound_profiles.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(self.sound_profiles, f)
        
        return self.sound_profiles
    
    def _interpret_cluster_characteristics(self, audio_features):
        """Interpret cluster characteristics based on audio features"""
        characteristics = []
        
        # Danceability
        if audio_features.get('danceability', 0) > 0.7:
            characteristics.append('Highly Danceable')
        elif audio_features.get('danceability', 0) < 0.3:
            characteristics.append('Low Danceability')
        
        # Energy
        if audio_features.get('energy', 0) > 0.7:
            characteristics.append('High Energy')
        elif audio_features.get('energy', 0) < 0.3:
            characteristics.append('Low Energy')
        
        # Valence (positivity)
        if audio_features.get('valence', 0) > 0.7:
            characteristics.append('Positive/Happy')
        elif audio_features.get('valence', 0) < 0.3:
            characteristics.append('Negative/Sad')
        
        # Acousticness
        if audio_features.get('acousticness', 0) > 0.7:
            characteristics.append('Acoustic')
        elif audio_features.get('acousticness', 0) < 0.2:
            characteristics.append('Electronic/Produced')
        
        # Instrumentalness
        if audio_features.get('instrumentalness', 0) > 0.5:
            characteristics.append('Instrumental')
        elif audio_features.get('instrumentalness', 0) < 0.1:
            characteristics.append('Vocal-Heavy')
        
        # Speechiness
        if audio_features.get('speechiness', 0) > 0.33:
            characteristics.append('Speech-like')
        elif audio_features.get('speechiness', 0) < 0.1:
            characteristics.append('Musical')
        
        # Liveness
        if audio_features.get('liveness', 0) > 0.8:
            characteristics.append('Live Performance')
        
        # Tempo
        tempo = audio_features.get('tempo', 0.5)  # Normalized tempo
        if tempo > 0.8:
            characteristics.append('Fast Tempo')
        elif tempo < 0.3:
            characteristics.append('Slow Tempo')
        else:
            characteristics.append('Moderate Tempo')
        
        return characteristics
    
    def analyze_demographic_clusters(self):
        """Analyze which clusters appeal to different demographics"""
        logger.info("Analyzing demographic cluster preferences...")
        
        if not hasattr(self, 'sound_profiles') or not self.sound_profiles:
            logger.error("No sound profiles found. Run create_sound_profiles() first.")
            return None
        
        # Define demographic preferences (from marketing insights)
        demographic_preferences = {
            'Gen Z (16-24)': {
                'danceability': 0.8, 'energy': 0.75, 'valence': 0.7,
                'speechiness': 0.3, 'acousticness': 0.2
            },
            'Millennials (25-40)': {
                'danceability': 0.65, 'energy': 0.6, 'valence': 0.6,
                'speechiness': 0.15, 'acousticness': 0.4
            },
            'Gen X (41-56)': {
                'danceability': 0.5, 'energy': 0.55, 'valence': 0.5,
                'speechiness': 0.1, 'acousticness': 0.6
            },
            'Boomers (57+)': {
                'danceability': 0.4, 'energy': 0.4, 'valence': 0.45,
                'speechiness': 0.05, 'acousticness': 0.7
            }
        }
        
        self.demographic_clusters = {}
        
        for demographic, preferences in demographic_preferences.items():
            cluster_scores = {}
            
            for cluster_id, profile in self.sound_profiles.items():
                # Calculate similarity between cluster and demographic preferences
                similarity_score = 0
                feature_count = 0
                
                for feature, ideal_value in preferences.items():
                    if feature in profile['audio_features']:
                        cluster_value = profile['audio_features'][feature]
                        feature_similarity = 1 - abs(cluster_value - ideal_value)
                        similarity_score += feature_similarity
                        feature_count += 1
                
                if feature_count > 0:
                    avg_similarity = similarity_score / feature_count
                    cluster_scores[cluster_id] = round(avg_similarity * 100, 2)
            
            # Sort clusters by similarity score
            sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
            
            self.demographic_clusters[demographic] = {
                'cluster_rankings': sorted_clusters,
                'best_cluster': sorted_clusters[0] if sorted_clusters else None,
                'cluster_distribution': cluster_scores
            }
        
        # Cache demographic analysis
        cache_file = os.path.join(self.cache_dir, 'demographic_clusters.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(self.demographic_clusters, f)
        
        return self.demographic_clusters
    
    def analyze_platform_preferences(self):
        """Analyze platform-specific audio preferences by cluster"""
        logger.info("Analyzing platform-specific cluster preferences...")
        
        if self.cluster_labels is None:
            logger.error("No clustering results found.")
            return None
        
        platforms = self.combined_df['source_platform'].unique()
        self.platform_preferences = {}
        
        for platform in platforms:
            platform_data = self.combined_df[self.combined_df['source_platform'] == platform]
            
            if len(platform_data) < 10:  # Skip platforms with too few songs
                continue
            
            # Get cluster distribution for this platform
            cluster_counts = platform_data['audio_cluster'].value_counts()
            total_songs = len(platform_data)
            
            # Calculate cluster preferences
            cluster_distribution = {}
            for cluster_id, count in cluster_counts.items():
                percentage = (count / total_songs) * 100
                cluster_distribution[int(cluster_id)] = {
                    'count': int(count),
                    'percentage': round(percentage, 2)
                }
            
            # Calculate average audio features for successful songs on this platform
            # (top 25% by popularity)
            top_quartile = platform_data.nlargest(len(platform_data) // 4, 'platform_popularity')
            
            platform_audio_profile = {}
            for feature in self.feature_matrix.columns:
                if feature in top_quartile.columns:
                    platform_audio_profile[feature] = round(top_quartile[feature].mean(), 3)
            
            best_cluster = None
            best_score = 0
            
            for cluster_id, profile in self.sound_profiles.items():
                # Calculate similarity between platform preferences and cluster
                similarity = 0
                feature_count = 0
                
                for feature, platform_value in platform_audio_profile.items():
                    if feature in profile['audio_features']:
                        cluster_value = profile['audio_features'][feature]
                        feature_similarity = 1 - abs(cluster_value - platform_value)
                        similarity += feature_similarity
                        feature_count += 1
                
                if feature_count > 0:
                    avg_similarity = similarity / feature_count
                    if avg_similarity > best_score:
                        best_score = avg_similarity
                        best_cluster = cluster_id
            
            self.platform_preferences[platform] = {
                'cluster_distribution': cluster_distribution,
                'best_matching_cluster': best_cluster,
                'platform_audio_profile': platform_audio_profile,
                'total_songs': total_songs
            }
        
        # Cache platform analysis
        cache_file = os.path.join(self.cache_dir, 'platform_preferences.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(self.platform_preferences, f)
        
        return self.platform_preferences
    
    def predict_song_cluster(self, song_features):
        """Predict which cluster a new song belongs to"""
        if self.kmeans_model is None:
            # Load cached model
            cache_file = os.path.join(self.cache_dir, 'clustering_model.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.kmeans_model = cached_data['kmeans_model']
                    self.scaler = cached_data['scaler']
            else:
                logger.error("No clustering model found. Run perform_clustering() first.")
                return None
        
        # Prepare song features
        feature_vector = []
        for feature in self.feature_matrix.columns:
            value = song_features.get(feature, 0)
            
            # Apply same normalization as training data
            if feature == 'tempo':
                # Normalize tempo to 0-1 scale
                value = (value - 60) / (200 - 60)  # Tempo range 60-200 BPM
                value = np.clip(value, 0, 1)
            elif feature == 'loudness':
                # Normalize loudness (typically -60 to 0 dB)
                value = (value + 60) / 60
                value = np.clip(value, 0, 1)
            
            feature_vector.append(value)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform([feature_vector]).astype(np.float64)
        # Predict cluster
        predicted_cluster = self.kmeans_model.predict(feature_vector_scaled)[0]
        
        # Get cluster probabilities (distances to all centroids)
        distances = self.kmeans_model.transform(feature_vector_scaled)[0]
        max_distance = np.max(distances)
        probabilities = (max_distance - distances) / max_distance  # Invert distances
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        return {
            'predicted_cluster': int(predicted_cluster),
            'cluster_probabilities': {int(i): round(float(prob), 3) for i, prob in enumerate(probabilities)},
            'confidence': round(float(probabilities[predicted_cluster]), 3)
        }
    
    def generate_clustering_insights(self, song_features=None):
        """Generate comprehensive clustering insights"""
        insights = {
            'sound_profiles': self.sound_profiles,
            'demographic_clusters': self.demographic_clusters,
            'platform_preferences': self.platform_preferences,
            'clustering_summary': self._generate_clustering_summary()
        }
        
        if song_features:
            prediction = self.predict_song_cluster(song_features)
            if prediction:
                predicted_cluster = prediction['predicted_cluster']
                # Robust cluster_profile lookup for int/str key mismatches
                cluster_profile = (
                    self.sound_profiles.get(predicted_cluster)
                    or self.sound_profiles.get(str(predicted_cluster))
                    or self.sound_profiles.get(int(predicted_cluster))
                    or {}
                )
                # Debug prints
                print('sound_profiles keys:', self.sound_profiles.keys())
                print('predicted_cluster:', predicted_cluster, type(predicted_cluster))
                print('cluster_profile:', cluster_profile)
                print('demographic_clusters:', self.demographic_clusters)
                print('platform_preferences:', self.platform_preferences)
                insights['song_prediction'] = {
                    'prediction': prediction,
                    'cluster_profile': cluster_profile,
                    'target_demographics': self._get_cluster_demographics(predicted_cluster),
                    'recommended_platforms': self._get_cluster_platforms(predicted_cluster)
                }
        
        return insights
    
    def _generate_clustering_summary(self):
        """Generate summary statistics for clustering"""
        if not self.sound_profiles:
            return {}
        
        summary = {
            'total_clusters': len(self.sound_profiles),
            'total_songs': sum(profile['size'] for profile in self.sound_profiles.values()),
            'cluster_sizes': {cid: profile['size'] for cid, profile in self.sound_profiles.items()},
            'largest_cluster': max(self.sound_profiles.items(), key=lambda x: x[1]['size']),
            'most_diverse_features': self._find_most_diverse_features()
        }
        
        return summary
    
    def _find_most_diverse_features(self):
        """Find features that vary most across clusters"""
        if not self.sound_profiles:
            return []
        
        feature_variations = {}
        features = list(self.sound_profiles[0]['audio_features'].keys())
        
        for feature in features:
            values = [profile['audio_features'][feature] for profile in self.sound_profiles.values()]
            variation = np.std(values)
            feature_variations[feature] = variation
        
        # Sort by descending variation 
        sorted_features = sorted(feature_variations.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features[:5]  # Top 5 most diverse features
    
    def _get_cluster_demographics(self, cluster_id):
        """Get best demographics for a cluster"""
        if not self.demographic_clusters:
            return []
        
        demographics = []
        for demo, data in self.demographic_clusters.items():
            for cid, score in data['cluster_rankings']:
                if cid == cluster_id:
                    demographics.append({'demographic': demo, 'score': score})
                    break
        
        return sorted(demographics, key=lambda x: x['score'], reverse=True)
    
    def _get_cluster_platforms(self, cluster_id):
        """Get best platforms for a cluster"""
        if not self.platform_preferences:
            return []
        
        platforms = []
        for platform, data in self.platform_preferences.items():
            if cluster_id in data['cluster_distribution']:
                percentage = data['cluster_distribution'][cluster_id]['percentage']
                platforms.append({'platform': platform, 'percentage': percentage})
        
        return sorted(platforms, key=lambda x: x['percentage'], reverse=True)
    
    def save_all_results(self, output_dir='clustering_results'):
        """Save all clustering results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sound profiles
        if self.sound_profiles:
            # Convert keys to int for JSON serialization
            self.sound_profiles = {int(k): v for k, v in self.sound_profiles.items()}
            sound_profiles_clean = convert_numpy_types(self.sound_profiles)
            with open(os.path.join(output_dir, 'sound_profiles.json'), 'w') as f:
                json.dump(sound_profiles_clean, f, indent=2)
        
        # Save demographic analysis
        if self.demographic_clusters:
            demographic_clusters_clean = convert_numpy_types(self.demographic_clusters)
            with open(os.path.join(output_dir, 'demographic_clusters.json'), 'w') as f:
                json.dump(demographic_clusters_clean, f, indent=2)
        
        # Save platform analysis
        if self.platform_preferences:
            platform_preferences_clean = convert_numpy_types(self.platform_preferences)
            with open(os.path.join(output_dir, 'platform_preferences.json'), 'w') as f:
                json.dump(platform_preferences_clean, f, indent=2)
        
        logger.info(f"All clustering results saved to {output_dir}")
    
    def run_complete_clustering_analysis(self, n_clusters=None):
        """Run complete clustering analysis pipeline"""
        logger.info("Starting complete clustering analysis...")
        
        # Prepare data
        if not self.prepare_clustering_data():
            return None
        
        # Perform clustering
        self.perform_clustering(n_clusters)
        
        # Create sound profiles
        self.create_sound_profiles()
        
        # Analyze demographics
        self.analyze_demographic_clusters()
        
        # Analyze platform preferences
        self.analyze_platform_preferences()
        
        # Save results
        self.save_all_results()
        
        logger.info("Complete clustering analysis finished!")
        
        return {
            'sound_profiles': self.sound_profiles,
            'demographic_clusters': self.demographic_clusters,
            'platform_preferences': self.platform_preferences
        }

    def load_clustering_results(self, output_dir='clustering_results'):
        """Load clustering results from JSON files"""
        import json
        import os
        # Load sound profiles
        sound_profiles_path = os.path.join(output_dir, 'sound_profiles.json')
        if os.path.exists(sound_profiles_path):
            with open(sound_profiles_path, 'r') as f:
                self.sound_profiles = json.load(f)
                # Convert keys to int for consistency
                self.sound_profiles = {int(k): v for k, v in self.sound_profiles.items()}
        # Load demographic clusters
        demographic_clusters_path = os.path.join(output_dir, 'demographic_clusters.json')
        if os.path.exists(demographic_clusters_path):
            with open(demographic_clusters_path, 'r') as f:
                self.demographic_clusters = json.load(f)
        # Load platform preferences
        platform_preferences_path = os.path.join(output_dir, 'platform_preferences.json')
        if os.path.exists(platform_preferences_path):
            with open(platform_preferences_path, 'r') as f:
                self.platform_preferences = json.load(f)

class ClusteringVisualizer:
    """Visualization utilities for clustering results"""
    def __init__(self, clustering_engine):
        self.engine = clustering_engine
    
    def create_cluster_visualization(self, output_file='cluster_visualization.html'):
        """Create interactive cluster visualization using PCA"""
        if self.engine.feature_matrix_scaled is None:
            logger.error("No clustering data available")
            return None
        
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(self.engine.feature_matrix_scaled)
        
        # Create DF for plotting
        viz_df = pd.DataFrame({
            'PC1': pca_features[:, 0],
            'PC2': pca_features[:, 1],
            'Cluster': self.engine.cluster_labels,
            'Track': self.engine.combined_df['track_name'].fillna('Unknown'),
            'Artist': self.engine.combined_df['artists'].fillna('Unknown'),
            'Platform': self.engine.combined_df['source_platform'].fillna('Unknown'),
            'Popularity': self.engine.combined_df['platform_popularity'].fillna(0)
        })
        
        # Create interactive scatter plot
        fig = px.scatter(
            viz_df, 
            x='PC1', 
            y='PC2',
            color='Cluster',
            hover_data=['Track', 'Artist', 'Platform', 'Popularity'],
            title='Audio Feature Clusters (PCA Visualization)',
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
        )
        
        # Add cluster centers
        if self.engine.kmeans_model is not None:
            centers_pca = pca.transform(self.engine.kmeans_model.cluster_centers_)
            fig.add_trace(go.Scatter(
                x=centers_pca[:, 0],
                y=centers_pca[:, 1],
                mode='markers',
                marker=dict(symbol='x', size=15, color='black'),
                name='Cluster Centers'
            ))
        
        fig.write_html(output_file)
        logger.info(f"Cluster visualization saved to {output_file}")
        return fig
    
    def create_feature_heatmap(self, output_file='cluster_features_heatmap.html'):
        """Create heatmap of cluster audio features"""
        if not self.engine.sound_profiles:
            logger.error("No sound profiles available")
            return None
        
        # Prepare data for heatmap
        clusters = list(self.engine.sound_profiles.keys())
        features = list(self.engine.sound_profiles[clusters[0]]['audio_features'].keys())
        
        heatmap_data = []
        for cluster_id in clusters:
            row = []
            for feature in features:
                value = self.engine.sound_profiles[cluster_id]['audio_features'][feature]
                row.append(value)
            heatmap_data.append(row)
        
        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=features,
            y=[f'Cluster {i}' for i in clusters],
            colorscale='RdYlBu_r',
            text=heatmap_data,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title='Audio Features by Cluster',
            xaxis_title='Audio Features',
            yaxis_title='Clusters'
        )
        
        fig.write_html(output_file)
        logger.info(f"Feature heatmap saved to {output_file}")
        return fig
    
    def create_demographic_analysis_chart(self, output_file='demographic_clusters.html'):
        """Create visualization of demographic cluster preferences"""
        if not self.engine.demographic_clusters:
            logger.error("No demographic analysis available")
            return None
        
        # Prepare data
        demographics = list(self.engine.demographic_clusters.keys())
        clusters = list(self.engine.sound_profiles.keys())
        
        data = []
        for demo in demographics:
            cluster_scores = self.engine.demographic_clusters[demo]['cluster_distribution']
            for cluster_id in clusters:
                score = cluster_scores.get(cluster_id, 0)
                data.append({
                    'Demographic': demo,
                    'Cluster': f'Cluster {cluster_id}',
                    'Score': score
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig = px.bar(
            df,
            x='Cluster',
            y='Score',
            color='Demographic',
            title='Cluster Appeal by Demographic',
            labels={'Score': 'Appeal Score (%)'}
        )
        
        fig.write_html(output_file)
        logger.info(f"Demographic analysis chart saved to {output_file}")
        return fig
    
    def create_platform_distribution_chart(self, output_file='platform_clusters.html'):
        """Create visualization of platform cluster distributions"""
        if not self.engine.platform_preferences:
            logger.error("No platform analysis available")
            return None
        
        # Create subplots for each platform
        platforms = list(self.engine.platform_preferences.keys())
        
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=platforms[:4],  # Show up to 4 platforms
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, platform in enumerate(platforms[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            cluster_dist = self.engine.platform_preferences[platform]['cluster_distribution']
            clusters = list(cluster_dist.keys())
            percentages = [cluster_dist[c]['percentage'] for c in clusters]
            
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {c}' for c in clusters],
                    y=percentages,
                    name=platform,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Cluster Distribution by Platform',
            height=600
        )
        
        fig.write_html(output_file)
        logger.info(f"Platform distribution chart saved to {output_file}")
        return fig
    
    def generate_all_visualizations(self, output_dir='clustering_visualizations'):
        """Generate all clustering visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = []
        
        # Cluster scatter plot
        cluster_viz = self.create_cluster_visualization(
            os.path.join(output_dir, 'cluster_visualization.html')
        )
        if cluster_viz:
            visualizations.append('cluster_visualization.html')
        
        # Feature heatmap
        feature_heatmap = self.create_feature_heatmap(
            os.path.join(output_dir, 'cluster_features_heatmap.html')
        )
        if feature_heatmap:
            visualizations.append('cluster_features_heatmap.html')
        
        # Demographic analysis
        demo_chart = self.create_demographic_analysis_chart(
            os.path.join(output_dir, 'demographic_clusters.html')
        )
        if demo_chart:
            visualizations.append('demographic_clusters.html')
        
        # Platform distribution
        platform_chart = self.create_platform_distribution_chart(
            os.path.join(output_dir, 'platform_clusters.html')
        )
        if platform_chart:
            visualizations.append('platform_clusters.html')
        
        logger.info(f"Generated {len(visualizations)} visualizations in {output_dir}")
        return visualizations

# Integration class to connect clustering with marketing insights
class IntegratedMusicAnalytics:
    """Integrated system combining marketing insights and clustering"""
    
    def __init__(self, data_path='cleaned_music', cache_dir='integrated_cache'):
        self.data_path = data_path
        self.cache_dir = cache_dir
        
        # Initialize components
        try:
            from scripts.features.hybrid_marketing_insights_generator import HybridMarketingInsightsGenerator
        except ImportError:
            # Fallback to relative import
            from hybrid_marketing_insights_generator import HybridMarketingInsightsGenerator
        
        self.marketing_engine = HybridMarketingInsightsGenerator(
            data_path=data_path,
            cache_dir=f"{cache_dir}/marketing"
        )
        
        self.clustering_engine = AudioFeatureClusteringEngine(
            data_path=data_path,
            cache_dir=f"{cache_dir}/clustering"
        )
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def run_complete_analysis(self):
        """Run both marketing insights and clustering analysis"""
        logger.info("Running complete integrated analysis...")
        
        # Run marketing insights analysis
        logger.info("Phase 1: Marketing Insights Analysis")
        marketing_results = self.marketing_engine.run_production_analysis()
        
        # Run clustering analysis
        logger.info("Phase 2: Clustering Analysis")
        clustering_results = self.clustering_engine.run_complete_clustering_analysis()
        
        # Create integrated insights
        logger.info("Phase 3: Creating Integrated Insights")
        integrated_insights = self._create_integrated_insights(marketing_results, clustering_results)
        
        # Save comprehensive results
        self._save_integrated_results(integrated_insights)
        
        return integrated_insights
    
    def _create_integrated_insights(self, marketing_results, clustering_results):
        """Create insights combining both marketing and clustering analysis"""
        integrated = {
            'marketing_insights': marketing_results,
            'clustering_insights': clustering_results,
            'integrated_recommendations': {}
        }
        
        # Cross-reference clusters with demographics and platforms
        if clustering_results and 'sound_profiles' in clustering_results:
            for cluster_id, profile in clustering_results['sound_profiles'].items():
                cluster_recommendations = {
                    'cluster_id': cluster_id,
                    'characteristics': profile['characteristics'],
                    'size': profile['size'],
                    'marketing_strategy': self._generate_cluster_marketing_strategy(cluster_id, profile)
                }
                
                integrated['integrated_recommendations'][cluster_id] = cluster_recommendations
        
        return integrated
    
    def _generate_cluster_marketing_strategy(self, cluster_id, cluster_profile):
        """Generate marketing strategy for a specific cluster"""
        strategy = {
            'target_demographics': [],
            'recommended_platforms': [],
            'content_strategy': [],
            'budget_allocation': {}
        }
        
        # Determine target demographics based on cluster characteristics
        characteristics = cluster_profile.get('characteristics', [])
        
        if 'Highly Danceable' in characteristics and 'High Energy' in characteristics:
            strategy['target_demographics'] = ['Gen Z (16-24)', 'Millennials (25-40)']
            strategy['content_strategy'].append('Focus on dance challenges and high-energy content')
        elif 'Acoustic' in characteristics:
            strategy['target_demographics'] = ['Gen X (41-56)', 'Boomers (57+)']
            strategy['content_strategy'].append('Emphasize intimate, authentic performances')
        
        # Platform recommendations based on cluster platform distribution
        platform_dist = cluster_profile.get('platforms', {})
        sorted_platforms = sorted(platform_dist.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        for platform, data in sorted_platforms[:3]:  # Top 3 platforms
            strategy['recommended_platforms'].append({
                'platform': platform,
                'percentage': data['percentage'],
                'priority': 'High' if data['percentage'] > 30 else 'Medium'
            })
        
        return strategy
    
    def _save_integrated_results(self, integrated_insights):
        """Save integrated analysis results"""
        # Save as JSON
        json_file = os.path.join(self.cache_dir, 'integrated_insights.json')
        with open(json_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            serializable_insights = convert_types(integrated_insights)
            json.dump(serializable_insights, f, indent=2)
        
        # Save as pickle (preserves all data types)
        pickle_file = os.path.join(self.cache_dir, 'integrated_insights.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(integrated_insights, f)
        
        logger.info(f"Integrated insights saved to {self.cache_dir}")
    
    def get_comprehensive_song_analysis(self, song_features, artist_name=None):
        """Get complete analysis for a new song using both systems"""
        # Get marketing insights
        marketing_strategy = self.marketing_engine.get_complete_marketing_strategy(
            song_features, artist_name
        )
        
        # Get clustering prediction
        clustering_insights = self.clustering_engine.generate_clustering_insights(song_features)
        
        # Combine insights
        comprehensive_analysis = {
            'song_features': song_features,
            'marketing_strategy': marketing_strategy,
            'clustering_insights': clustering_insights,
            'recommendations': self._combine_recommendations(marketing_strategy, clustering_insights)
        }
        
        return comprehensive_analysis
    
    def _combine_recommendations(self, marketing_strategy, clustering_insights):
        """Combine recommendations from both systems"""
        combined = {
            'confidence_score': 0,
            'primary_recommendations': [],
            'detailed_strategy': {}
        }
        
        # Calculate confidence based on agreement between systems
        if marketing_strategy and clustering_insights and 'song_prediction' in clustering_insights:
            # Check if demographic recommendations align
            marketing_demo = marketing_strategy.get('marketing_recommendations', {}).get('primary_demographic')
            cluster_demos = clustering_insights['song_prediction'].get('target_demographics', [])
            
            demographic_alignment = any(demo['demographic'] == marketing_demo for demo in cluster_demos)
            
            if demographic_alignment:
                combined['confidence_score'] = 85
                combined['primary_recommendations'].append(
                    f"High confidence: Both systems recommend targeting {marketing_demo}"
                )
            else:
                combined['confidence_score'] = 65
                combined['primary_recommendations'].append(
                    "Moderate confidence: Systems suggest different demographic targets"
                )
        
        return combined

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Audio Feature Clustering Analysis'
    )
    parser.add_argument(
        'command',
        choices=['analyze', 'predict', 'visualize', 'integrated'],
        help=(
            'analyze: run full clustering analysis, '
            'predict: predict cluster for song, '
            'visualize: generate all cluster visualizations, '
            'integrated: run combined clustering and marketing analysis'
        )
    )
    parser.add_argument(
        '--clusters', type=int, help='Number of clusters (auto-detect if not specified)'
    )
    parser.add_argument(
        '--cache-dir', default='clustering_cache', help='Cache directory'
    )
    parser.add_argument(
        '--output-dir', default='clustering_results', help='Output directory'
    )
    parser.add_argument(
        '--data-dir', default='cleaned_music', help='Data directory'
    )
    args = parser.parse_args()

    if args.command == 'analyze':
        clustering_engine = AudioFeatureClusteringEngine(
            data_path=args.data_dir, cache_dir=args.cache_dir
        )
        results = clustering_engine.run_complete_clustering_analysis(args.clusters)
        
        if results:
            print("\n=== CLUSTERING ANALYSIS COMPLETE ===")
            print(f"Sound profiles created: {len(results['sound_profiles'])}")
            print(f"Demographic analysis: {len(results['demographic_clusters'])} demographics")
            print(f"Platform analysis: {len(results['platform_preferences'])} platforms")
            
            # Show cluster summary
            print(f"\n=== CLUSTER SUMMARY ===")
            for cluster_id, profile in results['sound_profiles'].items():
                print(f"Cluster {cluster_id}: {profile['size']} songs ({profile['percentage']}%)")
                print(f"  Characteristics: {', '.join(profile['characteristics'])}")
                print(f"  Top platforms: {list(profile['platforms'].keys())[:3]}")
                print()
    
    elif args.command == 'predict':
        clustering_engine = AudioFeatureClusteringEngine(
            data_path=args.data_dir, cache_dir=args.cache_dir
        )
        
        # Example song prediction
        example_song = {
            'danceability': 0.8,
            'energy': 0.9,
            'valence': 0.85,
            'acousticness': 0.1,
            'instrumentalness': 0.0,
            'liveness': 0.2,
            'speechiness': 0.15,
            'tempo': 128,
            'loudness': -5.0
        }
        
        print("Predicting cluster for example song...")
        print(f"Song features: {example_song}")
        
        # Ensure feature matrix is prepared
        clustering_engine.prepare_clustering_data()
        # Load clustering results from disk
        clustering_engine.load_clustering_results()
        # Generate insights including prediction
        insights = clustering_engine.generate_clustering_insights(example_song)
        
        if 'song_prediction' in insights:
            prediction = insights['song_prediction']
            print(f"\n=== SONG CLUSTER PREDICTION ===")
            print(f"Predicted cluster: {prediction['prediction']['predicted_cluster']}")
            print(f"Confidence: {prediction['prediction']['confidence']:.1%}")
            
            cluster_profile = prediction['cluster_profile']
            print(f"\nCluster characteristics: {', '.join(cluster_profile.get('characteristics', []))}")
            
            print(f"\nTarget demographics:")
            for demo in prediction['target_demographics'][:3]:
                print(f"  {demo['demographic']}: {demo['score']:.1f}% match")
            
            print(f"\nRecommended platforms:")
            for platform in prediction['recommended_platforms'][:3]:
                print(f"  {platform['platform']}: {platform['percentage']:.1f}% of cluster")
        else:
            print("Could not generate prediction. Run clustering analysis first.")
    
    elif args.command == 'visualize':
        clustering_engine = AudioFeatureClusteringEngine(
            data_path=args.data_dir, cache_dir=args.cache_dir
        )
        # Ensure feature matrix is prepared
        clustering_engine.prepare_clustering_data()
        # Load clustering results
        clustering_engine.load_clustering_results(output_dir=args.output_dir)
        visualizer = ClusteringVisualizer(clustering_engine)
        print("Generating cluster scatter visualization...")
        visualizer.create_cluster_visualization(output_file=f"{args.output_dir}/cluster_visualization.html")
        print("Generating feature heatmap visualization...")
        visualizer.create_feature_heatmap(output_file=f"{args.output_dir}/cluster_features_heatmap.html")
        print("Generating demographic analysis chart...")
        visualizer.create_demographic_analysis_chart(output_file=f"{args.output_dir}/demographic_clusters.html")
        print("Generating platform distribution chart...")
        visualizer.create_platform_distribution_chart(output_file=f"{args.output_dir}/platform_clusters.html")
        print("All visualizations generated in:", args.output_dir)
    
    elif args.command == 'integrated':
        # Run integrated analysis combining clustering and marketing insights
        print("Running integrated analysis (clustering + marketing insights)...")
        
        try:
            # Initialize integrated system
            integrated_system = IntegratedMusicAnalytics(
                data_path=args.data_dir,
                cache_dir='integrated_cache'
            )
            
            # Run complete integrated analysis
            results = integrated_system.run_complete_analysis()
            
            if results:
                print("\n=== INTEGRATED ANALYSIS COMPLETE ===")
                print("Results saved to integrated_cache/")
                
                # Show summary of integrated recommendations
                if 'integrated_recommendations' in results:
                    print(f"\n=== INTEGRATED RECOMMENDATIONS ===")
                    for cluster_id, recommendation in results['integrated_recommendations'].items():
                        print(f"Cluster {cluster_id}:")
                        print(f"  Characteristics: {', '.join(recommendation['characteristics'])}")
                        print(f"  Size: {recommendation['size']} songs")
                        
                        strategy = recommendation['marketing_strategy']
                        if strategy['target_demographics']:
                            print(f"  Target demographics: {', '.join(strategy['target_demographics'])}")
                        if strategy['recommended_platforms']:
                            platforms = [f"{p['platform']} ({p['percentage']:.1f}%)" for p in strategy['recommended_platforms']]
                            print(f"  Recommended platforms: {', '.join(platforms)}")
                        print()
            else:
                print("Integrated analysis failed. Check that marketing insights analysis has been run first.")
                
        except ImportError as e:
            print(f"Error: Could not import marketing insights module. {e}")
            print("Make sure the marketing insights script is available and properly configured.")
        except Exception as e:
            print(f"Error during integrated analysis: {e}")
            print("Check that all required data and dependencies are available.")
    
    else:
        print("Invalid command. Use 'analyze', 'predict', 'visualize', or 'integrated'.")
        