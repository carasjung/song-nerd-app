import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
import json
warnings.filterwarnings('ignore')

class MarketingInsightsGenerator:
    df_type = pd.DataFrame | None
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "cross_platform_features"
        self.output_dir = self.data_dir / "marketing_insights"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comprehensive_features: MarketingInsightsGenerator.df_type = None
        self.virality_scores: MarketingInsightsGenerator.df_type = None
        self.platform_affinity: MarketingInsightsGenerator.df_type = None
        self.demographic_alignment: MarketingInsightsGenerator.df_type = None
        
        # Audio features for clustering and analysis
        self.audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo'
        ]
        
    def load_cross_platform_data(self):
        """Load cross-platform analysis results"""
        logging.info("Loading cross-platform analysis results...")
        
        try:
            self.comprehensive_features = pd.read_csv(
                self.features_dir / "comprehensive_cross_platform_features.csv"
            )
            
            if 'virality_score' not in self.comprehensive_features.columns:
                logging.info(
                    "Virality score not in comprehensive features, "
                    "attempting to load or create."
                )
                
                virality_file = self.features_dir / "virality_scores.csv"
                if virality_file.exists():
                    logging.info(
                        f"Loading virality scores from {virality_file}"
                    )
                    self.virality_scores = pd.read_csv(virality_file)
                else:
                    logging.info(
                        "Creating virality scores from existing data."
                    )
                    self.virality_scores = self._create_virality_scores()

                if self.virality_scores is not None and 'virality_score' in self.virality_scores.columns:
                    # Merge virality scores into comprehensive_features
                    if self.comprehensive_features is not None:
                        self.comprehensive_features = (
                            self.comprehensive_features.merge(
                                self.virality_scores[
                                    ['track_clean', 'artist_clean', 'virality_score']
                                ],
                                on=['track_clean', 'artist_clean'],
                                how='left'
                            )
                        )
                        # Fill missing virality scores with the mean
                        if self.comprehensive_features['virality_score'].isnull().any():
                            mean_virality = self.comprehensive_features[
                                'virality_score'
                            ].mean()
                            self.comprehensive_features['virality_score'].fillna(
                                mean_virality, inplace=True
                            )
                            logging.info(
                                "Filled missing virality scores with mean value "
                                f"({mean_virality:.2f})"
                            )
                else:
                    logging.error("Failed to load or create 'virality_score'.")
                    raise ValueError("Missing 'virality_score' in data.")
            
            # Now that virality_score is in comprehensive_features, 
            # we can create self.virality_scores from it
            if self.comprehensive_features is not None and 'virality_score' in self.comprehensive_features.columns:
                self.virality_scores = self.comprehensive_features[
                    ['track_clean', 'artist_clean', 'virality_score']
                ].copy()
            else:
                # This should not be reached if the logic above is correct
                raise ValueError("Virality score processing failed.")
            
            # Load other files if they exist
            platform_affinity_file = (
                self.features_dir / "platform_affinity_analysis.csv"
            )
            if platform_affinity_file.exists():
                self.platform_affinity = pd.read_csv(platform_affinity_file)
            else:
                logging.info(
                    "Platform affinity file not found, will create from "
                    "available data"
                )
                self.platform_affinity = self._create_platform_affinity()
            
            demographic_file = self.features_dir / "demographic_alignment.csv"
            if demographic_file.exists():
                self.demographic_alignment = pd.read_csv(demographic_file)
            else:
                logging.info(
                    "Demographic alignment file not found, will create from "
                    "available data"
                )
                self.demographic_alignment = self._create_demographic_alignment()
            
            logging.info("Successfully loaded all cross-platform data")

        except FileNotFoundError as e:
            logging.error(f"Cross-platform data not found: {e}")
            logging.info("Please run the CrossPlatformMetrics analysis first")
            raise
    
    def _create_virality_scores(self):
        """Create virality scores from existing popularity data"""
        logging.info("Creating virality scores from popularity data...")
        
        if self.comprehensive_features is None:
            raise ValueError(
                "comprehensive_features must be loaded "
                "before creating virality scores."
            )

        # Use existing popularity scores and create a composite virality score
        virality_data = self.comprehensive_features[
            ['track_clean', 'artist_clean', 'platform']
        ].copy()
        
        # Calculate virality score based on available metrics
        if 'popularity_score' in self.comprehensive_features.columns:
            virality_data['virality_score'] = self.comprehensive_features[
                'popularity_score'
            ]
        else:
            # Create a composite score from available features
            score_components = []
            
            # Use energy and danceability as proxies for virality
            if 'energy' in self.comprehensive_features.columns:
                score_components.append(self.comprehensive_features['energy'])
            if 'danceability' in self.comprehensive_features.columns:
                score_components.append(
                    self.comprehensive_features['danceability']
                )
            if 'valence' in self.comprehensive_features.columns:
                score_components.append(self.comprehensive_features['valence'])
            
            if score_components:
                virality_data['virality_score'] = np.mean(
                    score_components, axis=0
                )
            else:
                # Fallback: random scores for demonstration
                virality_data['virality_score'] = np.random.uniform(
                    0, 1, len(virality_data)
                )
        
        # Normalize to 0-1 range
        score = virality_data['virality_score']
        virality_data['virality_score'] = (score - score.min()) / (
            score.max() - score.min()
        )
        
        return virality_data
    
    def _create_platform_affinity(self):
        """Create platform affinity analysis from available data"""
        logging.info("Creating platform affinity analysis...")
        
        if self.comprehensive_features is None:
            raise ValueError(
                "comprehensive_features must be loaded "
                "before creating platform affinity."
            )

        platform_affinity_data = []
        
        # Get available platforms
        platforms = self.comprehensive_features['platform'].unique()
        available_features = [
            f for f in self.audio_features 
            if f in self.comprehensive_features.columns
        ]
        
        for platform in platforms:
            platform_data = self.comprehensive_features[
                self.comprehensive_features['platform'] == platform
            ]
            
            for feature in available_features:
                if feature in platform_data.columns:
                    # Calculate correlation with a success proxy
                    if 'popularity_score' in platform_data.columns:
                        correlation = platform_data[feature].corr(
                            platform_data['popularity_score']
                        )
                    else:
                        # Use energy as a proxy for engagement
                        proxy = platform_data.get(
                            'energy', pd.Series([0]*len(platform_data))
                        )
                        correlation = platform_data[feature].corr(proxy)
                    
                    if not np.isnan(correlation):
                        platform_affinity_data.append({
                            'platform': platform,
                            'feature': feature,
                            'correlation': correlation
                        })
        
        return pd.DataFrame(platform_affinity_data)
    
    def _create_demographic_alignment(self):
        """Create demographic alignment analysis from available data"""
        logging.info("Creating demographic alignment analysis...")
        
        if self.comprehensive_features is None:
            raise ValueError(
                "comprehensive_features must be loaded "
                "before creating demographic alignment."
            )

        # Create demographic alignment based on platform and audio features
        alignment_data = self.comprehensive_features[
            ['track_clean', 'artist_clean', 'platform']
        ].copy()
        
        # Gen Z alignment (TikTok-focused): high energy, danceability
        gen_z_score = 0
        if 'energy' in self.comprehensive_features.columns:
            gen_z_score += self.comprehensive_features['energy'] * 0.4
        if 'danceability' in self.comprehensive_features.columns:
            gen_z_score += self.comprehensive_features['danceability'] * 0.4
        if 'valence' in self.comprehensive_features.columns:
            gen_z_score += self.comprehensive_features['valence'] * 0.2
        alignment_data['Gen Z (TikTok-focused)_alignment'] = gen_z_score
        
        # Millennial alignment (Spotify-focused): balanced features
        millennial_score = 0
        if 'acousticness' in self.comprehensive_features.columns:
            millennial_score += self.comprehensive_features['acousticness'] * 0.3
        if 'energy' in self.comprehensive_features.columns:
            millennial_score += self.comprehensive_features['energy'] * 0.3
        if 'valence' in self.comprehensive_features.columns:
            millennial_score += self.comprehensive_features['valence'] * 0.2
        if 'instrumentalness' in self.comprehensive_features.columns:
            millennial_score += (
                1 - self.comprehensive_features['instrumentalness']
            ) * 0.2
        alignment_data['Millennials (Spotify-focused)_alignment'] = millennial_score
        
        # Cross-platform appeal: moderate values across features
        cross_platform_score = 0
        feature_count = 0
        for feature in ['energy', 'danceability', 'valence', 'acousticness']:
            if feature in self.comprehensive_features.columns:
                # Prefer moderate values (closer to 0.5)
                feature_values = self.comprehensive_features[feature]
                moderate_scores = 1 - abs(feature_values - 0.5) * 2
                cross_platform_score += moderate_scores
                feature_count += 1
        
        if feature_count > 0:
            alignment_data['Cross-Platform Appeal_alignment'] = (
                cross_platform_score / feature_count
            )
        else:
            alignment_data['Cross-Platform Appeal_alignment'] = 0.5
        
        return alignment_data
    
    def create_music_personas(self):
        """Create music personas using clustering"""
        logging.info("Creating music personas through clustering...")
        
        if self.comprehensive_features is None:
            raise ValueError(
                "comprehensive_features must be loaded "
                "before creating music personas."
            )

        # Prepare data for clustering
        cluster_data = self.comprehensive_features.copy()
        
        # Select features for clustering
        available_features = [
            f for f in self.audio_features if f in cluster_data.columns
        ]
        
        if len(available_features) < 3:
            logging.warning("Insufficient audio features for clustering")
            return None
        
        # Clean data for clustering
        cluster_features = cluster_data[available_features].copy()
        cluster_features = cluster_features.fillna(cluster_features.mean())
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_features)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        # Ensure k_range is valid
        max_k = min(11, len(cluster_features) // 10)
        if max_k < 2:
            logging.warning("Not enough data to form multiple clusters.")
            return None
        k_range = range(2, max_k)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method or default to 5 clusters
        optimal_k = 5  # Default
        if len(inertias) > 2:
            # Simple elbow detection
            diffs = np.diff(inertias)
            with np.errstate(divide='ignore', invalid='ignore'):
                diff_ratios = diffs[:-1] / diffs[1:]
            if len(diff_ratios) > 0 and np.all(np.isfinite(diff_ratios)):
                optimal_k = np.argmax(diff_ratios) + 2
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to data
        cluster_data['music_persona'] = cluster_labels
        
        # Analyze cluster characteristics
        persona_profiles = {}
        persona_names = [
            'High-Energy Dancers', 'Chill Acoustics', 'Electronic Beats', 
            'Vocal-Heavy Tracks', 'Instrumental Focus', 'Balanced Mainstream',
            'Experimental', 'Live Performance', 'Melancholic', 'Party Anthems'
        ]
        
        for i in range(optimal_k):
            cluster_mask = cluster_data['music_persona'] == i
            cluster_subset = cluster_data[cluster_mask]
            
            # Calculate cluster centroid characteristics
            profile = {}
            for feature in available_features:
                profile[feature] = {
                    'mean': cluster_subset[feature].mean(),
                    'std': cluster_subset[feature].std()
                }
            
            # Assign persona name based on characteristics
            persona_name = (
                persona_names[i] if i < len(persona_names) else f'Persona_{i}'
            )
            
            # Get virality score for this cluster
            if self.virality_scores is not None:
                cluster_tracks = self.virality_scores[
                    (self.virality_scores['track_clean'].isin(cluster_subset['track_clean'])) &
                    (self.virality_scores['artist_clean'].isin(cluster_subset['artist_clean']))
                ]
                avg_virality = cluster_tracks['virality_score'].mean() if len(cluster_tracks) > 0 else 0
            else:
                avg_virality = 0

            
            persona_profiles[persona_name] = {
                'cluster_id': i,
                'size': len(cluster_subset),
                'percentage': len(cluster_subset) / len(cluster_data) * 100,
                'audio_profile': profile,
                'avg_virality': avg_virality,
                'platform_distribution': cluster_subset['platform']
                .value_counts()
                .to_dict()
            }
        
        self.music_personas = persona_profiles
        self.comprehensive_features = cluster_data
        
        logging.info(f"Created {optimal_k} music personas")
        return persona_profiles
    
    def analyze_success_patterns(self):
        """Analyze patterns that lead to cross-platform success"""
        logging.info("Analyzing success patterns...")
        
        if self.comprehensive_features is None or self.virality_scores is None:
            raise ValueError(
                "Data must be loaded before analyzing success patterns."
            )

        success_patterns = {}
        
        # Define success metrics
        high_virality_threshold = (
            self.virality_scores['virality_score'].quantile(0.8)
        )
        
        # High virality tracks
        high_viral_tracks = self.virality_scores[
            self.virality_scores['virality_score'] >= high_virality_threshold
        ]
        
        # Merge with comprehensive features to get audio characteristics
        success_data = self.comprehensive_features.merge(
            high_viral_tracks[['track_clean', 'artist_clean', 'virality_score']],
            on=['track_clean', 'artist_clean'],
            how='inner'
        )
        
        if len(success_data) == 0:
            logging.warning("No high virality tracks found for analysis")
            return {}
        
        # Analyze audio features of successful tracks
        available_features = [
            f for f in self.audio_features if f in success_data.columns
        ]
        
        success_patterns['high_virality_profile'] = {}
        for feature in available_features:
            success_patterns['high_virality_profile'][feature] = {
                'mean': success_data[feature].mean(),
                'median': success_data[feature].median(),
                'std': success_data[feature].std()
            }
        
        # Cross-platform success analysis
        cross_platform_tracks = success_data[
            success_data.get('is_cross_platform', False)
        ]
        
        if len(cross_platform_tracks) > 0:
            success_patterns['cross_platform_profile'] = {}
            for feature in available_features:
                success_patterns['cross_platform_profile'][feature] = {
                    'mean': cross_platform_tracks[feature].mean(),
                    'median': cross_platform_tracks[feature].median(),
                    'std': cross_platform_tracks[feature].std()
                }
        
        # Platform-specific success patterns
        success_patterns['platform_specific'] = {}
        for platform in success_data['platform'].unique():
            platform_data = success_data[success_data['platform'] == platform]
            
            if len(platform_data) > 5:  # Minimum sample size
                success_patterns['platform_specific'][platform] = {}
                for feature in available_features:
                    success_patterns['platform_specific'][
                        platform
                    ][feature] = {
                        'mean': platform_data[feature].mean(),
                        'median': platform_data[feature].median()
                    }
        
        self.success_patterns = success_patterns
        logging.info(
            f"Analyzed success patterns for {len(success_data)} "
            "high-performing tracks"
        )
        
        return success_patterns
    
    def generate_marketing_recommendations(self, track_features=None, 
                                           artist_name=None, track_name=None):
        """Generate marketing recommendations for a specific track or general strategies"""
        logging.info("Generating marketing recommendations...")
        
        recommendations = {
            'general_strategies': self._generate_general_strategies(),
            'platform_specific': self._generate_platform_strategies(),
            'demographic_targeting': self._generate_demographic_strategies(),
            'timing_recommendations': self._generate_timing_recommendations()
        }
        
        # If specific track provided, generate personalized recommendations
        if track_features is not None:
            recommendations['personalized'] = (
                self._generate_personalized_recommendations(
                    track_features, artist_name, track_name
                )
            )
        
        self.marketing_recommendations = recommendations
        return recommendations
    
    def _generate_general_strategies(self):
        """Generate general marketing strategies based on success patterns"""
        strategies = []
        
        if hasattr(self, 'success_patterns') and \
                'high_virality_profile' in self.success_patterns:
            profile = self.success_patterns['high_virality_profile']
            
            # Energy-based strategies
            if 'energy' in profile:
                energy_mean = profile['energy']['mean']
                if energy_mean > 0.7:
                    strategies.append({
                        'strategy': 'High-Energy Campaign',
                        'description': 'Focus on energetic, dynamic content for '
                                       'high-energy tracks',
                        'tactics': ['Create dance challenges', 
                                    'Partner with fitness influencers', 
                                    'Target workout playlists']
                    })
                elif energy_mean < 0.4:
                    strategies.append({
                        'strategy': 'Chill/Ambient Marketing',
                        'description': 'Target relaxation and study contexts',
                        'tactics': ['Focus on study playlists', 
                                    'Partner with meditation apps', 
                                    'Target evening listening']
                    })
            
            # Danceability strategies
            if 'danceability' in profile:
                dance_mean = profile['danceability']['mean']
                if dance_mean > 0.6:
                    strategies.append({
                        'strategy': 'Dance-Focused Campaign', 
                        'description': 'Leverage high danceability for social '
                                       'media engagement',
                        'tactics': ['TikTok dance challenges', 
                                    'Partner with dance creators', 
                                    'Club/party playlist targeting']
                    })
            
            # Valence strategies
            if 'valence' in profile:
                valence_mean = profile['valence']['mean']
                if valence_mean > 0.6:
                    strategies.append({
                        'strategy': 'Feel-Good Marketing',
                        'description': 'Emphasize positive, uplifting messaging',
                        'tactics': ['Target happy/upbeat playlists', 
                                    'Partner with lifestyle brands', 
                                    'Morning routine content']
                    })
                elif valence_mean < 0.4:
                    strategies.append({
                        'strategy': 'Mood-Based Targeting',
                        'description': 'Target introspective and emotional '
                                       'contexts',
                        'tactics': ['Late-night listening', 
                                    'Emotional playlist targeting', 
                                    'Indie/alternative communities']
                    })
        
        return strategies
    
    def _generate_platform_strategies(self):
        """Generate platform-specific marketing strategies"""
        platform_strategies = {}
        
        if hasattr(self, 'platform_affinity') and self.platform_affinity is not None:
            for _, row in self.platform_affinity.iterrows():
                platform = row['platform']
                if platform not in platform_strategies:
                    platform_strategies[platform] = {
                        'primary_features': [],
                        'strategies': []
                    }
                
                feature = row['feature']
                correlation = row['correlation']
                
                if abs(correlation) > 0.3:  # Significant correlation
                    platform_strategies[platform]['primary_features'].append({
                        'feature': feature,
                        'correlation': correlation,
                        'importance': 'high' if abs(correlation) > 0.5 
                                      else 'moderate'
                    })
        
        # Add platform-specific strategies
        for platform in platform_strategies:
            if platform == 'tiktok':
                platform_strategies[platform]['strategies'] = [
                    'Focus on 15-30 second hooks',
                    'Create dance/challenge content',
                    'Partner with Gen Z influencers',
                    'Use trending sounds and hashtags',
                    'Optimize for mobile-first viewing'
                ]
            elif platform == 'spotify':
                platform_strategies[platform]['strategies'] = [
                    'Target playlist curators',
                    'Focus on full-song experience',
                    'Optimize for algorithm discovery',
                    'Create artist playlists',
                    'Leverage Spotify Canvas features'
                ]
            elif platform == 'youtube':
                platform_strategies[platform]['strategies'] = [
                    'Create engaging music videos',
                    'Optimize video thumbnails',
                    'Focus on storytelling',
                    'Leverage YouTube Shorts',
                    'Collaborate with music reactors'
                ]
        
        return platform_strategies
    
    def _generate_demographic_strategies(self):
        """Generate demographic-specific targeting strategies"""
        demographic_strategies = {}
        
        if hasattr(self, 'demographic_alignment') and self.demographic_alignment is not None:
            demo_cols = [
                col for col in self.demographic_alignment.columns 
                if col.endswith('_alignment')
            ]
            
            for col in demo_cols:
                demo_name = col.replace('_alignment', '')
                mean_alignment = self.demographic_alignment[col].mean()
                
                if 'Gen Z' in demo_name:
                    demographic_strategies['Gen Z'] = {
                        'avg_alignment': mean_alignment,
                        'targeting_strategy': [
                            'TikTok-first content strategy',
                            'Short-form video content',
                            'Trend-driven campaigns',
                            'Authentic, unpolished aesthetics',
                            'Social cause alignment'
                        ],
                        'key_platforms': ['TikTok', 'Instagram Reels', 
                                          'YouTube Shorts'],
                        'content_types': ['Dance challenges', 'Memes', 
                                          'Behind-the-scenes', 'Collaborations']
                    }
                elif 'Millennials' in demo_name:
                    demographic_strategies['Millennials'] = {
                        'avg_alignment': mean_alignment,
                        'targeting_strategy': [
                            'Nostalgic content themes',
                            'High-quality production',
                            'Playlist-focused marketing',
                            'Festival and live event tie-ins',
                            'Email marketing campaigns'
                        ],
                        'key_platforms': ['Spotify', 'Facebook', 'Instagram', 
                                          'YouTube'],
                        'content_types': ['Music videos', 'Acoustic sessions', 
                                          'Interviews', 'Live performances']
                    }
                elif 'Cross-Platform' in demo_name:
                    demographic_strategies['Cross-Platform'] = {
                        'avg_alignment': mean_alignment,
                        'targeting_strategy': [
                            'Integrated multi-platform campaigns',
                            'Consistent branding across platforms',
                            'Platform-specific content adaptations',
                            'Community building approach',
                            'Data-driven optimization'
                        ],
                        'key_platforms': ['All major platforms'],
                        'content_types': ['Adapted content for each platform', 
                                          'Cross-platform storytelling']
                    }
        
        return demographic_strategies
    
    def _generate_timing_recommendations(self):
        """Generate timing and release strategy recommendations"""
        timing_recommendations = {
            'optimal_release_times': {
                'TikTok': {
                    'days': ['Tuesday', 'Thursday', 'Sunday'],
                    'times': ['6-10 AM', '7-9 PM'],
                    'reasoning': 'Peak engagement times for Gen Z audience'
                },
                'Spotify': {
                    'days': ['Friday'],
                    'times': ['12 AM EST (New Music Friday)'],
                    'reasoning': 'Align with Spotify\'s algorithmic promotion'
                },
                'YouTube': {
                    'days': ['Thursday', 'Friday', 'Saturday'],
                    'times': ['2-4 PM', '8-11 PM'],
                    'reasoning': 'Peak viewing times for music content'
                }
            },
            'campaign_duration': {
                'pre_release': '2-4 weeks',
                'release_week': 'Heavy promotion across all platforms',
                'post_release': '4-6 weeks sustained promotion'
            },
            'seasonal_considerations': [
                'Summer: Focus on upbeat, high-energy tracks',
                'Fall: Transition to more introspective content',
                'Winter: Emphasize cozy, acoustic elements',
                'Spring: Fresh, optimistic messaging'
            ]
        }
        
        return timing_recommendations
    
    def _generate_personalized_recommendations(self, track_features, 
                                               artist_name=None, track_name=None):
        """Generate personalized recommendations for a specific track"""
        personalized = {
            'track_info': {
                'artist': artist_name,
                'track': track_name,
                'features': track_features
            },
            'predicted_success': {},
            'recommended_platforms': [],
            'marketing_focus': [],
            'similar_successful_tracks': []
        }
        
        # Predict success on each platform based on platform affinity
        if hasattr(self, 'platform_affinity') and self.platform_affinity is not None:
            for _, row in self.platform_affinity.iterrows():
                platform = row['platform']
                feature = row['feature']
                correlation = row['correlation']
                
                if feature in track_features:
                    feature_value = track_features[feature]
                    
                    if platform not in personalized['predicted_success']:
                        personalized['predicted_success'][platform] = 0
                    
                    # Simple prediction based on correlation and feature value
                    contribution = correlation * feature_value
                    personalized['predicted_success'][platform] += contribution
        
        # Recommend top platforms
        if personalized['predicted_success']:
            sorted_platforms = sorted(
                personalized['predicted_success'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            personalized['recommended_platforms'] = [
                p[0] for p in sorted_platforms[:3]
            ]
        
        # Generate marketing focus based on track characteristics
        if 'danceability' in track_features and \
                track_features['danceability'] > 0.7:
            personalized['marketing_focus'].append('Dance/Movement content')
        
        if 'energy' in track_features and track_features['energy'] > 0.8:
            personalized['marketing_focus'].append('High-energy campaigns')
        
        if 'acousticness' in track_features and \
                track_features['acousticness'] > 0.6:
            personalized['marketing_focus'].append('Intimate/Acoustic sessions')
        
        if 'valence' in track_features:
            if track_features['valence'] > 0.7:
                personalized['marketing_focus'].append('Feel-good messaging')
            elif track_features['valence'] < 0.3:
                personalized['marketing_focus'].append(
                    'Emotional/introspective content'
                )
        
        return personalized
    
    def create_success_prediction_model(self):
        """Create a simple model to predict track success"""
        logging.info("Creating success prediction model...")
        
        if self.comprehensive_features is None or self.virality_scores is None:
            logging.warning("Required data not available for prediction model")
            return None
        
        # Prepare training data
        model_data = self.comprehensive_features.merge(
            self.virality_scores[['track_clean', 'artist_clean', 'virality_score']],
            on=['track_clean', 'artist_clean'],
            how='inner'
        )
        
        available_features = [
            f for f in self.audio_features if f in model_data.columns
        ]
        
        if len(available_features) < 3:
            logging.warning("Insufficient features for prediction model")
            return None
        
        # Prepare features and target
        X = model_data[available_features].fillna(
            model_data[available_features].mean()
        )
        y = model_data['virality_score']
        
        # Simple clustering-based prediction (for demonstration)
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Create success tiers
        success_tiers = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
        
        self.prediction_model = {
            'scaler': scaler,
            'features': available_features,
            'success_distribution': success_tiers.value_counts().to_dict()
        }
        
        logging.info("Success prediction model created")
        return self.prediction_model
    
    def generate_comprehensive_report(self):
        """Generate comprehensive marketing insights report"""
        logging.info("Generating comprehensive marketing report...")
        
        report_path = self.output_dir / "marketing_insights_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE MARKETING INSIGHTS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Music Personas Section
            if hasattr(self, 'music_personas') and self.music_personas:
                f.write("MUSIC PERSONAS ANALYSIS\n")
                f.write("-" * 30 + "\n")
                
                for persona_name, profile in self.music_personas.items():
                    f.write(f"\n{persona_name}:\n")
                    f.write(
                        f"  Size: {profile['size']} tracks "
                        f"({profile['percentage']:.1f}%)\n"
                    )
                    f.write(f"  Avg Virality: {profile['avg_virality']:.2f}\n")
                    f.write(
                        "  Platform Distribution: "
                        f"{profile['platform_distribution']}\n"
                    )
                    
                    # Top audio characteristics
                    f.write("  Key Audio Characteristics:\n")
                    for feature, stats in profile['audio_profile'].items():
                        f.write(
                            f"    {feature}: {stats['mean']:.3f} "
                            f"(±{stats['std']:.3f})\n"
                        )
            
            # Success Patterns Section
            if hasattr(self, 'success_patterns') and self.success_patterns:
                f.write("\nSUCCESS PATTERNS ANALYSIS\n")
                f.write("-" * 30 + "\n")
                
                if 'high_virality_profile' in self.success_patterns:
                    f.write("\nHigh Virality Track Profile:\n")
                    profile = self.success_patterns['high_virality_profile']
                    for feature, stats in profile.items():
                        f.write(
                            f"  {feature}: {stats['mean']:.3f} "
                            f"(median: {stats['median']:.3f})\n"
                        )
                
                if 'platform_specific' in self.success_patterns:
                    f.write("\nPlatform-Specific Success Patterns:\n")
                    patterns = self.success_patterns['platform_specific']
                    for platform, features in patterns.items():
                        f.write(f"\n  {platform.upper()}:\n")
                        for feature, stats in features.items():
                            f.write(f"    {feature}: {stats['mean']:.3f}\n")
            
            # Marketing Recommendations Section
            if hasattr(self, 'marketing_recommendations') and self.marketing_recommendations:
                f.write("\nMARKETING RECOMMENDATIONS\n")
                f.write("-" * 35 + "\n")
                
                # General Strategies
                if 'general_strategies' in self.marketing_recommendations:
                    f.write("\nGeneral Marketing Strategies:\n")
                    strategies = self.marketing_recommendations['general_strategies']
                    for i, strategy in enumerate(strategies, 1):
                        f.write(f"\n{i}. {strategy['strategy']}\n")
                        f.write(
                            f"   Description: {strategy['description']}\n"
                        )
                        f.write(
                            f"   Tactics: {', '.join(strategy['tactics'])}\n"
                        )
                
                # Platform Strategies
                if 'platform_specific' in self.marketing_recommendations:
                    f.write("\nPlatform-Specific Strategies:\n")
                    strategies = self.marketing_recommendations['platform_specific']
                    for platform, details in strategies.items():
                        f.write(f"\n{platform.upper()}:\n")
                        f.write(
                            f"  Strategies: {', '.join(details['strategies'])}\n"
                        )
                        if 'primary_features' in details:
                            f.write("  Key Features: ")
                            features = [
                                f"{feat['feature']} ({feat['correlation']:.3f})" 
                                for feat in details['primary_features']
                            ]
                            f.write(f"{', '.join(features)}\n")
                
                # Demographic Strategies
                if 'demographic_targeting' in self.marketing_recommendations:
                    f.write("\nDemographic Targeting Strategies:\n")
                    strategies = self.marketing_recommendations['demographic_targeting']
                    for demo, details in strategies.items():
                        f.write(f"\n{demo}:\n")
                        f.write(
                            "  Alignment Score: "
                            f"{details['avg_alignment']:.3f}\n"
                        )
                        f.write(
                            "  Key Platforms: "
                            f"{', '.join(details['key_platforms'])}\n"
                        )
                        f.write(
                            "  Content Types: "
                            f"{', '.join(details['content_types'])}\n"
                        )
        
        logging.info(f"Comprehensive report saved to {report_path}")
    
    def create_visualization_dashboard(self):
        """Create visualization dashboard for marketing insights"""
        logging.info("Creating visualization dashboard...")
        
        # Set up the plotting style
        plt.style.use('default')
        plt.figure(figsize=(20, 16))
        
        # Music Personas Distribution
        if hasattr(self, 'music_personas') and self.music_personas:
            plt.subplot(2, 3, 1)
            persona_sizes = [p['size'] for p in self.music_personas.values()]
            persona_names = list(self.music_personas.keys())
            
            plt.pie(
                persona_sizes, labels=persona_names, 
                autopct='%1.1f%%', startangle=90
            )
            plt.title(
                'Music Personas Distribution', fontsize=14, fontweight='bold'
            )
        
        # Virality Score Distribution
        if hasattr(self, 'virality_scores') and self.virality_scores is not None:
            plt.subplot(2, 3, 2)
            plt.hist(
                self.virality_scores['virality_score'], bins=20, alpha=0.7, 
                color='skyblue', edgecolor='black'
            )
            plt.xlabel('Virality Score')
            plt.ylabel('Frequency')
            plt.title(
                'Virality Score Distribution', fontsize=14, fontweight='bold'
            )
            plt.grid(True, alpha=0.3)
        
        # Platform Affinity Heatmap
        if hasattr(self, 'platform_affinity') and self.platform_affinity is not None:
            plt.subplot(2, 3, 3)
            
            # Create pivot table for heatmap
            pivot_data = self.platform_affinity.pivot(
                index='platform', columns='feature', values='correlation'
            )
            
            sns.heatmap(
                pivot_data, annot=True, cmap='RdBu_r', center=0, 
                fmt='.2f', square=True, linewidths=0.5
            )
            plt.title(
                'Platform-Feature Affinity Heatmap', fontsize=14, 
                fontweight='bold'
            )
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        
        # Success Patterns Radar Chart (simplified)
        if hasattr(self, 'success_patterns') and self.success_patterns and \
                'high_virality_profile' in self.success_patterns:
            profile = self.success_patterns['high_virality_profile']
            features = list(profile.keys())[:6]  # Limit to 6 features
            values = [profile[f]['mean'] for f in features]
            
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            ax = plt.subplot(2, 3, 4, projection='polar')
            ax.plot(angles, values, 'o-', linewidth=2, color='red', alpha=0.7)
            ax.fill(angles, values, alpha=0.25, color='red')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features)
            ax.set_title(
                'High Virality Track Profile', fontsize=14, 
                fontweight='bold', pad=20
            )
        
        # Cross-Platform Success Analysis
        if hasattr(self, 'comprehensive_features') and self.comprehensive_features is not None:
            plt.subplot(2, 3, 5)
            
            if 'is_cross_platform' in self.comprehensive_features.columns:
                # Check if virality_score is available
                if 'virality_score' in self.comprehensive_features.columns:
                    grouped = self.comprehensive_features.groupby('is_cross_platform')
                    cross_platform_data = grouped['virality_score'].mean()
                else:
                    # Use popularity_score as fallback
                    grouped = self.comprehensive_features.groupby('is_cross_platform')
                    cross_platform_data = grouped['popularity_score'].mean()
                
                labels = (['Single Platform', 'Cross Platform'] 
                          if True in cross_platform_data.index 
                          else ['Single Platform'])
                values = cross_platform_data.values
                
                bars = plt.bar(
                    labels, values, 
                    color=['lightcoral', 'lightgreen'][:len(values)], 
                    alpha=0.7, edgecolor='black'
                )
                plt.ylabel('Average Success Score')
                plt.title(
                    'Cross-Platform vs Single Platform Success', fontsize=14, 
                    fontweight='bold'
                )
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom'
                    )
        
        # Demographic Alignment Scores
        if hasattr(self, 'demographic_alignment') and self.demographic_alignment is not None:
            plt.subplot(2, 3, 6)
            
            demo_cols = [
                col for col in self.demographic_alignment.columns 
                if col.endswith('_alignment')
            ]
            demo_names = [
                col.replace('_alignment', '').replace(
                    '(TikTok-focused)', ''
                ).replace('(Spotify-focused)', '') for col in demo_cols
            ]
            demo_scores = [self.demographic_alignment[col].mean() for col in demo_cols]
            
            bars = plt.barh(
                demo_names, demo_scores, color='gold', 
                alpha=0.7, edgecolor='black'
            )
            plt.xlabel('Average Alignment Score')
            plt.title(
                'Demographic Alignment Scores', fontsize=14, fontweight='bold'
            )
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(
                    width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{demo_scores[i]:.2f}', ha='left', va='center'
                )
        
        plt.tight_layout()
        
        # Save the dashboard
        dashboard_path = self.output_dir / "marketing_insights_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Visualization dashboard saved to {dashboard_path}")
    
    def run_complete_analysis(self):
        """Run complete marketing insights analysis"""
        logging.info("Starting complete marketing insights analysis...")
        
        try:
            self.load_cross_platform_data()
            
            self.create_music_personas()
            
            self.analyze_success_patterns()
            
            self.generate_marketing_recommendations()
            
            self.create_success_prediction_model()
            
            self.generate_comprehensive_report()

            self.create_visualization_dashboard()
            
            logging.info(
                "Complete marketing insights analysis finished successfully!"
            )
            
            return {
                'personas_created': len(self.music_personas) 
                if hasattr(self, 'music_personas') and self.music_personas 
                else 0,
                'success_patterns_analyzed': len(self.success_patterns) 
                if hasattr(self, 'success_patterns') and self.success_patterns 
                else 0,
                'recommendations_generated': len(self.marketing_recommendations) 
                if hasattr(self, 'marketing_recommendations') and self.marketing_recommendations 
                else 0,
                'prediction_model_created': hasattr(self, 'prediction_model'),
                'output_directory': str(self.output_dir)
            }
            
        except Exception as e:
            logging.error(f"Error in complete analysis: {e}")
            raise
    
    def predict_track_success(self, track_features, 
                            artist_name=None, track_name=None):
        """Predict success for a new track"""
        if not hasattr(self, 'prediction_model'):
            logging.warning("Prediction model not available. Creating one...")
            self.create_success_prediction_model()
        
        if not hasattr(self, 'prediction_model') or self.prediction_model is None:
            return None
        
        recommendations = self.generate_marketing_recommendations(
            track_features=track_features,
            artist_name=artist_name,
            track_name=track_name
        )
        
        return recommendations.get('personalized', {})
    
    def export_marketing_data(self, format='csv'):
        """Export marketing insights data in various formats"""
        logging.info(f"Exporting marketing data in {format} format...")
        
        exports = {}
        
        if hasattr(self, 'music_personas') and self.music_personas:
            personas_data = []
            for persona_name, profile in self.music_personas.items():
                row = {
                    'persona_name': persona_name,
                    'cluster_id': profile['cluster_id'],
                    'size': profile['size'],
                    'percentage': profile['percentage'],
                    'avg_virality': profile['avg_virality']
                }
                
                for feature, stats in profile['audio_profile'].items():
                    row[f'{feature}_mean'] = stats['mean']
                    row[f'{feature}_std'] = stats['std']
                
                personas_data.append(row)
            
            personas_df = pd.DataFrame(personas_data)
            
            if format == 'csv':
                export_path = self.output_dir / "music_personas_export.csv"
                personas_df.to_csv(export_path, index=False)
                exports['personas'] = export_path
            elif format == 'json':
                export_path = self.output_dir / "music_personas_export.json"
                personas_df.to_json(export_path, orient='records', indent=2)
                exports['personas'] = export_path
        
        # Export success patterns
        if hasattr(self, 'success_patterns') and self.success_patterns:
            export_path = self.output_dir / "success_patterns_export.json"
            with open(export_path, 'w') as f:
                json.dump(self.success_patterns, f, indent=2, default=str)
            exports['success_patterns'] = export_path
        
        # Export marketing recommendations
        if hasattr(self, 'marketing_recommendations') and self.marketing_recommendations:
            export_path = (
                self.output_dir / "marketing_recommendations_export.json"
            )
            with open(export_path, 'w') as f:
                json.dump(
                    self.marketing_recommendations, f, indent=2, default=str
                )
            exports['recommendations'] = export_path
        
        logging.info(f"Data exported to: {list(exports.values())}")
        return exports

# Example usage and testing
if __name__ == "__main__":
    # --- Command-Line Interface Setup ---
    # To use this script from the command line, you can run:
    #
    # 1. To run the complete analysis:
    #    python scripts/features/generate_marketing_recommendations.py analyze
    #
    # 2. To predict success for a new track:
    #    python scripts/features/generate_marketing_recommendations.py predict --track-features '{"danceability": 0.8, "energy": 0.9, "valence": 0.7}'
    #
    # You can also specify the data directory, artist, and track name:
    #    python scripts/features/generate_marketing_recommendations.py predict --data-dir "path/to/data" \
    #    --artist-name "New Artist" --track-name "Hit Song" \
    #    --track-features '{"danceability": 0.8, "energy": 0.9, "loudness": -5.0, "speechiness": 0.1, "acousticness": 0.2, "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.7, "tempo": 128.0}'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Marketing Insights Generator for music tracks."
    )
    parser.add_argument(
        '--data-dir', type=str, default='data', 
        help='The directory where the data is stored.'
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Sub-parser for the 'analyze' command
    subparsers.add_parser(
        'analyze', help='Run the complete analysis and generate reports.'
    )

    # Sub-parser for the 'predict' command
    predict_parser = subparsers.add_parser(
        'predict', help='Predict success for a single track.'
    )
    predict_parser.add_argument(
        '--track-features', type=str, required=True,
        help='A JSON string of audio features for the track.'
    )
    predict_parser.add_argument(
        '--artist-name', type=str, default='Sample Artist',
        help='The name of the artist.'
    )
    predict_parser.add_argument(
        '--track-name', type=str, default='Sample Track',
        help='The name of the track.'
    )

    args = parser.parse_args()
    
    insights_generator = MarketingInsightsGenerator(data_dir=args.data_dir)
    
    try:
        if args.command == 'analyze':
            results = insights_generator.run_complete_analysis()
            
            print("*** Marketing Insights Analysis Complete! ***")
            print(f"Results: {results}")

            exports = insights_generator.export_marketing_data(format='csv')
            print(f"\nExported files: {exports}")
        
        elif args.command == 'predict':
            print("Loading existing analysis data to make prediction...")
            insights_generator.load_cross_platform_data()
            insights_generator.create_success_prediction_model()
            
            try:
                track_features = json.loads(args.track_features)
            except json.JSONDecodeError:
                logging.error("Invalid JSON format for --track-features.")
                raise
            
            prediction = insights_generator.predict_track_success(
                track_features=track_features,
                artist_name=args.artist_name,
                track_name=args.track_name
            )
            
            if prediction:
                print("\n*** Personalized Marketing Recommendations ***")
                track_info = prediction.get('track_info', {})
                print(f"  Artist: {track_info.get('artist')}")
                print(f"  Track: {track_info.get('track')}")
                
                print("\n  Recommended Platforms:")
                for platform in prediction.get('recommended_platforms', []):
                    print(f"    - {platform.capitalize()}")
                    
                print("\n  Marketing Focus:")
                for focus in prediction.get('marketing_focus', []):
                    print(f"    - {focus}")
                print("\n*** End of Personalized Marketing Recommendations ***")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.error(f"Analysis failed: {e}", exc_info=True)