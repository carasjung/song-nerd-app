# integrated_analyzer.py
import pandas as pd
import numpy as np
from demographics_model_adapted import DemographicsPredictor
from robust_platform_model import RobustPlatformRecommender
from similar_artists_adapted import SimilarArtistFinder

class MusicMarketingAnalyzer:
    def __init__(self):
        self.demographics_model = DemographicsPredictor()
        self.platform_model = RobustPlatformRecommender()
        self.similar_artists_model = SimilarArtistFinder()
        self.models_loaded = False
        
    def load_models(self, models_dir='models/'):
        """Load all trained models"""
        try:
            self.demographics_model.load_model(f'{models_dir}demographics_predictor.pkl')
            self.platform_model.load_model(f'{models_dir}robust_platform_recommender.pkl')
            self.similar_artists_model.load_model(f'{models_dir}similar_artists.pkl')
            self.models_loaded = True
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure you've run the complete training pipeline first!")
            self.models_loaded = False
        
        return self
    
    def analyze_song(self, audio_features, song_metadata=None):
        """Complete marketing analysis for a song"""
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Ensure audio_features is a DataFrame
        if isinstance(audio_features, dict):
            audio_features = pd.DataFrame([audio_features])
        
        # Add required features if missing
        required_features = ['danceability', 'energy', 'valence', 'acousticness', 
                           'instrumentalness', 'liveness', 'speechiness']
        
        for feature in required_features:
            if feature not in audio_features.columns:
                audio_features[feature] = 0.5  # Default middle value
        
        # Add additional features expected by models
        if 'audio_appeal' not in audio_features.columns:
            # Calculate audio appeal based on features
            audio_features['audio_appeal'] = (
                audio_features['energy'] * 0.3 +
                audio_features['valence'] * 0.3 +
                audio_features['danceability'] * 0.4
            ) * 100
        
        if 'normalized_popularity' not in audio_features.columns:
            audio_features['normalized_popularity'] = 0.5  # Default
        
        if 'genre_clean' not in audio_features.columns and song_metadata and 'genre' in song_metadata:
            audio_features['genre_clean'] = song_metadata['genre']
        elif 'genre_clean' not in audio_features.columns:
            audio_features['genre_clean'] = 'pop'  # Default genre
        
        try:
            # Get demographics predictions 
            demographics = self.demographics_model.predict(audio_features)
            
            # Get platform recommendations
            platforms = self.platform_model.predict(audio_features)
            
            # Get similar artists
            input_dict = audio_features.iloc[0].to_dict()
            similar_artists = self.similar_artists_model.find_similar_artists(
                input_dict, top_k=8
            )
            
            # Generate marketing insights
            marketing_insights = self._generate_marketing_insights(
                demographics, platforms, similar_artists, audio_features.iloc[0]
            )
            
            # Compile complete analysis
            analysis = {
                'song_info': song_metadata or {},
                'audio_features': audio_features.iloc[0].to_dict(),
                'target_demographics': demographics,
                'platform_recommendations': platforms,
                'similar_artists': similar_artists,
                'marketing_insights': marketing_insights,
                'confidence_scores': {
                    'demographics': {
                        'age': demographics['confidence_scores']['age'],
                        'region': demographics['confidence_scores']['region']
                    },
                    'platforms': platforms['top_score'] / 100,
                    'platform_success': platforms['ranked_recommendations'][0]['success_probability'],
                    'similar_artists': similar_artists['similar_artists'][0]['similarity_score'] if similar_artists['similar_artists'] else 0
                },
                'analysis_summary': self._generate_summary(demographics, platforms, similar_artists)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return self._generate_fallback_analysis(audio_features, song_metadata)
    
    def _generate_marketing_insights(self, demographics, platforms, similar_artists, audio_features):
        """Generate comprehensive marketing insights"""
        insights = {
            'target_audience': {
                'primary': f"{demographics['primary_age_group']} in {demographics['primary_region']}",
                'confidence': f"{demographics['confidence_scores']['age']:.1%}",
                'secondary_markets': [
                    f"{age} ({prob:.1%})" 
                    for age, prob in sorted(demographics['age_groups'].items(), 
                                          key=lambda x: x[1], reverse=True)[1:4]
                ]
            },
            'platform_strategy': {
                'primary_platform': platforms['top_platform'],
                'expected_performance': f"{platforms['top_score']:.0f}/100",
                'success_probability': f"{platforms['ranked_recommendations'][0]['success_probability']:.1%}",
                'confidence': platforms['ranked_recommendations'][0]['confidence'],
                'strategy': platforms['ranked_recommendations'][0]['recommendation'],
                'platform_breakdown': [
                    {
                        'platform': rec['platform'],
                        'score': f"{rec['score']:.0f}/100",
                        'success_probability': f"{rec['success_probability']:.1%}",
                        'confidence': rec['confidence'],
                        'strategy': rec['recommendation']
                    }
                    for rec in platforms['ranked_recommendations']
                ],
                'platform_insights': self._generate_platform_insights(platforms, audio_features)
            },
            'positioning': {
                'similar_artists': [
                    f"{artist['artist_name']} ({artist['similarity_score']:.2f} similarity)"
                    for artist in similar_artists['similar_artists'][:3]
                ],
                'genre_positioning': similar_artists['similar_artists'][0]['genre'] if similar_artists['similar_artists'] else 'Unique',
                'sound_profile': self._describe_sound_profile(audio_features),
                'competitive_advantage': self._identify_competitive_advantage(audio_features, similar_artists)
            },
            'action_items': self._generate_action_items(demographics, platforms, similar_artists, audio_features)
        }
        
        return insights
    
    def _generate_platform_insights(self, platforms, audio_features):
        """Generate platform-specific insights based on robust model predictions"""
        insights = {}
        
        for platform_data in platforms['ranked_recommendations']:
            platform = platform_data['platform']
            score = platform_data['score']
            success_prob = platform_data['success_probability']
            
            if platform == 'tiktok':
                if audio_features['danceability'] > 0.7 and success_prob > 0.7:
                    insights[platform] = "High viral potential - create dance challenges and trending content"
                elif audio_features['energy'] > 0.7 and success_prob > 0.5:
                    insights[platform] = "Good energy for fitness/workout content and energetic trends"
                elif success_prob > 0.3:
                    insights[platform] = "Moderate potential - focus on creative storytelling or niche trends"
                else:
                    insights[platform] = "Limited TikTok appeal - consider other platforms first"
                    
            elif platform == 'spotify':
                if audio_features['valence'] > 0.6 and success_prob > 0.7:
                    insights[platform] = "Excellent for mood-based and feel-good playlists"
                elif audio_features['acousticness'] > 0.5 and success_prob > 0.5:
                    insights[platform] = "Strong fit for acoustic, chill, and indie playlists"
                elif success_prob > 0.4:
                    insights[platform] = "Good playlist potential - target genre-specific and algorithmic playlists"
                else:
                    insights[platform] = "Consider playlist pitching and acoustic versions"
                    
            elif platform == 'youtube':
                if audio_features['energy'] > 0.6 and success_prob > 0.7:
                    insights[platform] = "High potential for engaging music videos and visual content"
                elif audio_features['liveness'] > 0.3 and success_prob > 0.5:
                    insights[platform] = "Great for live performance videos and session content"
                elif success_prob > 0.4:
                    insights[platform] = "Good for lyric videos and storytelling content"
                else:
                    insights[platform] = "Focus on other platforms or try creative video concepts"
        
        return insights
    
    def _identify_competitive_advantage(self, audio_features, similar_artists):
        """Identify what makes this song unique"""
        if not similar_artists['similar_artists']:
            return "Unique sound with no close comparisons"
        
        advantages = []
        
        if audio_features['energy'] > 0.8:
            advantages.append("exceptionally high energy")
        if audio_features['danceability'] > 0.85:
            advantages.append("outstanding danceability")
        if audio_features['valence'] > 0.8:
            advantages.append("extremely positive mood")
        if audio_features['audio_appeal'] > 85:
            advantages.append("premium production quality")
        
        # Compare to top similar artist
        if similar_artists['similar_artists'] and len(advantages) == 0:
            similarity = similar_artists['similar_artists'][0]['similarity_score']
            if similarity < 0.8:
                advantages.append("distinctive sound profile")
        
        if advantages:
            return f"Key strengths: {', '.join(advantages)}"
        else:
            return "Well-balanced sound with broad appeal"
    
    def _generate_action_items(self, demographics, platforms, similar_artists, audio_features):
        """Generate specific, prioritized marketing actions"""
        actions = []
        
        # Platform-specific actions based on success probability
        top_platform = platforms['ranked_recommendations'][0]
        top_success_prob = top_platform['success_probability']
        
        if top_success_prob > 0.7:
            actions.append(f"HIGH PRIORITY: Focus marketing budget on {top_platform['platform'].title()} - {top_success_prob:.0%} success probability")
        elif top_success_prob > 0.4:
            actions.append(f"MEDIUM PRIORITY: Test {top_platform['platform'].title()} with modest budget - {top_success_prob:.0%} success probability")
        else:
            actions.append(f"LOW PRIORITY: Consider {top_platform['platform'].title()} after other platforms")
        
        # Demographics-based actions
        top_age = demographics['primary_age_group']
        age_confidence = demographics['confidence_scores']['age']
        if age_confidence > 0.8:
            actions.append(f"Target {top_age} demographic in advertising (high confidence: {age_confidence:.0%})")
        
        # Content creation actions based on audio features
        if audio_features['danceability'] > 0.75:
            actions.append("Create dance/choreography content for social media")
        
        if audio_features['valence'] > 0.7:
            actions.append("Pitch to upbeat, feel-good, and motivational playlists")
        
        if audio_features['energy'] > 0.7:
            actions.append("Target fitness, workout, and high-energy playlist curators")
        
        # Similar artists strategy
        if similar_artists['similar_artists']:
            top_similar = similar_artists['similar_artists'][0]
            if top_similar['similarity_score'] > 0.85:
                actions.append(f"Study {top_similar['artist_name']}'s recent campaigns and fan engagement")
                actions.append(f"Target playlists and audiences that feature {top_similar['artist_name']}")
        
        # Platform-specific content actions
        for rec in platforms['ranked_recommendations'][:2]:
            if rec['success_probability'] > 0.5:
                if rec['platform'] == 'tiktok':
                    actions.append("Create 15-30 second hook previews optimized for TikTok")
                elif rec['platform'] == 'spotify':
                    actions.append("Submit for Spotify Release Radar and Discover Weekly consideration")
                elif rec['platform'] == 'youtube':
                    actions.append("Plan high-quality music video or engaging visualizer")
        
        return actions[:8]  # Return top 8 most important actions
    
    def _describe_sound_profile(self, features):
        """Generate human-readable sound description"""
        descriptions = []
        
        # Energy level
        if features['energy'] > 0.75:
            descriptions.append("high-energy")
        elif features['energy'] < 0.35:
            descriptions.append("mellow")
        
        # Mood
        if features['valence'] > 0.7:
            descriptions.append("very upbeat")
        elif features['valence'] > 0.5:
            descriptions.append("positive")
        elif features['valence'] < 0.3:
            descriptions.append("melancholic")
        
        # Danceability
        if features['danceability'] > 0.75:
            descriptions.append("highly danceable")
        elif features['danceability'] > 0.5:
            descriptions.append("moderately danceable")
        
        # Sound characteristics
        if features['acousticness'] > 0.6:
            descriptions.append("acoustic")
        elif features['acousticness'] < 0.2:
            descriptions.append("electronic")
        
        if features['instrumentalness'] < 0.1:
            descriptions.append("vocal-focused")
        
        if len(descriptions) == 0:
            descriptions.append("balanced")
        
        return ", ".join(descriptions[:4])
    
    def _generate_summary(self, demographics, platforms, similar_artists):
        """Generate executive summary with robust platform insights"""
        top_platform_data = platforms['ranked_recommendations'][0]
        success_prob = top_platform_data['success_probability']
        
        confidence_level = 'high' if success_prob > 0.7 else 'medium' if success_prob > 0.4 else 'low'
        
        summary = {
            'headline': f"Target {demographics['primary_age_group']} on {platforms['top_platform'].title()} ({platforms['top_score']:.0f}/100, {success_prob:.0%} success probability)",
            'key_insight': f"{confidence_level.title()} confidence recommendation - {top_platform_data['recommendation']}",
            'competitive_positioning': f"Similar to {similar_artists['similar_artists'][0]['artist_name']}" if similar_artists['similar_artists'] else "Unique positioning opportunity",
            'confidence_level': confidence_level,
            'success_indicators': {
                'platform_fit': platforms['top_score'] / 100,
                'success_probability': success_prob,
                'demographic_confidence': demographics['confidence_scores']['age']
            },
            'next_steps': f"Prioritize {platforms['top_platform']}-specific content creation with {demographics['primary_age_group']} targeting"
        }
        
        return summary
    
    def _generate_fallback_analysis(self, audio_features, song_metadata, error_msg):
        """Generate basic analysis if models fail"""
        
        # Create basic insights from audio features
        features = audio_features.iloc[0]
        
        # Basic platform recommendations based on audio features
        basic_platform_scores = {
            'tiktok': min(100, (features.get('danceability', 0.5) * 40 + 
                               features.get('energy', 0.5) * 35 + 
                               features.get('valence', 0.5) * 25)),
            'spotify': min(100, (features.get('valence', 0.5) * 30 + 
                                features.get('energy', 0.5) * 25 + 
                                features.get('audio_appeal', 50) * 0.45)),
            'youtube': min(100, (features.get('energy', 0.5) * 35 + 
                                features.get('valence', 0.5) * 30 + 
                                (1 - features.get('instrumentalness', 0.5)) * 35))
        }
        
        # Sort platforms by score
        sorted_platforms = sorted(basic_platform_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'song_info': song_metadata or {},
            'audio_features': features.to_dict(),
            'error': f'Advanced model analysis failed: {error_msg}',
            'basic_insights': {
                'energy_level': 'high' if features.get('energy', 0) > 0.7 else 'moderate' if features.get('energy', 0) > 0.4 else 'low',
                'mood': 'positive' if features.get('valence', 0) > 0.6 else 'neutral' if features.get('valence', 0) > 0.4 else 'melancholic',
                'danceability': 'high' if features.get('danceability', 0) > 0.7 else 'moderate' if features.get('danceability', 0) > 0.4 else 'low',
                'sound_type': 'acoustic' if features.get('acousticness', 0) > 0.6 else 'electronic' if features.get('acousticness', 0) < 0.2 else 'balanced'
            },
            'basic_platform_recommendations': [
                {
                    'platform': platform,
                    'score': f"{score:.0f}/100",
                    'reason': self._get_basic_platform_reason(platform, features)
                }
                for platform, score in sorted_platforms
            ],
            'analysis_summary': {
                'headline': f"Basic analysis: {sorted_platforms[0][0].title()} recommended ({sorted_platforms[0][1]:.0f}/100)",
                'key_insight': f"Song has {self._get_sound_description(features)} characteristics",
                'competitive_positioning': "Run full analysis for similar artists",
                'confidence_level': 'basic',
                'next_steps': "Train models for detailed analysis using: python3 complete_training_pipeline.py"
            },
            'confidence_scores': {
                'demographics': {'age': 0.0, 'region': 0.0},
                'platforms': sorted_platforms[0][1] / 100,
                'platform_success': 0.0,
                'similar_artists': 0.0
            },
            'marketing_insights': {
                'target_audience': {
                    'primary': 'General audience (run full analysis for specifics)',
                    'confidence': '0%',
                    'secondary_markets': ['Requires model training']
                },
                'platform_strategy': {
                    'primary_platform': sorted_platforms[0][0],
                    'expected_performance': f"{sorted_platforms[0][1]:.0f}/100",
                    'success_probability': 'Unknown',
                    'confidence': 'basic',
                    'strategy': f"Basic recommendation for {sorted_platforms[0][0]}",
                    'platform_breakdown': [
                        {
                            'platform': platform,
                            'score': f"{score:.0f}/100",
                            'success_probability': 'Unknown',
                            'confidence': 'basic',
                            'strategy': self._get_basic_platform_reason(platform, features)
                        }
                        for platform, score in sorted_platforms
                    ]
                },
                'positioning': {
                    'similar_artists': ['Requires model training'],
                    'genre_positioning': features.get('genre_clean', 'Unknown'),
                    'sound_profile': self._get_sound_description(features),
                    'competitive_advantage': 'Run full analysis for competitive insights'
                },
                'action_items': [
                    f"Focus on {sorted_platforms[0][0]} as primary platform",
                    "Train models for detailed demographic targeting",
                    "Run complete analysis for similar artist insights",
                    "Develop platform-specific content strategy"
                ]
            }
        }
    
    def _get_basic_platform_reason(self, platform, features):
        """Get basic reasoning for platform recommendation"""
        if platform == 'tiktok':
            if features.get('danceability', 0) > 0.7:
                return "High danceability suggests good TikTok potential"
            elif features.get('energy', 0) > 0.7:
                return "High energy works well for TikTok content"
            else:
                return "Moderate TikTok potential"
        elif platform == 'spotify':
            if features.get('valence', 0) > 0.6:
                return "Positive mood fits Spotify playlists well"
            elif features.get('acousticness', 0) > 0.5:
                return "Acoustic elements work well on Spotify"
            else:
                return "Good general Spotify appeal"
        elif platform == 'youtube':
            if features.get('energy', 0) > 0.6:
                return "High energy content engages YouTube audiences"
            elif features.get('instrumentalness', 0) < 0.3:
                return "Vocal content works well for YouTube"
            else:
                return "Good YouTube potential"
        return f"Basic recommendation for {platform}"
    
    def _get_sound_description(self, features):
        """Get basic sound description"""
        descriptors = []
        
        if features.get('energy', 0) > 0.7:
            descriptors.append("high-energy")
        elif features.get('energy', 0) < 0.3:
            descriptors.append("mellow")
        
        if features.get('valence', 0) > 0.7:
            descriptors.append("upbeat")
        elif features.get('valence', 0) < 0.3:
            descriptors.append("moody")
        
        if features.get('danceability', 0) > 0.7:
            descriptors.append("danceable")
        
        if features.get('acousticness', 0) > 0.6:
            descriptors.append("acoustic")
        elif features.get('acousticness', 0) < 0.2:
            descriptors.append("electronic")
        
        return ", ".join(descriptors) if descriptors else "balanced"

# Test script for the complete analyzer
def test_analyzer():
    """Test the complete marketing analyzer with sample data"""
    
    # Sample audio features (danceable pop song) - include ALL required features
    sample_features = {
        'danceability': 0.85,
        'energy': 0.75,
        'valence': 0.70,
        'acousticness': 0.12,
        'instrumentalness': 0.03,
        'liveness': 0.15,
        'speechiness': 0.06,
        'audio_appeal': 82,
        'normalized_popularity': 0.6,
        'genre_clean': 'pop',
        'spotify': 0,      # Add these explicitly
        'tiktok': 0,       # Add these explicitly  
        'youtube': 0       # Add these explicitly
    }
    
    sample_metadata = {
        'track_name': 'Dance Floor Anthem',
        'artist_name': 'Rising Star',
        'genre': 'pop'
    }
    
    print("Testing Complete Music Marketing Analyzer...")
    print("=" * 60)
    
    # Initialize and test analyzer
    analyzer = MusicMarketingAnalyzer()
    analyzer.load_models()
    
    if analyzer.models_loaded:
        results = analyzer.analyze_song(sample_features, sample_metadata)
        
        print("COMPLETE MARKETING ANALYSIS RESULTS")
        print("=" * 50)
        print(f" {results['analysis_summary']['headline']}")
        print(f" {results['analysis_summary']['key_insight']}")
        print(f" {results['analysis_summary']['competitive_positioning']}")
        
        print(f"\n Platform Strategy:")
        for platform in results['marketing_insights']['platform_strategy']['platform_breakdown']:
            print(f"  • {platform['platform'].title()}: {platform['score']} "
                  f"(Success: {platform['success_probability']}, {platform['confidence']})")
        
        print(f"\n Top Action Items:")
        for i, action in enumerate(results['marketing_insights']['action_items'][:4], 1):
            print(f"  {i}. {action}")
        
        print(f"\n Confidence Scores:")
        print(f"  • Demographics: {results['confidence_scores']['demographics']['age']:.1%}")
        print(f"  • Platform Score: {results['confidence_scores']['platforms']:.1%}")
        print(f"  • Success Probability: {results['confidence_scores']['platform_success']:.1%}")
        print(f"  • Similar Artists: {results['confidence_scores']['similar_artists']:.1%}")
        
        print(f"\n Sound Profile: {results['marketing_insights']['positioning']['sound_profile']}")
        print(f" Competitive Advantage: {results['marketing_insights']['positioning']['competitive_advantage']}")
        
        return results
    else:
        print(" Models not loaded properly. Please run complete training pipeline first:")
        print("   python3 complete_training_pipeline.py")
        return None

if __name__ == "__main__":
    results = test_analyzer()