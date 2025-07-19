# ar_simulation_system.py
# Simulates record label talent discovery and market analysis

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ARSimulationSystem:
    def __init__(self, dataset_path='integration_cache/master_music_dataset_deduplicated.csv'):
        self.dataset_path = dataset_path
        self.df = None
        
        # General A&R criteria weights 
        self.ar_criteria = {
            'commercial_potential': 0.35,    # Popularity, platform presence
            'musical_quality': 0.25,         # Audio features, production quality
            'market_fit': 0.20,              # Genre trends, demographic appeal
            'uniqueness': 0.15,              # Standout features, originality
            'scalability': 0.05              # Multi-platform presence, source diversity
        }
        
        # Audio features that indicate commercial potential
        self.commercial_features = {
            'mainstream_appeal': ['danceability', 'energy', 'valence'],
            'production_quality': ['loudness', 'acousticness'],
            'catchiness': ['tempo', 'speechiness'],
            'emotional_impact': ['valence', 'energy']
        }
        
        # Market trend indicators
        self.market_trends = {
            'pop': {'weight': 1.2, 'features': {'danceability': 0.7, 'energy': 0.6, 'valence': 0.6}},
            'hip hop': {'weight': 1.1, 'features': {'danceability': 0.8, 'speechiness': 0.4, 'energy': 0.7}},
            'electronic': {'weight': 1.1, 'features': {'danceability': 0.8, 'energy': 0.9, 'acousticness': 0.1}},
            'rock': {'weight': 0.9, 'features': {'energy': 0.8, 'acousticness': 0.3, 'loudness': -5}},
            'acoustic': {'weight': 0.8, 'features': {'acousticness': 0.8, 'energy': 0.3, 'valence': 0.5}}
        }
    
    def load_dataset(self):
        """Load the master dataset"""
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found: {self.dataset_path}")
            return False
        
        print(f"Loading dataset: {os.path.basename(self.dataset_path)}")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Loaded {len(self.df):,} tracks from {self.df['artist_name'].nunique():,} artists")
        return True
    
    def calculate_commercial_potential(self):
        """Calculate commercial potential score for each track"""
        print("Calculating commercial potential scores...")
        
        scores = []
        
        for _, track in self.df.iterrows():
            score = 0
            
            # Popularity score (40% of commercial potential)
            popularity = track.get('normalized_popularity', 0)
            if pd.notna(popularity):
                popularity_score = min(popularity / 100, 1.0)  # Normalize to 0-1
            else:
                popularity_score = 0
            score += popularity_score * 0.4
            
            # Platform presence (30% of commercial potential)
            platform_count = sum([
                track.get('spotify', False),
                track.get('tiktok', False),
                track.get('youtube', False)
            ])
            platform_score = min(platform_count / 3, 1.0)  # Normalize to 0-1
            score += platform_score * 0.3
            
            # Audio appeal (30% of commercial potential)
            appeal_features = ['danceability', 'energy', 'valence']
            appeal_values = []
            
            for feature in appeal_features:
                value = track.get(feature)
                if pd.notna(value):
                    appeal_values.append(value)
            
            if appeal_values:
                appeal_score = np.mean(appeal_values)
                score += appeal_score * 0.3
            
            scores.append(min(score * 100, 100))  # Convert to 0-100 scale
        
        self.df['commercial_potential'] = scores
        return scores
    
    def calculate_musical_quality(self):
        """Calculate musical quality and production value"""
        print("🎵 Calculating musical quality scores...")
        
        scores = []
        
        for _, track in self.df.iterrows():
            score = 0
            
            # Production quality (50% of musical quality)
            # Based on loudness consistency and overall mix
            loudness = track.get('loudness', -60)
            if pd.notna(loudness):
                # Optimal loudness range for modern music: -14 to -6 dB
                if -14 <= loudness <= -6:
                    production_score = 1.0
                elif -20 <= loudness <= -14 or -6 <= loudness <= 0:
                    production_score = 0.7
                else:
                    production_score = 0.4
            else:
                production_score = 0.5
            
            score += production_score * 0.5
            
            # Musical complexity (25% of musical quality)
            # Balance between simplicity and sophistication
            complexity_features = ['acousticness', 'instrumentalness', 'speechiness']
            complexity_values = []
            
            for feature in complexity_features:
                value = track.get(feature)
                if pd.notna(value):
                    complexity_values.append(value)
            
            if complexity_values:
                # Optimal complexity is moderate (not too simple, but not too complex)
                avg_complexity = np.mean(complexity_values)
                if 0.2 <= avg_complexity <= 0.6:
                    complexity_score = 1.0
                else:
                    complexity_score = 1 - abs(avg_complexity - 0.4) * 2
                complexity_score = max(0, complexity_score)
            else:
                complexity_score = 0.5
            
            score += complexity_score * 0.25
            
            # Data quality (25% of musical quality)
            quality = track.get('data_quality_score', 0)
            if pd.notna(quality):
                quality_score = quality / 100
            else:
                quality_score = 0.5
            
            score += quality_score * 0.25
            
            scores.append(min(score * 100, 100))
        
        self.df['musical_quality'] = scores
        return scores
    
    def calculate_market_fit(self):
        """Calculate how well tracks fit current market trends"""
        print("Calculating market fit scores...")
        
        scores = []
        
        for _, track in self.df.iterrows():
            genre = track.get('genre', 'unknown').lower()
            
            # Base score
            base_score = 0.5
            
            # Genre trend multiplier
            trend_multiplier = 1.0
            if genre in self.market_trends:
                trend_data = self.market_trends[genre]
                trend_multiplier = trend_data['weight']
                
                # Calculate feature alignment with trend
                feature_alignment = 0
                feature_count = 0
                
                for feature, target_value in trend_data['features'].items():
                    track_value = track.get(feature)
                    if pd.notna(track_value):
                        if feature == 'loudness':
                            # Special handling for loudness (dB scale)
                            alignment = 1 - abs(track_value - target_value) / 20
                        else:
                            # Standard 0-1 features
                            alignment = 1 - abs(track_value - target_value)
                        
                        feature_alignment += max(0, alignment)
                        feature_count += 1
                
                if feature_count > 0:
                    base_score = feature_alignment / feature_count
            
            # Demographic appeal bonus
            if genre in ['pop', 'hip hop', 'electronic']:
                demographic_bonus = 0.1  # Genres popular with younger demographics
            else:
                demographic_bonus = 0
            
            final_score = (base_score * trend_multiplier + demographic_bonus) * 100
            scores.append(min(final_score, 100))
        
        self.df['market_fit'] = scores
        return scores
    
    def calculate_uniqueness(self):
        """Calculate uniqueness/standout potential"""
        print("✨ Calculating uniqueness scores...")
        
        # Calculate feature distances from median values
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'speechiness']
        
        available_features = [f for f in audio_features if f in self.df.columns]
        
        if not available_features:
            self.df['uniqueness'] = 50  # Default score
            return [50] * len(self.df)
        
        # Calculate median values for each feature
        medians = {}
        for feature in available_features:
            medians[feature] = self.df[feature].median()
        
        scores = []
        
        for _, track in self.df.iterrows():
            uniqueness_score = 0
            feature_count = 0
            
            for feature in available_features:
                track_value = track.get(feature)
                if pd.notna(track_value):
                    # Distance from median (0 = very common, 1 = very unique)
                    distance = abs(track_value - medians[feature])
                    
                    # Convert distance to uniqueness score
                    # Sweet spot: moderately unique (not too mainstream, but not too weird)
                    if 0.1 <= distance <= 0.3:
                        feature_uniqueness = 1.0  # Optimal uniqueness
                    elif 0.05 <= distance <= 0.1 or 0.3 <= distance <= 0.4:
                        feature_uniqueness = 0.8  # Good uniqueness
                    elif distance <= 0.05:
                        feature_uniqueness = 0.3  # Too mainstream
                    else:
                        feature_uniqueness = 0.5  # Too unique/experimental
                    
                    uniqueness_score += feature_uniqueness
                    feature_count += 1
            
            if feature_count > 0:
                final_uniqueness = (uniqueness_score / feature_count) * 100
            else:
                final_uniqueness = 50
            
            scores.append(final_uniqueness)
        
        self.df['uniqueness'] = scores
        return scores
    
    def calculate_scalability(self):
        """Calculate scalability potential"""
        print("Calculating scalability scores...")
        
        scores = []
        
        for _, track in self.df.iterrows():
            score = 0
            
            # Multi-source presence (50% of scalability)
            source_count = track.get('source_count', 1)
            source_score = min(source_count / 5, 1.0)  # Max score at 5+ sources
            score += source_score * 0.5
            
            # Multi-platform presence (40% of scalability)
            platform_count = sum([
                track.get('spotify', False),
                track.get('tiktok', False),
                track.get('youtube', False)
            ])
            platform_score = platform_count / 3
            score += platform_score * 0.4
            
            # Data completeness (10% of scalability)
            completeness = track.get('data_quality_score', 0) / 100
            score += completeness * 0.1
            
            scores.append(score * 100)
        
        self.df['scalability'] = scores
        return scores
    
    def calculate_overall_ar_score(self):
        """Calculate overall A&R potential score"""
        print("🎯 Calculating overall A&R scores...")
        
        # Ensure all component scores exist
        required_scores = ['commercial_potential', 'musical_quality', 'market_fit', 'uniqueness', 'scalability']
        
        for score_type in required_scores:
            if score_type not in self.df.columns:
                print(f"⚠️ Missing {score_type} scores, calculating...")
                getattr(self, f'calculate_{score_type}')()
        
        # Calculate weighted overall score
        overall_scores = []
        
        for _, track in self.df.iterrows():
            weighted_score = 0
            
            for score_type, weight in self.ar_criteria.items():
                component_score = track.get(score_type, 0)
                weighted_score += component_score * weight
            
            overall_scores.append(round(weighted_score, 2))
        
        self.df['ar_score'] = overall_scores
        return overall_scores
    
    def discover_promising_artists(self, min_tracks=2, top_n=20):
        """Discover promising artists based on A&R scores"""
        print(f"🔍 Discovering promising artists (min {min_tracks} tracks, top {top_n})...")
        
        if 'ar_score' not in self.df.columns:
            self.calculate_overall_ar_score()
        
        # Group by artist and calculate metrics
        artist_analysis = []
        
        for artist in self.df['artist_name'].unique():
            artist_tracks = self.df[self.df['artist_name'] == artist]
            
            if len(artist_tracks) < min_tracks:
                continue
            
            # Calculate artist-level metrics
            artist_metrics = {
                'artist_name': artist,
                'track_count': len(artist_tracks),
                'avg_ar_score': artist_tracks['ar_score'].mean(),
                'max_ar_score': artist_tracks['ar_score'].max(),
                'consistency': 100 - artist_tracks['ar_score'].std(),  # Lower std = more consistent
                'avg_popularity': artist_tracks['normalized_popularity'].mean(),
                'platform_diversity': artist_tracks[['spotify', 'tiktok', 'youtube']].sum().sum(),
                'genre_focus': artist_tracks['genre'].mode().iloc[0] if len(artist_tracks['genre'].mode()) > 0 else 'unknown',
                'total_sources': artist_tracks['source_count'].sum(),
                'best_track': artist_tracks.loc[artist_tracks['ar_score'].idxmax(), 'track_name']
            }
            
            # Calculate potential categories
            if artist_metrics['avg_ar_score'] >= 75:
                if artist_metrics['avg_popularity'] >= 70:
                    category = 'Established Hit-Maker'
                else:
                    category = 'High Potential Emerging'
            elif artist_metrics['avg_ar_score'] >= 60:
                if artist_metrics['consistency'] >= 80:
                    category = 'Consistent Performer'
                else:
                    category = 'Variable Quality'
            elif artist_metrics['max_ar_score'] >= 80:
                category = 'One Hit Wonder Potential'
            else:
                category = 'Development Needed'
            
            artist_metrics['category'] = category
            artist_analysis.append(artist_metrics)
        
        # Sort by A&R potential
        artist_analysis.sort(key=lambda x: x['avg_ar_score'], reverse=True)
        
        return artist_analysis[:top_n]
    
    def analyze_market_opportunities(self):
        """Analyze market opportunities by genre and demographics"""
        print("Analyzing market opportunities...")
        
        if 'ar_score' not in self.df.columns:
            self.calculate_overall_ar_score()
        
        opportunities = {}
        
        # Genre analysis
        genre_analysis = {}
        for genre in self.df['genre'].unique():
            if genre == 'unknown':
                continue
            
            genre_tracks = self.df[self.df['genre'] == genre]
            if len(genre_tracks) < 10:
                continue
            
            genre_metrics = {
                'track_count': len(genre_tracks),
                'avg_ar_score': genre_tracks['ar_score'].mean(),
                'avg_popularity': genre_tracks['normalized_popularity'].mean(),
                'market_saturation': len(genre_tracks) / len(self.df) * 100,
                'quality_distribution': {
                    'high_quality': (genre_tracks['ar_score'] >= 70).sum(),
                    'medium_quality': ((genre_tracks['ar_score'] >= 50) & (genre_tracks['ar_score'] < 70)).sum(),
                    'low_quality': (genre_tracks['ar_score'] < 50).sum()
                },
                'trend_alignment': self.market_trends.get(genre.lower(), {}).get('weight', 1.0)
            }
            
            # Calculate opportunity score
            opportunity_score = (
                (100 - genre_metrics['market_saturation']) * 0.4 +  # Less saturated = more opportunity
                genre_metrics['avg_ar_score'] * 0.3 +               # Higher quality = more opportunity
                (genre_metrics['trend_alignment'] - 0.5) * 100 * 0.3  # Trending genres = more opportunity
            )
            
            genre_metrics['opportunity_score'] = max(0, opportunity_score)
            genre_analysis[genre] = genre_metrics
        
        opportunities['genre_analysis'] = dict(sorted(genre_analysis.items(), 
                                                     key=lambda x: x[1]['opportunity_score'], 
                                                     reverse=True))
        
        return opportunities
    
    def generate_ar_report(self, output_file='ar_simulation_report.json'):
        """Generate comprehensive A&R report"""
        print("Generating comprehensive A&R report...")
        
        # Calculate all scores
        self.calculate_commercial_potential()
        self.calculate_musical_quality()
        self.calculate_market_fit()
        self.calculate_uniqueness()
        self.calculate_scalability()
        self.calculate_overall_ar_score()
        
        # Discover promising artists
        promising_artists = self.discover_promising_artists()
        
        # Analyze market opportunities
        market_opportunities = self.analyze_market_opportunities()
        
        # Get top tracks
        top_tracks = self.df.nlargest(20, 'ar_score')[
            ['track_name', 'artist_name', 'ar_score', 'commercial_potential', 
             'musical_quality', 'market_fit', 'uniqueness', 'scalability', 
             'genre', 'normalized_popularity']
        ].to_dict('records')
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset_size': len(self.df),
                'unique_artists': self.df['artist_name'].nunique(),
                'analysis_criteria': self.ar_criteria
            },
            'executive_summary': {
                'avg_ar_score': round(self.df['ar_score'].mean(), 2),
                'high_potential_tracks': (self.df['ar_score'] >= 70).sum(),
                'promising_artists_count': len([a for a in promising_artists if a['avg_ar_score'] >= 60]),
                'top_genres': list(market_opportunities['genre_analysis'].keys())[:5]
            },
            'promising_artists': promising_artists,
            'top_tracks': top_tracks,
            'market_opportunities': market_opportunities,
            'scoring_methodology': {
                'commercial_potential': 'Popularity + Platform Presence + Audio Appeal',
                'musical_quality': 'Production Quality + Complexity + Data Quality',
                'market_fit': 'Genre Trends + Feature Alignment + Demographics',
                'uniqueness': 'Feature Deviation + Standout Potential',
                'scalability': 'Multi-source + Multi-platform + Completeness'
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"A&R report saved to: {output_file}")
        return report
    
    def run_ar_simulation(self):
        """Run A&R simulation for testing"""
        if not self.load_dataset():
            return None
        
        print("\nA&R SIMULATION STARTING")
        print("=" * 50)
        print("Simulating record label talent discovery process...")
        print()
        
        # Generate comprehensive report
        report = self.generate_ar_report()
        
        # Display key findings
        print("A&R SIMULATION COMPLETE!")
        print("=" * 50)
        
        exec_summary = report['executive_summary']
        print(f"Dataset Analysis:")
        print(f"   • Total tracks analyzed: {report['report_metadata']['dataset_size']:,}")
        print(f"   • Unique artists: {report['report_metadata']['unique_artists']:,}")
        print(f"   • Average A&R score: {exec_summary['avg_ar_score']}/100")
        print(f"   • High potential tracks: {exec_summary['high_potential_tracks']:,}")
        print(f"   • Promising artists found: {exec_summary['promising_artists_count']}")
        
        print(f"\nTOP 5 PROMISING ARTISTS:")
        for i, artist in enumerate(report['promising_artists'][:5], 1):
            print(f"   {i}. {artist['artist_name']}")
            print(f"      A&R Score: {artist['avg_ar_score']:.1f}/100 | Category: {artist['category']}")
            print(f"      Best Track: '{artist['best_track']}' | Genre: {artist['genre_focus']}")
        
        print(f"\nTOP 5 MARKET OPPORTUNITIES:")
        opportunities = report['market_opportunities']['genre_analysis']
        for i, (genre, data) in enumerate(list(opportunities.items())[:5], 1):
            print(f"   {i}. {genre.title()}")
            print(f"      Opportunity Score: {data['opportunity_score']:.1f}/100")
            print(f"      Avg Quality: {data['avg_ar_score']:.1f} | Saturation: {data['market_saturation']:.1f}%")
        
        print(f"\nTOP 3 TRACKS TO SIGN:")
        for i, track in enumerate(report['top_tracks'][:3], 1):
            print(f"   {i}. '{track['track_name']}' by {track['artist_name']}")
            print(f"      A&R Score: {track['ar_score']:.1f}/100 | Genre: {track['genre']}")
            print(f"      Commercial: {track['commercial_potential']:.1f} | Quality: {track['musical_quality']:.1f}")
        
        print(f"\nFull report saved to: ar_simulation_report.json")
        
        return report

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='A&R Simulation System')
    parser.add_argument('--dataset', default='integration_cache/master_music_dataset_deduplicated.csv',
                       help='Path to master dataset')
    parser.add_argument('--output', default='ar_simulation_report.json',
                       help='Output report file')
    parser.add_argument('--min-tracks', type=int, default=2,
                       help='Minimum tracks per artist for analysis')
    parser.add_argument('--top-artists', type=int, default=20,
                       help='Number of top artists to identify')
    
    args = parser.parse_args()
    
    # Run A&R simulation
    ar_system = ARSimulationSystem(dataset_path=args.dataset)
    report = ar_system.run_ar_simulation()
    
    if report:
        print(f"\nA&R Simulation completed successfully!")
        print(f"Use this data to make informed talent acquisition decisions")
        print(f"Focus on artists in 'High Potential Emerging' category")