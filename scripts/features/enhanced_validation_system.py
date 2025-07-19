import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedValidationSystem:
    """
    Comprehensive validation system for the master music dataset
    with detailed quality metrics, cross-dataset consistency checks,
    and sophisticated demographic validation
    """
    
    def __init__(self, dataset_path='integration_cache/master_music_dataset_deduplicated.csv',
                 output_dir='validation_reports'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.df = None
        
        # Audio features for validation
        self.audio_features = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness'
        ]
        
        # Expected ranges for audio features (based on Spotify documentation)
        self.feature_ranges = {
            'danceability': (0.0, 1.0),
            'energy': (0.0, 1.0),
            'valence': (0.0, 1.0),
            'acousticness': (0.0, 1.0),
            'instrumentalness': (0.0, 1.0),
            'liveness': (0.0, 1.0),
            'speechiness': (0.0, 1.0),
            'tempo': (50.0, 250.0),  # BPM
            'loudness': (-60.0, 0.0)  # dB
        }
        
        # Demographics categories
        self.demographics = ['Gen Z (16-24)', 'Millennials (25-40)', 'Gen X (41-56)', 'Boomers (57+)']
        
        # Initialize validation results
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_overview': {},
            'quality_metrics': {},
            'audio_feature_consistency': {},
            'demographic_validation': {},
            'cross_dataset_analysis': {},
            'recommendations': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_dataset(self):
        """Load the master dataset"""
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset not found: {self.dataset_path}")
            return False
        
        logger.info(f"Loading dataset: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        logger.info(f"Loaded {len(self.df):,} tracks")
        return True
    
    def generate_dataset_overview(self):
        """Generate comprehensive dataset overview"""
        logger.info("Generating dataset overview...")
        
        overview = {
            'total_tracks': int(len(self.df)),
            'unique_artists': int(self.df['artist_name'].nunique()),
            'unique_tracks': int(self.df['track_name'].nunique()),
            'date_range': {
                'analysis_date': datetime.now().isoformat(),
                'dataset_created': os.path.getctime(self.dataset_path)
            },
            'source_distribution': {},
            'platform_coverage': {},
            'quality_score_distribution': {},
            'completeness_metrics': {}
        }
        
        # Source count distribution
        if 'source_count' in self.df.columns:
            source_dist = self.df['source_count'].value_counts().sort_index()
            overview['source_distribution'] = {
                'single_source': int(source_dist.get(1, 0)),
                'multi_source': int(source_dist[source_dist.index > 1].sum()),
                'max_sources': int(self.df['source_count'].max()),
                'avg_sources': float(self.df['source_count'].mean()),
                'distribution': source_dist.to_dict()
            }
        
        # Platform coverage
        platforms = ['spotify', 'tiktok', 'youtube']
        for platform in platforms:
            if platform in self.df.columns:
                count = int(self.df[platform].sum())
                percentage = (count / len(self.df)) * 100
                overview['platform_coverage'][platform] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }
        
        # Multi-platform tracks
        if all(p in self.df.columns for p in platforms):
            multi_platform = self.df[platforms].sum(axis=1) > 1
            overview['platform_coverage']['multi_platform'] = {
                'count': int(multi_platform.sum()),
                'percentage': round((multi_platform.sum() / len(self.df)) * 100, 2)
            }
        
        # Quality score distribution
        if 'data_quality_score' in self.df.columns:
            quality_scores = self.df['data_quality_score'].dropna()
            overview['quality_score_distribution'] = {
                'mean': float(quality_scores.mean()),
                'median': float(quality_scores.median()),
                'std': float(quality_scores.std()),
                'high_quality_tracks': int((quality_scores >= 80).sum()),
                'medium_quality_tracks': int(((quality_scores >= 60) & (quality_scores < 80)).sum()),
                'low_quality_tracks': int((quality_scores < 60).sum())
            }
        
        # Data completeness
        for col in ['track_name', 'artist_name', 'genre', 'normalized_popularity'] + self.audio_features:
            if col in self.df.columns:
                total_values = len(self.df)
                non_null_values = self.df[col].notna().sum()
                completeness = (non_null_values / total_values) * 100
                overview['completeness_metrics'][col] = {
                    'completeness_percentage': round(completeness, 2),
                    'missing_count': int(total_values - non_null_values)
                }
        
        self.validation_results['dataset_overview'] = overview
        return overview
    
    def validate_audio_feature_consistency(self):
        """Comprehensive audio feature validation"""
        logger.info("Validating audio feature consistency...")
        
        consistency_results = {
            'feature_statistics': {},
            'range_violations': {},
            'outlier_analysis': {},
            'correlation_analysis': {},
            'cross_source_consistency': {},
            'quality_flags': []
        }
        
        for feature in self.audio_features:
            if feature not in self.df.columns:
                continue
            
            values = self.df[feature].dropna()
            if len(values) == 0:
                continue
            
            # Basic statistics
            feature_stats = {
                'count': int(len(values)),
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values))
            }
            consistency_results['feature_statistics'][feature] = feature_stats
            
            # Range validation
            expected_min, expected_max = self.feature_ranges.get(feature, (None, None))
            if expected_min is not None and expected_max is not None:
                range_violations = {
                    'below_min': int((values < expected_min).sum()),
                    'above_max': int((values > expected_max).sum()),
                    'within_range_percentage': round(((values >= expected_min) & (values <= expected_max)).mean() * 100, 2)
                }
                consistency_results['range_violations'][feature] = range_violations
                
                if range_violations['within_range_percentage'] < 95:
                    consistency_results['quality_flags'].append({
                        'type': 'range_violation',
                        'feature': feature,
                        'severity': 'high' if range_violations['within_range_percentage'] < 90 else 'medium',
                        'description': f"{feature} has {100 - range_violations['within_range_percentage']:.1f}% values outside expected range"
                    })
            
            # Outlier detection (using IQR method)
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            outlier_analysis = {
                'outlier_count': int(len(outliers)),
                'outlier_percentage': round((len(outliers) / len(values)) * 100, 2),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            consistency_results['outlier_analysis'][feature] = outlier_analysis
            
            if outlier_analysis['outlier_percentage'] > 5:
                consistency_results['quality_flags'].append({
                    'type': 'outliers',
                    'feature': feature,
                    'severity': 'medium',
                    'description': f"{feature} has {outlier_analysis['outlier_percentage']:.1f}% outliers"
                })
        
        # Feature correlation analysis
        available_features = [f for f in self.audio_features if f in self.df.columns]
        if len(available_features) >= 2:
            correlation_matrix = self.df[available_features].corr()
            
            # Find highly correlated feature pairs
            high_correlations = []
            for i in range(len(available_features)):
                for j in range(i+1, len(available_features)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_correlations.append({
                            'feature1': available_features[i],
                            'feature2': available_features[j],
                            'correlation': round(float(corr_value), 3)
                        })
            
            consistency_results['correlation_analysis'] = {
                'high_correlations': high_correlations,
                'correlation_matrix': correlation_matrix.round(3).to_dict()
            }
        
        # Cross-source consistency (for tracks with multiple sources)
        if 'source_count' in self.df.columns:
            multi_source_tracks = self.df[self.df['source_count'] > 1]
            if len(multi_source_tracks) > 0:
                cross_source_consistency = {}
                
                for feature in available_features:
                    if feature in multi_source_tracks.columns:
                        # For multi-source tracks, check if audio features are consistent
                        # (This is a proxy check since we don't have original source values)
                        feature_variance = multi_source_tracks[feature].var()
                        single_source_variance = self.df[self.df['source_count'] == 1][feature].var()
                        
                        consistency_ratio = feature_variance / single_source_variance if single_source_variance > 0 else 1
                        
                        cross_source_consistency[feature] = {
                            'multi_source_variance': float(feature_variance),
                            'single_source_variance': float(single_source_variance),
                            'consistency_ratio': round(float(consistency_ratio), 3)
                        }
                
                consistency_results['cross_source_consistency'] = cross_source_consistency
        
        self.validation_results['audio_feature_consistency'] = consistency_results
        return consistency_results
    
    def validate_demographic_mapping(self):
        """Sophisticated demographic validation"""
        logger.info("Validating demographic mapping...")
        
        demographic_validation = {
            'genre_demographic_analysis': {},
            'audio_feature_demographic_consistency': {},
            'demographic_distribution': {},
            'validation_scores': {},
            'recommendations': []
        }
        
        # Define demographic audio preferences (research-based)
        demographic_profiles = {
            'Gen Z (16-24)': {
                'danceability': {'min': 0.6, 'max': 1.0, 'ideal': 0.8},
                'energy': {'min': 0.5, 'max': 1.0, 'ideal': 0.75},
                'valence': {'min': 0.4, 'max': 1.0, 'ideal': 0.7},
                'speechiness': {'min': 0.1, 'max': 0.5, 'ideal': 0.3},
                'acousticness': {'min': 0.0, 'max': 0.4, 'ideal': 0.2},
                'tempo': {'min': 100, 'max': 180, 'ideal': 128}
            },
            'Millennials (25-40)': {
                'danceability': {'min': 0.4, 'max': 0.8, 'ideal': 0.65},
                'energy': {'min': 0.3, 'max': 0.8, 'ideal': 0.6},
                'valence': {'min': 0.3, 'max': 0.8, 'ideal': 0.6},
                'speechiness': {'min': 0.05, 'max': 0.3, 'ideal': 0.15},
                'acousticness': {'min': 0.1, 'max': 0.7, 'ideal': 0.4},
                'tempo': {'min': 80, 'max': 140, 'ideal': 110}
            },
            'Gen X (41-56)': {
                'danceability': {'min': 0.2, 'max': 0.7, 'ideal': 0.5},
                'energy': {'min': 0.2, 'max': 0.8, 'ideal': 0.55},
                'valence': {'min': 0.2, 'max': 0.7, 'ideal': 0.5},
                'speechiness': {'min': 0.03, 'max': 0.2, 'ideal': 0.1},
                'acousticness': {'min': 0.2, 'max': 0.9, 'ideal': 0.6},
                'tempo': {'min': 70, 'max': 130, 'ideal': 100}
            },
            'Boomers (57+)': {
                'danceability': {'min': 0.1, 'max': 0.6, 'ideal': 0.4},
                'energy': {'min': 0.1, 'max': 0.6, 'ideal': 0.4},
                'valence': {'min': 0.2, 'max': 0.7, 'ideal': 0.45},
                'speechiness': {'min': 0.02, 'max': 0.15, 'ideal': 0.05},
                'acousticness': {'min': 0.3, 'max': 1.0, 'ideal': 0.7},
                'tempo': {'min': 60, 'max': 120, 'ideal': 90}
            }
        }
        
        # Analyze genre-demographic alignment
        if 'genre' in self.df.columns:
            genre_demographic_analysis = {}
            
            for genre in self.df['genre'].unique():
                if genre == 'unknown' or pd.isna(genre):
                    continue
                
                genre_tracks = self.df[self.df['genre'] == genre]
                if len(genre_tracks) < 10:  # Skip genres with too few tracks
                    continue
                
                genre_profile = {}
                for feature in ['danceability', 'energy', 'valence', 'speechiness', 'acousticness', 'tempo']:
                    if feature in genre_tracks.columns:
                        values = genre_tracks[feature].dropna()
                        if len(values) > 0:
                            genre_profile[feature] = {
                                'mean': float(values.mean()),
                                'std': float(values.std()),
                                'median': float(values.median())
                            }
                
                # Calculate demographic fit scores
                demographic_scores = {}
                for demo, demo_profile in demographic_profiles.items():
                    score = 0
                    feature_count = 0
                    
                    for feature, genre_stats in genre_profile.items():
                        if feature in demo_profile:
                            genre_mean = genre_stats['mean']
                            demo_ideal = demo_profile[feature]['ideal']
                            demo_min = demo_profile[feature]['min']
                            demo_max = demo_profile[feature]['max']
                            
                            # Calculate fit score (0-1)
                            if demo_min <= genre_mean <= demo_max:
                                # Within range, calculate how close to ideal
                                distance_from_ideal = abs(genre_mean - demo_ideal)
                                max_distance = max(abs(demo_max - demo_ideal), abs(demo_min - demo_ideal))
                                fit_score = 1 - (distance_from_ideal / max_distance) if max_distance > 0 else 1
                            else:
                                # Outside range
                                if genre_mean < demo_min:
                                    fit_score = max(0, 1 - (demo_min - genre_mean) / demo_min)
                                else:
                                    fit_score = max(0, 1 - (genre_mean - demo_max) / demo_max)
                            
                            score += fit_score
                            feature_count += 1
                    
                    if feature_count > 0:
                        demographic_scores[demo] = round((score / feature_count) * 100, 2)
                
                genre_demographic_analysis[genre] = {
                    'track_count': len(genre_tracks),
                    'audio_profile': genre_profile,
                    'demographic_fit_scores': demographic_scores,
                    'best_demographic': max(demographic_scores.items(), key=lambda x: x[1]) if demographic_scores else None
                }
            
            demographic_validation['genre_demographic_analysis'] = genre_demographic_analysis
        
        # Overall demographic distribution validation
        demographic_distribution = {}
        
        for demo, demo_profile in demographic_profiles.items():
            matching_tracks = 0
            total_tracks = 0
            
            for _, track in self.df.iterrows():
                is_match = True
                feature_count = 0
                
                for feature, constraints in demo_profile.items():
                    if feature in track and pd.notna(track[feature]):
                        feature_value = track[feature]
                        if not (constraints['min'] <= feature_value <= constraints['max']):
                            is_match = False
                            break
                        feature_count += 1
                
                if feature_count >= 3:  # Require at least 3 features to match
                    total_tracks += 1
                    if is_match:
                        matching_tracks += 1
            
            if total_tracks > 0:
                demographic_distribution[demo] = {
                    'matching_tracks': matching_tracks,
                    'total_evaluated_tracks': total_tracks,
                    'percentage': round((matching_tracks / total_tracks) * 100, 2)
                }
        
        demographic_validation['demographic_distribution'] = demographic_distribution
        
        # Generate validation scores and recommendations
        validation_scores = {}
        
        # Genre coverage score
        if 'genre_demographic_analysis' in demographic_validation:
            genres_analyzed = len(demographic_validation['genre_demographic_analysis'])
            total_genres = self.df['genre'].nunique()
            genre_coverage = (genres_analyzed / total_genres) * 100 if total_genres > 0 else 0
            validation_scores['genre_coverage'] = round(genre_coverage, 2)
        
        # Demographic balance score
        if demographic_distribution:
            demo_percentages = [d['percentage'] for d in demographic_distribution.values()]
            balance_score = 100 - (np.std(demo_percentages) * 2)  # Lower std = better balance
            validation_scores['demographic_balance'] = round(max(0, balance_score), 2)
        
        demographic_validation['validation_scores'] = validation_scores
        
        # Generate recommendations
        recommendations = []
        
        if validation_scores.get('genre_coverage', 0) < 70:
            recommendations.append({
                'type': 'genre_coverage',
                'priority': 'medium',
                'description': 'Low genre coverage in demographic analysis. Consider adding more genre diversity or improving genre classification.'
            })
        
        if validation_scores.get('demographic_balance', 0) < 60:
            recommendations.append({
                'type': 'demographic_balance',
                'priority': 'medium', 
                'description': 'Demographic distribution is unbalanced. Dataset may be skewed toward certain age groups.'
            })
        
        demographic_validation['recommendations'] = recommendations
        
        self.validation_results['demographic_validation'] = demographic_validation
        return demographic_validation
    
    def generate_quality_metrics(self):
        """Generate detailed quality metrics"""
        logger.info("Generating quality metrics...")
        
        quality_metrics = {
            'overall_quality_score': 0,
            'component_scores': {},
            'quality_categories': {},
            'data_freshness': {},
            'reliability_metrics': {}
        }
        
        # Component scores
        component_scores = {}
        
        # Data Completeness Score (0-100)
        completeness_scores = []
        critical_fields = ['track_name', 'artist_name', 'normalized_popularity'] + self.audio_features[:5]
        
        for field in critical_fields:
            if field in self.df.columns:
                completeness = (self.df[field].notna().sum() / len(self.df)) * 100
                completeness_scores.append(completeness)
        
        component_scores['data_completeness'] = round(np.mean(completeness_scores), 2) if completeness_scores else 0
        
        # Data Consistency Score (0-100)
        consistency_score = 100
        
        # Check for range violations
        for feature in self.audio_features:
            if feature in self.df.columns and feature in self.feature_ranges:
                values = self.df[feature].dropna()
                expected_min, expected_max = self.feature_ranges[feature]
                violations = ((values < expected_min) | (values > expected_max)).sum()
                violation_rate = violations / len(values) if len(values) > 0 else 0
                consistency_score -= violation_rate * 10  # Deduct points for violations
        
        component_scores['data_consistency'] = round(max(0, consistency_score), 2)
        
        # Source Diversity Score (0-100)
        if 'source_count' in self.df.columns:
            multi_source_pct = (self.df['source_count'] > 1).mean() * 100
            avg_sources = self.df['source_count'].mean()
            diversity_score = min(100, multi_source_pct + (avg_sources * 10))
        else:
            diversity_score = 0
        
        component_scores['source_diversity'] = round(diversity_score, 2)
        
        # Platform Coverage Score (0-100)
        platforms = ['spotify', 'tiktok', 'youtube']
        platform_coverage = []
        
        for platform in platforms:
            if platform in self.df.columns:
                coverage = (self.df[platform].sum() / len(self.df)) * 100
                platform_coverage.append(coverage)
        
        coverage_score = np.mean(platform_coverage) if platform_coverage else 0
        component_scores['platform_coverage'] = round(coverage_score, 2)
        
        # Feature Quality Score (0-100)
        feature_quality_scores = []
        
        for feature in self.audio_features:
            if feature in self.df.columns:
                values = self.df[feature].dropna()
                if len(values) > 0:
                    # Check for reasonable distribution
                    cv = values.std() / values.mean() if values.mean() != 0 else 0
                    outlier_pct = self._calculate_outlier_percentage(values)
                    
                    # Score based on coefficient of variation and outliers
                    feature_score = 100 - (cv * 50) - (outlier_pct * 2)
                    feature_quality_scores.append(max(0, feature_score))
        
        component_scores['feature_quality'] = round(np.mean(feature_quality_scores), 2) if feature_quality_scores else 0
        
        quality_metrics['component_scores'] = component_scores
        
        # Overall quality score (weighted average)
        weights = {
            'data_completeness': 0.3,
            'data_consistency': 0.25,
            'source_diversity': 0.2,
            'platform_coverage': 0.15,
            'feature_quality': 0.1
        }
        
        overall_score = sum(component_scores[component] * weight 
                          for component, weight in weights.items() 
                          if component in component_scores)
        
        quality_metrics['overall_quality_score'] = round(overall_score, 2)
        
        # Quality categories
        if overall_score >= 90:
            quality_category = 'Excellent'
        elif overall_score >= 80:
            quality_category = 'Good'
        elif overall_score >= 70:
            quality_category = 'Fair'
        elif overall_score >= 60:
            quality_category = 'Poor'
        else:
            quality_category = 'Critical'
        
        quality_metrics['quality_categories'] = {
            'overall_category': quality_category,
            'excellent_tracks': int((self.df.get('data_quality_score', pd.Series([0])) >= 90).sum()),
            'good_tracks': int(((self.df.get('data_quality_score', pd.Series([0])) >= 80) & 
                               (self.df.get('data_quality_score', pd.Series([0])) < 90)).sum()),
            'fair_tracks': int(((self.df.get('data_quality_score', pd.Series([0])) >= 70) & 
                               (self.df.get('data_quality_score', pd.Series([0])) < 80)).sum()),
            'poor_tracks': int((self.df.get('data_quality_score', pd.Series([0])) < 70).sum())
        }
        
        # Data freshness (based on file modification time)
        file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.dataset_path))).days
        freshness_score = max(0, 100 - (file_age_days * 2))  # Deduct 2 points per day
        
        quality_metrics['data_freshness'] = {
            'file_age_days': file_age_days,
            'freshness_score': round(freshness_score, 2)
        }
        
        # Reliability metrics
        reliability_metrics = {
            'duplicate_rate': 0,  # Should be 0 after deduplication
            'missing_data_rate': round((self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100, 2),
            'outlier_rate': round(self._calculate_overall_outlier_rate(), 2),
            'consistency_violations': self._count_consistency_violations()
        }
        
        quality_metrics['reliability_metrics'] = reliability_metrics
        
        self.validation_results['quality_metrics'] = quality_metrics
        return quality_metrics
    
    def _calculate_outlier_percentage(self, values):
        """Calculate percentage of outliers using IQR method"""
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        return (len(outliers) / len(values)) * 100 if len(values) > 0 else 0
    
    def _calculate_overall_outlier_rate(self):
        """Calculate overall outlier rate across all audio features"""
        outlier_rates = []
        
        for feature in self.audio_features:
            if feature in self.df.columns:
                values = self.df[feature].dropna()
                if len(values) > 0:
                    outlier_rate = self._calculate_outlier_percentage(values)
                    outlier_rates.append(outlier_rate)
        
        return np.mean(outlier_rates) if outlier_rates else 0
    
    def _count_consistency_violations(self):
        """Count various consistency violations"""
        violations = 0
        
        # Range violations
        for feature in self.audio_features:
            if feature in self.df.columns and feature in self.feature_ranges:
                values = self.df[feature].dropna()
                expected_min, expected_max = self.feature_ranges[feature]
                violations += ((values < expected_min) | (values > expected_max)).sum()
        
        return int(violations)
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on validation results"""
        logger.info("Generating recommendations...")
        
        recommendations = []
        
        # Quality-based recommendations
        quality_score = self.validation_results['quality_metrics']['overall_quality_score']
        
        if quality_score < 80:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'title': 'Improve Overall Data Quality',
                'description': f'Overall quality score is {quality_score:.1f}%. Focus on data completeness and consistency.',
                'actions': [
                    'Review data collection processes',
                    'Implement additional data validation rules',
                    'Consider additional data sources'
                ]
            })
        
        # Component-specific recommendations
        component_scores = self.validation_results['quality_metrics']['component_scores']
        
        for component, score in component_scores.items():
            if score < 70:
                if component == 'data_completeness':
                    recommendations.append({
                        'category': 'Data Completeness',
                        'priority': 'High',
                        'title': 'Address Missing Data',
                        'description': f'Data completeness score is {score:.1f}%.',
                        'actions': [
                            'Identify fields with high missing data rates',
                            'Implement better imputation strategies',
                            'Seek additional data sources for missing fields'
                        ]
                    })
                elif component == 'source_diversity':
                    recommendations.append({
                        'category': 'Data Integration',
                        'priority': 'Medium',
                        'title': 'Improve Source Diversity',
                        'description': f'Source diversity score is {score:.1f}%. Many tracks come from single sources.',
                        'actions': [
                            'Improve fuzzy matching algorithms',
                            'Add more diverse data sources',
                            'Review deduplication thresholds'
                        ]
                    })
                elif component == 'platform_coverage':
                    recommendations.append({
                        'category': 'Platform Coverage',
                        'priority': 'Medium',
                        'title': 'Expand Platform Coverage',
                        'description': f'Platform coverage score is {score:.1f}%.',
                        'actions': [
                            'Add data from underrepresented platforms',
                            'Improve platform identification logic',
                            'Consider emerging music platforms'
                        ]
                    })
        
        # Audio feature consistency recommendations
        audio_consistency = self.validation_results.get('audio_feature_consistency', {})
        quality_flags = audio_consistency.get('quality_flags', [])
        
        for flag in quality_flags:
            if flag['severity'] == 'high':
                recommendations.append({
                    'category': 'Audio Features',
                    'priority': 'High',
                    'title': f'Fix {flag["feature"]} Issues',
                    'description': flag['description'],
                    'actions': [
                        f'Review {flag["feature"]} extraction process',
                        'Check for data source inconsistencies',
                        'Implement feature-specific validation rules'
                    ]
                })
        
        # Demographic validation recommendations
        demo_validation = self.validation_results.get('demographic_validation', {})
        demo_recommendations = demo_validation.get('recommendations', [])
        
        for demo_rec in demo_recommendations:
            recommendations.append({
                'category': 'Demographics',
                'priority': demo_rec['priority'].title(),
                'title': demo_rec['type'].replace('_', ' ').title(),
                'description': demo_rec['description'],
                'actions': [
                    'Review demographic classification algorithms',
                    'Consider additional demographic data sources',
                    'Validate genre-demographic mappings'
                ]
            })
        
        # Data freshness recommendations
        freshness = self.validation_results['quality_metrics']['data_freshness']
        if freshness['file_age_days'] > 30:
            recommendations.append({
                'category': 'Data Freshness',
                'priority': 'Medium',
                'title': 'Update Dataset',
                'description': f'Dataset is {freshness["file_age_days"]} days old.',
                'actions': [
                    'Refresh data from source APIs',
                    'Implement automated data updates',
                    'Monitor data freshness metrics'
                ]
            })
        
        self.validation_results['recommendations'] = recommendations
        return recommendations
    
    def generate_visualization_data(self):
        """Generate data for visualizations"""
        logger.info("Generating visualization data...")
        
        viz_data = {
            'quality_distribution': {},
            'feature_distributions': {},
            'platform_coverage': {},
            'source_distribution': {},
            'demographic_mapping': {}
        }
        
        # Quality score distribution
        if 'data_quality_score' in self.df.columns:
            quality_bins = pd.cut(self.df['data_quality_score'], bins=[0, 60, 70, 80, 90, 100], 
                                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
            viz_data['quality_distribution'] = quality_bins.value_counts().to_dict()
        
        # Feature distributions (for histograms)
        for feature in self.audio_features[:6]:  # Top 6 features
            if feature in self.df.columns:
                values = self.df[feature].dropna()
                if len(values) > 0:
                    hist, bins = np.histogram(values, bins=20)
                    viz_data['feature_distributions'][feature] = {
                        'values': hist.tolist(),
                        'bins': bins.tolist(),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
        
        # Platform coverage
        platforms = ['spotify', 'tiktok', 'youtube']
        for platform in platforms:
            if platform in self.df.columns:
                count = int(self.df[platform].sum())
                viz_data['platform_coverage'][platform] = count
        
        # Source count distribution
        if 'source_count' in self.df.columns:
            source_dist = self.df['source_count'].value_counts().sort_index()
            viz_data['source_distribution'] = source_dist.to_dict()
        
        # Genre-demographic mapping
        if 'genre' in self.df.columns:
            genre_counts = self.df['genre'].value_counts().head(10)
            viz_data['demographic_mapping'] = genre_counts.to_dict()
        
        return viz_data
    
    def save_validation_report(self):
        """Save comprehensive validation report"""
        logger.info("Saving validation report...")
        
        # Save JSON report
        report_file = os.path.join(self.output_dir, 'enhanced_validation_report.json')
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = os.path.join(self.output_dir, 'validation_summary.txt')
        self._generate_text_summary(summary_file)
        
        # Generate visualization data
        viz_data = self.generate_visualization_data()
        viz_file = os.path.join(self.output_dir, 'visualization_data.json')
        with open(viz_file, 'w') as f:
            json.dump(viz_data, f, indent=2, default=str)
        
        logger.info(f"Validation reports saved to {self.output_dir}")
        return {
            'json_report': report_file,
            'summary_report': summary_file,
            'visualization_data': viz_file
        }
    
    def _generate_text_summary(self, output_file):
        """Generate human-readable text summary"""
        overview = self.validation_results['dataset_overview']
        quality = self.validation_results['quality_metrics']
        recommendations = self.validation_results['recommendations']
        
        with open(output_file, 'w') as f:
            f.write("ENHANCED VALIDATION REPORT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Tracks: {overview['total_tracks']:,}\n")
            f.write(f"Unique Artists: {overview['unique_artists']:,}\n")
            f.write(f"Multi-source Tracks: {overview['source_distribution']['multi_source']:,}\n")
            f.write(f"Average Sources per Track: {overview['source_distribution']['avg_sources']:.1f}\n\n")
            
            # Quality Metrics
            f.write("QUALITY METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Quality Score: {quality['overall_quality_score']:.1f}/100 ({quality['quality_categories']['overall_category']})\n\n")
            
            f.write("Component Scores:\n")
            for component, score in quality['component_scores'].items():
                f.write(f"  {component.replace('_', ' ').title()}: {score:.1f}/100\n")
            
            f.write(f"\nData Freshness: {quality['data_freshness']['freshness_score']:.1f}/100 ({quality['data_freshness']['file_age_days']} days old)\n\n")
            
            # Platform Coverage
            f.write("PLATFORM COVERAGE\n")
            f.write("-" * 20 + "\n")
            for platform, data in overview['platform_coverage'].items():
                f.write(f"{platform.title()}: {data['count']:,} tracks ({data['percentage']:.1f}%)\n")
            f.write("\n")
            
            # Top Recommendations
            f.write("TOP RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            high_priority = [r for r in recommendations if r['priority'] == 'High']
            for i, rec in enumerate(high_priority[:5], 1):
                f.write(f"{i}. [{rec['category']}] {rec['title']}\n")
                f.write(f"   {rec['description']}\n\n")
            
            # Audio Feature Summary
            audio_consistency = self.validation_results.get('audio_feature_consistency', {})
            if 'quality_flags' in audio_consistency:
                f.write("AUDIO FEATURE ISSUES\n")
                f.write("-" * 20 + "\n")
                flags = audio_consistency['quality_flags']
                if flags:
                    for flag in flags[:5]:
                        f.write(f"- {flag['feature']}: {flag['description']}\n")
                else:
                    f.write("No significant audio feature issues detected.\n")
                f.write("\n")
            
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def run_complete_validation(self):
        """Run the complete enhanced validation suite"""
        logger.info("Starting enhanced validation suite...")
        
        # Load dataset
        if not self.load_dataset():
            return None
        
        print("Enhanced validation suite")
        print("=" * 50)
        print(f"Dataset: {os.path.basename(self.dataset_path)}")
        print(f"Tracks: {len(self.df):,}")
        print()
        
        # Step 1: Dataset Overview
        print("Generating dataset overview...")
        self.generate_dataset_overview()
        
        # Step 2: Audio Feature Consistency
        print("Validating audio feature consistency...")
        self.validate_audio_feature_consistency()
        
        # Step 3: Demographic Validation
        print("Validating demographic mapping...")
        self.validate_demographic_mapping()
        
        # Step 4: Quality Metrics
        print("Generating quality metrics...")
        self.generate_quality_metrics()
        
        # Step 5: Recommendations
        print("Generating recommendations...")
        self.generate_recommendations()
        
        # Step 6: Save Reports
        print("Saving validation reports...")
        report_files = self.save_validation_report()
        
        print("\nValidation complete!")
        print("=" * 50)
        
        # Display summary
        quality_score = self.validation_results['quality_metrics']['overall_quality_score']
        quality_category = self.validation_results['quality_metrics']['quality_categories']['overall_category']
        
        print(f"Overall Quality: {quality_score:.1f}/100 ({quality_category})")
        
        component_scores = self.validation_results['quality_metrics']['component_scores']
        print(f"\nComponent Scores:")
        for component, score in component_scores.items():
            status = "OK" if score >= 80 else "Warning" if score >= 60 else "Error"
            print(f"  {status} {component.replace('_', ' ').title()}: {score:.1f}/100")
        
        recommendations = self.validation_results['recommendations']
        high_priority = [r for r in recommendations if r['priority'] == 'High']
        
        if high_priority:
            print(f"\nHigh Priority Issues ({len(high_priority)}):")
            for rec in high_priority[:3]:
                print(f"  • {rec['title']}")
        else:
            print(f"\nNo high priority issues found!")
        
        print(f"\nReports saved to: {self.output_dir}")
        for desc, file_path in report_files.items():
            print(f"  • {desc}: {os.path.basename(file_path)}")
        
        return self.validation_results

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Validation System')
    parser.add_argument('--dataset', default='integration_cache/master_music_dataset_deduplicated.csv',
                       help='Path to master dataset')
    parser.add_argument('--output-dir', default='validation_reports',
                       help='Output directory for reports')
    parser.add_argument('--component', choices=['overview', 'audio', 'demographic', 'quality', 'all'],
                       default='all', help='Validation component to run')
    
    args = parser.parse_args()
    
    validator = EnhancedValidationSystem(
        dataset_path=args.dataset,
        output_dir=args.output_dir
    )
    
    if args.component == 'all':
        # Run complete validation
        results = validator.run_complete_validation()
    else:
        # Run specific component
        validator.load_dataset()
        
        if args.component == 'overview':
            results = validator.generate_dataset_overview()
        elif args.component == 'audio':
            results = validator.validate_audio_feature_consistency()
        elif args.component == 'demographic':
            results = validator.validate_demographic_mapping()
        elif args.component == 'quality':
            results = validator.generate_quality_metrics()
        
        print(json.dumps(results, indent=2, default=str))