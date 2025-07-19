# platform_data_analysis.py
# Analyze platform data to understand the zero-heavy distribution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_platform_data():
    try:
        platform_data = pd.read_csv('integrated_platform_training.csv')
        print(f"Platform training data shape: {platform_data.shape}")
    except FileNotFoundError:
        print("integrated_platform_training.csv not found. Running data integration first...")
        exec(open('data_integration.py').read())
        platform_data = pd.read_csv('integrated_platform_training.csv')
    
    # Analyze platform score columns
    platform_cols = ['spotify_combined', 'tiktok_combined', 'youtube_combined']
    
    print("\n" + "="*60)
    print("PLATFORM DATA ANALYSIS")
    print("="*60)
    
    for col in platform_cols:
        if col in platform_data.columns:
            values = platform_data[col].dropna()
            non_zero_count = (values > 0).sum()
            zero_count = (values == 0).sum()
            
            print(f"\n{col.upper()}:")
            print(f"  Total records: {len(values)}")
            print(f"  Non-zero values: {non_zero_count} ({non_zero_count/len(values)*100:.1f}%)")
            print(f"  Zero values: {zero_count} ({zero_count/len(values)*100:.1f}%)")
            print(f"  Mean (all): {values.mean():.2f}")
            print(f"  Mean (non-zero only): {values[values > 0].mean():.2f}")
            print(f"  Std (all): {values.std():.2f}")
            print(f"  Range: {values.min():.1f} - {values.max():.1f}")
            
            if non_zero_count > 0:
                print(f"  Non-zero distribution:")
                print(f"    25%: {values[values > 0].quantile(0.25):.1f}")
                print(f"    50%: {values[values > 0].quantile(0.50):.1f}")
                print(f"    75%: {values[values > 0].quantile(0.75):.1f}")
    
    return platform_data

def create_cleaned_platform_dataset(platform_data):
    """Create a cleaned dataset that's better for machine learning"""
    
    print(f"\n" + "="*60)
    print("CREATING CLEANED PLATFORM DATASET")
    print("="*60)
    
    platform_cols = ['spotify_combined', 'tiktok_combined', 'youtube_combined']
    
    # Strategy 1: Filter to songs with at least some platform data
    # Keep rows where at least one platform has a score > 5
    mask = (
        (platform_data['spotify_combined'] > 5) |
        (platform_data['tiktok_combined'] > 1) |
        (platform_data['youtube_combined'] > 1)
    )
    
    filtered_data = platform_data[mask].copy()
    print(f"After filtering for songs with real platform data: {filtered_data.shape}")
    
    # Strategy 2: Transform zero-heavy data using different approaches
    
    # Option A: Convert to binary classification (success/no success)
    for col in platform_cols:
        if col in filtered_data.columns:
            # Create binary success indicators
            threshold = {'spotify_combined': 30, 'tiktok_combined': 5, 'youtube_combined': 10}
            filtered_data[f'{col}_success'] = (filtered_data[col] > threshold.get(col, 10)).astype(int)
    
    # Option B: Use log transformation for non-zero values
    for col in platform_cols:
        if col in filtered_data.columns:
            # Log transform: log(x + 1) to handle zeros
            filtered_data[f'{col}_log'] = np.log1p(filtered_data[col])
    
    # Option C: Use ranking/percentile scores within non-zero values
    for col in platform_cols:
        if col in filtered_data.columns:
            non_zero_mask = filtered_data[col] > 0
            if non_zero_mask.sum() > 0:
                # Rank within non-zero values, scale to 0-100
                ranks = filtered_data.loc[non_zero_mask, col].rank(pct=True) * 100
                filtered_data[f'{col}_rank'] = 0
                filtered_data.loc[non_zero_mask, f'{col}_rank'] = ranks
    
    # Analyze the cleaned data
    print(f"\nCleaned dataset statistics:")
    for col in platform_cols:
        if col in filtered_data.columns:
            values = filtered_data[col]
            non_zero = (values > 0).sum()
            print(f"{col}: {non_zero}/{len(values)} non-zero ({non_zero/len(values)*100:.1f}%)")
    
    # Save cleaned dataset
    filtered_data.to_csv('cleaned_platform_training.csv', index=False)
    print(f"\nCleaned dataset saved as 'cleaned_platform_training.csv'")
    
    return filtered_data

def recommend_modeling_strategy(cleaned_data):
    """Recommend the best modeling approach based on data analysis"""
    
    print(f"\n" + "="*60)
    print("MODELING STRATEGY RECOMMENDATIONS")
    print("="*60)
    
    platform_cols = ['spotify_combined', 'tiktok_combined', 'youtube_combined']
    
    recommendations = {}
    
    for col in platform_cols:
        if col in cleaned_data.columns:
            values = cleaned_data[col]
            non_zero_pct = (values > 0).sum() / len(values) * 100
            
            platform = col.replace('_combined', '')
            
            if non_zero_pct < 10:
                recommendations[platform] = {
                    'approach': 'Binary Classification',
                    'reason': f'Only {non_zero_pct:.1f}% non-zero values - predict success/failure',
                    'target': f'{col}_success',
                    'model': 'Random Forest Classifier'
                }
            elif non_zero_pct < 30:
                recommendations[platform] = {
                    'approach': 'Two-Stage Model',
                    'reason': f'{non_zero_pct:.1f}% non-zero - first predict if score > 0, then predict value',
                    'target': f'{col}_success + {col}_rank',
                    'model': 'Classifier + Regressor'
                }
            else:
                recommendations[platform] = {
                    'approach': 'Log-Transformed Regression',
                    'reason': f'{non_zero_pct:.1f}% non-zero - sufficient data for regression',
                    'target': f'{col}_log',
                    'model': 'Random Forest Regressor'
                }
    
    for platform, rec in recommendations.items():
        print(f"\n{platform.upper()}:")
        print(f"  Recommended approach: {rec['approach']}")
        print(f"  Reason: {rec['reason']}")
        print(f"  Target variable: {rec['target']}")
        print(f"  Model type: {rec['model']}")
    
    return recommendations

def create_synthetic_platform_data(cleaned_data):
    """Create synthetic platform data to improve model training"""
    
    print(f"\n" + "="*60)
    print("CREATING SYNTHETIC PLATFORM DATA")
    print("="*60)
    
    synthetic_data = []
    
    # Define platform preferences based on audio features
    platform_rules = {
        'spotify': {
            'high_score_conditions': [
                ('valence > 0.6 and energy > 0.5', 'feel-good music'),
                ('acousticness > 0.6', 'acoustic/chill music'),
                ('danceability > 0.7 and energy > 0.6', 'dance/pop music')
            ],
            'base_score': 40,
            'bonus_range': (20, 40)
        },
        'tiktok': {
            'high_score_conditions': [
                ('danceability > 0.8 and energy > 0.7', 'viral dance potential'),
                ('speechiness > 0.15 and energy > 0.6', 'rap/hip-hop appeal'),
                ('valence > 0.7 and danceability > 0.6', 'upbeat viral content')
            ],
            'base_score': 10,
            'bonus_range': (30, 60)
        },
        'youtube': {
            'high_score_conditions': [
                ('energy > 0.7 and instrumentalness < 0.3', 'music video potential'),
                ('liveness > 0.3', 'live performance appeal'),
                ('valence > 0.6 and energy > 0.5', 'engaging content')
            ],
            'base_score': 15,
            'bonus_range': (25, 50)
        }
    }
    
    # Generate scores for each song based on rules
    enhanced_data = cleaned_data.copy()
    
    for platform, rules in platform_rules.items():
        col_name = f'{platform}_synthetic'
        scores = []
        
        for idx, row in enhanced_data.iterrows():
            base_score = rules['base_score']
            bonus = 0
            
            # Check each condition
            for condition, description in rules['high_score_conditions']:
                try:
                    # Create local variables for evaluation
                    local_vars = {
                        'danceability': row.get('danceability', 0.5),
                        'energy': row.get('energy', 0.5),
                        'valence': row.get('valence', 0.5),
                        'acousticness': row.get('acousticness', 0.5),
                        'speechiness': row.get('speechiness', 0.1),
                        'instrumentalness': row.get('instrumentalness', 0.1),
                        'liveness': row.get('liveness', 0.1),
                        'audio_appeal': row.get('audio_appeal', 50)
                    }
                    
                    if eval(condition, {"__builtins__": {}}, local_vars):
                        bonus += np.random.uniform(*rules['bonus_range'])
                        
                except:
                    pass  # Skip if evaluation fails
            
            # Add some randomness
            final_score = base_score + bonus + np.random.normal(0, 5)
            final_score = max(0, min(100, final_score))  # Clamp to 0-100
            scores.append(final_score)
        
        enhanced_data[col_name] = scores
        
        # Show statistics
        non_zero = (enhanced_data[col_name] > 10).sum()
        print(f"{platform}: {non_zero}/{len(enhanced_data)} songs above threshold ({non_zero/len(enhanced_data)*100:.1f}%)")
        print(f"  Mean score: {enhanced_data[col_name].mean():.1f}")
        print(f"  Score range: {enhanced_data[col_name].min():.1f} - {enhanced_data[col_name].max():.1f}")
    
    # Save enhanced dataset
    enhanced_data.to_csv('enhanced_platform_training.csv', index=False)
    print(f"\nEnhanced dataset saved as 'enhanced_platform_training.csv'")
    
    return enhanced_data

def main():
    """Run complete platform data analysis and improvement"""
    
    # Analyze current data
    platform_data = analyze_platform_data()
    
    # Create cleaned dataset
    cleaned_data = create_cleaned_platform_dataset(platform_data)
    
    # Get modeling recommendations
    recommendations = recommend_modeling_strategy(cleaned_data)
    
    # Create synthetic data for better training
    enhanced_data = create_synthetic_platform_data(cleaned_data)
    
    print(f"\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Use 'enhanced_platform_training.csv' for training")
    print("2. Consider the modeling strategy recommendations above")
    print("3. Update your platform model to handle the data characteristics")
    print("\nRecommended approach:")
    print("- Use synthetic scores for initial training")
    print("- Implement binary classification for platforms with sparse data")
    print("- Focus on relative ranking rather than absolute scores")

if __name__ == "__main__":
    main()