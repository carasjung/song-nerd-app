# data_integration.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def integrate_datasets():
    """Integrate all datasets for model training"""
    
    # Load datasets
    master_data = pd.read_csv('final_datasets/master_music_data.csv')
    platform_data = pd.read_csv('final_datasets/platform_performance.csv')
    demographics_data = pd.read_csv('final_datasets/demographic_preferences.csv')
    trend_data = pd.read_csv('final_datasets/trend_analysis.csv')
    test_data = pd.read_csv('final_datasets/test_dataset.csv')
    
    print("Dataset shapes:")
    print(f"Master: {master_data.shape}")
    print(f"Platform: {platform_data.shape}")
    print(f"Demographics: {demographics_data.shape}")
    print(f"Trends: {trend_data.shape}")
    print(f"Test: {test_data.shape}")
    
    return master_data, platform_data, demographics_data, trend_data, test_data

def prepare_demographics_training_data(master_data, demographics_data):
    """Prepare training data for demographics model"""
    
    # Clean and merge demographics data
    demo_processed = demographics_data.copy()
    
    # Extract age groups and regions from demographic column
    # Assuming format like "18-24_US" or "25-34_UK"
    demo_processed['age_group'] = demo_processed['demographic'].str.extract(r'(\d+-\d+|\d+\+)')
    demo_processed['region'] = demo_processed['demographic'].str.extract(r'_([A-Z]{2,3})$')
    
    # Handle cases where demographic might be just age or just region
    demo_processed['age_group'] = demo_processed['age_group'].fillna('unknown')
    demo_processed['region'] = demo_processed['region'].fillna('global')
    
    # Group by track to get primary demographics
    demo_summary = demo_processed.groupby(['track_name', 'artist_name']).agg({
        'age_group': lambda x: x.value_counts().index[0],  # Most common age group
        'region': lambda x: x.value_counts().index[0],     # Most common region
        'match_score': 'mean',                             # Average appeal
        'preferred_platform': lambda x: x.value_counts().index[0]  # Most preferred platform
    }).reset_index()
    
    # Merge with master data for audio features
    training_data = master_data.merge(
        demo_summary,
        left_on=['track_name_clean', 'artist_name_clean'],
        right_on=['track_name', 'artist_name'],
        how='inner'
    )
    
    # Add demographic confidence based on match_score
    training_data['demo_confidence'] = training_data['match_score'] / 100
    
    print(f"Demographics training data shape: {training_data.shape}")
    print(f"Age groups found: {training_data['age_group'].unique()}")
    print(f"Regions found: {training_data['region'].unique()}")
    
    return training_data

def prepare_platform_training_data(master_data, platform_data):
    """Prepare training data for platform recommendation model"""
    
    # Pivot platform data to get scores for each platform
    platform_pivot = platform_data.pivot_table(
        index=['track_name', 'artist_name'],
        columns='platform',
        values='popularity_score',
        aggfunc='mean'
    ).reset_index()
    
    # Fill missing platforms with 0
    for platform in ['spotify', 'tiktok', 'youtube', 'instagram']:
        if platform not in platform_pivot.columns:
            platform_pivot[platform] = 0
        else:
            platform_pivot[platform] = platform_pivot[platform].fillna(0)
    
    # Merge with master data
    platform_training = master_data.merge(
        platform_pivot,
        left_on=['track_name_clean', 'artist_name_clean'],
        right_on=['track_name', 'artist_name'],
        how='inner'
    )
    
    # Use existing platform scores from master_data if available
    for platform in ['spotify', 'tiktok', 'youtube']:
        if f'{platform}_x' in platform_training.columns:
            # Combine scores, prioritizing platform_data
            platform_training[f'{platform}_combined'] = platform_training[f'{platform}_y'].fillna(
                platform_training[f'{platform}_x']
            )
        else:
            platform_training[f'{platform}_combined'] = platform_training.get(platform, 0)
    
    print(f"Platform training data shape: {platform_training.shape}")
    return platform_training

# Run data integration
master_data, platform_data, demographics_data, trend_data, test_data = integrate_datasets()
demographics_training = prepare_demographics_training_data(master_data, demographics_data)
platform_training = prepare_platform_training_data(master_data, platform_data)

# Save integrated datasets
demographics_training.to_csv('integrated_demographics_training.csv', index=False)
platform_training.to_csv('integrated_platform_training.csv', index=False)
print("Integrated datasets saved!")