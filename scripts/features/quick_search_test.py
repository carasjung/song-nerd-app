# quick_search_test.py
# This script is used to test the quick search functionality of the master dataset.
# It uses the fuzzy matching algorithm to match the songs from the different sources.
# It also uses the audio features of the song to generate insights on the song's target demographic,
# platform recommendations, and marketing suggestions.
# It also uses the audio features of the song to generate insights on the song's trend alignment, 
# similar artists, viral potential and overall trend alignment.

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re
import os

def clean_text_for_matching(text):
    """Clean text for fuzzy matching"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)  # Remove parentheses/brackets
    text = re.sub(r'feat\.?\s*|ft\.?\s*|featuring\s*', ' ', text)  # Remove feat
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Clean spaces
    
    return text

def search_deduplicated_dataset(query, limit=10):
    """Search the deduplicated dataset directly"""
    
    # Check which file to use
    deduplicated_path = 'integration_cache/master_music_dataset_deduplicated.csv'
    fixed_path = 'integration_cache/master_music_dataset_fixed.csv'
    
    if os.path.exists(deduplicated_path):
        dataset_path = deduplicated_path
        print(f"Using deduplicated dataset: {dataset_path}")
    elif os.path.exists(fixed_path):
        dataset_path = fixed_path
        print(f"Deduplicated dataset not found, using fixed: {dataset_path}")
    else:
        print("No dataset found!")
        return []
    
    # Load dataset
    print(f"Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset size: {len(df):,} tracks")
    
    # Check for duplicates in the dataset
    exact_dupes = df.groupby(['track_name', 'artist_name']).size()
    duplicate_count = (exact_dupes > 1).sum()
    print(f"Exact duplicates still in dataset: {duplicate_count}")
    
    # Perform search
    query_clean = clean_text_for_matching(query)
    results = []
    
    print(f"Searching for: '{query}'")
    
    for idx, row in df.iterrows():
        max_similarity = 0
        
        # Search in track name and artist name
        for field in ['track_name', 'artist_name']:
            if field in row and pd.notna(row[field]):
                field_value = clean_text_for_matching(str(row[field]))
                similarity = fuzz.partial_ratio(query_clean, field_value)
                max_similarity = max(max_similarity, similarity)
        
        if max_similarity > 60:  # Threshold for search results
            popularity = row.get('normalized_popularity', row.get('avg_popularity', 0))
            if pd.isna(popularity):
                popularity = 0
            
            results.append({
                'track_name': row.get('track_name', ''),
                'artist_name': row.get('artist_name', ''),
                'similarity_score': max_similarity,
                'source_count': row.get('source_count', 1),
                'popularity': float(popularity),
                'genre': row.get('genre', 'unknown'),
                'spotify': bool(row.get('spotify', False)),
                'tiktok': bool(row.get('tiktok', False)),
                'youtube': bool(row.get('youtube', False)),
                'quality_score': row.get('data_quality_score', 0)
            })
    
    # Sort by similarity and popularity
    results.sort(key=lambda x: (x['similarity_score'], x['popularity']), reverse=True)
    
    return results[:limit]

def analyze_dataset_quality(dataset_path):
    """Analyze the quality of the dataset"""
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    
    print(f"\nDataset Analysis: {os.path.basename(dataset_path)}")
    print(f"Total tracks: {len(df):,}")
    
    # Check for exact duplicates
    exact_dupes = df.groupby(['track_name', 'artist_name']).size()
    duplicate_groups = exact_dupes[exact_dupes > 1]
    print(f"Exact duplicate groups: {len(duplicate_groups)}")
    print(f"Total duplicate tracks: {(exact_dupes - 1).sum()}")
    
    # Source count distribution
    if 'source_count' in df.columns:
        source_dist = df['source_count'].value_counts().sort_index()
        print(f"\nSource count distribution:")
        for count, tracks in source_dist.items():
            print(f"  {count} source(s): {tracks:,} tracks")
    
    # Platform distribution
    platforms = ['spotify', 'tiktok', 'youtube']
    print(f"\nPlatform coverage:")
    for platform in platforms:
        if platform in df.columns:
            count = df[platform].sum()
            percentage = (count / len(df)) * 100
            print(f"  {platform.title()}: {count:,} tracks ({percentage:.1f}%)")
    
    # Show examples of duplicates if they exist
    if len(duplicate_groups) > 0:
        print(f"\nExample duplicate groups:")
        for i, ((track, artist), count) in enumerate(duplicate_groups.head(5).items()):
            print(f"  {i+1}. '{track}' by '{artist}': {count} copies")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 quick_search_test.py <search_query>")
        print("       python3 quick_search_test.py analyze")
        sys.exit(1)
    
    query = sys.argv[1]
    
    if query.lower() == 'analyze':
        # Analyze both datasets
        print("Analyzing Datasets")
        analyze_dataset_quality('integration_cache/master_music_dataset_fixed.csv')
        analyze_dataset_quality('integration_cache/master_music_dataset_deduplicated.csv')
    else:
        # Perform search
        results = search_deduplicated_dataset(query, limit=10)
        
        if results:
            print(f"\nSearch Results ({len(results)} found)")
            
            for i, result in enumerate(results, 1):
                platforms = []
                if result['spotify']: platforms.append('Spotify')
                if result['tiktok']: platforms.append('TikTok')  
                if result['youtube']: platforms.append('YouTube')
                
                print(f"{i:2d}. {result['track_name']} by {result['artist_name']}")
                print(f"     Match: {result['similarity_score']}% | Genre: {result['genre']}")
                print(f"     Sources: {result['source_count']} | Platforms: {', '.join(platforms) if platforms else 'Unknown'}")
                
                if result['popularity'] > 0:
                    print(f"     Popularity: {result['popularity']:.1f}")
                if result['quality_score'] > 0:
                    print(f"     Quality: {result['quality_score']:.1f}/100")
                print()
        else:
            print(f"No results found for '{query}'")