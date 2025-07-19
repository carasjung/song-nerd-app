# prepare_audio_datasets.py
# Apply preprocessing and save cleaned versions

import os
from preprocess_audio_features import preprocess_audio_file
import pandas as pd
import json

def main():
    os.makedirs("data/cleaned", exist_ok=True)
    
    files = {
        'tiktok_2021': 'data/raw/market/tiktok_songs_2021.csv',
        'tiktok_2022': 'data/raw/market/tiktok_songs_2022.csv',
        'spotify_yt': 'data/raw/market/spotify_youtube.csv',
        'spotify_ds': 'data/raw/music/spotify-dataset.csv',
        'music_info': 'data/raw/music/million-spotify-lastfm/music_info.csv'
    }
    
    cleaned_dfs = {}
    
    for name, path in files.items():
        try:
            print(f"Processing {name}")
            
            df = preprocess_audio_file(path)
            cleaned_dfs[name] = df
            
            output_path = f"data/cleaned/{name}_clean.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved cleaned dataset to: {output_path}")
            
        except FileNotFoundError:
            print(f"Warning: File not found - {path}")
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
    
    create_summary_report(cleaned_dfs)

def create_summary_report(cleaned_dfs):
    """Create a summary report of all cleaned datasets"""
    from audio_feature_config import AUDIO_FEATURES
    
    summary = {}
    
    for name, df in cleaned_dfs.items():
        audio_features = [col for col in df.columns if col in AUDIO_FEATURES]
        
        summary[name] = {
            'shape': df.shape,
            'audio_features_count': len(audio_features),
            'audio_features': audio_features,
            'missing_audio_features': [col for col in AUDIO_FEATURES if col not in df.columns]
        }
    
    with open("data/cleaned/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("PROCESSING SUMMARY")
    for name, stats in summary.items():
        print(f"{name}: {stats['shape'][0]:,} rows, {stats['audio_features_count']} audio features")
        if stats['missing_audio_features']:
            print(f"  Missing: {stats['missing_audio_features']}")

if __name__ == "__main__":
    main()