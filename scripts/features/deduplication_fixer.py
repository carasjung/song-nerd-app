import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re
import os
import logging

logger = logging.getLogger(__name__)

class DuplicationFixer:
    """
    Fix duplicate songs that weren't properly merged during integration
    """
    
    def __init__(self, dataset_path='integration_cache/master_music_dataset_fixed.csv'):
        self.dataset_path = dataset_path
        self.df = None
        
    def load_dataset(self):
        """Load the master dataset"""
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset not found: {self.dataset_path}")
            return False
        
        logger.info("Loading master dataset...")
        self.df = pd.read_csv(self.dataset_path)
        logger.info(f"Loaded {len(self.df)} tracks")
        return True
    
    def analyze_duplicates(self):
        """Analyze duplicate patterns in the dataset"""
        if self.df is None:
            self.load_dataset()
        
        analysis = {
            'total_tracks': len(self.df),
            'potential_duplicates': {},
            'exact_duplicates': {},
            'single_source_tracks': 0,
            'multi_source_tracks': 0
        }
        
        # Count source distribution
        analysis['single_source_tracks'] = (self.df['source_count'] == 1).sum()
        analysis['multi_source_tracks'] = (self.df['source_count'] > 1).sum()
        
        # Find exact track+artist duplicates
        track_artist_combos = self.df.groupby(['track_name', 'artist_name']).size()
        exact_dupes = track_artist_combos[track_artist_combos > 1]
        
        analysis['exact_duplicates'] = {
            'count': len(exact_dupes),
            'examples': exact_dupes.head(10).to_dict()
        }
        
        # Find potential fuzzy duplicates
        potential_dupes = self._find_fuzzy_duplicates()
        analysis['potential_duplicates'] = {
            'count': len(potential_dupes),
            'examples': list(potential_dupes.keys())[:10]
        }
        
        return analysis
    
    def _find_fuzzy_duplicates(self, sample_size=1000):
        """Find fuzzy duplicate groups (limited sample for speed)"""
        # Take a sample for analysis
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        
        duplicate_groups = {}
        processed_indices = set()
        
        for i, row1 in sample_df.iterrows():
            if i in processed_indices:
                continue
            
            track1 = self._clean_for_matching(row1['track_name'])
            artist1 = self._clean_for_matching(row1['artist_name'])
            key1 = f"{artist1}||{track1}"
            
            similar_tracks = [i]
            processed_indices.add(i)
            
            for j, row2 in sample_df.iterrows():
                if j in processed_indices or i == j:
                    continue
                
                track2 = self._clean_for_matching(row2['track_name'])
                artist2 = self._clean_for_matching(row2['artist_name'])
                key2 = f"{artist2}||{track2}"
                
                # Check if they're similar
                similarity = fuzz.ratio(key1, key2)
                if similarity >= 90:  # High threshold for duplicates
                    similar_tracks.append(j)
                    processed_indices.add(j)
            
            if len(similar_tracks) > 1:
                group_key = f"{row1['artist_name']} - {row1['track_name']}"
                duplicate_groups[group_key] = similar_tracks
        
        return duplicate_groups
    
    def _clean_for_matching(self, text):
        """Clean text for more aggressive matching"""
        if pd.isna(text):
            return ''
        
        text = str(text).lower()
        
        # More aggressive cleaning for duplicates
        replacements = {
            r'\(.*?\)': '',  # Remove all parentheses content
            r'\[.*?\]': '',  # Remove all bracket content
            r'feat\.?\s*.*$': '',  # Remove everything after feat
            r'ft\.?\s*.*$': '',   # Remove everything after ft
            r'featuring\s*.*$': '',  # Remove everything after featuring
            r'with\s+.*$': '',    # Remove everything after "with"
            r'&.*$': '',          # Remove everything after &
            r'[^\w\s]': ' ',      # Remove all special characters
            r'\s+': ' '           # Multiple spaces to single
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def remove_exact_duplicates(self):
        """Remove exact duplicates and merge their data"""
        if self.df is None:
            self.load_dataset()
        
        logger.info("Removing exact duplicates...")
        original_count = len(self.df)
        
        # Group by exact track+artist matches
        grouped = self.df.groupby(['track_name', 'artist_name'])
        
        merged_records = []
        
        for (track, artist), group in grouped:
            if len(group) == 1:
                # No duplicates, keep as is
                merged_records.append(group.iloc[0].to_dict())
            else:
                # Merge duplicates
                merged_record = self._merge_duplicate_group(group)
                merged_records.append(merged_record)
        
        # Create new dataframe
        self.df = pd.DataFrame(merged_records)
        
        final_count = len(self.df)
        removed_count = original_count - final_count
        
        logger.info(f"Removed {removed_count} exact duplicates. {final_count} tracks remaining.")
        
        return removed_count
    
    def remove_fuzzy_duplicates(self, similarity_threshold=95):
        """Remove fuzzy duplicates using aggressive matching"""
        if self.df is None:
            self.load_dataset()
        
        logger.info(f"Removing fuzzy duplicates (threshold: {similarity_threshold})...")
        original_count = len(self.df)
        
        # Create clean keys for all tracks
        self.df['clean_key'] = self.df.apply(
            lambda row: f"{self._clean_for_matching(row['artist_name'])}||{self._clean_for_matching(row['track_name'])}", 
            axis=1
        )
        
        # Find duplicate groups
        duplicate_groups = {}
        processed_keys = set()
        
        for idx, row in self.df.iterrows():
            key = row['clean_key']
            if key in processed_keys or not key.strip():
                continue
            
            # Find all similar keys
            similar_indices = [idx]
            processed_keys.add(key)
            
            for idx2, row2 in self.df.iterrows():
                if idx2 == idx or row2['clean_key'] in processed_keys:
                    continue
                
                similarity = fuzz.ratio(key, row2['clean_key'])
                if similarity >= similarity_threshold:
                    similar_indices.append(idx2)
                    processed_keys.add(row2['clean_key'])
            
            if len(similar_indices) > 1:
                duplicate_groups[key] = similar_indices
        
        # Merge duplicate groups
        merged_records = []
        all_duplicate_indices = set()
        
        # First, process duplicates
        for group_key, indices in duplicate_groups.items():
            all_duplicate_indices.update(indices)
            group_df = self.df.loc[indices]
            merged_record = self._merge_duplicate_group(group_df)
            merged_records.append(merged_record)
        
        # Then, add non-duplicates
        non_duplicate_indices = set(self.df.index) - all_duplicate_indices
        for idx in non_duplicate_indices:
            merged_records.append(self.df.loc[idx].to_dict())
        
        # Create new dataframe
        self.df = pd.DataFrame(merged_records)
        self.df = self.df.drop(columns=['clean_key'], errors='ignore')
        
        final_count = len(self.df)
        removed_count = original_count - final_count
        
        logger.info(f"Removed {removed_count} fuzzy duplicates. {final_count} tracks remaining.")
        
        return removed_count
    
    def _merge_duplicate_group(self, group_df):
        """Merge a group of duplicate tracks into a single record"""
        if len(group_df) == 1:
            return group_df.iloc[0].to_dict()
        
        # Use the record with highest quality as base
        if 'data_quality_score' in group_df.columns:
            base_record = group_df.loc[group_df['data_quality_score'].idxmax()]
        else:
            base_record = group_df.iloc[0]
        
        merged = base_record.to_dict()
        
        # Merge specific fields
        merged['source_count'] = len(group_df)
        merged['all_sources'] = group_df['primary_source'].unique().tolist()
        
        # Platform presence (OR logic)
        merged['spotify'] = group_df['spotify'].any()
        merged['tiktok'] = group_df['tiktok'].any()  
        merged['youtube'] = group_df['youtube'].any()
        
        # Audio features (median for robustness)
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness']
        
        for feature in audio_features:
            if feature in group_df.columns:
                feature_values = group_df[feature].dropna()
                if len(feature_values) > 0:
                    merged[feature] = feature_values.median()
        
        # Popularity (use max for best representation)
        popularity_cols = ['normalized_popularity', 'avg_popularity', 'max_popularity']
        for pop_col in popularity_cols:
            if pop_col in group_df.columns:
                pop_values = group_df[pop_col].dropna()
                if len(pop_values) > 0:
                    merged[pop_col] = pop_values.max()
        
        # Genre (use most common non-unknown)
        genres = group_df['genre'][group_df['genre'] != 'unknown']
        if len(genres) > 0:
            merged['genre'] = genres.mode().iloc[0] if len(genres.mode()) > 0 else genres.iloc[0]
        
        # Recalculate quality score
        merged['data_quality_score'] = self._calculate_quality_score(merged)
        
        return merged
    
    def _calculate_quality_score(self, record):
        """Recalculate quality score for merged record"""
        score = 0
        
        # Audio features (40 points)
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
        audio_count = sum(1 for f in audio_features if pd.notna(record.get(f)))
        score += (audio_count / len(audio_features)) * 40
        
        # Has popularity (20 points)
        if pd.notna(record.get('normalized_popularity', record.get('avg_popularity'))):
            score += 20
        
        # Known genre (15 points)
        if record.get('genre', 'unknown') != 'unknown':
            score += 15
        
        # Multi-platform (15 points)
        platform_count = sum([
            record.get('spotify', False),
            record.get('tiktok', False),
            record.get('youtube', False)
        ])
        score += min(platform_count * 5, 15)
        
        # Multiple sources (10 points)
        if record.get('source_count', 1) > 1:
            score += 10
        
        return round(score, 1)
    
    def create_deduplicated_dataset(self, output_path='integration_cache/master_music_dataset_deduplicated.csv'):
        """Create a deduplicated version of the dataset"""
        if self.df is None:
            self.load_dataset()
        
        logger.info("Creating deduplicated dataset...")
        original_count = len(self.df)
        
        # Step 1: Remove exact duplicates
        exact_removed = self.remove_exact_duplicates()
        
        # Step 2: Remove fuzzy duplicates
        fuzzy_removed = self.remove_fuzzy_duplicates()
        
        # Step 3: Recalculate quality scores
        logger.info("Recalculating quality scores...")
        self.df['data_quality_score'] = self.df.apply(
            lambda row: self._calculate_quality_score(row.to_dict()), axis=1
        )
        
        # Step 4: Sort by quality (best first)
        self.df = self.df.sort_values('data_quality_score', ascending=False)
        
        # Step 5: Save deduplicated dataset
        self.df.to_csv(output_path, index=False)
        
        final_count = len(self.df)
        total_removed = original_count - final_count
        
        logger.info(f"Deduplication complete!")
        logger.info(f"Original: {original_count:,} tracks")
        logger.info(f"Final: {final_count:,} tracks")
        logger.info(f"Removed: {total_removed:,} duplicates ({total_removed/original_count*100:.1f}%)")
        logger.info(f"Saved to: {output_path}")
        
        return {
            'original_count': original_count,
            'final_count': final_count,
            'total_removed': total_removed,
            'exact_removed': exact_removed,
            'fuzzy_removed': fuzzy_removed,
            'output_path': output_path
        }

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix duplicate songs in master dataset')
    parser.add_argument('command', choices=['analyze', 'deduplicate'], 
                       help='analyze: show duplicates, deduplicate: remove duplicates')
    parser.add_argument('--input', default='integration_cache/master_music_dataset_fixed.csv',
                       help='Input dataset path')
    parser.add_argument('--output', default='integration_cache/master_music_dataset_deduplicated.csv',
                       help='Output deduplicated dataset path')
    parser.add_argument('--threshold', type=int, default=95,
                       help='Fuzzy matching threshold for duplicates (95-99)')
    
    args = parser.parse_args()
    
    fixer = DuplicationFixer(args.input)
    
    if args.command == 'analyze':
        print("Analyzing duplicate patterns...")
        analysis = fixer.analyze_duplicates()
        
        print(f"\nDuplicate analysis")
        print(f"Total tracks: {analysis['total_tracks']:,}")
        print(f"Single source tracks: {analysis['single_source_tracks']:,}")
        print(f"Multi-source tracks: {analysis['multi_source_tracks']:,}")
        
        print(f"\nExact duplicates")
        exact = analysis['exact_duplicates']
        print(f"Groups found: {exact['count']}")
        if exact['examples']:
            print("Examples:")
            for combo, count in list(exact['examples'].items())[:5]:
                print(f"  '{combo[1]}' by '{combo[0]}': {count} copies")
        
        print(f"\nFuzzy duplicates (sample)")
        fuzzy = analysis['potential_duplicates']
        print(f"Groups found: {fuzzy['count']}")
        if fuzzy['examples']:
            print("Examples:")
            for example in fuzzy['examples'][:5]:
                print(f"  {example}")
    
    elif args.command == 'deduplicate':
        print(f"Removing duplicates (threshold: {args.threshold})...")
        results = fixer.create_deduplicated_dataset(args.output)
        
        print(f"\nDeduplication complete")
        print(f"Original: {results['original_count']:,} tracks")
        print(f"Final: {results['final_count']:,} tracks")
        print(f"Removed: {results['total_removed']:,} duplicates")
        print(f"   - Exact duplicates: {results['exact_removed']:,}")
        print(f"   - Fuzzy duplicates: {results['fuzzy_removed']:,}")
        
        reduction_pct = (results['total_removed'] / results['original_count']) * 100
        print(f"Reduction: {reduction_pct:.1f}%")
        
        print(f"\nDeduplicated dataset: {results['output_path']}")
        print(f"Test search: python3 scripts/features/optimized_integration_system.py search --query \"Bad Guy\" --cache-dir integration_cache")