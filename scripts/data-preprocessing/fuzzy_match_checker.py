# fuzzy_match_checker.py

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_multi_score(query: str, candidate: str) -> Dict[str, float]:
    """Calculate multiple similarity scores for better matching."""
    return {
        'ratio': fuzz.ratio(query, candidate),
        'partial_ratio': fuzz.partial_ratio(query, candidate),
        'token_sort_ratio': fuzz.token_sort_ratio(query, candidate),
        'token_set_ratio': fuzz.token_set_ratio(query, candidate)
    }

def weighted_score(scores: Dict[str, float]) -> float:
    """Calculate weighted score from multiple algorithms."""
    # Weights based on empirical testing for music data
    weights = {
        'ratio': 0.25,
        'partial_ratio': 0.15,
        'token_sort_ratio': 0.35,
        'token_set_ratio': 0.25
    }
    
    return sum(scores[alg] * weight for alg, weight in weights.items())

def process_chunk_worker(df1_chunk: pd.DataFrame, df2: pd.DataFrame, 
                        chunk_id: int, min_score: int, top_n: int) -> List[Dict]:
    """Standalone function for parallel processing of chunks."""
    matches = []
    
    # Create candidate list once
    candidates = df2['search_key'].tolist()
    
    for idx, row in df1_chunk.iterrows():
        query = row['search_key']
        
        # Skip empty queries
        if not query:
            continue
        
        # Get best matches using rapidfuzz
        best_matches = process.extract(
            query, 
            candidates, 
            limit=top_n,
            scorer=fuzz.token_sort_ratio  # Generally best for music
        )
        
        for match_text, score, match_idx in best_matches:
            if score >= min_score:
                # Calculate detailed scores for high-confidence matches
                detailed_scores = calculate_multi_score(query, match_text)
                w_score = weighted_score(detailed_scores)
                
                # Get original row data
                df2_row = df2.iloc[match_idx]
                
                match_data = {
                    # Source data
                    'df1_index': idx,
                    'df1_artist': row['artist_clean'],
                    'df1_track': row['track_clean'],
                    'df1_search_key': query,
                    
                    # Match data
                    'df2_index': match_idx,
                    'df2_artist': df2_row['artist_clean'],
                    'df2_track': df2_row['track_clean'],
                    'df2_search_key': match_text,
                    
                    # Scoring
                    'primary_score': score,
                    'weighted_score': w_score,
                    **{f'score_{k}': v for k, v in detailed_scores.items()},
                    
                    # Metadata
                    'match_rank': len([m for m in best_matches if m[1] > score]) + 1,
                    'chunk_id': chunk_id
                }
                
                matches.append(match_data)
    
    return matches

class FuzzyMatcher:
    """Enhanced fuzzy matching with multiple algorithms and performance optimizations."""
    
    def __init__(self, min_score: int = 85, top_n: int = 3):
        self.min_score = min_score
        self.top_n = top_n
        
    def create_search_key(self, artist: str, track: str, separator: str = " - ") -> str:
        """Create search key from artist and track, handling missing values."""
        artist = str(artist).strip() if pd.notna(artist) else ""
        track = str(track).strip() if pd.notna(track) else ""
        
        if not artist and not track:
            return ""
        elif not artist:
            return track
        elif not track:
            return artist
        else:
            return f"{artist}{separator}{track}"
    
    def preprocess_dataframe(self, df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
        """Clean and validate dataframe before matching."""
        logger.info(f"Preprocessing {name}: {len(df)} rows")
        
        # Ensure required columns exist
        required_cols = ['artist_clean', 'track_clean']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {name}: {missing_cols}")
        
        # Remove rows where both artist and track are empty
        df_clean = df.copy()
        df_clean['search_key'] = df_clean.apply(
            lambda row: self.create_search_key(row['artist_clean'], row['track_clean']), 
            axis=1
        )
        
        # Filter out empty search keys
        df_clean = df_clean[df_clean['search_key'].str.len() > 0].copy()
        
        # Add normalized versions for better matching
        df_clean['artist_tokens'] = df_clean['artist_clean'].str.lower().str.split()
        df_clean['track_tokens'] = df_clean['track_clean'].str.lower().str.split()
        
        logger.info(f"After preprocessing {name}: {len(df_clean)} valid rows")
        return df_clean
    
    def calculate_multi_score(self, query: str, candidate: str) -> Dict[str, float]:
        """Calculate multiple similarity scores for better matching."""
        return calculate_multi_score(query, candidate)
    
    def weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted score from multiple algorithms."""
        return weighted_score(scores)
    
    def match_chunk(self, df1_chunk: pd.DataFrame, df2: pd.DataFrame, 
                   chunk_id: int) -> List[Dict]:
        """Process a chunk of df1 against df2 for parallel processing."""
        return process_chunk_worker(df1_chunk, df2, chunk_id, self.min_score, self.top_n)
    
    def match_across_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                            df1_name: str = "dataset1", df2_name: str = "dataset2",
                            use_parallel: bool = True, chunk_size: int = 1000) -> pd.DataFrame:
        """
        Match tracks across two datasets with enhanced scoring and parallel processing.
        """
        start_time = time.time()
        logger.info(f"Starting fuzzy matching: {df1_name} vs {df2_name}")
        
        df1_clean = self.preprocess_dataframe(df1, df1_name)
        df2_clean = self.preprocess_dataframe(df2, df2_name)
        
        if df1_clean.empty or df2_clean.empty:
            logger.warning("One or both datasets are empty after preprocessing")
            return pd.DataFrame()
        
        # Remove exact duplicates within df2 to speed up matching
        df2_deduped = df2_clean.drop_duplicates(subset=['search_key']).copy()
        df2_deduped.reset_index(drop=True, inplace=True)
        
        logger.info(f"Matching {len(df1_clean)} records against {len(df2_deduped)} unique candidates")
        
        all_matches = []
        
        if use_parallel and len(df1_clean) > chunk_size:
            num_cores = min(mp.cpu_count() - 1, 4)  # Leave one core free, max 4
            logger.info(f"Using parallel processing with {num_cores} cores")
            
            # Split df1 into chunks
            chunks = [df1_clean[i:i + chunk_size] for i in range(0, len(df1_clean), chunk_size)]
            
            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = [
                    executor.submit(process_chunk_worker, chunk, df2_deduped, chunk_id, self.min_score, self.top_n) 
                    for chunk_id, chunk in enumerate(chunks)
                ]
                
                for future in futures:
                    chunk_matches = future.result()
                    all_matches.extend(chunk_matches)
                    
        else:
            # Sequential processing
            logger.info("Using sequential processing")
            chunk_matches = process_chunk_worker(df1_clean, df2_deduped, 0, self.min_score, self.top_n)
            all_matches.extend(chunk_matches)
        
        # Convert to DF
        if all_matches:
            result_df = pd.DataFrame(all_matches)
            
            # Sort by descending score and remove duplicates
            result_df = result_df.sort_values(['df1_index', 'weighted_score'], 
                                            ascending=[True, False])
            
            # Add confidence categories
            result_df['confidence'] = pd.cut(
                result_df['weighted_score'],
                bins=[0, 70, 85, 95, 100],
                labels=['low', 'medium', 'high', 'very_high'],
                include_lowest=True
            )
            
        else:
            logger.warning("No matches found above threshold")
            result_df = pd.DataFrame()
        
        processing_time = time.time() - start_time
        logger.info(f"Matching completed in {processing_time:.1f}s. Found {len(result_df)} matches.")
        
        return result_df
    
    def find_reciprocal_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Find matches that appear in both directions (A->B and B->A)."""
        if matches_df.empty:
            return pd.DataFrame()
        
        # Create lookup for reverse matches
        reverse_lookup = set()
        for _, row in matches_df.iterrows():
            reverse_lookup.add((row['df2_index'], row['df1_index']))
        
        # Find reciprocal matches
        reciprocal_matches = []
        for _, row in matches_df.iterrows():
            if (row['df1_index'], row['df2_index']) in reverse_lookup:
                reciprocal_matches.append(row)
        
        if reciprocal_matches:
            reciprocal_df = pd.DataFrame(reciprocal_matches)
            reciprocal_df['is_reciprocal'] = True
            return reciprocal_df
        else:
            return pd.DataFrame()
    
    def generate_match_report(self, matches_df: pd.DataFrame, 
                            output_dir: str = "data/match_reports") -> Dict:
        """Generate comprehensive matching report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if matches_df.empty:
            logger.warning("No matches to report")
            return {}
        
        # Basic statistics
        stats = {
            'total_matches': len(matches_df),
            'unique_df1_matches': matches_df['df1_index'].nunique(),
            'unique_df2_matches': matches_df['df2_index'].nunique(),
            'avg_score': matches_df['weighted_score'].mean(),
            'median_score': matches_df['weighted_score'].median(),
        }
        
        # Confidence distribution
        conf_dist = matches_df['confidence'].value_counts().to_dict()
        stats['confidence_distribution'] = conf_dist
        
        # Score distribution
        score_bins = pd.cut(matches_df['weighted_score'], bins=10)
        score_dist = score_bins.value_counts().sort_index()
        
        # Save detailed results
        matches_df.to_csv(output_path / "detailed_matches.csv", index=False)
        
        # Save high-confidence matches only
        high_conf = matches_df[matches_df['confidence'].isin(['high', 'very_high'])]
        if not high_conf.empty:
            high_conf.to_csv(output_path / "high_confidence_matches.csv", index=False)
        
        # Save summary
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(output_path / "match_summary.csv", index=False)
        
        logger.info(f"Match report saved to {output_path}")
        logger.info(f"Statistics: {stats}")
        
        return stats

def main():
    """Main execution function with error handling."""
    try:
        # Configuration
        matcher = FuzzyMatcher(min_score=85, top_n=3)
        
        # Load datasets
        logger.info("Loading datasets...")
        df1 = pd.read_csv("data/cleaned_normalized/spotify_ds_normalized.csv")
        df2 = pd.read_csv("data/cleaned_normalized/spotify_yt_normalized.csv")
        
        # Perform matching
        matches = matcher.match_across_datasets(
            df1, df2, 
            df1_name="spotify_ds", 
            df2_name="spotify_yt",
            use_parallel=True
        )
        
        if not matches.empty:
            # Save results
            output_file = "data/cleaned_normalized/fuzzy_matches_spotifyds_vs_spotifyyt.csv"
            matches.to_csv(output_file, index=False)
            
            # Create report
            stats = matcher.generate_match_report(matches)
            
            # Find reciprocal matches
            reciprocal = matcher.find_reciprocal_matches(matches)
            if not reciprocal.empty:
                logger.info(f"Found {len(reciprocal)} reciprocal matches")
                reciprocal.to_csv("data/cleaned_normalized/reciprocal_matches.csv", index=False)
            
            print(f"Found {len(matches)} matches above threshold.")
            print(f"High confidence matches: {len(matches[matches['confidence'].isin(['high', 'very_high'])])}")
            
        else:
            print("No matches found above the specified threshold.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()