# normalize_cleaned_datasets.py

import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from normalize_names import clean_artist_name, clean_track_name, extract_featured_artists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('normalization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetNormalizer:
    """Class to handle dataset normalization with configurable column mappings."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurable column mappings 
        self.artist_columns = ['artist', 'artist_name', 'artists', 'main_artist', 'performer']
        self.track_columns = ['track', 'track_name', 'name', 'song', 'song_name', 'title']
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'total_rows': 0,
            'rows_with_artist': 0,
            'rows_with_track': 0,
            'errors': 0
        }
    
    def find_column(self, df: pd.DataFrame, possible_columns: List[str]) -> Optional[str]:
        """Find the first matching column from a list of possibilities."""
        df_columns_lower = [col.lower() for col in df.columns]
        
        for col in possible_columns:
            if col.lower() in df_columns_lower:
                # Return the actual column name (preserving case)
                return df.columns[df_columns_lower.index(col.lower())]
        return None
    
    def validate_dataframe(self, df: pd.DataFrame, filename: str) -> bool:
        """Validate dataframe before processing."""
        if df.empty:
            logger.warning(f"Empty dataframe in {filename}")
            return False
        
        if len(df.columns) == 0:
            logger.warning(f"No columns found in {filename}")
            return False
        
        return True
    
    def normalize_artist_track_columns(self, df: pd.DataFrame, filename: str = "") -> pd.DataFrame:
        """
        Normalize artist and track columns in dataframe.
        
        Args:
            df: Input dataframe
            filename: Optional filename for logging
            
        Returns:
            DF with normalized columns added
        """
        if not self.validate_dataframe(df, filename):
            return df
        
        # Find artist and track columns
        artist_col = self.find_column(df, self.artist_columns)
        track_col = self.find_column(df, self.track_columns)
        
        # Log column detection
        logger.info(f"File: {filename}")
        logger.info(f"  Artist column: {artist_col}")
        logger.info(f"  Track column: {track_col}")
        
        # Process artist column
        if artist_col:
            try:
                # Handle missing values
                df['artist_clean'] = df[artist_col].fillna('').astype(str).apply(
                    lambda x: clean_artist_name(x) if x and x.strip() else ''
                )
                
                # Extract featured artists
                featured_data = df[artist_col].fillna('').astype(str).apply(extract_featured_artists)
                df['main_artist'] = featured_data.apply(lambda x: x[0])
                df['featured_artists'] = featured_data.apply(lambda x: ', '.join(x[1]) if x[1] else '')
                
                self.stats['rows_with_artist'] += df['artist_clean'].str.len().gt(0).sum()
                
            except Exception as e:
                logger.error(f"Error processing artist column in {filename}: {e}")
                df['artist_clean'] = ''
                df['main_artist'] = ''
                df['featured_artists'] = ''
                self.stats['errors'] += 1
        else:
            df['artist_clean'] = ''
            df['main_artist'] = ''
            df['featured_artists'] = ''
        
        # Process track column
        if track_col:
            try:
                df['track_clean'] = df[track_col].fillna('').astype(str).apply(
                    lambda x: clean_track_name(x) if x and x.strip() else ''
                )
                
                self.stats['rows_with_track'] += df['track_clean'].str.len().gt(0).sum()
                
            except Exception as e:
                logger.error(f"Error processing track column in {filename}: {e}")
                df['track_clean'] = ''
                self.stats['errors'] += 1
        else:
            df['track_clean'] = ''
        
        # Add metadata columns
        df['normalization_source'] = filename
        df['has_featured_artists'] = df['featured_artists'].str.len() > 0
        
        return df
    
    def get_input_files(self, pattern: str = "*_clean.csv") -> List[Path]:
        """Get list of input files matching pattern."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        
        files = list(self.input_dir.glob(pattern))
        if not files:
            logger.warning(f"No files found matching pattern '{pattern}' in {self.input_dir}")
        
        return sorted(files)  # Sort for consistent processing order
    
    def process_file(self, filepath: Path) -> bool:
        """
        Process a single file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing: {filepath.name}")
            
            # Read CSV with error handling
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {filepath.name}, trying latin-1")
                df = pd.read_csv(filepath, encoding='latin-1')
            
            original_rows = len(df)
            self.stats['total_rows'] += original_rows
            
            # Normalize dataframe
            df_normalized = self.normalize_artist_track_columns(df, filepath.name)
            
            # Create output filename
            output_filename = filepath.name.replace("_clean.csv", "_normalized.csv")
            output_path = self.output_dir / output_filename
            
            # Save normalized data
            df_normalized.to_csv(output_path, index=False, encoding='utf-8')
            
            # Log summary
            clean_artists = (df_normalized['artist_clean'].str.len() > 0).sum()
            clean_tracks = (df_normalized['track_clean'].str.len() > 0).sum()
            featured_count = df_normalized['has_featured_artists'].sum()
            
            logger.info(f"  Rows: {original_rows}")
            logger.info(f"  Clean artists: {clean_artists}")
            logger.info(f"  Clean tracks: {clean_tracks}")
            logger.info(f"  With features: {featured_count}")
            logger.info(f"  Saved to: {output_path}")
            
            self.stats['files_processed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {filepath.name}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_all_files(self, file_pattern: str = "*_clean.csv") -> Dict[str, Any]:
        """
        Process all files matching the pattern.
        
        Returns:
            Processing statistics
        """
        logger.info("Starting dataset normalization...")
        
        input_files = self.get_input_files(file_pattern)
        
        if not input_files:
            logger.warning("No input files found")
            return self.stats
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Process each file
        successful = 0
        for filepath in input_files:
            if self.process_file(filepath):
                successful += 1
        
        # Final statistics
        logger.info("=" * 50)
        logger.info("NORMALIZATION COMPLETE")
        logger.info(f"Files processed successfully: {successful}/{len(input_files)}")
        logger.info(f"Total rows processed: {self.stats['total_rows']}")
        logger.info(f"Rows with clean artists: {self.stats['rows_with_artist']}")
        logger.info(f"Rows with clean tracks: {self.stats['rows_with_track']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        
        return self.stats
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Create summary report of the normalization process."""
        # This could be expanded to include per-file statistics
        summary_data = {
            'metric': ['files_processed', 'total_rows', 'rows_with_artist', 'rows_with_track', 'errors'],
            'value': [self.stats[key] for key in ['files_processed', 'total_rows', 'rows_with_artist', 'rows_with_track', 'errors']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "normalization_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Summary report saved to: {summary_path}")
        return summary_df

def main():
    """Main execution function."""
    # Configuration
    input_dir = "data/cleaned"
    output_dir = "data/cleaned_normalized"
    
    # Initialize normalizer
    normalizer = DatasetNormalizer(input_dir, output_dir)
    
    # Process all files
    stats = normalizer.process_all_files()
    
    # Generate summary report
    normalizer.generate_summary_report()
    
    return stats

if __name__ == "__main__":
    main()