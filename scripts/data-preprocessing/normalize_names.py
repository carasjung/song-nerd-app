# normalize_names.py
# Clean artist and track names, supports fuzzy matching

import re
import unicodedata
from rapidfuzz import fuzz
from typing import Tuple, List, Optional

def clean_artist_name(name: str, preserve_features: bool = False) -> str:
    """
    Clean and normalize artist names for matching.
    
    Args:
        name: Raw artist name
        preserve_features: If True, extracts featured artists separately
    
    Returns:
        Cleaned primary artist name
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    
    # Normalize unicode and convert to lowercase
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode().lower()
    
    # Handle common prefixes that should be preserved
    name = re.sub(r'^(the|a|an)\s+', '', name)
    
    # Extract featured artists if needed (before removing them)
    if preserve_features:
        featured_match = re.search(r'\s*(feat\.|ft\.|featuring|with|&|\+)\s+(.+)', name)
        if featured_match:
            # Store featured artists for later use if needed
            pass
    
    # Remove featured artists and collaborations
    name = re.sub(r'\s*(feat\.|ft\.|featuring|with|x|&|\+)\s+.*', '', name)
    
    # Remove common suffixes and prefixes
    name = re.sub(r'\s*(official|music|band|group|the|crew)$', '', name)
    
    # Clean special characters but preserve spaces
    name = re.sub(r'[^\w\s]', '', name)
    
    # Normalize multiple spaces
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()

def clean_track_name(name: str) -> str:
    """
    Clean and normalize track names for matching.
    
    Args:
        name: Raw track name
        
    Returns:
        Cleaned track name
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    
    # Normalize unicode and convert to lowercase
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode().lower()
    
    # Remove content in parentheses and brackets (versions, features, etc.)
    name = re.sub(r'\s*[\(\[].*?[\)\]]', '', name)
    
    # Remove common suffixes with separators
    suffixes_pattern = r'\s*[-–—\s]+(explicit|radio\s+edit|clean\s+version|instrumental|acoustic|live|remix|remaster|extended|original|official).*$'
    name = re.sub(suffixes_pattern, '', name)
    
    # Remove ft. from tracks
    name = re.sub(r'\s*(feat\.|ft\.|featuring)\s+.*', '', name)
    
    # Clean special characters but preserve spaces
    name = re.sub(r'[^\w\s]', '', name)
    
    # Normalize multiple spaces
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()

def extract_featured_artists(artist_name: str) -> Tuple[str, List[str]]:
    """
    Extract main artist and featured artists separately.
    
    Args:
        artist_name: Raw artist name with potential features
        
    Returns:
        Tuple of (main_artist, list_of_featured_artists)
    """
    if not isinstance(artist_name, str):
        return '', []
    
    # Find ft artists
    featured_pattern = r'\s*(feat\.|ft\.|featuring|with|x|&|\+)\s+(.+)'
    match = re.search(featured_pattern, artist_name, re.IGNORECASE)
    
    if match:
        main_artist = artist_name[:match.start()].strip()
        featured_part = match.group(2)
        
        # Split multiple ft artists
        featured_artists = re.split(r'\s*[,&\+]\s*|\s+and\s+|\s+x\s+', featured_part)
        featured_artists = [clean_artist_name(artist) for artist in featured_artists if artist.strip()]
        
        return clean_artist_name(main_artist), featured_artists
    
    return clean_artist_name(artist_name), []

def fuzzy_match_score(str1: str, str2: str, algorithm: str = 'ratio') -> int:
    """
    Calculate fuzzy matching score between 2 strings.
    
    Args:
        str1, str2: Strings to compare
        algorithm: Matching algorithm ('ratio', 'partial_ratio', 'token_sort_ratio')
        
    Returns:
        Similarity score (0-100)
    """
    if not str1 or not str2:
        return 0
    
    if algorithm == 'partial_ratio':
        return fuzz.partial_ratio(str1, str2)
    elif algorithm == 'token_sort_ratio':
        return fuzz.token_sort_ratio(str1, str2)
    else:
        return fuzz.ratio(str1, str2)

def create_matching_variants(artist_name: str, track_name: str) -> List[Tuple[str, str]]:
    """
    Create multiple variants for fuzzy matching to improve match rates.
    
    Args:
        artist_name: Clean artist name
        track_name: Clean track name
        
    Returns:
        List of (artist, track) variants for matching
    """
    variants = []
    
    # Original cleaned versions
    variants.append((artist_name, track_name))
    
    # Without common words
    artist_no_common = re.sub(r'\b(the|and|of|in|on|at|to|for|with|by)\b', '', artist_name).strip()
    track_no_common = re.sub(r'\b(the|and|of|in|on|at|to|for|with|by)\b', '', track_name).strip()
    
    if artist_no_common != artist_name or track_no_common != track_name:
        variants.append((artist_no_common, track_no_common))
    
    # Abbreviated versions for long names
    if len(artist_name) > 20:
        words = artist_name.split()
        if len(words) > 1:
            abbreviated = ' '.join(word[0] for word in words if word)
            variants.append((abbreviated, track_name))
    
    return variants

def is_likely_match(artist1: str, track1: str, artist2: str, track2: str, 
                   threshold: int = 85, strict_artist: bool = True) -> bool:
    """
    Determine if two artist/track combinations are likely the same song.
    
    Args:
        artist1, track1: First song details
        artist2, track2: Second song details  
        threshold: Minimum similarity score for match
        strict_artist: Require high artist similarity
        
    Returns:
        True if likely match
    """
    # Clean inputs
    artist1, artist2 = clean_artist_name(artist1), clean_artist_name(artist2)
    track1, track2 = clean_track_name(track1), clean_track_name(track2)
    
    if not all([artist1, track1, artist2, track2]):
        return False
    
    artist_score = fuzzy_match_score(artist1, artist2, 'token_sort_ratio')
    track_score = fuzzy_match_score(track1, track2, 'token_sort_ratio')
    
    # Strict artist matching for high confidence
    if strict_artist:
        return artist_score >= 90 and track_score >= threshold
    
    # More lenient matching
    combined_score = (artist_score * 0.6) + (track_score * 0.4)
    return combined_score >= threshold and artist_score >= 70 and track_score >= 70

# Example use and testing
if __name__ == "__main__":
    test_cases = [
        ("The Weeknd feat. Daft Punk", "I Feel It Coming (Explicit Version)"),
        ("Post Malone ft. 21 Savage", "rockstar - Radio Edit"),
        ("Beyoncé", "Crazy In Love (Featuring Jay-Z)"),
        ("ARTIST NAME", "Track Name [Official Music Video]")
    ]
    
    print("Testing normalization:")
    for artist, track in test_cases:
        clean_artist = clean_artist_name(artist)
        clean_track = clean_track_name(track)
        main_artist, featured = extract_featured_artists(artist)
        
        print(f"Original: {artist} - {track}")
        print(f"Cleaned:  {clean_artist} - {clean_track}")
        print(f"Main: {main_artist}, Featured: {featured}")
        print("-" * 50)