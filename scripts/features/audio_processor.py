# audio_processor.py

import librosa
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def extract_features(self, audio_path):
        """Extract all audio features needed by your ML models"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            features = {}
            
            features.update(self._extract_basic_features(y, sr))
            
            features.update(self._extract_spotify_features(y, sr))
            
            features.update(self._extract_quality_features(y, sr))
            
            return features
            
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")
    
    def _extract_basic_features(self, y, sr):
        """Extract basic audio characteristics"""
        features = {}
        
        features['duration'] = len(y) / sr
        
        # RMS (loudness proxy)
        rms = librosa.feature.rms(y=y)[0]
        features['loudness'] = -60 + 60 * np.mean(rms)  # Convert to dB-like scale
        
        # Zero crossing rate (speechiness proxy)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate'] = np.mean(zcr)
        
        return features
    
    def _extract_spotify_features(self, y, sr):
        """Extract Spotify-like audio features"""
        features = {}
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # Key and mode
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = np.argmax(np.sum(chroma, axis=1))
        features['key'] = int(key)
        
        # Mode (major=1, minor=0) - simplified approach
        major_profile = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        minor_profile = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
        
        chroma_mean = np.mean(chroma, axis=1)
        major_correlation = np.corrcoef(chroma_mean, major_profile)[0, 1]
        minor_correlation = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        
        features['mode'] = 1 if major_correlation > minor_correlation else 0
        
        # Time signature (simplified)
        features['time_signature'] = 4  # Default to 4/4
        
        # Danceability (based on beat strength and regularity)
        beat_strength = librosa.feature.tempogram(y=y, sr=sr)
        features['danceability'] = min(1.0, np.mean(beat_strength) * 2)
        
        # Energy (based on spectral characteristics)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rms_energy = librosa.feature.rms(y=y)[0]
        features['energy'] = min(1.0, np.mean(rms_energy) * 10)
        
        # Valence (positivity - simplified approach using spectral features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Combine features that correlate with perceived positivity
        brightness = np.mean(spectral_centroids) / (sr/2)  # Normalized brightness
        rhythm_strength = np.std(beats) if len(beats) > 1 else 0
        
        features['valence'] = min(1.0, max(0.0, brightness + 0.3))
        
        # Acousticness (inverse of spectral energy in higher frequencies)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['acousticness'] = max(0.0, min(1.0, 1 - np.mean(spectral_bandwidth) / 4000))
        
        # Instrumentalness (based on vocal detection)
        # Look for harmonic content that suggests vocals
        harmonic, percussive = librosa.effects.hpss(y)
        vocal_likelihood = np.mean(librosa.feature.spectral_centroid(y=harmonic, sr=sr))
        features['instrumentalness'] = max(0.0, min(1.0, 1 - vocal_likelihood / 3000))
        
        # Liveness (based on spectral characteristics and reverb)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['liveness'] = min(1.0, np.mean(spectral_flatness) * 10)
        
        # Speechiness (based on zero crossing rate and spectral characteristics)
        features['speechiness'] = min(1.0, features['zero_crossing_rate'] * 5)
        
        return features
    
    def _extract_quality_features(self, y, sr):
        """Extract audio quality and appeal metrics"""
        features = {}
        
        # Audio appeal (composite quality score)
        # Based on dynamic range, frequency balance, and clarity
        
        # Dynamic range
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range = np.max(rms) - np.min(rms)
        
        # Frequency balance (how well distributed energy is across spectrum)
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        freq_balance = 1 - np.std(np.mean(magnitude, axis=1)) / np.mean(magnitude)
        
        # Clarity (inverse of noise)
        harmonic, percussive = librosa.effects.hpss(y)
        clarity = np.mean(librosa.feature.rms(y=harmonic)) / (np.mean(librosa.feature.rms(y=y)) + 1e-8)
        
        # Combine into appeal score (0-100)
        features['audio_appeal'] = min(100, max(0, 
            (dynamic_range * 30 + freq_balance * 40 + clarity * 30)))
        
        return features
    
    def process_file(self, audio_path, metadata=None):
        """Main processing function that returns ML-ready features"""
        features = self.extract_features(audio_path)
        
        # Add metadata if provided
        if metadata:
            features.update(metadata)
        
        # Add default values for features expected by ML models
        if 'genre_clean' not in features:
            features['genre_clean'] = 'unknown'
        
        if 'normalized_popularity' not in features:
            features['normalized_popularity'] = 0.5
        
        # Ensure all required features are present
        required_features = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 
            'audio_appeal', 'normalized_popularity', 'genre_clean'
        ]
        
        for feature in required_features:
            if feature not in features:
                features[feature] = 0.5  # Default value
        
        return features

# Example use and testing
def test_audio_processor():
    """Test the audio processor with a sample file"""
    processor = AudioFeatureExtractor()
    
    # Test with a sample file (you'll need to provide an actual audio file)
    try:
        features = processor.process_file('sample_song.mp3', 
                                        metadata={'genre_clean': 'pop'})
        
        print("Extracted Features:")
        for key, value in features.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
        return features
    except Exception as e:
        print(f"Test failed: {e}")
        return None

if __name__ == "__main__":
    test_audio_processor()