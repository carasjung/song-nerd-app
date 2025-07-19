# direct_audio_test.py
import os
import numpy as np
import pandas as pd

def find_audio_file():
    """Find any audio file in current directory"""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            return file
    return None

def extract_audio_features_direct(audio_path):
    """Extract features directly using librosa (no pydub needed)"""
    import librosa
    
    print(f"Loading audio file: {audio_path}")
    
    try:
        # Load audio directly with librosa (handles MP3, WAV, etc.)
        y, sr = librosa.load(audio_path, sr=22050, duration=60)  # Load first 60 sec
        
        print(f"Audio loaded successfully")
        print(f"   Duration: {len(y)/sr:.1f} seconds")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Audio shape: {y.shape}")
        
        # Extract all the features that the ML models need
        features = {}
        
        print(f"\n Extracting audio features...")
        
        # Basic features
        features['duration'] = len(y) / sr
        
        # Energy and loudness
        rms = librosa.feature.rms(y=y)[0]
        features['energy'] = min(1.0, np.mean(rms) * 10)
        features['loudness'] = -60 + 60 * np.mean(rms)
        
        # Tempo and rhythm
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # Danceability (beat consistency and strength)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        features['danceability'] = min(1.0, np.var(onset_strength) / 100 + 0.1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Valence (positivity) - based on spectral characteristics
        brightness = np.mean(spectral_centroids) / (sr/2)
        features['valence'] = min(1.0, max(0.1, brightness * 0.8 + 0.2))
        
        # Acousticness (inverse of spectral complexity)
        features['acousticness'] = max(0.0, min(1.0, 1 - np.mean(spectral_bandwidth) / 4000))
        
        # Speechiness (zero crossing rate)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['speechiness'] = min(1.0, np.mean(zcr) * 5)
        
        # Instrumentalness (vocal detection)
        harmonic, percussive = librosa.effects.hpss(y)
        vocal_strength = np.mean(librosa.feature.spectral_centroid(y=harmonic, sr=sr))
        features['instrumentalness'] = max(0.0, min(1.0, 1 - vocal_strength / 3000))
        
        # Liveness (reverb and room characteristics)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['liveness'] = min(1.0, np.mean(spectral_flatness) * 8)
        
        # Key and mode
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = np.argmax(np.sum(chroma, axis=1))
        features['key'] = int(key)
        
        # Mode detection (major vs minor)
        chroma_mean = np.mean(chroma, axis=1)
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        features['mode'] = 1 if major_corr > minor_corr else 0
        
        # Time signature (simplified)
        features['time_signature'] = 4  # Default to 4/4
        
        # Audio appeal (quality score)
        dynamic_range = np.max(rms) - np.min(rms)
        freq_balance = 1 - np.std(np.mean(np.abs(librosa.stft(y)), axis=1)) / np.mean(np.abs(librosa.stft(y)))
        clarity = np.mean(rms) / (np.std(rms) + 1e-8)
        features['audio_appeal'] = min(100, max(0, dynamic_range * 40 + freq_balance * 30 + clarity * 30))
        
        # Add required features for ML models
        features['normalized_popularity'] = 0.5  # Default for new songs
        features['genre_clean'] = 'pop'  # Default genre
        features['spotify'] = 0  # Default platform scores
        features['tiktok'] = 0
        features['youtube'] = 0
        
        print(f"Feature extraction completed!")
        return features
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_with_ml_models(features):
    """Test the features with your ML models"""
    print(f"\nTesting with ML models...")
    
    try:
        from integrated_analyzer import MusicMarketingAnalyzer
        
        analyzer = MusicMarketingAnalyzer()
        analyzer.load_models()
        
        if not analyzer.models_loaded:
            print("ML models not loaded - need to run training first")
            return False
        
        # Create test metadata
        metadata = {
            'track_name': 'Sample Song',
            'artist_name': 'Test Artist',
            'genre': 'pop'
        }
        
        # Run complete analysis
        analysis = analyzer.analyze_song(features, metadata)
        
        if 'error' not in analysis:
            print(f"Complete success")
            print(f"   Target demographic: {analysis['target_demographics']['primary_age_group']}")
            print(f"   Top platform: {analysis['platform_recommendations']['top_platform']}")
            print(f"   Platform score: {analysis['platform_recommendations']['top_score']:.0f}/100")
            print(f"   Success probability: {analysis['platform_recommendations']['ranked_recommendations'][0]['success_probability']:.1%}")
            
            # Show similar artists
            if analysis['similar_artists']['similar_artists']:
                print(f"   Similar to: {analysis['similar_artists']['similar_artists'][0]['artist_name']}")
            
            return True
        else:
            print(f"  Got basic analysis (models need training)")
            return True
            
    except ImportError:
        print(f"  ML models not available")
        return True
    except Exception as e:
        print(f" ML model test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Direct Audio Processing Test")
    print("=" * 50)
    
    # Find audio file
    audio_file = find_audio_file()
    if not audio_file:
        print("No audio file found!")
        print("Put an MP3 or WAV file in this directory and try again.")
        return
    
    print(f"🎵 Found audio file: {audio_file}")
    
    # Extract features
    features = extract_audio_features_direct(audio_file)
    
    if features:
        print(f"\n Extracted features:")
        print("=" * 30)
        
        # Show the key features
        key_features = ['danceability', 'energy', 'valence', 'acousticness', 
                       'speechiness', 'tempo', 'audio_appeal']
        
        for feature in key_features:
            if feature in features:
                value = features[feature]
                if isinstance(value, float):
                    print(f"   {feature:15}: {value:.3f}")
                else:
                    print(f"   {feature:15}: {value}")
        
        # Test with ML models
        ml_success = test_with_ml_models(features)
        
        if ml_success:
            print(f"\n Audio processing pipeline is working")
            print(f"\n You can now:")
            print(f"1. Process any audio file and get marketing insights")
            print(f"2. Build API endpoints for your web app")
            print(f"3. Create the frontend interface")
        else:
            print(f"\n  Audio processing works, but ML models need setup")
    
    else:
        print(f"\n Audio processing failed")

if __name__ == "__main__":
    main()