# test_individual_models.py
import pandas as pd
from demographics_model_adapted import DemographicsPredictor
from robust_platform_model import RobustPlatformRecommender
from similar_artists_adapted import SimilarArtistFinder

def test_demographics_model():
    """Test demographics model separately"""
    print("=" * 50)
    print("TESTING DEMOGRAPHICS MODEL")
    print("=" * 50)
    
    try:
        demo_model = DemographicsPredictor()
        demo_model.load_model('models/demographics_predictor.pkl')
        
        # Create test data exactly like the debug script
        test_data = pd.DataFrame([{
            'danceability': 0.85,
            'energy': 0.75,
            'valence': 0.70,
            'acousticness': 0.12,
            'instrumentalness': 0.03,
            'liveness': 0.15,
            'speechiness': 0.06,
            'audio_appeal': 82,
            'normalized_popularity': 0.6,
            'genre_clean': 'pop',
            'spotify': 0,
            'tiktok': 0,
            'youtube': 0
        }])
        
        result = demo_model.predict(test_data)
        print("Demographics prediction successful")
        print(f"   Primary age group: {result['primary_age_group']}")
        print(f"   Primary region: {result['primary_region']}")
        print(f"   Confidence: {result['confidence_scores']['age']:.1%}")
        return True
        
    except Exception as e:
        print(f" Demographics model failed: {e}")
        return False

def test_platform_model():
    """Test platform model separately"""
    print("\n" + "=" * 50)
    print("Testing platform model")
    print("=" * 50)
    
    try:
        platform_model = RobustPlatformRecommender()
        platform_model.load_model('models/robust_platform_recommender.pkl')
        
        test_data = pd.DataFrame([{
            'danceability': 0.85,
            'energy': 0.75,
            'valence': 0.70,
            'acousticness': 0.12,
            'instrumentalness': 0.03,
            'liveness': 0.15,
            'speechiness': 0.06,
            'audio_appeal': 82,
            'genre_clean': 'pop'
        }])
        
        result = platform_model.predict(test_data)
        print(" Platform prediction successful!")
        print(f"   Top platform: {result['top_platform']}")
        print(f"   Top score: {result['top_score']:.0f}/100")
        for rec in result['ranked_recommendations']:
            print(f"   {rec['platform']}: {rec['score']:.0f}/100 ({rec['success_probability']:.1%})")
        return True
        
    except Exception as e:
        print(f" Platform model failed: {e}")
        return False

def test_similar_artists_model():
    """Test similar artists model separately"""
    print("\n" + "=" * 50)
    print("Testing similar artists model")
    print("=" * 50)
    
    try:
        similar_model = SimilarArtistFinder()
        similar_model.load_model('models/similar_artists.pkl')
        
        test_features = {
            'danceability': 0.85,
            'energy': 0.75,
            'valence': 0.70,
            'acousticness': 0.12,
            'instrumentalness': 0.03,
            'liveness': 0.15,
            'speechiness': 0.06
        }
        
        result = similar_model.find_similar_artists(test_features, top_k=3)
        print(" Similar artists prediction successful!")
        for artist in result['similar_artists']:
            print(f"   {artist['artist_name']} (similarity: {artist['similarity_score']:.3f})")
        return True
        
    except Exception as e:
        print(f" Similar artists model failed: {e}")
        return False

def test_integrated_prediction():
    """Test the exact same data through each model step by step"""
    print("\n" + "=" * 50)
    print("Testing integrated step-by-step")
    print("=" * 50)
    
    # Exact same data as in the integrated analyzer
    audio_features = pd.DataFrame([{
        'danceability': 0.85,
        'energy': 0.75,
        'valence': 0.70,
        'acousticness': 0.12,
        'instrumentalness': 0.03,
        'liveness': 0.15,
        'speechiness': 0.06,
        'audio_appeal': 82,
        'normalized_popularity': 0.6,
        'genre_clean': 'pop',
        'spotify': 0,
        'tiktok': 0,
        'youtube': 0
    }])
    
    print(f"Input data shape: {audio_features.shape}")
    print(f"Input columns: {list(audio_features.columns)}")
    
    # Test demographics step by step
    try:
        demo_model = DemographicsPredictor()
        demo_model.load_model('models/demographics_predictor.pkl')
        
        print("\n1. Testing demographics prediction...")
        demographics = demo_model.predict(audio_features)
        print(" Demographics OK")
        
        print("\n2. Testing platform prediction...")
        platform_model = RobustPlatformRecommender()
        platform_model.load_model('models/robust_platform_recommender.pkl')
        platforms = platform_model.predict(audio_features)
        print(" Platform OK")
        
        print("\n3. Testing similar artists...")
        similar_model = SimilarArtistFinder()
        similar_model.load_model('models/similar_artists.pkl')
        input_dict = audio_features.iloc[0].to_dict()
        similar_artists = similar_model.find_similar_artists(input_dict, top_k=3)
        print(" Similar artists OK")
        
        print("\n All models working individually!")
        return True
        
    except Exception as e:
        print(f" Integrated test failed at step: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all individual tests"""
    print("Testing individual models")
    
    demo_ok = test_demographics_model()
    platform_ok = test_platform_model()
    similar_ok = test_similar_artists_model()
    integrated_ok = test_integrated_prediction()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Demographics Model: {'OK' if demo_ok else 'Failed'}")
    print(f"Platform Model: {'OK' if platform_ok else 'Failed'}")
    print(f"Similar Artists Model: {'OK' if similar_ok else 'Failed'}")
    print(f"Integrated Test: {'OK' if integrated_ok else 'Failed'}")
    
    if all([demo_ok, platform_ok, similar_ok, integrated_ok]):
        print("\n All models are working! The issue is in the integrated analyzer.")
        print(" Try running the integrated analyzer again.")
    else:
        print("\n Some models have issues that need to be fixed first.")

if __name__ == "__main__":
    main()