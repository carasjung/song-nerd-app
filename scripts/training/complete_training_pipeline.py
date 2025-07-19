# complete_training_pipeline.py
import os
import pandas as pd
from demographics_model_adapted import DemographicsPredictor, train_demographics_model
from robust_platform_model import RobustPlatformRecommender, train_with_current_data
from similar_artists_adapted import SimilarArtistFinder, build_similar_artists_database

def setup_directories():
    """Create necessary directories"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("Directories created!")

def run_complete_training():
    """Run complete model training pipeline with robust platform model"""
    print("Starting complete model training pipeline...")
    
    # Setup
    setup_directories()
    
    # Step 1: Integrate datasets
    print("\n" + "="*50)
    print("STEP 1: Data Integration")
    print("="*50)
    exec(open('data_integration.py').read())
    
    # Step 2: Train demographics model
    print("\n" + "="*50)
    print("STEP 2: Training Demographics Model")
    print("="*50)
    demographics_model = train_demographics_model()
    
    # Step 3: Train robust platform model
    print("\n" + "="*50)
    print("STEP 3: Training Robust Platform Model")
    print("="*50)
    platform_model = train_with_current_data()
    
    # Step 4: Build similar artists database
    print("\n" + "="*50)
    print("STEP 4: Building Similar Artists Database")
    print("="*50)
    similar_artists_model = build_similar_artists_database()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("Models saved in ./models/ directory:")
    print("- demographics_predictor.pkl")
    print("- robust_platform_recommender.pkl")
    print("- similar_artists.pkl")
    
    return demographics_model, platform_model, similar_artists_model

# Quick test function
def test_models():
    """Test the models with sample data"""
    
    # Load models
    demographics_model = DemographicsPredictor().load_model('models/demographics_predictor.pkl')
    platform_model = RobustPlatformRecommender().load_model('models/robust_platform_recommender.pkl')
    similar_artists_model = SimilarArtistFinder().load_model('models/similar_artists.pkl')
    
    # Sample test data (upbeat pop song)
    test_features = pd.DataFrame([{
        'danceability': 0.8,
        'energy': 0.7,
        'valence': 0.6,
        'acousticness': 0.1,
        'instrumentalness': 0.02,
        'liveness': 0.15,
        'speechiness': 0.05,
        'audio_appeal': 75,
        'normalized_popularity': 0.5,
        'genre_clean': 'pop'
    }])
    
    print("\n" + "="*50)
    print("TESTING MODELS")
    print("="*50)
    
    # Test demographics
    demo_results = demographics_model.predict(test_features)
    print(f"Demographics: {demo_results['primary_age_group']} in {demo_results['primary_region']}")
    print(f"Platform preference: {demo_results['preferred_platform']}")
    
    # Test robust platform model
    platform_results = platform_model.predict(test_features)
    print(f"\nPlatform Recommendations:")
    for rec in platform_results['ranked_recommendations']:
        print(f"- {rec['platform'].title()}: {rec['score']:.0f}/100 "
              f"(Success: {rec['success_probability']:.1%}, {rec['confidence']})")
    
    # Test similar artists
    similar_results = similar_artists_model.find_similar_artists(test_features.iloc[0].to_dict(), top_k=3)
    print(f"\nSimilar Artists:")
    for artist in similar_results['similar_artists']:
        print(f"- {artist['artist_name']} (similarity: {artist['similarity_score']:.3f})")
    
    return demo_results, platform_results, similar_results

if __name__ == "__main__":
    # Run training
    models = run_complete_training()
    
    # Test the models
    print("\n" + "="*60)
    print("RUNNING TESTS...")
    print("="*60)
    test_results = test_models()