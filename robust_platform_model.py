# robust_platform_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

class RobustPlatformRecommender:
    def __init__(self):
        # Use a two-stage approach: classification + regression
        self.success_models = {}  # Binary classifiers for each platform
        self.score_models = {}    # Regressors for scoring successful tracks
        self.scaler = StandardScaler()
        self.platform_names = ['spotify', 'tiktok', 'youtube']
        self.label_encoders = {}
        
    def engineer_platform_features(self, df):
        """Create robust platform-specific features"""
        features = df.copy()
        
        # Core platform affinity scores
        features['spotify_fit'] = (
            features['valence'] * 0.25 +           # Mood-based playlists
            features['energy'] * 0.2 +             # Energy-based playlists  
            features['acousticness'] * 0.2 +       # Acoustic playlists
            (1 - features['instrumentalness']) * 0.2 +  # Vocal content
            (features['audio_appeal'] / 100) * 0.15     # Quality factor
        )
        
        features['tiktok_fit'] = (
            features['danceability'] * 0.4 +       # Dance content
            features['energy'] * 0.3 +             # High energy
            (features['speechiness'] > 0.1).astype(float) * 0.15 +  # Some vocal/rap
            features['valence'] * 0.15             # Positive mood
        )
        
        features['youtube_fit'] = (
            features['energy'] * 0.3 +             # Engaging content
            (1 - features['instrumentalness']) * 0.25 +  # Vocal content
            features['valence'] * 0.2 +            # Positive/engaging
            (features['liveness'] > 0.2).astype(float) * 0.15 +  # Live appeal
            (features['audio_appeal'] / 100) * 0.1      # Production quality
        )
        
        # Genre-platform compatibility
        genre_compatibility = {
            'pop': {'spotify': 0.9, 'tiktok': 0.8, 'youtube': 0.8},
            'hip hop': {'spotify': 0.7, 'tiktok': 1.0, 'youtube': 0.7},
            'electronic': {'spotify': 0.8, 'tiktok': 0.9, 'youtube': 0.6},
            'rock': {'spotify': 0.8, 'tiktok': 0.4, 'youtube': 0.9},
            'country': {'spotify': 0.9, 'tiktok': 0.5, 'youtube': 0.8},
            'r&b': {'spotify': 0.9, 'tiktok': 0.7, 'youtube': 0.7},
            'indie': {'spotify': 0.9, 'tiktok': 0.5, 'youtube': 0.8}
        }
        
        # Apply genre compatibility
        for platform in self.platform_names:
            col_name = f'genre_{platform}_fit'
            features[col_name] = features['genre_clean'].map(
                lambda x: genre_compatibility.get(
                    x.lower() if isinstance(x, str) else 'pop', 
                    {'spotify': 0.7, 'tiktok': 0.7, 'youtube': 0.7}
                )[platform]
            )
        
        # Viral potential indicators
        features['hook_strength'] = (
            features['danceability'] * features['energy'] * 
            (1 - features['instrumentalness'])
        )
        
        features['mood_appeal'] = np.where(
            features['valence'] > 0.6, 
            features['valence'] * features['energy'],
            features['valence'] * 0.5  # Penalty for sad songs
        )
        
        # Platform-specific thresholds
        features['tempo_tiktok_sweet_spot'] = np.where(
            (features.get('tempo', features['energy'] * 140) >= 100) & 
            (features.get('tempo', features['energy'] * 140) <= 140), 
            1.0, 0.5
        )
        
        return features
    
    def prepare_training_data(self, df, use_synthetic=True):
        """Prepare training data with option to use synthetic scores"""
        
        # Engineer features
        df_featured = self.engineer_platform_features(df)
        
        # Select features
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
        
        platform_features = [
            'spotify_fit', 'tiktok_fit', 'youtube_fit',
            'genre_spotify_fit', 'genre_tiktok_fit', 'genre_youtube_fit',
            'hook_strength', 'mood_appeal', 'tempo_tiktok_sweet_spot'
        ]
        
        quality_features = ['audio_appeal']
        
        # Encode genre
        if 'genre_clean' in df_featured.columns:
            from sklearn.preprocessing import LabelEncoder
            if 'genre_encoder' not in self.label_encoders:
                self.label_encoders['genre_encoder'] = LabelEncoder()
            df_featured['genre_encoded'] = self.label_encoders['genre_encoder'].fit_transform(
                df_featured['genre_clean'].fillna('unknown')
            )
            platform_features.append('genre_encoded')
        
        # Combine features
        feature_cols = audio_features + platform_features + quality_features
        X = df_featured[feature_cols].copy()
        
        # Handle missing values
        for col in feature_cols:
            if col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
        
        # Prepare targets - use synthetic data if available and sparse real data
        target_data = {}
        
        for platform in self.platform_names:
            synthetic_col = f'{platform}_synthetic'
            combined_col = f'{platform}_combined'
            
            if use_synthetic and synthetic_col in df_featured.columns:
                # Use synthetic data
                scores = df_featured[synthetic_col].fillna(0)
                print(f"Using synthetic data for {platform}")
            elif combined_col in df_featured.columns:
                # Use real data
                scores = df_featured[combined_col].fillna(0)
                print(f"Using real data for {platform}")
            else:
                # Create default scores
                scores = pd.Series(np.zeros(len(df_featured)))
                print(f"No data found for {platform}, using zeros")
            
            # Create binary success labels (platform-specific thresholds)
            thresholds = {'spotify': 20, 'tiktok': 5, 'youtube': 10}
            success_labels = (scores > thresholds[platform]).astype(int)
            
            target_data[platform] = {
                'scores': scores,
                'success': success_labels,
                'threshold': thresholds[platform]
            }
        
        # Filter out rows where all platforms have zero scores (if using real data)
        if not use_synthetic:
            all_scores = sum(target_data[p]['scores'] for p in self.platform_names)
            valid_mask = all_scores > 0
            X = X[valid_mask]
            for platform in self.platform_names:
                target_data[platform]['scores'] = target_data[platform]['scores'][valid_mask]
                target_data[platform]['success'] = target_data[platform]['success'][valid_mask]
        
        print(f"Final training data shape: {X.shape}")
        for platform in self.platform_names:
            success_rate = target_data[platform]['success'].mean()
            mean_score = target_data[platform]['scores'].mean()
            print(f"{platform}: {success_rate:.1%} success rate, mean score {mean_score:.1f}")
        
        return X, target_data
    
    def train(self, training_data, use_synthetic=True):
        """Train robust two-stage platform models"""
        
        X, target_data = self.prepare_training_data(training_data, use_synthetic)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nTraining models on {X_scaled.shape[0]} samples...")
        
        # Train models for each platform
        for platform in self.platform_names:
            print(f"\nTraining {platform} models...")
            
            scores = target_data[platform]['scores']
            success_labels = target_data[platform]['success']
            
            # Binary classification (will this song be successful on this platform?)
            self.success_models[platform] = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            self.success_models[platform].fit(X_scaled, success_labels)
            
            # Regression for successful songs only
            successful_mask = success_labels == 1
            if successful_mask.sum() > 10:  # Need at least 10 successful examples
                X_successful = X_scaled[successful_mask]
                scores_successful = scores[successful_mask]
                
                self.score_models[platform] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
                self.score_models[platform].fit(X_successful, scores_successful)
                
                print(f"  Success classifier trained on {len(success_labels)} samples")
                print(f"  Score regressor trained on {len(scores_successful)} successful samples")
            else:
                print(f"  Warning: Only {successful_mask.sum()} successful examples for {platform}")
                print(f"  Using simple scoring based on success probability")
                self.score_models[platform] = None
        
        # Print feature importance for first platform
        if self.platform_names and self.platform_names[0] in self.success_models:
            feature_importance = self.success_models[self.platform_names[0]].feature_importances_
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 features for platform success prediction:")
            print(importance_df.head(10))
        
        return self
    
    def predict(self, audio_features):
        """Predict platform performance using two-stage approach"""
        
        # Engineer features
        df_featured = self.engineer_platform_features(audio_features)
        
        # Prepare features (same as training)
        audio_cols = ['danceability', 'energy', 'valence', 'acousticness', 
                     'instrumentalness', 'liveness', 'speechiness']
        
        platform_cols = [
            'spotify_fit', 'tiktok_fit', 'youtube_fit',
            'genre_spotify_fit', 'genre_tiktok_fit', 'genre_youtube_fit',
            'hook_strength', 'mood_appeal', 'tempo_tiktok_sweet_spot'
        ]
        
        quality_cols = ['audio_appeal']
        
        if 'genre_encoded' in df_featured.columns:
            platform_cols.append('genre_encoded')
        
        feature_cols = audio_cols + platform_cols + quality_cols
        X = df_featured[feature_cols].copy()
        
        # Handle missing values
        for col in feature_cols:
            if col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].median() if not X[col].empty else 0)
                else:
                    X[col] = X[col].fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict for each platform
        platform_results = {}
        
        for platform in self.platform_names:
            if platform in self.success_models:
                # Predict success probability
                success_prob = self.success_models[platform].predict_proba(X_scaled)[0][1]
                
                # Predict score if likely to be successful
                if self.score_models[platform] is not None and success_prob > 0.3:
                    predicted_score = self.score_models[platform].predict(X_scaled)[0]
                    # Weight by success probability
                    final_score = predicted_score * success_prob
                else:
                    # Use simple scoring based on success probability and platform fit
                    platform_fit = df_featured[f'{platform}_fit'].iloc[0]
                    final_score = success_prob * platform_fit * 100
                
                # Ensure reasonable bounds
                final_score = max(0, min(100, final_score))
                
                platform_results[platform] = {
                    'score': float(final_score),
                    'success_probability': float(success_prob),
                    'confidence': self._calculate_confidence(final_score, success_prob),
                    'recommendation': self._generate_recommendation(platform, final_score, success_prob)
                }
        
        # Rank platforms
        sorted_platforms = sorted(
            platform_results.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        recommendations = {
            'platform_scores': platform_results,
            'ranked_recommendations': [
                {
                    'platform': platform,
                    'score': data['score'],
                    'success_probability': data['success_probability'],
                    'confidence': data['confidence'],
                    'recommendation': data['recommendation']
                }
                for platform, data in sorted_platforms
            ],
            'top_platform': sorted_platforms[0][0] if sorted_platforms else 'spotify',
            'top_score': sorted_platforms[0][1]['score'] if sorted_platforms else 0
        }
        
        return recommendations
    
    def _calculate_confidence(self, score, success_prob):
        """Calculate confidence based on score and success probability"""
        if success_prob > 0.7 and score > 60:
            return 'high'
        elif success_prob > 0.4 and score > 30:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendation(self, platform, score, success_prob):
        """Generate platform-specific recommendations"""
        recommendations = {
            'spotify': {
                'high': "Excellent Spotify potential! Target editorial playlists and Release Radar.",
                'medium': "Good Spotify fit. Focus on algorithmic playlists and genre-specific lists.",
                'low': "Limited Spotify appeal. Consider acoustic versions or different positioning."
            },
            'tiktok': {
                'high': "High TikTok viral potential! Create dance challenges and trend content.",
                'medium': "Moderate TikTok appeal. Focus on specific hooks or storytelling.",
                'low': "Low TikTok fit. Consider creative adaptations or focus on other platforms."
            },
            'youtube': {
                'high': "Great YouTube potential! Invest in high-quality music videos.",
                'medium': "Good YouTube fit. Consider lyric videos or live performance content.",
                'low': "Limited YouTube appeal. Focus on other platforms or try different content types."
            }
        }
        
        confidence = self._calculate_confidence(score, success_prob)
        return recommendations.get(platform, {}).get(confidence, f"Score: {score:.0f}, Success Probability: {success_prob:.1%}")
    
    def evaluate_model(self, test_data, use_synthetic=True):
        """Evaluate the two-stage model performance"""
        
        X_test, target_data = self.prepare_training_data(test_data, use_synthetic)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for platform in self.platform_names:
            if platform in self.success_models:
                # Evaluate success prediction
                true_success = target_data[platform]['success']
                pred_success = self.success_models[platform].predict(X_test_scaled)
                success_accuracy = accuracy_score(true_success, pred_success)
                
                # Evaluate score prediction for successful cases
                if self.score_models[platform] is not None:
                    successful_mask = true_success == 1
                    if successful_mask.sum() > 0:
                        true_scores = target_data[platform]['scores'][successful_mask]
                        pred_scores = self.score_models[platform].predict(X_test_scaled[successful_mask])
                        score_mae = mean_absolute_error(true_scores, pred_scores)
                        score_r2 = r2_score(true_scores, pred_scores)
                    else:
                        score_mae = np.nan
                        score_r2 = np.nan
                else:
                    score_mae = np.nan
                    score_r2 = np.nan
                
                results[platform] = {
                    'success_accuracy': success_accuracy,
                    'score_mae': score_mae,
                    'score_r2': score_r2,
                    'success_rate': true_success.mean(),
                    'n_successful': true_success.sum()
                }
                
                print(f"{platform.upper()}:")
                print(f"  Success prediction accuracy: {success_accuracy:.3f}")
                print(f"  Success rate in data: {true_success.mean():.1%}")
                print(f"  Number of successful examples: {true_success.sum()}")
                if not np.isnan(score_mae):
                    print(f"  Score prediction MAE: {score_mae:.2f}")
                    print(f"  Score prediction R²: {score_r2:.3f}")
                else:
                    print(f"  Score prediction: Not enough successful examples")
                print()
        
        return results
    
    def save_model(self, filepath):
        """Save the robust two-stage model"""
        model_data = {
            'success_models': self.success_models,
            'score_models': self.score_models,
            'scaler': self.scaler,
            'platform_names': self.platform_names,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load the robust two-stage model"""
        model_data = joblib.load(filepath)
        self.success_models = model_data['success_models']
        self.score_models = model_data['score_models']
        self.scaler = model_data['scaler']
        self.platform_names = model_data['platform_names']
        self.label_encoders = model_data['label_encoders']
        return self

# Training script for robust platform model
def train_robust_platform_model():
    """Train the robust platform model with data analysis"""
    
    print("="*60)
    print("ROBUST PLATFORM MODEL TRAINING")
    print("="*60)
    
    # Run data analysis to understand the platform data first
    print("Step 1: Analyzing platform data...")
    try:
        exec(open('platform_data_analysis.py').read())
    except FileNotFoundError:
        print("Warning: platform_data_analysis.py not found, proceeding with available data")
    
    # Try to load enhanced data first, fall back to integrated data
    training_files = [
        'enhanced_platform_training.csv',
        'cleaned_platform_training.csv', 
        'integrated_platform_training.csv'
    ]
    
    training_data = None
    use_synthetic = False
    
    for filename in training_files:
        try:
            training_data = pd.read_csv(filename)
            print(f"Step 2: Loaded training data from {filename}")
            if 'synthetic' in filename or 'enhanced' in filename:
                use_synthetic = True
                print("  Using enhanced/synthetic data for better training")
            break
        except FileNotFoundError:
            continue
    
    if training_data is None:
        print("Error: No training data found. Please run data integration first.")
        return None
    
    print(f"Training data shape: {training_data.shape}")
    
    # Check what platform data is available
    platform_cols = [col for col in training_data.columns if any(p in col for p in ['spotify', 'tiktok', 'youtube'])]
    print(f"Available platform columns: {platform_cols}")
    
    # Split data
    train_df, test_df = train_test_split(training_data, test_size=0.2, random_state=42)
    
    # Initialize and train robust model
    print(f"\nStep 3: Training robust two-stage platform model...")
    platform_model = RobustPlatformRecommender()
    platform_model.train(train_df, use_synthetic=use_synthetic)
    
    # Evaluate model
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    platform_model.evaluate_model(test_df, use_synthetic=use_synthetic)
    
    # Save model
    platform_model.save_model('models/robust_platform_recommender.pkl')
    print("Robust platform model saved!")
    
    # Test with sample data
    print("\n" + "="*60)
    print("SAMPLE PREDICTION TEST")
    print("="*60)
    
    sample_features = pd.DataFrame([{
        'danceability': 0.8,
        'energy': 0.75,
        'valence': 0.7,
        'acousticness': 0.1,
        'instrumentalness': 0.02,
        'liveness': 0.12,
        'speechiness': 0.05,
        'audio_appeal': 80,
        'genre_clean': 'pop'
    }])
    
    sample_results = platform_model.predict(sample_features)
    print("Sample song (upbeat pop):")
    for rec in sample_results['ranked_recommendations']:
        print(f"- {rec['platform'].title()}: {rec['score']:.0f}/100 "
              f"(Success: {rec['success_probability']:.1%}, {rec['confidence']})")
    
    return platform_model

# Alternative training function that works with your current files
def train_with_current_data():
    """Train using your current integrated_platform_training.csv with robustness improvements"""
    
    print("Training with current data")
    
    # Load current data
    try:
        training_data = pd.read_csv('integrated_platform_training.csv')
        print(f"Loaded training data: {training_data.shape}")
    except FileNotFoundError:
        print("Error: integrated_platform_training.csv not found")
        print("Please run data_integration.py first")
        return None
    
    # Analyze the data quickly
    platform_cols = ['spotify_combined', 'tiktok_combined', 'youtube_combined']
    for col in platform_cols:
        if col in training_data.columns:
            non_zero = (training_data[col] > 0).sum()
            total = len(training_data[col].dropna())
            print(f"{col}: {non_zero}/{total} non-zero ({non_zero/total*100:.1f}%)")
    
    # Create simple synthetic data to augment sparse real data
    print("\nCreating synthetic data to improve training...")
    
    # Create rule-based platform scores for each song
    enhanced_data = training_data.copy()
    
    for platform in ['spotify', 'tiktok', 'youtube']:
        synthetic_col = f'{platform}_synthetic'
        
        if platform == 'spotify':
            # Spotify likes diverse music, quality matters
            enhanced_data[synthetic_col] = (
                enhanced_data['valence'] * 30 +
                enhanced_data['energy'] * 20 +
                enhanced_data['audio_appeal'] * 0.3 +
                np.random.normal(0, 10, len(enhanced_data))
            ).clip(0, 100)
            
        elif platform == 'tiktok':
            # TikTok likes danceable, high-energy music
            enhanced_data[synthetic_col] = (
                enhanced_data['danceability'] * 40 +
                enhanced_data['energy'] * 30 +
                enhanced_data['valence'] * 20 +
                np.random.normal(0, 15, len(enhanced_data))
            ).clip(0, 100)
            
        else:  # youtube
            # YouTube likes engaging, vocal content
            enhanced_data[synthetic_col] = (
                enhanced_data['energy'] * 25 +
                (1 - enhanced_data['instrumentalness']) * 35 +
                enhanced_data['valence'] * 20 +
                np.random.normal(0, 12, len(enhanced_data))
            ).clip(0, 100)
        
        # Show improvement
        original_nonzero = (training_data.get(f'{platform}_combined', pd.Series([0])) > 0).sum()
        synthetic_nonzero = (enhanced_data[synthetic_col] > 20).sum()
        print(f"{platform}: {original_nonzero} -> {synthetic_nonzero} songs above threshold")
    
    # Split and train
    train_df, test_df = train_test_split(enhanced_data, test_size=0.2, random_state=42)
    
    # Train model
    platform_model = RobustPlatformRecommender()
    platform_model.train(train_df, use_synthetic=True)
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    platform_model.evaluate_model(test_df, use_synthetic=True)
    
    # Save
    platform_model.save_model('models/robust_platform_recommender.pkl')
    print("Robust platform model saved!")
    
    return platform_model

if __name__ == "__main__":
    # Try robust training first, fall back to current data approach
    try:
        model = train_robust_platform_model()
    except Exception as e:
        print(f"Robust training failed: {e}")
        print("Falling back to current data approach...")
        model = train_with_current_data()