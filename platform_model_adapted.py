# platform_model_adapted.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

class PlatformRecommender:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.platform_names = ['spotify', 'tiktok', 'youtube']
        self.label_encoders = {}
        
    def engineer_platform_specific_features(self, df):
        """Create more sophisticated platform-specific features"""
        features = df.copy()
        
        # TikTok features (15-60 second engagement, viral potential)
        features['tiktok_viral_score'] = (
            features['danceability'] * 0.35 +  # Dance trends
            features['energy'] * 0.25 +        # High energy
            (features['speechiness'] > 0.1).astype(int) * 0.15 +  # Some vocal elements
            (features['instrumentalness'] < 0.5).astype(int) * 0.25  # Not purely instrumental
        )
        
        # Spotify features (playlist placement, algorithmic discovery)
        features['spotify_algorithm_score'] = (
            features['valence'] * 0.2 +        # Mood-based playlists
            features['energy'] * 0.2 +         # Energy-based playlists
            features['acousticness'] * 0.15 +  # Acoustic playlists
            (1 - features['instrumentalness']) * 0.25 +  # Vocal songs
            features['audio_appeal'] / 100 * 0.2  # Quality factor
        )
        
        # YouTube features (music videos, longer content)
        features['youtube_engagement_score'] = (
            features['energy'] * 0.3 +         # Engaging content
            (1 - features['instrumentalness']) * 0.3 +  # Vocal content
            features['valence'] * 0.2 +        # Positive content
            (features['liveness'] > 0.2).astype(int) * 0.2  # Live performance appeal
        )
        
        # Genre-platform affinity (simplified)
        genre_platform_affinity = {
            'pop': {'spotify': 1.0, 'tiktok': 0.9, 'youtube': 0.8},
            'hip hop': {'spotify': 0.8, 'tiktok': 1.0, 'youtube': 0.7},
            'electronic': {'spotify': 0.9, 'tiktok': 1.0, 'youtube': 0.6},
            'rock': {'spotify': 0.9, 'tiktok': 0.5, 'youtube': 1.0},
            'country': {'spotify': 1.0, 'tiktok': 0.6, 'youtube': 0.9},
            'r&b': {'spotify': 0.9, 'tiktok': 0.8, 'youtube': 0.7}
        }
        
        # Apply genre affinity
        for platform in self.platform_names:
            col_name = f'genre_{platform}_affinity'
            features[col_name] = features['genre_clean'].map(
                lambda x: genre_platform_affinity.get(x.lower() if isinstance(x, str) else 'pop', 
                                                    {'spotify': 0.7, 'tiktok': 0.7, 'youtube': 0.7})[platform]
            )
        
        # Audio feature combinations that work for different platforms
        features['hook_potential'] = (
            features['energy'] * features['danceability'] * 
            (1 - features['instrumentalness'])
        )
        
        features['mood_appeal'] = (
            features['valence'] * (1 - features['acousticness']) * 
            features['energy']
        )
        
        # Tempo-based features - estimate tempo from energy if not available
        estimated_tempo = features.get('tempo', features['energy'] * 140)
        features['tempo_tiktok_fit'] = np.where(
            (estimated_tempo >= 110) & (estimated_tempo <= 140), 1.0, 0.5
        )
        features['tempo_spotify_fit'] = np.where(
            (estimated_tempo >= 80) & (estimated_tempo <= 160), 1.0, 0.7
        )
        
        return features
    
    def prepare_training_data(self, df):
        """Prepare features and targets with improved feature engineering"""
        # Engineer features
        df_featured = self.engineer_platform_specific_features(df)
        
        # Core audio features
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
        
        # Engineered platform features
        platform_features = [
            'tiktok_viral_score', 'spotify_algorithm_score', 'youtube_engagement_score',
            'genre_spotify_affinity', 'genre_tiktok_affinity', 'genre_youtube_affinity',
            'hook_potential', 'mood_appeal', 'tempo_tiktok_fit', 'tempo_spotify_fit'
        ]
        
        # Quality features (but limit their impact)
        quality_features = ['audio_appeal']
        
        # Exclude popularity to prevent overfitting
        # normalized_popularity removed from features
        
        # Genre encoding
        if 'genre_clean' in df_featured.columns:
            if 'genre_encoder' not in self.label_encoders:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoders['genre_encoder'] = LabelEncoder()
            df_featured['genre_encoded'] = self.label_encoders['genre_encoder'].fit_transform(
                df_featured['genre_clean'].fillna('unknown')
            )
            platform_features.append('genre_encoded')
        
        # Combine features
        feature_cols = audio_features + platform_features + quality_features
        
        # Prepare features
        X = df_featured[feature_cols].copy()
        
        # Handle missing values
        for col in feature_cols:
            if col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
        
        # Prepare targets
        y_cols = []
        y_data = []
        
        for platform in self.platform_names:
            if f'{platform}_combined' in df_featured.columns:
                y_cols.append(f'{platform}_combined')
                y_data.append(df_featured[f'{platform}_combined'].fillna(0))
            elif platform in df_featured.columns:
                y_cols.append(platform)
                y_data.append(df_featured[platform].fillna(0))
            else:
                # Create dummy column
                y_cols.append(f'{platform}_score')
                y_data.append(pd.Series(np.zeros(len(df_featured))))
        
        y = pd.DataFrame(dict(zip(y_cols, y_data)))
        
        # Remove rows where all platform scores are 0 (likely missing data)
        valid_rows = (y.sum(axis=1) > 0)
        X = X[valid_rows]
        y = y[valid_rows]
        
        print(f"Features shape after cleaning: {X.shape}")
        print(f"Targets shape after cleaning: {y.shape}")
        print(f"Feature columns: {X.columns.tolist()}")
        print(f"Target columns: {y.columns.tolist()}")
        print(f"Target statistics:")
        print(y.describe())
        
        return X, y
    
    def train(self, training_data):
        """Train improved platform recommendation model"""
        X, y = self.prepare_training_data(training_data)
        
        # Feature selection to prevent overfitting
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(12, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y.iloc[:, 0])  # Use first platform for selection
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()]
        print(f"Selected features: {selected_features.tolist()}")
        
        # Scale 
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Train with Random Forest 
        base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_split=20,  # Require more samples to split
            min_samples_leaf=10,   # Require more samples in leaf
            max_features='sqrt',   # Use sqrt of features
            random_state=42,
            n_jobs=-1
        )
        
        self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        self.model.fit(X_scaled, y)
        
        # Print feature importance for first estimator
        feature_importance = self.model.estimators_[0].feature_importances_
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 features for platform prediction:")
        print(importance_df.head(10))
        
        return self
    
    def predict(self, audio_features):
        """Predict platform performance scores"""
        # Engineer features
        df_featured = self.engineer_platform_specific_features(audio_features)
        
        # Prepare features (same as training)
        audio_cols = ['danceability', 'energy', 'valence', 'acousticness', 
                     'instrumentalness', 'liveness', 'speechiness']
        
        platform_cols = [
            'tiktok_viral_score', 'spotify_algorithm_score', 'youtube_engagement_score',
            'genre_spotify_affinity', 'genre_tiktok_affinity', 'genre_youtube_affinity',
            'hook_potential', 'mood_appeal', 'tempo_tiktok_fit', 'tempo_spotify_fit'
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
        
        # Apply feature selection
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)[0]
        
        # Format results with more realistic scoring
        platform_scores = {}
        for i, platform in enumerate(self.platform_names):
            raw_score = predictions[i]
            
            # Apply platform-specific scaling and bounds
            if platform == 'tiktok':
                # TikTok scores tend to be more variable
                score = max(0, min(100, raw_score * 20))  # Scale up TikTok predictions
            elif platform == 'spotify':
                # Spotify scores are more stable
                score = max(0, min(100, raw_score))
            else:  # YouTube
                # YouTube in middle
                score = max(0, min(100, raw_score * 0.8))
            
            platform_scores[platform] = {
                'score': float(score),
                'confidence': self._calculate_confidence(score, platform),
                'recommendation': self._generate_platform_advice(platform, score)
            }
        
        # Rank platforms
        sorted_platforms = sorted(
            platform_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        recommendations = {
            'platform_scores': platform_scores,
            'ranked_recommendations': [
                {
                    'platform': platform,
                    'score': data['score'],
                    'confidence': data['confidence'],
                    'recommendation': data['recommendation']
                }
                for platform, data in sorted_platforms
            ],
            'top_platform': sorted_platforms[0][0],
            'top_score': sorted_platforms[0][1]['score']
        }
        
        return recommendations
    
    def _calculate_confidence(self, score, platform):
        """Calculate confidence based on score and platform characteristics"""
        if platform == 'tiktok':
            # TikTok is more unpredictable
            return 'high' if score > 80 else 'medium' if score > 50 else 'low'
        elif platform == 'spotify':
            # Spotify is more predictable
            return 'high' if score > 70 else 'medium' if score > 40 else 'low'
        else:  # YouTube
            return 'high' if score > 75 else 'medium' if score > 45 else 'low'
    
    def _generate_platform_advice(self, platform, score):
        """Generate platform-specific marketing advice"""
        advice_templates = {
            'spotify': {
                'high': "Excellent for Spotify! Target editorial and algorithmic playlists.",
                'medium': "Good Spotify potential. Focus on genre playlists and Release Radar.",
                'low': "Consider playlist pitching. May work better with remixes or acoustic versions."
            },
            'tiktok': {
                'high': "High TikTok viral potential! Create dance content and trend challenges.",
                'medium': "Good TikTok fit. Focus on hook moments and short-form content.",
                'low': "Requires creative approach. Try storytelling or behind-the-scenes content."
            },
            'youtube': {
                'high': "Perfect for YouTube! Create music videos and visual content.",
                'medium': "Good YouTube potential. Try lyric videos or live sessions.",
                'low': "Focus on other platforms first. Consider podcast or interview content."
            }
        }
        
        confidence = self._calculate_confidence(score, platform)
        return advice_templates.get(platform, {}).get(confidence, f"Score: {score:.0f}")
    
    def evaluate_model(self, test_data):
        """Evaluate model performance with detailed metrics"""
        X_test, y_test = self.prepare_training_data(test_data)
        X_test_selected = self.feature_selector.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics for each platform
        results = {}
        for i, platform in enumerate(self.platform_names):
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            y_true = y_test.iloc[:, i]
            y_pred_platform = y_pred[:, i]
            
            # Avoid division by zero
            non_zero_mask = y_true != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred_platform[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                mape = np.inf
            
            results[platform] = {
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'score_range': f"{y_true.min():.1f} - {y_true.max():.1f}"
            }
            
            print(f"{platform.title()}:")
            print(f"  MAE: {mae:.3f}")
            print(f"  R²: {r2:.3f}")
            print(f"  MAPE: {mape:.1f}%")
            print(f"  Score Range: {y_true.min():.1f} - {y_true.max():.1f}")
            print()
        
        return results
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'platform_names': self.platform_names,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.platform_names = model_data['platform_names']
        self.label_encoders = model_data['label_encoders']
        return self

# Training script with improved validation
def train_platform_model():
    # Load integrated data
    training_data = pd.read_csv('integrated_platform_training.csv')
    
    print(f"Training data shape: {training_data.shape}")
    print("\nPlatform score distributions:")
    platform_cols = [col for col in training_data.columns if any(p in col.lower() for p in ['spotify', 'tiktok', 'youtube'])]
    for col in platform_cols:
        if training_data[col].dtype in ['float64', 'int64']:
            print(f"{col}: mean={training_data[col].mean():.2f}, std={training_data[col].std():.2f}, range={training_data[col].min():.1f}-{training_data[col].max():.1f}")
    
    # Split data with stratification by platform performance
    train_df, test_df = train_test_split(training_data, test_size=0.2, random_state=42)
    
    # Initialize and train model
    platform_model = PlatformRecommender()
    platform_model.train(train_df)
    
    # Evaluate model
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    platform_model.evaluate_model(test_df)
    
    # Save 
    platform_model.save_model('models/platform_recommender.pkl')
    print("Platform model saved!")
    
    return platform_model

if __name__ == "__main__":
    model = train_platform_model()