# demographics_model_adapted.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class DemographicsPredictor:
    def __init__(self):
        self.age_model = None
        self.region_model = None
        self.platform_pref_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_features(self, df):
        """Prepare features using actual data structure"""
        # Audio features from master dataset
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
        
        # Additional features from your data
        additional_features = ['audio_appeal', 'normalized_popularity']
        
        # Genre encoding
        if 'genre_clean' in df.columns:
            if 'genre_encoder' not in self.label_encoders:
                self.label_encoders['genre_encoder'] = LabelEncoder()
            df['genre_encoded'] = self.label_encoders['genre_encoder'].fit_transform(
                df['genre_clean'].fillna('unknown')
            )
            additional_features.append('genre_encoded')
        
        # Platform performance features (if available)
        platform_features = []
        for platform in ['spotify', 'tiktok', 'youtube']:
            if platform in df.columns:
                platform_features.append(platform)
                df[platform] = df[platform].fillna(0)
        
        # Combine all features
        all_features = audio_features + additional_features + platform_features
        
        # Handle missing values
        for feature in audio_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())
        
        for feature in additional_features:
            if feature in df.columns and feature != 'genre_encoded':
                df[feature] = df[feature].fillna(df[feature].median())
        
        return df[all_features]
    
    def train(self, training_data):
        """Train demographics prediction models"""
        print("Preparing features...")
        X = self.prepare_features(training_data)
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare target variables
        y_age = training_data['age_group']
        y_region = training_data['region'] 
        y_platform = training_data['preferred_platform']
        
        print(f"Training data shape: {X_scaled.shape}")
        print(f"Age groups: {y_age.value_counts()}")
        print(f"Regions: {y_region.value_counts()}")
        
        # Train age group prediction model
        print("Training age group model...")
        self.age_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.age_model.fit(X_scaled, y_age)
        
        # Train region prediction model
        print("Training region model...")
        self.region_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.region_model.fit(X_scaled, y_region)
        
        # Train platform preference model
        print("Training platform preference model...")
        self.platform_pref_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.platform_pref_model.fit(X_scaled, y_platform)
        
        # Print feature importance
        feature_names = X.columns
        age_importance = self.age_model.feature_importances_
        
        print("\nTop 5 features for age prediction:")
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': age_importance
        }).sort_values('importance', ascending=False)
        print(importance_df.head())
        
        return self
    
    def predict(self, audio_features):
        """Predict demographics for new song"""
        X = self.prepare_features(audio_features)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions with probabilities
        age_probs = self.age_model.predict_proba(X_scaled)[0]
        age_classes = self.age_model.classes_
        
        region_probs = self.region_model.predict_proba(X_scaled)[0]
        region_classes = self.region_model.classes_
        
        platform_probs = self.platform_pref_model.predict_proba(X_scaled)[0]
        platform_classes = self.platform_pref_model.classes_
        
        # Format results
        demographics = {
            'age_groups': {
                age_classes[i]: float(age_probs[i]) 
                for i in range(len(age_classes))
            },
            'regions': {
                region_classes[i]: float(region_probs[i]) 
                for i in range(len(region_classes))
            },
            'platform_preferences': {
                platform_classes[i]: float(platform_probs[i]) 
                for i in range(len(platform_classes))
            },
            'primary_age_group': age_classes[np.argmax(age_probs)],
            'primary_region': region_classes[np.argmax(region_probs)],
            'preferred_platform': platform_classes[np.argmax(platform_probs)],
            'confidence_scores': {
                'age': float(np.max(age_probs)),
                'region': float(np.max(region_probs)),
                'platform': float(np.max(platform_probs))
            }
        }
        
        return demographics
    
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        X_test = self.prepare_features(test_data)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Age prediction accuracy
        age_pred = self.age_model.predict(X_test_scaled)
        age_accuracy = accuracy_score(test_data['age_group'], age_pred)
        
        # Region prediction accuracy
        region_pred = self.region_model.predict(X_test_scaled)
        region_accuracy = accuracy_score(test_data['region'], region_pred)
        
        print(f"Age Group Prediction Accuracy: {age_accuracy:.3f}")
        print(f"Region Prediction Accuracy: {region_accuracy:.3f}")
        
        print("\nAge Group Classification Report:")
        print(classification_report(test_data['age_group'], age_pred))
        
        return {
            'age_accuracy': age_accuracy,
            'region_accuracy': region_accuracy
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'age_model': self.age_model,
            'region_model': self.region_model,
            'platform_pref_model': self.platform_pref_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.age_model = model_data['age_model']
        self.region_model = model_data['region_model']
        self.platform_pref_model = model_data['platform_pref_model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        return self

# Training script
def train_demographics_model():
    # Load integrated data
    training_data = pd.read_csv('integrated_demographics_training.csv')
    
    # Split data
    train_df, test_df = train_test_split(training_data, test_size=0.2, random_state=42)
    
    # Initialize and train model
    demo_model = DemographicsPredictor()
    demo_model.train(train_df)
    
    # Evaluate model
    demo_model.evaluate_model(test_df)
    
    # Save model
    demo_model.save_model('models/demographics_predictor.pkl')
    print("Demographics model saved!")
    
    return demo_model

if __name__ == "__main__":
    model = train_demographics_model()