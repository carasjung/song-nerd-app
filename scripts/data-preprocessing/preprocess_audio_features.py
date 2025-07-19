# preprocess_audio_features.py
import pandas as pd
import logging
from audio_feature_config import AUDIO_FEATURES, RENAME_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardize_columns(df):
    df = df.rename(columns=lambda col: RENAME_MAP.get(col, col.lower()))
    return df

def normalize_features(df):
    for col, (min_val, max_val) in AUDIO_FEATURES.items():
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            # Log outliers before clipping
            outliers = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if outliers > 0:
                logger.warning(f"{col}: {outliers} values outside expected range")
            
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            if max_val != min_val and col not in ['key', 'mode', 'time_signature', 'duration_ms', 'loudness']:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

def clean_missing(df):
    audio_cols = [col for col in AUDIO_FEATURES if col in df.columns]
    before = len(df)
    df = df.dropna(subset=audio_cols)
    after = len(df)
    if before != after:
        logger.info(f"Dropped {before - after} rows with missing values")
    return df

def preprocess_audio_file(path):
    logger.info(f"Processing: {path}")
    df = pd.read_csv(path)
    logger.info(f"Original shape: {df.shape}")
    
    df = standardize_columns(df)
    df = normalize_features(df)
    df = clean_missing(df)
    
    logger.info(f"Final shape: {df.shape}")
    return df