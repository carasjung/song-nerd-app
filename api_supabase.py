# api_supabase.py

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
import shutil
import json
import numpy as np
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def convert_to_json_safe(data):
    """Convert data to JSON-safe format"""
    return json.loads(json.dumps(data, cls=NumpyEncoder))

from supabase_config import get_supabase_client, get_admin_client
from supabase import Client

from direct_audio_test import extract_audio_features_direct
from integrated_analyzer import MusicMarketingAnalyzer

analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global analyzer
    
    print("Starting Music Marketing API with Supabase...")
    
    # Test Supabase connection
    try:
        supabase = get_supabase_client()
        result = supabase.table('songs').select('id').limit(1).execute()
        print("Supabase connection successful!")
    except Exception as e:
        print(f"Supabase connection failed: {e}")
        print("Make sure you've:")
        print("1. Created your Supabase project")
        print("2. Added credentials to .env file") 
        print("3. Created the database tables")
    
    # Load ML models
    print("Loading ML models...")
    try:
        analyzer = MusicMarketingAnalyzer()
        analyzer.load_models()
        
        if analyzer.models_loaded:
            print("ML models loaded successfully!")
        else:
            print("ML models not fully loaded, will use basic analysis")
    except Exception as e:
        print(f"Error loading ML models: {e}")
        analyzer = None
    
    yield
    
    # Shutdown
    print("Shutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="Music Marketing AI",
    description="AI-powered music marketing insights with Supabase",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_supabase() -> Client:
    """Dependency to get Supabase client"""
    return get_supabase_client()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Music Marketing AI with Supabase",
        "status": "running",
        "version": "1.0.0",
        "models_loaded": analyzer is not None and getattr(analyzer, 'models_loaded', False),
        "database": "supabase",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check(supabase: Client = Depends(get_supabase)):
    """Detailed health check"""
    try:
        songs_result = supabase.table('songs').select('id', count='exact').execute()
        
        return {
            "status": "healthy",
            "models_loaded": analyzer is not None and getattr(analyzer, 'models_loaded', False),
            "database": "supabase-connected",
            "songs_in_db": songs_result.count if hasattr(songs_result, 'count') else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database_error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

def validate_audio_file(file: UploadFile) -> tuple[bool, list[str]]:
    """Validate uploaded audio file"""
    errors = []
    
    allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    if file.filename:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            errors.append(f"Unsupported format: {file_ext}")
    else:
        errors.append("No filename provided")
    
    return len(errors) == 0, errors

async def save_uploaded_file(upload_file: UploadFile, song_id: str) -> str:
    """Save uploaded file locally (will move to Supabase Storage later)"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_extension = os.path.splitext(upload_file.filename)[1] if upload_file.filename else '.mp3'
    safe_filename = f"{song_id}{file_extension}"
    file_path = os.path.join(upload_dir, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

def process_song_with_supabase(song_id: str, file_path: str, metadata: dict):
    """Process song and save results to Supabase"""
    global analyzer
    
    supabase = get_admin_client()
    
    try:
        # Update status to processing
        supabase.table('songs').update({
            'processing_status': 'processing'
        }).eq('id', song_id).execute()
        
        print(f"Processing song {song_id}...")
        start_time = datetime.utcnow()
        
        # Extract audio features
        features = extract_audio_features_direct(file_path)
        
        if not features:
            raise Exception("Failed to extract audio features")
        
        # Run marketing analysis
        if analyzer and getattr(analyzer, 'models_loaded', False):
            analysis_result = analyzer.analyze_song(features, metadata)
        else:
            analysis_result = create_basic_analysis(features, metadata)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Save analysis to Supabase
        analysis_data = {
            'song_id': song_id,
            'danceability': float(features.get('danceability', 0)),
            'energy': float(features.get('energy', 0)),
            'valence': float(features.get('valence', 0)),
            'acousticness': float(features.get('acousticness', 0)),
            'instrumentalness': float(features.get('instrumentalness', 0)),
            'liveness': float(features.get('liveness', 0)),
            'speechiness': float(features.get('speechiness', 0)),
            'tempo': float(features.get('tempo', 0)),
            'loudness': float(features.get('loudness', 0)),
            'key': int(features.get('key', 0)),
            'mode': int(features.get('mode', 0)),
            'time_signature': int(features.get('time_signature', 4)),
            'audio_appeal': float(features.get('audio_appeal', 0)),
            'processing_time': processing_time,
            'raw_features': convert_to_json_safe(features)
        }
        
        supabase.table('analysis').insert(analysis_data).execute()
        
        # Save marketing insights
        if 'target_demographics' in analysis_result:
            insights_data = {
                'song_id': song_id,
                'primary_age_group': analysis_result['target_demographics']['primary_age_group'],
                'age_confidence': analysis_result['target_demographics']['confidence_scores']['age'],
                'primary_region': analysis_result['target_demographics']['primary_region'],
                'region_confidence': analysis_result['target_demographics']['confidence_scores']['region'],
                'top_platform': analysis_result['platform_recommendations']['top_platform'],
                'platform_scores': analysis_result['platform_recommendations'].get('platform_scores', {}),
                'similar_artists': [
                    {
                        'artist_name': artist['artist_name'],
                        'similarity_score': artist['similarity_score'],
                        'genre': artist['genre']
                    }
                    for artist in analysis_result['similar_artists']['similar_artists'][:5]
                ],
                'action_items': analysis_result['marketing_insights']['action_items'],
                'sound_profile': analysis_result['marketing_insights']['positioning']['sound_profile'],
                'competitive_advantage': analysis_result['marketing_insights']['positioning']['competitive_advantage'],
                'overall_confidence': analysis_result['confidence_scores']['platforms']
            }
            
            supabase.table('marketing_insights').insert(insights_data).execute()
        
        # Update song status
        supabase.table('songs').update({
            'processing_status': 'completed',
            'duration': features.get('duration')
        }).eq('id', song_id).execute()
        
        print(f"Song {song_id} processed successfully in {processing_time:.2f}s")
        
    except Exception as e:
        print(f"Error processing song {song_id}: {e}")
        supabase.table('songs').update({
            'processing_status': 'failed'
        }).eq('id', song_id).execute()

def create_basic_analysis(features: dict, metadata: dict) -> dict:
    """Basic analysis fallback"""
    platforms = []
    
    if features.get('danceability', 0) > 0.7:
        platforms.append(('tiktok', 75))
    if features.get('valence', 0) > 0.6:
        platforms.append(('spotify', 70))
    if features.get('energy', 0) > 0.6:
        platforms.append(('youtube', 65))
    
    if not platforms:
        platforms = [('spotify', 50), ('tiktok', 45), ('youtube', 40)]
    
    platforms.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'song_info': metadata,
        'target_demographics': {
            'primary_age_group': '18-24',
            'primary_region': 'global',
            'confidence_scores': {'age': 0.5, 'region': 0.5}
        },
        'platform_recommendations': {
            'top_platform': platforms[0][0],
            'top_score': platforms[0][1],
            'platform_scores': {p[0]: {'score': p[1]} for p in platforms}
        },
        'similar_artists': {
            'similar_artists': []
        },
        'marketing_insights': {
            'action_items': [f"Focus on {platforms[0][0]} for marketing"],
            'positioning': {
                'sound_profile': 'Basic analysis available',
                'competitive_advantage': 'Run full analysis for detailed insights'
            }
        },
        'confidence_scores': {
            'platforms': 0.5
        }
    }

@app.post("/api/songs/upload")
async def upload_song(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    artist_name: Optional[str] = None,
    genre: Optional[str] = None,
    supabase: Client = Depends(get_supabase)
):
    """Upload and analyze song"""
    
    is_valid, errors = validate_audio_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail={"errors": errors})
    
    song_id = str(uuid.uuid4())
    
    if not title and file.filename:
        title = os.path.splitext(file.filename)[0]
    if not artist_name:
        artist_name = "Unknown Artist"
    if not genre:
        genre = "pop"
    
    try:
        file_path = await save_uploaded_file(file, song_id)
        
        song_data = {
            'id': song_id,
            'title': title,
            'artist_name': artist_name,
            'genre': genre,
            'file_path': file_path,
            'file_size': getattr(file, 'size', 0),
            'processing_status': 'pending',
            'user_id': None
        }
        
        supabase.table('songs').insert(song_data).execute()
        
        metadata = {
            'track_name': title,
            'artist_name': artist_name,
            'genre': genre
        }
        
        background_tasks.add_task(process_song_with_supabase, song_id, file_path, metadata)
        
        return song_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/songs/{song_id}/status")
async def get_song_status(song_id: str, supabase: Client = Depends(get_supabase)):
    """Get song status"""
    result = supabase.table('songs').select('*').eq('id', song_id).execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Song not found")
    
    return result.data[0]

@app.get("/api/songs/{song_id}/analysis")
async def get_song_analysis(song_id: str, supabase: Client = Depends(get_supabase)):
    """Get complete analysis"""
    
    song_result = supabase.table('songs').select('*').eq('id', song_id).execute()
    
    if not song_result.data:
        raise HTTPException(status_code=404, detail="Song not found")
    
    song = song_result.data[0]
    
    if song['processing_status'] == "pending":
        return {"id": song_id, "status": "pending", "message": "Analysis queued"}
    elif song['processing_status'] == "processing":
        return {"id": song_id, "status": "processing", "message": "Analysis in progress"}
    elif song['processing_status'] == "failed":
        raise HTTPException(status_code=422, detail="Analysis failed")
    
    analysis_result = supabase.table('analysis').select('*').eq('song_id', song_id).execute()
    insights_result = supabase.table('marketing_insights').select('*').eq('song_id', song_id).execute()
    
    if not analysis_result.data:
        raise HTTPException(status_code=500, detail="Analysis data not found")
    
    analysis = analysis_result.data[0]
    insights = insights_result.data[0] if insights_result.data else None
    
    response = {
        "song_id": song_id,
        "status": "completed",
        "audio_features": analysis['raw_features'],
        "processing_time": analysis['processing_time'],
        "created_at": analysis['created_at']
    }
    
    if insights:
        response["marketing_analysis"] = {
            "target_demographics": {
                "primary_age_group": insights['primary_age_group'],
                "primary_region": insights['primary_region'],
                "confidence_scores": {
                    "age": insights['age_confidence'],
                    "region": insights['region_confidence']
                }
            },
            "platform_recommendations": {
                "top_platform": insights['top_platform'],
                "platform_scores": insights['platform_scores'],
                "ranked_recommendations": [
                    {
                        "platform": platform,
                        "score": data.get("score", 0),
                        "confidence": "medium"
                    }
                    for platform, data in insights['platform_scores'].items()
                ] if insights['platform_scores'] else []
            },
            "similar_artists": {
                "similar_artists": insights['similar_artists'] or []
            },
            "marketing_insights": {
                "action_items": insights['action_items'] or [],
                "positioning": {
                    "sound_profile": insights['sound_profile'],
                    "competitive_advantage": insights['competitive_advantage']
                }
            },
            "analysis_summary": {
                "headline": f"Target {insights['primary_age_group']} on {insights['top_platform']}",
                "confidence_level": "high" if insights['overall_confidence'] > 0.7 else "medium"
            },
            "confidence_scores": {
                "platforms": insights['overall_confidence']
            }
        }
    
    return response

@app.get("/api/songs")
async def list_songs(supabase: Client = Depends(get_supabase)):
    """List all songs"""
    result = supabase.table('songs').select('*').order('upload_timestamp', desc=True).execute()
    return {"songs": result.data, "total": len(result.data)}

if __name__ == "__main__":
    print("Starting Music Marketing AI with Supabase...")
    uvicorn.run(app, host="127.0.0.1", port=8000)