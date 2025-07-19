from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import asyncio
import tempfile
from supabase import create_client, Client
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Song Nerd API", version="1.0.0")

# CORS middleware - Update with your Vercel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing only)
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    logger.warning("SUPABASE_URL and SUPABASE_SERVICE_KEY not set - using mock mode")
    supabase = None
else:
    supabase: Client = create_client(supabase_url, supabase_key)

class SongAnalysisRequest(BaseModel):
    song_id: str
    file_url: str
    metadata: dict

class DirectUploadRequest(BaseModel):
    song_id: str
    metadata: dict

@app.get("/")
async def root():
    return {
        "message": "Song Nerd API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": [
            "/api/songs/analyze",
            "/api/songs/upload",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "song-nerd-api",
        "supabase_connected": supabase is not None
    }

@app.post("/api/songs/analyze")
async def analyze_song_endpoint(request: SongAnalysisRequest, background_tasks: BackgroundTasks):
    """Trigger AI analysis for a song from URL"""
    try:
        logger.info(f"Starting analysis for song {request.song_id}")
        
        # Update song status to processing (if Supabase is available)
        if supabase:
            supabase.table("songs").update({
                "processing_status": "processing"
            }).eq("id", request.song_id).execute()
        
        # Add analysis to background task
        background_tasks.add_task(
            process_song_analysis, 
            request.song_id, 
            request.file_url, 
            request.metadata
        )
        
        return {
            "message": "Analysis started",
            "song_id": request.song_id,
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        await update_song_status(request.song_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/songs/upload")
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    song_id: str = None,
    metadata: str = "{}"
):
    """Upload audio file directly and analyze"""
    try:
        import json
        metadata_dict = json.loads(metadata) if metadata else {}
        
        if not song_id:
            raise HTTPException(status_code=400, detail="song_id is required")
        
        logger.info(f"Starting upload analysis for song {song_id}")
        
        # Update song status to processing
        await update_song_status(song_id, "processing")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Add analysis to background task
        background_tasks.add_task(
            process_uploaded_file_analysis,
            song_id,
            temp_file_path,
            metadata_dict
        )
        
        return {
            "message": "File uploaded and analysis started",
            "song_id": song_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        if song_id:
            await update_song_status(song_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def update_song_status(song_id: str, status: str, error_message: str = None):
    """Update song processing status in database"""
    try:
        if not supabase:
            logger.info(f"Mock mode: Song {song_id} status: {status}")
            return
            
        update_data = {"processing_status": status}
        if error_message:
            update_data["error_message"] = error_message
        
        supabase.table("songs").update(update_data).eq("id", song_id).execute()
    except Exception as e:
        logger.error(f"Failed to update song status: {e}")

async def download_audio_file(file_url: str) -> str:
    """Download audio file from URL to temporary file"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(response.content)
                return temp_file.name
                
    except Exception as e:
        logger.error(f"Failed to download audio file: {e}")
        raise

async def process_song_analysis(song_id: str, file_url: str, metadata: dict):
    """Background task to process song analysis from URL"""
    temp_file_path = None
    try:
        logger.info(f"Processing analysis for song {song_id}")
        
        # Download the audio file
        temp_file_path = await download_audio_file(file_url)
        
        # Process the analysis
        await run_analysis(song_id, temp_file_path, metadata)
        
    except Exception as e:
        logger.error(f"Analysis failed for song {song_id}: {e}")
        await update_song_status(song_id, "failed", str(e))
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

async def process_uploaded_file_analysis(song_id: str, temp_file_path: str, metadata: dict):
    """Background task to process uploaded file analysis"""
    try:
        logger.info(f"Processing uploaded file analysis for song {song_id}")
        await run_analysis(song_id, temp_file_path, metadata)
        
    except Exception as e:
        logger.error(f"Analysis failed for song {song_id}: {e}")
        await update_song_status(song_id, "failed", str(e))
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

async def run_analysis(song_id: str, audio_file_path: str, metadata: dict):
    """Run the actual AI analysis on the audio file"""
    try:
        # For now, create mock analysis results
        # TODO: Replace with actual analysis logic when dependencies are working
        mock_analysis = {
            "danceability": 0.7,
            "energy": 0.8,
            "valence": 0.6,
            "acousticness": 0.3,
            "instrumentalness": 0.1,
            "liveness": 0.2,
            "speechiness": 0.1,
            "tempo": 120.0,
            "loudness": -5.0,
            "key": 7,
            "mode": 1,
            "time_signature": 4,
            "audio_appeal": 85.0,
            "processing_time": 2.5,
            "raw_features": {},
            "sound_profile": "Modern pop with strong melodic elements"
        }
        
        mock_insights = {
            "primary_age_group": "18-24",
            "age_confidence": 0.85,
            "primary_region": "North America",
            "region_confidence": 0.78,
            "top_platform": "spotify",
            "platform_scores": {
                "spotify": 88,
                "tiktok": 75,
                "instagram": 82,
                "youtube": 79
            },
            "similar_artists": [
                {"artist_name": "Taylor Swift", "genre": metadata.get("genre", "pop"), "similarity_score": 0.8},
                {"artist_name": "Ed Sheeran", "genre": metadata.get("genre", "pop"), "similarity_score": 0.75}
            ],
            "action_items": [
                "Focus marketing on 18-24 age group",
                "Prioritize Spotify playlist submissions",
                "Create TikTok-friendly content",
                "Develop Instagram story campaigns"
            ],
            "competitive_advantage": "High danceability with emotional depth",
            "overall_confidence": 0.82,
            "model_version": "1.0"
        }
        
        if supabase:
            # Insert analysis results
            supabase.table("analysis").insert({
                "song_id": song_id,
                **mock_analysis
            }).execute()
            
            supabase.table("marketing_insights").insert({
                "song_id": song_id,
                **mock_insights
            }).execute()
        else:
            logger.info(f"Mock mode: Analysis complete for song {song_id}")
        
        # Update song status to completed
        await update_song_status(song_id, "completed")
        
        logger.info(f"Analysis completed for song {song_id}")
        
    except Exception as e:
        logger.error(f"Analysis processing failed for song {song_id}: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)