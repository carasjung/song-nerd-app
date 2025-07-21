# Song Nerd

**Song Nerd** is an AI-powered music marketing platform that processes and analyzes audio data to generate high-quality training datasets and actionable insights for artists. It leverages advanced machine learning models to help artists, managers, and marketers optimize their music's reach and impact across streaming and social platforms.

---

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
  - [Frontend Setup (Next.js)](#frontend-setup-nextjs)
  - [Backend Setup (FastAPI)](#backend-setup-fastapi)
  - [Environment Variables](#environment-variables)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **AI-Powered Audio Analysis**: Extracts key audio features and automates data labeling using ML models.
- **High-Quality Training Datasets**: Generates structured datasets with confidence scoring and robust validation.
- **Real-time Data Processing Pipeline**: Automated, with built-in quality control, progress tracking, and systematic validation.
- **Visual Performance Metrics**: Offers visual representation of performance metrics and actionable insights.
- **Marketing Insights**: Recommends optimal platforms, target demographics, and marketing actions for each song.
- **Similar Artist Discovery**: Suggests similar artists for playlisting and collaboration opportunities.
- **User Dashboard**: Track uploads, analysis results, and historical performance.
- **Exportable Reports**: Download marketing analysis as PDF for sharing or record-keeping.

---

## Tech Stack
- **Frontend**: [Next.js](https://nextjs.org/), React, Tailwind CSS
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), Python 3.10+
- **Database & Auth**: [Supabase](https://supabase.com/)
- **Deployment**: Vercel (frontend), Railway (backend)
- **ML/AI**: Custom Python models (demographics, platform recommendation, similar artists)

---

## Architecture Overview

```mermaid
graph TD;
  A[User Uploads Song] --> B[Frontend (Next.js)]
  B -->|File & Metadata| C[Supabase Storage/DB]
  B -->|Trigger| D[Backend API (FastAPI)]
  D -->|Fetches File| C
  D -->|Runs ML Analysis| E[ML Models]
  E -->|Results| D
  D -->|Stores Results| C
  B -->|Fetches Results| C
  B -->|Displays Insights| F[User Dashboard]
```

---

## Getting Started

### Prerequisites
- Node.js (v18+ recommended)
- Python 3.10+
- Supabase account & project
- Vercel/Railway accounts (for deployment)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/song-nerd-app.git
cd song-nerd-app
```

### 2. Frontend Setup (Next.js)
```bash
cd song-nerd-app
npm install
```

#### Start the frontend locally:
```bash
npm run dev
```

### 3. Backend Setup (FastAPI)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Start the backend locally:
```bash
uvicorn main:app --reload --port 8000
```

---

## Environment Variables

### Frontend (`.env.local`)
```
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000  # or your deployed backend URL
```

### Backend (`.env` in `backend/`)
```
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-service-role-key
SUPABASE_SERVICE_KEY=your-supabase-service-role-key
PORT=8000
```

---

## Usage Guide

1. **Upload a Song**: Drag & drop or select an audio file (MP3, WAV, M4A, FLAC) and enter song metadata (title, artist, genre).
2. **Processing**: The backend fetches the file, runs ML analysis, and updates the status in real-time.
3. **View Results**: See detailed audio features, platform recommendations, target demographics, similar artists, and marketing action items.
4. **Export**: Download a PDF report of the analysis for sharing or record-keeping.
5. **Dashboard**: Track all your uploads, analysis history, and performance analytics.

---

## API Reference

### Backend Endpoints (FastAPI)
- `GET /` — Health/status and available endpoints
- `GET /health` — Health check (returns Supabase connection status)
- `POST /api/songs/analyze` — Trigger analysis for a song (from file URL)
  - **Body:** `{ song_id: str, file_url: str, metadata: dict }`
- `POST /api/songs/upload` — Upload audio file directly and trigger analysis
  - **Form Data:** `file`, `song_id`, `metadata` (JSON string)

#### Example: Trigger Analysis
```json
POST /api/songs/analyze
{
  "song_id": "abc123",
  "file_url": "https://.../mysong.mp3",
  "metadata": { "title": "My Song", "artist": "Me", "genre": "pop" }
}
```

### Frontend API Usage
- Uses `/src/services/api.ts` to communicate with backend and Supabase
- Handles file upload, song record creation, analysis trigger, and status polling

---

## Deployment

### Frontend (Vercel)
- Connect your repo to Vercel
- Set environment variables in Vercel dashboard
- Deploy (Vercel auto-detects Next.js)

### Backend (Railway)
- Deploy `backend/` as a Python service
- Set environment variables in Railway dashboard
- Health check path: `/health`
- Expose port `8000`

---

## Troubleshooting & FAQ

**Q: My analysis is stuck in processing.**
- Check backend logs for errors
- Ensure Supabase credentials are correct and tables exist
- Confirm the backend can access the uploaded file URL

**Q: I get a Supabase error.**
- Double-check all environment variables
- Make sure your Supabase project has the required tables: `songs`, `analysis`, `marketing_insights`

**Q: How do I add new ML models?**
- Place your model files in `backend/models/`
- Update `integrated_analyzer.py` to load and use your models

**Q: Can I use a different database?**
- The app is tightly integrated with Supabase, but you can adapt the backend for other databases with some refactoring.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or documentation improvements.

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your fork and open a PR

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. 