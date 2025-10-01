# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Whisper Speech-to-Text API service with a modern React frontend. The project consists of:

- **Backend (remote_server/)**: FastAPI-based service for speech-to-text processing using Faster-Whisper and Pyannote
- **Frontend (frontend/)**: React + TypeScript + Vite application for uploading audio and monitoring progress

## Commands

### Backend Development

```bash
# Navigate to backend directory
cd remote_server

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
python api.py
```

The backend API runs on `http://localhost:8000` with interactive docs at `/docs` and `/redoc`.

### Frontend Development

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The frontend development server runs on `http://localhost:5173` with API proxy to backend.

### Environment Setup

The backend requires a `.env` file in `remote_server/` with:
```env
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

## Architecture Overview

### Backend Architecture (remote_server/)

- **api.py**: FastAPI main application with all API endpoints
- **database.py**: SQLite database management for task tracking
- **task_processor.py**: Core speech processing logic using Faster-Whisper and Pyannote
- **whisper_api.py**: Additional Whisper-related utilities

The backend uses an asynchronous task queue system where:
1. Tasks are submitted via `/api/tasks` endpoint
2. Files are stored in `uploads/{task_id}/`
3. Processing happens asynchronously with real-time progress via SSE
4. Results are stored in `result/{task_id}/`

### Frontend Architecture (frontend/src/)

- **ui/App.tsx**: Main application component
- **components/**: Reusable React components
  - **UploadSection.tsx**: File upload with drag-and-drop
  - **TaskProgress.tsx**: Real-time progress display with SSE
  - **TaskHistory.tsx**: Historical task management
  - **ServiceStats.tsx**: Service statistics display
- **api.ts**: Axios-based API client
- **types.ts**: TypeScript type definitions
- **styles/main.css**: Global styles

### Key Features

1. **Real-time Progress**: Uses Server-Sent Events (SSE) for live progress updates
2. **Speaker Diarization**: Optional multi-speaker recognition using Pyannote
3. **Task Management**: Complete lifecycle management with SQLite persistence
4. **File Handling**: Supports MP3, WAV, M4A, FLAC formats
5. **Proxy Configuration**: Vite proxy routes `/api` to backend at `localhost:8000`

### Processing Pipeline

1. **Model Loading** (0-5%): Load Whisper and optional Pyannote models
2. **Audio Conversion** (20-25%): Convert to appropriate format using FFmpeg
3. **Speech Recognition** (30-60%): Transcribe audio using Faster-Whisper
4. **Speaker Diarization** (70-85%): Optional speaker separation
5. **Integration** (85-95%): Combine transcription with speaker information
6. **Completion** (100%): Save results and update database

### Database Schema

SQLite database (`tasks.db`) tracks:
- Task metadata (ID, filename, status, progress)
- Client IP for task history
- Processing timestamps and error handling
- File paths and configuration options

### API Integration Points

- **POST /api/tasks**: Submit new transcription tasks
- **GET /api/tasks/{id}**: Query task status
- **GET /api/tasks/{id}/stream**: SSE progress updates
- **GET /api/tasks/{id}/download**: Download results
- **DELETE /api/tasks/{id}**: Cancel tasks
- **GET /api/my-tasks**: Client task history
- **GET /api/stats**: Service statistics

### Development Notes

- Backend uses CUDA if available, falls back to CPU
- First run downloads models automatically (requires time and storage)
- FFmpeg path is auto-detected from project directory if present
- Frontend uses Vite's proxy for seamless API integration during development
- Both applications support hot reloading during development
