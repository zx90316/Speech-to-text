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
ADMIN_TOKEN=your_admin_token_here
```

Reference [.env.txt](.env.txt) for the template.

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

- **ui/App.tsx**: Main application component with view mode switching (main/admin)
- **components/**: Reusable React components
  - **UploadSection.tsx**: File upload with drag-and-drop, audio player for time range selection
  - **TaskProgress.tsx**: Real-time progress display with SSE, partial result preview
  - **TaskHistory.tsx**: Historical task management with batch operations
  - **ServiceStats.tsx**: Service statistics display
  - **AudioPlayer.tsx**: Audio playback with time range selection
- **pages/AdminPage.tsx**: Admin interface for viewing all tasks and system stats
- **api.ts**: Axios-based API client
- **types.ts**: TypeScript type definitions
- **utils/taskStorage.ts**: LocalStorage management for task IDs
- **styles/main.css**: Global styles

### Key Features

1. **Real-time Progress**: Uses Server-Sent Events (SSE) for live progress updates with partial results
2. **Speaker Diarization**: Optional multi-speaker recognition using Pyannote
3. **Task Management**: Complete lifecycle management with SQLite persistence
4. **File Handling**: Supports MP3, WAV, M4A, FLAC formats
5. **Time Range Selection**: Can process specific segments of audio files
6. **Multiple Models**: Support for different Whisper models and languages
7. **Admin Dashboard**: Token-based admin interface for system monitoring
8. **Proxy Configuration**: Vite proxy routes `/api` to backend at `localhost:8000`

### Processing Pipeline

1. **Model Loading** (0-5%): Load Whisper and optional Pyannote models
2. **Audio Conversion** (20-25%): Convert to appropriate format using FFmpeg
3. **Speech Recognition** (30-60%): Transcribe audio using Faster-Whisper
4. **Speaker Diarization** (70-85%): Optional speaker separation
5. **Integration** (85-95%): Combine transcription with speaker information
6. **Completion** (100%): Save results and update database

### Database Schema

SQLite database (`tasks.db`) in [remote_server/database.py](remote_server/database.py) tracks:
- Task metadata (ID, filename, status, progress, current_stage)
- Client IP for task history
- Processing configuration (enable_diarization, start_time, end_time, language, task, model)
- Processing timestamps (created_at, started_at, completed_at)
- Error handling and partial results
- Result file paths and queue position

### API Integration Points

- **POST /api/tasks**: Submit new transcription tasks with optional parameters (enable_diarization, start_time, end_time, language, task, model)
- **GET /api/tasks/{id}**: Query task status
- **GET /api/tasks/{id}/stream**: SSE progress updates with partial results
- **GET /api/tasks/{id}/download**: Download results (transcript or raw)
- **DELETE /api/tasks/{id}**: Cancel tasks (or permanently delete with `?permanent=true`)
- **POST /api/tasks/batch**: Batch query multiple tasks
- **GET /api/my-tasks**: Client task history based on IP
- **GET /api/stats**: Service statistics (queue size, processing count)
- **GET /api/admin/tasks**: Admin endpoint to view all tasks (requires ADMIN_TOKEN)
- **GET /api/admin/stats**: Admin system statistics

See full API documentation at `http://localhost:8000/docs` when server is running.

### Development Notes

- Backend uses CUDA if available, falls back to CPU
- First run downloads models automatically (requires time and storage)
- FFmpeg path is auto-detected from `ffmpeg-7.1.1-full_build-shared/bin` if present in project root
- Default Whisper model: `XA9/Belle-faster-whisper-large-v3-zh-punct` (Chinese-focused)
- Diarization model: `pyannote/speaker-diarization-community-1`
- Frontend uses Vite's proxy for seamless API integration during development
- Both applications support hot reloading during development
- Task queue processes one task at a time; multiple submissions will queue
- Traditional Chinese conversion using OpenCC (s2twp)

## Important Task Processing Details

### Task Processor ([remote_server/task_processor.py](remote_server/task_processor.py))

The `TaskProcessor` class handles:
- **Model Management**: Singleton pattern for Whisper and Pyannote models to prevent multiple instances
- **Dynamic Model Loading**: Supports switching between different Whisper models
- **Cancellation Checks**: Regularly checks database for task cancellation during processing
- **Progress Updates**: Updates database with progress percentage and current stage
- **Audio Conversion**: Uses FFmpeg for format conversion and time-based trimming
- **Memory Management**: Explicitly clears CUDA cache when switching models

### API Server ([remote_server/api.py](remote_server/api.py))

The FastAPI application uses:
- **Async Queue System**: `asyncio.Queue` for task management
- **SSE Streaming**: Server-Sent Events for real-time progress updates
- **Client IP Tracking**: Uses `X-Forwarded-For` header for task history
- **Lifespan Management**: Starts queue processor on application startup
- **CORS Middleware**: Configured for cross-origin requests

### Frontend State Management

- **Task IDs**: Stored in localStorage via `taskStorage.ts` for persistence across sessions
- **SSE Connections**: EventSource API for real-time updates, auto-reconnect on failure
- **Batch Operations**: Supports batch querying multiple task statuses for history view
- **Admin Mode**: Separate view mode with token-based authentication
