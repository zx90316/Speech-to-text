# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Backend (FastAPI)
- Start backend: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` (from `backend/` directory)
- Install dependencies: `pip install -r backend/requirements.txt`

### Frontend (React + Vite)
- Development server: `npm run dev` (from `frontend/` directory)
- Build production: `npm run build`
- Install dependencies: `npm install`

### Remote Server (Whisper)
- Start remote inference: `python -m uvicorn remote_inference_server:app --host 0.0.0.0 --port 8001` (from `remote_server/` directory)
- Install dependencies: `pip install -r remote_server/requirements.txt`

### All-in-one Launcher
- Start all services: `python start.py`
- Backend only: `python start.py --no-frontend`
- With remote server: `python start.py --with-remote`
- Custom ports: `python start.py --backend-port 8000 --frontend-port 8002 --remote-port 8001`

## Architecture

This is a speech-to-text web application with the following structure:

### Backend (`backend/app/`)
- **FastAPI** server with WebSocket support for real-time progress updates
- **Celery** + **Redis** for background task processing (optional, controlled by `USE_CELERY` env var)
- Two transcription services:
  - **Vertex AI**: Google Cloud Vertex AI integration
  - **Remote LLM**: Self-hosted Whisper model on RTX GPU
- Core modules:
  - `main.py`: FastAPI app with REST and WebSocket endpoints
  - `tasks.py`: Celery background tasks
  - `storage.py`: In-memory task state management
  - `services/`: Transcription service implementations
  - `utils/`: Audio processing and formatting utilities

### Frontend (`frontend/src/`)
- **React** + **TypeScript** + **Vite**
- Real-time progress tracking via WebSocket
- File upload with audio format validation
- Multiple output formats (plain text, timestamped, SRT)

### Remote Server (`remote_server/`)
- **FastAPI** server running optimized Whisper models
- Designed for NVIDIA GPU inference (RTX 5060 Ti)
- Uses PyTorch with CUDA support

## Key API Endpoints

- `POST /api/v1/transcribe`: Submit audio file for transcription
  - Support for speaker diarization with parameters:
    - `enable_diarization`: Enable/disable speaker separation (remote_llm only)
    - `min_speakers`: Minimum number of speakers to detect
    - `max_speakers`: Maximum number of speakers to detect
- `WS /ws/v1/status/{task_id}`: WebSocket for real-time progress
- `GET /api/v1/result/{task_id}`: Download transcription results
- `POST /api/v1/cancel/{task_id}`: Cancel running task

## Environment Configuration

Environment variables are loaded from `.env` files (root and backend directories):

- `REDIS_URL`: Redis connection for Celery (default: redis://localhost:6379/0)
- `REMOTE_SERVER_URL`: URL for remote Whisper server (default: http://localhost:8001)
- `VERTEX_PROJECT`: Google Cloud project ID
- `VERTEX_LOCATION`: Vertex AI region (default: global)
- `VERTEX_GENAI_MODEL`: Vertex AI model name (default: gemini-2.5-flash-lite)
- `USE_CELERY`: Enable Celery background tasks (default: false)
- `CORS_ORIGINS`: Allowed CORS origins for frontend
- `VITE_BACKEND_URL`: Backend URL for frontend (default: http://localhost:8000)
- `HUGGINGFACE_TOKEN`: Required for speaker diarization model access

## Audio Processing

- Supported formats: WAV, MP3, M4A, FLAC
- Audio chunking for long files (configurable chunk length)
- Time range selection support (start_time/end_time parameters)
- FFmpeg integration for audio manipulation
- Traditional Chinese text conversion support (OpenCC)

## Task Management

Tasks are managed through `TaskStore` with the following states:
- `pending`: Task created but not started
- `processing`: Currently being processed
- `completed`: Successfully finished
- `failed`: Error occurred
- `canceled`: User canceled

Progress updates include:
- Percentage completion
- Partial transcription text
- Token usage statistics
- Segment data with timestamps

## Speaker Diarization

The system supports speaker diarization (speaker separation) through the remote server:

- **Technology**: Uses `pyannote.audio` with `pyannote/speaker-diarization-3.1` model
- **Requirements**: 
  - NVIDIA GPU for optimal performance
  - Hugging Face token with access to pyannote models
  - Additional dependencies: `pyannote.audio`, `speechbrain`, `librosa`, `soundfile`
- **Features**:
  - Automatic speaker detection or manual min/max speaker constraints
  - Speaker labels added to transcription segments
  - Available in all output formats (plain text, timestamped, SRT)
- **Frontend Integration**: Speaker diarization options are available only for remote_llm model

## Development Notes

- Backend supports both synchronous (BackgroundTasks) and asynchronous (Celery) processing modes
- WebSocket connections provide real-time updates during transcription
- Audio files are temporarily stored during processing
- The system handles both local and remote model inference
- Traditional Chinese text conversion is applied to transcription results
- Speaker diarization adds computational overhead but provides valuable speaker identification
- Segments now include optional `speaker` field for diarization results