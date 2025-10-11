# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Whisper Speech-to-Text API service with a modern React frontend and audio preprocessing capabilities. The project consists of:

- **Backend (remote_server/)**: FastAPI-based service for speech-to-text processing using Faster-Whisper and Pyannote
- **Audio Preprocessing (remote_server/audio_preprocessor.py)**: FFmpeg-based audio enhancement engine with noise reduction, normalization, vocal enhancement, and more
- **Frontend (frontend/)**: React + TypeScript + Vite application for uploading audio, preprocessing, and monitoring progress

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

- **api.py**: FastAPI main application with all API endpoints (transcription + preprocessing)
- **database.py**: SQLite database management for task tracking (tasks and preprocess_tasks tables)
- **task_processor.py**: Core speech processing logic using Faster-Whisper and Pyannote
- **audio_preprocessor.py**: Audio preprocessing engine with FFmpeg-based filters
- **preprocess_processor.py**: Preprocessing task queue processor
- **whisper_api.py**: Additional Whisper-related utilities

The backend uses an asynchronous task queue system where:
1. Tasks are submitted via `/api/tasks` endpoint
2. Files are stored in `uploads/{task_id}/`
3. Processing happens asynchronously with real-time progress via SSE
4. Results are stored in `result/{task_id}/`

Audio preprocessing flow:
1. Upload audio via `/api/preprocess` with configuration
2. Files stored in `preprocessed/{preprocess_id}/`
3. Processing applies FFmpeg filters (denoise, normalize, EQ, etc.)
4. Download processed audio or use for transcription

### Frontend Architecture (frontend/src/)

- **src/ui/App.tsx**: Main application component with view mode switching (main/admin/preprocess)
- **src/components/**: Reusable React components
  - **UploadSection.tsx**: File upload with drag-and-drop, model selection, advanced parameters
  - **TaskProgress.tsx**: Real-time progress display with SSE, partial result preview
  - **TaskHistory.tsx**: Historical task management with batch operations
  - **ServiceStats.tsx**: Service statistics display
  - **AudioPlayer.tsx**: Smart audio player selection based on file size
  - **SimpleAudioPlayer.tsx**: HTML5 audio player for regular files
  - **NativeAudioPlayer.tsx**: Native audio player for large files
  - **AudioPreprocessor.tsx**: Audio preprocessing interface with parameter controls and A/B comparison
  - **PreprocessHistory.tsx**: Preprocessing task history management
- **src/pages/AdminPage.tsx**: Admin interface for viewing all tasks and system stats
- **src/api.ts**: Axios-based API client (transcription + preprocessing methods)
- **src/types.ts**: TypeScript type definitions
- **src/utils/taskStorage.ts**: LocalStorage management for task IDs
- **src/styles/main.css**: Global styles

### Key Features

1. **Real-time Progress**: Uses Server-Sent Events (SSE) for live progress updates with partial results
2. **Speaker Diarization**: Optional multi-speaker recognition using Pyannote with configurable speaker count
3. **Confidence Score Visualization**: Word-level confidence scores with interactive HTML visualization
4. **Audio Preprocessing**: FFmpeg-based audio enhancement with 15+ processing options
   - Noise reduction (FFT denoising)
   - Volume normalization (peak/LUFS)
   - Silence removal
   - Vocal enhancement
   - Echo removal
   - EQ (3-band)
   - Dynamic range compression
   - Speed/pitch adjustment
   - A/B audio comparison
5. **Task Management**: Complete lifecycle management with SQLite persistence
6. **File Handling**: Supports MP3, WAV, M4A, FLAC formats with smart audio player selection
7. **Time Range Selection**: Can process specific segments of audio files
8. **Multiple Models**: Support for different Whisper models, languages, and compute types (float32, int8, float16)
9. **Advanced Parameters**: Configurable VAD sensitivity, beam size based on compute type
10. **Admin Dashboard**: Token-based admin interface with batch operations and cleanup utilities
11. **Proxy Configuration**: Vite proxy routes `/api` to backend at `localhost:8000`

### Processing Pipeline

1. **Model Loading** (0-5%): Load Whisper and optional Pyannote models
2. **Audio Conversion** (20-25%): Convert to appropriate format using FFmpeg
3. **Speech Recognition** (30-60%): Transcribe audio using Faster-Whisper
4. **Speaker Diarization** (70-85%): Optional speaker separation
5. **Integration** (85-95%): Combine transcription with speaker information
6. **Completion** (100%): Save results and update database

### Database Schema

SQLite database (`tasks.db`) in [remote_server/database.py](remote_server/database.py) with two main tables:

**tasks table** (transcription tasks):
- Task metadata (task_id, filename, status, progress, current_stage)
- Client IP for task history tracking
- Processing configuration (enable_diarization, start_time, end_time, language, task, model)
- Advanced parameters (vad_onset, vad_offset, min_speakers, max_speakers, enable_confidence_score, compute_type)
- Processing timestamps (created_at, started_at, completed_at)
- Error handling and partial results (stored as JSON)
- Result file paths and queue position

**preprocess_tasks table** (audio preprocessing tasks):
- Preprocessing metadata (preprocess_id, filename, status, progress, current_stage)
- Client IP for history tracking
- Configuration JSON with all preprocessing parameters
- File paths (original_path, processed_path)
- Processing timestamps and results (original_info, processed_info, filters_applied as JSON)

### API Integration Points

#### Transcription APIs
- **POST /api/tasks**: Submit new transcription tasks with optional parameters (enable_diarization, start_time, end_time, language, task, model)
- **GET /api/tasks/{id}**: Query task status
- **GET /api/tasks/{id}/stream**: SSE progress updates with partial results
- **GET /api/tasks/{id}/download**: Download results (transcript or raw)
- **DELETE /api/tasks/{id}**: Cancel tasks (or permanently delete with `?permanent=true`)
- **POST /api/tasks/batch**: Batch query multiple tasks
- **GET /api/my-tasks**: Client task history based on IP
- **GET /api/stats**: Service statistics (queue size, processing count)

#### Preprocessing APIs
- **POST /api/preprocess**: Submit audio preprocessing task with configuration JSON
- **GET /api/preprocess/{id}**: Query preprocessing task status
- **GET /api/preprocess/{id}/stream**: SSE progress updates for preprocessing
- **GET /api/preprocess/{id}/download**: Download preprocessed audio (original or processed)
- **GET /api/preprocess/{id}/info**: Get preprocessing details and audio info
- **GET /api/my-preprocess-tasks**: Client preprocessing task history
- **DELETE /api/preprocess/{id}**: Cancel/delete preprocessing task (with optional permanent flag)

#### Admin APIs
- **GET /api/admin/tasks**: Admin endpoint to view all tasks with pagination (requires ADMIN_TOKEN)
- **GET /api/admin/stats**: Admin system statistics and status counts
- **POST /api/admin/tasks/batch-delete**: Batch delete tasks and their files
- **POST /api/admin/cleanup**: Cleanup old completed tasks by age (in days)

See full API documentation at `http://localhost:8000/docs` when server is running.
See preprocessing usage guide at [PREPROCESS_USAGE.md](PREPROCESS_USAGE.md) for detailed examples.

### Development Notes

- Backend uses CUDA if available, falls back to CPU
- First run downloads models automatically (requires time and storage)
- FFmpeg path is auto-detected from `ffmpeg-7.1.1-full_build-shared/bin` if present in project root
- Default Whisper model: `CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32` (Chinese-focused)
- Diarization model: `pyannote/speaker-diarization-community-1`
- Frontend uses Vite's proxy for seamless API integration during development
- Both applications support hot reloading during development
- Task queue processes one task at a time; multiple submissions will queue
- Traditional Chinese conversion using OpenCC (s2twp)
- Confidence score visualization generates interactive HTML with word-level confidence coloring

## Important Task Processing Details

### Task Processor ([remote_server/task_processor.py](remote_server/task_processor.py))

The `TaskProcessor` class handles:
- **Model Management**: Singleton pattern for Whisper and Pyannote models to prevent multiple instances
- **Dynamic Model Loading**: Supports switching between different Whisper models with configurable compute types
- **Compute Type Optimization**: Automatically adjusts beam size based on compute type (float32=1, int8=10, float16=5)
- **Cancellation Checks**: Regularly checks database for task cancellation during processing
- **Progress Updates**: Updates database with progress percentage and current stage
- **Audio Conversion**: Uses FFmpeg for format conversion and time-based trimming
- **Memory Management**: Explicitly clears CUDA cache when switching models or unloading diarization
- **Confidence Score Generation**: Creates word-level confidence visualization HTML with color-coded confidence levels

### API Server ([remote_server/api.py](remote_server/api.py))

The FastAPI application uses:
- **Dual Async Queue System**: Separate `asyncio.Queue` instances for transcription and preprocessing tasks
- **SSE Streaming**: Server-Sent Events for real-time progress updates on both task types
- **Client IP Tracking**: Uses `X-Forwarded-For` header for task history
- **Lifespan Management**: Starts both queue processors on application startup
- **CORS Middleware**: Configured for cross-origin requests

### Preprocessing Processor ([remote_server/preprocess_processor.py](remote_server/preprocess_processor.py))

Handles audio preprocessing tasks:
- **Async Processing**: Runs FFmpeg filters asynchronously without blocking API
- **Progress Callbacks**: Updates database with progress percentage and current stage
- **Audio Info Extraction**: Gets audio metadata before and after processing
- **Filter Chain Application**: Applies configured FFmpeg filters in sequence
- **Error Handling**: Captures and reports FFmpeg errors to database

### Frontend State Management

- **Task IDs**: Stored in localStorage via `taskStorage.ts` for persistence across sessions
- **SSE Connections**: EventSource API for real-time updates, auto-reconnect on failure
- **Batch Operations**: Supports batch querying multiple task statuses for history view
- **Admin Mode**: Separate view mode with token-based authentication
