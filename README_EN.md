# Whisper Speech-to-Text API Service

English | [繁體中文](README.md)

A complete speech-to-text API service based on Faster-Whisper and Pyannote, featuring a modern React frontend and audio preprocessing capabilities.

## ✨ Features

### Core Features
- 🎯 **High-Quality Transcription**: Using Belle-Whisper-Large-V3 Chinese model
- 👥 **Speaker Diarization**: Optional multi-speaker recognition with configurable speaker count
- 📊 **Real-time Progress**: Live progress updates with partial results via SSE (Server-Sent Events)
- 🎨 **Confidence Visualization**: Word-level confidence scores with interactive HTML visualization
- 📁 **Task Management**: Complete task lifecycle management
- 🔍 **History Tracking**: Query past tasks by IP address
- 💾 **Result Storage**: Automatic organization and storage of transcription results
- 🚀 **Async Processing**: Background task queue without blocking requests

### Audio Preprocessing
- 🎵 **Noise Reduction**: FFT denoising algorithm
- 🔊 **Volume Normalization**: Peak or LUFS normalization
- 🔇 **Silence Removal**: Configurable threshold silence detection
- 🎤 **Vocal Enhancement**: Speech frequency band enhancement
- 🔔 **Echo Removal**: Remove clicks and echoes
- 🎛️ **Frequency Equalization**: 3-band EQ adjustment
- 🎚️ **Dynamic Compression**: Dynamic range compression
- ⚡ **Speed Adjustment**: Speed change while maintaining pitch
- 🎹 **Pitch Shift**: Semitone adjustment
- 🔄 **Sample Rate Conversion**: Support for multiple sample rates

### Frontend Interface
- 💻 **Modern UI**: React + TypeScript + Vite
- 📤 **Drag & Drop Upload**: Support for drag-and-drop file upload
- 🎬 **Time Range Selection**: Select audio segments for transcription
- 📊 **Live Preview**: Display partial results during processing
- 📜 **Task History**: Batch management and queries
- 🔧 **Advanced Parameters**: VAD sensitivity, compute types, etc.
- 🎯 **Smart Audio Player**: Automatic player selection based on file size
- 🛠️ **Admin Interface**: Admin dashboard with batch operations

## 🚀 Quick Start

### 1. Requirements

**Backend:**
- Python 3.8+
- CUDA (recommended for GPU acceleration)
- FFmpeg (audio processing)
- Hugging Face Token (for speaker diarization model)

**Frontend:**
- Node.js 16+
- npm or yarn

### 2. Backend Setup

#### Install Dependencies

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```env
HUGGINGFACE_TOKEN=your_huggingface_token_here
ADMIN_TOKEN=your_admin_token_here
```

Get Hugging Face Token: Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)

#### Start Backend Service

```bash
cd backend
python api.py
```

The backend API will start at `http://localhost:8000` with documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start at `http://localhost:5173` and automatically proxy API requests to the backend.

#### Production Build

```bash
cd frontend
npm run build
npm run preview
```

## 📖 API Documentation

### Complete API Docs

After starting the service, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main Endpoints

#### 1. Submit Transcription Task

```bash
POST /api/tasks
```

**Basic Parameters:**
- `file`: Audio file (supports mp3, wav, m4a, flac)
- `enable_diarization`: Enable speaker diarization (default: true)
- `start_time`: Start time in seconds (optional)
- `end_time`: End time in seconds (optional)
- `language`: Language code (e.g., zh, en, ja, optional)
- `task`: Task type (transcribe or translate)
- `model`: Whisper model name
- `compute_type`: Compute type (float32, int8, float16)

**Advanced Parameters:**
- `vad_onset`: VAD speech detection sensitivity (0-1, default 0.5)
- `vad_offset`: VAD speech end threshold (0-1, default 0.363)
- `min_speakers`: Minimum number of speakers (optional)
- `max_speakers`: Maximum number of speakers (optional)
- `enable_confidence_score`: Enable confidence scores (default false)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/tasks?enable_diarization=true" \
  -F "file=@audio.mp3"
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "queue_position": 1,
  "message": "Task submitted, queued for processing"
}
```

#### 2. Query Task Status

```bash
GET /api/tasks/{task_id}
```

**Example:**
```bash
curl "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "audio.mp3",
  "status": "processing",
  "progress": 45.5,
  "current_stage": "Speech recognition (processed 25 segments)",
  "queue_position": 0,
  "enable_diarization": true,
  "created_at": "2025-10-01T10:30:00",
  "started_at": "2025-10-01T10:31:00",
  "completed_at": null,
  "error_message": null,
  "has_result": false
}
```

#### 3. Real-time Progress Stream (SSE)

```bash
GET /api/tasks/{task_id}/stream
```

**Example (JavaScript):**
```javascript
const eventSource = new EventSource(`http://localhost:8000/api/tasks/${taskId}/stream`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress, '%');
  console.log('Status:', data.current_stage);

  if (data.status === 'completed') {
    console.log('Task completed!');
    eventSource.close();
  }
};

eventSource.onerror = (error) => {
  console.error('Connection error:', error);
  eventSource.close();
};
```

**Stream Data Format:**
```json
{
  "status": "processing",
  "progress": 45.5,
  "current_stage": "Speech recognition (processed 25 segments)",
  "queue_position": 0,
  "error_message": null,
  "timestamp": "2025-10-01T10:32:15.123456",
  "partial_result": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Hello everyone",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

#### 4. Cancel Task

```bash
DELETE /api/tasks/{task_id}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000"
```

#### 5. Download Transcription Results

```bash
GET /api/tasks/{task_id}/download?file_type=transcript
```

**Parameters:**
- `file_type`:
  - `transcript`: Final result (with speaker info if enabled)
  - `raw`: Raw ASR transcription result
  - `confidence_html`: Confidence visualization HTML (requires enable_confidence_score)

**Example:**
```bash
# Download final result
curl -O "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000/download?file_type=transcript"

# Download raw ASR result
curl -O "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000/download?file_type=raw"

# Download confidence visualization HTML
curl -O "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000/download?file_type=confidence_html"
```

#### 6. Query My Task History

```bash
GET /api/my-tasks?limit=50
```

**Example:**
```bash
curl "http://localhost:8000/api/my-tasks?limit=10"
```

**Response:**
```json
{
  "client_ip": "192.168.1.100",
  "total": 10,
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "audio1.mp3",
      "status": "completed",
      "progress": 100.0,
      "enable_diarization": true,
      "created_at": "2025-10-01T10:30:00",
      "completed_at": "2025-10-01T10:35:00",
      "has_result": true
    }
  ]
}
```

#### 7. Service Statistics

```bash
GET /api/stats
```

**Response:**
```json
{
  "queue_size": 3,
  "processing_count": 1,
  "is_processing": true,
  "total_waiting": 4
}
```

### Audio Preprocessing API

#### 8. Submit Preprocessing Task

```bash
POST /api/preprocess
```

**Parameters:**
- `file`: Audio file
- `config`: Preprocessing configuration in JSON format

**Configuration Example:**
```json
{
  "enable_denoise": true,
  "denoise_strength": 0.5,
  "enable_normalize": true,
  "normalize_type": "peak",
  "target_level": -3.0,
  "enable_silence_removal": true,
  "silence_threshold": -50.0,
  "enable_vocal_enhancement": true,
  "enhancement_strength": 0.5,
  "enable_mono": true,
  "enable_resample": true,
  "target_sample_rate": 16000
}
```

#### 9. Query Preprocessing Status

```bash
GET /api/preprocess/{preprocess_id}
```

#### 10. Download Preprocessed Audio

```bash
GET /api/preprocess/{preprocess_id}/download?file_type=processed
```

**Parameters:**
- `file_type`: `original` or `processed`

## 📁 Directory Structure

```
Speech-to-text/
├── backend/              # Backend service
│   ├── api.py                 # FastAPI main application
│   ├── database.py            # SQLite database management
│   ├── task_processor.py      # Transcription task processor
│   ├── audio_preprocessor.py  # Audio preprocessing engine
│   ├── preprocess_processor.py # Preprocessing task processor
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Environment variables (create manually)
│   ├── tasks.db               # SQLite database (auto-created)
│   ├── uploads/               # Uploaded audio files
│   │   └── {task_id}/
│   ├── result/                # Transcription results
│   │   └── {task_id}/
│   │       ├── transcript_raw.txt
│   │       ├── transcript_with_speakers.txt
│   │       └── confidence_visualization.html
│   └── preprocessed/          # Preprocessed files
│       └── {preprocess_id}/
│           ├── original_*.mp3
│           └── processed_*.wav
│
├── frontend/                   # Frontend application
│   ├── src/
│   │   ├── ui/
│   │   │   └── App.tsx        # Main application component
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── api.ts             # API client
│   │   └── types.ts           # TypeScript type definitions
│   ├── package.json
│   └── vite.config.ts
│
├── CLAUDE.md                   # Claude Code guide
├── README.md                   # Chinese README
└── README_EN.md               # This file
```

## 🔄 Task Status Flow

```
pending → processing → completed
                    ↘ failed
                    ↘ canceled
```

- **pending**: Task submitted, waiting for processing
- **processing**: Currently processing
- **completed**: Processing completed
- **failed**: Processing failed
- **canceled**: Task canceled

## 🎯 Processing Stages

1. **Load Models** (0-5%): Load Whisper and optional Pyannote models
2. **Convert Audio Format** (20-25%): Convert to appropriate format
3. **Speech Recognition (ASR)** (30-60%): Transcribe using Faster-Whisper
4. **Speaker Diarization** (70-85%, if enabled): Separate speakers
5. **Integrate Speaker Info** (85-95%, if enabled): Combine transcription with speakers
6. **Completion** (100%): Save results and update database

## 💡 Usage Tips

### Transcription
1. **First Use**: Models will be downloaded automatically on first run (requires time and storage)
2. **GPU Acceleration**: Highly recommended to use CUDA acceleration, 10x+ faster
3. **Memory Requirements**: At least 8GB RAM recommended, more required for diarization
4. **Compute Type Selection**:
   - `float32`: Highest quality, large memory requirement, for GPUs with sufficient VRAM
   - `int8`: Balanced option, medium memory requirement, good quality
   - `float16`: Small memory requirement, suitable for GPUs with limited VRAM
5. **Concurrent Processing**: Currently single-task processing, multiple tasks will queue

### Audio Preprocessing
1. **Audio Format**: Recommend using 16kHz mono WAV for best transcription results
2. **Preprocessing Order**: Recommend preprocessing audio before transcription to improve accuracy
3. **Denoise Strength**: Start with 0.3-0.5, too high may affect audio quality
4. **Silence Removal**: Threshold should be -50dB or higher to avoid removing normal speech
5. **A/B Comparison**: Use frontend interface for real-time comparison of original vs. processed audio

### Frontend Usage
1. **Large Files**: System automatically selects native audio player for large files
2. **Time Range Selection**: Select specific segments in audio player for transcription
3. **Confidence Visualization**: When enabled, download interactive HTML to view word-level confidence scores

## 🐛 Troubleshooting

### Model Download Failure
```bash
# Set Hugging Face mirror (for users in China)
export HF_ENDPOINT=https://hf-mirror.com
```

### FFmpeg Not Found
Ensure FFmpeg is correctly installed and added to system PATH, or place the `ffmpeg-7.1.1-full_build-shared` folder in the project root directory

### CUDA Out of Memory
Select lower compute type in frontend:
- Choose `int8` or `float16` compute type
- Disable speaker diarization
- Use smaller Whisper model

### Frontend Cannot Connect to Backend
Verify:
1. Backend service is running at `http://localhost:8000`
2. Frontend dev server started with `npm run dev`
3. Vite proxy configuration is correct (pre-configured)

### Audio Quality Degraded After Preprocessing
Adjust parameters:
- Reduce denoise strength (denoise_strength)
- Adjust silence removal threshold (silence_threshold)
- Disable unnecessary filters

## 🛠️ Technology Stack

### Backend Technologies
- **FastAPI**: Modern Python web framework
- **Faster-Whisper**: High-performance Whisper inference engine
- **Pyannote Audio**: Speaker diarization model
- **FFmpeg**: Audio processing tool
- **SQLite**: Task database
- **Server-Sent Events**: Real-time progress updates

### Frontend Technologies
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Vite**: Fast build tool
- **Axios**: HTTP client
- **Lucide React**: Icon library

### Main Models
- **Whisper**: CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32
- **Diarization**: pyannote/speaker-diarization-community-1
- **Text Conversion**: OpenCC (s2twp)

## 📝 License

This project is for learning and research purposes only.

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [BELLE-2/Belle-whisper-large-v3-zh-punct](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh-punct)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [Vite](https://vitejs.dev/)
