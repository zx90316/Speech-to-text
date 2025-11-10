# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Whisper Speech-to-Text API service with **email-based verification** and a modern React frontend. The project uses **memory-based storage** instead of databases for better security and simplicity. The project consists of:

- **Backend (remote_server/)**: FastAPI-based service for speech-to-text processing using Faster-Whisper and Pyannote, with memory storage and email delivery
- **Frontend (frontend/)**: React + TypeScript + Vite application with email verification, file upload, and progress monitoring

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

The backend requires a `.env` file in `remote_server/` with SMTP configuration for email service:

```env
HUGGINGFACE_TOKEN=your_huggingface_token_here
ADMIN_TOKEN=your_admin_token_here

# SMTP Email Service Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
```

Reference [.env.txt](.env.txt) for the template.

## Architecture Overview

### Backend Architecture (remote_server/)

- **api.py**: FastAPI main application with all API endpoints (email verification, task submission, progress streaming)
- **memory_storage.py**: In-memory task storage using OrderedDict (no database required)
- **email_service.py**: SMTP-based email service for verification codes and result delivery
- **task_processor.py**: Core speech processing logic using Faster-Whisper and Pyannote

The backend uses a **memory-based** architecture where:
1. Email verification required before task submission (6-digit code, 5-minute expiry)
2. Verified emails have 24-hour validity
3. Tasks stored in memory (OrderedDict) for fast access
4. Files temporarily stored during processing in `uploads/{task_id}/` and `result/{task_id}/`
5. Processing happens asynchronously with real-time progress via SSE
6. Results delivered via email upon completion
7. **All temporary files auto-deleted** after email sent or on failure
8. No persistent database - clean shutdown removes all task data

### Frontend Architecture (frontend/src/)

- **src/ui/App.tsx**: Main application component with view mode switching (main/admin)
- **src/components/**: Reusable React components
  - **EmailVerification.tsx**: Email verification UI with code input
  - **UploadSection.tsx**: File upload with drag-and-drop, model selection, advanced parameters
  - **TaskProgress.tsx**: Real-time progress display with SSE, partial result preview
  - **TaskHistory.tsx**: Historical task management with email-based query
  - **ServiceStats.tsx**: Service statistics display
  - **AudioPlayer.tsx**: Smart audio player selection based on file size
  - **SimpleAudioPlayer.tsx**: HTML5 audio player for regular files
  - **NativeAudioPlayer.tsx**: Native audio player for large files
- **src/pages/AdminPage.tsx**: Admin interface for viewing all tasks (with masked emails)
- **src/api.ts**: Axios-based API client for email verification and transcription tasks
- **src/types.ts**: TypeScript type definitions
- **src/utils/taskStorage.ts**: LocalStorage management for task IDs
- **src/utils/emailStorage.ts**: LocalStorage management for verified email (24-hour persistence)
- **src/styles/main.css**: Global styles

### Key Features

1. **Email Verification System**: 6-digit verification codes with 5-minute expiry, 24-hour validity after verification
2. **Persistent Email State**: Frontend stores verified email in localStorage, auto-unlocks both upload and history on page refresh
3. **Memory-Based Storage**: No database required - all task metadata stored in memory for faster access
4. **Email Delivery**: Results sent directly to user's email with transcript attachment and confidence visualization HTML
5. **Auto Cleanup**: All temporary files automatically deleted after email sent or on task failure
6. **Real-time Progress**: Uses Server-Sent Events (SSE) for live progress updates with partial results
7. **Speaker Diarization**: Optional multi-speaker recognition using Pyannote with configurable speaker count
8. **Confidence Score Visualization**: Word-level confidence scores with interactive HTML visualization (emailed as attachment)
9. **File Handling**: Supports MP3, WAV, M4A, FLAC formats with smart audio player selection
10. **Time Range Selection**: Can process specific segments of audio files
11. **Multiple Models**: Support for different Whisper models, languages, and compute types (float32, int8, float16)
12. **Advanced Parameters**: Configurable VAD sensitivity, speaker count, confidence scores
13. **Privacy Protection**: Admin view shows masked emails only (e.g., `ab***@domain.com`)

### Processing Pipeline

1. **Email Verification** (0%): User receives and verifies 6-digit code
2. **Model Loading** (0-5%): Load Whisper and optional Pyannote models
3. **Audio Conversion** (20-25%): Convert to appropriate format using FFmpeg
4. **Speech Recognition** (30-60%): Transcribe audio using Faster-Whisper
5. **Speaker Diarization** (70-85%): Optional speaker separation
6. **Integration** (85-95%): Combine transcription with speaker information
7. **Email Delivery** (95-100%): Send results to user's email with attachments
8. **Cleanup** (100%): Delete all temporary files

### Memory Storage Schema

In-memory OrderedDict storage in [remote_server/memory_storage.py](remote_server/memory_storage.py):

**Task Data Structure**:
- Task metadata (task_id, email, filename, status, progress, current_stage)
- Processing configuration (enable_diarization, start_time, end_time, language, task, model)
- Advanced parameters (vad_onset, vad_offset, min_speakers, max_speakers, enable_confidence_score, compute_type)
- Processing timestamps (created_at, started_at, completed_at)
- Error handling and partial results (stored as lists/dicts)
- Temporary file paths (upload_path, result_path) - cleaned after completion

**Email Verification Storage** (in memory):
- Verification codes with 5-minute expiry
- Verified status with 24-hour expiry after successful verification

### API Integration Points

#### Email Verification APIs
- **POST /api/email/send-verification**: Send verification code to email (6-digit, 5-minute expiry)
- **POST /api/email/verify-code**: Verify email with code (extends validity to 24 hours on success)

#### Transcription APIs
- **POST /api/tasks**: Submit new transcription tasks (requires verified email)
  - Parameters: email, file, enable_diarization, start_time, end_time, language, task, model, etc.
  - Returns: task_id, status, queue_position
- **GET /api/tasks/{id}**: Query task status
- **GET /api/tasks/{id}/stream**: SSE progress updates with partial results
- **DELETE /api/tasks/{id}**: Cancel tasks (with optional `?permanent=true` for deletion)
- **POST /api/tasks/batch**: Batch query multiple tasks
- **GET /api/my-tasks**: Get task history by email (requires `?email=xxx`)
- **GET /api/stats**: Service statistics (queue size, processing count)

#### Admin APIs
- **GET /api/admin/tasks**: Admin endpoint to view all tasks with masked emails (requires ADMIN_TOKEN)
- **GET /api/admin/stats**: Admin system statistics and status counts

See full API documentation at `http://localhost:8000/docs` when server is running.

### Development Notes

- **No Database**: All task data stored in memory (OrderedDict) - server restart clears all tasks
- **Auto Cleanup**: Temporary files in `uploads/` and `result/` auto-deleted on startup and after tasks complete
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
- **Cancellation Checks**: Regularly checks memory storage for task cancellation during processing
- **Progress Updates**: Updates memory storage with progress percentage and current stage
- **Audio Conversion**: Uses FFmpeg for format conversion and time-based trimming
- **Memory Management**: Explicitly clears CUDA cache when switching models or unloading diarization
- **Confidence Score Generation**: Creates word-level confidence visualization HTML with color-coded confidence levels
- **Email Delivery**: Sends results via email with transcript.txt and optional confidence_report.html attachments
- **Auto Cleanup**: Deletes all temporary files after successful email delivery

### API Server ([remote_server/api.py](remote_server/api.py))

The FastAPI application uses:
- **Async Queue System**: `asyncio.Queue` for transcription task processing
- **SSE Streaming**: Server-Sent Events for real-time progress updates
- **Email Verification**: Required before task submission, validates 24-hour validity
- **Lifespan Management**: Cleans up temporary files on startup, starts queue processor
- **CORS Middleware**: Configured for cross-origin requests
- **Memory Storage**: Uses `memory_manager` singleton for all task data

### Email Service ([remote_server/email_service.py](remote_server/email_service.py))

Handles all email operations:
- **Verification Codes**: Generates 6-digit codes with 5-minute expiry
- **Extended Validity**: Successful verification extends to 24-hour validity
- **SMTP Integration**: Uses environment variables for SMTP configuration
- **Result Delivery**: Sends email with:
  - Task completion notification
  - Transcript.txt attachment
  - Optional confidence_report.html attachment (if enabled)
  - Preview of first 500 characters in email body

### Memory Storage ([remote_server/memory_storage.py](remote_server/memory_storage.py))

In-memory task management:
- **OrderedDict**: Maintains task insertion order for queue management
- **Thread-Safe**: Uses RLock for concurrent access
- **Email-Based Queries**: Can retrieve tasks by email address
- **Auto Cleanup**: Provides methods to cleanup temporary files
- **Privacy Protection**: Admin views return masked emails
- **No Persistence**: All data lost on server restart (by design)

### Frontend State Management

- **Verified Email**: Stored in localStorage via `emailStorage.ts` for 24-hour persistence across page refreshes
- **Task IDs**: Stored in localStorage via `taskStorage.ts` for tracking user's own tasks
- **SSE Connections**: EventSource API for real-time updates, auto-reconnect on failure
- **Batch Operations**: Supports batch querying multiple task statuses for history view
- **Shared Verification State**: Both UploadSection and TaskHistory share same verified email from localStorage
- **Admin Mode**: Separate view mode with token-based authentication, shows masked emails only

---

## 🔒 Security Features (SSDLC Compliant - v2.1.0)

### Security Modules (New in v2.1.0)

#### 1. Security Logger ([remote_server/security_logger.py](remote_server/security_logger.py))

Comprehensive logging system with 5 log types:
- **auth.log** (180 days retention): Authentication attempts, verification codes, session management
- **operation.log** (180 days): Task creation, file uploads, task completion
- **security.log** (365 days): Security events, rate limiting, unauthorized access
- **error.log** (180 days): System errors, processing failures
- **audit.log** (1825 days / 5 years): Personal data access, data deletion (GDPR compliant)

All logs include: event_type, timestamp, ip_address, user_id, action, result, details

#### 2. Input Validator ([remote_server/input_validator.py](remote_server/input_validator.py))

Validates all user inputs to prevent attacks:
- **Email validation**: RFC 5322 standard, dangerous character detection
- **File validation**: Size limits (500MB), type whitelist, magic number verification
- **Filename security**: Path traversal prevention, null byte injection protection
- **Parameter validation**: Time ranges, language codes, model names, VAD parameters
- **Task ID validation**: UUID format verification

#### 3. Rate Limiter ([remote_server/rate_limiter.py](remote_server/rate_limiter.py))

Prevents brute force attacks and DoS:
- **IP-based limiting**: 100 requests/minute (general endpoints)
- **Email verification**: 5 codes/hour per email
- **Task creation**: 10 tasks/hour per email
- **Brute force protection**: 5 failed attempts → 30-minute email ban, 10-minute IP ban
- **Blacklist management**: Automatic temporary bans with expiry

#### 4. Crypto Utils ([remote_server/crypto_utils.py](remote_server/crypto_utils.py))

Encryption and data protection:
- **Password hashing**: PBKDF2-SHA256 (100,000 iterations)
- **Data encryption**: Fernet (AES-128-CBC + HMAC-SHA256)
- **Email hashing**: SHA-256 + salt for storage
- **Secure deletion**: 3-pass overwrite before file deletion
- **Data masking**: Email and IP address masking for logs
- **Constant-time comparison**: Prevents timing attacks

### API Security Enhancements

#### HTTP Security Headers

All responses include security headers:
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

#### CORS Configuration

- **Whitelist mode**: Only configured origins allowed (via `ALLOWED_ORIGINS` env var)
- **Method restrictions**: Only GET, POST, DELETE
- **Header restrictions**: Only Content-Type, Authorization
- **Credentials support**: With secure configuration

#### Trusted Host Protection

- Prevents Host Header attacks
- Configurable via `TRUSTED_HOSTS` environment variable

#### Enhanced API Endpoints

**Email Verification** (`/api/email/send-verification`, `/api/email/verify-code`):
- Input validation (email format, verification code format)
- Rate limiting (5 codes/hour per email)
- Blacklist checking (IP and email)
- Brute force protection (5 attempts → ban)
- Detailed logging (all attempts, successes, failures)
- Remaining attempts display

**Admin APIs** (`/api/admin/*`):
- Token-based authentication (ADMIN_TOKEN from env)
- Token validation (min 16 chars, max 256 chars)
- Masked email display in responses
- Rate limiting (20 requests/minute)
- Admin action logging

### Environment Variables (.env)

Security-related configuration:

```env
# Security (Required)
ADMIN_TOKEN=<min 32-char random string>
EMAIL_HASH_SALT=<min 32-char random string>
ENCRYPTION_KEY=<auto-generated on first run>

# CORS & Host Security
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
TRUSTED_HOSTS=localhost,127.0.0.1

# Features
ENABLE_DOCS=true  # Set to false in production
```

See [remote_server/.env.example](remote_server/.env.example) for full configuration template.

### SSDLC Compliance

This project complies with **45 SSDLC (Secure Software Development Lifecycle) requirements**:

- ✅ **30 items fully compliant** (66.7%)
- ⚠️ **11 items recommended/partial** (24.4%)
- ❌ **2 items not applicable** (4.4% - non-core system, single deployment)
- ➖ **2 items not applicable** (4.4% - no traditional database, test data)

**Overall compliance rate: 91.1%**

See detailed compliance documentation:
- [SECURITY.md](SECURITY.md) - Complete security documentation
- [SSDLC-COMPLIANCE.md](SSDLC-COMPLIANCE.md) - Detailed compliance mapping
- [README-SSDLC.md](README-SSDLC.md) - Implementation summary
- [WI-GA215-附件一、SSDLC檢核表-已填寫.xlsx](WI-GA215-附件一、SSDLC檢核表-已填寫.xlsx) - Filled checklist

### Security Best Practices

When deploying this application:

1. **Use HTTPS/TLS 1.2+** in production (see [INSTALL.md](INSTALL.md))
2. **Set strong ADMIN_TOKEN** (min 32 chars, use `secrets.token_hex(32)`)
3. **Disable API docs** in production (`ENABLE_DOCS=false`)
4. **Configure proper CORS** (whitelist only necessary origins)
5. **Set up log monitoring** (check logs/security.log regularly)
6. **Regular security updates** (run `pip-audit` periodically)
7. **Backup logs** regularly (especially audit.log - 5-year retention required)
8. **Monitor rate limits** and adjust if needed
9. **Review admin access logs** in audit.log
10. **Follow deployment guide** in INSTALL.md for production setup

### Security Testing

Recommended security testing tools:

```bash
# Python dependency vulnerability check
pip install pip-audit
pip-audit

# Code security analysis
pip install bandit
bandit -r remote_server -ll

# OWASP ZAP for penetration testing
# https://www.zaproxy.org/

# SSL/TLS testing
# https://www.ssllabs.com/ssltest/
```

### Logging and Monitoring

**Log locations**: `remote_server/logs/`

**Key logs to monitor**:
- `security.log` - Security events (rate limits, unauthorized access)
- `auth.log` - Authentication attempts
- `error.log` - System errors
- `audit.log` - Personal data operations (GDPR compliance)

**Log retention**:
- General logs: 6 months (180 days)
- Personal data logs: 5 years (1825 days)

See [SECURITY.md](SECURITY.md) for complete logging documentation.
