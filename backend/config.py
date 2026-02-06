# -*- coding: utf-8 -*-
"""
Backend V2 配置管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 路徑配置
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

# 確保目錄存在
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Hugging Face Token (語者分離必需)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# 模型配置
QWEN_ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B": "ASR 主模型（效能最佳）",
    "Qwen/Qwen3-ASR-0.6B": "ASR 輕量模型（速度較快）",
}
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

# 語者分離模型
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# 音訊處理配置
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# API 配置
API_PREFIX = "/api"
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
