# -*- coding: utf-8 -*-
"""
Pydantic 資料模型
定義 API 請求/回應的資料結構
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """任務狀態"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskCreate(BaseModel):
    """建立任務請求"""
    enable_diarization: bool = Field(True, description="是否啟用語者分離")
    enable_timestamps: bool = Field(False, description="是否啟用時間戳輸出")
    language: Optional[str] = Field(None, description="語言代碼 (zh/en/ja 等)")
    model: str = Field("Qwen/Qwen3-ASR-1.7B", description="ASR 模型名稱")
    min_speakers: Optional[int] = Field(None, ge=1, le=20, description="最小語者數")
    max_speakers: Optional[int] = Field(None, ge=1, le=20, description="最大語者數")


class WordInfo(BaseModel):
    """詞級資訊"""
    word: str
    start: float
    end: float
    probability: float = 1.0


class TranscriptSegment(BaseModel):
    """轉錄片段"""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: Optional[List[WordInfo]] = None


class TaskResponse(BaseModel):
    """任務回應"""
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    filename: Optional[str] = None
    # 結果
    text: Optional[str] = None
    language: Optional[str] = None
    segments: Optional[List[TranscriptSegment]] = None
    has_diarization: bool = False


class TaskListResponse(BaseModel):
    """任務列表回應"""
    tasks: List[TaskResponse]
    total: int


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str = "ok"
    version: str = "2.0.0"
    gpu_available: bool = False
    qwen_available: bool = False


class ModelsResponse(BaseModel):
    """可用模型回應"""
    asr_models: Dict[str, str]
    default_model: str
