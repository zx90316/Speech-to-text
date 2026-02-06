# -*- coding: utf-8 -*-
"""
健康檢查路由
"""
from fastapi import APIRouter
import torch

from models import HealthResponse, ModelsResponse
from config import QWEN_ASR_MODELS, DEFAULT_QWEN_MODEL
from asr_processor import QwenASRProcessor

router = APIRouter(tags=["健康檢查"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        gpu_available=torch.cuda.is_available(),
        qwen_available=QwenASRProcessor.is_available(),
    )


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """取得可用模型列表"""
    return ModelsResponse(
        asr_models=QWEN_ASR_MODELS,
        default_model=DEFAULT_QWEN_MODEL,
    )
