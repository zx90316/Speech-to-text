# -*- coding: utf-8 -*-
"""
Backend V2 - Qwen ASR + 語者分離後端服務
FastAPI 應用主檔案
"""
import sys
from pathlib import Path

# 確保模組可以被導入
sys.path.insert(0, str(Path(__file__).parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_PREFIX, CORS_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    print("🚀 Backend V2 啟動中...")
    print(f"   API 前綴: {API_PREFIX}")
    
    # 在這裡進行啟動時的初始化
    yield
    
    # 在這裡進行關閉時的清理
    print("🛑 Backend V2 關閉中...")


# 建立 FastAPI 應用
app = FastAPI(
    title="Speech-to-Text Backend V2",
    description="Qwen ASR + 語者分離後端服務",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
from routes.health import router as health_router
from routes.tasks import router as tasks_router
from routes.email import router as email_router

app.include_router(health_router, prefix=API_PREFIX)
app.include_router(tasks_router, prefix=API_PREFIX)
app.include_router(email_router, prefix=API_PREFIX)


@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "Speech-to-Text Backend V2",
        "version": "2.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
    )
