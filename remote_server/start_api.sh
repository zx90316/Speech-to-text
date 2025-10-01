#!/bin/bash

echo "================================================"
echo "Whisper 語音轉文字 API 服務啟動中..."
echo "================================================"
echo ""

cd "$(dirname "$0")"

# 檢查虛擬環境是否存在
if [ ! -d ".venv" ]; then
    echo "未找到虛擬環境，正在創建..."
    python3 -m venv .venv
    echo "虛擬環境創建完成"
    echo ""
fi

# 啟動虛擬環境
source .venv/bin/activate

# 安裝或更新依賴
echo "檢查並安裝依賴套件..."
pip install -r requirements.txt
echo ""

# 檢查 .env 檔案
if [ ! -f ".env" ]; then
    echo "警告: 未找到 .env 檔案"
    echo "請創建 .env 檔案並設定 HUGGINGFACE_TOKEN"
    echo ""
    echo "範例:"
    echo "HUGGINGFACE_TOKEN=your_token_here"
    echo ""
    read -p "按 Enter 繼續..."
fi

# 啟動 API 服務
echo "================================================"
echo "正在啟動 API 服務..."
echo "API 地址: http://localhost:8000"
echo "API 文檔: http://localhost:8000/docs"
echo "================================================"
echo ""

python api.py

