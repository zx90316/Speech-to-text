# -*- coding: utf-8 -*-
"""
上傳 CTranslate2 模型到 Hugging Face Hub
"""
from huggingface_hub import HfApi, login
import os
import sys

# 設定 UTF-8 輸出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 模型設定
MODEL_PATH = "Belle-faster-whisper-large-v3-zh-punct"
REPO_NAME = "Belle-whisper-large-v3-zh-punct-ct2-float32"  # 只填 repo 名稱，不含用戶名
REPO_TYPE = "model"

def upload_model():
    print("=" * 60)
    print("上傳 CTranslate2 模型到 Hugging Face Hub")
    print("=" * 60)
    
    # 步驟 1: 登入
    print("\n[1/3] 登入 Hugging Face...")
    print("請輸入你的 Hugging Face Token (需要有 write 權限):")
    print("Token 可從這裡取得: https://huggingface.co/settings/tokens")
    
    try:
        login()
        print("[OK] 登入成功!")
    except Exception as e:
        print(f"[FAIL] 登入失敗: {e}")
        return
    
    # 步驟 2: 初始化 API
    print("\n[2/3] 初始化 API...")
    api = HfApi()
    
    # 取得當前使用者名稱
    try:
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{REPO_NAME}"
        print(f"[OK] 使用者: {username}")
        print(f"[OK] Repository: {repo_id}")
    except Exception as e:
        print(f"[FAIL] 無法取得使用者資訊: {e}")
        return
    
    # 步驟 3: 建立並上傳
    print("\n[3/3] 建立 repository 並上傳模型...")
    print("[WARNING] 注意: 模型檔案很大 (約 5.8GB)，上傳可能需要一些時間...")
    
    try:
        # 建立 repository（如果不存在）
        api.create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            exist_ok=True,
            private=False  # 設為 True 如果想要私人 repo
        )
        print(f"[OK] Repository 已建立/已存在")
        
        # 上傳整個資料夾
        print("正在上傳檔案...")
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        
        print("\n" + "=" * 60)
        print("[OK] 上傳完成!")
        print("=" * 60)
        print(f"\n模型已上傳到: https://huggingface.co/{repo_id}")
        print(f"\n使用方式:")
        print(f"  from faster_whisper import WhisperModel")
        print(f'  model = WhisperModel("{repo_id}")')
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAIL] 上傳失敗: {e}")
        print("\n可能的原因:")
        print("  1. Token 權限不足 (需要 write 權限)")
        print("  2. Repository 名稱已被使用")
        print("  3. 網路連線問題")
        print("  4. 檔案過大超時")

if __name__ == "__main__":
    # 檢查模型資料夾是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] 錯誤: 找不到模型資料夾 '{MODEL_PATH}'")
        exit(1)
    
    # 檢查必要檔案
    required_files = ["model.bin", "config.json", "tokenizer.json"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(MODEL_PATH, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"[FAIL] 錯誤: 缺少必要檔案: {', '.join(missing_files)}")
        exit(1)
    
    print(f"[OK] 模型資料夾檢查通過")
    print(f"[OK] 找到模型檔案:")
    for file in os.listdir(MODEL_PATH):
        file_path = os.path.join(MODEL_PATH, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.1f} MB)")
    
    print()
    response = input("確定要上傳模型到 Hugging Face 嗎? (yes/no): ")
    if response.lower() in ['yes', 'y', '是']:
        upload_model()
    else:
        print("已取消上傳")

