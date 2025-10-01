"""
API 使用範例腳本
示範如何使用 Whisper 語音轉文字 API
"""
import requests
import time
import sys

# API 基礎 URL
BASE_URL = "http://localhost:8000"


def upload_audio(file_path: str, enable_diarization: bool = True):
    """
    上傳音訊檔案並提交轉錄任務
    
    Args:
        file_path: 音訊檔案路徑
        enable_diarization: 是否啟用語者分離
    
    Returns:
        task_id: 任務ID
    """
    print(f"正在上傳檔案: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        params = {'enable_diarization': enable_diarization}
        
        response = requests.post(
            f"{BASE_URL}/api/tasks",
            files=files,
            params=params
        )
    
    if response.status_code == 200:
        data = response.json()
        task_id = data['task_id']
        print(f"✓ 任務已提交")
        print(f"  任務ID: {task_id}")
        print(f"  佇列位置: {data['queue_position']}")
        return task_id
    else:
        print(f"✗ 上傳失敗: {response.text}")
        return None


def check_status(task_id: str):
    """
    查詢任務狀態
    
    Args:
        task_id: 任務ID
    
    Returns:
        dict: 任務狀態資訊
    """
    response = requests.get(f"{BASE_URL}/api/tasks/{task_id}")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"✗ 查詢失敗: {response.text}")
        return None


def wait_for_completion(task_id: str, check_interval: float = 2.0):
    """
    等待任務完成並顯示進度
    
    Args:
        task_id: 任務ID
        check_interval: 檢查間隔（秒）
    """
    print("\n正在處理中...")
    print("-" * 60)
    
    last_progress = -1
    
    while True:
        status = check_status(task_id)
        
        if not status:
            print("✗ 無法獲取任務狀態")
            break
        
        current_status = status['status']
        progress = status['progress']
        stage = status['current_stage']
        
        # 只在進度改變時顯示
        if progress != last_progress:
            print(f"[{progress:5.1f}%] {stage or current_status}")
            last_progress = progress
        
        if current_status == 'completed':
            print("-" * 60)
            print("✓ 任務完成！")
            break
        elif current_status == 'failed':
            print("-" * 60)
            print(f"✗ 任務失敗: {status.get('error_message', '未知錯誤')}")
            break
        elif current_status == 'canceled':
            print("-" * 60)
            print("⚠ 任務已取消")
            break
        
        time.sleep(check_interval)


def download_result(task_id: str, file_type: str = "transcript", output_path: str = None):
    """
    下載轉錄結果
    
    Args:
        task_id: 任務ID
        file_type: 檔案類型 (transcript 或 raw)
        output_path: 輸出檔案路徑
    """
    if not output_path:
        output_path = f"{task_id}_{file_type}.txt"
    
    response = requests.get(
        f"{BASE_URL}/api/tasks/{task_id}/download",
        params={'file_type': file_type}
    )
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ 結果已下載至: {output_path}")
        return True
    else:
        print(f"✗ 下載失敗: {response.text}")
        return False


def get_my_tasks(limit: int = 10):
    """
    查詢我的任務歷史
    
    Args:
        limit: 返回數量
    """
    response = requests.get(
        f"{BASE_URL}/api/my-tasks",
        params={'limit': limit}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n我的任務歷史 (IP: {data['client_ip']})")
        print("=" * 60)
        
        for task in data['tasks']:
            print(f"\n任務ID: {task['task_id']}")
            print(f"檔案名: {task['filename']}")
            print(f"狀態: {task['status']} ({task['progress']:.1f}%)")
            print(f"提交時間: {task['created_at']}")
            if task['completed_at']:
                print(f"完成時間: {task['completed_at']}")
            print(f"有結果: {'是' if task['has_result'] else '否'}")
    else:
        print(f"✗ 查詢失敗: {response.text}")


def cancel_task(task_id: str):
    """
    取消任務
    
    Args:
        task_id: 任務ID
    """
    response = requests.delete(f"{BASE_URL}/api/tasks/{task_id}")
    
    if response.status_code == 200:
        print(f"✓ 任務已取消")
        return True
    else:
        print(f"✗ 取消失敗: {response.text}")
        return False


def main():
    """主函數 - 完整工作流程範例"""
    
    print("=" * 60)
    print("Whisper 語音轉文字 API - 使用範例")
    print("=" * 60)
    
    # 檢查參數
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python example_usage.py <音訊檔案路徑> [enable_diarization]")
        print("\n範例:")
        print("  python example_usage.py audio.mp3")
        print("  python example_usage.py audio.mp3 false")
        print("\n其他功能:")
        print("  python example_usage.py --history  # 查看任務歷史")
        return
    
    # 查看歷史記錄
    if sys.argv[1] == "--history":
        get_my_tasks()
        return
    
    audio_file = sys.argv[1]
    enable_diarization = True if len(sys.argv) < 3 else sys.argv[2].lower() == 'true'
    
    # 1. 上傳音訊檔案
    print(f"\n步驟 1: 上傳音訊檔案")
    task_id = upload_audio(audio_file, enable_diarization)
    
    if not task_id:
        return
    
    # 2. 等待處理完成
    print(f"\n步驟 2: 等待處理完成")
    wait_for_completion(task_id)
    
    # 3. 查詢最終狀態
    print(f"\n步驟 3: 查詢最終狀態")
    final_status = check_status(task_id)
    
    if final_status and final_status['status'] == 'completed':
        # 4. 下載結果
        print(f"\n步驟 4: 下載轉錄結果")
        
        if enable_diarization:
            download_result(task_id, "transcript", f"result_with_speakers.txt")
        else:
            download_result(task_id, "transcript", f"result.txt")
        
        download_result(task_id, "raw", f"result_raw.txt")
        
        print("\n" + "=" * 60)
        print("✓ 所有步驟完成！")
        print("=" * 60)


if __name__ == "__main__":
    main()

