"""
記憶體任務存儲管理模組
使用記憶體存儲任務資訊
處理中的檔案暫存於實體檔案系統，完成後刪除
"""
import threading
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import OrderedDict
from security_logger import security_logger


class MemoryTaskManager:
    """記憶體任務管理類別，使用單例模式"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 使用 OrderedDict 保持插入順序
            self.tasks: OrderedDict[str, Dict[str, Any]] = OrderedDict()
            self.task_lock = threading.RLock()

            # 設定臨時檔案目錄
            self.upload_dir = Path(__file__).parent / "uploads"
            self.result_dir = Path(__file__).parent / "result"
            self.upload_dir.mkdir(exist_ok=True)
            self.result_dir.mkdir(exist_ok=True)

            self.initialized = True

    def create_task(
        self,
        task_id: str,
        email: str,
        filename: str,
        enable_diarization: bool = True,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        language: Optional[str] = None,
        task: str = 'transcribe',
        model: str = 'CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32',
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        enable_confidence_score: bool = False,
        compute_type: Optional[str] = None,
        enable_llm_correction: bool = False,
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """創建新任務"""
        with self.task_lock:
            task_data = {
                'task_id': task_id,
                'email': email,
                'filename': filename,
                'status': 'pending',
                'progress': 0.0,
                'current_stage': None,
                'asr_progress': None,
                'enable_diarization': enable_diarization,
                'start_time': start_time,
                'end_time': end_time,
                'language': language,
                'task': task,
                'model': model,
                'vad_onset': vad_onset,
                'vad_offset': vad_offset,
                'min_speakers': min_speakers,
                'max_speakers': max_speakers,
                'enable_confidence_score': enable_confidence_score,
                'compute_type': compute_type,
                'enable_llm_correction': enable_llm_correction,
                'llm_model': llm_model or 'gemma3:4b',
                'created_at': datetime.now().isoformat(),
                'started_at': None,
                'completed_at': None,
                'error_message': None,
                'partial_result': [],
                # 檔案路徑（臨時）
                'upload_path': str(self.upload_dir / task_id),
                'result_path': str(self.result_dir / task_id)
            }

            # 創建任務目錄
            Path(task_data['upload_path']).mkdir(parents=True, exist_ok=True)
            Path(task_data['result_path']).mkdir(parents=True, exist_ok=True)

            self.tasks[task_id] = task_data
            return task_data.copy()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """獲取任務詳情（不包含敏感資料）"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if task:
                # 返回副本，不包含檔案路徑
                safe_task = task.copy()
                safe_task.pop('upload_path', None)
                safe_task.pop('result_path', None)
                return safe_task
            return None

    def get_task_full(self, task_id: str) -> Optional[Dict[str, Any]]:
        """獲取完整任務詳情（包含所有資料，僅供內部使用）"""
        with self.task_lock:
            return self.tasks.get(task_id)

    def cleanup_task_files(self, task_id: str) -> bool:
        """清理任務的臨時檔案"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            try:
                # 刪除上傳目錄
                upload_path = Path(task['upload_path'])
                if upload_path.exists():
                    shutil.rmtree(upload_path, ignore_errors=True)

                # 刪除結果目錄
                result_path = Path(task['result_path'])
                if result_path.exists():
                    shutil.rmtree(result_path, ignore_errors=True)

                # 記錄文件刪除日誌
                security_logger.log_file_deleted(
                    task_id=task_id,
                    reason=f"Task {task['status']} - automatic cleanup"
                )

                return True
            except Exception as e:
                print(f"清理任務 {task_id} 檔案時發生錯誤: {e}")
                return False

    def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None,
        asr_progress: Optional[Dict[str, Any]] = None
    ):
        """更新任務狀態"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return

            task['status'] = status

            if progress is not None:
                task['progress'] = progress

            if asr_progress is not None:
                task['asr_progress'] = asr_progress

            if current_stage is not None:
                task['current_stage'] = current_stage

            if error_message is not None:
                task['error_message'] = error_message

            # 更新時間戳記
            if status == "processing" and not task.get('started_at'):
                task['started_at'] = datetime.now().isoformat()
            elif status in ("completed", "failed", "canceled"):
                task['completed_at'] = datetime.now().isoformat()

                # 任務結束時，自動清理臨時檔案
                # 注意：completed 狀態的清理應該在發送郵件後進行
                if status in ("failed", "canceled"):
                    self.cleanup_task_files(task_id)

    def update_task_result(
        self,
        task_id: str,
        partial_result: Optional[List[Dict]] = None
    ):
        """更新任務部分結果"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return

            if partial_result is not None:
                task['partial_result'] = partial_result

    def cleanup_all_temporary_files(self):
        """清理所有臨時檔案（啟動時調用）"""
        print("清理殘留的臨時檔案...")

        try:
            # 清理 uploads 目錄
            if self.upload_dir.exists():
                for item in self.upload_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                        print(f"已刪除: {item}")

            # 清理 result 目錄
            if self.result_dir.exists():
                for item in self.result_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                        print(f"已刪除: {item}")

            print("臨時檔案清理完成")
        except Exception as e:
            print(f"清理臨時檔案時發生錯誤: {e}")

    def get_tasks_by_email(self, email: str, limit: int = 50) -> List[Dict[str, Any]]:
        """根據郵箱獲取任務列表"""
        with self.task_lock:
            tasks = []
            for task in reversed(list(self.tasks.values())):
                if task['email'] == email:
                    # 返回安全版本（不包含檔案路徑）
                    safe_task = task.copy()
                    safe_task.pop('upload_path', None)
                    safe_task.pop('result_path', None)
                    tasks.append(safe_task)

                    if len(tasks) >= limit:
                        break

            return tasks

    def get_queue_position(self, task_id: str) -> int:
        """獲取任務在佇列中的位置"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return -1

            position = 0
            for t in self.tasks.values():
                if t['created_at'] < task['created_at'] and t['status'] in ('pending', 'processing'):
                    position += 1

            return position

    def get_processing_count(self) -> int:
        """獲取正在處理的任務數量"""
        with self.task_lock:
            return sum(1 for task in self.tasks.values() if task['status'] == 'processing')

    def cancel_task(self, task_id: str) -> bool:
        """取消任務"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            if task['status'] in ('completed', 'failed', 'canceled'):
                return False

            task['status'] = 'canceled'
            task['completed_at'] = datetime.now().isoformat()
            return True

    def delete_task(self, task_id: str) -> bool:
        """刪除任務（從記憶體中移除）"""
        with self.task_lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                return True
            return False

    def get_all_tasks_summary(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """獲取所有任務摘要（僅管理員可見，不包含敏感資料）"""
        with self.task_lock:
            all_tasks = list(reversed(list(self.tasks.values())))
            tasks = all_tasks[offset:offset + limit]

            # 只返回佇列資訊，不包含檔案內容和完整郵箱
            summaries = []
            for task in tasks:
                # 郵箱脫敏處理：只顯示部分
                email = task['email']
                if '@' in email:
                    local, domain = email.split('@', 1)
                    if len(local) > 3:
                        masked_email = f"{local[:2]}***@{domain}"
                    else:
                        masked_email = f"***@{domain}"
                else:
                    masked_email = "***"

                summaries.append({
                    'task_id': task['task_id'],
                    'email_masked': masked_email,  # 脫敏郵箱
                    'filename': task['filename'],  # 檔名仍然顯示（方便管理）
                    'status': task['status'],
                    'progress': task['progress'],
                    'current_stage': task['current_stage'],
                    'enable_diarization': task['enable_diarization'],
                    'created_at': task['created_at'],
                    'started_at': task['started_at'],
                    'completed_at': task['completed_at'],
                    'error_message': task['error_message']
                })

            return summaries

    def get_total_tasks_count(self) -> int:
        """獲取任務總數"""
        with self.task_lock:
            return len(self.tasks)

    def get_stats_summary(self) -> Dict[str, Any]:
        """獲取統計摘要"""
        with self.task_lock:
            status_counts = {}
            unique_emails = set()

            for task in self.tasks.values():
                status = task['status']
                status_counts[status] = status_counts.get(status, 0) + 1
                unique_emails.add(task['email'])

            # 今日任務數
            today = datetime.now().date()
            today_count = sum(
                1 for task in self.tasks.values()
                if datetime.fromisoformat(task['created_at']).date() == today
            )

            return {
                'total_tasks': len(self.tasks),
                'today_tasks': today_count,
                'unique_users': len(unique_emails),
                'status_counts': status_counts
            }

    def cleanup_old_completed_tasks(self, keep_count: int = 100):
        """清理舊的已完成任務，保留最近的 N 個"""
        with self.task_lock:
            completed_tasks = [
                (task_id, task) for task_id, task in self.tasks.items()
                if task['status'] in ('completed', 'failed', 'canceled')
            ]

            # 按完成時間排序
            completed_tasks.sort(
                key=lambda x: x[1].get('completed_at', ''),
                reverse=True
            )

            # 刪除舊的已完成任務
            for task_id, _ in completed_tasks[keep_count:]:
                del self.tasks[task_id]

            return len(completed_tasks) - min(keep_count, len(completed_tasks))


# 全局記憶體管理實例
memory_manager = MemoryTaskManager()
