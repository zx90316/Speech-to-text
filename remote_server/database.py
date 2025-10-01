"""
SQLite 資料庫管理模組
用於儲存和查詢任務資訊
"""
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading


class DatabaseManager:
    """資料庫管理類別，使用單例模式"""
    
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
            self.db_path = Path(__file__).parent / "tasks.db"
            self.init_database()
            self.initialized = True
    
    def get_connection(self):
        """獲取資料庫連接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """初始化資料庫表格"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 創建任務表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                client_ip TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                current_stage TEXT,
                enable_diarization BOOLEAN DEFAULT 1,
                queue_position INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                result_path TEXT,
                partial_result TEXT
            )
        """)
        
        # 創建索引以加快查詢速度
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_client_ip ON tasks(client_ip)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def create_task(
        self, 
        task_id: str, 
        client_ip: str, 
        filename: str,
        enable_diarization: bool = True
    ) -> Dict[str, Any]:
        """創建新任務"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        created_at = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO tasks (
                task_id, client_ip, filename, status, 
                enable_diarization, created_at
            )
            VALUES (?, ?, ?, 'pending', ?, ?)
        """, (task_id, client_ip, filename, enable_diarization, created_at))
        
        conn.commit()
        conn.close()
        
        return self.get_task(task_id)
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """獲取任務詳情"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            task = dict(row)
            # 解析 JSON 欄位
            if task.get('partial_result'):
                try:
                    task['partial_result'] = json.loads(task['partial_result'])
                except:
                    task['partial_result'] = []
            return task
        return None
    
    def update_task_status(
        self, 
        task_id: str, 
        status: str,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """更新任務狀態"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        updates = ["status = ?"]
        params = [status]
        
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        
        if current_stage is not None:
            updates.append("current_stage = ?")
            params.append(current_stage)
        
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        
        # 更新時間戳記
        if status == "processing" and not self.get_task(task_id).get('started_at'):
            updates.append("started_at = ?")
            params.append(datetime.now().isoformat())
        elif status in ("completed", "failed", "canceled"):
            updates.append("completed_at = ?")
            params.append(datetime.now().isoformat())
        
        params.append(task_id)
        
        query = f"UPDATE tasks SET {', '.join(updates)} WHERE task_id = ?"
        cursor.execute(query, params)
        conn.commit()
        conn.close()
    
    def update_task_result(self, task_id: str, result_path: str, partial_result: List[Dict]):
        """更新任務結果"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks 
            SET result_path = ?, partial_result = ?
            WHERE task_id = ?
        """, (result_path, json.dumps(partial_result, ensure_ascii=False), task_id))
        
        conn.commit()
        conn.close()
    
    def get_tasks_by_ip(self, client_ip: str, limit: int = 50) -> List[Dict[str, Any]]:
        """根據 IP 獲取任務列表"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM tasks 
            WHERE client_ip = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (client_ip, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            task = dict(row)
            if task.get('partial_result'):
                try:
                    task['partial_result'] = json.loads(task['partial_result'])
                except:
                    task['partial_result'] = []
            tasks.append(task)
        
        return tasks
    
    def get_queue_position(self, task_id: str) -> int:
        """獲取任務在佇列中的位置"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        task = self.get_task(task_id)
        if not task:
            return -1
        
        # 計算在此任務之前創建且仍在處理中的任務數量
        cursor.execute("""
            SELECT COUNT(*) as count FROM tasks 
            WHERE created_at < ? 
            AND status IN ('pending', 'processing')
        """, (task['created_at'],))
        
        result = cursor.fetchone()
        conn.close()
        
        return result['count'] if result else 0
    
    def get_processing_count(self) -> int:
        """獲取正在處理的任務數量"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as count FROM tasks 
            WHERE status = 'processing'
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return result['count'] if result else 0
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任務"""
        task = self.get_task(task_id)
        if not task:
            return False

        if task['status'] in ('completed', 'failed', 'canceled'):
            return False

        self.update_task_status(task_id, 'canceled')
        return True

    def delete_task(self, task_id: str) -> bool:
        """永久刪除任務記錄"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"刪除任務記錄失敗: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def cleanup_old_tasks(self, days: int = 7):
        """清理舊任務（可選功能）"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM tasks 
            WHERE completed_at IS NOT NULL 
            AND datetime(completed_at) < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count


# 全局資料庫實例
db_manager = DatabaseManager()

