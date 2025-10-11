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
                start_time REAL,
                end_time REAL,
                language TEXT,
                task TEXT DEFAULT 'transcribe',
                model TEXT DEFAULT 'XA9/Belle-faster-whisper-large-v3-zh-punct',
                queue_position INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                result_path TEXT,
                partial_result TEXT
            )
        """)

        # 遷移：檢查並添加新欄位（如果表已存在但沒有這些欄位）
        try:
            # 檢查欄位是否存在
            cursor.execute("PRAGMA table_info(tasks)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'start_time' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN start_time REAL")
                print("已添加 start_time 欄位")

            if 'end_time' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN end_time REAL")
                print("已添加 end_time 欄位")

            if 'language' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN language TEXT")
                print("已添加 language 欄位")

            if 'task' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN task TEXT DEFAULT 'transcribe'")
                print("已添加 task 欄位")

            if 'model' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN model TEXT DEFAULT 'XA9/Belle-faster-whisper-large-v3-zh-punct'")
                print("已添加 model 欄位")

            # 新增進階參數欄位
            if 'vad_onset' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN vad_onset REAL DEFAULT 0.5")
                print("已添加 vad_onset 欄位")

            if 'vad_offset' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN vad_offset REAL DEFAULT 0.363")
                print("已添加 vad_offset 欄位")

            if 'min_speakers' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN min_speakers INTEGER")
                print("已添加 min_speakers 欄位")

            if 'max_speakers' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN max_speakers INTEGER")
                print("已添加 max_speakers 欄位")

            if 'enable_confidence_score' not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN enable_confidence_score BOOLEAN DEFAULT 0")
                print("已添加 enable_confidence_score 欄位")
        except Exception as e:
            print(f"資料庫遷移警告: {e}")

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

        # 創建預處理任務表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preprocess_tasks (
                preprocess_id TEXT PRIMARY KEY,
                client_ip TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                current_stage TEXT,
                config_json TEXT,
                original_path TEXT,
                processed_path TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                original_info TEXT,
                processed_info TEXT,
                filters_applied TEXT
            )
        """)

        # 創建預處理任務索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_preprocess_client_ip ON preprocess_tasks(client_ip)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_preprocess_status ON preprocess_tasks(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_preprocess_created_at ON preprocess_tasks(created_at DESC)
        """)

        conn.commit()
        conn.close()
    
    def create_task(
        self,
        task_id: str,
        client_ip: str,
        filename: str,
        enable_diarization: bool = True,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        language: Optional[str] = None,
        task: str = 'transcribe',
        model: str = 'CWTchen/Belle-whisper-large-v3-zh-punct-ct2-faster-whisper-float32',
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        enable_confidence_score: bool = False
    ) -> Dict[str, Any]:
        """創建新任務"""
        conn = self.get_connection()
        cursor = conn.cursor()

        created_at = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO tasks (
                task_id, client_ip, filename, status,
                enable_diarization, start_time, end_time, language, task, model,
                vad_onset, vad_offset, min_speakers, max_speakers,
                enable_confidence_score, created_at
            )
            VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (task_id, client_ip, filename, enable_diarization, start_time, end_time, language, task, model, 
              vad_onset, vad_offset, min_speakers, max_speakers, enable_confidence_score, created_at))

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
    
    def update_task_result(self, task_id: str, result_path: str, partial_result: Optional[List[Dict]] = None):
        """更新任務結果，並處理不可序列化的資料類型"""
        
        serializable_result_list = partial_result
        if isinstance(partial_result, list):
            serializable_result_list = []
            for segment in partial_result:
                if not isinstance(segment, dict):
                    serializable_result_list.append(segment)
                    continue

                segment_copy = segment.copy()
                if 'words' in segment_copy and isinstance(segment_copy['words'], list):
                    serializable_words = []
                    for word in segment_copy['words']:
                        if hasattr(word, 'start') and hasattr(word, 'word'):  # Duck typing for Word object
                            serializable_words.append({
                                'start': float(word.start),
                                'end': float(word.end),
                                'word': word.word,
                                'probability': float(word.probability)
                            })
                        elif isinstance(word, dict): # Handles dict with numpy floats
                            word_copy = word.copy()
                            if 'start' in word_copy: word_copy['start'] = float(word_copy['start'])
                            if 'end' in word_copy: word_copy['end'] = float(word_copy['end'])
                            if 'probability' in word_copy: word_copy['probability'] = float(word_copy['probability'])
                            serializable_words.append(word_copy)
                        else:
                            serializable_words.append(word)
                    segment_copy['words'] = serializable_words
                serializable_result_list.append(segment_copy)

        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks 
            SET result_path = ?, partial_result = ?
            WHERE task_id = ?
        """, (result_path, json.dumps(serializable_result_list, ensure_ascii=False), task_id))
        
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
    
    def get_all_tasks(self, limit: int = 100, offset: int = 0, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """獲取所有任務（管理者用）"""
        conn = self.get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT * FROM tasks
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (status, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM tasks
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            task = dict(row)
            if task.get('partial_result'):
                try:
                    task['partial_result'] = json.loads(task['partial_result'])
                except:
                    pass
            tasks.append(task)

        return tasks

    def get_total_tasks_count(self, status: Optional[str] = None) -> int:
        """獲取任務總數"""
        conn = self.get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute("SELECT COUNT(*) as count FROM tasks WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT COUNT(*) as count FROM tasks")

        result = cursor.fetchone()
        conn.close()
        return result['count'] if result else 0

    def get_stats_summary(self) -> Dict[str, Any]:
        """獲取統計摘要（管理者用）"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 各狀態任務數量
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM tasks
            GROUP BY status
        """)
        status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

        # 總任務數
        cursor.execute("SELECT COUNT(*) as count FROM tasks")
        total = cursor.fetchone()['count']

        # 今日任務數
        cursor.execute("""
            SELECT COUNT(*) as count FROM tasks
            WHERE date(created_at) = date('now')
        """)
        today = cursor.fetchone()['count']

        # 獨特 IP 數量
        cursor.execute("SELECT COUNT(DISTINCT client_ip) as count FROM tasks")
        unique_ips = cursor.fetchone()['count']

        conn.close()

        return {
            "total_tasks": total,
            "today_tasks": today,
            "unique_users": unique_ips,
            "status_counts": status_counts
        }

    def batch_delete_tasks(self, task_ids: List[str]) -> int:
        """批量刪除任務"""
        conn = self.get_connection()
        cursor = conn.cursor()

        deleted_count = 0
        for task_id in task_ids:
            try:
                cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
                if cursor.rowcount > 0:
                    deleted_count += 1
            except Exception as e:
                print(f"刪除任務 {task_id} 失敗: {e}")

        conn.commit()
        conn.close()
        return deleted_count

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

    # ===== 預處理任務方法 =====

    def create_preprocess_task(
        self,
        preprocess_id: str,
        client_ip: str,
        filename: str,
        config_json: str,
        original_path: str,
        processed_path: str
    ) -> Dict[str, Any]:
        """創建預處理任務"""
        conn = self.get_connection()
        cursor = conn.cursor()

        created_at = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO preprocess_tasks (
                preprocess_id, client_ip, filename, status,
                config_json, original_path, processed_path, created_at
            )
            VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)
        """, (preprocess_id, client_ip, filename, config_json, original_path, processed_path, created_at))

        conn.commit()
        conn.close()

        return self.get_preprocess_task(preprocess_id)

    def get_preprocess_task(self, preprocess_id: str) -> Optional[Dict[str, Any]]:
        """獲取預處理任務詳情"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM preprocess_tasks WHERE preprocess_id = ?", (preprocess_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            task = dict(row)
            # 解析 JSON 欄位
            if task.get('config_json'):
                try:
                    task['config'] = json.loads(task['config_json'])
                except:
                    task['config'] = {}
            if task.get('original_info'):
                try:
                    task['original_info'] = json.loads(task['original_info'])
                except:
                    pass
            if task.get('processed_info'):
                try:
                    task['processed_info'] = json.loads(task['processed_info'])
                except:
                    pass
            if task.get('filters_applied'):
                try:
                    task['filters_applied'] = json.loads(task['filters_applied'])
                except:
                    pass
            return task
        return None

    def update_preprocess_status(
        self,
        preprocess_id: str,
        status: str,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """更新預處理任務狀態"""
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
        if status == "processing":
            task = self.get_preprocess_task(preprocess_id)
            if task and not task.get('started_at'):
                updates.append("started_at = ?")
                params.append(datetime.now().isoformat())
        elif status in ("completed", "failed", "canceled"):
            updates.append("completed_at = ?")
            params.append(datetime.now().isoformat())

        params.append(preprocess_id)

        query = f"UPDATE preprocess_tasks SET {', '.join(updates)} WHERE preprocess_id = ?"
        cursor.execute(query, params)
        conn.commit()
        conn.close()

    def update_preprocess_result(
        self,
        preprocess_id: str,
        original_info: Dict,
        processed_info: Dict,
        filters_applied: List[str]
    ):
        """更新預處理結果"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE preprocess_tasks
            SET original_info = ?, processed_info = ?, filters_applied = ?
            WHERE preprocess_id = ?
        """, (
            json.dumps(original_info, ensure_ascii=False),
            json.dumps(processed_info, ensure_ascii=False),
            json.dumps(filters_applied, ensure_ascii=False),
            preprocess_id
        ))

        conn.commit()
        conn.close()

    def get_preprocess_tasks_by_ip(self, client_ip: str, limit: int = 50) -> List[Dict[str, Any]]:
        """根據 IP 獲取預處理任務列表"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM preprocess_tasks
            WHERE client_ip = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (client_ip, limit))

        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            task = dict(row)
            if task.get('config_json'):
                try:
                    task['config'] = json.loads(task['config_json'])
                except:
                    task['config'] = {}
            if task.get('filters_applied'):
                try:
                    task['filters_applied'] = json.loads(task['filters_applied'])
                except:
                    pass
            tasks.append(task)

        return tasks

    def delete_preprocess_task(self, preprocess_id: str) -> bool:
        """刪除預處理任務記錄"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM preprocess_tasks WHERE preprocess_id = ?", (preprocess_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"刪除預處理任務記錄失敗: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def cancel_preprocess_task(self, preprocess_id: str) -> bool:
        """取消預處理任務"""
        task = self.get_preprocess_task(preprocess_id)
        if not task:
            return False

        if task['status'] in ('completed', 'failed', 'canceled'):
            return False

        self.update_preprocess_status(preprocess_id, 'canceled')
        return True


# 全局資料庫實例
db_manager = DatabaseManager()

