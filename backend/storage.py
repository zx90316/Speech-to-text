# -*- coding: utf-8 -*-
"""
SQLite 資料庫儲存模組
持久化任務狀態、結果與郵件驗證資訊
"""
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path
from contextlib import contextmanager

from config import DATA_DIR


# 資料庫路徑
DB_PATH = DATA_DIR / "backend.db"


class Database:
    """SQLite 資料庫管理"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def get_connection(self):
        """取得資料庫連線"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """初始化資料庫表格"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 任務表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    current_stage TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    filename TEXT,
                    audio_path TEXT,
                    email TEXT,
                    
                    -- 任務參數
                    enable_diarization INTEGER DEFAULT 1,
                    enable_timestamps INTEGER DEFAULT 0,
                    language TEXT,
                    model TEXT,
                    min_speakers INTEGER,
                    max_speakers INTEGER,
                    
                    -- 結果
                    text TEXT,
                    detected_language TEXT,
                    segments TEXT,
                    has_diarization INTEGER DEFAULT 0
                )
            """)
            
            # 郵件驗證表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_verifications (
                    email TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    verified INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
            print(f"✅ 資料庫初始化完成: {self.db_path}")


class TaskStorage:
    """任務儲存管理（SQLite 版本）"""
    
    def __init__(self, db: Database = None):
        self.db = db or Database()
    
    def create_task(
        self,
        filename: str,
        email: Optional[str] = None,
        enable_diarization: bool = True,
        enable_timestamps: bool = False,
        language: Optional[str] = None,
        model: str = "Qwen/Qwen3-ASR-1.7B",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> str:
        """建立新任務"""
        task_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tasks (
                    task_id, status, progress, current_stage, created_at,
                    filename, email, enable_diarization, enable_timestamps,
                    language, model, min_speakers, max_speakers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id, "pending", 0.0, "等待處理", created_at,
                filename, email, int(enable_diarization), int(enable_timestamps),
                language, model, min_speakers, max_speakers
            ))
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """取得任務資訊"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """將資料庫列轉換為字典"""
        data = dict(row)
        
        # 轉換布林值
        data["enable_diarization"] = bool(data.get("enable_diarization", 0))
        data["enable_timestamps"] = bool(data.get("enable_timestamps", 0))
        data["has_diarization"] = bool(data.get("has_diarization", 0))
        
        # 解析 segments JSON
        if data.get("segments"):
            try:
                data["segments"] = json.loads(data["segments"])
            except json.JSONDecodeError:
                data["segments"] = None
        
        # 解析日期
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        
        return data
    
    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None,
        text: Optional[str] = None,
        detected_language: Optional[str] = None,
        segments: Optional[List[Dict]] = None,
        has_diarization: Optional[bool] = None,
        audio_path: Optional[str] = None,
    ) -> bool:
        """更新任務"""
        updates = []
        values = []
        
        if status is not None:
            updates.append("status = ?")
            values.append(status)
            if status == "completed":
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())
        
        if progress is not None:
            updates.append("progress = ?")
            values.append(progress)
        
        if current_stage is not None:
            updates.append("current_stage = ?")
            values.append(current_stage)
        
        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)
        
        if text is not None:
            updates.append("text = ?")
            values.append(text)
        
        if detected_language is not None:
            updates.append("detected_language = ?")
            values.append(detected_language)
        
        if segments is not None:
            updates.append("segments = ?")
            values.append(json.dumps(segments, ensure_ascii=False))
        
        if has_diarization is not None:
            updates.append("has_diarization = ?")
            values.append(int(has_diarization))
        
        if audio_path is not None:
            updates.append("audio_path = ?")
            values.append(audio_path)
        
        if not updates:
            return True
        
        values.append(task_id)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE tasks SET {', '.join(updates)} WHERE task_id = ?",
                values
            )
            return cursor.rowcount > 0
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任務"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tasks SET status = 'canceled', current_stage = '已取消'
                WHERE task_id = ? AND status IN ('pending', 'processing')
            """, (task_id,))
            return cursor.rowcount > 0
    
    def delete_task(self, task_id: str) -> bool:
        """刪除任務"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            return cursor.rowcount > 0
    
    def get_all_tasks(self, limit: int = 100, email: Optional[str] = None) -> List[Dict[str, Any]]:
        """取得所有任務"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if email:
                cursor.execute(
                    "SELECT * FROM tasks WHERE email = ? ORDER BY created_at DESC LIMIT ?",
                    (email, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def is_canceled(self, task_id: str) -> bool:
        """檢查任務是否已取消"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
            return row is not None and row["status"] == "canceled"


class EmailVerificationStorage:
    """郵件驗證儲存管理"""
    
    def __init__(self, db: Database = None):
        self.db = db or Database()
    
    def generate_code(self, email: str) -> str:
        """生成驗證碼"""
        import secrets
        code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        expires_at = (datetime.now() + timedelta(minutes=5)).isoformat()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO email_verifications (email, code, expires_at, verified)
                VALUES (?, ?, ?, 0)
            """, (email, code, expires_at))
        
        return code
    
    def verify_code(self, email: str, code: str) -> bool:
        """驗證碼驗證"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT code, expires_at FROM email_verifications WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()
            
            if not row:
                return False
            
            # 檢查過期
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now() > expires_at:
                cursor.execute("DELETE FROM email_verifications WHERE email = ?", (email,))
                return False
            
            # 檢查驗證碼
            if row["code"] == code:
                # 驗證成功，延長有效期到 24 小時
                new_expires = (datetime.now() + timedelta(hours=24)).isoformat()
                cursor.execute("""
                    UPDATE email_verifications SET verified = 1, expires_at = ?
                    WHERE email = ?
                """, (new_expires, email))
                return True
            
            return False
    
    def is_verified(self, email: str) -> bool:
        """檢查郵件是否已驗證"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT verified, expires_at FROM email_verifications WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()
            
            if not row:
                return False
            
            # 檢查過期
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now() > expires_at:
                return False
            
            return bool(row["verified"])


# 建立全域實例
db = Database()
storage = TaskStorage(db)
email_verification = EmailVerificationStorage(db)
