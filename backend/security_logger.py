"""
安全日誌模組
符合 SSDLC 要求：記錄所有安全相關事件
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class SecurityLogger:
    """安全日誌記錄器"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 創建日誌目錄
            self.log_dir = Path(__file__).parent / "logs"
            self.log_dir.mkdir(exist_ok=True)

            # 設置不同類型的日誌記錄器
            self._setup_loggers()
            self.initialized = True

    def _setup_loggers(self):
        """設置多個專用日誌記錄器"""

        # 1. 身份驗證日誌（保存 6 個月）
        self.auth_logger = self._create_logger(
            'auth',
            self.log_dir / 'auth.log',
            max_bytes=10*1024*1024,  # 10MB
            backup_count=180  # 保存 180 天
        )

        # 2. 操作日誌（保存 6 個月）
        self.operation_logger = self._create_logger(
            'operation',
            self.log_dir / 'operation.log',
            max_bytes=50*1024*1024,  # 50MB
            backup_count=180
        )

        # 3. 安全事件日誌（保存 1 年）
        self.security_logger = self._create_logger(
            'security',
            self.log_dir / 'security.log',
            max_bytes=10*1024*1024,
            backup_count=365
        )

        # 4. 錯誤日誌（保存 6 個月）
        self.error_logger = self._create_logger(
            'error',
            self.log_dir / 'error.log',
            max_bytes=20*1024*1024,
            backup_count=180
        )

        # 5. 審計日誌（保存 5 年，用於個資相關操作）
        self.audit_logger = self._create_logger(
            'audit',
            self.log_dir / 'audit.log',
            max_bytes=50*1024*1024,
            backup_count=1825  # 5 年
        )

    def _create_logger(
        self,
        name: str,
        log_file: Path,
        max_bytes: int = 10*1024*1024,
        backup_count: int = 180
    ) -> logging.Logger:
        """創建日誌記錄器"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # 避免重複添加 handler
        if logger.handlers:
            return logger

        # 使用 RotatingFileHandler 進行日誌輪替
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )

        # 日誌格式：時間 | 級別 | 事件類型 | 詳細信息
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _format_log_entry(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: Optional[str],
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """格式化日誌條目"""
        log_data = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id or 'anonymous',
            'ip_address': ip_address or 'unknown',
            'action': action,
            'result': result,
            'details': details or {}
        }
        return json.dumps(log_data, ensure_ascii=False)

    # ==================== 身份驗證日誌 ====================

    def log_auth_attempt(
        self,
        email: str,
        ip_address: str,
        success: bool,
        reason: Optional[str] = None
    ):
        """記錄身份驗證嘗試"""
        log_entry = self._format_log_entry(
            event_type='AUTH_ATTEMPT',
            user_id=email,
            ip_address=ip_address,
            action='email_verification',
            result='success' if success else 'failed',
            details={'reason': reason} if reason else None
        )
        self.auth_logger.info(log_entry)

    def log_verification_code_sent(self, email: str, ip_address: str):
        """記錄驗證碼發送"""
        log_entry = self._format_log_entry(
            event_type='VERIFICATION_CODE_SENT',
            user_id=email,
            ip_address=ip_address,
            action='send_verification_code',
            result='success'
        )
        self.auth_logger.info(log_entry)

    def log_session_expired(self, email: str):
        """記錄會話過期"""
        log_entry = self._format_log_entry(
            event_type='SESSION_EXPIRED',
            user_id=email,
            ip_address=None,
            action='session_check',
            result='expired'
        )
        self.auth_logger.info(log_entry)

    # ==================== 操作日誌 ====================

    def log_task_created(
        self,
        task_id: str,
        email: str,
        ip_address: str,
        filename: str,
        parameters: Dict[str, Any]
    ):
        """記錄任務創建"""
        log_entry = self._format_log_entry(
            event_type='TASK_CREATED',
            user_id=email,
            ip_address=ip_address,
            action='create_task',
            result='success',
            details={
                'task_id': task_id,
                'filename': filename,
                'parameters': parameters
            }
        )
        self.operation_logger.info(log_entry)

    def log_task_completed(
        self,
        task_id: str,
        email: str,
        success: bool,
        error: Optional[str] = None
    ):
        """記錄任務完成"""
        log_entry = self._format_log_entry(
            event_type='TASK_COMPLETED',
            user_id=email,
            ip_address=None,
            action='process_task',
            result='success' if success else 'failed',
            details={'task_id': task_id, 'error': error} if error else {'task_id': task_id}
        )
        self.operation_logger.info(log_entry)

    def log_file_upload(
        self,
        email: str,
        ip_address: str,
        filename: str,
        file_size: int,
        file_type: str
    ):
        """記錄文件上傳"""
        log_entry = self._format_log_entry(
            event_type='FILE_UPLOAD',
            user_id=email,
            ip_address=ip_address,
            action='upload_file',
            result='success',
            details={
                'filename': filename,
                'file_size': file_size,
                'file_type': file_type
            }
        )
        self.operation_logger.info(log_entry)

    def log_file_deleted(self, task_id: str, reason: str):
        """記錄文件刪除"""
        log_entry = self._format_log_entry(
            event_type='FILE_DELETED',
            user_id='system',
            ip_address=None,
            action='delete_file',
            result='success',
            details={'task_id': task_id, 'reason': reason}
        )
        self.operation_logger.info(log_entry)

    # ==================== 安全事件日誌 ====================

    def log_security_event(
        self,
        event_type: str,
        ip_address: str,
        severity: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """記錄安全事件"""
        log_entry = self._format_log_entry(
            event_type=event_type,
            user_id=None,
            ip_address=ip_address,
            action='security_event',
            result=severity,
            details={'description': description, **(details or {})}
        )

        if severity == 'critical':
            self.security_logger.error(log_entry)
        elif severity == 'warning':
            self.security_logger.warning(log_entry)
        else:
            self.security_logger.info(log_entry)

    def log_rate_limit_exceeded(self, ip_address: str, endpoint: str):
        """記錄速率限制超出"""
        self.log_security_event(
            event_type='RATE_LIMIT_EXCEEDED',
            ip_address=ip_address,
            severity='warning',
            description='Rate limit exceeded',
            details={'endpoint': endpoint}
        )

    def log_invalid_request(
        self,
        ip_address: str,
        endpoint: str,
        reason: str
    ):
        """記錄無效請求"""
        self.log_security_event(
            event_type='INVALID_REQUEST',
            ip_address=ip_address,
            severity='warning',
            description='Invalid request detected',
            details={'endpoint': endpoint, 'reason': reason}
        )

    def log_unauthorized_access(
        self,
        ip_address: str,
        endpoint: str,
        user_id: Optional[str] = None
    ):
        """記錄未授權訪問"""
        log_entry = self._format_log_entry(
            event_type='UNAUTHORIZED_ACCESS',
            user_id=user_id,
            ip_address=ip_address,
            action='access_attempt',
            result='denied',
            details={'endpoint': endpoint}
        )
        self.security_logger.warning(log_entry)

    # ==================== 錯誤日誌 ====================

    def log_error(
        self,
        error_type: str,
        error_message: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """記錄錯誤"""
        log_entry = self._format_log_entry(
            event_type='ERROR',
            user_id=user_id,
            ip_address=None,
            action=error_type,
            result='error',
            details={'message': error_message, **(details or {})}
        )
        self.error_logger.error(log_entry)

    # ==================== 審計日誌（個資相關）====================

    def log_personal_data_access(
        self,
        email: str,
        ip_address: str,
        action: str,
        data_type: str
    ):
        """記錄個資訪問（保存 5 年）"""
        log_entry = self._format_log_entry(
            event_type='PERSONAL_DATA_ACCESS',
            user_id=email,
            ip_address=ip_address,
            action=action,
            result='success',
            details={'data_type': data_type}
        )
        self.audit_logger.info(log_entry)

    def log_data_deletion(
        self,
        task_id: str,
        email: str,
        reason: str,
        data_type: str
    ):
        """記錄數據刪除（符合個資法要求）"""
        log_entry = self._format_log_entry(
            event_type='DATA_DELETION',
            user_id=email,
            ip_address=None,
            action='delete_data',
            result='success',
            details={
                'task_id': task_id,
                'reason': reason,
                'data_type': data_type
            }
        )
        self.audit_logger.info(log_entry)

    # ==================== 管理員操作日誌 ====================

    def log_admin_action(
        self,
        admin_token: str,
        ip_address: str,
        action: str,
        target: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """記錄管理員操作"""
        log_entry = self._format_log_entry(
            event_type='ADMIN_ACTION',
            user_id=f'admin_{admin_token[:8]}',  # 只記錄 token 前 8 位
            ip_address=ip_address,
            action=action,
            result='success',
            details={'target': target, **(details or {})}
        )
        self.audit_logger.info(log_entry)


# 全局安全日誌實例
security_logger = SecurityLogger()
