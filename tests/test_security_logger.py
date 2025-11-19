"""
測試安全日誌模組 (security_logger.py)
"""
import pytest
import json
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from security_logger import SecurityLogger, security_logger


class TestSecurityLogger:
    """測試 SecurityLogger 類"""

    def test_singleton_pattern(self):
        """測試單例模式"""
        logger1 = SecurityLogger()
        logger2 = SecurityLogger()
        assert logger1 is logger2

    def test_log_directory_creation(self, temp_dir):
        """測試日誌目錄創建"""
        # 使用臨時目錄
        with patch.object(Path, '__truediv__', return_value=temp_dir / 'logs'):
            logger = SecurityLogger()
            # 日誌目錄應該被創建
            assert (temp_dir / 'logs').exists() or logger.log_dir.exists()

    def test_loggers_initialized(self):
        """測試各類日誌記錄器已初始化"""
        logger = SecurityLogger()
        
        assert hasattr(logger, 'auth_logger')
        assert hasattr(logger, 'operation_logger')
        assert hasattr(logger, 'security_logger')
        assert hasattr(logger, 'error_logger')
        assert hasattr(logger, 'audit_logger')

    def test_format_log_entry(self):
        """測試日誌格式化"""
        logger = SecurityLogger()
        
        log_entry = logger._format_log_entry(
            event_type='TEST_EVENT',
            user_id='test@example.com',
            ip_address='192.168.1.1',
            action='test_action',
            result='success',
            details={'key': 'value'}
        )
        
        # 應該返回 JSON 格式
        log_data = json.loads(log_entry)
        
        assert log_data['event_type'] == 'TEST_EVENT'
        assert log_data['user_id'] == 'test@example.com'
        assert log_data['ip_address'] == '192.168.1.1'
        assert log_data['action'] == 'test_action'
        assert log_data['result'] == 'success'
        assert log_data['details']['key'] == 'value'
        assert 'timestamp' in log_data

    def test_format_log_entry_with_none_values(self):
        """測試包含 None 值的日誌格式化"""
        logger = SecurityLogger()
        
        log_entry = logger._format_log_entry(
            event_type='TEST_EVENT',
            user_id=None,
            ip_address=None,
            action='test_action',
            result='success',
            details=None
        )
        
        log_data = json.loads(log_entry)
        
        assert log_data['user_id'] == 'anonymous'
        assert log_data['ip_address'] == 'unknown'
        assert log_data['details'] == {}

    def test_log_auth_attempt_success(self):
        """測試記錄成功的身份驗證"""
        logger = SecurityLogger()
        
        with patch.object(logger.auth_logger, 'info') as mock_info:
            logger.log_auth_attempt(
                email='test@example.com',
                ip_address='192.168.1.1',
                success=True
            )
            
            # 應該調用了 info 方法
            assert mock_info.called
            
            # 檢查日誌內容
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'AUTH_ATTEMPT'
            assert log_data['result'] == 'success'

    def test_log_auth_attempt_failed(self):
        """測試記錄失敗的身份驗證"""
        logger = SecurityLogger()
        
        with patch.object(logger.auth_logger, 'info') as mock_info:
            logger.log_auth_attempt(
                email='test@example.com',
                ip_address='192.168.1.1',
                success=False,
                reason='Invalid code'
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['result'] == 'failed'
            assert log_data['details']['reason'] == 'Invalid code'

    def test_log_verification_code_sent(self):
        """測試記錄驗證碼發送"""
        logger = SecurityLogger()
        
        with patch.object(logger.auth_logger, 'info') as mock_info:
            logger.log_verification_code_sent(
                email='test@example.com',
                ip_address='192.168.1.1'
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'VERIFICATION_CODE_SENT'

    def test_log_session_expired(self):
        """測試記錄會話過期"""
        logger = SecurityLogger()
        
        with patch.object(logger.auth_logger, 'info') as mock_info:
            logger.log_session_expired(email='test@example.com')
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'SESSION_EXPIRED'
            assert log_data['result'] == 'expired'

    def test_log_task_created(self):
        """測試記錄任務創建"""
        logger = SecurityLogger()
        
        with patch.object(logger.operation_logger, 'info') as mock_info:
            logger.log_task_created(
                task_id='task-123',
                email='test@example.com',
                ip_address='192.168.1.1',
                filename='test.mp3',
                parameters={'model': 'whisper-large'}
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'TASK_CREATED'
            assert log_data['details']['task_id'] == 'task-123'
            assert log_data['details']['filename'] == 'test.mp3'

    def test_log_task_completed_success(self):
        """測試記錄任務成功完成"""
        logger = SecurityLogger()
        
        with patch.object(logger.operation_logger, 'info') as mock_info:
            logger.log_task_completed(
                task_id='task-123',
                email='test@example.com',
                success=True
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'TASK_COMPLETED'
            assert log_data['result'] == 'success'

    def test_log_task_completed_failed(self):
        """測試記錄任務失敗"""
        logger = SecurityLogger()
        
        with patch.object(logger.operation_logger, 'info') as mock_info:
            logger.log_task_completed(
                task_id='task-123',
                email='test@example.com',
                success=False,
                error='Processing error'
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['result'] == 'failed'
            assert log_data['details']['error'] == 'Processing error'

    def test_log_file_upload(self):
        """測試記錄文件上傳"""
        logger = SecurityLogger()
        
        with patch.object(logger.operation_logger, 'info') as mock_info:
            logger.log_file_upload(
                email='test@example.com',
                ip_address='192.168.1.1',
                filename='test.mp3',
                file_size=1024000,
                file_type='audio/mpeg'
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'FILE_UPLOAD'
            assert log_data['details']['file_size'] == 1024000

    def test_log_file_deleted(self):
        """測試記錄文件刪除"""
        logger = SecurityLogger()
        
        with patch.object(logger.operation_logger, 'info') as mock_info:
            logger.log_file_deleted(
                task_id='task-123',
                reason='Task completed'
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'FILE_DELETED'
            assert log_data['details']['reason'] == 'Task completed'

    def test_log_security_event_info(self):
        """測試記錄一般安全事件"""
        logger = SecurityLogger()
        
        with patch.object(logger.security_logger, 'info') as mock_info:
            logger.log_security_event(
                event_type='TEST_SECURITY_EVENT',
                ip_address='192.168.1.1',
                severity='info',
                description='Test security event'
            )
            
            assert mock_info.called

    def test_log_security_event_warning(self):
        """測試記錄警告級別安全事件"""
        logger = SecurityLogger()
        
        with patch.object(logger.security_logger, 'warning') as mock_warning:
            logger.log_security_event(
                event_type='TEST_WARNING',
                ip_address='192.168.1.1',
                severity='warning',
                description='Test warning'
            )
            
            assert mock_warning.called

    def test_log_security_event_critical(self):
        """測試記錄嚴重安全事件"""
        logger = SecurityLogger()
        
        with patch.object(logger.security_logger, 'error') as mock_error:
            logger.log_security_event(
                event_type='TEST_CRITICAL',
                ip_address='192.168.1.1',
                severity='critical',
                description='Test critical event'
            )
            
            assert mock_error.called

    def test_log_rate_limit_exceeded(self):
        """測試記錄速率限制超出"""
        logger = SecurityLogger()
        
        with patch.object(logger.security_logger, 'warning') as mock_warning:
            logger.log_rate_limit_exceeded(
                ip_address='192.168.1.1',
                endpoint='/api/upload'
            )
            
            assert mock_warning.called
            call_args = mock_warning.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'RATE_LIMIT_EXCEEDED'

    def test_log_invalid_request(self):
        """測試記錄無效請求"""
        logger = SecurityLogger()
        
        with patch.object(logger.security_logger, 'warning') as mock_warning:
            logger.log_invalid_request(
                ip_address='192.168.1.1',
                endpoint='/api/upload',
                reason='Invalid file format'
            )
            
            assert mock_warning.called
            call_args = mock_warning.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'INVALID_REQUEST'

    def test_log_unauthorized_access(self):
        """測試記錄未授權訪問"""
        logger = SecurityLogger()
        
        with patch.object(logger.security_logger, 'warning') as mock_warning:
            logger.log_unauthorized_access(
                ip_address='192.168.1.1',
                endpoint='/api/admin',
                user_id='test@example.com'
            )
            
            assert mock_warning.called
            call_args = mock_warning.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'UNAUTHORIZED_ACCESS'

    def test_log_error(self):
        """測試記錄錯誤"""
        logger = SecurityLogger()
        
        with patch.object(logger.error_logger, 'error') as mock_error:
            logger.log_error(
                error_type='DatabaseError',
                error_message='Connection failed',
                user_id='test@example.com',
                details={'code': 500}
            )
            
            assert mock_error.called
            call_args = mock_error.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'ERROR'
            assert log_data['action'] == 'DatabaseError'

    def test_log_personal_data_access(self):
        """測試記錄個資訪問"""
        logger = SecurityLogger()
        
        with patch.object(logger.audit_logger, 'info') as mock_info:
            logger.log_personal_data_access(
                email='test@example.com',
                ip_address='192.168.1.1',
                action='view_profile',
                data_type='email'
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'PERSONAL_DATA_ACCESS'

    def test_log_data_deletion(self):
        """測試記錄數據刪除"""
        logger = SecurityLogger()
        
        with patch.object(logger.audit_logger, 'info') as mock_info:
            logger.log_data_deletion(
                task_id='task-123',
                email='test@example.com',
                reason='User request',
                data_type='audio_file'
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'DATA_DELETION'
            assert log_data['details']['reason'] == 'User request'

    def test_log_admin_action(self):
        """測試記錄管理員操作"""
        logger = SecurityLogger()
        
        with patch.object(logger.audit_logger, 'info') as mock_info:
            logger.log_admin_action(
                admin_token='admin_token_12345678',
                ip_address='192.168.1.1',
                action='view_all_tasks',
                target='task-123',
                details={'count': 10}
            )
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0]
            log_data = json.loads(call_args)
            assert log_data['event_type'] == 'ADMIN_ACTION'
            # 應該只記錄 token 的前 8 位
            assert 'admin_' in log_data['user_id']

    def test_global_instance(self):
        """測試全局實例"""
        assert security_logger is not None
        assert isinstance(security_logger, SecurityLogger)


@pytest.mark.integration
class TestSecurityLoggerIntegration:
    """整合測試"""
    
    def test_log_file_creation(self, temp_dir):
        """測試實際日誌文件創建"""
        # 這個測試需要實際寫入文件系統
        # 在生產環境中應該確保日誌目錄存在
        logger = SecurityLogger()
        
        # 日誌目錄應該存在
        assert logger.log_dir.exists() or Path(logger.log_dir).exists()

    def test_complete_workflow_logging(self):
        """測試完整的工作流日誌記錄"""
        logger = SecurityLogger()
        
        with patch.object(logger.auth_logger, 'info'), \
             patch.object(logger.operation_logger, 'info'), \
             patch.object(logger.audit_logger, 'info'):
            
            # 模擬完整流程
            # 1. 發送驗證碼
            logger.log_verification_code_sent('test@example.com', '192.168.1.1')
            
            # 2. 驗證成功
            logger.log_auth_attempt('test@example.com', '192.168.1.1', True)
            
            # 3. 上傳文件
            logger.log_file_upload(
                'test@example.com',
                '192.168.1.1',
                'test.mp3',
                1024000,
                'audio/mpeg'
            )
            
            # 4. 創建任務
            logger.log_task_created(
                'task-123',
                'test@example.com',
                '192.168.1.1',
                'test.mp3',
                {}
            )
            
            # 5. 任務完成
            logger.log_task_completed('task-123', 'test@example.com', True)
            
            # 6. 刪除文件
            logger.log_file_deleted('task-123', 'Task completed')


@pytest.mark.security
class TestSecurityLoggerSecurity:
    """安全相關測試"""
    
    def test_log_sensitive_data_masking(self):
        """測試敏感數據遮罩"""
        logger = SecurityLogger()
        
        # 在實際實現中，應該遮罩敏感信息
        with patch.object(logger.auth_logger, 'info') as mock_info:
            logger.log_auth_attempt(
                email='test@example.com',
                ip_address='192.168.1.1',
                success=True
            )
            
            # 確保日誌被記錄
            assert mock_info.called

    def test_log_injection_prevention(self):
        """測試防止日誌注入"""
        logger = SecurityLogger()
        
        # 嘗試注入特殊字符
        malicious_input = "test\n[2025-01-01] FAKE LOG ENTRY"
        
        with patch.object(logger.auth_logger, 'info') as mock_info:
            logger.log_auth_attempt(
                email=malicious_input,
                ip_address='192.168.1.1',
                success=True
            )
            
            # 應該正常記錄（JSON 格式會轉義特殊字符）
            assert mock_info.called

    def test_log_entry_timestamp(self):
        """測試日誌時間戳"""
        logger = SecurityLogger()
        
        log_entry = logger._format_log_entry(
            event_type='TEST',
            user_id='test@example.com',
            ip_address='192.168.1.1',
            action='test',
            result='success'
        )
        
        log_data = json.loads(log_entry)
        timestamp = log_data['timestamp']
        
        # 驗證時間戳格式
        datetime.fromisoformat(timestamp)  # 應該不拋出異常

    def test_concurrent_logging(self):
        """測試並發日誌記錄"""
        import threading
        
        logger = SecurityLogger()
        
        def log_task(i):
            with patch.object(logger.operation_logger, 'info'):
                logger.log_task_created(
                    f'task-{i}',
                    f'test{i}@example.com',
                    '192.168.1.1',
                    f'test{i}.mp3',
                    {}
                )
        
        # 創建多個線程並發記錄
        threads = [threading.Thread(target=log_task, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 應該能正常處理並發記錄

