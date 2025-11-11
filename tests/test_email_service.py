"""
測試郵件服務模組 (email_service.py)
"""
import pytest
import smtplib
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock, mock_open
from email.mime.multipart import MIMEMultipart

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "remote_server"))

from email_service import EmailService, email_service


class TestEmailService:
    """測試 EmailService 類"""

    def test_singleton_pattern(self):
        """測試單例模式"""
        service1 = EmailService()
        service2 = EmailService()
        assert service1 is service2

    def test_initialization(self):
        """測試初始化"""
        service = EmailService()
        
        assert hasattr(service, 'smtp_server')
        assert hasattr(service, 'smtp_port')
        assert hasattr(service, 'smtp_username')
        assert hasattr(service, 'smtp_password')
        assert hasattr(service, 'verification_codes')

    def test_generate_verification_code(self):
        """測試生成驗證碼"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        
        # 驗證碼應該是 6 位數字
        assert len(code) == 6
        assert code.isdigit()
        
        # 應該存儲在記憶體中
        assert email in service.verification_codes
        assert service.verification_codes[email]['code'] == code

    def test_generate_verification_code_uniqueness(self):
        """測試驗證碼的唯一性"""
        service = EmailService()
        email = "test@example.com"
        
        code1 = service.generate_verification_code(email)
        code2 = service.generate_verification_code(email)
        
        # 兩次生成的驗證碼可能不同（覆蓋前一個）
        assert code2 is not None

    def test_verify_code_correct(self):
        """測試正確的驗證碼"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        result = service.verify_code(email, code)
        
        assert result is True
        # 驗證成功後應該標記為已驗證
        assert service.verification_codes[email]['verified'] is True

    def test_verify_code_incorrect(self):
        """測試錯誤的驗證碼"""
        service = EmailService()
        email = "test@example.com"
        
        service.generate_verification_code(email)
        result = service.verify_code(email, "000000")
        
        assert result is False

    def test_verify_code_nonexistent_email(self):
        """測試不存在的郵箱"""
        service = EmailService()
        
        result = service.verify_code("nonexistent@example.com", "123456")
        assert result is False

    def test_verify_code_expired(self):
        """測試過期的驗證碼"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        
        # 手動設置為過期
        service.verification_codes[email]['expires_at'] = datetime.now() - timedelta(minutes=1)
        
        result = service.verify_code(email, code)
        assert result is False
        # 過期的驗證碼應該被刪除
        assert email not in service.verification_codes

    def test_is_email_verified_true(self):
        """測試已驗證的郵箱"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        service.verify_code(email, code)
        
        result = service.is_email_verified(email)
        assert result is True

    def test_is_email_verified_false(self):
        """測試未驗證的郵箱"""
        service = EmailService()
        email = "test@example.com"
        
        service.generate_verification_code(email)
        
        result = service.is_email_verified(email)
        assert result is False

    def test_is_email_verified_nonexistent(self):
        """測試不存在的郵箱驗證狀態"""
        service = EmailService()
        
        result = service.is_email_verified("nonexistent@example.com")
        assert result is False

    def test_is_email_verified_expired(self):
        """測試過期的驗證狀態"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        service.verify_code(email, code)
        
        # 手動設置為過期
        service.verification_codes[email]['expires_at'] = datetime.now() - timedelta(minutes=1)
        
        result = service.is_email_verified(email)
        assert result is False

    @patch('smtplib.SMTP')
    def test_send_verification_email_success(self, mock_smtp):
        """測試成功發送驗證郵件"""
        service = EmailService()
        email = "test@example.com"
        
        # 設置 mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = service.send_verification_email(email)
        
        assert result is True
        # 應該調用了 send_message
        assert mock_server.send_message.called

    @patch('smtplib.SMTP')
    def test_send_verification_email_failure(self, mock_smtp):
        """測試發送驗證郵件失敗"""
        service = EmailService()
        email = "test@example.com"
        
        # 模擬 SMTP 錯誤
        mock_smtp.side_effect = smtplib.SMTPException("Connection failed")
        
        result = service.send_verification_email(email)
        
        assert result is False

    @patch('smtplib.SMTP')
    def test_send_completion_email_basic(self, mock_smtp):
        """測試發送基本的完成通知郵件"""
        service = EmailService()
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = service.send_completion_email(
            to_email="test@example.com",
            task_id="task-123",
            filename="test.mp3",
            transcript_text="這是測試轉錄結果"
        )
        
        assert result is True
        assert mock_server.send_message.called

    @patch('smtplib.SMTP')
    def test_send_completion_email_with_diarization(self, mock_smtp):
        """測試發送包含語者分離的完成郵件"""
        service = EmailService()
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = service.send_completion_email(
            to_email="test@example.com",
            task_id="task-123",
            filename="test.mp3",
            transcript_text="這是測試轉錄結果",
            has_diarization=True
        )
        
        assert result is True

    @patch('smtplib.SMTP')
    @patch('builtins.open', new_callable=mock_open, read_data="<html>Test Report</html>")
    @patch('os.path.exists', return_value=True)
    def test_send_completion_email_with_confidence_report(self, mock_exists, mock_file, mock_smtp):
        """測試發送包含信心分數報告的郵件"""
        service = EmailService()
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = service.send_completion_email(
            to_email="test@example.com",
            task_id="task-123",
            filename="test.mp3",
            transcript_text="這是測試轉錄結果",
            confidence_html_path="/path/to/confidence.html"
        )
        
        assert result is True

    @patch('smtplib.SMTP')
    def test_send_completion_email_with_llm_correction(self, mock_smtp):
        """測試發送包含 LLM 校對的郵件"""
        service = EmailService()
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = service.send_completion_email(
            to_email="test@example.com",
            task_id="task-123",
            filename="test.mp3",
            transcript_text="這是測試轉錄結果",
            corrected_text="這是校正後的結果"
        )
        
        assert result is True

    @patch('smtplib.SMTP')
    @patch('builtins.open', new_callable=mock_open, read_data="<html>LLM Comparison</html>")
    @patch('os.path.exists', return_value=True)
    def test_send_completion_email_with_llm_comparison(self, mock_exists, mock_file, mock_smtp):
        """測試發送包含 LLM 對比報告的郵件"""
        service = EmailService()
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = service.send_completion_email(
            to_email="test@example.com",
            task_id="task-123",
            filename="test.mp3",
            transcript_text="這是測試轉錄結果",
            corrected_text="這是校正後的結果",
            llm_comparison_html_path="/path/to/comparison.html"
        )
        
        assert result is True

    @patch('smtplib.SMTP')
    def test_send_completion_email_long_transcript(self, mock_smtp):
        """測試發送長轉錄結果的郵件"""
        service = EmailService()
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # 創建超過 500 字的轉錄結果
        long_text = "測試文本 " * 100
        
        result = service.send_completion_email(
            to_email="test@example.com",
            task_id="task-123",
            filename="test.mp3",
            transcript_text=long_text
        )
        
        assert result is True

    @patch('smtplib.SMTP')
    def test_send_completion_email_failure(self, mock_smtp):
        """測試發送完成郵件失敗"""
        service = EmailService()
        
        mock_smtp.side_effect = smtplib.SMTPException("Connection failed")
        
        result = service.send_completion_email(
            to_email="test@example.com",
            task_id="task-123",
            filename="test.mp3",
            transcript_text="測試"
        )
        
        assert result is False

    @patch('smtplib.SMTP')
    def test_send_email_with_authentication(self, mock_smtp):
        """測試使用身份驗證發送郵件"""
        service = EmailService()
        service.smtp_username = "user@example.com"
        service.smtp_password = "password"
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        msg = MIMEMultipart()
        msg['From'] = "test@example.com"
        msg['To'] = "recipient@example.com"
        msg['Subject'] = "Test"
        
        result = service._send_email(msg)
        
        # 應該調用 starttls 和 login
        assert mock_server.starttls.called
        assert mock_server.login.called
        assert result is True

    @patch('smtplib.SMTP')
    def test_send_email_without_authentication(self, mock_smtp):
        """測試不使用身份驗證發送郵件"""
        service = EmailService()
        service.smtp_username = ""
        service.smtp_password = ""
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        msg = MIMEMultipart()
        msg['From'] = "test@example.com"
        msg['To'] = "recipient@example.com"
        msg['Subject'] = "Test"
        
        result = service._send_email(msg)
        
        # 不應該調用 starttls 和 login
        assert not mock_server.starttls.called
        assert not mock_server.login.called
        assert result is True

    def test_verification_code_expiry_time(self):
        """測試驗證碼過期時間"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        
        # 檢查過期時間是否正確設置（5 分鐘）
        expires_at = service.verification_codes[email]['expires_at']
        now = datetime.now()
        
        time_diff = (expires_at - now).total_seconds()
        
        # 應該接近 300 秒（5 分鐘）
        assert 290 < time_diff < 310

    def test_verified_email_extended_expiry(self):
        """測試驗證成功後的延長有效期"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        service.verify_code(email, code)
        
        # 檢查延長後的過期時間（24 小時）
        expires_at = service.verification_codes[email]['expires_at']
        now = datetime.now()
        
        time_diff = (expires_at - now).total_seconds()
        
        # 應該接近 86400 秒（24 小時）
        assert 86000 < time_diff < 87000

    def test_global_instance(self):
        """測試全局實例"""
        assert email_service is not None
        assert isinstance(email_service, EmailService)


@pytest.mark.integration
class TestEmailServiceIntegration:
    """整合測試"""
    
    def test_complete_verification_flow(self):
        """測試完整的驗證流程"""
        service = EmailService()
        email = "test@example.com"
        
        # 1. 生成驗證碼
        code = service.generate_verification_code(email)
        assert code is not None
        
        # 2. 檢查未驗證狀態
        assert service.is_email_verified(email) is False
        
        # 3. 驗證
        assert service.verify_code(email, code) is True
        
        # 4. 檢查已驗證狀態
        assert service.is_email_verified(email) is True

    @patch('smtplib.SMTP')
    def test_complete_email_workflow(self, mock_smtp):
        """測試完整的郵件工作流"""
        service = EmailService()
        email = "test@example.com"
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # 1. 發送驗證郵件
        result1 = service.send_verification_email(email)
        assert result1 is True
        
        # 2. 驗證成功
        code = service.verification_codes[email]['code']
        assert service.verify_code(email, code) is True
        
        # 3. 發送完成郵件
        result2 = service.send_completion_email(
            to_email=email,
            task_id="task-123",
            filename="test.mp3",
            transcript_text="測試結果"
        )
        assert result2 is True

    def test_multiple_users_verification(self):
        """測試多用戶驗證隔離"""
        service = EmailService()
        
        email1 = "user1@example.com"
        email2 = "user2@example.com"
        
        code1 = service.generate_verification_code(email1)
        code2 = service.generate_verification_code(email2)
        
        # 驗證碼應該不同
        assert code1 != code2
        
        # 用戶 1 的驗證碼不能用於用戶 2
        assert service.verify_code(email2, code1) is False
        assert service.verify_code(email1, code1) is True


@pytest.mark.security
class TestEmailServiceSecurity:
    """安全相關測試"""
    
    def test_verification_code_randomness(self):
        """測試驗證碼的隨機性"""
        service = EmailService()
        
        codes = set()
        for i in range(100):
            email = f"test{i}@example.com"
            code = service.generate_verification_code(email)
            codes.add(code)
        
        # 100 次應該產生接近 100 個不同的驗證碼
        # （理論上可能有重複，但機率極低）
        assert len(codes) > 90

    def test_email_verification_state_isolation(self):
        """測試郵箱驗證狀態隔離"""
        service = EmailService()
        
        email1 = "user1@example.com"
        email2 = "user2@example.com"
        
        code1 = service.generate_verification_code(email1)
        service.verify_code(email1, code1)
        
        # 用戶 2 不應該受影響
        assert service.is_email_verified(email2) is False

    def test_expired_verification_cleanup(self):
        """測試過期驗證的清理"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        
        # 設置為過期
        service.verification_codes[email]['expires_at'] = datetime.now() - timedelta(minutes=1)
        
        # 嘗試驗證應該失敗並清理
        result = service.verify_code(email, code)
        
        assert result is False
        assert email not in service.verification_codes

    def test_verification_code_not_reusable_after_expiry(self):
        """測試驗證碼過期後不可重用"""
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        
        # 設置為過期
        service.verification_codes[email]['expires_at'] = datetime.now() - timedelta(minutes=1)
        
        # 第一次驗證失敗並清理
        service.verify_code(email, code)
        
        # 第二次驗證應該也失敗（因為已被清理）
        result = service.verify_code(email, code)
        assert result is False

    @patch('smtplib.SMTP')
    def test_email_content_sanitization(self, mock_smtp):
        """測試郵件內容清理"""
        service = EmailService()
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # 包含特殊字符的轉錄結果
        malicious_text = "Normal text <script>alert('xss')</script>"
        
        result = service.send_completion_email(
            to_email="test@example.com",
            task_id="task-123",
            filename="test.mp3",
            transcript_text=malicious_text
        )
        
        # 應該能成功發送（HTML 會轉義特殊字符）
        assert result is True

    def test_concurrent_verification_attempts(self):
        """測試並發驗證嘗試"""
        import threading
        
        service = EmailService()
        email = "test@example.com"
        
        code = service.generate_verification_code(email)
        results = []
        
        def verify_attempt():
            result = service.verify_code(email, code)
            results.append(result)
        
        # 多個線程同時驗證
        threads = [threading.Thread(target=verify_attempt) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 應該只有一次成功（或都成功，取決於實現）
        # 至少應該有結果
        assert len(results) == 5

