"""
測試輸入驗證模組 (input_validator.py)
"""
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, AsyncMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "remote_server"))

from input_validator import InputValidator, input_validator


class TestInputValidator:
    """測試 InputValidator 類"""

    def test_validate_email_valid(self):
        """測試有效的郵箱格式"""
        valid_emails = [
            "test@example.com",
            "user.name@example.com",
            "user+tag@example.co.uk",
            "test123@test-domain.com",
            "a@b.co"
        ]
        
        for email in valid_emails:
            is_valid, error = InputValidator.validate_email(email)
            assert is_valid is True, f"郵箱 {email} 應該有效"
            assert error is None

    def test_validate_email_invalid(self):
        """測試無效的郵箱格式"""
        invalid_emails = [
            "",
            "notanemail",
            "@example.com",
            "test@",
            "test @example.com",
            "test@exam ple.com",
            "test<script>@example.com",
            "test;@example.com"
        ]
        
        for email in invalid_emails:
            is_valid, error = InputValidator.validate_email(email)
            assert is_valid is False, f"郵箱 {email} 應該無效"
            assert error is not None

    def test_validate_email_too_long(self):
        """測試過長的郵箱"""
        long_email = "a" * 250 + "@example.com"
        is_valid, error = InputValidator.validate_email(long_email)
        assert is_valid is False
        assert "過長" in error

    def test_validate_email_dangerous_chars(self):
        """測試包含危險字符的郵箱"""
        dangerous_emails = [
            "test<@example.com",
            "test>@example.com",
            'test"@example.com',
            "test'@example.com",
            "test\\@example.com",
            "test;@example.com",
            "test&@example.com"
        ]
        
        for email in dangerous_emails:
            is_valid, error = InputValidator.validate_email(email)
            assert is_valid is False
            assert error is not None

    def test_validate_verification_code_valid(self):
        """測試有效的驗證碼"""
        valid_codes = ["123456", "000000", "999999"]
        
        for code in valid_codes:
            is_valid, error = InputValidator.validate_verification_code(code)
            assert is_valid is True
            assert error is None

    def test_validate_verification_code_invalid(self):
        """測試無效的驗證碼"""
        invalid_codes = [
            "",
            "12345",  # 太短
            "1234567",  # 太長
            "12345a",  # 包含字母
            "12-456",  # 包含特殊字符
        ]
        
        for code in invalid_codes:
            is_valid, error = InputValidator.validate_verification_code(code)
            assert is_valid is False
            assert error is not None

    def test_validate_task_id_valid(self):
        """測試有效的任務 ID（UUID）"""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11"
        ]
        
        for uuid in valid_uuids:
            is_valid, error = InputValidator.validate_task_id(uuid)
            assert is_valid is True
            assert error is None

    def test_validate_task_id_invalid(self):
        """測試無效的任務 ID"""
        invalid_uuids = [
            "",
            "not-a-uuid",
            "550e8400-e29b-41d4-a716",  # 不完整
            "550e8400-e29b-41d4-a716-446655440000-extra"  # 太長
        ]
        
        for uuid in invalid_uuids:
            is_valid, error = InputValidator.validate_task_id(uuid)
            assert is_valid is False
            assert error is not None

    def test_validate_filename_valid(self):
        """測試有效的文件名"""
        valid_filenames = [
            "test.mp3",
            "audio.wav",
            "recording.m4a",
            "file_123.flac",
            "語音檔案.ogg"
        ]
        
        for filename in valid_filenames:
            is_valid, error = InputValidator.validate_filename(filename)
            assert is_valid is True, f"文件名 {filename} 應該有效"
            assert error is None

    def test_validate_filename_invalid_extension(self):
        """測試不支持的文件擴展名"""
        invalid_filenames = [
            "test.txt",
            "audio.mp4",
            "file.exe",
            "script.sh"
        ]
        
        for filename in invalid_filenames:
            is_valid, error = InputValidator.validate_filename(filename)
            assert is_valid is False
            assert "不支持的文件格式" in error

    def test_validate_filename_path_traversal(self):
        """測試路徑遍歷攻擊"""
        malicious_filenames = [
            "../etc/passwd",
            "..\\windows\\system32",
            "test/../../../file.mp3",
            "/etc/passwd.mp3"
        ]
        
        for filename in malicious_filenames:
            is_valid, error = InputValidator.validate_filename(filename)
            assert is_valid is False
            assert error is not None

    def test_validate_filename_null_byte(self):
        """測試包含 null 字節的文件名"""
        filename_with_null = "test\x00.mp3"
        is_valid, error = InputValidator.validate_filename(filename_with_null)
        assert is_valid is False

    def test_validate_filename_too_long(self):
        """測試過長的文件名"""
        long_filename = "a" * 300 + ".mp3"
        is_valid, error = InputValidator.validate_filename(long_filename)
        assert is_valid is False
        assert "過長" in error

    @pytest.mark.asyncio
    async def test_validate_upload_file_valid(self, mock_upload_file):
        """測試有效的上傳文件"""
        is_valid, error = await InputValidator.validate_upload_file(mock_upload_file)
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_upload_file_empty(self):
        """測試空文件"""
        mock_file = MagicMock()
        mock_file.filename = "test.mp3"
        mock_file.read = AsyncMock(return_value=b"")
        mock_file.seek = AsyncMock()
        
        is_valid, error = await InputValidator.validate_upload_file(mock_file)
        assert is_valid is False
        assert "為空" in error

    @pytest.mark.asyncio
    async def test_validate_upload_file_too_large(self):
        """測試過大的文件"""
        mock_file = MagicMock()
        mock_file.filename = "test.mp3"
        # 創建一個超過限制的文件（> 500MB）
        mock_file.read = AsyncMock(return_value=b"x" * (501 * 1024 * 1024))
        mock_file.seek = AsyncMock()
        
        is_valid, error = await InputValidator.validate_upload_file(mock_file)
        assert is_valid is False
        assert "過大" in error

    @pytest.mark.asyncio
    async def test_validate_upload_file_no_file(self):
        """測試未提供文件"""
        is_valid, error = await InputValidator.validate_upload_file(None)
        assert is_valid is False
        assert "未提供文件" in error

    def test_check_audio_file_type_mp3(self):
        """測試 MP3 文件類型檢查"""
        mp3_headers = [
            b'ID3' + b'\x00' * 60,
            b'\xff\xfb' + b'\x00' * 60,
            b'\xff\xf3' + b'\x00' * 60
        ]
        
        for header in mp3_headers:
            is_valid = InputValidator._check_audio_file_type(header, "test.mp3")
            assert is_valid is True

    def test_check_audio_file_type_wav(self):
        """測試 WAV 文件類型檢查"""
        wav_header = b'RIFF' + b'\x00' * 60
        is_valid = InputValidator._check_audio_file_type(wav_header, "test.wav")
        assert is_valid is True

    def test_check_audio_file_type_m4a(self):
        """測試 M4A 文件類型檢查"""
        m4a_header = b'ftyp' + b'\x00' * 60
        is_valid = InputValidator._check_audio_file_type(m4a_header, "test.m4a")
        assert is_valid is True

    def test_check_audio_file_type_flac(self):
        """測試 FLAC 文件類型檢查"""
        flac_header = b'fLaC' + b'\x00' * 60
        is_valid = InputValidator._check_audio_file_type(flac_header, "test.flac")
        assert is_valid is True

    def test_check_audio_file_type_ogg(self):
        """測試 OGG 文件類型檢查"""
        ogg_header = b'OggS' + b'\x00' * 60
        is_valid = InputValidator._check_audio_file_type(ogg_header, "test.ogg")
        assert is_valid is True

    def test_check_audio_file_type_mismatch(self):
        """測試文件類型不匹配"""
        # MP3 標頭但聲稱是 WAV
        mp3_header = b'ID3' + b'\x00' * 60
        is_valid = InputValidator._check_audio_file_type(mp3_header, "test.wav")
        assert is_valid is False

    def test_validate_time_range_valid(self):
        """測試有效的時間範圍"""
        is_valid, error = InputValidator.validate_time_range(0.0, 100.0)
        assert is_valid is True
        assert error is None
        
        is_valid, error = InputValidator.validate_time_range(None, None)
        assert is_valid is True

    def test_validate_time_range_negative(self):
        """測試負數時間"""
        is_valid, error = InputValidator.validate_time_range(-10.0, 100.0)
        assert is_valid is False
        assert "負數" in error

    def test_validate_time_range_too_long(self):
        """測試超過 24 小時"""
        is_valid, error = InputValidator.validate_time_range(0.0, 90000.0)
        assert is_valid is False

    def test_validate_time_range_invalid_order(self):
        """測試開始時間大於結束時間"""
        is_valid, error = InputValidator.validate_time_range(100.0, 50.0)
        assert is_valid is False
        assert "必須小於" in error

    def test_validate_time_range_too_large_span(self):
        """測試時間範圍超過 4 小時"""
        is_valid, error = InputValidator.validate_time_range(0.0, 15000.0)
        assert is_valid is False
        assert "不能超過" in error

    def test_validate_language_valid(self):
        """測試有效的語言代碼"""
        valid_languages = ["zh", "en", "ja", "ko", "es", "fr"]
        
        for lang in valid_languages:
            is_valid, error = InputValidator.validate_language(lang)
            assert is_valid is True
            assert error is None

    def test_validate_language_none(self):
        """測試 None（自動檢測）"""
        is_valid, error = InputValidator.validate_language(None)
        assert is_valid is True

    def test_validate_language_invalid(self):
        """測試無效的語言代碼"""
        invalid_languages = ["xx", "invalid", "zh-CN"]
        
        for lang in invalid_languages:
            is_valid, error = InputValidator.validate_language(lang)
            assert is_valid is False
            assert "不支持的語言" in error

    def test_validate_task_type_valid(self):
        """測試有效的任務類型"""
        is_valid, error = InputValidator.validate_task_type("transcribe")
        assert is_valid is True
        
        is_valid, error = InputValidator.validate_task_type("translate")
        assert is_valid is True

    def test_validate_task_type_invalid(self):
        """測試無效的任務類型"""
        is_valid, error = InputValidator.validate_task_type("invalid")
        assert is_valid is False
        assert "無效的任務類型" in error

    def test_validate_vad_parameters_valid(self):
        """測試有效的 VAD 參數"""
        is_valid, error = InputValidator.validate_vad_parameters(0.5, 0.363)
        assert is_valid is True
        assert error is None

    def test_validate_vad_parameters_out_of_range(self):
        """測試超出範圍的 VAD 參數"""
        is_valid, error = InputValidator.validate_vad_parameters(1.5, 0.5)
        assert is_valid is False
        
        is_valid, error = InputValidator.validate_vad_parameters(0.5, -0.1)
        assert is_valid is False

    def test_validate_vad_parameters_invalid_order(self):
        """測試 VAD 參數順序錯誤"""
        is_valid, error = InputValidator.validate_vad_parameters(0.3, 0.5)
        assert is_valid is False
        assert "onset 不能小於" in error

    def test_validate_speaker_count_valid(self):
        """測試有效的語者數量"""
        is_valid, error = InputValidator.validate_speaker_count(1, 5)
        assert is_valid is True
        
        is_valid, error = InputValidator.validate_speaker_count(None, None)
        assert is_valid is True

    def test_validate_speaker_count_invalid(self):
        """測試無效的語者數量"""
        is_valid, error = InputValidator.validate_speaker_count(0, 5)
        assert is_valid is False
        
        is_valid, error = InputValidator.validate_speaker_count(1, 25)
        assert is_valid is False

    def test_validate_speaker_count_invalid_range(self):
        """測試最小值大於最大值"""
        is_valid, error = InputValidator.validate_speaker_count(5, 2)
        assert is_valid is False
        assert "不能大於" in error

    def test_validate_compute_type_valid(self):
        """測試有效的計算類型"""
        valid_types = ["float32", "float16", "int8", "int8_float16"]
        
        for compute_type in valid_types:
            is_valid, error = InputValidator.validate_compute_type(compute_type)
            assert is_valid is True

    def test_validate_compute_type_none(self):
        """測試 None"""
        is_valid, error = InputValidator.validate_compute_type(None)
        assert is_valid is True

    def test_validate_compute_type_invalid(self):
        """測試無效的計算類型"""
        is_valid, error = InputValidator.validate_compute_type("invalid")
        assert is_valid is False

    def test_validate_model_name_valid(self):
        """測試有效的模型名稱"""
        valid_models = [
            "whisper-large-v3",
            "Belle-whisper-large-v3-zh",
            "model_123"
        ]
        
        for model in valid_models:
            is_valid, error = InputValidator.validate_model_name(model)
            assert is_valid is True

    def test_validate_model_name_invalid(self):
        """測試無效的模型名稱"""
        invalid_models = [
            "",
            "model/../../../etc/passwd",
            "model;rm -rf /",
            "a" * 250
        ]
        
        for model in invalid_models:
            is_valid, error = InputValidator.validate_model_name(model)
            assert is_valid is False

    def test_sanitize_string_normal(self):
        """測試清理正常字符串"""
        text = "這是測試文本"
        sanitized = InputValidator.sanitize_string(text)
        assert sanitized == text

    def test_sanitize_string_with_newlines(self):
        """測試保留換行符"""
        text = "第一行\n第二行\t製表符"
        sanitized = InputValidator.sanitize_string(text)
        assert "\n" in sanitized
        assert "\t" in sanitized

    def test_sanitize_string_max_length(self):
        """測試最大長度限制"""
        text = "a" * 2000
        sanitized = InputValidator.sanitize_string(text, max_length=100)
        assert len(sanitized) == 100

    def test_sanitize_string_empty(self):
        """測試空字符串"""
        sanitized = InputValidator.sanitize_string("")
        assert sanitized == ""

    def test_validate_admin_token_valid(self):
        """測試有效的管理員 Token"""
        valid_token = "a" * 32
        is_valid, error = InputValidator.validate_admin_token(valid_token)
        assert is_valid is True

    def test_validate_admin_token_too_short(self):
        """測試過短的 Token"""
        short_token = "short"
        is_valid, error = InputValidator.validate_admin_token(short_token)
        assert is_valid is False
        assert "長度不足" in error

    def test_validate_admin_token_too_long(self):
        """測試過長的 Token"""
        long_token = "a" * 300
        is_valid, error = InputValidator.validate_admin_token(long_token)
        assert is_valid is False
        assert "過長" in error

    def test_validate_admin_token_dangerous_chars(self):
        """測試包含危險字符的 Token"""
        dangerous_tokens = [
            "token;rm -rf /",
            "token&whoami",
            "token|cat /etc/passwd"
        ]
        
        for token in dangerous_tokens:
            is_valid, error = InputValidator.validate_admin_token(token)
            assert is_valid is False

    def test_global_instance(self):
        """測試全局實例"""
        assert input_validator is not None
        assert isinstance(input_validator, InputValidator)

    def test_allowed_extensions(self):
        """測試允許的擴展名列表"""
        expected_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus'}
        assert InputValidator.ALLOWED_AUDIO_EXTENSIONS == expected_extensions

    def test_max_file_size(self):
        """測試最大文件大小常量"""
        assert InputValidator.MAX_FILE_SIZE == 500 * 1024 * 1024  # 500MB


@pytest.mark.security
class TestInputValidatorSecurity:
    """安全相關的特殊測試"""
    
    def test_sql_injection_in_email(self):
        """測試 SQL 注入嘗試"""
        malicious_emails = [
            "test' OR '1'='1@example.com",
            "admin'--@example.com",
            "test@example.com; DROP TABLE users--"
        ]
        
        for email in malicious_emails:
            is_valid, _ = InputValidator.validate_email(email)
            assert is_valid is False

    def test_xss_in_filename(self):
        """測試 XSS 攻擊嘗試"""
        malicious_filenames = [
            "<script>alert('xss')</script>.mp3",
            "test<img src=x onerror=alert(1)>.wav"
        ]
        
        for filename in malicious_filenames:
            is_valid, _ = InputValidator.validate_filename(filename)
            assert is_valid is False

    def test_command_injection_in_model_name(self):
        """測試命令注入嘗試"""
        malicious_models = [
            "model;ls -la",
            "model|cat /etc/passwd",
            "model`whoami`",
            "model$(rm -rf /)"
        ]
        
        for model in malicious_models:
            is_valid, _ = InputValidator.validate_model_name(model)
            assert is_valid is False

