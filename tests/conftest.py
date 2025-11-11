"""
pytest 配置和共用 fixtures
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock
from datetime import datetime, timedelta

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_server"))

# 設置測試環境變數
os.environ['TESTING'] = '1'
os.environ['ENCRYPTION_KEY'] = 'test_encryption_key_32_bytes_long_123456789012='
os.environ['EMAIL_HASH_SALT'] = 'test_salt_for_testing'
os.environ['ADMIN_TOKEN'] = 'test_admin_token_for_testing'
os.environ['SMTP_SERVER'] = 'smtp.test.com'
os.environ['SMTP_PORT'] = '587'
os.environ['SMTP_USERNAME'] = 'test@test.com'
os.environ['SMTP_PASSWORD'] = 'test_password'
os.environ['FROM_EMAIL'] = 'test@test.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


@pytest.fixture(scope="session")
def temp_dir():
    """創建臨時目錄用於測試"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # 清理
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def clean_temp_dir(temp_dir):
    """每個測試前清理臨時目錄"""
    for item in temp_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    yield temp_dir


@pytest.fixture
def mock_datetime():
    """模擬的 datetime 物件"""
    mock_dt = MagicMock()
    mock_dt.now.return_value = datetime(2025, 1, 1, 12, 0, 0)
    mock_dt.fromisoformat.side_effect = lambda x: datetime.fromisoformat(x)
    return mock_dt


@pytest.fixture
def sample_audio_file(temp_dir):
    """創建測試用的模擬音訊檔案"""
    audio_path = temp_dir / "test_audio.mp3"
    # MP3 檔案標頭（ID3）
    mp3_header = b'ID3' + b'\x03\x00\x00\x00\x00\x00\x00'
    with open(audio_path, 'wb') as f:
        f.write(mp3_header)
        f.write(b'\x00' * 1024)  # 1KB 的模擬資料
    return audio_path


@pytest.fixture
def sample_wav_file(temp_dir):
    """創建測試用的 WAV 檔案"""
    wav_path = temp_dir / "test_audio.wav"
    # WAV 檔案標頭（RIFF）
    wav_header = b'RIFF' + b'\x00\x00\x00\x00' + b'WAVE'
    with open(wav_path, 'wb') as f:
        f.write(wav_header)
        f.write(b'\x00' * 1024)
    return wav_path


@pytest.fixture
def valid_email():
    """有效的測試郵箱"""
    return "test@example.com"


@pytest.fixture
def valid_task_id():
    """有效的任務 ID（UUID 格式）"""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def valid_verification_code():
    """有效的驗證碼"""
    return "123456"


@pytest.fixture
def mock_smtp_server():
    """模擬 SMTP 伺服器"""
    mock_server = MagicMock()
    mock_server.starttls = MagicMock()
    mock_server.login = MagicMock()
    mock_server.send_message = MagicMock()
    mock_server.__enter__ = MagicMock(return_value=mock_server)
    mock_server.__exit__ = MagicMock(return_value=False)
    return mock_server


@pytest.fixture
def mock_request():
    """模擬 FastAPI Request 物件"""
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {}
    return request


@pytest.fixture
def mock_upload_file(sample_audio_file):
    """模擬 UploadFile 物件"""
    from io import BytesIO
    
    mock_file = MagicMock()
    mock_file.filename = "test_audio.mp3"
    
    with open(sample_audio_file, 'rb') as f:
        content = f.read()
    
    mock_file.read = Mock(return_value=content)
    mock_file.seek = Mock()
    mock_file.file = BytesIO(content)
    
    return mock_file


@pytest.fixture(autouse=True)
def reset_singletons():
    """每個測試後重置單例實例"""
    yield
    
    # 重置所有單例
    from remote_server import crypto_utils, rate_limiter, security_logger, memory_storage, email_service
    
    # 清理 RateLimiter 的內部數據
    if hasattr(rate_limiter.RateLimiter, '_instance') and rate_limiter.RateLimiter._instance is not None:
        instance = rate_limiter.RateLimiter._instance
        if hasattr(instance, 'ip_requests'):
            instance.ip_requests.clear()
        if hasattr(instance, 'email_verification_failures'):
            instance.email_verification_failures.clear()
        if hasattr(instance, 'email_code_requests'):
            instance.email_code_requests.clear()
        if hasattr(instance, 'ip_blacklist'):
            instance.ip_blacklist.clear()
        if hasattr(instance, 'email_blacklist'):
            instance.email_blacklist.clear()
    
    # 清理單例狀態
    if hasattr(crypto_utils.CryptoUtils, '_instance'):
        crypto_utils.CryptoUtils._instance = None
    
    if hasattr(rate_limiter.RateLimiter, '_instance'):
        rate_limiter.RateLimiter._instance = None
    
    if hasattr(security_logger.SecurityLogger, '_instance'):
        security_logger.SecurityLogger._instance = None
    
    if hasattr(memory_storage.MemoryTaskManager, '_instance'):
        memory_storage.MemoryTaskManager._instance = None
    
    if hasattr(email_service.EmailService, '_instance'):
        email_service.EmailService._instance = None


@pytest.fixture
def api_client():
    """FastAPI 測試客戶端"""
    from fastapi.testclient import TestClient
    # 延遲導入以避免初始化問題
    import remote_server.api as api_module
    
    client = TestClient(api_module.app)
    return client


# 輔助函數
def create_mock_task_data(
    task_id: str = "test-task-id",
    email: str = "test@example.com",
    status: str = "pending"
):
    """創建模擬任務資料"""
    return {
        'task_id': task_id,
        'email': email,
        'filename': 'test_audio.mp3',
        'status': status,
        'progress': 0.0,
        'current_stage': None,
        'enable_diarization': True,
        'created_at': datetime.now().isoformat(),
        'started_at': None,
        'completed_at': None,
        'error_message': None,
        'partial_result': []
    }

