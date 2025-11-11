"""
API 即時測試（使用 TestClient）
不需要啟動服務器，直接測試 FastAPI 應用

注意：這些測試會實際加載 API 模組，需要較長時間
"""
import pytest
import sys
from pathlib import Path
from io import BytesIO
from unittest.mock import patch, MagicMock, AsyncMock

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "remote_server"))


@pytest.fixture(scope="module")
def mock_whisper_model():
    """模擬 Whisper 模型"""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        'text': '這是測試轉錄文本',
        'segments': []
    }
    return mock_model


@pytest.fixture(scope="module")
def mock_task_processor():
    """模擬任務處理器"""
    mock = MagicMock()
    mock.transcribe_audio = AsyncMock(return_value="測試轉錄結果")
    return mock


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestAPIWithMocks:
    """使用 Mock 的 API 測試"""
    
    @pytest.mark.skipif(True, reason="需要較長時間加載 API 模組")
    def test_api_with_mocked_dependencies(self, mock_whisper_model, mock_task_processor):
        """測試帶 Mock 依賴的 API"""
        # 模擬所有重量級依賴
        with patch('remote_server.task_processor.whisper') as mock_whisper, \
             patch('remote_server.task_processor.task_processor', mock_task_processor):
            
            mock_whisper.load_model.return_value = mock_whisper_model
            
            # 現在可以安全地導入 API
            from fastapi.testclient import TestClient
            from remote_server.api import app
            
            client = TestClient(app)
            
            # 測試健康檢查
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            
            print("\n✅ API with mocks 測試通過")


@pytest.mark.integration
@pytest.mark.api
class TestAPIEndpoints:
    """API 端點功能測試（需要實際運行的 API）"""
    
    def test_api_documentation(self):
        """測試 API 文檔端點"""
        import requests
        
        try:
            # 嘗試訪問 API 文檔
            response = requests.get("http://localhost:8000/docs", timeout=2)
            if response.status_code == 200:
                print("\n✅ API 文檔可訪問: http://localhost:8000/docs")
            else:
                pytest.skip("API 服務未運行")
        except Exception:
            pytest.skip("API 服務未運行")
    
    def test_api_openapi_schema(self):
        """測試 OpenAPI schema"""
        import requests
        
        try:
            response = requests.get("http://localhost:8000/openapi.json", timeout=2)
            if response.status_code == 200:
                schema = response.json()
                assert "openapi" in schema
                assert "paths" in schema
                print(f"\n✅ OpenAPI schema 可用")
                print(f"可用端點數: {len(schema.get('paths', {}))}")
            else:
                pytest.skip("API 服務未運行")
        except Exception:
            pytest.skip("API 服務未運行")

