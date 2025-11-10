"""
Ollama LLM 校對服務模組
使用 Ollama API 對轉錄文本進行校對和改正
"""
import requests
from typing import Optional, Dict
import threading


class OllamaService:
    """Ollama LLM 校對服務類別"""

    _instance = None
    _lock = threading.Lock()

    # 可用的模型列表
    AVAILABLE_MODELS = [
        "gpt-oss:20b",
        "gemma3:4b",
        "qwen3:4b"
    ]

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.ollama_url = "http://localhost:11434/api/generate"
            self.initialized = True

    def get_correction_prompt(self, text: str, has_diarization: bool = False) -> str:
        """
        生成高效的校對提示詞
        針對低參數量 LLM 優化的提示工程
        """
        if has_diarization:
            # 帶語者分離的格式
            prompt = f"""你是一個專業的文本校對助手。請仔細檢查並修正以下語音轉錄文本中的錯誤。

**重要規則：**
1. 保持原文的語者標記格式（SPEAKER_XX）不變
2. 僅修正明顯的錯字、語法錯誤、標點符號、中文近似音錯置問題
3. 保持原文的時間戳格式不變
4. 如果某句話完全正確，保持不變
5. 直接輸出校正後的完整文本，不要添加任何說明或註解

**原文：**
{text}

**校正後文本：**"""
        else:
            # 純文本格式
            prompt = f"""你是一個專業的文本校對助手。請仔細檢查並修正以下語音轉錄文本中的錯誤。

**重要規則：**
1. 僅修正明顯的錯字、語法錯誤、標點符號、中文近似音錯置問題
2. 保持原文的時間戳格式不變（如果有）
3. 如果某句話完全正確，保持不變
4. 直接輸出校正後的完整文本，不要添加任何說明或註解

**原文：**
{text}

**校正後文本：**"""

        return prompt

    def correct_text(
        self,
        text: str,
        model: str = "gemma3:4b",
        has_diarization: bool = False,
        progress_callback: Optional[callable] = None,
        _force_direct: bool = False
    ) -> Dict[str, str]:
        """
        使用 Ollama LLM 校對文本

        Args:
            text: 需要校對的原始文本
            model: 使用的 LLM 模型
            has_diarization: 是否包含語者分離信息
            progress_callback: 進度回調函數
            _force_direct: 強制直接處理，不進行分段（內部使用）

        Returns:
            Dict 包含 original（原文）和 corrected（校正後）
        """
        # 驗證模型
        if model not in self.AVAILABLE_MODELS:
            print(f"警告: 模型 {model} 不在可用列表中，使用默認模型 gemma3:4b")
            model = "gemma3:4b"

        # 分段處理長文本（每段最多 2000 字）
        if len(text) > 2000 and not _force_direct:
            return self._correct_long_text(text, model, has_diarization, progress_callback)

        try:
            if progress_callback:
                progress_callback("準備 LLM 校對...")

            prompt = self.get_correction_prompt(text, has_diarization)

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # 低溫度，更保守的修正
                    "top_p": 0.9,
                    "top_k": 20,
                    "num_predict": len(text) + 500  # 預測長度稍大於原文
                }
            }

            if progress_callback:
                progress_callback(f"正在使用 {model} 進行校對...")

            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=300  # 5分鐘超時
            )

            if response.status_code == 200:
                result = response.json()
                corrected_text = result.get('response', '').strip()

                if progress_callback:
                    progress_callback("LLM 校對完成")

                return {
                    'original': text,
                    'corrected': corrected_text if corrected_text else text
                }
            else:
                print(f"Ollama API 錯誤: {response.status_code} - {response.text}")
                return {
                    'original': text,
                    'corrected': text,
                    'error': f"API 錯誤: {response.status_code}"
                }

        except requests.exceptions.Timeout:
            print("Ollama API 請求超時")
            return {
                'original': text,
                'corrected': text,
                'error': "請求超時"
            }
        except requests.exceptions.ConnectionError:
            print("無法連接到 Ollama 服務，請確保 Ollama 正在運行")
            return {
                'original': text,
                'corrected': text,
                'error': "無法連接到 Ollama"
            }
        except Exception as e:
            print(f"LLM 校對時發生錯誤: {e}")
            return {
                'original': text,
                'corrected': text,
                'error': str(e)
            }

    def _correct_long_text(
        self,
        text: str,
        model: str,
        has_diarization: bool,
        progress_callback: Optional[callable]
    ) -> Dict[str, str]:
        """
        分段處理長文本
        """
        # 按段落分割（保持語者分離的完整性）
        if has_diarization:
            # 按 SPEAKER 標記分段
            lines = text.split('\n')
            segments = []
            current_segment = []

            for line in lines:
                current_segment.append(line)
                if len('\n'.join(current_segment)) > 1500:  # 每段約1500字
                    segments.append('\n'.join(current_segment))
                    current_segment = []

            if current_segment:
                segments.append('\n'.join(current_segment))
        else:
            # 按句子分段
            sentences = text.replace('。', '。\n').replace('？', '？\n').replace('！', '！\n').split('\n')
            segments = []
            current_segment = []

            for sentence in sentences:
                current_segment.append(sentence)
                if len(''.join(current_segment)) > 1500:
                    segments.append(''.join(current_segment))
                    current_segment = []

            if current_segment:
                segments.append(''.join(current_segment))

        # 逐段校對
        corrected_segments = []
        total_segments = len(segments)

        for i, segment in enumerate(segments):
            if progress_callback:
                progress_callback(f"LLM 校對進度: {i + 1}/{total_segments}")

            # 使用 _force_direct=True 避免遞歸
            result = self.correct_text(segment, model, has_diarization, None, _force_direct=True)
            corrected_segments.append(result['corrected'])

        # 合併結果
        corrected_text = '\n'.join(corrected_segments) if has_diarization else ''.join(corrected_segments)

        return {
            'original': text,
            'corrected': corrected_text
        }

    def generate_comparison_html(
        self,
        original: str,
        corrected: str,
        has_diarization: bool = False
    ) -> str:
        """
        生成校對前後對比的 HTML 報告
        """
        html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM 校對報告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Microsoft JhengHei', 'PingFang TC', sans-serif;
            line-height: 1.8;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 16px;
            opacity: 0.9;
        }}
        .comparison-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            background: #e0e0e0;
        }}
        .panel {{
            background: white;
            padding: 30px;
        }}
        .panel-header {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .original-header {{
            color: #ff6b6b;
        }}
        .corrected-header {{
            color: #51cf66;
        }}
        .content {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 16px;
            color: #333;
        }}
        .speaker {{
            color: #667eea;
            font-weight: bold;
            margin-top: 15px;
            display: block;
        }}
        @media (max-width: 768px) {{
            .comparison-container {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 LLM 校對報告</h1>
            <p>語音轉錄文本校對結果對比</p>
        </div>
        <div class="comparison-container">
            <div class="panel">
                <div class="panel-header original-header">📝 校對前（原始轉錄）</div>
                <div class="content">{self._format_text_for_html(original, has_diarization)}</div>
            </div>
            <div class="panel">
                <div class="panel-header corrected-header">✅ 校對後（LLM 修正）</div>
                <div class="content">{self._format_text_for_html(corrected, has_diarization)}</div>
            </div>
        </div>
    </div>
</body>
</html>"""
        return html

    def _format_text_for_html(self, text: str, has_diarization: bool) -> str:
        """格式化文本用於 HTML 顯示"""
        # HTML 轉義
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        if has_diarization:
            # 高亮語者標記
            import re
            text = re.sub(
                r'(SPEAKER_\d+)',
                r'<span class="speaker">\1</span>',
                text
            )

        return text

    def check_ollama_availability(self) -> bool:
        """檢查 Ollama 服務是否可用"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# 創建全局單例實例
ollama_service = OllamaService()
