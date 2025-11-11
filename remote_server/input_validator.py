"""
輸入驗證模組
符合 SSDLC 要求：驗證所有用戶輸入，防止注入攻擊
"""
import re
import os
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException


class InputValidator:
    """輸入驗證器"""

    # 允許的文件擴展名
    ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus'}

    # 最大文件大小（500MB）
    MAX_FILE_SIZE = 500 * 1024 * 1024

    # 郵箱格式
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    # 驗證碼格式（6 位數字）
    VERIFICATION_CODE_PATTERN = re.compile(r'^\d{6}$')

    # UUID 格式
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )

    # 文件名安全字符（防止路徑遍歷）
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._\-\u4e00-\u9fff]+$')

    # 允許的語言代碼
    ALLOWED_LANGUAGES = {
        'zh', 'en', 'ja', 'ko', 'es', 'fr', 'de', 'ru', 'ar', 'pt', 'it',
        'nl', 'pl', 'tr', 'vi', 'id', 'th', 'ms', 'fil', 'uk', 'el'
    }

    # 允許的任務類型
    ALLOWED_TASKS = {'transcribe', 'translate'}

    # 允許的計算類型
    ALLOWED_COMPUTE_TYPES = {'float32', 'float16', 'int8', 'int8_float16'}

    @staticmethod
    def validate_email(email: str) -> Tuple[bool, Optional[str]]:
        """
        驗證郵箱格式

        Returns:
            (is_valid, error_message)
        """
        if not email:
            return False, "郵箱地址不能為空"

        if len(email) > 255:
            return False, "郵箱地址過長"

        if not InputValidator.EMAIL_PATTERN.match(email):
            return False, "無效的郵箱格式"

        # 檢查是否包含危險字符
        dangerous_chars = ['<', '>', '"', "'", '\\', ';', '&', '|', '$', '`']
        if any(char in email for char in dangerous_chars):
            return False, "郵箱包含非法字符"

        return True, None

    @staticmethod
    def validate_verification_code(code: str) -> Tuple[bool, Optional[str]]:
        """
        驗證驗證碼格式

        Returns:
            (is_valid, error_message)
        """
        if not code:
            return False, "驗證碼不能為空"

        if not InputValidator.VERIFICATION_CODE_PATTERN.match(code):
            return False, "驗證碼必須是 6 位數字"

        return True, None

    @staticmethod
    def validate_task_id(task_id: str) -> Tuple[bool, Optional[str]]:
        """
        驗證任務 ID 格式（UUID）

        Returns:
            (is_valid, error_message)
        """
        if not task_id:
            return False, "任務 ID 不能為空"

        if not InputValidator.UUID_PATTERN.match(task_id):
            return False, "無效的任務 ID 格式"

        return True, None

    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
        """
        驗證文件名安全性

        Returns:
            (is_valid, error_message)
        """
        if not filename:
            return False, "文件名不能為空"

        # 檢查長度
        if len(filename) > 255:
            return False, "文件名過長（最多 255 字符）"

        # 檢查路徑遍歷攻擊（在移除路徑之前檢查）
        if '..' in filename or '/' in filename or '\\' in filename:
            return False, "文件名包含非法字符"

        # 檢查是否包含 null 字節
        if '\x00' in filename:
            return False, "文件名包含非法字符"

        # 移除路徑部分，只保留文件名
        filename = os.path.basename(filename)

        # 檢查文件擴展名
        ext = Path(filename).suffix.lower()
        if ext not in InputValidator.ALLOWED_AUDIO_EXTENSIONS:
            return False, f"不支持的文件格式。允許的格式: {', '.join(InputValidator.ALLOWED_AUDIO_EXTENSIONS)}"

        return True, None

    @staticmethod
    async def validate_upload_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
        """
        驗證上傳文件

        Returns:
            (is_valid, error_message)
        """
        if not file:
            return False, "未提供文件"

        if not file.filename:
            return False, "文件名為空"

        # 驗證文件名
        is_valid, error = InputValidator.validate_filename(file.filename)
        if not is_valid:
            return False, error

        # 檢查文件大小
        # 注意：這需要讀取文件內容，會消耗內存
        content = await file.read()
        file_size = len(content)

        # 重置文件指針
        await file.seek(0)

        if file_size == 0:
            return False, "文件為空"

        if file_size > InputValidator.MAX_FILE_SIZE:
            return False, f"文件過大（最大 {InputValidator.MAX_FILE_SIZE // 1024 // 1024} MB）"

        # 檢查文件魔術數字（文件頭）
        is_valid_type = InputValidator._check_audio_file_type(content[:64], file.filename)
        if not is_valid_type:
            return False, "文件類型不匹配或已損壞"

        return True, None

    @staticmethod
    def _check_audio_file_type(header: bytes, filename: str) -> bool:
        """檢查音頻文件類型的魔術數字"""
        ext = Path(filename).suffix.lower()

        # 定義各種音頻格式的魔術數字
        magic_numbers = {
            '.mp3': [b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'],
            '.wav': [b'RIFF'],
            '.m4a': [b'ftyp'],
            '.flac': [b'fLaC'],
            '.ogg': [b'OggS'],
            '.opus': [b'OggS']
        }

        if ext not in magic_numbers:
            return False

        # 檢查文件頭是否匹配
        for magic in magic_numbers[ext]:
            if magic in header:
                return True

        return False

    @staticmethod
    def validate_time_range(
        start_time: Optional[float],
        end_time: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """
        驗證時間範圍

        Returns:
            (is_valid, error_message)
        """
        if start_time is not None:
            if start_time < 0:
                return False, "開始時間不能為負數"
            if start_time > 86400:  # 24 小時
                return False, "開始時間不能超過 24 小時"

        if end_time is not None:
            if end_time < 0:
                return False, "結束時間不能為負數"
            if end_time > 86400:
                return False, "結束時間不能超過 24 小時"

        if start_time is not None and end_time is not None:
            if start_time >= end_time:
                return False, "開始時間必須小於結束時間"
            if end_time - start_time > 14400:  # 4 小時
                return False, "處理時間範圍不能超過 4 小時"

        return True, None

    @staticmethod
    def validate_language(language: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        驗證語言代碼

        Returns:
            (is_valid, error_message)
        """
        if language is None:
            return True, None  # 允許為空（自動檢測）

        if language not in InputValidator.ALLOWED_LANGUAGES:
            return False, f"不支持的語言。允許的語言: {', '.join(sorted(InputValidator.ALLOWED_LANGUAGES))}"

        return True, None

    @staticmethod
    def validate_task_type(task: str) -> Tuple[bool, Optional[str]]:
        """
        驗證任務類型

        Returns:
            (is_valid, error_message)
        """
        if task not in InputValidator.ALLOWED_TASKS:
            return False, f"無效的任務類型。允許的類型: {', '.join(InputValidator.ALLOWED_TASKS)}"

        return True, None

    @staticmethod
    def validate_vad_parameters(
        vad_onset: float,
        vad_offset: float
    ) -> Tuple[bool, Optional[str]]:
        """
        驗證 VAD 參數

        Returns:
            (is_valid, error_message)
        """
        if not 0 <= vad_onset <= 1:
            return False, "VAD onset 必須在 0 到 1 之間"

        if not 0 <= vad_offset <= 1:
            return False, "VAD offset 必須在 0 到 1 之間"

        if vad_onset < vad_offset:
            return False, "VAD onset 不能小於 VAD offset"

        return True, None

    @staticmethod
    def validate_speaker_count(
        min_speakers: Optional[int],
        max_speakers: Optional[int]
    ) -> Tuple[bool, Optional[str]]:
        """
        驗證語者數量範圍

        Returns:
            (is_valid, error_message)
        """
        if min_speakers is not None:
            if min_speakers < 1:
                return False, "最小語者數不能小於 1"
            if min_speakers > 20:
                return False, "最小語者數不能超過 20"

        if max_speakers is not None:
            if max_speakers < 1:
                return False, "最大語者數不能小於 1"
            if max_speakers > 20:
                return False, "最大語者數不能超過 20"

        if min_speakers is not None and max_speakers is not None:
            if min_speakers > max_speakers:
                return False, "最小語者數不能大於最大語者數"

        return True, None

    @staticmethod
    def validate_compute_type(compute_type: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        驗證計算類型

        Returns:
            (is_valid, error_message)
        """
        if compute_type is None:
            return True, None

        if compute_type not in InputValidator.ALLOWED_COMPUTE_TYPES:
            return False, f"無效的計算類型。允許的類型: {', '.join(InputValidator.ALLOWED_COMPUTE_TYPES)}"

        return True, None

    @staticmethod
    def validate_model_name(model: str) -> Tuple[bool, Optional[str]]:
        """
        驗證模型名稱

        Returns:
            (is_valid, error_message)
        """
        if not model:
            return False, "模型名稱不能為空"

        # 檢查長度
        if len(model) > 200:
            return False, "模型名稱過長"

        # 檢查是否包含危險字符（防止路徑遍歷）
        dangerous_chars = ['..', '/', '\\', '\x00', ';', '&', '|', '$', '`', '<', '>']
        if any(char in model for char in dangerous_chars):
            return False, "模型名稱包含非法字符"

        return True, None

    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """
        清理字符串，移除潛在危險字符

        Args:
            text: 要清理的文本
            max_length: 最大長度

        Returns:
            清理後的文本
        """
        if not text:
            return ""

        # 截斷過長的文本
        text = text[:max_length]

        # 移除控制字符（保留換行和製表符）
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

        return text

    @staticmethod
    def validate_admin_token(token: str) -> Tuple[bool, Optional[str]]:
        """
        驗證管理員 Token

        Returns:
            (is_valid, error_message)
        """
        if not token:
            return False, "Token 不能為空"

        if len(token) < 16:
            return False, "Token 長度不足"

        if len(token) > 256:
            return False, "Token 過長"

        # 檢查是否包含危險字符
        dangerous_chars = [';', '&', '|', '$', '`', '<', '>', '"', "'", '\\', '\x00']
        if any(char in token for char in dangerous_chars):
            return False, "Token 包含非法字符"

        return True, None


# 創建全局驗證器實例
input_validator = InputValidator()
