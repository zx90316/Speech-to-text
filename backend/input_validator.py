# -*- coding: utf-8 -*-
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
        對用戶提供的文件名進行嚴格的安全驗證。

        防範威脅：
        1. Null 字節注入 (Null Byte Injection)
        2. 路徑遍歷 (Path Traversal)
        3. 控制字符 (Control Characters)
        4. 系統保留名稱 (Reserved OS Filenames)
        5. 特殊/非法字符 (用於命令注入、FS 錯誤)
        6. 長度攻擊 (Length Attacks)
        7. 空白/點結尾 (Whitespace/Dot Suffix)
        8. 擴展名白名單 (Extension Whitelisting)

        Returns:
            (is_valid, error_message)
        """

        # --- 安全配置 ---

        # 1. Windows 保留文件名（不區分大小寫）
        # 這些名稱在 Windows 上被系統保留，不能用作文件名
        RESERVED_FILENAMES = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
        }

        # 2. 檢查路徑分隔符
        #    拒絕任何 *試圖* 包含路徑的輸入
        #    這將捕獲 "/etc/passwd.mp3" 和 "test/file.mp3"
        if '/' in filename or '\\' in filename:
            return False, "文件名不應包含路徑分隔符 ( / 或 \\ )"

        # 3. 檢查 '..'
        #    明確拒絕 '..' 以防止路徑遍歷
        #    這將捕獲 "../etc/passwd"
        if '..' in filename:
            return False, "文件名不應包含 '..' (路徑遍歷)"

        # 2. C0 和 C1 控制字符的正則表達式
        # 這些是不可見字符，可能用於欺騙系統或日誌
        # 禁止 \x00-\x1F (C0) 和 \x7F-\x9F (C1)
        # \x00 (null) 已被單獨檢查，但包含在此處可作為多層防禦
        CONTROL_CHAR_RE = re.compile(r'[\x00-\x1F\x7F-\x9F]')

        # 3. 在多數文件系統或 shell 中非法的/危險的字符
        # 包括： < > : " / \ | ? * ; & $ ` ( ) [ ] { }
        # os.path.basename 會處理 / 和 \，但我們在黑名單中保留它們以防萬一
        # 這些字符可能用於路徑遍歷、命令注入或創建無效文件
        INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*;&$()`[\]{}]')

        # 0. 基本類型檢查
        if not filename or not isinstance(filename, str):
            return False, "文件名不能為空"

        # 1. 檢查 Null 字節 (高優先級)
        # Null 字節 (`\x00`) 可以截斷 C 語言中的字符串，導致安全繞過
        if '\x00' in filename:
            return False, "文件名包含非法的 null 字節"

        # 2. 移除路徑部分 (核心的路徑遍歷防禦)
        # os.path.basename 會安全地提取文件名，丟棄所有前面的路徑
        # 例如 "..\..\secret.txt" -> "secret.txt"
        # 例如 "/etc/passwd" -> "passwd"
        # 例如 "C:\\Windows\\system32.dll" -> "system32.dll"
        base_filename = os.path.basename(filename)

        # 3. 檢查 `basename` 後的結果
        # 如果原始輸入是 ".." 或 "." 或 "///" 之類的，basename 可能返回 "." 或 ".." 或空
        if not base_filename or base_filename in {".", ".."}:
            return False, "文件名無效（例如，僅包含點或路徑分隔符）"

        # 4. 檢查長度
        if len(base_filename) > 255:
            return False, "文件名過長（最多 255 字符）"

        # 5. 檢查控制字符
        if CONTROL_CHAR_RE.search(base_filename):
            return False, "文件名包含非法的控制字符"

        # 6. 檢查非法/危險字符 (黑名單)
        if INVALID_FILENAME_CHARS_RE.search(base_filename):
            return False, "文件名包含非法字符 (例如: <, >, :, \", |, ?, *, ;, &)"

        # 7. 檢查 Windows 保留文件名
        # 移除擴展名後，檢查名稱是否在保留列表中（不區分大小寫）
        name_part = os.path.splitext(base_filename)[0]
        if name_part.upper() in RESERVED_FILENAMES:
            return False, "文件名是系統保留名稱（例如 CON, PRN）"

        # 8. 檢查開頭或結尾的空白/點 (Windows 特有問題)
        # Windows 會自動移除這些，可能導致 'file.txt' 和 'file.txt ' 混淆
        if base_filename.startswith(' ') or base_filename.endswith(' ') or base_filename.endswith('.'):
            return False, "文件名不能以空格開頭或結尾，也不能以點結尾"

        # 9. 檢查文件擴展名 (白名單)
        ext = Path(base_filename).suffix.lower()
        if not ext:
            return False, "文件缺少擴展名"
            
        if ext not in InputValidator.ALLOWED_AUDIO_EXTENSIONS:
            return False, f"不支持的文件格式。允許的格式: {', '.join(InputValidator.ALLOWED_AUDIO_EXTENSIONS)}"

        # 10. (可選) 原始函數中的 '..'/'/'/'\' 檢查
        # 雖然 os.path.basename 已處理，但多一層檢查無害
        # 確保文件名 *本身* 不包含這些
        if '..' in base_filename or '/' in base_filename or '\\' in base_filename:
            return False, "文件名本身不應包含 '..' 或路徑分隔符"

        # 所有檢查通過
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
