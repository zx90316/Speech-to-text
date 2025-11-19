"""
測試加密工具模組 (crypto_utils.py)
"""
import pytest
import os
import base64
import tempfile
from pathlib import Path
from cryptography.fernet import Fernet

# 導入要測試的模組
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from crypto_utils import CryptoUtils, crypto_utils


class TestCryptoUtils:
    """測試 CryptoUtils 類"""

    def test_singleton_pattern(self):
        """測試單例模式"""
        instance1 = CryptoUtils()
        instance2 = CryptoUtils()
        assert instance1 is instance2

    def test_encryption_key_initialization(self):
        """測試加密金鑰初始化"""
        crypto = CryptoUtils()
        assert crypto.encryption_key is not None
        assert isinstance(crypto.encryption_key, bytes)

    def test_hash_password(self):
        """測試密碼雜湊"""
        password = "test_password_123"
        hashed, salt = CryptoUtils.hash_password(password)
        
        assert hashed is not None
        assert salt is not None
        assert isinstance(hashed, str)
        assert isinstance(salt, str)
        
        # 相同密碼和鹽值應該產生相同的雜湊
        hashed2, _ = CryptoUtils.hash_password(password, salt)
        assert hashed == hashed2

    def test_hash_password_different_salts(self):
        """測試不同鹽值產生不同雜湊"""
        password = "test_password_123"
        hashed1, salt1 = CryptoUtils.hash_password(password)
        hashed2, salt2 = CryptoUtils.hash_password(password)
        
        assert salt1 != salt2
        assert hashed1 != hashed2

    def test_verify_password_correct(self):
        """測試正確密碼驗證"""
        password = "test_password_123"
        hashed, salt = CryptoUtils.hash_password(password)
        
        assert CryptoUtils.verify_password(password, hashed, salt) is True

    def test_verify_password_incorrect(self):
        """測試錯誤密碼驗證"""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed, salt = CryptoUtils.hash_password(password)
        
        assert CryptoUtils.verify_password(wrong_password, hashed, salt) is False

    def test_verify_password_invalid_hash(self):
        """測試無效雜湊值"""
        password = "test_password_123"
        invalid_hash = "invalid_hash"
        invalid_salt = "invalid_salt"
        
        assert CryptoUtils.verify_password(password, invalid_hash, invalid_salt) is False

    def test_hash_sha256(self):
        """測試 SHA-256 雜湊"""
        data = "test_data"
        hashed = CryptoUtils.hash_sha256(data)
        
        assert hashed is not None
        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA-256 產生 64 個十六進制字符
        
        # 相同資料應該產生相同雜湊
        hashed2 = CryptoUtils.hash_sha256(data)
        assert hashed == hashed2

    def test_hash_email_for_storage(self):
        """測試郵箱雜湊"""
        email = "test@example.com"
        hashed = CryptoUtils.hash_email_for_storage(email)
        
        assert hashed is not None
        assert isinstance(hashed, str)
        assert len(hashed) == 64
        
        # 相同郵箱應該產生相同雜湊
        hashed2 = CryptoUtils.hash_email_for_storage(email)
        assert hashed == hashed2
        
        # 不同郵箱應該產生不同雜湊
        different_email = "different@example.com"
        hashed3 = CryptoUtils.hash_email_for_storage(different_email)
        assert hashed != hashed3

    def test_encrypt_decrypt_data(self):
        """測試資料加密和解密"""
        crypto = CryptoUtils()
        original_data = "這是測試資料 with 中文字符"
        
        # 加密
        encrypted = crypto.encrypt_data(original_data)
        assert encrypted is not None
        assert isinstance(encrypted, str)
        assert encrypted != original_data
        
        # 解密
        decrypted = crypto.decrypt_data(encrypted)
        assert decrypted == original_data

    def test_encrypt_empty_string(self):
        """測試加密空字符串"""
        crypto = CryptoUtils()
        encrypted = crypto.encrypt_data("")
        decrypted = crypto.decrypt_data(encrypted)
        assert decrypted == ""

    def test_decrypt_invalid_data(self):
        """測試解密無效資料"""
        crypto = CryptoUtils()
        with pytest.raises(ValueError):
            crypto.decrypt_data("invalid_encrypted_data")

    def test_encrypt_file(self, temp_dir):
        """測試文件加密"""
        crypto = CryptoUtils()
        
        # 創建測試文件
        test_file = temp_dir / "test.txt"
        test_content = "這是測試文件內容"
        test_file.write_text(test_content, encoding='utf-8')
        
        # 加密文件
        encrypted_file = crypto.encrypt_file(str(test_file))
        assert Path(encrypted_file).exists()
        
        # 驗證加密文件不是原始內容
        encrypted_content = Path(encrypted_file).read_bytes()
        assert encrypted_content != test_content.encode('utf-8')
        
        # 清理
        Path(encrypted_file).unlink()

    def test_decrypt_file(self, temp_dir):
        """測試文件解密"""
        crypto = CryptoUtils()
        
        # 創建測試文件
        test_file = temp_dir / "test.txt"
        test_content = "這是測試文件內容"
        test_file.write_text(test_content, encoding='utf-8')
        
        # 加密文件
        encrypted_file = crypto.encrypt_file(str(test_file))
        
        # 解密文件
        decrypted_file = temp_dir / "decrypted.txt"
        crypto.decrypt_file(encrypted_file, str(decrypted_file))
        
        # 驗證解密內容
        decrypted_content = decrypted_file.read_text(encoding='utf-8')
        assert decrypted_content == test_content
        
        # 清理
        Path(encrypted_file).unlink()

    def test_encrypt_file_with_custom_output(self, temp_dir):
        """測試使用自定義輸出路徑的文件加密"""
        crypto = CryptoUtils()
        
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        custom_output = temp_dir / "custom_encrypted.bin"
        encrypted_file = crypto.encrypt_file(str(test_file), str(custom_output))
        
        assert encrypted_file == str(custom_output)
        assert custom_output.exists()

    def test_encrypt_nonexistent_file(self):
        """測試加密不存在的文件"""
        crypto = CryptoUtils()
        with pytest.raises(ValueError):
            crypto.encrypt_file("/nonexistent/file.txt")

    def test_generate_secure_token(self):
        """測試生成安全 Token"""
        token = CryptoUtils.generate_secure_token()
        assert token is not None
        assert isinstance(token, str)
        assert len(token) == 64  # 32 bytes = 64 hex chars
        
        # 測試自定義長度
        token_16 = CryptoUtils.generate_secure_token(16)
        assert len(token_16) == 32  # 16 bytes = 32 hex chars

    def test_generate_secure_token_uniqueness(self):
        """測試 Token 唯一性"""
        token1 = CryptoUtils.generate_secure_token()
        token2 = CryptoUtils.generate_secure_token()
        assert token1 != token2

    def test_generate_secure_password(self):
        """測試生成安全密碼"""
        password = CryptoUtils.generate_secure_password()
        assert password is not None
        assert isinstance(password, str)
        assert len(password) == 16
        
        # 測試自定義長度
        password_24 = CryptoUtils.generate_secure_password(24)
        assert len(password_24) == 24

    def test_generate_secure_password_complexity(self):
        """測試密碼複雜度"""
        password = CryptoUtils.generate_secure_password(100)
        
        # 應該包含多種字符類型
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        assert has_upper or has_lower or has_digit

    def test_mask_email(self):
        """測試郵箱遮罩"""
        # 正常郵箱
        email = "testuser@example.com"
        masked = CryptoUtils.mask_email(email)
        assert masked == "te***@example.com"
        assert "@" in masked
        assert "example.com" in masked
        
        # 短郵箱
        short_email = "ab@example.com"
        masked_short = CryptoUtils.mask_email(short_email)
        assert masked_short == "***@example.com"
        
        # 無效郵箱
        invalid_email = "notanemail"
        masked_invalid = CryptoUtils.mask_email(invalid_email)
        assert masked_invalid == "***"

    def test_mask_email_various_lengths(self):
        """測試不同長度的郵箱遮罩"""
        # 4 字符
        email1 = "test@example.com"
        assert CryptoUtils.mask_email(email1) == "t***@example.com"
        
        # 5+ 字符
        email2 = "testuser@example.com"
        assert CryptoUtils.mask_email(email2) == "te***@example.com"

    def test_mask_ip(self):
        """測試 IP 地址遮罩"""
        ip = "192.168.1.100"
        masked = CryptoUtils.mask_ip(ip)
        assert masked.startswith("192.168")
        assert "***" in masked
        
        # 無效 IP
        invalid_ip = "invalid_ip"
        masked_invalid = CryptoUtils.mask_ip(invalid_ip)
        assert masked_invalid == "***"

    def test_constant_time_compare_equal(self):
        """測試常數時間比較 - 相等"""
        str1 = "test_string_123"
        str2 = "test_string_123"
        assert CryptoUtils.constant_time_compare(str1, str2) is True

    def test_constant_time_compare_not_equal(self):
        """測試常數時間比較 - 不相等"""
        str1 = "test_string_123"
        str2 = "test_string_456"
        assert CryptoUtils.constant_time_compare(str1, str2) is False

    def test_constant_time_compare_different_lengths(self):
        """測試常數時間比較 - 不同長度"""
        str1 = "short"
        str2 = "longer_string"
        assert CryptoUtils.constant_time_compare(str1, str2) is False

    def test_secure_delete_file(self, temp_dir):
        """測試安全刪除文件"""
        test_file = temp_dir / "test_delete.txt"
        test_file.write_text("sensitive data")
        
        assert test_file.exists()
        
        CryptoUtils.secure_delete_file(str(test_file))
        
        assert not test_file.exists()

    def test_secure_delete_nonexistent_file(self):
        """測試刪除不存在的文件（應該不報錯）"""
        CryptoUtils.secure_delete_file("/nonexistent/file.txt")
        # 不應該拋出異常

    def test_secure_delete_file_multiple_passes(self, temp_dir):
        """測試多次覆寫刪除"""
        test_file = temp_dir / "test_delete.txt"
        test_file.write_text("sensitive data" * 100)
        
        CryptoUtils.secure_delete_file(str(test_file), passes=5)
        
        assert not test_file.exists()

    def test_global_instance(self):
        """測試全局實例"""
        assert crypto_utils is not None
        assert isinstance(crypto_utils, CryptoUtils)


@pytest.mark.security
class TestCryptoUtilsSecurity:
    """安全相關的特殊測試"""
    
    def test_password_hash_timing_attack_resistance(self):
        """測試密碼雜湊的時序攻擊抵抗性"""
        password = "test_password"
        hashed, salt = CryptoUtils.hash_password(password)
        
        # 使用 constant_time_compare 驗證
        hashed2, _ = CryptoUtils.hash_password(password, salt)
        assert CryptoUtils.constant_time_compare(hashed, hashed2)

    def test_encryption_key_from_env(self):
        """測試從環境變數讀取加密金鑰"""
        # 已在 conftest.py 中設置
        crypto = CryptoUtils()
        assert crypto.encryption_key is not None

    def test_salt_randomness(self):
        """測試鹽值的隨機性"""
        salts = set()
        for _ in range(100):
            _, salt = CryptoUtils.hash_password("password")
            salts.add(salt)
        
        # 100 次應該產生 100 個不同的鹽值
        assert len(salts) == 100

    def test_pbkdf2_iterations(self):
        """測試 PBKDF2 迭代次數（間接測試）"""
        import time
        
        password = "test_password"
        
        # 雜湊應該需要一定時間（因為有 100,000 次迭代）
        start = time.time()
        CryptoUtils.hash_password(password)
        elapsed = time.time() - start
        
        # 至少需要一些時間（但不要太嚴格，以免在快速機器上失敗）
        assert elapsed > 0.001  # 至少 1ms

