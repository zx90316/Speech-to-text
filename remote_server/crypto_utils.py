"""
加密工具模組
符合 SSDLC 要求：保護敏感數據的儲存和傳輸
"""
import os
import hashlib
import secrets
from typing import Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64


class CryptoUtils:
    """加密工具類"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 從環境變數讀取加密金鑰，如果不存在則生成一個
            self.encryption_key = self._get_or_create_encryption_key()
            self.fernet = Fernet(self.encryption_key)
            self.initialized = True

    def _get_or_create_encryption_key(self) -> bytes:
        """獲取或創建加密金鑰"""
        # 從環境變數讀取
        key_str = os.getenv('ENCRYPTION_KEY')

        if key_str:
            try:
                # Fernet key is already base64-encoded, just convert string to bytes
                return key_str.encode()
            except Exception:
                print("警告：無效的 ENCRYPTION_KEY，將生成新的金鑰")

        # 生成新的金鑰
        key = Fernet.generate_key()
        print("=" * 60)
        print("警告：未設置 ENCRYPTION_KEY，已生成新的加密金鑰")
        print("請將以下金鑰添加到 .env 文件中：")
        print(f"ENCRYPTION_KEY={key.decode()}")
        print("=" * 60)
        return key

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """
        使用 PBKDF2 雜湊密碼

        Args:
            password: 明文密碼
            salt: 鹽值（可選，不提供則自動生成）

        Returns:
            (hashed_password, salt) 兩者都是 base64 編碼的字符串
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        elif isinstance(salt, str):
            salt = base64.b64decode(salt)

        # 使用 PBKDF2 進行 100,000 次迭代
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        hashed = kdf.derive(password.encode('utf-8'))

        return (
            base64.b64encode(hashed).decode('utf-8'),
            base64.b64encode(salt).decode('utf-8')
        )

    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """
        驗證密碼

        Args:
            password: 明文密碼
            hashed_password: 雜湊後的密碼（base64）
            salt: 鹽值（base64）

        Returns:
            密碼是否正確
        """
        try:
            new_hash, _ = CryptoUtils.hash_password(password, salt)
            return secrets.compare_digest(new_hash, hashed_password)
        except Exception:
            return False

    @staticmethod
    def hash_sha256(data: str) -> str:
        """
        使用 SHA-256 雜湊數據

        Args:
            data: 要雜湊的數據

        Returns:
            十六進制雜湊值
        """
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    @staticmethod
    def hash_email_for_storage(email: str) -> str:
        """
        雜湊郵箱用於存儲（單向）

        Args:
            email: 郵箱地址

        Returns:
            雜湊後的郵箱
        """
        # 使用 SHA-256 並加鹽
        salt = os.getenv('EMAIL_HASH_SALT', 'default_salt_change_me')
        return hashlib.sha256(f"{email}{salt}".encode('utf-8')).hexdigest()

    def encrypt_data(self, data: str) -> str:
        """
        加密數據（對稱加密）

        Args:
            data: 明文數據

        Returns:
            加密後的數據（base64 編碼）
        """
        try:
            encrypted = self.fernet.encrypt(data.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        except Exception as e:
            raise ValueError(f"加密失敗: {e}")

    def decrypt_data(self, encrypted_data: str) -> str:
        """
        解密數據

        Args:
            encrypted_data: 加密的數據（base64 編碼）

        Returns:
            明文數據
        """
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data)
            decrypted = self.fernet.decrypt(encrypted)
            return decrypted.decode('utf-8')
        except Exception as e:
            raise ValueError(f"解密失敗: {e}")

    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        加密文件

        Args:
            file_path: 要加密的文件路徑
            output_path: 輸出文件路徑（可選，默認為 原文件.encrypted）

        Returns:
            加密後的文件路徑
        """
        if output_path is None:
            output_path = f"{file_path}.encrypted"

        try:
            # 讀取文件
            with open(file_path, 'rb') as f:
                data = f.read()

            # 加密
            encrypted = self.fernet.encrypt(data)

            # 寫入加密文件
            with open(output_path, 'wb') as f:
                f.write(encrypted)

            return output_path
        except Exception as e:
            raise ValueError(f"文件加密失敗: {e}")

    def decrypt_file(self, encrypted_file_path: str, output_path: str) -> str:
        """
        解密文件

        Args:
            encrypted_file_path: 加密的文件路徑
            output_path: 輸出文件路徑

        Returns:
            解密後的文件路徑
        """
        try:
            # 讀取加密文件
            with open(encrypted_file_path, 'rb') as f:
                encrypted = f.read()

            # 解密
            decrypted = self.fernet.decrypt(encrypted)

            # 寫入解密文件
            with open(output_path, 'wb') as f:
                f.write(decrypted)

            return output_path
        except Exception as e:
            raise ValueError(f"文件解密失敗: {e}")

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """
        生成安全的隨機 Token

        Args:
            length: Token 長度（字節）

        Returns:
            十六進制 Token
        """
        return secrets.token_hex(length)

    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """
        生成安全的隨機密碼

        Args:
            length: 密碼長度

        Returns:
            隨機密碼（包含大小寫字母、數字和特殊字符）
        """
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def mask_email(email: str) -> str:
        """
        遮罩郵箱地址（用於日誌和顯示）

        Args:
            email: 郵箱地址

        Returns:
            遮罩後的郵箱（例如：ab***@domain.com）
        """
        if '@' not in email:
            return "***"

        local, domain = email.split('@', 1)

        if len(local) <= 2:
            masked_local = "***"
        elif len(local) <= 4:
            masked_local = local[0] + "***"
        else:
            masked_local = local[:2] + "***"

        return f"{masked_local}@{domain}"

    @staticmethod
    def mask_ip(ip_address: str) -> str:
        """
        遮罩 IP 地址（保留前兩段）

        Args:
            ip_address: IP 地址

        Returns:
            遮罩後的 IP（例如：192.168.***.***）
        """
        parts = ip_address.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.***. ***"
        return "***"

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """
        常數時間字符串比較（防止時序攻擊）

        Args:
            a: 字符串 A
            b: 字符串 B

        Returns:
            是否相等
        """
        return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

    @staticmethod
    def secure_delete_file(file_path: str, passes: int = 3):
        """
        安全刪除文件（覆寫後刪除）

        Args:
            file_path: 文件路徑
            passes: 覆寫次數（預設 3 次）
        """
        try:
            if not os.path.exists(file_path):
                return

            # 獲取文件大小
            file_size = os.path.getsize(file_path)

            # 多次覆寫
            with open(file_path, 'ba+', buffering=0) as f:
                for _ in range(passes):
                    f.seek(0)
                    f.write(os.urandom(file_size))

            # 最後刪除
            os.remove(file_path)

        except Exception as e:
            print(f"安全刪除文件失敗: {e}")
            # 如果安全刪除失敗，至少要正常刪除
            try:
                os.remove(file_path)
            except:
                pass

crypto_utils = CryptoUtils()
    
