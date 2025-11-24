# -*- coding: utf-8 -*-
"""
速率限制模組
符合 SSDLC 要求：防止暴力破解和 DoS 攻擊
"""
import time
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import threading


class RateLimiter:
    """速率限制器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # IP 級別的請求記錄
            self.ip_requests: Dict[str, deque] = defaultdict(lambda: deque())

            # 郵箱驗證失敗記錄
            self.email_verification_failures: Dict[str, deque] = defaultdict(lambda: deque())

            # 郵箱驗證碼發送記錄
            self.email_code_requests: Dict[str, deque] = defaultdict(lambda: deque())

            # IP 黑名單（臨時封禁）
            self.ip_blacklist: Dict[str, datetime] = {}

            # 郵箱黑名單（臨時封禁）
            self.email_blacklist: Dict[str, datetime] = {}

            self.lock = threading.RLock()
            self.initialized = True

    def _clean_old_records(self, records: deque, time_window: int):
        """清理過期的記錄"""
        current_time = time.time()
        while records and current_time - records[0] > time_window:
            records.popleft()

    def check_ip_rate_limit(
        self,
        ip_address: str,
        endpoint: str = "general",
        max_requests: int = 100,
        time_window: int = 60
    ) -> Tuple[bool, Optional[str]]:
        """
        檢查 IP 速率限制

        Args:
            ip_address: IP 地址
            endpoint: 端點名稱
            max_requests: 時間窗口內的最大請求數
            time_window: 時間窗口（秒）

        Returns:
            (is_allowed, error_message)
        """
        with self.lock:
            # 檢查 IP 是否在黑名單中
            if ip_address in self.ip_blacklist:
                ban_until = self.ip_blacklist[ip_address]
                if datetime.now() < ban_until:
                    remaining = int((ban_until - datetime.now()).total_seconds())
                    return False, f"IP 已被臨時封禁，請在 {remaining} 秒後再試"
                else:
                    # 封禁已過期，移除
                    del self.ip_blacklist[ip_address]

            # 獲取記錄鍵
            record_key = f"{ip_address}:{endpoint}"
            records = self.ip_requests[record_key]

            # 清理過期記錄
            self._clean_old_records(records, time_window)

            # 檢查是否超過限制
            if len(records) >= max_requests:
                # 超過限制，加入黑名單（封禁 10 分鐘）
                self.ip_blacklist[ip_address] = datetime.now() + timedelta(minutes=10)
                return False, f"請求過於頻繁，已被臨時封禁 10 分鐘"

            # 記錄本次請求
            records.append(time.time())
            return True, None

    def check_email_verification_rate_limit(
        self,
        email: str,
        max_requests: int = 5,
        time_window: int = 3600
    ) -> Tuple[bool, Optional[str]]:
        """
        檢查郵箱驗證碼發送速率限制

        Args:
            email: 郵箱地址
            max_requests: 時間窗口內的最大請求數（預設 1 小時 5 次）
            time_window: 時間窗口（秒）

        Returns:
            (is_allowed, error_message)
        """
        with self.lock:
            # 檢查郵箱是否在黑名單中
            if email in self.email_blacklist:
                ban_until = self.email_blacklist[email]
                if datetime.now() < ban_until:
                    remaining = int((ban_until - datetime.now()).total_seconds())
                    return False, f"該郵箱已被臨時封禁，請在 {remaining} 秒後再試"
                else:
                    del self.email_blacklist[email]

            records = self.email_code_requests[email]

            # 清理過期記錄
            self._clean_old_records(records, time_window)

            # 檢查是否超過限制
            if len(records) >= max_requests:
                return False, f"驗證碼發送過於頻繁，請稍後再試（1 小時內最多 {max_requests} 次）"

            # 記錄本次請求
            records.append(time.time())
            return True, None

    def record_verification_failure(self, email: str, ip_address: str) -> bool:
        """
        記錄驗證失敗

        Args:
            email: 郵箱地址
            ip_address: IP 地址

        Returns:
            是否應該封禁（連續失敗超過閾值）
        """
        with self.lock:
            record_key = f"{email}:{ip_address}"
            records = self.email_verification_failures[record_key]

            # 清理 15 分鐘前的記錄
            self._clean_old_records(records, 900)

            # 記錄本次失敗
            records.append(time.time())

            # 檢查是否連續失敗超過 5 次
            if len(records) >= 5:
                # 封禁該郵箱 30 分鐘
                self.email_blacklist[email] = datetime.now() + timedelta(minutes=30)
                # 封禁該 IP 10 分鐘
                self.ip_blacklist[ip_address] = datetime.now() + timedelta(minutes=10)
                return True

            return False

    def reset_verification_failures(self, email: str, ip_address: str):
        """重置驗證失敗記錄（驗證成功時調用）"""
        with self.lock:
            record_key = f"{email}:{ip_address}"
            if record_key in self.email_verification_failures:
                self.email_verification_failures[record_key].clear()

    def check_task_creation_rate_limit(
        self,
        email: str,
        max_tasks: int = 10,
        time_window: int = 3600
    ) -> Tuple[bool, Optional[str]]:
        """
        檢查任務創建速率限制

        Args:
            email: 郵箱地址
            max_tasks: 時間窗口內的最大任務數（預設 1 小時 10 個）
            time_window: 時間窗口（秒）

        Returns:
            (is_allowed, error_message)
        """
        with self.lock:
            record_key = f"task:{email}"
            records = self.ip_requests[record_key]

            # 清理過期記錄
            self._clean_old_records(records, time_window)

            # 檢查是否超過限制
            if len(records) >= max_tasks:
                return False, f"任務創建過於頻繁（1 小時內最多 {max_tasks} 個任務）"

            # 記錄本次請求
            records.append(time.time())
            return True, None

    def check_admin_access_rate_limit(
        self,
        ip_address: str,
        max_requests: int = 20,
        time_window: int = 60
    ) -> Tuple[bool, Optional[str]]:
        """
        檢查管理員 API 訪問速率限制

        Args:
            ip_address: IP 地址
            max_requests: 時間窗口內的最大請求數
            time_window: 時間窗口（秒）

        Returns:
            (is_allowed, error_message)
        """
        return self.check_ip_rate_limit(
            ip_address,
            endpoint="admin",
            max_requests=max_requests,
            time_window=time_window
        )

    def get_remaining_attempts(self, email: str, ip_address: str) -> int:
        """獲取剩餘驗證嘗試次數"""
        with self.lock:
            record_key = f"{email}:{ip_address}"
            records = self.email_verification_failures.get(record_key, deque())

            # 清理過期記錄
            self._clean_old_records(records, 900)

            return max(0, 5 - len(records))

    def is_ip_blacklisted(self, ip_address: str) -> Tuple[bool, Optional[int]]:
        """
        檢查 IP 是否在黑名單中

        Returns:
            (is_blacklisted, remaining_seconds)
        """
        with self.lock:
            if ip_address in self.ip_blacklist:
                ban_until = self.ip_blacklist[ip_address]
                if datetime.now() < ban_until:
                    remaining = int((ban_until - datetime.now()).total_seconds())
                    return True, remaining
                else:
                    del self.ip_blacklist[ip_address]

            return False, None

    def is_email_blacklisted(self, email: str) -> Tuple[bool, Optional[int]]:
        """
        檢查郵箱是否在黑名單中

        Returns:
            (is_blacklisted, remaining_seconds)
        """
        with self.lock:
            if email in self.email_blacklist:
                ban_until = self.email_blacklist[email]
                if datetime.now() < ban_until:
                    remaining = int((ban_until - datetime.now()).total_seconds())
                    return True, remaining
                else:
                    del self.email_blacklist[email]

            return False, None

    def cleanup_expired_records(self):
        """清理所有過期記錄（定期調用）"""
        with self.lock:
            current_time = time.time()

            # 清理 IP 請求記錄
            for key in list(self.ip_requests.keys()):
                records = self.ip_requests[key]
                self._clean_old_records(records, 3600)
                if not records:
                    del self.ip_requests[key]

            # 清理郵箱驗證失敗記錄
            for key in list(self.email_verification_failures.keys()):
                records = self.email_verification_failures[key]
                self._clean_old_records(records, 900)
                if not records:
                    del self.email_verification_failures[key]

            # 清理郵箱驗證碼請求記錄
            for key in list(self.email_code_requests.keys()):
                records = self.email_code_requests[key]
                self._clean_old_records(records, 3600)
                if not records:
                    del self.email_code_requests[key]

            # 清理過期的黑名單
            current_datetime = datetime.now()
            self.ip_blacklist = {
                ip: ban_until
                for ip, ban_until in self.ip_blacklist.items()
                if ban_until > current_datetime
            }
            self.email_blacklist = {
                email: ban_until
                for email, ban_until in self.email_blacklist.items()
                if ban_until > current_datetime
            }

    def get_stats(self) -> Dict:
        """獲取速率限制統計信息"""
        with self.lock:
            return {
                'ip_records_count': len(self.ip_requests),
                'email_failure_records': len(self.email_verification_failures),
                'email_code_requests': len(self.email_code_requests),
                'ip_blacklist_count': len(self.ip_blacklist),
                'email_blacklist_count': len(self.email_blacklist)
            }


# 全局速率限制器實例
rate_limiter = RateLimiter()
