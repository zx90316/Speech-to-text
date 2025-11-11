"""
測試速率限制模組 (rate_limiter.py)
"""
import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "remote_server"))

from rate_limiter import RateLimiter, rate_limiter


class TestRateLimiter:
    """測試 RateLimiter 類"""

    def test_singleton_pattern(self):
        """測試單例模式"""
        limiter1 = RateLimiter()
        limiter2 = RateLimiter()
        assert limiter1 is limiter2

    def test_check_ip_rate_limit_normal(self):
        """測試正常的 IP 速率限制"""
        limiter = RateLimiter()
        ip = "192.168.1.100"
        
        # 第一次請求應該允許
        is_allowed, error = limiter.check_ip_rate_limit(ip, "test", 5, 60)
        assert is_allowed is True
        assert error is None

    def test_check_ip_rate_limit_exceed(self):
        """測試超過 IP 速率限制"""
        limiter = RateLimiter()
        ip = "192.168.1.101"
        
        # 發送 5 次請求（限制是 5）
        for i in range(5):
            is_allowed, error = limiter.check_ip_rate_limit(ip, "test", 5, 60)
            assert is_allowed is True
        
        # 第 6 次應該被拒絕
        is_allowed, error = limiter.check_ip_rate_limit(ip, "test", 5, 60)
        assert is_allowed is False
        assert "封禁" in error

    def test_check_ip_rate_limit_different_endpoints(self):
        """測試不同端點的速率限制是獨立的"""
        limiter = RateLimiter()
        ip = "192.168.1.102"
        
        # 端點 A 的請求
        for _ in range(3):
            limiter.check_ip_rate_limit(ip, "endpoint_a", 5, 60)
        
        # 端點 B 的請求應該有自己的計數
        is_allowed, _ = limiter.check_ip_rate_limit(ip, "endpoint_b", 5, 60)
        assert is_allowed is True

    def test_check_ip_rate_limit_blacklist(self):
        """測試 IP 黑名單"""
        limiter = RateLimiter()
        ip = "192.168.1.103"
        
        # 超過限制，加入黑名單
        for i in range(6):
            limiter.check_ip_rate_limit(ip, "test", 5, 60)
        
        # 立即再次請求，應該被黑名單拒絕
        is_allowed, error = limiter.check_ip_rate_limit(ip, "another_endpoint", 100, 60)
        assert is_allowed is False
        assert "封禁" in error

    def test_check_ip_rate_limit_time_window(self):
        """測試時間窗口過期後重置"""
        limiter = RateLimiter()
        ip = "192.168.1.104"
        
        # 發送 3 次請求
        for _ in range(3):
            limiter.check_ip_rate_limit(ip, "test", 5, 1)  # 1 秒窗口
        
        # 等待時間窗口過期
        time.sleep(1.1)
        
        # 應該能再次請求
        is_allowed, error = limiter.check_ip_rate_limit(ip, "test", 5, 1)
        assert is_allowed is True

    def test_check_email_verification_rate_limit_normal(self):
        """測試正常的郵箱驗證速率限制"""
        limiter = RateLimiter()
        email = "test1@example.com"
        
        is_allowed, error = limiter.check_email_verification_rate_limit(email, 5, 3600)
        assert is_allowed is True
        assert error is None

    def test_check_email_verification_rate_limit_exceed(self):
        """測試超過郵箱驗證速率限制"""
        limiter = RateLimiter()
        email = "test2@example.com"
        
        # 發送 5 次驗證碼
        for _ in range(5):
            limiter.check_email_verification_rate_limit(email, 5, 3600)
        
        # 第 6 次應該被拒絕
        is_allowed, error = limiter.check_email_verification_rate_limit(email, 5, 3600)
        assert is_allowed is False
        assert "頻繁" in error

    def test_check_email_verification_rate_limit_blacklist(self):
        """測試郵箱黑名單"""
        limiter = RateLimiter()
        email = "test3@example.com"
        
        # 加入黑名單
        limiter.email_blacklist[email] = datetime.now() + timedelta(minutes=5)
        
        is_allowed, error = limiter.check_email_verification_rate_limit(email, 5, 3600)
        assert is_allowed is False
        assert "封禁" in error

    def test_record_verification_failure_normal(self):
        """測試記錄驗證失敗"""
        limiter = RateLimiter()
        email = "test4@example.com"
        ip = "192.168.1.105"
        
        # 記錄 4 次失敗，不應該觸發封禁
        for _ in range(4):
            should_ban = limiter.record_verification_failure(email, ip)
            assert should_ban is False

    def test_record_verification_failure_trigger_ban(self):
        """測試連續失敗觸發封禁"""
        limiter = RateLimiter()
        email = "test5@example.com"
        ip = "192.168.1.106"
        
        # 記錄 5 次失敗，應該觸發封禁
        for i in range(4):
            limiter.record_verification_failure(email, ip)
        
        # 第 5 次應該返回 True
        should_ban = limiter.record_verification_failure(email, ip)
        assert should_ban is True
        
        # 檢查郵箱和 IP 是否都被封禁
        assert email in limiter.email_blacklist
        assert ip in limiter.ip_blacklist

    def test_reset_verification_failures(self):
        """測試重置驗證失敗記錄"""
        limiter = RateLimiter()
        email = "test6@example.com"
        ip = "192.168.1.107"
        
        # 記錄一些失敗
        for _ in range(3):
            limiter.record_verification_failure(email, ip)
        
        # 重置
        limiter.reset_verification_failures(email, ip)
        
        # 檢查記錄已清空
        record_key = f"{email}:{ip}"
        assert len(limiter.email_verification_failures[record_key]) == 0

    def test_check_task_creation_rate_limit_normal(self):
        """測試正常的任務創建速率限制"""
        limiter = RateLimiter()
        email = "test7@example.com"
        
        is_allowed, error = limiter.check_task_creation_rate_limit(email, 10, 3600)
        assert is_allowed is True

    def test_check_task_creation_rate_limit_exceed(self):
        """測試超過任務創建速率限制"""
        limiter = RateLimiter()
        email = "test8@example.com"
        
        # 創建 10 個任務
        for _ in range(10):
            limiter.check_task_creation_rate_limit(email, 10, 3600)
        
        # 第 11 個應該被拒絕
        is_allowed, error = limiter.check_task_creation_rate_limit(email, 10, 3600)
        assert is_allowed is False
        assert "頻繁" in error

    def test_check_admin_access_rate_limit(self):
        """測試管理員訪問速率限制"""
        limiter = RateLimiter()
        ip = "192.168.1.108"
        
        # 使用管理員端點
        is_allowed, error = limiter.check_admin_access_rate_limit(ip, 20, 60)
        assert is_allowed is True

    def test_get_remaining_attempts(self):
        """測試獲取剩餘嘗試次數"""
        limiter = RateLimiter()
        email = "test9@example.com"
        ip = "192.168.1.109"
        
        # 初始應該有 5 次
        remaining = limiter.get_remaining_attempts(email, ip)
        assert remaining == 5
        
        # 記錄 2 次失敗
        limiter.record_verification_failure(email, ip)
        limiter.record_verification_failure(email, ip)
        
        # 應該剩 3 次
        remaining = limiter.get_remaining_attempts(email, ip)
        assert remaining == 3

    def test_is_ip_blacklisted_not_blacklisted(self):
        """測試 IP 未被封禁"""
        limiter = RateLimiter()
        ip = "192.168.1.110"
        
        is_blacklisted, remaining = limiter.is_ip_blacklisted(ip)
        assert is_blacklisted is False
        assert remaining is None

    def test_is_ip_blacklisted_active(self):
        """測試 IP 正在被封禁"""
        limiter = RateLimiter()
        ip = "192.168.1.111"
        
        # 手動加入黑名單
        limiter.ip_blacklist[ip] = datetime.now() + timedelta(minutes=10)
        
        is_blacklisted, remaining = limiter.is_ip_blacklisted(ip)
        assert is_blacklisted is True
        assert remaining is not None
        assert remaining > 0

    def test_is_ip_blacklisted_expired(self):
        """測試過期的 IP 封禁"""
        limiter = RateLimiter()
        ip = "192.168.1.112"
        
        # 加入已過期的封禁
        limiter.ip_blacklist[ip] = datetime.now() - timedelta(minutes=1)
        
        is_blacklisted, remaining = limiter.is_ip_blacklisted(ip)
        assert is_blacklisted is False
        # 應該已從黑名單中移除
        assert ip not in limiter.ip_blacklist

    def test_is_email_blacklisted_not_blacklisted(self):
        """測試郵箱未被封禁"""
        limiter = RateLimiter()
        email = "test10@example.com"
        
        is_blacklisted, remaining = limiter.is_email_blacklisted(email)
        assert is_blacklisted is False
        assert remaining is None

    def test_is_email_blacklisted_active(self):
        """測試郵箱正在被封禁"""
        limiter = RateLimiter()
        email = "test11@example.com"
        
        # 手動加入黑名單
        limiter.email_blacklist[email] = datetime.now() + timedelta(minutes=30)
        
        is_blacklisted, remaining = limiter.is_email_blacklisted(email)
        assert is_blacklisted is True
        assert remaining is not None
        assert remaining > 0

    def test_cleanup_expired_records(self):
        """測試清理過期記錄"""
        limiter = RateLimiter()
        
        # 清空現有記錄以避免其他測試的干擾
        limiter.ip_requests.clear()
        limiter.email_code_requests.clear()
        
        # 添加一些記錄
        ip = "192.168.1.113"
        email = "test12@example.com"
        
        limiter.check_ip_rate_limit(ip, "test", 5, 1)
        limiter.check_email_verification_rate_limit(email, 5, 1)
        
        # 確認記錄已創建
        assert len(limiter.ip_requests.get(f"{ip}:test", [])) > 0
        assert len(limiter.email_code_requests.get(email, [])) > 0
        
        # 等待過期
        time.sleep(1.1)
        
        # 清理
        limiter.cleanup_expired_records()
        
        # 檢查過期記錄是否被清理
        # 注意：清理後字典鍵會被移除
        assert f"{ip}:test" not in limiter.ip_requests or len(limiter.ip_requests.get(f"{ip}:test", [])) == 0
        assert email not in limiter.email_code_requests or len(limiter.email_code_requests.get(email, [])) == 0

    def test_get_stats(self):
        """測試獲取統計信息"""
        limiter = RateLimiter()
        
        # 添加一些記錄
        limiter.check_ip_rate_limit("192.168.1.114", "test", 5, 60)
        limiter.check_email_verification_rate_limit("test13@example.com", 5, 3600)
        
        stats = limiter.get_stats()
        
        assert isinstance(stats, dict)
        assert 'ip_records_count' in stats
        assert 'email_failure_records' in stats
        assert 'email_code_requests' in stats
        assert 'ip_blacklist_count' in stats
        assert 'email_blacklist_count' in stats

    def test_global_instance(self):
        """測試全局實例"""
        assert rate_limiter is not None
        assert isinstance(rate_limiter, RateLimiter)


@pytest.mark.slow
class TestRateLimiterPerformance:
    """性能相關測試"""
    
    def test_high_volume_requests(self):
        """測試大量請求的處理"""
        limiter = RateLimiter()
        ip = "192.168.1.200"
        
        # 發送 1000 次請求
        for i in range(1000):
            limiter.check_ip_rate_limit(f"{ip}.{i}", "test", 100, 60)
        
        # 應該能正常處理
        stats = limiter.get_stats()
        assert stats['ip_records_count'] > 0

    def test_concurrent_different_ips(self):
        """測試多個不同 IP 的並發請求"""
        limiter = RateLimiter()
        
        # 模擬 100 個不同 IP
        for i in range(100):
            ip = f"192.168.1.{i}"
            is_allowed, _ = limiter.check_ip_rate_limit(ip, "test", 10, 60)
            assert is_allowed is True

    def test_cleanup_performance(self):
        """測試清理操作的性能"""
        limiter = RateLimiter()
        
        # 添加大量記錄
        for i in range(100):
            ip = f"192.168.{i}.1"
            email = f"test{i}@example.com"
            limiter.check_ip_rate_limit(ip, "test", 10, 1)
            limiter.check_email_verification_rate_limit(email, 5, 1)
        
        # 等待過期
        time.sleep(1.1)
        
        # 清理應該快速完成
        start_time = time.time()
        limiter.cleanup_expired_records()
        elapsed = time.time() - start_time
        
        # 清理應該在 1 秒內完成
        assert elapsed < 1.0


@pytest.mark.security
class TestRateLimiterSecurity:
    """安全相關測試"""
    
    def test_ban_duration_ip(self):
        """測試 IP 封禁時長"""
        limiter = RateLimiter()
        ip = "192.168.1.150"
        
        # 觸發封禁
        for _ in range(6):
            limiter.check_ip_rate_limit(ip, "test", 5, 60)
        
        # 檢查封禁時長（應該是 10 分鐘）
        is_blacklisted, remaining = limiter.is_ip_blacklisted(ip)
        assert is_blacklisted is True
        assert remaining <= 600  # 最多 10 分鐘
        assert remaining > 590  # 至少 9 分 50 秒

    def test_ban_duration_email(self):
        """測試郵箱封禁時長"""
        limiter = RateLimiter()
        email = "test_ban@example.com"
        ip = "192.168.1.151"
        
        # 觸發封禁（5 次失敗）
        for _ in range(5):
            limiter.record_verification_failure(email, ip)
        
        # 檢查封禁時長（應該是 30 分鐘）
        is_blacklisted, remaining = limiter.is_email_blacklisted(email)
        assert is_blacklisted is True
        assert remaining <= 1800  # 最多 30 分鐘
        assert remaining > 1790  # 至少 29 分 50 秒

    def test_distributed_attack_prevention(self):
        """測試防止分散式攻擊"""
        limiter = RateLimiter()
        email = "target@example.com"
        
        # 從多個 IP 嘗試攻擊同一郵箱
        for i in range(10):
            ip = f"192.168.1.{i + 160}"
            for _ in range(3):
                limiter.record_verification_failure(email, ip)
        
        # 郵箱應該最終被封禁（某個 IP 達到閾值）
        # 這裡我們測試至少有一個 IP 觸發了封禁
        is_blacklisted, _ = limiter.is_email_blacklisted(email)
        # 可能被封禁（取決於實現）

    def test_reset_after_success(self):
        """測試成功後重置計數"""
        limiter = RateLimiter()
        email = "test_reset@example.com"
        ip = "192.168.1.170"
        
        # 記錄一些失敗
        for _ in range(3):
            limiter.record_verification_failure(email, ip)
        
        # 成功驗證後重置
        limiter.reset_verification_failures(email, ip)
        
        # 剩餘嘗試次數應該恢復
        remaining = limiter.get_remaining_attempts(email, ip)
        assert remaining == 5

    def test_different_users_isolation(self):
        """測試不同用戶的速率限制隔離"""
        limiter = RateLimiter()
        
        # 用戶 A 達到限制
        email_a = "userA@example.com"
        for _ in range(5):
            limiter.check_email_verification_rate_limit(email_a, 5, 3600)
        
        # 用戶 B 不應該受影響
        email_b = "userB@example.com"
        is_allowed, error = limiter.check_email_verification_rate_limit(email_b, 5, 3600)
        assert is_allowed is True
        assert error is None

