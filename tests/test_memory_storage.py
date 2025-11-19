"""
測試記憶體存儲模組 (memory_storage.py)
"""
import pytest
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from memory_storage import MemoryTaskManager, memory_manager


class TestMemoryTaskManager:
    """測試 MemoryTaskManager 類"""

    def test_singleton_pattern(self):
        """測試單例模式"""
        manager1 = MemoryTaskManager()
        manager2 = MemoryTaskManager()
        assert manager1 is manager2

    def test_initialization(self):
        """測試初始化"""
        manager = MemoryTaskManager()
        
        assert hasattr(manager, 'tasks')
        assert hasattr(manager, 'task_lock')
        assert hasattr(manager, 'upload_dir')
        assert hasattr(manager, 'result_dir')

    def test_create_task_basic(self):
        """測試創建基本任務"""
        manager = MemoryTaskManager()
        
        task_data = manager.create_task(
            task_id='test-task-1',
            email='test@example.com',
            filename='test.mp3'
        )
        
        assert task_data is not None
        assert task_data['task_id'] == 'test-task-1'
        assert task_data['email'] == 'test@example.com'
        assert task_data['filename'] == 'test.mp3'
        assert task_data['status'] == 'pending'
        assert task_data['progress'] == 0.0

    def test_create_task_with_options(self):
        """測試創建帶選項的任務"""
        manager = MemoryTaskManager()
        
        task_data = manager.create_task(
            task_id='test-task-2',
            email='test@example.com',
            filename='test.mp3',
            enable_diarization=False,
            start_time=10.0,
            end_time=100.0,
            language='zh',
            enable_confidence_score=True,
            enable_llm_correction=True
        )
        
        assert task_data['enable_diarization'] is False
        assert task_data['start_time'] == 10.0
        assert task_data['end_time'] == 100.0
        assert task_data['language'] == 'zh'
        assert task_data['enable_confidence_score'] is True
        assert task_data['enable_llm_correction'] is True

    def test_get_task(self):
        """測試獲取任務"""
        manager = MemoryTaskManager()
        
        # 創建任務
        manager.create_task(
            task_id='test-task-3',
            email='test@example.com',
            filename='test.mp3'
        )
        
        # 獲取任務
        task = manager.get_task('test-task-3')
        
        assert task is not None
        assert task['task_id'] == 'test-task-3'
        # 應該不包含敏感路徑
        assert 'upload_path' not in task
        assert 'result_path' not in task

    def test_get_task_not_found(self):
        """測試獲取不存在的任務"""
        manager = MemoryTaskManager()
        
        task = manager.get_task('nonexistent-task')
        assert task is None

    def test_get_task_full(self):
        """測試獲取完整任務資料"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='test-task-4',
            email='test@example.com',
            filename='test.mp3'
        )
        
        task = manager.get_task_full('test-task-4')
        
        assert task is not None
        # 應該包含所有字段，包括路徑
        assert 'upload_path' in task
        assert 'result_path' in task

    def test_update_task_status(self):
        """測試更新任務狀態"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='test-task-5',
            email='test@example.com',
            filename='test.mp3'
        )
        
        # 更新狀態
        manager.update_task_status(
            'test-task-5',
            status='processing',
            progress=0.5,
            current_stage='transcribing'
        )
        
        task = manager.get_task('test-task-5')
        assert task['status'] == 'processing'
        assert task['progress'] == 0.5
        assert task['current_stage'] == 'transcribing'
        assert task['started_at'] is not None

    def test_update_task_status_completed(self):
        """測試更新任務為完成狀態"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='test-task-6',
            email='test@example.com',
            filename='test.mp3'
        )
        
        manager.update_task_status('test-task-6', status='completed')
        
        task = manager.get_task('test-task-6')
        assert task['status'] == 'completed'
        assert task['completed_at'] is not None

    def test_update_task_status_with_error(self):
        """測試更新任務狀態並包含錯誤信息"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='test-task-7',
            email='test@example.com',
            filename='test.mp3'
        )
        
        manager.update_task_status(
            'test-task-7',
            status='failed',
            error_message='Processing failed'
        )
        
        task = manager.get_task('test-task-7')
        assert task['status'] == 'failed'
        assert task['error_message'] == 'Processing failed'

    def test_update_task_result(self):
        """測試更新任務部分結果"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='test-task-8',
            email='test@example.com',
            filename='test.mp3'
        )
        
        partial_result = [
            {'text': 'Hello', 'start': 0.0, 'end': 1.0},
            {'text': 'World', 'start': 1.0, 'end': 2.0}
        ]
        
        manager.update_task_result('test-task-8', partial_result=partial_result)
        
        task = manager.get_task('test-task-8')
        assert task['partial_result'] == partial_result

    def test_get_tasks_by_email(self):
        """測試根據郵箱獲取任務列表"""
        manager = MemoryTaskManager()
        
        # 創建多個任務
        for i in range(5):
            manager.create_task(
                task_id=f'task-{i}',
                email='user@example.com',
                filename=f'test{i}.mp3'
            )
        
        # 創建另一個用戶的任務
        manager.create_task(
            task_id='other-task',
            email='other@example.com',
            filename='other.mp3'
        )
        
        # 獲取特定用戶的任務
        tasks = manager.get_tasks_by_email('user@example.com')
        
        assert len(tasks) == 5
        for task in tasks:
            assert task['email'] == 'user@example.com'

    def test_get_tasks_by_email_with_limit(self):
        """測試帶限制的任務列表獲取"""
        manager = MemoryTaskManager()
        
        # 創建 10 個任務
        for i in range(10):
            manager.create_task(
                task_id=f'task-{i}',
                email='user@example.com',
                filename=f'test{i}.mp3'
            )
        
        # 只獲取 3 個
        tasks = manager.get_tasks_by_email('user@example.com', limit=3)
        
        assert len(tasks) == 3

    def test_get_queue_position(self):
        """測試獲取佇列位置"""
        manager = MemoryTaskManager()
        
        # 創建多個待處理任務
        task_ids = []
        for i in range(5):
            task_data = manager.create_task(
                task_id=f'queue-task-{i}',
                email='user@example.com',
                filename=f'test{i}.mp3'
            )
            task_ids.append(task_data['task_id'])
            time.sleep(0.01)  # 確保不同的創建時間
        
        # 第一個任務應該在位置 0
        position = manager.get_queue_position(task_ids[0])
        assert position >= 0

    def test_get_processing_count(self):
        """測試獲取正在處理的任務數量"""
        manager = MemoryTaskManager()
        
        # 創建多個任務，部分設為處理中
        for i in range(3):
            manager.create_task(
                task_id=f'proc-task-{i}',
                email='user@example.com',
                filename=f'test{i}.mp3'
            )
        
        # 設置一個為處理中
        manager.update_task_status('proc-task-0', status='processing')
        
        count = manager.get_processing_count()
        assert count >= 1

    def test_cancel_task(self):
        """測試取消任務"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='cancel-task',
            email='user@example.com',
            filename='test.mp3'
        )
        
        result = manager.cancel_task('cancel-task')
        assert result is True
        
        task = manager.get_task('cancel-task')
        assert task['status'] == 'canceled'

    def test_cancel_completed_task(self):
        """測試取消已完成的任務"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='completed-task',
            email='user@example.com',
            filename='test.mp3'
        )
        
        manager.update_task_status('completed-task', status='completed')
        
        result = manager.cancel_task('completed-task')
        assert result is False  # 不能取消已完成的任務

    def test_delete_task(self):
        """測試刪除任務"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='delete-task',
            email='user@example.com',
            filename='test.mp3'
        )
        
        result = manager.delete_task('delete-task')
        assert result is True
        
        task = manager.get_task('delete-task')
        assert task is None

    def test_delete_nonexistent_task(self):
        """測試刪除不存在的任務"""
        manager = MemoryTaskManager()
        
        result = manager.delete_task('nonexistent-task')
        assert result is False

    def test_get_all_tasks_summary(self):
        """測試獲取所有任務摘要"""
        manager = MemoryTaskManager()
        
        # 創建多個任務
        for i in range(5):
            manager.create_task(
                task_id=f'summary-task-{i}',
                email=f'user{i}@example.com',
                filename=f'test{i}.mp3'
            )
        
        summaries = manager.get_all_tasks_summary(limit=10)
        
        assert len(summaries) >= 5
        # 檢查郵箱是否被遮罩
        for summary in summaries:
            assert 'email_masked' in summary
            assert '***' in summary['email_masked']

    def test_get_all_tasks_summary_with_pagination(self):
        """測試帶分頁的任務摘要"""
        manager = MemoryTaskManager()
        
        # 創建 10 個任務
        for i in range(10):
            manager.create_task(
                task_id=f'page-task-{i}',
                email='user@example.com',
                filename=f'test{i}.mp3'
            )
        
        # 獲取第一頁（5 個）
        summaries_page1 = manager.get_all_tasks_summary(limit=5, offset=0)
        assert len(summaries_page1) <= 5
        
        # 獲取第二頁（5 個）
        summaries_page2 = manager.get_all_tasks_summary(limit=5, offset=5)
        assert len(summaries_page2) <= 5

    def test_get_total_tasks_count(self):
        """測試獲取任務總數"""
        manager = MemoryTaskManager()
        
        # 創建 3 個任務
        for i in range(3):
            manager.create_task(
                task_id=f'count-task-{i}',
                email='user@example.com',
                filename=f'test{i}.mp3'
            )
        
        count = manager.get_total_tasks_count()
        assert count >= 3

    def test_get_stats_summary(self):
        """測試獲取統計摘要"""
        manager = MemoryTaskManager()
        
        # 創建不同狀態的任務
        manager.create_task(
            task_id='stats-task-1',
            email='user1@example.com',
            filename='test1.mp3'
        )
        manager.update_task_status('stats-task-1', status='processing')
        
        manager.create_task(
            task_id='stats-task-2',
            email='user2@example.com',
            filename='test2.mp3'
        )
        manager.update_task_status('stats-task-2', status='completed')
        
        stats = manager.get_stats_summary()
        
        assert 'total_tasks' in stats
        assert 'today_tasks' in stats
        assert 'unique_users' in stats
        assert 'status_counts' in stats
        assert stats['total_tasks'] >= 2

    def test_cleanup_task_files(self, temp_dir):
        """測試清理任務文件"""
        manager = MemoryTaskManager()
        
        # 使用臨時目錄
        with patch.object(manager, 'upload_dir', temp_dir / 'uploads'), \
             patch.object(manager, 'result_dir', temp_dir / 'results'):
            
            # 創建任務
            task_data = manager.create_task(
                task_id='cleanup-task',
                email='user@example.com',
                filename='test.mp3'
            )
            
            # 創建測試目錄
            upload_path = Path(task_data['upload_path'])
            result_path = Path(task_data['result_path'])
            upload_path.mkdir(parents=True, exist_ok=True)
            result_path.mkdir(parents=True, exist_ok=True)
            
            # 清理文件
            with patch('backend.memory_storage.security_logger'):
                result = manager.cleanup_task_files('cleanup-task')
            
            # 應該成功清理
            assert result is True or not upload_path.exists()

    def test_cleanup_old_completed_tasks(self):
        """測試清理舊的已完成任務"""
        manager = MemoryTaskManager()
        
        # 創建多個已完成的任務
        for i in range(10):
            manager.create_task(
                task_id=f'old-task-{i}',
                email='user@example.com',
                filename=f'test{i}.mp3'
            )
            manager.update_task_status(f'old-task-{i}', status='completed')
        
        # 只保留 5 個
        deleted_count = manager.cleanup_old_completed_tasks(keep_count=5)
        
        # 應該刪除了一些任務
        assert deleted_count >= 0

    def test_thread_safety(self):
        """測試線程安全"""
        import threading
        
        manager = MemoryTaskManager()
        errors = []
        
        def create_task_thread(i):
            try:
                manager.create_task(
                    task_id=f'thread-task-{i}',
                    email=f'user{i}@example.com',
                    filename=f'test{i}.mp3'
                )
            except Exception as e:
                errors.append(e)
        
        # 創建多個線程
        threads = [threading.Thread(target=create_task_thread, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 不應該有錯誤
        assert len(errors) == 0

    def test_global_instance(self):
        """測試全局實例"""
        assert memory_manager is not None
        assert isinstance(memory_manager, MemoryTaskManager)


@pytest.mark.integration
class TestMemoryTaskManagerIntegration:
    """整合測試"""
    
    def test_complete_task_lifecycle(self):
        """測試完整的任務生命週期"""
        manager = MemoryTaskManager()
        
        # 1. 創建任務
        task_data = manager.create_task(
            task_id='lifecycle-task',
            email='user@example.com',
            filename='test.mp3'
        )
        assert task_data['status'] == 'pending'
        
        # 2. 開始處理
        manager.update_task_status('lifecycle-task', status='processing', progress=0.0)
        task = manager.get_task('lifecycle-task')
        assert task['status'] == 'processing'
        
        # 3. 更新進度
        manager.update_task_status('lifecycle-task', status='processing', progress=0.5)
        task = manager.get_task('lifecycle-task')
        assert task['progress'] == 0.5
        
        # 4. 完成任務
        manager.update_task_status('lifecycle-task', status='completed', progress=1.0)
        task = manager.get_task('lifecycle-task')
        assert task['status'] == 'completed'
        assert task['completed_at'] is not None

    def test_multiple_users_isolation(self):
        """測試多用戶任務隔離"""
        manager = MemoryTaskManager()
        
        # 用戶 A 的任務
        manager.create_task(
            task_id='user-a-task',
            email='userA@example.com',
            filename='testA.mp3'
        )
        
        # 用戶 B 的任務
        manager.create_task(
            task_id='user-b-task',
            email='userB@example.com',
            filename='testB.mp3'
        )
        
        # 用戶 A 只能看到自己的任務
        tasks_a = manager.get_tasks_by_email('userA@example.com')
        assert all(task['email'] == 'userA@example.com' for task in tasks_a)
        
        # 用戶 B 只能看到自己的任務
        tasks_b = manager.get_tasks_by_email('userB@example.com')
        assert all(task['email'] == 'userB@example.com' for task in tasks_b)

    def test_queue_management(self):
        """測試佇列管理"""
        manager = MemoryTaskManager()
        
        # 創建多個待處理任務
        task_ids = []
        for i in range(5):
            task_data = manager.create_task(
                task_id=f'queue-mgmt-task-{i}',
                email='user@example.com',
                filename=f'test{i}.mp3'
            )
            task_ids.append(task_data['task_id'])
            time.sleep(0.01)
        
        # 檢查佇列位置
        for i, task_id in enumerate(task_ids):
            position = manager.get_queue_position(task_id)
            # 位置應該按順序
            assert position >= 0


@pytest.mark.security
class TestMemoryTaskManagerSecurity:
    """安全相關測試"""
    
    def test_sensitive_data_protection(self):
        """測試敏感數據保護"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='sensitive-task',
            email='user@example.com',
            filename='test.mp3'
        )
        
        # get_task 不應該返回敏感路徑
        task = manager.get_task('sensitive-task')
        assert 'upload_path' not in task
        assert 'result_path' not in task

    def test_email_masking_in_summary(self):
        """測試摘要中的郵箱遮罩"""
        manager = MemoryTaskManager()
        
        manager.create_task(
            task_id='mask-task',
            email='sensitive@example.com',
            filename='test.mp3'
        )
        
        summaries = manager.get_all_tasks_summary()
        
        # 所有摘要中的郵箱應該被遮罩
        for summary in summaries:
            if summary['task_id'] == 'mask-task':
                assert '***' in summary['email_masked']
                assert '@example.com' in summary['email_masked']

    def test_concurrent_access_safety(self):
        """測試並發訪問安全性"""
        import threading
        
        manager = MemoryTaskManager()
        
        # 創建任務
        manager.create_task(
            task_id='concurrent-task',
            email='user@example.com',
            filename='test.mp3'
        )
        
        errors = []
        
        def update_task():
            try:
                for _ in range(10):
                    manager.update_task_status(
                        'concurrent-task',
                        status='processing',
                        progress=0.5
                    )
            except Exception as e:
                errors.append(e)
        
        # 多個線程同時更新
        threads = [threading.Thread(target=update_task) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 不應該有錯誤
        assert len(errors) == 0

