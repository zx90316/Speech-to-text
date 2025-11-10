/**
 * 任務歷史組件
 */
import { useEffect, useState } from 'react';
import { History, RefreshCw, Clock, CheckCircle2, XCircle, Loader2, Trash2, Lock } from 'lucide-react';
import { api } from '../api';
import { getStoredTaskIds, removeTaskId } from '../utils/taskStorage';
import { getVerifiedEmail } from '../utils/emailStorage';
import type { Task } from '../types';

interface TaskHistoryProps {
  onSelectTask: (taskId: string) => void;
  refreshTrigger: number;
}

export function TaskHistory({ onSelectTask, refreshTrigger }: TaskHistoryProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(false);
  const [verifiedEmail, setVerifiedEmail] = useState<string | null>(getVerifiedEmail());

  const loadTasks = async (email: string) => {
    setLoading(true);
    try {
      // 1. 從 localStorage 獲取任務
      const storedIds = getStoredTaskIds();
      let allTasks: Task[] = [];

      if (storedIds.length > 0) {
        const batchResult = await api.getTasksBatch(storedIds);
        allTasks = batchResult.tasks;
      }

      // 2. 從郵箱獲取任務
      const history = await api.getMyTasks(email, 20);

      // 3. 合併兩個來源的任務，去重
      const taskMap = new Map<string, Task>();

      // 先加入 localStorage 的任務
      allTasks.forEach(task => {
        taskMap.set(task.task_id, task);
      });

      // 再加入郵箱的任務（不覆蓋已存在的）
      history.tasks.forEach(task => {
        if (!taskMap.has(task.task_id)) {
          taskMap.set(task.task_id, task);
        }
      });

      // 按建立時間排序
      const mergedTasks = Array.from(taskMap.values()).sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

      setTasks(mergedTasks);
    } catch (error) {
      console.error('載入任務歷史失敗:', error);
    } finally {
      setLoading(false);
    }
  };

  // 初始加載：如果有已驗證的郵箱，自動加載任務
  useEffect(() => {
    const email = getVerifiedEmail();
    if (email) {
      setVerifiedEmail(email);
      loadTasks(email);
    }
  }, []);

  // 刷新觸發：當 refreshTrigger 變化時重新加載
  useEffect(() => {
    const email = getVerifiedEmail();
    if (email) {
      setVerifiedEmail(email);
      if (refreshTrigger > 0) {
        loadTasks(email);
      }
    }
  }, [refreshTrigger]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 size={16} className="status-icon success" />;
      case 'failed':
      case 'canceled':
        return <XCircle size={16} className="status-icon error" />;
      case 'processing':
        return <Loader2 size={16} className="status-icon processing spin" />;
      case 'pending':
        return <Clock size={16} className="status-icon pending" />;
      default:
        return null;
    }
  };

  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      pending: '等待中',
      processing: '處理中',
      completed: '已完成',
      failed: '失敗',
      canceled: '已取消'
    };
    return statusMap[status] || status;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor(diff / (1000 * 60));

    if (minutes < 1) return '剛剛';
    if (minutes < 60) return `${minutes} 分鐘前`;
    if (hours < 24) return `${hours} 小時前`;
    
    return date.toLocaleString('zh-TW', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleDelete = async (taskId: string, filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`確定要刪除任務「${filename}」嗎？\n此操作將永久刪除任務記錄和相關檔案。`)) {
      try {
        await api.deleteTask(taskId);
        // 從 localStorage 移除
        removeTaskId(taskId);
        // 刷新任務列表
        if (verifiedEmail) {
          await loadTasks(verifiedEmail);
        }
      } catch (error) {
        console.error('刪除任務失敗:', error);
        alert('刪除任務失敗，請稍後再試。');
      }
    }
  };

  return (
    <div className="task-history">
      <div className="history-header">
        <h2 className="section-title">
          <History size={24} />
          任務歷史
        </h2>
        {verifiedEmail && (
          <button className="refresh-btn" onClick={() => loadTasks(verifiedEmail)} title="重新整理">
            <RefreshCw size={18} />
          </button>
        )}
      </div>

      {!verifiedEmail ? (
        <div className="locked-state">
          <Lock size={48} className="lock-icon" />
          <p>請先在左側「語音轉文字」區塊驗證您的電子郵件</p>
          <p className="sub-text">驗證後即可查看任務歷史</p>
        </div>
      ) : loading ? (
        <div className="loading-state">
          <Loader2 className="spin" size={32} />
          <p>載入中...</p>
        </div>
      ) : (
        <>
      {tasks.length === 0 ? (
        <div className="empty-state">
          <History size={48} />
          <p>尚無任務記錄</p>
          <p className="sub-text">上傳音訊檔案開始使用</p>
        </div>
      ) : (
        <div className="task-list">
          {tasks.map((task) => (
            <div
              key={task.task_id}
              className="task-item"
              onClick={() => onSelectTask(task.task_id)}
            >
              <div className="task-item-header">
                <div className="task-item-title">
                  <span className="filename">{task.filename}</span>
                  <div className="task-item-status">
                    {getStatusIcon(task.status)}
                    <span>{getStatusText(task.status)}</span>
                  </div>
                </div>
                <div className="task-item-actions">
                  {task.status === 'completed' && (
                    <span className="completed-badge" title="結果已發送至您的郵箱">
                      📧 已發送
                    </span>
                  )}
                  <button
                    className="delete-icon-btn"
                    onClick={(e) => handleDelete(task.task_id, task.filename, e)}
                    title="刪除任務"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>

              {task.status === 'processing' && (
                <div className="task-item-progress">
                  <div className="mini-progress-bar">
                    <div 
                      className="mini-progress-fill" 
                      style={{ width: `${task.progress}%` }}
                    />
                  </div>
                  <span className="progress-label">{task.progress.toFixed(0)}%</span>
                </div>
              )}

              <div className="task-item-meta">
                <span className="timestamp">{formatDate(task.created_at)}</span>
                <span className="task-id-short">ID: {task.task_id.slice(0, 8)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
        </>
      )}
    </div>
  );
}

