/**
 * 預處理任務歷史組件
 */
import { useEffect, useState } from 'react';
import { History, Download, RefreshCw, Clock, CheckCircle2, XCircle, Loader2, Trash2 } from 'lucide-react';
import { api } from '../api';
import { getStoredPreprocessTaskIds, removePreprocessTaskId } from '../utils/taskStorage';
import './PreprocessHistory.css';

interface PreprocessTask {
  preprocess_id: string;
  filename: string;
  status: string;
  progress: number;
  current_stage?: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
}

interface PreprocessHistoryProps {
  onSelectTask?: (preprocessId: string) => void;
  refreshTrigger?: number;
}

export function PreprocessHistory({ onSelectTask, refreshTrigger = 0 }: PreprocessHistoryProps) {
  const [tasks, setTasks] = useState<PreprocessTask[]>([]);
  const [loading, setLoading] = useState(true);

  const loadTasks = async () => {
    setLoading(true);
    try {
      // 1. 從 localStorage 獲取任務 ID
      const storedIds = getStoredPreprocessTaskIds();
      let allTasks: PreprocessTask[] = [];

      // 批量查詢任務（使用單個查詢）
      if (storedIds.length > 0) {
        const taskPromises = storedIds.map(id =>
          api.getPreprocessTask(id).catch(() => null)
        );
        const results = await Promise.all(taskPromises);
        allTasks = results.filter(t => t !== null) as PreprocessTask[];
      }

      // 2. 從 IP 獲取任務（作為備份）
      const history = await api.getMyPreprocessTasks(20);

      // 3. 合併兩個來源的任務，去重
      const taskMap = new Map<string, PreprocessTask>();

      // 先加入 localStorage 的任務
      allTasks.forEach(task => {
        taskMap.set(task.preprocess_id, task);
      });

      // 再加入 IP 的任務（不覆蓋已存在的）
      history.tasks.forEach(task => {
        if (!taskMap.has(task.preprocess_id)) {
          taskMap.set(task.preprocess_id, task);
        }
      });

      // 按建立時間排序
      const mergedTasks = Array.from(taskMap.values()).sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

      setTasks(mergedTasks);
    } catch (error) {
      console.error('載入預處理任務歷史失敗:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTasks();
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

  const handleDownload = (preprocessId: string, filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const url = api.downloadPreprocessedAudio(preprocessId, 'processed');
    const a = document.createElement('a');
    a.href = url;
    a.download = `processed_${filename}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleDelete = async (preprocessId: string, filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`確定要刪除預處理任務「${filename}」嗎？\n此操作將永久刪除任務記錄和相關檔案。`)) {
      try {
        await api.deletePreprocess(preprocessId, true);
        // 從 localStorage 移除
        removePreprocessTaskId(preprocessId);
        // 刷新任務列表
        await loadTasks();
      } catch (error) {
        console.error('刪除預處理任務失敗:', error);
        alert('刪除預處理任務失敗，請稍後再試。');
      }
    }
  };

  return (
    <div className="task-history">
      <div className="history-header">
        <div className="header-left">
          <History size={20} />
          <h3>預處理歷史</h3>
          <span className="task-count">({tasks.length})</span>
        </div>
        <button onClick={loadTasks} className="btn-refresh" disabled={loading}>
          <RefreshCw size={16} className={loading ? 'spin' : ''} />
          刷新
        </button>
      </div>

      {loading ? (
        <div className="history-loading">
          <Loader2 size={24} className="spin" />
          <p>載入中...</p>
        </div>
      ) : tasks.length === 0 ? (
        <div className="history-empty">
          <History size={48} />
          <p>尚無預處理任務</p>
        </div>
      ) : (
        <div className="history-list">
          {tasks.map(task => (
            <div
              key={task.preprocess_id}
              className={`history-item status-${task.status}`}
              onClick={() => onSelectTask?.(task.preprocess_id)}
            >
              <div className="item-header">
                <div className="item-status">
                  {getStatusIcon(task.status)}
                  <span className="status-text">{getStatusText(task.status)}</span>
                  {task.status === 'processing' && (
                    <span className="progress-badge">{Math.round(task.progress)}%</span>
                  )}
                </div>
                <span className="item-time">{formatDate(task.created_at)}</span>
              </div>

              <div className="item-filename">{task.filename}</div>

              {task.current_stage && task.status === 'processing' && (
                <div className="item-stage">{task.current_stage}</div>
              )}

              {task.error_message && (
                <div className="item-error">{task.error_message}</div>
              )}

              <div className="item-actions">
                {task.status === 'completed' && (
                  <button
                    onClick={(e) => handleDownload(task.preprocess_id, task.filename, e)}
                    className="btn-download"
                    title="下載處理後音訊"
                  >
                    <Download size={14} />
                    下載
                  </button>
                )}
                <button
                  onClick={(e) => handleDelete(task.preprocess_id, task.filename, e)}
                  className="btn-delete"
                  title="刪除任務"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
