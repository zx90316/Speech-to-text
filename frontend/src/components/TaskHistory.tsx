/**
 * 任務歷史組件
 */
import { useEffect, useState } from 'react';
import { History, Download, RefreshCw, Clock, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { api } from '../api';
import type { Task } from '../types';

interface TaskHistoryProps {
  onSelectTask: (taskId: string) => void;
  refreshTrigger: number;
}

export function TaskHistory({ onSelectTask, refreshTrigger }: TaskHistoryProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [clientIp, setClientIp] = useState('');

  const loadTasks = async () => {
    setLoading(true);
    try {
      const history = await api.getMyTasks(20);
      setTasks(history.tasks);
      setClientIp(history.client_ip);
    } catch (error) {
      console.error('載入任務歷史失敗:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    
    const loadTasksIfMounted = async () => {
      if (!isMounted) return;
      setLoading(true);
      try {
        const history = await api.getMyTasks(20);
        if (isMounted) {
          setTasks(history.tasks);
          setClientIp(history.client_ip);
        }
      } catch (error) {
        console.error('載入任務歷史失敗:', error);
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };
    
    loadTasksIfMounted();
    
    return () => {
      isMounted = false;
    };
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

  const handleDownload = (taskId: string, filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const url = api.downloadResult(taskId, 'transcript');
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}_transcript.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (loading) {
    return (
      <div className="task-history">
        <div className="history-header">
          <h2 className="section-title">
            <History size={24} />
            任務歷史
          </h2>
        </div>
        <div className="loading-state">
          <Loader2 className="spin" size={32} />
          <p>載入中...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="task-history">
      <div className="history-header">
        <h2 className="section-title">
          <History size={24} />
          任務歷史
        </h2>
        <button className="refresh-btn" onClick={loadTasks} title="重新整理">
          <RefreshCw size={18} />
        </button>
      </div>

      {clientIp && (
        <div className="client-info">
          <span>您的 IP: {clientIp}</span>
        </div>
      )}

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
                {task.status === 'completed' && task.has_result && (
                  <button
                    className="download-icon-btn"
                    onClick={(e) => handleDownload(task.task_id, task.filename, e)}
                    title="下載結果"
                  >
                    <Download size={16} />
                  </button>
                )}
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
    </div>
  );
}

