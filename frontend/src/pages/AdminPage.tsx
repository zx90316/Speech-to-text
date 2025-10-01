/**
 * 管理者頁面
 */
import { useState, useEffect } from 'react';
import {
  Shield, Users, FileText, CheckCircle2, XCircle, Clock,
  Loader2, Trash2, RefreshCw, Download, AlertTriangle
} from 'lucide-react';

interface AdminTask {
  task_id: string;
  client_ip: string;
  filename: string;
  status: string;
  progress: number;
  enable_diarization: boolean;
  created_at: string;
  completed_at: string | null;
  has_result: boolean;
}

interface AdminStats {
  total_tasks: number;
  today_tasks: number;
  unique_users: number;
  status_counts: Record<string, number>;
  queue_size: number;
  processing_count: number;
  is_processing: boolean;
}

export function AdminPage() {
  const [adminToken, setAdminToken] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [tasks, setTasks] = useState<AdminTask[]>([]);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState(0);
  const [total, setTotal] = useState(0);
  const limit = 50;

  const handleLogin = () => {
    if (adminToken.trim()) {
      localStorage.setItem('admin_token', adminToken);
      setIsAuthenticated(true);
      loadData();
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    setAdminToken('');
    setIsAuthenticated(false);
  };

  useEffect(() => {
    const saved = localStorage.getItem('admin_token');
    if (saved) {
      setAdminToken(saved);
      setIsAuthenticated(true);
    }
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      loadData();
    }
  }, [isAuthenticated, statusFilter, currentPage]);

  const loadData = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('admin_token') || adminToken;

      // 載入任務列表
      const tasksParams = new URLSearchParams({
        token,
        limit: limit.toString(),
        offset: (currentPage * limit).toString(),
      });
      if (statusFilter) {
        tasksParams.append('status', statusFilter);
      }

      const tasksRes = await fetch(`/api/admin/tasks?${tasksParams}`);
      if (tasksRes.status === 403) {
        handleLogout();
        return;
      }
      const tasksData = await tasksRes.json();
      setTasks(tasksData.tasks);
      setTotal(tasksData.total);

      // 載入統計資訊
      const statsRes = await fetch(`/api/admin/stats?token=${token}`);
      const statsData = await statsRes.json();
      setStats(statsData);
    } catch (error) {
      console.error('載入資料失敗:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectTask = (taskId: string) => {
    const newSelected = new Set(selectedTasks);
    if (newSelected.has(taskId)) {
      newSelected.delete(taskId);
    } else {
      newSelected.add(taskId);
    }
    setSelectedTasks(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedTasks.size === tasks.length) {
      setSelectedTasks(new Set());
    } else {
      setSelectedTasks(new Set(tasks.map(t => t.task_id)));
    }
  };

  const handleBatchDelete = async () => {
    if (selectedTasks.size === 0) return;

    if (!confirm(`確定要刪除 ${selectedTasks.size} 個任務嗎？`)) return;

    try {
      const token = localStorage.getItem('admin_token');
      const taskIds = Array.from(selectedTasks).join('&task_ids=');
      await fetch(`/api/admin/tasks/batch-delete?token=${token}&task_ids=${taskIds}`, {
        method: 'POST'
      });

      setSelectedTasks(new Set());
      loadData();
    } catch (error) {
      console.error('批量刪除失敗:', error);
    }
  };

  const handleCleanup = async () => {
    const days = prompt('清理幾天前的任務？', '7');
    if (!days) return;

    try {
      const token = localStorage.getItem('admin_token');
      const res = await fetch(`/api/admin/cleanup?token=${token}&days=${days}`, {
        method: 'POST'
      });
      const data = await res.json();
      alert(data.message);
      loadData();
    } catch (error) {
      console.error('清理失敗:', error);
    }
  };

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

  const formatDate = (dateStr: string) => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleString('zh-TW');
  };

  if (!isAuthenticated) {
    return (
      <div className="admin-login">
        <div className="login-card">
          <Shield size={48} className="login-icon" />
          <h1>管理者登入</h1>
          <input
            type="password"
            placeholder="請輸入管理者 Token"
            value={adminToken}
            onChange={(e) => setAdminToken(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
          />
          <button onClick={handleLogin}>登入</button>
        </div>
      </div>
    );
  }

  const totalPages = Math.ceil(total / limit);

  return (
    <div className="admin-page">
      <header className="admin-header">
        <div className="header-content">
          <div className="header-left">
            <Shield size={32} />
            <div>
              <h1>系統管理面板</h1>
              <p className="subtitle">Whisper 語音轉文字管理</p>
            </div>
          </div>
          <button className="logout-btn" onClick={handleLogout}>登出</button>
        </div>
      </header>

      {stats && (
        <div className="admin-stats-grid">
          <div className="stat-card">
            <FileText size={24} />
            <div>
              <div className="stat-value">{stats.total_tasks}</div>
              <div className="stat-label">總任務數</div>
            </div>
          </div>
          <div className="stat-card">
            <Clock size={24} />
            <div>
              <div className="stat-value">{stats.today_tasks}</div>
              <div className="stat-label">今日任務</div>
            </div>
          </div>
          <div className="stat-card">
            <Users size={24} />
            <div>
              <div className="stat-value">{stats.unique_users}</div>
              <div className="stat-label">獨特用戶</div>
            </div>
          </div>
          <div className="stat-card">
            <Loader2 size={24} className={stats.is_processing ? 'spin' : ''} />
            <div>
              <div className="stat-value">{stats.queue_size + stats.processing_count}</div>
              <div className="stat-label">處理中/等待</div>
            </div>
          </div>
        </div>
      )}

      <div className="admin-controls">
        <div className="controls-left">
          <button className="control-btn" onClick={loadData} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spin' : ''} />
            刷新
          </button>
          <button
            className="control-btn danger"
            onClick={handleBatchDelete}
            disabled={selectedTasks.size === 0}
          >
            <Trash2 size={16} />
            刪除選中 ({selectedTasks.size})
          </button>
          <button className="control-btn warning" onClick={handleCleanup}>
            <AlertTriangle size={16} />
            清理舊任務
          </button>
        </div>

        <select
          className="status-filter"
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setCurrentPage(0);
          }}
        >
          <option value="">全部狀態</option>
          <option value="pending">等待中</option>
          <option value="processing">處理中</option>
          <option value="completed">已完成</option>
          <option value="failed">失敗</option>
          <option value="canceled">已取消</option>
        </select>
      </div>

      <div className="admin-table-container">
        <table className="admin-table">
          <thead>
            <tr>
              <th>
                <input
                  type="checkbox"
                  checked={selectedTasks.size === tasks.length && tasks.length > 0}
                  onChange={handleSelectAll}
                />
              </th>
              <th>檔案名稱</th>
              <th>狀態</th>
              <th>進度</th>
              <th>客戶端 IP</th>
              <th>建立時間</th>
              <th>完成時間</th>
              <th>任務 ID</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={8} className="loading-cell">
                  <Loader2 className="spin" size={32} />
                  <p>載入中...</p>
                </td>
              </tr>
            ) : tasks.length === 0 ? (
              <tr>
                <td colSpan={8} className="empty-cell">
                  <FileText size={48} />
                  <p>無任務記錄</p>
                </td>
              </tr>
            ) : (
              tasks.map((task) => (
                <tr key={task.task_id}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedTasks.has(task.task_id)}
                      onChange={() => handleSelectTask(task.task_id)}
                    />
                  </td>
                  <td className="filename-cell">{task.filename}</td>
                  <td>
                    <div className="status-badge">
                      {getStatusIcon(task.status)}
                      {task.status}
                    </div>
                  </td>
                  <td>{task.progress.toFixed(0)}%</td>
                  <td>{task.client_ip}</td>
                  <td>{formatDate(task.created_at)}</td>
                  <td>{formatDate(task.completed_at || '')}</td>
                  <td className="task-id-cell">{task.task_id.slice(0, 8)}...</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button
            onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
            disabled={currentPage === 0}
          >
            上一頁
          </button>
          <span>
            第 {currentPage + 1} 頁，共 {totalPages} 頁 (總共 {total} 筆)
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={currentPage >= totalPages - 1}
          >
            下一頁
          </button>
        </div>
      )}
    </div>
  );
}
