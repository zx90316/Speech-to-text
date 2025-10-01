/**
 * 服務統計組件
 */
import { useEffect, useState } from 'react';
import { Activity, Users, Clock } from 'lucide-react';
import { api } from '../api';
import type { ServiceStats as StatsType } from '../types';

export function ServiceStats() {
  const [stats, setStats] = useState<StatsType | null>(null);

  useEffect(() => {
    const loadStats = async () => {
      try {
        const data = await api.getStats();
        setStats(data);
      } catch (error) {
        console.error('載入統計資訊失敗:', error);
      }
    };

    loadStats();
    const interval = setInterval(loadStats, 5000); // 每 5 秒更新

    return () => clearInterval(interval);
  }, []);

  if (!stats) return null;

  return (
    <div className="service-stats">
      <div className="stat-item">
        <Activity size={20} />
        <div>
          <div className="stat-value">{stats.is_processing ? '處理中' : '閒置'}</div>
          <div className="stat-label">服務狀態</div>
        </div>
      </div>
      <div className="stat-item">
        <Users size={20} />
        <div>
          <div className="stat-value">{stats.processing_count}</div>
          <div className="stat-label">正在處理</div>
        </div>
      </div>
      <div className="stat-item">
        <Clock size={20} />
        <div>
          <div className="stat-value">{stats.queue_size}</div>
          <div className="stat-label">排隊中</div>
        </div>
      </div>
    </div>
  );
}

