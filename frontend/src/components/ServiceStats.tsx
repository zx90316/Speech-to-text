/**
 * 服務統計組件
 */
import { useEffect, useState, useRef } from 'react';
import { Activity, Users, Clock } from 'lucide-react';
import { api } from '../api';
import type { ServiceStats as StatsType } from '../types';

export function ServiceStats() {
  const [stats, setStats] = useState<StatsType | null>(null);
  const mountedRef = useRef(true);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isLoadingRef = useRef(false); // 使用 ref 避免 useEffect 依賴

  useEffect(() => {
    mountedRef.current = true;
    let intervalId: NodeJS.Timeout;

    const loadStats = async () => {
      // 防止重複請求：如果正在載入，跳過此次輪詢
      if (isLoadingRef.current) {
        console.log('[ServiceStats] 跳過重複請求');
        return;
      }

      // 取消上一次未完成的請求
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      isLoadingRef.current = true;
      abortControllerRef.current = new AbortController();

      try {
        const data = await api.getStats();
        if (mountedRef.current) {
          setStats(data);
        }
      } catch (error: any) {
        // 如果是手動取消或組件已卸載，不輸出錯誤
        if (error.name !== 'CanceledError' && error.name !== 'AbortError' && mountedRef.current) {
          console.error('[ServiceStats] 載入統計資訊失敗:', error.message || error);
        }
      } finally {
        isLoadingRef.current = false;
        abortControllerRef.current = null;
      }
    };

    // 立即執行一次
    loadStats();

    // 設定輪詢（增加到 10 秒，減少請求頻率）
    intervalId = setInterval(loadStats, 10000);

    // 清理函數
    return () => {
      mountedRef.current = false;
      clearInterval(intervalId);
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
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

