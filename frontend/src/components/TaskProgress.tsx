/**
 * 任務進度組件
 */
import { useEffect, useState, useRef } from 'react';
import { Loader2, CheckCircle2, XCircle, Download, X, Clock, Users, Eye } from 'lucide-react';
import { api } from '../api';
import type { Task, ProgressEvent } from '../types';

interface TaskProgressProps {
  taskId: string;
  onClose: () => void;
  onComplete: () => void;
}

export function TaskProgress({ taskId, onClose, onComplete }: TaskProgressProps) {
  const [task, setTask] = useState<Task | null>(null);
  const [progressEvent, setProgressEvent] = useState<ProgressEvent | null>(null);
  const [canceling, setCanceling] = useState(false);
  const [showPartialModal, setShowPartialModal] = useState(false);
  const hasCompletedRef = useRef(false);

  useEffect(() => {
    let isMounted = true;
    hasCompletedRef.current = false;
    
    // 首次載入任務資訊
    api.getTask(taskId).then(data => {
      if (isMounted) setTask(data);
    }).catch(console.error);

    // 建立 SSE 連接
    const eventSource = api.createProgressStream(taskId);

    eventSource.onmessage = (event) => {
      if (!isMounted) return;

      const data: ProgressEvent = JSON.parse(event.data);
      setProgressEvent(data);

      // 當任務完成時，重新獲取完整的任務資訊以確保 has_result 正確
      if (data.status === 'completed' && !hasCompletedRef.current) {
        hasCompletedRef.current = true;
        api.getTask(taskId).then(taskData => {
          if (isMounted) setTask(taskData);
        }).catch(console.error);
        eventSource.close();
        // 延遲調用以確保狀態更新完成
        setTimeout(() => {
          if (isMounted) onComplete();
        }, 500);
      } else if (data.status === 'failed' || data.status === 'canceled') {
        // 任務失敗或取消時，也要更新任務資訊並刷新歷史
        if (!hasCompletedRef.current) {
          hasCompletedRef.current = true;
          api.getTask(taskId).then(taskData => {
            if (isMounted) setTask(taskData);
          }).catch(console.error);
          eventSource.close();
          setTimeout(() => {
            if (isMounted) onComplete();
          }, 500);
        }
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE 錯誤:', error);
      eventSource.close();
    };

    return () => {
      isMounted = false;
      eventSource.close();
    };
  }, [taskId, onComplete]);

  const handleCancel = async () => {
    if (confirm('確定要取消此任務嗎？')) {
      setCanceling(true);
      try {
        await api.cancelTask(taskId);
      } catch (error) {
        console.error('取消任務失敗:', error);
      } finally {
        setCanceling(false);
      }
    }
  };

  const handleDownload = (fileType: 'transcript' | 'raw' | 'confidence_html') => {
    const url = api.downloadResult(taskId, fileType);
    const a = document.createElement('a');
    a.href = url;
    const extension = fileType === 'confidence_html' ? 'html' : 'txt';
    a.download = `${task?.filename}_${fileType}.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (!task) {
    return (
      <div className="task-progress loading">
        <Loader2 className="spin" size={32} />
        <p>載入中...</p>
      </div>
    );
  }

  const currentProgress = progressEvent?.progress ?? task.progress;
  const currentStage = progressEvent?.current_stage ?? task.current_stage ?? task.status;
  const queuePosition = progressEvent?.queue_position ?? task.queue_position;
  const currentStatus = progressEvent?.status ?? task.status;
  const hasResult = progressEvent?.has_result ?? task.has_result;

  return (
    <div className="task-progress">
      <div className="task-header">
        <div className="task-info">
          <h3>{task.filename}</h3>
          <div className="task-meta">
            {task.enable_diarization && (
              <span className="badge">
                <Users size={14} />
                語者分離
              </span>
            )}
            <span className="task-id">ID: {taskId.slice(0, 8)}...</span>
          </div>
        </div>
        <button className="close-btn" onClick={onClose} title="關閉">
          <X size={20} />
        </button>
      </div>

      <div className="status-section">
        {currentStatus === 'pending' && (
          <div className="status pending">
            <Clock className="status-icon" size={24} />
            <div>
              <p className="status-text">等待處理中</p>
              <p className="status-detail">前方還有 {queuePosition} 個任務</p>
            </div>
          </div>
        )}

        {currentStatus === 'processing' && (
          <div className="status processing">
            <Loader2 className="status-icon spin" size={24} />
            <div>
              <p className="status-text">處理中</p>
              <p className="status-detail">{currentStage}</p>
            </div>
          </div>
        )}

        {currentStatus === 'completed' && (
          <div className="status completed">
            <CheckCircle2 className="status-icon" size={24} />
            <div>
              <p className="status-text">轉錄完成！</p>
              <p className="status-detail">已完成處理</p>
            </div>
          </div>
        )}

        {currentStatus === 'failed' && (
          <div className="status failed">
            <XCircle className="status-icon" size={24} />
            <div>
              <p className="status-text">處理失敗</p>
              <p className="status-detail">{task.error_message || '未知錯誤'}</p>
            </div>
          </div>
        )}

        {currentStatus === 'canceled' && (
          <div className="status canceled">
            <XCircle className="status-icon" size={24} />
            <div>
              <p className="status-text">已取消</p>
              <p className="status-detail">任務已被取消</p>
            </div>
          </div>
        )}
      </div>

      <div className="progress-section">
        <div className="progress-bar-container">
          <div 
            className="progress-bar" 
            style={{ width: `${currentProgress}%` }}
          />
        </div>
        <div className="progress-text">{currentProgress.toFixed(1)}%</div>
      </div>

      {progressEvent?.partial_result && progressEvent.partial_result.length > 0 && (
        <div className="partial-result-preview">
          <div className="preview-header">
            <span>部分轉錄結果預覽 ({progressEvent.partial_result.length} 段)</span>
            <button
              className="view-all-btn"
              onClick={() => setShowPartialModal(true)}
            >
              <Eye size={16} />
              查看全部
            </button>
          </div>
          <div className="result-preview-mini">
            {progressEvent.partial_result.slice(-3).map((segment, idx) => (
              <div key={idx} className="segment-mini">
                <span className="timestamp">
                  [{segment.start.toFixed(1)}s]
                </span>
                {segment.speaker && <span className="speaker">{segment.speaker}:</span>}
                <span className="text">{segment.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {showPartialModal && progressEvent?.partial_result && (
        <div className="modal-overlay" onClick={() => setShowPartialModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>部分轉錄結果</h3>
              <button className="modal-close-btn" onClick={() => setShowPartialModal(false)}>
                <X size={24} />
              </button>
            </div>
            <div className="modal-body">
              <div className="modal-stats">
                <span>共 {progressEvent.partial_result.length} 段</span>
                <span>進度: {currentProgress.toFixed(1)}%</span>
              </div>
              <div className="result-list">
                {progressEvent.partial_result.map((segment, idx) => (
                  <div key={idx} className="segment-full">
                    <div className="segment-header">
                      <span className="segment-number">#{idx + 1}</span>
                      <span className="timestamp">
                        {segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s
                      </span>
                      {segment.speaker && (
                        <span className="speaker-badge">{segment.speaker}</span>
                      )}
                    </div>
                    <div className="segment-text">{segment.text}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="actions">
        {(currentStatus === 'pending' || currentStatus === 'processing') && (
          <button
            className="cancel-btn"
            onClick={handleCancel}
            disabled={canceling}
          >
            {canceling ? <Loader2 className="spin" size={16} /> : <X size={16} />}
            取消任務
          </button>
        )}

        {currentStatus === 'completed' && hasResult && (
          <>
            <button 
              className="download-btn primary" 
              onClick={() => handleDownload('transcript')}
            >
              <Download size={16} />
              下載結果
            </button>
            <button 
              className="download-btn secondary" 
              onClick={() => handleDownload('raw')}
            >
              <Download size={16} />
              下載原始 ASR
            </button>
            <button 
              className="download-btn secondary" 
              onClick={() => handleDownload('confidence_html')}
              title="下載信心度視覺化 HTML（需啟用詞級時間戳或信心分數）"
            >
              <Download size={16} />
              下載信心度視覺化
            </button>
          </>
        )}
      </div>
    </div>
  );
}

