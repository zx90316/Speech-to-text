/**
 * 主應用組件
 */
import { useState, useCallback } from 'react';
import { UploadSection } from '../components/UploadSection';
import { TaskProgress } from '../components/TaskProgress';
import { TaskHistory } from '../components/TaskHistory';
import { ServiceStats } from '../components/ServiceStats';
import { api } from '../api';
import { addTaskId } from '../utils/taskStorage';
import { Mic } from 'lucide-react';

function App() {
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [uploading, setUploading] = useState(false);

  const handleUpload = useCallback(async (file: File, enableDiarization: boolean) => {
    setUploading(true);
    try {
      const response = await api.createTask(file, enableDiarization);
      setCurrentTaskId(response.task_id);
      // 儲存任務 ID 到 localStorage
      addTaskId(response.task_id);
      // 提交新任務後刷新歷史
      setRefreshTrigger(prev => prev + 1);
    } catch (error: any) {
      console.error('上傳失敗:', error);
      alert('上傳失敗: ' + (error.response?.data?.detail || error.message));
    } finally {
      setUploading(false);
    }
  }, []);

  const handleTaskComplete = useCallback(() => {
    // 任務完成時重新載入歷史記錄
    setRefreshTrigger(prev => prev + 1);
  }, []);

  const handleCloseProgress = useCallback(() => {
    setCurrentTaskId(null);
    setRefreshTrigger(prev => prev + 1);
  }, []);

  const handleSelectHistoryTask = useCallback((taskId: string) => {
    setCurrentTaskId(taskId);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <Mic size={32} />
            <div>
              <h1>Whisper 語音轉文字</h1>
              <p className="subtitle">高品質中文語音辨識 • 支援語者分離</p>
            </div>
          </div>
          <ServiceStats />
        </div>
      </header>

      <main className="app-main">
        <div className="main-content">
          <div className="left-panel">
            <UploadSection 
              onUpload={handleUpload} 
              disabled={uploading}
            />

            {currentTaskId && (
              <TaskProgress
                taskId={currentTaskId}
                onClose={handleCloseProgress}
                onComplete={handleTaskComplete}
              />
            )}
          </div>

          <div className="right-panel">
            <TaskHistory
              onSelectTask={handleSelectHistoryTask}
              refreshTrigger={refreshTrigger}
            />
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Powered by <strong>Faster-Whisper</strong> • <strong>Pyannote Audio</strong>
        </p>
        <p className="api-docs">
          API 文檔：<a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
            http://localhost:8000/docs
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
