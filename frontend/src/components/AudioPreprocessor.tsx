/**
 * 音訊預處理組件 - 現代化左右佈局
 */
import React, { useState } from 'react';
import { Upload, Settings, Download, Trash2, FileAudio } from 'lucide-react';
import { api } from '../api';
import { WaveformPlayer } from './WaveformPlayer';
import { NativeAudioPlayer } from './NativeAudioPlayer';
import { PreprocessHistory } from './PreprocessHistory';
import { addPreprocessTaskId } from '../utils/taskStorage';
import './AudioPreprocessor.css';

interface PreprocessConfig {
  enable_denoise: boolean;
  denoise_strength: number;
  enable_normalize: boolean;
  normalize_type: 'peak' | 'lufs';
  target_level: number;
  enable_silence_removal: boolean;
  silence_threshold: number;
  min_silence_duration: number;
  enable_vocal_enhancement: boolean;
  enhancement_strength: number;
  enable_echo_removal: boolean;
  enable_eq: boolean;
  eq_low_gain: number;
  eq_mid_gain: number;
  eq_high_gain: number;
  enable_speed_change: boolean;
  speed_factor: number;
  enable_pitch_shift: boolean;
  pitch_semitones: number;
  enable_resample: boolean;
  target_sample_rate: number;
  enable_mono: boolean;
  enable_compression: boolean;
  compression_ratio: number;
  compression_threshold: number;
}

interface AudioPreprocessorProps {
  onPreprocessComplete?: (preprocessId: string, processedFile: File) => void;
}

export const AudioPreprocessor: React.FC<AudioPreprocessorProps> = ({ onPreprocessComplete }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [preprocessId, setPreprocessId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<string>(''); // 追蹤任務狀態
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [progress, setProgress] = useState<number>(0);
  const [currentStage, setCurrentStage] = useState<string>('');
  const [historyRefreshTrigger, setHistoryRefreshTrigger] = useState(0);
  const [isLargeFile, setIsLargeFile] = useState(false); // 標記是否為大文件
  const [audioDuration, setAudioDuration] = useState<number>(0); // 音頻時長

  const [config, setConfig] = useState<PreprocessConfig>({
    enable_denoise: false,
    denoise_strength: 0.5,
    enable_normalize: false,
    normalize_type: 'peak',
    target_level: -3.0,
    enable_silence_removal: false,
    silence_threshold: -50.0,
    min_silence_duration: 1.0,
    enable_vocal_enhancement: false,
    enhancement_strength: 0.5,
    enable_echo_removal: false,
    enable_eq: false,
    eq_low_gain: 0.0,
    eq_mid_gain: 0.0,
    eq_high_gain: 0.0,
    enable_speed_change: false,
    speed_factor: 1.0,
    enable_pitch_shift: false,
    pitch_semitones: 0,
    enable_resample: false,
    target_sample_rate: 16000,
    enable_mono: true,
    enable_compression: false,
    compression_ratio: 4.0,
    compression_threshold: -20.0,
  });

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setPreprocessId(null);
    setError(null);

    // 檢查文件大小 (超過 10MB 視為大文件)
    const fileSizeMB = selectedFile.size / (1024 * 1024);
    setIsLargeFile(fileSizeMB > 10);

    // 獲取音頻時長
    const audioUrl = URL.createObjectURL(selectedFile);
    const audio = new Audio(audioUrl);
    audio.addEventListener('loadedmetadata', () => {
      setAudioDuration(audio.duration);
      URL.revokeObjectURL(audioUrl);

      // 如果超過 5 分鐘也視為大文件
      if (audio.duration > 300) {
        setIsLargeFile(true);
      }
    });
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handlePreprocess = async () => {
    if (!file) return;

    setIsProcessing(true);
    setError(null);
    setProgress(0);
    setCurrentStage('提交任務中...');

    try {
      // 提交預處理任務
      const result = await api.preprocessAudio(file, config);
      const taskId = result.preprocess_id;

      // 儲存任務 ID 到 localStorage
      addPreprocessTaskId(taskId);

      // 連接 SSE 進度推送
      const eventSource = api.connectPreprocessStream(taskId);

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
          setError(data.error);
          eventSource.close();
          setIsProcessing(false);
          return;
        }

        setProgress(data.progress || 0);
        setCurrentStage(data.current_stage || '');
        setTaskStatus(data.status);

        if (data.status === 'completed') {
          setPreprocessId(taskId);
          setIsProcessing(false);
          eventSource.close();

          // 刷新歷史列表
          setHistoryRefreshTrigger(prev => prev + 1);

          // 處理完成回調
          if (onPreprocessComplete) {
            const processedUrl = api.downloadPreprocessedAudio(taskId, 'processed');
            fetch(processedUrl)
              .then(res => res.blob())
              .then(blob => {
                const processedFile = new File([blob], file.name, { type: 'audio/wav' });
                onPreprocessComplete(taskId, processedFile);
              });
          }
        } else if (data.status === 'failed') {
          setError(data.error_message || '預處理失敗');
          setIsProcessing(false);
          eventSource.close();
        } else if (data.status === 'canceled') {
          setError('任務已取消');
          setIsProcessing(false);
          eventSource.close();
        }
      };

      eventSource.onerror = () => {
        setError('連接中斷，請重試');
        setIsProcessing(false);
        eventSource.close();
      };

    } catch (err: any) {
      setError(err.response?.data?.detail || '提交任務失敗');
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreprocessId(null);
    setError(null);
  };

  const updateConfig = (updates: Partial<PreprocessConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
  };

  // 預設方案
  const applyPreset = (preset: string) => {
    const presets: Record<string, Partial<PreprocessConfig>> = {
      clean: {
        enable_denoise: true,
        denoise_strength: 0.7,
        enable_normalize: true,
        normalize_type: 'lufs',
        target_level: -16.0,
        enable_vocal_enhancement: true,
        enhancement_strength: 0.6,
        enable_compression: true,
        compression_ratio: 6.0,
        enable_mono: true,
      },
      outdoor: {
        enable_denoise: true,
        denoise_strength: 0.8,
        enable_eq: true,
        eq_low_gain: -6,
        eq_mid_gain: 3,
        eq_high_gain: -3,
        enable_vocal_enhancement: true,
        enhancement_strength: 0.7,
        enable_mono: true,
      },
      meeting: {
        enable_denoise: true,
        denoise_strength: 0.5,
        enable_echo_removal: true,
        enable_normalize: true,
        normalize_type: 'lufs',
        target_level: -16.0,
        enable_vocal_enhancement: true,
        enhancement_strength: 0.5,
        silence_threshold: -50.0,
        min_silence_duration: 1.0,
        enable_mono: true,
      },
    };

    if (presets[preset]) {
      updateConfig(presets[preset]);
    }
  };

  return (
    <div className="audio-preprocessor-modern">
      {/* 左側：操作面板 */}
      <div className="preprocessor-left">
        <div className="section-header">
          <Settings size={24} />
          <h2>音訊預處理</h2>
        </div>

        {/* 檔案上傳 */}
        <div
          className={`upload-zone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {file ? (
            <div className="file-info">
              <FileAudio size={32} />
              <div className="file-details">
                <p className="file-name">{file.name}</p>
                <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
              <button className="btn-remove" onClick={handleReset} title="移除檔案">
                <Trash2 size={18} />
              </button>
            </div>
          ) : (
            <>
              <Upload size={48} />
              <p>拖放音訊檔案至此</p>
              <p className="upload-hint">或點擊選擇檔案</p>
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => {
                  const selectedFile = e.target.files?.[0];
                  if (selectedFile) handleFileSelect(selectedFile);
                }}
                className="file-input"
              />
            </>
          )}
        </div>

        {/* 預設方案 */}
        <div className="preset-section">
          <h3>快速預設</h3>
          <div className="preset-buttons">
            <button onClick={() => applyPreset('clean')} className="preset-btn">
              通話清理
            </button>
            <button onClick={() => applyPreset('outdoor')} className="preset-btn">
              戶外錄音
            </button>
            <button onClick={() => applyPreset('meeting')} className="preset-btn">
              會議錄音
            </button>
          </div>
        </div>

        {/* 參數設定 */}
        <div className="params-section">
          <h3>處理參數</h3>

          <div className="params-grid">
            {/* 降噪 */}
            <div className="param-group">
            <label className="param-toggle">
              <input
                type="checkbox"
                checked={config.enable_denoise}
                onChange={(e) => updateConfig({ enable_denoise: e.target.checked })}
              />
              <span>降噪處理</span>
            </label>
            {config.enable_denoise && (
              <div className="param-control">
                <label>
                  強度: {(config.denoise_strength * 100).toFixed(0)}%
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.denoise_strength}
                    onChange={(e) => updateConfig({ denoise_strength: parseFloat(e.target.value) })}
                  />
                </label>
              </div>
            )}
          </div>

          {/* 音量正規化 */}
          <div className="param-group">
            <label className="param-toggle">
              <input
                type="checkbox"
                checked={config.enable_normalize}
                onChange={(e) => updateConfig({ enable_normalize: e.target.checked })}
              />
              <span>音量正規化</span>
            </label>
            {config.enable_normalize && (
              <div className="param-control">
                <label>
                  類型:
                  <select
                    value={config.normalize_type}
                    onChange={(e) => updateConfig({ normalize_type: e.target.value as 'peak' | 'lufs' })}
                  >
                    <option value="peak">峰值</option>
                    <option value="lufs">響度 (LUFS)</option>
                  </select>
                </label>
              </div>
            )}
          </div>

          {/* 人聲增強 */}
          <div className="param-group">
            <label className="param-toggle">
              <input
                type="checkbox"
                checked={config.enable_vocal_enhancement}
                onChange={(e) => updateConfig({ enable_vocal_enhancement: e.target.checked })}
              />
              <span>人聲增強</span>
            </label>
            {config.enable_vocal_enhancement && (
              <div className="param-control">
                <label>
                  強度: {(config.enhancement_strength * 100).toFixed(0)}%
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.enhancement_strength}
                    onChange={(e) => updateConfig({ enhancement_strength: parseFloat(e.target.value) })}
                  />
                </label>
              </div>
            )}
          </div>

          {/* 迴聲消除 */}
          <div className="param-group">
            <label className="param-toggle">
              <input
                type="checkbox"
                checked={config.enable_echo_removal}
                onChange={(e) => updateConfig({ enable_echo_removal: e.target.checked })}
              />
              <span>迴聲消除</span>
            </label>
          </div>

          {/* 靜音移除 */}
          <div className="param-group">
            <label className="param-toggle">
              <input
                type="checkbox"
                checked={config.enable_silence_removal}
                onChange={(e) => updateConfig({ enable_silence_removal: e.target.checked })}
              />
              <span>靜音移除</span>
            </label>
          </div>

          {/* 動態壓縮 */}
          <div className="param-group">
            <label className="param-toggle">
              <input
                type="checkbox"
                checked={config.enable_compression}
                onChange={(e) => updateConfig({ enable_compression: e.target.checked })}
              />
              <span>動態壓縮</span>
            </label>
          </div>

          {/* 單聲道轉換 */}
          <div className="param-group">
            <label className="param-toggle">
              <input
                type="checkbox"
                checked={config.enable_mono}
                onChange={(e) => updateConfig({ enable_mono: e.target.checked })}
              />
              <span>轉單聲道</span>
            </label>
          </div>
          </div>
        </div>

        {/* 操作按鈕 */}
        <div className="action-buttons">
          <button
            onClick={handlePreprocess}
            disabled={!file || isProcessing}
            className="btn-process"
          >
            {isProcessing ? `處理中... ${Math.round(progress)}%` : '開始預處理'}
          </button>

          {isProcessing && preprocessId && (
            <button
              onClick={async () => {
                if (confirm('確定要取消預處理任務嗎？')) {
                  try {
                    await api.deletePreprocess(preprocessId, false);
                    setIsProcessing(false);
                    setError('任務已取消');
                    setPreprocessId(null);
                  } catch (err) {
                    console.error('取消任務失敗:', err);
                  }
                }
              }}
              className="btn-cancel"
            >
              取消任務
            </button>
          )}
        </div>

        {isProcessing && (
          <div className="progress-container">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            </div>
            <div className="progress-text">{currentStage}</div>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}
      </div>

      {/* 右側：預覽面板 */}
      <div className="preprocessor-right">
        <div className="section-header">
          <FileAudio size={24} />
          <h2>音訊預覽</h2>
        </div>

        {!file && !preprocessId && (
          <div className="preview-placeholder">
            <FileAudio size={64} />
            <p>上傳音訊檔案以查看預覽</p>
          </div>
        )}

        {file && !preprocessId && (
          <div className="preview-content">
            <h3>原始音訊</h3>
            {isLargeFile ? (
              <NativeAudioPlayer
                audioUrl={URL.createObjectURL(file)}
                title={file.name}
              />
            ) : (
              <WaveformPlayer
                audioUrl={URL.createObjectURL(file)}
                title={file.name}
                color="#6366f1"
              />
            )}
            <p className="preview-hint">點擊「開始預處理」以查看處理後效果</p>
          </div>
        )}

        {preprocessId && taskStatus === 'completed' && (
          <div className="preview-content">
            <div className="comparison-section">
              {file && (
                <div className="comparison-item">
                  <h3>原始音訊</h3>
                  {isLargeFile ? (
                    <NativeAudioPlayer
                      audioUrl={URL.createObjectURL(file)}
                      title="原始"
                    />
                  ) : (
                    <WaveformPlayer
                      audioUrl={URL.createObjectURL(file)}
                      title="原始"
                      color="#94a3b8"
                    />
                  )}
                </div>
              )}

              {!file && (
                <div className="comparison-item">
                  <h3>原始音訊</h3>
                  <NativeAudioPlayer
                    audioUrl={api.downloadPreprocessedAudio(preprocessId, 'original')}
                    title="原始"
                  />
                </div>
              )}

              <div className="comparison-item">
                <h3>處理後音訊</h3>
                <NativeAudioPlayer
                  audioUrl={api.downloadPreprocessedAudio(preprocessId, 'processed')}
                  title="處理後"
                />
              </div>
            </div>

            <div className="download-section">
              <a
                href={api.downloadPreprocessedAudio(preprocessId, 'processed')}
                download
                className="btn-download"
              >
                <Download size={20} />
                下載處理後音訊
              </a>
            </div>
          </div>
        )}

        {preprocessId && (taskStatus === 'processing' || isProcessing) && (
          <div className="preview-content">
            <div className="processing-status">
              <h3>處理中</h3>
              <div className="progress-container">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                </div>
                <div className="progress-text">
                  {Math.round(progress)}% - {currentStage}
                </div>
              </div>
            </div>
          </div>
        )}

        {preprocessId && (taskStatus === 'failed' || taskStatus === 'canceled') && !isProcessing && (
          <div className="preview-content">
            <div className="error-state">
              <h3>任務{taskStatus === 'failed' ? '失敗' : '已取消'}</h3>
              {error && <p className="error-message">{error}</p>}
            </div>
          </div>
        )}
      </div>

      {/* 歷史記錄區塊 - 跨越整個佈局 */}
      <div className="history-section-wrapper">
        <PreprocessHistory
          onSelectTask={(selectedPreprocessId) => {
            // 選擇歷史任務，加載其結果
            setPreprocessId(selectedPreprocessId);
            setFile(null); // 清除當前文件

            api.getPreprocessTask(selectedPreprocessId).then(task => {
              setTaskStatus(task.status);
              setProgress(task.progress || 0);
              setCurrentStage(task.current_stage || '');

              if (task.status === 'processing') {
                // 如果是處理中的任務，連接 SSE 監聽進度
                setIsProcessing(true);
                const eventSource = api.connectPreprocessStream(selectedPreprocessId);

                eventSource.onmessage = (event) => {
                  const data = JSON.parse(event.data);

                  if (data.error) {
                    setError(data.error);
                    eventSource.close();
                    setIsProcessing(false);
                    return;
                  }

                  setProgress(data.progress || 0);
                  setCurrentStage(data.current_stage || '');
                  setTaskStatus(data.status);

                  if (data.status === 'completed') {
                    setIsProcessing(false);
                    eventSource.close();
                    setHistoryRefreshTrigger(prev => prev + 1);
                  } else if (data.status === 'failed') {
                    setError(data.error_message || '預處理失敗');
                    setIsProcessing(false);
                    eventSource.close();
                  } else if (data.status === 'canceled') {
                    setError('任務已取消');
                    setIsProcessing(false);
                    eventSource.close();
                  }
                };

                eventSource.onerror = () => {
                  setError('連接中斷');
                  setIsProcessing(false);
                  eventSource.close();
                };
              } else if (task.status === 'completed') {
                // 完成的任務，顯示預覽
                setIsProcessing(false);
              } else if (task.status === 'failed' || task.status === 'canceled') {
                // 失敗或取消的任務
                setIsProcessing(false);
                setError(task.error_message || `任務${task.status === 'failed' ? '失敗' : '已取消'}`);
              }
            }).catch(err => {
              console.error('獲取任務失敗:', err);
              setError('無法獲取任務資訊');
            });
          }}
          refreshTrigger={historyRefreshTrigger}
        />
      </div>
    </div>
  );
};
