/**
 * 上傳區域組件
 */
import { useState, useRef, useEffect } from 'react';
import { Upload, FileAudio, X } from 'lucide-react';
import { AudioPlayer } from './AudioPlayer';
import { EmailVerification } from './EmailVerification';
import { getVerifiedEmail, saveVerifiedEmail, clearVerifiedEmail } from '../utils/emailStorage';

interface UploadSectionProps {
  onUpload: (
    email: string,
    file: File,
    enableDiarization: boolean,
    startTime?: number,
    endTime?: number,
    language?: string,
    task?: string,
    model?: string,
    vadOnset?: number,
    vadOffset?: number,
    minSpeakers?: number,
    maxSpeakers?: number,
    enableConfidenceScore?: boolean,
    computeType?: string,
    enableLlmCorrection?: boolean,
    llmModel?: string,
    // QWEN ASR 參數
    asrEngine?: string,
    qwenModel?: string,
    enableQwenTimestamps?: boolean,
  ) => void;
  onEmailVerified?: () => void;
  disabled?: boolean;
}

export function UploadSection({ onUpload, onEmailVerified, disabled }: UploadSectionProps) {
  const [verifiedEmail, setVerifiedEmail] = useState<string | null>(getVerifiedEmail());
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [enableDiarization, setEnableDiarization] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [startTime, setStartTime] = useState<number | undefined>(undefined);
  const [endTime, setEndTime] = useState<number | undefined>(undefined);
  const [language, setLanguage] = useState<string>('');
  const [taskType, setTaskType] = useState<string>('transcribe');
  const [model, setModel] = useState<string>('CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32');

  // 進階參數
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [vadOnset, setVadOnset] = useState<number>(0.5);
  const [vadOffset, setVadOffset] = useState<number>(0.363);
  const [minSpeakers, setMinSpeakers] = useState<number | undefined>(undefined);
  const [maxSpeakers, setMaxSpeakers] = useState<number | undefined>(undefined);
  const [enableConfidenceScore, setEnableConfidenceScore] = useState(true);
  const [computeType, setComputeType] = useState<string>('float16');

  // LLM 校對參數
  const [enableLlmCorrection, setEnableLlmCorrection] = useState(false);
  const [llmModel, setLlmModel] = useState<string>('gemma3:4b');

  // QWEN ASR 參數
  const [asrEngine, setAsrEngine] = useState<string>('whisper');
  const [qwenModel, setQwenModel] = useState<string>('Qwen/Qwen3-ASR-1.7B');
  const [enableQwenTimestamps, setEnableQwenTimestamps] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleEmailVerified = (email: string) => {
    saveVerifiedEmail(email);
    setVerifiedEmail(email);
    // 通知父組件驗證成功，以便 TaskHistory 可以解鎖
    if (onEmailVerified) {
      onEmailVerified();
    }
  };

  const handleTimeRangeChange = (start: number, end: number) => {
    setStartTime(start);
    setEndTime(end);
  };

  const handleFileSelect = (file: File) => {
    const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/flac', 'audio/mp3'];
    const allowedExtensions = ['.mp3', '.wav', '.m4a', '.flac'];

    const hasValidExtension = allowedExtensions.some(ext =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidExtension) {
      alert('不支援的檔案格式。請上傳 MP3, WAV, M4A 或 FLAC 檔案。');
      return;
    }

    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleSubmit = () => {
    if (selectedFile && verifiedEmail) {
      onUpload(
        verifiedEmail,
        selectedFile,
        enableDiarization,
        startTime,
        endTime,
        language || undefined,
        taskType,
        asrEngine === 'whisper' ? model : undefined,
        vadOnset,
        vadOffset,
        minSpeakers,
        maxSpeakers,
        enableConfidenceScore,
        computeType || undefined,
        enableLlmCorrection,
        llmModel,
        // QWEN ASR 參數
        asrEngine,
        qwenModel,
        enableQwenTimestamps,
      );
      setSelectedFile(null);
      setStartTime(undefined);
      setEndTime(undefined);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="upload-section">
      <h2 className="section-title">
        <FileAudio size={24} />
        語音轉文字
      </h2>

      {!verifiedEmail ? (
        <EmailVerification onVerified={handleEmailVerified} />
      ) : (
        <>
          <div className="verified-email-badge">
            ✓ 已驗證: {verifiedEmail}
            <button
              className="change-email-btn"
              onClick={() => {
                clearVerifiedEmail();
                setVerifiedEmail(null);
              }}
            >
              更改
            </button>
          </div>

          <div className={`upload-player-container ${selectedFile ? 'has-file' : 'no-file'}`}>
            <div
              className={`drop-zone ${isDragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''} ${selectedFile ? 'compact' : 'full'}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => !disabled && fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp3,.wav,.m4a,.flac,audio/*"
                onChange={handleFileInputChange}
                style={{ display: 'none' }}
                disabled={disabled}
              />

              <Upload size={selectedFile ? 32 : 48} className="upload-icon" />

              {selectedFile ? (
                <div className="file-info compact">
                  <div className="file-name" title={selectedFile.name}>{selectedFile.name}</div>
                  <div className="file-size">{formatFileSize(selectedFile.size)}</div>
                  <button
                    className="remove-file-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFile(null);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                    title="移除檔案"
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <div className="drop-zone-text">
                  <p className="primary-text">點擊或拖曳檔案至此</p>
                  <p className="secondary-text">支援 MP3, WAV, M4A, FLAC 格式</p>
                </div>
              )}
            </div>

            {selectedFile && (
              <div className="audio-player-wrapper">
                <AudioPlayer file={selectedFile} onTimeRangeChange={handleTimeRangeChange} />
              </div>
            )}
          </div>

          <div className="options">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={enableDiarization}
                onChange={(e) => setEnableDiarization(e.target.checked)}
                disabled={disabled}
              />
              <span>啟用語者分離（多人對話識別）</span>
            </label>

            <div className="language-selector">
              <label htmlFor="language">語言：</label>
              <select
                id="language"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={disabled}
              >
                <option value="">自動偵測</option>
                <option value="zh">中文</option>
                <option value="en">英文</option>
                <option value="ja">日文</option>
                <option value="ko">韓文</option>
                <option value="es">西班牙文</option>
                <option value="fr">法文</option>
                <option value="de">德文</option>
                <option value="ru">俄文</option>
              </select>
            </div>

            <div className="task-type-selector">
              <label>任務類型：</label>
              <div className="radio-group">
                <label className="radio-label">
                  <input
                    type="radio"
                    name="taskType"
                    value="transcribe"
                    checked={taskType === 'transcribe'}
                    onChange={(e) => setTaskType(e.target.value)}
                    disabled={disabled}
                  />
                  <span>轉錄</span>
                </label>
                <label className="radio-label">
                  <input
                    type="radio"
                    name="taskType"
                    value="translate"
                    checked={taskType === 'translate'}
                    onChange={(e) => setTaskType(e.target.value)}
                    disabled={disabled}
                  />
                  <span>翻譯成英文</span>
                </label>
              </div>
            </div>
          </div>

          {/* 進階參數區域 */}
          <div className="advanced-section">
            <button
              type="button"
              className="advanced-toggle"
              onClick={() => setShowAdvanced(!showAdvanced)}
              disabled={disabled}
            >
              {showAdvanced ? '▼' : '▶'} 進階參數
            </button>

            {showAdvanced && (
              <div className="advanced-options">
                {/* ASR 引擎選擇 */}
                <div className="asr-engine-selector">
                  <label htmlFor="asrEngine">ASR 引擎：</label>
                  <select
                    id="asrEngine"
                    value={asrEngine}
                    onChange={(e) => setAsrEngine(e.target.value)}
                    disabled={disabled}
                  >
                    <option value="whisper">Whisper（推薦）</option>
                    <option value="qwen">QWEN ASR</option>
                  </select>
                </div>

                {/* Whisper 模型選項（僅當選擇 Whisper 時顯示） */}
                {asrEngine === 'whisper' && (
                  <div className="model-selector">
                    <label htmlFor="model">Whisper 模型：</label>
                    <select
                      id="model"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      disabled={disabled}
                    >
                      <option value="CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32">CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32</option>
                      <option value="XA9/Belle-faster-whisper-large-v3-zh-punct">XA9/Belle-faster-whisper-large-v3-zh-punct</option>
                    </select>
                  </div>
                )}

                {/* QWEN 模型選項（僅當選擇 QWEN 時顯示） */}
                {asrEngine === 'qwen' && (
                  <>
                    <div className="model-selector">
                      <label htmlFor="qwenModel">QWEN 模型：</label>
                      <select
                        id="qwenModel"
                        value={qwenModel}
                        onChange={(e) => setQwenModel(e.target.value)}
                        disabled={disabled}
                      >
                        <option value="Qwen/Qwen3-ASR-1.7B">Qwen3-ASR-1.7B（效能最佳）</option>
                        <option value="Qwen/Qwen3-ASR-0.6B">Qwen3-ASR-0.6B（速度更快）</option>
                      </select>
                    </div>

                    <div className="advanced-row">
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={enableQwenTimestamps}
                          onChange={(e) => setEnableQwenTimestamps(e.target.checked)}
                          disabled={disabled}
                        />
                        <span>啟用詞級時間戳（需額外模型）</span>
                      </label>
                    </div>
                  </>
                )}

                <div className="compute-type-selector">
                  <label htmlFor="computeType">計算類型：</label>
                  <select
                    id="computeType"
                    value={computeType}
                    onChange={(e) => setComputeType(e.target.value)}
                    disabled={disabled}
                  >
                    <option value="float32">float32 (較慢，精度高)</option>
                    <option value="float16">float16 (較快，精度中等)</option>
                    <option value="int8">int8 (快，經度低，但部分 GPU 不支持)</option>
                    <option value="default">默認</option>
                  </select>
                </div>
                <div className="advanced-row">

                  <div className="param-group">
                    <label htmlFor="vadOnset">
                      VAD 檢測敏感度：
                      <span className="param-value">{vadOnset.toFixed(2)}</span>
                    </label>
                    <input
                      type="range"
                      id="vadOnset"
                      min="0"
                      max="1"
                      step="0.01"
                      value={vadOnset}
                      onChange={(e) => setVadOnset(parseFloat(e.target.value))}
                      disabled={disabled}
                    />
                    <span className="param-hint">較低值檢測更多語音片段</span>
                  </div>

                  <div className="param-group">
                    <label htmlFor="vadOffset">
                      VAD 結束閾值：
                      <span className="param-value">{vadOffset.toFixed(3)}</span>
                    </label>
                    <input
                      type="range"
                      id="vadOffset"
                      min="0"
                      max="1"
                      step="0.001"
                      value={vadOffset}
                      onChange={(e) => setVadOffset(parseFloat(e.target.value))}
                      disabled={disabled}
                    />
                    <span className="param-hint">控制語音片段結束判定</span>
                  </div>
                </div>

                <div className="advanced-row">
                  <div className="param-group">
                    <label htmlFor="minSpeakers">最小語者數：</label>
                    <input
                      type="number"
                      id="minSpeakers"
                      min="1"
                      max="10"
                      value={minSpeakers || ''}
                      onChange={(e) => setMinSpeakers(e.target.value ? parseInt(e.target.value) : undefined)}
                      placeholder="自動"
                      disabled={disabled || !enableDiarization}
                    />
                  </div>

                  <div className="param-group">
                    <label htmlFor="maxSpeakers">最大語者數：</label>
                    <input
                      type="number"
                      id="maxSpeakers"
                      min="1"
                      max="10"
                      value={maxSpeakers || ''}
                      onChange={(e) => setMaxSpeakers(e.target.value ? parseInt(e.target.value) : undefined)}
                      placeholder="自動"
                      disabled={disabled || !enableDiarization}
                    />
                  </div>
                </div>

                <div className="advanced-row">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={enableConfidenceScore}
                      onChange={(e) => setEnableConfidenceScore(e.target.checked)}
                      disabled={disabled}
                    />
                    <span>啟用信心分數輸出</span>
                  </label>
                </div>

                <div className="advanced-row">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={enableLlmCorrection}
                      onChange={(e) => setEnableLlmCorrection(e.target.checked)}
                      disabled={disabled}
                    />
                    <span>啟用 LLM 文本校對</span>
                  </label>
                </div>

                {enableLlmCorrection && (
                  <div className="advanced-row">
                    <div className="param-group">
                      <label htmlFor="llmModel">LLM 模型：</label>
                      <select
                        id="llmModel"
                        value={llmModel}
                        onChange={(e) => setLlmModel(e.target.value)}
                        disabled={disabled}
                      >
                        <option value="gemma3:4b">gemma3:4b (推薦)</option>
                        <option value="qwen3:4b">qwen3:4b</option>
                        <option value="gpt-oss:20b">gpt-oss:20b</option>
                      </select>
                      <span className="param-hint">使用 LLM 對轉錄結果進行校對改正</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <button
            className="submit-btn"
            onClick={handleSubmit}
            disabled={!selectedFile || disabled}
          >
            <Upload size={20} />
            開始轉錄
          </button>
        </>
      )}
    </div>
  );
}

