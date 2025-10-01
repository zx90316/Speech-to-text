/**
 * 上傳區域組件
 */
import { useState, useRef } from 'react';
import { Upload, FileAudio, X } from 'lucide-react';
import { AudioPlayer } from './AudioPlayer';

interface UploadSectionProps {
  onUpload: (file: File, enableDiarization: boolean, startTime?: number, endTime?: number, language?: string, task?: string, model?: string) => void;
  disabled?: boolean;
}

export function UploadSection({ onUpload, disabled }: UploadSectionProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [enableDiarization, setEnableDiarization] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [startTime, setStartTime] = useState<number | undefined>(undefined);
  const [endTime, setEndTime] = useState<number | undefined>(undefined);
  const [language, setLanguage] = useState<string>('');
  const [taskType, setTaskType] = useState<string>('transcribe');
  const [model, setModel] = useState<string>('XA9/Belle-faster-whisper-large-v3-zh-punct');
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    if (selectedFile) {
      onUpload(
        selectedFile,
        enableDiarization,
        startTime,
        endTime,
        language || undefined,
        taskType,
        model
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
      
      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`}
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
        
        <Upload size={48} className="upload-icon" />
        
        {selectedFile ? (
          <div className="file-info">
            <div className="file-name">{selectedFile.name}</div>
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
            >
              <X size={20} />
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
        <AudioPlayer file={selectedFile} onTimeRangeChange={handleTimeRangeChange} />
      )}

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

        <div className="model-selector">
          <label htmlFor="model">Whisper 模型：</label>
          <select
            id="model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={disabled}
          >
            <option value="XA9/Belle-faster-whisper-large-v3-zh-punct">Belle Large V3 (推薦)</option>
            <option value="XA9/Belle-faster-whisper-large-v3-zh-punct-int8">Belle Large V3 INT8 (較快)</option>
          </select>
        </div>
      </div>

      <button
        className="submit-btn"
        onClick={handleSubmit}
        disabled={!selectedFile || disabled}
      >
        <Upload size={20} />
        開始轉錄
      </button>
    </div>
  );
}

