/**
 * Backend V2 簡化版上傳頁面
 * 用於測試 Qwen ASR + 語者分離後端
 */
import { useState, useEffect, useRef } from 'react';
import { apiV2, type TaskV2, type ModelsResponse, type HealthResponse } from '../apiV2';
import './UploadV2.css';

export default function UploadV2() {
    // 狀態
    const [file, setFile] = useState<File | null>(null);
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [models, setModels] = useState<ModelsResponse | null>(null);
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [enableDiarization, setEnableDiarization] = useState(true);
    const [enableTimestamps, setEnableTimestamps] = useState(false);
    const [language, setLanguage] = useState<string>('');
    const [minSpeakers, setMinSpeakers] = useState<number | undefined>();
    const [maxSpeakers, setMaxSpeakers] = useState<number | undefined>();
    const [email, setEmail] = useState<string>('');

    const [isUploading, setIsUploading] = useState(false);
    const [task, setTask] = useState<TaskV2 | null>(null);
    const [error, setError] = useState<string>('');

    const eventSourceRef = useRef<EventSource | null>(null);

    // 初始化：取得健康狀態與模型列表
    useEffect(() => {
        const init = async () => {
            try {
                const [healthRes, modelsRes] = await Promise.all([
                    apiV2.health(),
                    apiV2.getModels()
                ]);
                setHealth(healthRes);
                setModels(modelsRes);
                setSelectedModel(modelsRes.default_model);
            } catch (err) {
                setError('無法連接後端服務，請確認 backend 已啟動');
            }
        };
        init();
    }, []);

    // 清理 SSE 連接
    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, []);

    // 處理檔案選擇
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
            setError('');
            setTask(null);
        }
    };

    // 處理上傳
    const handleUpload = async () => {
        if (!file) {
            setError('請先選擇檔案');
            return;
        }

        setIsUploading(true);
        setError('');
        setTask(null);

        try {
            const result = await apiV2.createTask(file, {
                email: email || undefined,
                enableDiarization,
                enableTimestamps,
                language: language || undefined,
                model: selectedModel,
                minSpeakers,
                maxSpeakers,
            });

            setTask(result);

            // 建立 SSE 連接
            const es = apiV2.createProgressStream(result.task_id);
            eventSourceRef.current = es;

            es.addEventListener('progress', (event) => {
                const data = JSON.parse(event.data);
                setTask(prev => prev ? { ...prev, ...data } : null);
            });

            es.addEventListener('complete', (event) => {
                const data = JSON.parse(event.data);
                setTask(data);
                es.close();
                setIsUploading(false);
            });

            es.addEventListener('error', () => {
                es.close();
                setIsUploading(false);
            });

        } catch (err: any) {
            setError(err.response?.data?.detail || err.message || '上傳失敗');
            setIsUploading(false);
        }
    };

    // 取消任務
    const handleCancel = async () => {
        if (!task) return;

        try {
            await apiV2.cancelTask(task.task_id);
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
            setTask(prev => prev ? { ...prev, status: 'canceled' } : null);
            setIsUploading(false);
        } catch (err: any) {
            setError(err.response?.data?.detail || '取消失敗');
        }
    };

    return (
        <div className="upload-v2">
            <header className="header">
                <h1>🎙️ Speech-to-Text V2</h1>
                <p className="subtitle">Qwen ASR + 語者分離</p>

                {health && (
                    <div className="status-bar">
                        <span className={`status-item ${health.gpu_available ? 'ok' : 'warn'}`}>
                            GPU: {health.gpu_available ? '✅' : '❌'}
                        </span>
                        <span className={`status-item ${health.qwen_available ? 'ok' : 'warn'}`}>
                            Qwen ASR: {health.qwen_available ? '✅' : '❌'}
                        </span>
                        <span className="status-item">v{health.version}</span>
                    </div>
                )}
            </header>

            <main className="main-content">
                {error && (
                    <div className="error-message">
                        ❌ {error}
                    </div>
                )}

                {/* 上傳區 */}
                <section className="upload-section">
                    <div className="file-input-wrapper">
                        <input
                            type="file"
                            accept=".mp3,.wav,.m4a,.flac,.ogg,.wma,.aac,.webm,.mp4"
                            onChange={handleFileChange}
                            disabled={isUploading}
                            id="file-input"
                        />
                        <label htmlFor="file-input" className="file-input-label">
                            {file ? file.name : '📁 選擇音訊檔案'}
                        </label>
                    </div>

                    {/* 郵件通知（可選） */}
                    <div className="option-group">
                        <label>
                            📧 完成通知:
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="輸入郵件地址（可選）"
                                disabled={isUploading}
                                className="email-input"
                            />
                        </label>
                    </div>

                    {/* 選項 */}
                    <div className="options">
                        <div className="option-group">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={enableDiarization}
                                    onChange={(e) => setEnableDiarization(e.target.checked)}
                                    disabled={isUploading}
                                />
                                語者分離
                            </label>

                            <label>
                                <input
                                    type="checkbox"
                                    checked={enableTimestamps}
                                    onChange={(e) => setEnableTimestamps(e.target.checked)}
                                    disabled={isUploading}
                                />
                                時間戳
                            </label>
                        </div>

                        <div className="option-group">
                            <label>
                                語言:
                                <select
                                    value={language}
                                    onChange={(e) => setLanguage(e.target.value)}
                                    disabled={isUploading}
                                >
                                    <option value="">自動偵測</option>
                                    <option value="zh">中文</option>
                                    <option value="en">英文</option>
                                    <option value="ja">日文</option>
                                    <option value="ko">韓文</option>
                                </select>
                            </label>

                            <label>
                                模型:
                                <select
                                    value={selectedModel}
                                    onChange={(e) => setSelectedModel(e.target.value)}
                                    disabled={isUploading}
                                >
                                    {models && Object.entries(models.asr_models).map(([model, desc]) => (
                                        <option key={model} value={model}>{model.split('/')[1]}</option>
                                    ))}
                                </select>
                            </label>
                        </div>

                        {enableDiarization && (
                            <div className="option-group">
                                <label>
                                    最少語者:
                                    <input
                                        type="number"
                                        min="1"
                                        max="20"
                                        value={minSpeakers || ''}
                                        onChange={(e) => setMinSpeakers(e.target.value ? parseInt(e.target.value) : undefined)}
                                        disabled={isUploading}
                                        placeholder="自動"
                                    />
                                </label>

                                <label>
                                    最多語者:
                                    <input
                                        type="number"
                                        min="1"
                                        max="20"
                                        value={maxSpeakers || ''}
                                        onChange={(e) => setMaxSpeakers(e.target.value ? parseInt(e.target.value) : undefined)}
                                        disabled={isUploading}
                                        placeholder="自動"
                                    />
                                </label>
                            </div>
                        )}
                    </div>

                    <button
                        className="upload-button"
                        onClick={handleUpload}
                        disabled={!file || isUploading}
                    >
                        {isUploading ? '處理中...' : '🚀 開始轉錄'}
                    </button>

                    {isUploading && task && (
                        <button className="cancel-button" onClick={handleCancel}>
                            ❌ 取消
                        </button>
                    )}
                </section>

                {/* 進度與結果 */}
                {task && (
                    <section className="result-section">
                        <div className="progress-info">
                            <div className="progress-bar">
                                <div
                                    className="progress-fill"
                                    style={{ width: `${task.progress}%` }}
                                />
                            </div>
                            <div className="progress-text">
                                <span>{task.progress.toFixed(0)}%</span>
                                <span>{task.current_stage || task.status}</span>
                            </div>
                        </div>

                        {task.status === 'completed' && (
                            <div className="transcript">
                                <h3>📝 轉錄結果</h3>
                                {task.language && (
                                    <p className="language-info">🌐 偵測語言: {task.language}</p>
                                )}
                                {task.has_diarization && <p className="diarization-info">👥 已啟用語者分離</p>}

                                <div className="transcript-content">
                                    {task.segments?.map((seg, idx) => (
                                        <div key={idx} className="segment">
                                            <span className="time">
                                                [{seg.start.toFixed(2)}s → {seg.end.toFixed(2)}s]
                                            </span>
                                            {seg.speaker && <span className="speaker">{seg.speaker}:</span>}
                                            <span className="text">{seg.text}</span>
                                        </div>
                                    )) || <p>{task.text}</p>}
                                </div>
                            </div>
                        )}

                        {task.status === 'failed' && (
                            <div className="error-result">
                                ❌ 錯誤: {task.error_message}
                            </div>
                        )}

                        {task.status === 'canceled' && (
                            <div className="canceled-result">
                                ⚠️ 任務已取消
                            </div>
                        )}
                    </section>
                )}
            </main>
        </div>
    );
}
