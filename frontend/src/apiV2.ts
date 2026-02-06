/**
 * API 客戶端
 * Qwen ASR + 語者分離後端
 */
import axios from 'axios';

// API 基礎 URL
// 生產環境：設定 VITE_API_URL 環境變數
// 開發環境：使用 /api 透過 vite proxy 轉發
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export interface TaskV2 {
    task_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'canceled';
    progress: number;
    current_stage: string | null;
    error_message: string | null;
    created_at: string;
    completed_at: string | null;
    filename: string | null;
    text: string | null;
    language: string | null;
    segments: Array<{
        start: number;
        end: number;
        text: string;
        speaker?: string;
        words?: Array<{
            word: string;
            start: number;
            end: number;
            probability: number;
        }>;
    }> | null;
    has_diarization: boolean;
}

export interface HealthResponse {
    status: string;
    version: string;
    gpu_available: boolean;
    qwen_available: boolean;
}

export interface ModelsResponse {
    asr_models: Record<string, string>;
    default_model: string;
}

export interface EmailResponse {
    success: boolean;
    message: string;
    email?: string;
}

export const apiV2 = {
    /**
     * 發送郵件驗證碼
     */
    async sendVerificationEmail(email: string): Promise<EmailResponse> {
        const response = await axios.post<EmailResponse>(
            `${API_BASE_URL}/email/send-verification`,
            null,
            { params: { email } }
        );
        return response.data;
    },

    /**
     * 驗證郵件驗證碼
     */
    async verifyEmailCode(email: string, code: string): Promise<EmailResponse> {
        const response = await axios.post<EmailResponse>(
            `${API_BASE_URL}/email/verify-code`,
            null,
            { params: { email, code } }
        );
        return response.data;
    },

    /**
     * 健康檢查
     */
    async health(): Promise<HealthResponse> {
        const response = await axios.get<HealthResponse>(`${API_BASE_URL}/health`);
        return response.data;
    },

    /**
     * 取得可用模型列表
     */
    async getModels(): Promise<ModelsResponse> {
        const response = await axios.get<ModelsResponse>(`${API_BASE_URL}/models`);
        return response.data;
    },

    /**
     * 提交轉錄任務
     */
    async createTask(
        file: File,
        options: {
            email?: string;
            enableDiarization?: boolean;
            enableTimestamps?: boolean;
            language?: string;
            model?: string;
            minSpeakers?: number;
            maxSpeakers?: number;
        } = {}
    ): Promise<TaskV2> {
        const formData = new FormData();
        formData.append('file', file);

        const params: Record<string, any> = {};
        if (options.email) params.email = options.email;
        if (options.enableDiarization !== undefined) params.enable_diarization = options.enableDiarization;
        if (options.enableTimestamps !== undefined) params.enable_timestamps = options.enableTimestamps;
        if (options.language) params.language = options.language;
        if (options.model) params.model = options.model;
        if (options.minSpeakers !== undefined) params.min_speakers = options.minSpeakers;
        if (options.maxSpeakers !== undefined) params.max_speakers = options.maxSpeakers;

        const response = await axios.post<TaskV2>(
            `${API_BASE_URL}/tasks`,
            formData,
            {
                params,
                headers: { 'Content-Type': 'multipart/form-data' }
            }
        );

        return response.data;
    },

    /**
     * 查詢任務狀態
     */
    async getTask(taskId: string): Promise<TaskV2> {
        const response = await axios.get<TaskV2>(`${API_BASE_URL}/tasks/${taskId}`);
        return response.data;
    },

    /**
     * 取消任務
     */
    async cancelTask(taskId: string): Promise<void> {
        await axios.delete(`${API_BASE_URL}/tasks/${taskId}`);
    },

    /**
     * 取得任務列表
     */
    async getTasks(limit: number = 50): Promise<{ total: number; tasks: TaskV2[] }> {
        const response = await axios.get<{ total: number; tasks: TaskV2[] }>(
            `${API_BASE_URL}/tasks`,
            { params: { limit } }
        );
        return response.data;
    },

    /**
     * 建立 SSE 連接以接收進度更新
     */
    createProgressStream(taskId: string): EventSource {
        // 如果 API_BASE_URL 是相對路徑，使用當前 origin；否則直接使用設定的 URL
        const baseUrl = API_BASE_URL.startsWith('http')
            ? API_BASE_URL
            : `${window.location.origin}${API_BASE_URL}`;
        return new EventSource(`${baseUrl}/tasks/${taskId}/stream`);
    }
};
