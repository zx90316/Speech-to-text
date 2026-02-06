/**
 * API 客戶端
 */
import axios from 'axios';
import type { Task, TaskCreateResponse, TaskHistory, ServiceStats } from './types';
import { clearVerifiedEmail } from './utils/emailStorage';

// API 基礎 URL
// 生產環境：設定 VITE_API_URL 環境變數指向後端（例如 http://192.168.1.100:8100/api）
// 開發環境：使用相對路徑 /api，透過 Vite proxy 轉發
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

// 添加 axios 響應攔截器，處理 403 錯誤
let hasShown403Alert = false; // 防止重複彈窗
let alertTimeout: ReturnType<typeof setTimeout> | null = null;

axios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 403 && !hasShown403Alert) {
      hasShown403Alert = true;

      // 伺服器記憶已被重置，清除本地驗證信息
      clearVerifiedEmail();

      // 清除之前的 timeout（如果有）
      if (alertTimeout) {
        clearTimeout(alertTimeout);
      }

      // 延遲彈窗和刷新，避免多次觸發
      alertTimeout = setTimeout(() => {
        const errorMessage = error.response?.data?.detail || '伺服器已重啟，驗證信息已失效';
        alert(`🔒 驗證已失效\n\n${errorMessage}\n\n頁面將自動刷新，請重新驗證您的電子郵件。`);

        // 刷新頁面以返回驗證界面
        window.location.reload();

        hasShown403Alert = false;
        alertTimeout = null;
      }, 100);
    }
    return Promise.reject(error);
  }
);

export const api = {
  /**
   * 發送郵件驗證碼
   */
  async sendVerificationEmail(email: string): Promise<{ success: boolean; message: string }> {
    const response = await axios.post<{ success: boolean; message: string }>(
      `${API_BASE_URL}/email/send-verification`,
      null,
      { params: { email } }
    );
    return response.data;
  },

  /**
   * 驗證郵件驗證碼
   */
  async verifyEmailCode(email: string, code: string): Promise<{ success: boolean; message: string; email: string }> {
    const response = await axios.post<{ success: boolean; message: string; email: string }>(
      `${API_BASE_URL}/email/verify-code`,
      null,
      { params: { email, code } }
    );
    return response.data;
  },

  /**
   * 提交轉錄任務
   */
  async createTask(
    email: string,
    file: File,
    enableDiarization: boolean = true,
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
  ): Promise<TaskCreateResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const params: any = {
      email,
      enable_diarization: enableDiarization
    };
    if (startTime !== undefined) params.start_time = startTime;
    if (endTime !== undefined) params.end_time = endTime;
    if (language) params.language = language;
    if (task) params.task = task;
    if (model) params.model = model;
    if (vadOnset !== undefined) params.vad_onset = vadOnset;
    if (vadOffset !== undefined) params.vad_offset = vadOffset;
    if (minSpeakers !== undefined) params.min_speakers = minSpeakers;
    if (maxSpeakers !== undefined) params.max_speakers = maxSpeakers;
    if (enableConfidenceScore !== undefined) params.enable_confidence_score = enableConfidenceScore;
    if (computeType !== undefined) params.compute_type = computeType;
    if (enableLlmCorrection !== undefined) params.enable_llm_correction = enableLlmCorrection;
    if (llmModel) params.llm_model = llmModel;
    // QWEN ASR 參數
    if (asrEngine) params.asr_engine = asrEngine;
    if (qwenModel) params.qwen_model = qwenModel;
    if (enableQwenTimestamps !== undefined) params.enable_qwen_timestamps = enableQwenTimestamps;

    const response = await axios.post<TaskCreateResponse>(
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
  async getTask(taskId: string): Promise<Task> {
    const response = await axios.get<Task>(`${API_BASE_URL}/tasks/${taskId}`);
    return response.data;
  },

  /**
   * 取消任務
   */
  async cancelTask(taskId: string): Promise<void> {
    await axios.delete(`${API_BASE_URL}/tasks/${taskId}`);
  },

  /**
   * 永久刪除任務（包含檔案）
   */
  async deleteTask(taskId: string): Promise<void> {
    await axios.delete(`${API_BASE_URL}/tasks/${taskId}`, {
      params: { permanent: true }
    });
  },

  /**
   * 查詢我的任務歷史（基於郵箱）
   */
  async getMyTasks(email: string, limit: number = 50): Promise<TaskHistory> {
    const response = await axios.get<TaskHistory>(`${API_BASE_URL}/my-tasks`, {
      params: { email, limit }
    });
    return response.data;
  },

  /**
   * 批量查詢任務（基於任務 ID 列表）
   */
  async getTasksBatch(taskIds: string[]): Promise<{ total: number; tasks: Task[] }> {
    const response = await axios.post<{ total: number; tasks: Task[] }>(
      `${API_BASE_URL}/tasks/batch`,
      taskIds
    );
    return response.data;
  },

  /**
   * 獲取服務統計
   */
  async getStats(): Promise<ServiceStats> {
    const response = await axios.get<ServiceStats>(`${API_BASE_URL}/stats`);
    return response.data;
  },

  /**
   * 建立 SSE 連接以接收進度更新
   * 注意：SSE 需要完整 URL，當使用遠端後端時需要特別處理
   */
  createProgressStream(taskId: string): EventSource {
    // 如果 API_BASE_URL 是相對路徑，使用當前 origin；否則直接使用設定的 URL
    const baseUrl = API_BASE_URL.startsWith('http')
      ? API_BASE_URL
      : `${window.location.origin}${API_BASE_URL}`;
    return new EventSource(`${baseUrl}/tasks/${taskId}/stream`);
  }
};

