/**
 * API 客戶端
 */
import axios from 'axios';
import type { Task, TaskCreateResponse, TaskHistory, ServiceStats } from './types';

const API_BASE_URL = '/api';

export const api = {
  /**
   * 提交轉錄任務
   */
  async createTask(
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
    computeType?: string
  ): Promise<TaskCreateResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const params: any = { enable_diarization: enableDiarization };
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
   * 下載結果
   */
  downloadResult(taskId: string, fileType: 'transcript' | 'raw' | 'confidence_html' = 'transcript'): string {
    return `${API_BASE_URL}/tasks/${taskId}/download?file_type=${fileType}`;
  },

  /**
   * 查詢我的任務歷史（基於 IP）
   */
  async getMyTasks(limit: number = 50): Promise<TaskHistory> {
    const response = await axios.get<TaskHistory>(`${API_BASE_URL}/my-tasks`, {
      params: { limit }
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
   */
  createProgressStream(taskId: string): EventSource {
    return new EventSource(`${API_BASE_URL}/tasks/${taskId}/stream`);
  },

  /**
   * 音訊預處理
   */
  /**
   * 提交音訊預處理任務（異步）
   */
  async preprocessAudio(
    file: File,
    config: any
  ): Promise<{
    preprocess_id: string;
    status: string;
    message: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(
      `${API_BASE_URL}/preprocess`,
      formData,
      {
        params: {
          config: JSON.stringify(config)
        },
        headers: { 'Content-Type': 'multipart/form-data' }
      }
    );

    return response.data;
  },

  /**
   * 查詢預處理任務狀態
   */
  async getPreprocessTask(preprocessId: string): Promise<any> {
    const response = await axios.get(`${API_BASE_URL}/preprocess/${preprocessId}`);
    return response.data;
  },

  /**
   * 連接預處理任務進度 SSE 串流
   */
  connectPreprocessStream(preprocessId: string): EventSource {
    return new EventSource(`${API_BASE_URL}/preprocess/${preprocessId}/stream`);
  },

  /**
   * 下載預處理音訊
   */
  downloadPreprocessedAudio(preprocessId: string, fileType: 'original' | 'processed' = 'processed'): string {
    return `${API_BASE_URL}/preprocess/${preprocessId}/download?file_type=${fileType}`;
  },

  /**
   * 獲取預處理資訊（已棄用，使用 getPreprocessTask）
   */
  async getPreprocessInfo(preprocessId: string): Promise<{
    preprocess_id: string;
    original_info: any;
    processed_info: any;
    original_file: string;
    processed_file: string;
  }> {
    const response = await axios.get(`${API_BASE_URL}/preprocess/${preprocessId}/info`);
    return response.data;
  },

  /**
   * 取消/刪除預處理任務
   */
  async deletePreprocess(preprocessId: string, permanent: boolean = false): Promise<void> {
    await axios.delete(`${API_BASE_URL}/preprocess/${preprocessId}`, {
      params: { permanent }
    });
  },

  /**
   * 獲取我的預處理任務歷史
   */
  async getMyPreprocessTasks(limit: number = 50): Promise<{
    tasks: any[];
    total: number;
  }> {
    const response = await axios.get(`${API_BASE_URL}/my-preprocess-tasks`, {
      params: { limit }
    });
    return response.data;
  }
};

