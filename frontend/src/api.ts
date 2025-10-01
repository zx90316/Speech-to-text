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
    model?: string
  ): Promise<TaskCreateResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const params: any = { enable_diarization: enableDiarization };
    if (startTime !== undefined) params.start_time = startTime;
    if (endTime !== undefined) params.end_time = endTime;
    if (language) params.language = language;
    if (task) params.task = task;
    if (model) params.model = model;

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
  downloadResult(taskId: string, fileType: 'transcript' | 'raw' = 'transcript'): string {
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
  }
};

