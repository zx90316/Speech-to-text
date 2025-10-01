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
  async createTask(file: File, enableDiarization: boolean = true): Promise<TaskCreateResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post<TaskCreateResponse>(
      `${API_BASE_URL}/tasks`,
      formData,
      {
        params: { enable_diarization: enableDiarization },
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
   * 下載結果
   */
  downloadResult(taskId: string, fileType: 'transcript' | 'raw' = 'transcript'): string {
    return `${API_BASE_URL}/tasks/${taskId}/download?file_type=${fileType}`;
  },

  /**
   * 查詢我的任務歷史
   */
  async getMyTasks(limit: number = 50): Promise<TaskHistory> {
    const response = await axios.get<TaskHistory>(`${API_BASE_URL}/my-tasks`, {
      params: { limit }
    });
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

