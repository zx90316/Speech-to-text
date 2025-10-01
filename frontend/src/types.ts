/**
 * API 類型定義
 */

export interface Task {
  task_id: string;
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'canceled';
  progress: number;
  current_stage: string | null;
  queue_position: number;
  enable_diarization: boolean;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  has_result: boolean;
}

export interface TaskCreateResponse {
  task_id: string;
  status: string;
  queue_position: number;
  message: string;
}

export interface TaskHistory {
  client_ip: string;
  total: number;
  tasks: Task[];
}

export interface ProgressEvent {
  status: string;
  progress: number;
  current_stage: string | null;
  queue_position: number;
  error_message: string | null;
  has_result: boolean;
  timestamp: string;
  partial_result?: Array<{
    start: number;
    end: number;
    text: string;
    speaker?: string;
  }>;
}

export interface ServiceStats {
  queue_size: number;
  processing_count: number;
  is_processing: boolean;
  total_waiting: number;
}

