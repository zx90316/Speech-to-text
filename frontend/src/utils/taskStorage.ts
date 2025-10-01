/**
 * localStorage 任務管理工具
 * 用於在本地儲存任務 ID 列表，避免 IP 變更導致任務丟失
 */

const STORAGE_KEY = 'whisper_task_ids';
const MAX_TASKS = 100; // 最多儲存 100 個任務 ID

/**
 * 獲取所有已儲存的任務 ID
 */
export function getStoredTaskIds(): string[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    return JSON.parse(stored);
  } catch (error) {
    console.error('讀取任務 ID 失敗:', error);
    return [];
  }
}

/**
 * 新增任務 ID
 */
export function addTaskId(taskId: string): void {
  try {
    const taskIds = getStoredTaskIds();

    // 如果已存在，先移除
    const filtered = taskIds.filter(id => id !== taskId);

    // 新增到最前面
    filtered.unshift(taskId);

    // 限制數量
    const limited = filtered.slice(0, MAX_TASKS);

    localStorage.setItem(STORAGE_KEY, JSON.stringify(limited));
  } catch (error) {
    console.error('儲存任務 ID 失敗:', error);
  }
}

/**
 * 移除任務 ID
 */
export function removeTaskId(taskId: string): void {
  try {
    const taskIds = getStoredTaskIds();
    const filtered = taskIds.filter(id => id !== taskId);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  } catch (error) {
    console.error('移除任務 ID 失敗:', error);
  }
}

/**
 * 清空所有任務 ID
 */
export function clearTaskIds(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error('清空任務 ID 失敗:', error);
  }
}
