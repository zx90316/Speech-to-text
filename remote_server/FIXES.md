# 修復記錄

## 問題：API 在處理任務時無響應

### 原因分析

當後端 API 收到任務並開始處理時，整個服務會變得無響應，無法處理其他 API 請求。

**根本原因：**
- `process_queue()` 雖然是 async 函數，但內部調用的 `task_processor.process_task()` 包含 CPU 密集型操作
- Whisper 語音轉錄和 Pyannote 語者分離是同步的、阻塞的操作
- 這些操作會阻塞 asyncio 事件循環，導致其他協程無法執行
- 結果：當一個任務在處理時，所有其他 API 請求（包括查詢、SSE 等）都會被阻塞

### 解決方案

使用 `asyncio.to_thread()` 將 CPU 密集型任務移到後台線程執行，避免阻塞事件循環。

### 修改內容

#### 1. `api.py` - 使用線程池執行任務

```python
# 修改前
await task_processor.process_task(
    task_id=task_id,
    audio_path=str(upload_path),
    enable_diarization=task['enable_diarization']
)

# 修改後
await asyncio.to_thread(
    task_processor.process_task_sync,
    task_id=task_id,
    audio_path=str(upload_path),
    enable_diarization=task['enable_diarization']
)
```

**優點：**
- 任務處理在獨立線程中運行
- 主事件循環保持響應
- 其他 API 請求可以正常處理

#### 2. `task_processor.py` - 改為同步函數

```python
# 修改前
async def process_task(self, task_id: str, ...):

# 修改後
def process_task_sync(self, task_id: str, ...):
```

**原因：**
- 函數內部是同步操作，不需要 async/await
- 在線程池中運行，不需要是協程

#### 3. 添加錯誤追蹤

```python
except Exception as e:
    print(f"處理任務 {task_id} 時發生錯誤: {e}")
    import traceback
    traceback.print_exc()  # 新增：打印完整錯誤堆疊
```

### 效果

**修復前：**
- ❌ 處理任務時 API 完全無響應
- ❌ 無法查詢任務狀態
- ❌ SSE 連接中斷
- ❌ 無法提交新任務

**修復後：**
- ✅ 任務處理時 API 保持響應
- ✅ 可以查詢任務狀態
- ✅ SSE 即時更新正常
- ✅ 可以同時提交多個任務
- ✅ 健康檢查和統計端點正常

### 技術說明

#### `asyncio.to_thread()` 的工作原理

```python
await asyncio.to_thread(blocking_function, *args, **kwargs)
```

1. 將阻塞函數提交到 ThreadPoolExecutor
2. 等待結果時，事件循環可以處理其他協程
3. 完成後將結果返回給調用者

#### 線程安全性

- 資料庫操作（SQLite）在每個線程中使用獨立連接
- `DatabaseManager` 使用 `check_same_thread=False`
- 狀態更新通過資料庫同步，線程安全

### 性能影響

- **無額外開銷**：線程在 Python 線程池中重用
- **並發處理**：雖然仍是單任務處理，但 API 響應不受影響
- **資源使用**：每個任務使用一個線程，不會創建過多線程

### 未來改進建議

如果需要真正的並發任務處理，可以考慮：

1. **使用 Celery + Redis**
   - 真正的分布式任務隊列
   - 支援多個 Worker 並行處理
   - 更好的任務管理和監控

2. **使用 Python 多進程**
   - 繞過 GIL 限制
   - 每個任務獨立進程
   - 更好的 CPU 利用率

3. **使用 Ray 或 Dask**
   - 分布式計算框架
   - 適合大規模部署

### 測試建議

1. **並發 API 測試**
   ```bash
   # 提交任務
   curl -X POST http://localhost:8000/api/tasks -F "file=@test.mp3"
   
   # 同時查詢狀態（應該立即響應）
   curl http://localhost:8000/api/tasks/{task_id}
   
   # 健康檢查（應該立即響應）
   curl http://localhost:8000/health
   ```

2. **SSE 連接測試**
   - 提交任務後立即建立 SSE 連接
   - 應該持續接收進度更新
   - 不應該中斷或超時

3. **多任務提交測試**
   - 連續提交多個任務
   - 所有任務應該正常排隊
   - API 始終保持響應

### 日期

2025-10-01

### 版本

API 版本：2.0.0

