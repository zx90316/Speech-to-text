import React, { useEffect, useMemo, useRef, useState } from 'react'

const BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

type TaskStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed'
type Tokens = { input: number; output: number }

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [model, setModel] = useState<'vertex_ai' | 'remote_llm'>('remote_llm')
  const [startTime, setStartTime] = useState('')
  const [endTime, setEndTime] = useState('')
  const [taskStatus, setTaskStatus] = useState<TaskStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [partialText, setPartialText] = useState('')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [tokens, setTokens] = useState<Tokens>({ input: 0, output: 0 })
  const [errorMsg, setErrorMsg] = useState<string>('')

  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!taskId) return
    const ws = new WebSocket(`${BACKEND.replace('http', 'ws')}/ws/v1/status/${taskId}`)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        if (typeof data.progress === 'number') setProgress(data.progress)
        if (typeof data.partial_text === 'string') setPartialText(data.partial_text)
        if (data.tokens && typeof data.tokens.input === 'number' && typeof data.tokens.output === 'number') {
          setTokens({ input: data.tokens.input, output: data.tokens.output })
        }
        if (data.status === 'completed') setTaskStatus('completed')
        if (data.status === 'failed') {
          setTaskStatus('failed')
          if (typeof data.error === 'string') setErrorMsg(data.error)
        }
      } catch {}
    }
    ws.onerror = () => setTaskStatus('failed')
    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [taskId])

  async function handleStart() {
    if (!file) return
    setTaskStatus('uploading')
    setProgress(0)
    setPartialText('')
    setTokens({ input: 0, output: 0 })
    setErrorMsg('')

    const form = new FormData()
    form.append('file', file)

    const q = new URLSearchParams()
    q.set('model_choice', model)
    if (startTime) q.set('start_time', startTime)
    if (endTime) q.set('end_time', endTime)

    const res = await fetch(`${BACKEND}/api/v1/transcribe?${q.toString()}`, {
      method: 'POST',
      body: form,
    })
    if (!res.ok) {
      setTaskStatus('failed')
      return
    }
    const data = await res.json()
    setTaskId(data.task_id)
    setTaskStatus('processing')
  }

  const downloadUrlPlain = useMemo(() => taskId ? `${BACKEND}/api/v1/result/${taskId}?format=plain` : '' , [taskId])
  const downloadUrlTimestamped = useMemo(() => taskId ? `${BACKEND}/api/v1/result/${taskId}?format=timestamped` : '' , [taskId])
  const downloadUrlSrt = useMemo(() => taskId ? `${BACKEND}/api/v1/result/${taskId}?format=srt` : '' , [taskId])

  return (
    <div className="container">
      <div className="row" style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>語音轉文字</h2>
        <span style={{ color: '#9ca3af' }}>支援 Vertex AI 與遠端 Whisper，含即時 token 顯示</span>
      </div>
      <div className="card row">
        <div className="row">
          <div>
            <label>音訊檔案</label>
            <input type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <div>
            <label>模型選擇</label>
            <select value={model} onChange={(e) => setModel(e.target.value as any)}>
              <option value="remote_llm">遠端本地模型（Whisper）</option>
              <option value="vertex_ai">Google Vertex AI</option>
            </select>
          </div>
          <div className="inline">
            <div>
              <label>開始時間（HH:MM:SS，可空）</label>
              <input value={startTime} onChange={(e) => setStartTime(e.target.value)} placeholder="00:00:00" />
            </div>
            <div>
              <label>結束時間（HH:MM:SS，可空）</label>
              <input value={endTime} onChange={(e) => setEndTime(e.target.value)} placeholder="00:01:00" />
            </div>
          </div>
          <div>
            <button onClick={handleStart} disabled={!file || taskStatus === 'uploading' || taskStatus === 'processing'}>
              {taskStatus === 'uploading' ? '上傳中...' : taskStatus === 'processing' ? '處理中...' : '開始轉錄'}
            </button>
          </div>
        </div>
      </div>

      <div className="card row" style={{ marginTop: 16 }}>
        {taskStatus === 'failed' && (
          <div className="alert">{errorMsg || '處理失敗，請稍後重試。'}</div>
        )}
        <div>
          <label>進度</label>
          <progress value={progress} max={100} />
        </div>
        <div className="inline" style={{ justifyContent: 'space-between' }}>
          <div>
            <label>Tokens</label>
            <div style={{ color: '#9ca3af' }}>Input: {tokens.input} | Output: {tokens.output}</div>
          </div>
          <div style={{ fontSize: 12, color: '#9ca3af' }}>以 WebSocket 即時更新</div>
        </div>
        <div>
          <label>即時 Tokens</label>
          <div className="tokens" id="live-tokens">{partialText}</div>
          <div className="toolbar" style={{ marginTop: 8 }}>
            <button className="ghost" onClick={() => navigator.clipboard.writeText(partialText)}>複製</button>
            <button className="ghost" onClick={() => setPartialText('')}>清空</button>
          </div>
        </div>
        {taskStatus === 'completed' && (
          <div className="downloads inline">
            <a href={downloadUrlPlain}>下載純文字</a>
            <a href={downloadUrlTimestamped}>下載帶時間戳</a>
            <a href={downloadUrlSrt}>下載 SRT</a>
          </div>
        )}
      </div>
    </div>
  )
}


