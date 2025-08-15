import React, { useEffect, useMemo, useRef, useState } from 'react'

const BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://192.168.80.24:8000'

type TaskStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed'
type Tokens = { input: number; output: number }

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [model, setModel] = useState<'vertex_ai' | 'remote_llm'>('vertex_ai')
  const [startTime, setStartTime] = useState('')
  const [endTime, setEndTime] = useState('')
  const [prompt, setPrompt] = useState(`Your task is to provide a direct transcription of an audio file. The output must contain ONLY the transcribed text. Omit any preambles, introductory phrases, or notes.
Key requirements:
1.  Extract speech only and ignore background sounds.
2.  If any speech is in Chinese, transcribe it using Traditional Chinese characters.`)
  const [temperature, setTemperature] = useState(0)
  const [topP, setTopP] = useState(0.95)
  const [maxTokens, setMaxTokens] = useState(65535)
  const [chunkLength, setChunkLength] = useState(30)
  const [taskStatus, setTaskStatus] = useState<TaskStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [partialText, setPartialText] = useState('')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [tokens, setTokens] = useState<Tokens>({ input: 0, output: 0 })
  const [errorMsg, setErrorMsg] = useState<string>('')
  const [canceling, setCanceling] = useState(false)
  const [segments, setSegments] = useState<Array<{ start: number; end: number; text: string }>>([])

  const wsRef = useRef<WebSocket | null>(null)
  const liveTokensRef = useRef<HTMLDivElement | null>(null)
  const segmentsRef = useRef<HTMLDivElement | null>(null)

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
        if (Array.isArray(data.segments)) {
          setSegments(data.segments as any)
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

  useEffect(() => {
    if (liveTokensRef.current) {
      liveTokensRef.current.scrollTop = liveTokensRef.current.scrollHeight
    }
  }, [partialText])

  useEffect(() => {
    if (segmentsRef.current) {
      segmentsRef.current.scrollTop = segmentsRef.current.scrollHeight
    }
  }, [segments])

  async function handleStart() {
    if (!file) return
    setTaskStatus('uploading')
    setProgress(0)
    setPartialText('')
    setTokens({ input: 0, output: 0 })
    setErrorMsg('')
    setSegments([])

    const form = new FormData()
    form.append('file', file)

    const q = new URLSearchParams()
    q.set('model_choice', model)
    if (startTime) q.set('start_time', startTime)
    if (endTime) q.set('end_time', endTime)
    if (chunkLength) q.set('chunk_length', String(chunkLength))
    if (model === 'vertex_ai') {
      if (prompt) q.set('prompt', prompt)
      if (temperature != null) q.set('temperature', String(temperature))
      if (topP != null) q.set('top_p', String(topP))
      if (maxTokens != null) q.set('max_output_tokens', String(maxTokens))
      q.set('thinking_budget', String(0))
      q.set('safety_off', String(true))
    }

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

  async function handleCancel() {
    if (!taskId) return
    setCanceling(true)
    try {
      await fetch(`${BACKEND}/api/v1/cancel/${taskId}`, { method: 'POST' })
    } finally {
      setCanceling(false)
    }
  }

  return (
    <div className="container">
      <header className="top-bar">
        <h2 style={{ margin: 0 }}>語音轉文字</h2>
        <span style={{ color: '#9ca3af' }}>支援 Vertex AI 與遠端 Whisper，含即時 token 顯示</span>
      </header>
      <div className="main-content">
        <div className="left-panel">
            <div>
              <label>音訊檔案</label>
              <input type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </div>
            <div>
              <label>模型選擇</label>
              <select value={model} onChange={(e) => setModel(e.target.value as any)}>
                <option value="vertex_ai">Google Vertex AI</option>
                <option value="remote_llm">遠端本地模型（Whisper）</option>
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
              <div>
                <label>每段分塊秒數</label>
                <input type="number" min={5} max={120} value={chunkLength} onChange={(e) => setChunkLength(Number(e.target.value))} />
              </div>
            </div>
            {model === 'vertex_ai' && (
              <div className="row">
                <div>
                  <label>提示詞（Prompt）</label>
                  <textarea style={{ width: '90%' }} rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Generate a transcription of the audio, only extract speech and ignore background audio. If any part of the speech is in Chinese, please transcribe it using Traditional Chinese characters." />
                </div>
                <div className="inline">
                  <div>
                    <label>Temperature</label>
                    <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
                  </div>
                  <div>
                    <label>Top P</label>
                    <input type="number" step="0.05" value={topP} onChange={(e) => setTopP(Number(e.target.value))} />
                  </div>
                  <div>
                    <label>Max Output Tokens</label>
                    <input type="number" min={128} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
                  </div>
                </div>
              </div>
            )}
            <div>
              <button onClick={handleStart} disabled={!file || taskStatus === 'uploading' || taskStatus === 'processing'}>
                {taskStatus === 'uploading' ? '上傳中...' : taskStatus === 'processing' ? '處理中...' : '開始轉錄'}
              </button>
              {taskStatus === 'processing' && (
                <button className="ghost" style={{ marginLeft: 8 }} onClick={handleCancel} disabled={canceling}>取消</button>
              )}
            </div>
        </div>

        <div className="right-display">
          {taskStatus === 'failed' && (
            <div className="alert">{errorMsg || '處理失敗，請稍後重試。'}</div>
          )}
            <div className="left-col row">
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
              {taskStatus === 'completed' && (
                <div className="downloads inline">
                  <a href={downloadUrlPlain}>下載純文字</a>
                  <a href={downloadUrlTimestamped}>下載帶時間戳</a>
                  <a href={downloadUrlSrt}>下載 SRT</a>
                </div>
              )}
            </div>
            <div className="right-panel">
              <div className="panel-section">
                <label>即時 Tokens</label>
                <div className="tokens scrollable" id="live-tokens" ref={liveTokensRef}>{partialText}</div>
                <div className="toolbar" style={{ marginTop: 8 }}>
                  <button className="ghost" onClick={() => navigator.clipboard.writeText(partialText)}>複製</button>
                  <button className="ghost" onClick={() => setPartialText('')}>清空</button>
                </div>
              </div>
              <div className="panel-section">
                <label>逐段結果</label>
                <div className="tokens scrollable" ref={segmentsRef}>
                  {segments.length === 0 && <div style={{ color: '#9ca3af' }}>尚無資料</div>}
                  {segments.map((s, i) => (
                    <div key={i} style={{ marginBottom: 8 }}>
                      <div style={{ color: '#9ca3af', fontSize: 12 }}>
                        [{s.start.toFixed(2)} - {s.end.toFixed(2)}]
                      </div>
                      <div>{s.text}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
        </div>
      </div>
    </div>
  )
}


