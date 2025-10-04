1. 實現詞級時間戳對齊
# 建議新增
def align_timestamps(self, segments, audio, language):
    """使用 wav2vec2 進行詞級對齊"""
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=self.device
    )
    
    result = whisperx.align(
        segments, 
        align_model, 
        metadata, 
        audio, 
        self.device,
        return_char_alignments=False
    )
    
    return result

2. 新增進階 VAD 參數配置
# API 新增參數
vad_onset: float = 0.5  # 語音檢測敏感度
vad_offset: float = 0.363  # 語音結束閾值

# transcribe 調用
vad_parameters = {
    "vad_onset": vad_onset,
    "vad_offset": vad_offset,
    "min_silence_duration_ms": 500
}

segments, info = model.transcribe(
    audio,
    vad_filter=True,
    vad_parameters=vad_parameters
)

3. 語者分離參數控制
# API 新增
min_speakers: Optional[int] = None
max_speakers: Optional[int] = None

# 調用
diarize_segments = diarization_model(
    audio,
    min_speakers=min_speakers,
    max_speakers=max_speakers
)

6. 音訊預處理增強
# 建議新增
def preprocess_audio_for_whisper(audio_path):
    """針對 Whisper 優化的音訊預處理"""
    - 重採樣至 16kHz
    - 轉換為單聲道
    - 音量正規化至 -20dB LUFS
    return processed_audio

7. 信心分數輸出
# 輸出信心分數
for segment in segments:
    print(f"Text: {segment.text}")
    print(f"Confidence: {segment.avg_logprob}")  # 平均對數概率

