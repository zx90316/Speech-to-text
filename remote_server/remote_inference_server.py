from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import os
import librosa
import soundfile as sf
from opencc import OpenCC

from pyannote.audio import Pipeline

# 初始化轉換器，'s2twp' 表示從簡體（s）轉換到台灣繁體（tw），並包含詞彙轉換（p）
# s2t: 簡轉繁
# s2tw: 簡轉臺
cc = OpenCC('s2twp')  

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load .env from project root or current folder
root_env = Path(__file__).resolve().parents[1] / ".env"
local_env = Path(__file__).resolve().parent / ".env"
for dotenv_path in (root_env, local_env):
    load_dotenv(dotenv_path=dotenv_path, override=True)

# Optional: Add FFMPEG to PATH from env or fallback
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg-7.1.1-essentials_build/bin")
if FFMPEG_PATH:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

app = FastAPI(title="Remote Whisper Inference Server", version="0.1.0")


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_id = os.getenv("MODEL_NAME", "BELLE-2/Belle-whisper-large-v3-zh-punct")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    #,local_files_only=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# 初始化語者分離模型
pipeline = None

def initialize_diarization():
    global pipeline
    
    try:
        # 檢查環境變數
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("警告：未設定 HUGGINGFACE_TOKEN，語者分離功能將不可用")
            return

        # 嘗試載入語者分離模型
        model_id = "pyannote/speaker-diarization-3.1"

        try:
            print(f"🔄 嘗試載入 {model_id}...")
            
            # 增加更多錯誤處理
            pipeline = Pipeline.from_pretrained(model_id,use_auth_token=hf_token)     
        except Exception as model_error:
            print(f"❌ 無法載入 {model_id}: {type(model_error).__name__}: {model_error}")
        
    except Exception as e:
        print(f"❌ 語者分離初始化失敗: {type(e).__name__}: {e}")
        import traceback
        print("詳細錯誤追蹤:")
        traceback.print_exc()

# 嘗試初始化語者分離
print("🚀 初始化語者分離功能...")
initialize_diarization()

if pipeline is None:
    print("⚠️ 語者分離功能將不可用，但語音轉錄功能正常")


class Chunk(BaseModel):
    text: str
    timestamp: tuple[float | None, float | None]
    speaker: Optional[str] = None

class TranscriptionResponse(BaseModel):
    chunks: list[Chunk]


@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    enable_diarization: bool = Query(default=False),
    min_speakers: Optional[int] = Query(default=None),
    max_speakers: Optional[int] = Query(default=None),
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 如果啟用語者分離且模型可用
        diarization = None
        if enable_diarization:
            if pipeline is None:
                print("⚠️ 語者分離已啟用但模型不可用，將回退到無語者分離模式")
            else:
                try:
                    print("🔄 開始語者分離處理...")
                    # 先將音訊轉換為正確的格式
                    audio, sr = librosa.load(tmp_path, sr=16000)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_tmp:
                        sf.write(audio_tmp.name, audio, sr)
                        audio_tmp_path = audio_tmp.name
                    
                    # 執行語者分離
                    diarization_params = {}
                    if min_speakers is not None:
                        diarization_params['min_speakers'] = min_speakers
                    if max_speakers is not None:
                        diarization_params['max_speakers'] = max_speakers
                    
                    diarization = pipeline(audio_tmp_path, **diarization_params)
                    os.remove(audio_tmp_path)
                    
                    segment_count = len(list(diarization.itertracks())) if diarization else 0
                    print(f"✅ 語者分離完成: {segment_count} 個語音段落")
                    
                except Exception as e:
                    print(f"❌ 語者分離失敗: {type(e).__name__}: {e}")
                    diarization = None

        pipe.model.config.forced_decoder_ids = (
            pipe.tokenizer.get_decoder_prompt_ids(
                language="chinese", 
                task="transcribe"
            )
        )

        outputs = pipe(tmp_path, return_timestamps=True)
        
        chunks_output = []
        for chunk in outputs.get("chunks", []):
            timestamp_start = None
            timestamp_end = None
            if chunk.get('timestamp') is not None:
                timestamp_start = chunk['timestamp'][0]
                timestamp_end = chunk['timestamp'][1]

            # 尋找對應的語者
            speaker = None
            if diarization and timestamp_start is not None:
                # 找到時間戳記中點對應的語者
                mid_time = timestamp_start + (timestamp_end - timestamp_start) / 2 if timestamp_end else timestamp_start
                for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                    if turn.start <= mid_time <= turn.end:
                        speaker = speaker_label
                        break

            chunks_output.append({
                "text": cc.convert(chunk.get('text', '')),
                "timestamp": (timestamp_start, timestamp_end),
                "speaker": speaker
            })
        
        os.remove(tmp_path)
        return {"chunks": chunks_output}
    except Exception as e:
        return {"chunks": [{"text": f"Error: {str(e)}", "timestamp": (0, 0), "speaker": None}]}


@app.get("/healthz")
async def healthz():
        return JSONResponse({"ok": True, "device": device, "model": model_id})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


