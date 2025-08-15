from __future__ import annotations

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import os
from opencc import OpenCC

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
    ,local_files_only=True
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


class TranscriptionResponse(BaseModel):
    text: str


@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        pipe.model.config.forced_decoder_ids = (
            pipe.tokenizer.get_decoder_prompt_ids(
                language="chinese", 
                task="transcribe"
            )
        )

        outputs = pipe(tmp_path)

        os.remove(tmp_path)
        return {"text": cc.convert(outputs["text"])}
    except Exception as e:
        return {"text": f"Error: {str(e)}"}


@app.get("/healthz")
async def healthz():
        return JSONResponse({"ok": True, "device": device, "model": model_id})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


