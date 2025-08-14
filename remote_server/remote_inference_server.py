from __future__ import annotations

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import pipeline
import tempfile
import os


app = FastAPI(title="Remote Whisper Inference Server", version="0.1.0")


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8


pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
    device=0 if DEVICE.startswith("cuda") else -1,
)


class TranscriptionResponse(BaseModel):
    text: str
    language: str


@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        outputs = pipe(
            tmp_path,
            chunk_length_s=30,
            batch_size=BATCH_SIZE,
            return_timestamps=False,
        )

        os.remove(tmp_path)

        return {"text": outputs["text"], "language": outputs.get("language", "unknown")}
    except Exception as e:
        return {"text": f"Error: {str(e)}", "language": "error"}


