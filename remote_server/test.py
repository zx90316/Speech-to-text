from pyannote.core import Segment
import torch
import time
import os
import subprocess
from opencc import OpenCC
from faster_whisper import WhisperModel

from pyannote.audio import Pipeline
from dotenv import load_dotenv

file = "語音 250814_133051.m4a"

# 新增 FFmpeg 可執行檔的路徑到 PATH
# 注意要把bin底下的DLL複製到.venv/Lib/site-packages/torchcodec底下
# https://www.gyan.dev/ffmpeg/builds/
os.environ["PATH"] += os.pathsep + "ffmpeg-7.1.1-full_build-shared/bin"

# 也添加本地 ffmpeg 路徑
ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg-7.1.1-full_build-shared", "bin")
os.environ["PATH"] += os.pathsep + ffmpeg_path
# 載入環境變數
load_dotenv()

# 初始化繁簡轉換器
cc = OpenCC('s2twp')

# 進度追蹤類別（為 FastAPI 轉換做準備）
class ProgressTracker:
    """追蹤音訊處理的進度狀態"""

    def __init__(self):
        self.current_stage = ""
        self.progress_percentage = 0.0
        self.status = "pending"  # pending, processing, completed, failed
        self.partial_result = []
        self.error_message = None

    def update(self, stage: str, percentage: float, status: str = "processing", message: str = None):
        """更新進度"""
        self.current_stage = stage
        self.progress_percentage = percentage
        self.status = status
        if message:
            print(f"[{percentage:5.1f}%] {stage}: {message}")
        else:
            print(f"[{percentage:5.1f}%] {stage}")

    def add_partial_result(self, segment_data: dict):
        """添加部分結果（用於即時顯示）"""
        self.partial_result.append(segment_data)

    def set_error(self, error: str):
        """設置錯誤訊息"""
        self.status = "failed"
        self.error_message = error
        print(f"❌ 錯誤: {error}")

    def complete(self):
        """標記為完成"""
        self.status = "completed"
        self.progress_percentage = 100.0
        print(f"[100.0%] 處理完成")

    def get_state(self) -> dict:
        """獲取當前狀態（為 FastAPI WebSocket 準備）"""
        return {
            "stage": self.current_stage,
            "percentage": self.progress_percentage,
            "status": self.status,
            "partial_result": self.partial_result,
            "error": self.error_message
        }

def convert_audio_to_wav(input_audio_path: str, output_audio_path: str):
    """
    使用 FFmpeg 將任何音訊格式轉換為 16kHz 單聲道 WAV 格式。
    """
    if not os.path.exists(output_audio_path):
        command = [
            "ffmpeg",
            "-i", input_audio_path,
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            output_audio_path
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"成功將 {input_audio_path} 轉換為 {output_audio_path}")
        except subprocess.CalledProcessError as e:
            print(f"轉換音訊時發生錯誤: {e}")
            print(f"FFmpeg 輸出: {e.stderr.decode()}")
            raise

def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        # print(ann.crop(seg))  # 註解掉以避免干擾進度顯示
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text
 
def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = round(text_cache[0][0].start, 1)
    end = round(text_cache[-1][0].end, 1)
    return Segment(start, end), spk, sentence
 
def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk
        elif spk == pre_spk and text == text_cache[-1][2]:
            # print(text_cache[-1][2])  # 註解掉以避免干擾進度顯示
            continue
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text
 
 
def diarize_text(timestamp_texts, diarization_result):
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed

if __name__ == "__main__":
    # 初始化進度追蹤器
    progress = ProgressTracker()

    print("=" * 60)
    print(f"開始處理音訊檔案: {file}")
    print("=" * 60)

    # 設定設備和數據類型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 載入模型
    progress.update("初始化", 0.0, message="載入 BELLE Whisper ASR 模型")
    model = WhisperModel("k1nto/Belle-whisper-large-v3-zh-punct-ct2", device=device , compute_type="float16") #int8_float16 int8

    progress.update("初始化", 5.0, message=f"Whisper Large V3 模型載入完成 (設備: {device})")

    progress.update("初始化", 10.0, message="載入語者分離模型")
    diarization_model_id = "pyannote/speaker-diarization-community-1"
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    speaker_diarization = Pipeline.from_pretrained(diarization_model_id, token=hf_token)
    speaker_diarization.to(torch.device(device))
    progress.update("初始化", 15.0, message="語者分離模型載入完成")

    start_time = time.time()
 
    dialogue_path = "./audios_txt/" + file.split(".")[0] + ".txt"
    audio = "./audios_wav/" + file

    # 轉換音訊檔案為標準 WAV 格式
    progress.update("音訊預處理", 20.0, message="轉換音訊格式為 WAV")
    converted_audio_path = "./audios_wav/converted_" + file.split(".")[0] + ".wav"
    convert_audio_to_wav(audio, converted_audio_path)
    progress.update("音訊預處理", 25.0, message="音訊格式轉換完成")

    # 語音轉文字
    progress.update("語音轉文字", 30.0, message="開始執行 ASR 轉錄")


    segments, info = model.transcribe(
        audio=converted_audio_path,
        language="zh",
        task="transcribe",
        log_progress=True,
    )
    timestamp_texts=[]
    for segment in segments:
        timestamp_texts.append((Segment(segment.start, segment.end), cc.convert(segment.text)))
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, cc.convert(segment.text)))

    asr_time = time.time()
    progress.update("語音轉文字", 60.0, message=f"ASR 轉錄完成，耗時 {asr_time - start_time:.2f} 秒")

    progress.update("語者分離", 65.0, message="開始執行語者分離")

    diarization_output = speaker_diarization(converted_audio_path)
    diarization_result = diarization_output.speaker_diarization

    diarization_time = time.time()
    progress.update("語者分離", 80.0, message=f"語者分離完成，耗時 {diarization_time - asr_time:.2f} 秒")

    progress.update("文字整合", 85.0, message="整合轉錄文字與語者資訊")
    final_result = diarize_text(timestamp_texts, diarization_result)
    os.remove(converted_audio_path)
    progress.update("文字整合", 90.0, message="文字與語者整合完成")

    progress.update("結果輸出", 95.0, message="生成最終結果")

    print("\n" + "=" * 60)
    print("轉錄結果:")
    print("=" * 60)

    dialogue = []
    for segment, spk, sent in final_result:
        content = {'speaker': spk, 'start': segment.start, 'end': segment.end, 'text': sent}
        dialogue.append(content)
        # 添加到進度追蹤器的部分結果（為 FastAPI 準備）
        progress.add_partial_result(content)
        print(f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] {spk}: {sent}")

    with open(dialogue_path, 'w', encoding='utf-8') as f:
         f.write(str(dialogue))
    end_time = time.time()

    progress.complete()
    
    print("\n" + "=" * 60)
    print("處理完成統計:")
    print("=" * 60)
    print(f"音訊檔案: {file}")
    print(f"總處理時間: {end_time - start_time:.2f} 秒")
    print(f"輸出檔案: {dialogue_path}")
    print("=" * 60)