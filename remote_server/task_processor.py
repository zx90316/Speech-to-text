"""
任務處理器模組
將原有的 Whisper 轉錄邏輯改造為異步任務
"""
import os
import time
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from pyannote.core import Segment
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from opencc import OpenCC
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from dotenv import load_dotenv

from database import db_manager

# 載入環境變數
load_dotenv()

# 初始化繁簡轉換器
cc = OpenCC('s2twp')

# 添加 FFmpeg 路徑
ffmpeg_path = Path(__file__).parent.parent / "ffmpeg-7.1.1-full_build-shared" / "bin"
if ffmpeg_path.exists():
    os.environ["PATH"] += os.pathsep + str(ffmpeg_path)


class TaskProcessor:
    """任務處理器類別"""

    def __init__(self):
        self.whisper_model = None
        self.current_model_name = None
        self.diarization_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diarization_loaded = False
        self.current_task_id = None
        self._cancelled = False
    
    def load_whisper_model(self, model_name: str):
        """載入 Whisper 模型（確保只有一個模型實例）"""
        if self.whisper_model is not None and self.current_model_name == model_name:
            # 模型已載入且相同，無需重新載入
            return

        # 釋放舊模型
        if self.whisper_model is not None:
            print(f"釋放舊模型: {self.current_model_name}")
            del self.whisper_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 載入新模型
        print(f"正在載入 Whisper 模型: {model_name} (設備: {self.device})")
        self.whisper_model = WhisperModel(
            model_name,
            device=self.device,
            compute_type="float16" if "int8" not in model_name.lower() else "int8",
            local_files_only= True
        )
        self.current_model_name = model_name
        print(f"Whisper 模型載入完成: {model_name}")

    def load_diarization_model(self):
        """載入語者分離模型（延遲載入）"""
        if self.diarization_loaded:
            return

        print("正在載入語者分離模型...")
        diarization_model_id = "pyannote/speaker-diarization-community-1"
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.diarization_model = Pipeline.from_pretrained(
            diarization_model_id,
            token=hf_token
        )
        self.diarization_model.to(torch.device(self.device))
        print("語者分離模型載入完成")
        self.diarization_loaded = True
    
    def check_cancelled(self, task_id: str) -> bool:
        """檢查任務是否被取消"""
        task = db_manager.get_task(task_id)
        return task and task['status'] == 'canceled'
    
    def convert_audio_to_wav(self, input_path: str, output_path: str, start_time: Optional[float] = None, end_time: Optional[float] = None):
        """轉換音訊為 WAV 格式，支援時間裁切"""
        if os.path.exists(output_path):
            return

        command = ["ffmpeg"]

        # 添加開始時間參數
        if start_time is not None:
            command.extend(["-ss", str(start_time)])

        command.extend(["-i", input_path])

        # 添加結束時間參數（使用 -t 指定持續時間）
        if end_time is not None:
            if start_time is not None:
                duration = end_time - start_time
            else:
                duration = end_time
            command.extend(["-t", str(duration)])

        command.extend([
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            output_path
        ])

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"音訊轉換失敗: {e.stderr.decode()}")
    
    def add_speaker_info_to_text(self, timestamp_texts, ann):
        """添加語者資訊到文字"""
        spk_text = []
        for seg, text in timestamp_texts:
            spk = ann.crop(seg).argmax()
            spk_text.append((seg, spk, text))
        return spk_text
    
    def merge_cache(self, text_cache):
        """合併文字快取"""
        sentence = ''.join([item[-1] for item in text_cache])
        spk = text_cache[0][1]
        start = round(text_cache[0][0].start, 1)
        end = round(text_cache[-1][0].end, 1)
        return Segment(start, end), spk, sentence
    
    def merge_sentence(self, spk_text):
        """合併句子"""
        merged_spk_text = []
        pre_spk = None
        text_cache = []
        for seg, spk, text in spk_text:
            if spk != pre_spk and len(text_cache) > 0:
                merged_spk_text.append(self.merge_cache(text_cache))
                text_cache = [(seg, spk, text)]
                pre_spk = spk
            elif spk == pre_spk and text == text_cache[-1][2]:
                continue
            else:
                text_cache.append((seg, spk, text))
                pre_spk = spk
        if len(text_cache) > 0:
            merged_spk_text.append(self.merge_cache(text_cache))
        return merged_spk_text
    
    def diarize_text(self, timestamp_texts, diarization_result):
        """語者分離與文字整合"""
        spk_text = self.add_speaker_info_to_text(timestamp_texts, diarization_result)
        res_processed = self.merge_sentence(spk_text)
        return res_processed
    
    def process_task_sync(
        self,
        task_id: str,
        audio_path: str,
        enable_diarization: bool = True,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        language: Optional[str] = None,
        task: str = 'transcribe',
        model: str = 'XA9/Belle-faster-whisper-large-v3-zh-punct'
    ):
        """
        處理轉錄任務（同步版本，在後台線程中運行）

        Args:
            task_id: 任務ID
            audio_path: 音訊檔案路徑
            enable_diarization: 是否啟用語者分離
            start_time: 開始時間（秒）
            end_time: 結束時間（秒）
            language: 語言代碼（如 "zh", "en"）
            task: 任務類型（"transcribe" 或 "translate"）
            model: Whisper 模型名稱
        """
        try:
            # 檢查是否被取消
            if self.check_cancelled(task_id):
                return

            # 載入 Whisper 模型
            db_manager.update_task_status(
                task_id,
                'processing',
                progress=0.0,
                current_stage='載入 Whisper 模型'
            )
            self.load_whisper_model(model)

            # 載入語者分離模型（如果需要）
            if enable_diarization:
                db_manager.update_task_status(
                    task_id,
                    'processing',
                    progress=3.0,
                    current_stage='載入語者分離模型'
                )
                self.load_diarization_model()
            
            # 檢查是否被取消
            if self.check_cancelled(task_id):
                return
            
            # 創建結果資料夾
            result_dir = Path(__file__).parent / "result" / task_id
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 轉換音訊格式
            db_manager.update_task_status(
                task_id, 
                'processing', 
                progress=20.0, 
                current_stage='轉換音訊格式'
            )
            
            converted_audio_path = result_dir / "converted_audio.wav"
            self.convert_audio_to_wav(audio_path, str(converted_audio_path), start_time, end_time)
            
            # 檢查是否被取消
            if self.check_cancelled(task_id):
                return
            
            # 語音轉文字
            db_manager.update_task_status(
                task_id, 
                'processing', 
                progress=30.0, 
                current_stage='語音轉文字 (ASR)'
            )
            
            segments, info = self.whisper_model.transcribe(
                audio=str(converted_audio_path),
                language=language,
                task=task,
                log_progress=True,
            )
            
            timestamp_texts = []
            asr_lines = []
            partial_result = []
            
            # 處理 segments（這是一個生成器）
            segment_count = 0
            for segment in segments:
                # 檢查是否被取消
                if self.check_cancelled(task_id):
                    return
                
                segment_count += 1
                converted_text = cc.convert(segment.text)
                timestamp_texts.append((Segment(segment.start, segment.end), converted_text))
                asr_line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, converted_text)
                asr_lines.append(asr_line)
                
                # 添加部分結果
                partial_result.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': converted_text
                })
                
                # 每處理 5 個片段更新一次進度
                if segment_count % 5 == 0:
                    current_progress = min(30.0 + (segment_count * 0.5), 55.0)
                    db_manager.update_task_status(
                        task_id, 
                        'processing', 
                        progress=current_progress, 
                        current_stage=f'語音轉文字 (已處理 {segment_count} 個片段)'
                    )
                    db_manager.update_task_result(task_id, "", partial_result)
            
            db_manager.update_task_status(
                task_id, 
                'processing', 
                progress=60.0, 
                current_stage='ASR 轉錄完成'
            )
            
            # 保存原始 ASR 結果
            asr_raw_path = result_dir / "transcript_raw.txt"
            with open(asr_raw_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(asr_lines))
            
            # 檢查是否被取消
            if self.check_cancelled(task_id):
                return
            
            # 語者分離（如果啟用）
            final_result = None
            if enable_diarization and self.diarization_model:
                db_manager.update_task_status(
                    task_id, 
                    'processing', 
                    progress=70.0, 
                    current_stage='執行語者分離'
                )
                
                diarization_output = self.diarization_model(str(converted_audio_path))
                diarization_result = diarization_output.speaker_diarization
                
                db_manager.update_task_status(
                    task_id, 
                    'processing', 
                    progress=85.0, 
                    current_stage='整合語者資訊'
                )
                
                final_result = self.diarize_text(timestamp_texts, diarization_result)
                
                # 更新部分結果（包含語者資訊）
                partial_result = []
                dialogue_lines = []
                for segment, spk, sent in final_result:
                    partial_result.append({
                        'start': segment.start,
                        'end': segment.end,
                        'speaker': spk,
                        'text': sent
                    })
                    formatted_line = f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] {spk}: {sent}"
                    dialogue_lines.append(formatted_line)
                
                # 保存語者分離結果
                dialogue_path = result_dir / "transcript_with_speakers.txt"
                with open(dialogue_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(dialogue_lines))
            else:
                # 沒有語者分離，使用原始 ASR 結果
                dialogue_path = result_dir / "transcript.txt"
                with open(dialogue_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(asr_lines))
            
            # 刪除臨時轉換的音訊檔案
            if converted_audio_path.exists():
                os.remove(converted_audio_path)
            
            # 檢查是否被取消
            if self.check_cancelled(task_id):
                return
            
            # 任務完成
            db_manager.update_task_status(
                task_id, 
                'completed', 
                progress=100.0, 
                current_stage='處理完成'
            )
            db_manager.update_task_result(
                task_id, 
                str(result_dir), 
                partial_result
            )
            
        except Exception as e:
            error_msg = str(e)
            print(f"任務 {task_id} 處理失敗: {error_msg}")
            db_manager.update_task_status(
                task_id, 
                'failed', 
                error_message=error_msg
            )


# 全局任務處理器實例
task_processor = TaskProcessor()

