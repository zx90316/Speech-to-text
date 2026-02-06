# -*- coding: utf-8 -*-
"""
整合處理管道
協調 ASR 和語者分離的完整處理流程
"""
import os
from pathlib import Path
from typing import Optional, Callable

from opencc import OpenCC

from config import RESULT_DIR
from models import TaskStatus
from storage import storage
from audio_utils import convert_to_wav
from asr_processor import qwen_processor
from diarization_processor import diarizer
from email_service import email_service


# 繁簡轉換器
cc = OpenCC('s2twp')


class TranscriptionPipeline:
    """轉錄處理管道"""
    
    def __init__(self):
        self.asr = qwen_processor
        self.diarizer = diarizer
    
    def process(
        self,
        task_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> bool:
        """
        執行完整處理流程
        
        Args:
            task_id: 任務 ID
            progress_callback: 進度回調 (progress, stage)
            
        Returns:
            是否成功
        """
        task = storage.get_task(task_id)
        if not task:
            return False
        
        audio_path = task["audio_path"]
        enable_diarization = task["enable_diarization"]
        enable_timestamps = task["enable_timestamps"]
        language = task["language"]
        model = task["model"]
        min_speakers = task["min_speakers"]
        max_speakers = task["max_speakers"]
        
        def update_progress(progress: float, stage: str):
            storage.update_task(task_id, progress=progress, current_stage=stage)
            if progress_callback:
                progress_callback(progress, stage)
        
        try:
            # 檢查是否已取消
            if storage.is_canceled(task_id):
                return False
            
            storage.update_task(task_id, status=TaskStatus.PROCESSING)
            
            # Step 1: 轉換音訊格式
            update_progress(10.0, "轉換音訊格式")
            result_dir = RESULT_DIR / task_id
            result_dir.mkdir(parents=True, exist_ok=True)
            
            converted_path = result_dir / "converted.wav"
            convert_to_wav(audio_path, str(converted_path))
            
            if storage.is_canceled(task_id):
                return False
            
            # Step 2: 載入 ASR 模型
            update_progress(20.0, "載入 ASR 模型")
            self.asr.load_model(
                model_name=model,
                enable_timestamps=enable_timestamps,
            )
            
            if storage.is_canceled(task_id):
                self.asr.unload_model()
                return False
            
            # Step 3: 執行轉錄
            update_progress(30.0, "語音轉文字中")
            
            # 轉換語言代碼
            qwen_language = self._convert_language_code(language)
            
            asr_result = self.asr.transcribe(
                audio_path=str(converted_path),
                language=qwen_language,
                return_timestamps=enable_timestamps,
            )
            
            # 轉換為繁體中文
            text = cc.convert(asr_result["text"])
            segments = asr_result.get("segments", [])
            timestamps = asr_result.get("timestamps", [])
            
            # 保存 ASR debug 資訊
            debug_log_path = result_dir / "debug.log"
            with open(debug_log_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("ASR 輸出結果\n")
                f.write("="*60 + "\n")
                f.write(f"Text length: {len(text)} chars\n")
                f.write(f"Segments count: {len(segments)}\n")
                f.write(f"Timestamps count: {len(timestamps)}\n")
                for i, seg in enumerate(segments):
                    f.write(f"\nSegment {i}:\n")
                    f.write(f"  start: {seg.get('start')}\n")
                    f.write(f"  end: {seg.get('end')}\n")
                    f.write(f"  text: {seg.get('text', '')[:100]}...\n")
                    words = seg.get('words', [])
                    f.write(f"  words count: {len(words) if words else 0}\n")
                    if words and len(words) > 0:
                        f.write(f" first 10 words: {words[:10]}\n")
                        f.write(f" last 10 words: {words[-10:]}\n")

                f.write("\n")
            
            # 轉換 segments 中的文字
            for seg in segments:
                seg["text"] = cc.convert(seg["text"])
                if seg.get("words"):
                    for word in seg["words"]:
                        word["word"] = cc.convert(word["word"])
                        
            
            detected_language = asr_result.get("language", "Unknown")
            
            # 卸載 ASR 模型
            self.asr.unload_model()
            
            if storage.is_canceled(task_id):
                return False
            
            # Step 4: 語者分離 (如果啟用)
            has_diarization = False
            
            if enable_diarization:
                try:
                    update_progress(60.0, "載入語者分離模型")
                    self.diarizer.load_model()
                    
                    if storage.is_canceled(task_id):
                        self.diarizer.unload_model()
                        return False
                    
                    update_progress(70.0, "執行語者分離")
                    
                    def diarization_progress(step_name: str, progress: float):
                        # 映射進度到 70-90 區間
                        mapped_progress = 70.0 + (progress / 100.0) * 20.0
                        update_progress(mapped_progress, f"語者分離: {step_name}")
                    
                    diarization = self.diarizer.diarize(
                        audio_path=str(converted_path),
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        progress_callback=diarization_progress,
                    )
                    
                    update_progress(90.0, "整合語者資訊")
                    
                    segments = self.diarizer.merge_with_transcript(
                        segments, diarization, debug_log_path=debug_log_path
                    )
                    
                    has_diarization = True
                    
                    self.diarizer.unload_model()
                    
                except Exception as e:
                    print(f"⚠️ 語者分離失敗: {e}")
                    # 繼續使用 ASR 結果
                    self.diarizer.unload_model()
            
            # Step 5: 保存結果
            update_progress(95.0, "保存結果")
            
            # 保存文字檔
            transcript_path = result_dir / "transcript.txt"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(f"[ASR: Qwen | 語言: {detected_language}]\n\n")
                
                for seg in segments:
                    if has_diarization and seg.get("speaker"):
                        line = f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}"
                    elif seg.get("start") and seg.get("end"):
                        line = f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}"
                    else:
                        line = seg['text']
                    f.write(line + "\n")
            
            # 刪除臨時音訊檔
            if converted_path.exists():
                os.remove(converted_path)
            
            # 更新任務結果
            storage.update_task(
                task_id,
                status="completed",
                progress=100.0,
                current_stage="完成",
                text=text,
                detected_language=detected_language,
                segments=segments,
                has_diarization=has_diarization,
            )
            
            # 發送郵件通知（如果有設定 email）
            email = task.get("email")
            if email:
                try:
                    email_service.send_completion_email(
                        to_email=email,
                        task_id=task_id,
                        filename=task["filename"],
                        transcript_text=text,
                        has_diarization=has_diarization,
                        detected_language=detected_language,
                    )
                except Exception as email_error:
                    print(f"⚠️ 發送郵件通知失敗: {email_error}")
            
            print(f"✅ 任務 {task_id} 處理完成")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ 任務 {task_id} 處理失敗: {error_msg}")
            
            storage.update_task(
                task_id,
                status="failed",
                error_message=error_msg,
            )
            
            # 清理
            try:
                self.asr.unload_model()
            except Exception:
                pass
            try:
                self.diarizer.unload_model()
            except Exception:
                pass
            
            return False
    
    def _convert_language_code(self, language: Optional[str]) -> Optional[str]:
        """轉換語言代碼為 Qwen 格式"""
        if not language:
            return None
        
        language_map = {
            'zh': 'Chinese',
            'en': 'English',
            'ja': 'Japanese',
            'ko': 'Korean',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
        }
        
        return language_map.get(language.lower(), language)


# 建立全域實例
pipeline = TranscriptionPipeline()
