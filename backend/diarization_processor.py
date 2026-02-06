# -*- coding: utf-8 -*-
"""
語者分離處理器模組
使用 pyannote.audio 進行語者分離
"""
import gc
from typing import Optional, List, Dict, Tuple, Any

import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation

from config import DIARIZATION_MODEL, HUGGINGFACE_TOKEN
from audio_utils import load_audio


class ProgressHook:
    """語者分離進度追蹤"""
    
    def __init__(self, callback=None):
        self.callback = callback
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def __call__(self, step_name: str, step_artefact=None, file=None, total=None, completed=None):
        if self.callback and completed is not None and total is not None:
            progress = (completed / total) * 100
            self.callback(step_name, progress)


class SpeakerDiarizer:
    """語者分離處理器類別"""
    
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
    
    def load_model(self) -> None:
        """載入語者分離模型"""
        if self.is_loaded:
            print("✓ 語者分離模型已載入")
            return
        
        if not HUGGINGFACE_TOKEN:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN 環境變數未設定。"
                "請在 .env 檔案中設定: HUGGINGFACE_TOKEN=hf_xxx"
            )
        
        print(f"🔄 正在載入語者分離模型: {DIARIZATION_MODEL}")
        
        self.pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            token=HUGGINGFACE_TOKEN
        )
        self.pipeline.to(torch.device(self.device))
        self.is_loaded = True
        
        print("✅ 語者分離模型載入完成")
    
    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback=None,
    ) -> Annotation:
        """
        執行語者分離
        
        Args:
            audio_path: 音訊檔案路徑
            min_speakers: 最小語者數
            max_speakers: 最大語者數
            progress_callback: 進度回調函式
            
        Returns:
            pyannote Annotation 物件
        """
        if not self.is_loaded:
            self.load_model()
        
        print(f"🔄 執行語者分離: {audio_path}")
        
        waveform, sample_rate = load_audio(audio_path)
        
        params = {}
        if min_speakers is not None:
            params['min_speakers'] = min_speakers
        if max_speakers is not None:
            params['max_speakers'] = max_speakers
        
        with ProgressHook(progress_callback) as hook:
            output = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
                **params
            )
        
        # 取得語者分離結果
        diarization = output.speaker_diarization
        
        # 統計語者數量
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
        
        print(f"✅ 語者分離完成，識別到 {len(speakers)} 位語者")
        
        return diarization
    
    def merge_with_transcript(
        self,
        segments: List[Dict],
        diarization: Annotation,
        debug_log_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        將 ASR 結果與語者標記整合
        
        Args:
            segments: ASR 轉錄片段 [{"start", "end", "text", "words"}, ...]
            diarization: 語者分離結果
            debug_log_path: debug log 檔案路徑（可選）
            
        Returns:
            帶有語者標記的片段列表
        """
        print("🔄 整合語者資訊...")
        
        # 取得所有語者分離片段
        diar_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diar_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
        
        # 寫入 debug log
        if debug_log_path:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*60 + "\n")
                f.write("語者分離結果\n")
                f.write("="*60 + "\n")
                f.write(f"Diarization segments count: {len(diar_segments)}\n")
                for i, diar in enumerate(diar_segments):
                    f.write(f"  Diar {i}: {diar['start']:.2f}s - {diar['end']:.2f}s, Speaker: {diar['speaker']}\n")
                f.write("\n")
                
                # ASR segments info
                f.write("="*60 + "\n")
                f.write("ASR Segments 輸入\n")
                f.write("="*60 + "\n")
                f.write(f"Segments count: {len(segments)}\n")
                for i, seg in enumerate(segments):
                    f.write(f"  Seg {i}: start={seg.get('start')}, end={seg.get('end')}\n")
                    words = seg.get("words", [])
                    f.write(f"          words count: {len(words) if words else 0}\n")
                    if words and len(words) > 0:
                        f.write(f"          first word: {words[0]}\n")
                        f.write(f"          last word: {words[-1]}\n")
                f.write("\n")
        
        if not diar_segments:
            # 沒有語者分離結果，直接返回原始 segments
            for seg in segments:
                seg["speaker"] = "UNKNOWN"
            return segments
        
        result = []
        
        for seg in segments:
            seg_words = seg.get("words", None)
            
            # 如果有 word-level 時間戳，使用更精細的語者分配
            if seg_words and len(seg_words) > 0:
                # 基於 word 時間戳分配語者
                word_speaker_segments = self._assign_speakers_by_words(seg_words, diar_segments)
                result.extend(word_speaker_segments)
            else:
                # 沒有 word 時間戳，使用整個 segment 的時間範圍
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                seg_text = seg.get("text", "")
                
                # 找出該片段的主要語者
                segment = Segment(seg_start, seg_end)
                try:
                    cropped = diarization.crop(segment)
                    if len(cropped) > 0:
                        speaker = cropped.argmax()
                    else:
                        speaker = "UNKNOWN"
                except (IndexError, ValueError):
                    speaker = "UNKNOWN"
                
                result.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text,
                    "speaker": speaker,
                    "words": seg_words,
                })
        
        # 寫入整合結果 debug log
        if debug_log_path:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("整合結果（合併前）\n")
                f.write("="*60 + "\n")
                f.write(f"Result segments count: {len(result)}\n")
                for i, seg in enumerate(result[:20]):
                    f.write(f"  Seg {i}: [{seg.get('start'):.2f}s - {seg.get('end'):.2f}s] {seg.get('speaker')}: {seg.get('text', '')[:50]}...\n")
                if len(result) > 20:
                    f.write(f"  ... and {len(result) - 20} more\n")
                f.write("\n")
        
        # 合併相鄰的相同語者片段
        merged = self._merge_same_speaker(result)
        
        # 寫入合併後結果
        if debug_log_path:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("整合結果（合併後）\n")
                f.write("="*60 + "\n")
                f.write(f"Merged segments count: {len(merged)}\n")
                for i, seg in enumerate(merged):
                    f.write(f"  Seg {i}: [{seg.get('start'):.2f}s - {seg.get('end'):.2f}s] {seg.get('speaker')}: {seg.get('text', '')[:50]}...\n")
                f.write("\n")
        
        print(f"✅ 整合完成，共 {len(merged)} 個片段")
        
        return merged
    
    def _assign_speakers_by_words(
        self,
        words: List[Dict],
        diar_segments: List[Dict],
    ) -> List[Dict]:
        """基於 word 時間戳分配語者"""
        if not words or not diar_segments:
            return []
        
        # 為每個 word 分配語者
        last_speaker = diar_segments[0]["speaker"]  # 預設使用第一個語者
        
        for word in words:
            word_mid = (word["start"] + word["end"]) / 2
            word["speaker"] = None
            
            for diar in diar_segments:
                if diar["start"] <= word_mid <= diar["end"]:
                    word["speaker"] = diar["speaker"]
                    last_speaker = diar["speaker"]
                    break
            
            # 如果沒有找到匹配的區間，使用最後一個已知的語者
            if word["speaker"] is None:
                word["speaker"] = last_speaker
        
        # 按語者分組
        result = []
        current_speaker = words[0].get("speaker", "UNKNOWN")
        current_words = [words[0]]
        
        for word in words[1:]:
            word_speaker = word.get("speaker", "UNKNOWN")
            if word_speaker == current_speaker:
                current_words.append(word)
            else:
                # 結束當前片段
                result.append(self._words_to_segment(current_words, current_speaker))
                current_speaker = word_speaker
                current_words = [word]
        
        # 添加最後一個片段
        if current_words:
            result.append(self._words_to_segment(current_words, current_speaker))
        
        return result
    
    def _words_to_segment(self, words: List[Dict], speaker: str) -> Dict:
        """將 words 轉換為 segment"""
        text = "".join(w.get("word", "") for w in words)
        return {
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": text,
            "speaker": speaker,
            "words": words,
        }
    
    def _merge_same_speaker(self, segments: List[Dict]) -> List[Dict]:
        """合併相鄰的相同語者片段"""
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            if seg["speaker"] == current["speaker"]:
                # 合併
                current["end"] = seg["end"]
                current["text"] = current["text"] + seg["text"]
                if current.get("words") and seg.get("words"):
                    current["words"] = current["words"] + seg["words"]
            else:
                merged.append(current)
                current = seg.copy()
        
        merged.append(current)
        
        return merged
    
    def unload_model(self) -> None:
        """卸載模型並釋放 VRAM"""
        if not self.is_loaded:
            return
        
        print("🔄 正在卸載語者分離模型...")
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024**3
            except Exception:
                memory_before = 0
        else:
            memory_before = 0
        
        try:
            if self.pipeline is not None:
                self.pipeline.to(torch.device("cpu"))
                del self.pipeline
                self.pipeline = None
        except Exception as e:
            print(f"   ⚠️ 刪除模型時發生錯誤: {e}")
            self.pipeline = None
        
        self.is_loaded = False
        
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / 1024**3
                print(f"   已釋放 GPU 記憶體: {memory_before - memory_after:.2f} GB")
            except Exception:
                pass
        
        print("✅ 語者分離模型已卸載")


# 建立全域實例
diarizer = SpeakerDiarizer()
