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

# 導入用於詞級對齊的依賴
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torchaudio
    import numpy as np
    WAV2VEC2_AVAILABLE = True
except ImportError:
    WAV2VEC2_AVAILABLE = False
    print("Wav2Vec2 未安裝，詞級對齊功能將不可用")

# 嘗試導入 ProgressHook
try:
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    PROGRESS_HOOK_AVAILABLE = True
except ImportError:
    PROGRESS_HOOK_AVAILABLE = False
    print("ProgressHook 不可用，將使用備選進度追蹤方案")

from database import db_manager

# 載入環境變數
load_dotenv()

# 初始化繁簡轉換器
cc = OpenCC('s2twp')

# 添加 FFmpeg 路徑
ffmpeg_path = Path(__file__).parent.parent / "ffmpeg-7.1.1-full_build-shared" / "bin"
if ffmpeg_path.exists():
    os.environ["PATH"] += os.pathsep + str(ffmpeg_path)


class CustomProgressHook:
    """自訂進度追蹤 Hook"""

    def __init__(self, task_id: str, start_progress: float = 70.0, end_progress: float = 85.0):
        self.task_id = task_id
        self.start_progress = start_progress
        self.end_progress = end_progress
        self.progress_range = end_progress - start_progress

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, step_name: str, step_artefact=None, file=None, total=None, completed=None):
        """
        進度回調函數
        step_name: 當前步驟名稱
        completed: 已完成數量
        total: 總數量
        """
        if total is not None and completed is not None:
            # 計算當前步驟的進度
            step_progress = (completed / total) if total > 0 else 0
            current_progress = self.start_progress + (step_progress * self.progress_range)

            db_manager.update_task_status(
                self.task_id,
                'processing',
                progress=current_progress,
                current_stage=f'語者分離: {step_name} ({completed}/{total})'
            )
        else:
            # 無法取得詳細進度，顯示步驟名稱
            db_manager.update_task_status(
                self.task_id,
                'processing',
                progress=self.start_progress + (self.progress_range * 0.5),
                current_stage=f'語者分離: {step_name}'
            )


class TaskProcessor:
    """任務處理器類別"""

    def __init__(self):
        self.whisper_model = None
        self.current_model_name = None
        self.diarization_model = None
        self.align_model = None
        self.align_metadata = None
        self.current_language = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diarization_loaded = False
        self.align_loaded = False
        self.current_task_id = None
        self._cancelled = False
    
    def load_whisper_model(self, model_name: str ,compute_type: str = "float16"):
        """載入 Whisper 模型（確保只有一個模型實例）"""
        if self.whisper_model is not None and self.current_model_name == model_name:
            # 模型已載入且相同，無需重新載入
            return
        
        # 卸載舊模型
        self.unload_whisper_model()

        # 載入新模型
        print(f"正在載入 Whisper 模型: {model_name} (設備: {self.device})")
        print(f"使用 compute_type: {compute_type} 以優化記憶體使用")

        self.whisper_model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=compute_type,
            local_files_only=True
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

    def unload_diarization_model(self):
        """卸載語者分離模型以釋放 VRAM"""
        if not self.diarization_loaded:
            return

        print("正在卸載語者分離模型以釋放 VRAM...")
        if self.diarization_model is not None:
            del self.diarization_model
            self.diarization_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.diarization_loaded = False
        print("語者分離模型已卸載")

    def load_align_model(self, language_code: str):
        """載入對齊模型（用於詞級時間戳）"""
        if not WAV2VEC2_AVAILABLE:
            print("Wav2Vec2 未安裝，無法使用詞級對齊功能")
            return False

        if self.align_loaded and self.current_language == language_code:
            return True

        # 卸載舊的對齊模型
        self.unload_align_model()

        print(f"正在載入對齊模型，語言: {language_code}")
        try:
            # 根據語言選擇適合的 Wav2Vec2 模型
            model_mapping = {
                'zh': 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
                'en': 'facebook/wav2vec2-large-960h-lv60-self',
                'ja': 'jonatasgrosman/wav2vec2-large-xlsr-53-japanese',
                'ko': 'kresnik/wav2vec2-large-xlsr-korean',
                'fr': 'facebook/wav2vec2-large-xlsr-53-french',
                'de': 'facebook/wav2vec2-large-xlsr-53-german',
                'es': 'facebook/wav2vec2-large-xlsr-53-spanish',
            }

            model_name = model_mapping.get(language_code, 'facebook/wav2vec2-large-960h-lv60-self')

            self.align_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.align_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.align_model.to(self.device)
            self.align_model.eval()

            self.current_language = language_code
            self.align_loaded = True
            print(f"對齊模型載入完成: {model_name}")
            return True
        except Exception as e:
            print(f"載入對齊模型失敗: {e}")
            return False

    def unload_align_model(self):
        """卸載對齊模型以釋放 VRAM"""
        if not self.align_loaded:
            return

        print("正在卸載對齊模型以釋放 VRAM...")
        if self.align_model is not None:
            del self.align_model
            del self.align_processor
            self.align_model = None
            self.align_processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.align_loaded = False
        self.current_language = None
        print("對齊模型已卸載")

    def align_timestamps(self, segments: List[Dict], audio_path: str, language: str) -> Optional[List[Dict]]:
        """使用 wav2vec2 進行詞級對齊（優化版，減少記憶體使用）"""
        if not WAV2VEC2_AVAILABLE:
            return None

        try:
            # 載入對齊模型
            if not self.load_align_model(language):
                return None

            # 載入完整音訊（一次性載入，避免重複 I/O）
            waveform, sample_rate = torchaudio.load(audio_path)

            # 如果是立體聲，轉換為單聲道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 重採樣至 16kHz（Wav2Vec2 的標準採樣率）
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            aligned_segments = []

            # 逐個處理 segment（避免批次處理造成 OOM）
            for idx, segment in enumerate(segments):
                print(f"正在對齊 segment {idx+1}/{len(segments)}")

                # 提取片段時間範圍內的音訊
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                segment_audio = waveform[:, start_sample:end_sample].squeeze()

                # 限制音訊長度，超過 30 秒的片段跳過模型推理
                max_duration = 30.0  # 秒
                segment_duration = (end_sample - start_sample) / sample_rate

                if segment_duration > max_duration:
                    print(f"  片段過長 ({segment_duration:.1f}s)，使用簡化對齊")
                    words_with_timestamps = self._simple_word_alignment(
                        segment['text'],
                        segment['start'],
                        segment['end']
                    )
                else:
                    # 使用 Wav2Vec2 進行強制對齊
                    words_with_timestamps = self._align_segment_words(
                        segment_audio.cpu().numpy(),
                        segment['text'],
                        segment['start'],
                        sample_rate
                    )

                # 添加詞級時間戳到片段
                aligned_segment = segment.copy()
                if words_with_timestamps:
                    aligned_segment['words'] = words_with_timestamps

                aligned_segments.append(aligned_segment)

                # 每處理 5 個 segment 清理一次記憶體
                if (idx + 1) % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 釋放音訊記憶體
            del waveform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return aligned_segments
        except Exception as e:
            print(f"詞級對齊失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _simple_word_alignment(self, text: str, segment_start: float, segment_end: float) -> List[Dict]:
        """簡化的詞級對齊（基於字符比例，不使用模型推理）"""
        try:
            # 分詞（根據空格分割）
            words = text.strip().split()

            if not words:
                return []

            # 計算每個詞的字符數
            chars_per_word = [len(word) for word in words]
            total_chars = sum(chars_per_word)

            if total_chars == 0:
                return []

            # 片段總時長
            duration = segment_end - segment_start

            # 根據字符比例分配時間
            words_with_timestamps = []
            current_time = segment_start

            for i, word in enumerate(words):
                # 根據字符比例估算時間
                word_duration = (chars_per_word[i] / total_chars) * duration
                word_start = current_time
                word_end = current_time + word_duration

                words_with_timestamps.append({
                    'word': word,
                    'start': round(word_start, 3),
                    'end': round(word_end, 3),
                    'probability': 1.0  # 簡化版本沒有真實的置信度
                })

                current_time = word_end

            return words_with_timestamps

        except Exception as e:
            print(f"詞級對齊失敗: {e}")
            return []

    def unload_whisper_model(self):
        """卸載 Whisper 模型以釋放 VRAM"""
        if self.whisper_model is None:
            return

        print("正在卸載 Whisper 模型以釋放 VRAM...")
        del self.whisper_model
        self.whisper_model = None
        self.current_model_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Whisper 模型已卸載")
    
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
    
    def add_speaker_info_to_text(self, timestamp_texts, ann, confidence_map=None):
        """添加語者資訊到文字"""
        spk_text = []
        for i, (seg, text) in enumerate(timestamp_texts):
            spk = ann.crop(seg).argmax()
            confidence = confidence_map.get(i) if confidence_map else None
            spk_text.append((seg, spk, text, confidence))
        return spk_text

    def merge_cache(self, text_cache):
        """合併文字快取"""
        sentence = ''.join([item[2] for item in text_cache])
        spk = text_cache[0][1]
        start = round(text_cache[0][0].start, 1)
        end = round(text_cache[-1][0].end, 1)
        # 計算平均信心分數
        confidences = [item[3] for item in text_cache if item[3] is not None]
        avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
        return Segment(start, end), spk, sentence, avg_confidence

    def merge_sentence(self, spk_text):
        """合併句子"""
        merged_spk_text = []
        pre_spk = None
        text_cache = []
        for seg, spk, text, confidence in spk_text:
            if spk != pre_spk and len(text_cache) > 0:
                merged_spk_text.append(self.merge_cache(text_cache))
                text_cache = [(seg, spk, text, confidence)]
                pre_spk = spk
            elif spk == pre_spk and text == text_cache[-1][2]:
                continue
            else:
                text_cache.append((seg, spk, text, confidence))
                pre_spk = spk
        if len(text_cache) > 0:
            merged_spk_text.append(self.merge_cache(text_cache))
        return merged_spk_text

    def diarize_text(self, timestamp_texts, diarization_result, confidence_map=None):
        """語者分離與文字整合"""
        spk_text = self.add_speaker_info_to_text(timestamp_texts, diarization_result, confidence_map)
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
        model: str = 'CWTchen/Belle-whisper-large-v3-zh-punct-ct2-faster-whisper-float32',
        # 新增進階參數
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        enable_word_timestamps: bool = False,
        enable_confidence_score: bool = False
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
            vad_onset: VAD 語音檢測敏感度 (0-1)
            vad_offset: VAD 語音結束閾值
            min_speakers: 最小語者數
            max_speakers: 最大語者數
            enable_word_timestamps: 是否啟用詞級時間戳
            enable_confidence_score: 是否啟用信心分數輸出
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

            # 自動選擇 compute_type 以優化記憶體使用
            if self.device == "cuda":
                if "float32" in model.lower():
                    compute_type = "float32"
                    beam_size = 1
                elif "int8" in model.lower():
                    compute_type = "int8"
                    beam_size = 10
                else:
                    compute_type = "float16"
                    beam_size = 5
            else:
                # CPU 使用 int8 以減少 RAM 使用
                compute_type = "int8"
                beam_size = 1

            self.load_whisper_model(model, compute_type)

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
            
            # 卸載語者分離模型以節省 VRAM
            self.unload_diarization_model()

            # 語音轉文字
            db_manager.update_task_status(
                task_id, 
                'processing', 
                progress=30.0, 
                current_stage='語音轉文字 (ASR)'
            )
            
            # 使用優化參數以避免 OOM
            # 構建 VAD 參數（faster-whisper 使用的參數名稱）
            vad_parameters = {
                "threshold": vad_onset,  # 語音檢測敏感度（faster-whisper 使用 threshold）
                "min_speech_duration_ms": int((1 - vad_offset) * 1000),  # 最小語音持續時間
                "min_silence_duration_ms": 500
            }

            segments, info = self.whisper_model.transcribe(
                audio=str(converted_audio_path),
                language=language,
                task=task,
                beam_size=beam_size,  # 減少 beam size 以降低記憶體使用
                vad_filter=True,  # 啟用 VAD 過濾靜音片段
                vad_parameters=vad_parameters,  # 使用進階 VAD 參數
                word_timestamps=enable_word_timestamps,  # 根據參數決定是否啟用詞級時間戳
                log_progress=True,
            )
            
            timestamp_texts = []
            asr_lines = []
            partial_result = []
            confidence_map = {}  # 儲存每個 segment 的信心分數

            # 處理 segments（這是一個生成器）
            segment_count = 0
            segments_list = []  # 儲存所有 segment 供後續詞級對齊使用

            for segment in segments:
                # 檢查是否被取消
                if self.check_cancelled(task_id):
                    return

                segment_count += 1
                converted_text = cc.convert(segment.text)
                timestamp_texts.append((Segment(segment.start, segment.end), converted_text))

                # 計算信心分數百分比
                confidence_pct = None
                if enable_confidence_score and hasattr(segment, 'avg_logprob'):
                    # 將對數概率轉換為百分比
                    # avg_logprob 範圍約在 -1 到 0 之間，我們將其映射到 0-100%
                    # 使用 exp(avg_logprob) 轉換，然後乘以 100
                    import math
                    confidence_pct = round(math.exp(segment.avg_logprob) * 100, 1)
                    # 儲存到 confidence_map 供語者分離使用
                    confidence_map[len(timestamp_texts) - 1] = confidence_pct
                    asr_line = "[%6.2fs -> %6.2fs] (%.1f%%) %s" % (segment.start, segment.end, confidence_pct, converted_text)
                else:
                    asr_line = "[%6.2fs -> %6.2fs] %s" % (segment.start, segment.end, converted_text)

                asr_lines.append(asr_line)

                # 儲存 segment 資料
                seg_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': converted_text
                }

                # 添加信心分數（如果啟用）
                if confidence_pct is not None:
                    seg_data['confidence'] = confidence_pct

                # 添加詞級時間戳（如果有）
                if hasattr(segment, 'words') and segment.words:
                    # 將 Word 對象轉換為字典，確保可以 JSON 序列化
                    seg_data['words'] = [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        }
                        for word in segment.words
                    ]

                segments_list.append(seg_data)
                partial_result.append(seg_data)
                
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

            # 詞級對齊（如果啟用）
            if enable_word_timestamps and WAV2VEC2_AVAILABLE:
                # 使用 Wav2Vec2 進行詞級對齊
                detected_language = language if language else info.language

                db_manager.update_task_status(
                    task_id,
                    'processing',
                    progress=62.0,
                    current_stage='執行詞級對齊'
                )

                aligned_segments = self.align_timestamps(
                    segments_list,
                    str(converted_audio_path),
                    detected_language
                )

                if aligned_segments:
                    segments_list = aligned_segments
                    partial_result = aligned_segments
                    db_manager.update_task_result(task_id, "", partial_result)
                    print("詞級對齊完成")

                db_manager.update_task_status(
                    task_id,
                    'processing',
                    progress=65.0,
                    current_stage='詞級對齊完成'
                )
            
            # 語者分離（如果啟用）
            final_result = None
            diarization_success = False

            if enable_diarization:
                self.unload_whisper_model()

                try:
                    # 載入語者分離模型
                    db_manager.update_task_status(
                        task_id,
                        'processing',
                        progress=65.0,
                        current_stage='載入語者分離模型'
                    )
                    self.load_diarization_model()

                    db_manager.update_task_status(
                        task_id,
                        'processing',
                        progress=70.0,
                        current_stage='執行語者分離'
                    )

                    # 使用 Hook 追蹤語者分離進度
                    # 準備語者分離參數
                    diarization_params = {}
                    if min_speakers is not None:
                        diarization_params['min_speakers'] = min_speakers
                    if max_speakers is not None:
                        diarization_params['max_speakers'] = max_speakers

                    with CustomProgressHook(task_id, start_progress=70.0, end_progress=85.0) as hook:
                        diarization_output = self.diarization_model(
                            str(converted_audio_path),
                            hook=hook,
                            **diarization_params
                        )
                    diarization_result = diarization_output.speaker_diarization

                    db_manager.update_task_status(
                        task_id,
                        'processing',
                        progress=85.0,
                        current_stage='整合語者資訊'
                    )

                    final_result = self.diarize_text(timestamp_texts, diarization_result, confidence_map if enable_confidence_score else None)

                    # 更新部分結果（包含語者資訊）
                    partial_result = []
                    dialogue_lines = []
                    for segment, spk, sent, confidence in final_result:
                        result_dict = {
                            'start': segment.start,
                            'end': segment.end,
                            'speaker': spk,
                            'text': sent
                        }
                        if confidence is not None:
                            result_dict['confidence'] = confidence
                            formatted_line = f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] ({confidence:.1f}%) {spk}: {sent}"
                        else:
                            formatted_line = f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] {spk}: {sent}"

                        partial_result.append(result_dict)
                        dialogue_lines.append(formatted_line)

                    # 保存語者分離結果
                    dialogue_path = result_dir / "transcript_with_speakers.txt"
                    with open(dialogue_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(dialogue_lines))

                    diarization_success = True

                except Exception as e:
                    # 語者分離失敗，記錄錯誤並使用 ASR 結果
                    print(f"語者分離失敗: {str(e)}")
                    print("將僅返回 ASR 轉錄結果")
                    db_manager.update_task_status(
                        task_id,
                        'processing',
                        progress=65.0,
                        current_stage='語者分離失敗，使用 ASR 結果'
                    )

            # 如果沒有啟用語者分離或語者分離失敗，使用原始 ASR 結果
            if not diarization_success:
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

