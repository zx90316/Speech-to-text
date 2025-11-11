"""
任務處理器模組
將原有的 Whisper 轉錄邏輯改造為異步任務
"""
import os
import sys

# ============================================================================
# 重要：在導入任何第三方庫之前，必須先設置離線模式環境變量！
# 否則 pyannote、transformers 等庫在導入時就會嘗試連接網路
# ============================================================================

# 強制 Hugging Face Hub 使用離線模式（僅使用本地緩存）
# 這是為了符合服務器限制對外聯網的安全要求
# 所有模型應該預先下載到 ~/.cache/huggingface/ 中
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 禁用 telemetry 和自動更新檢查
os.environ['DO_NOT_TRACK'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# 移除代理設置，避免嘗試通過代理連接網路
# 某些環境變量可能已設置代理，在離線模式下需要清除
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                   'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

# ============================================================================
# 現在可以安全地導入其他模組
# ============================================================================

import time
import subprocess
import asyncio
from pathlib import Path
import faulthandler

# 啟用 faulthandler 以捕獲 segmentation fault 等嚴重錯誤
# 將錯誤信息輸出到 stderr，幫助診斷 C++ 擴展模組的崩潰問題
faulthandler.enable(file=sys.stderr, all_threads=True)

#https://github.com/tencent-ailab/SongPrep/issues/5#issuecomment-3478738144
# Add FFmpeg DLL directory for Windows (Python 3.8+)
ffmpeg_dll_dir = Path(__file__).parent.parent / "ffmpeg-master-latest-win64-gpl-shared" / "bin"
if ffmpeg_dll_dir.exists():
    os.add_dll_directory(str(ffmpeg_dll_dir))
    os.environ["PATH"] += os.pathsep + str(ffmpeg_dll_dir)

from typing import Optional, List, Dict, Any
from pyannote.core import Segment
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from opencc import OpenCC
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from dotenv import load_dotenv

from pyannote.audio.pipelines.utils.hook import ProgressHook

from memory_storage import memory_manager
from email_service import email_service
from ollama_service import ollama_service

# 載入環境變數
load_dotenv()

# 初始化繁簡轉換器
cc = OpenCC('s2twp')

# 注意：本檔案中所有的 memory_manager 都應替換為 memory_manager

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

            memory_manager.update_task_status(
                self.task_id,
                'processing',
                progress=current_progress,
                current_stage=f'語者分離: {step_name} ({completed}/{total})'
            )
        else:
            # 無法取得詳細進度，顯示步驟名稱
            memory_manager.update_task_status(
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
    
    def load_whisper_model(self, model_name: str ,compute_type: str = "default"):
        """載入 Whisper 模型（僅使用本地緩存）"""
        if self.whisper_model is not None and self.current_model_name == model_name:
            # 模型已載入且相同，無需重新載入
            return
        
        # 卸載舊模型
        self.unload_model()

        # 載入新模型（從本地緩存）
        print(f"正在載入 Whisper 模型: {model_name} (設備: {self.device}) compute_type: {compute_type}")

        self.whisper_model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=compute_type,
            local_files_only=True  # 僅使用本地緩存，避免網路請求
        )
        self.current_model_name = model_name
        print(f"Whisper 模型載入完成: {model_name}")

    def load_diarization_model(self):
        """載入語者分離模型"""
        if self.diarization_loaded:
            return

        print("正在載入語者分離模型（從本地緩存）...")
        diarization_model_id = "pyannote/speaker-diarization-community-1"
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.diarization_model = Pipeline.from_pretrained(
            diarization_model_id,
            token=hf_token
        )
        self.diarization_model.to(torch.device(self.device))
        print("語者分離模型載入完成")
        self.diarization_loaded = True

    def unload_model(self):
        """卸載所有模型以釋放 VRAM（Windows 安全版本）"""
        import gc
        import sys

        if not self.diarization_loaded and self.whisper_model is None:
            return

        print("正在卸載所有模型以釋放 VRAM...")

        # 卸載 diarization 模型
        if self.diarization_model is not None:
            try:
                print("  - 卸載語者分離模型...")
                # 先移到 CPU 避免 CUDA 鎖死
                try:
                    self.diarization_model.to(torch.device("cpu"))
                except:
                    pass  # 如果已經在 CPU 上就忽略

                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except:
                        pass  # 忽略同步錯誤

                # 使用 weakref 讓 Python GC 自動處理，避免 fatal error
                try:
                    del self.diarization_model
                except:
                    pass

                print("  - 語者分離模型已卸載")
            except Exception as e:
                print(f"  - 卸載語者分離模型時發生錯誤: {e}")
            finally:
                self.diarization_model = None

        # 卸載 Whisper 模型 - Windows 安全方式
        if self.whisper_model is not None:
            try:
                print("  - 準備卸載 Whisper 模型...")

                # 確保所有 CUDA 操作完成
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        memory_before = torch.cuda.memory_allocated() / 1024**3
                        print(f"  - 卸載前 GPU 記憶體: {memory_before:.2f} GB")
                    except:
                        pass  # 忽略記憶體查詢錯誤

                # 方法1: 直接設為 None，讓 GC 處理（最安全）
                # 避免使用 del，因為 CTranslate2 的 C++ 解構函數可能在 Windows 上有問題
                self.whisper_model = None
                self.current_model_name = None

                # 立即觸發垃圾回收（多次以確保清理）
                gc.collect()
                gc.collect()
                gc.collect()

                # 清理 CUDA 緩存
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        memory_after = torch.cuda.memory_allocated() / 1024**3
                        print(f"  - 卸載後 GPU 記憶體: {memory_after:.2f} GB")
                    except:
                        pass  # 忽略 CUDA 操作錯誤

                print("  - Whisper 模型已成功卸載")

            except Exception as e:
                print(f"  - 卸載 Whisper 模型時發生錯誤: {type(e).__name__}: {e}")
                # 不要 print stack trace，避免觸發更多錯誤
            finally:
                self.whisper_model = None
                self.current_model_name = None

        # 最終清理（多次 GC 確保釋放）
        try:
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass  # 忽略最終清理錯誤

        self.diarization_loaded = False
        print("所有模型已卸載")
        sys.stdout.flush()
    
    def unload_diarization_model(self):
        """只卸載語者分離模型以釋放 VRAM（保留 Whisper 模型）"""
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
    
    def check_cancelled(self, task_id: str) -> bool:
        """檢查任務是否被取消"""
        task = memory_manager.get_task(task_id)
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
        for i, (seg, text, words) in enumerate(timestamp_texts):
            spk = ann.crop(seg).argmax()
            confidence = confidence_map.get(i) if confidence_map else None
            spk_text.append((seg, spk, text, confidence, words))
        return spk_text

    def merge_cache(self, text_cache):
        """合併文字快取"""
        sentence = ''.join([item[2] for item in text_cache])
        spk = text_cache[0][1]
        start = round(text_cache[0][0].start, 1)
        end = round(text_cache[-1][0].end, 1)
        # 計算平均信心分數
        confidences = [item[3] for item in text_cache if item[3] is not None]
        words = []
        for item in text_cache:
            if item[4] is not None:
                # Assuming item[4] is a single Word object, or an iterable of Word objects
                if isinstance(item[4], list):
                    words.extend(item[4])
                else:
                    words.append(item[4])
        avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
        return Segment(start, end), spk, sentence, avg_confidence, words

    def merge_sentence(self, spk_text):
        """合併句子"""
        merged_spk_text = []
        pre_spk = None
        text_cache = []
        for seg, spk, text, confidence,words in spk_text:
            if spk != pre_spk and len(text_cache) > 0:
                merged_spk_text.append(self.merge_cache(text_cache))
                text_cache = [(seg, spk, text, confidence,words)]
                pre_spk = spk
            elif spk == pre_spk and text == text_cache[-1][2]:
                continue
            else:
                text_cache.append((seg, spk, text, confidence,words))
                pre_spk = spk
        if len(text_cache) > 0:
            merged_spk_text.append(self.merge_cache(text_cache))
        return merged_spk_text

    def generate_confidence_html(self, segments: List[Dict], output_path: str, enable_diarization: bool = False):
        """
        生成詞級信心度視覺化 HTML 檔案
        
        Args:
            segments: 包含詞級時間戳和信心度的片段列表
            output_path: 輸出 HTML 檔案路徑
            enable_diarization: 是否包含語者資訊
        """
        def confidence_to_color(confidence: float) -> str:
            """
            將信心度轉換為顏色（0-1）
            高信心度 -> 綠色
            中等信心度 -> 黃色
            低信心度 -> 紅色
            """
            if confidence >= 0.8:
                # 高信心度：綠色
                r = int((1 - confidence) * 255 * 5)  # 0-51
                g = 200
                b = 100
            elif confidence >= 0.5:
                # 中等信心度：黃色到綠色
                ratio = (confidence - 0.5) / 0.3
                r = int(255 - ratio * 55)
                g = int(150 + ratio * 50)
                b = 50
            else:
                # 低信心度：紅色到黃色
                ratio = confidence / 0.5
                r = 255
                g = int(ratio * 150)
                b = 50
            
            return f"rgb({r}, {g}, {b})"
        
        html_content = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>詞級信心度視覺化</title>
    <style>
        body {
            font-family: 'Microsoft JhengHei', 'PingFang TC', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .legend {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .legend h3 {
            margin-top: 0;
            color: #333;
        }
        .legend-items {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .legend-color {
            width: 40px;
            height: 20px;
            border-radius: 3px;
        }
        .segment {
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .segment-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        .timestamp {
            font-family: 'Courier New', monospace;
            color: #666;
            font-size: 14px;
            background: #f8f8f8;
            padding: 5px 10px;
            border-radius: 4px;
        }
        .speaker {
            font-weight: bold;
            color: #667eea;
            background: #e8ebff;
            padding: 5px 12px;
            border-radius: 4px;
        }
        .confidence-badge {
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            background: #f0f0f0;
            color: #666;
        }
        .content {
            font-size: 16px;
        }
        .word {
            border-radius: 4px;
            display: inline-block;
            transition: all 0.2s;
            cursor: pointer;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        .word:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .word-tooltip {
            position: relative;
        }
        .word-tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            white-space: nowrap;
            font-size: 13px;
            z-index: 1000;
            margin-bottom: 5px;
        }
        .stats {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats h3 {
            margin-top: 0;
            color: #333;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-item {
            background: #f8f8f8;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="legend">
        <h3>📊 信心度圖例</h3>
        <div class="legend-items">
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgb(145, 200, 100);"></div>
                <span>高信心度 (80-100%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgb(255, 180, 50);"></div>
                <span>中等信心度 (50-80%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgb(255, 75, 50);"></div>
                <span>低信心度 (0-50%)</span>
            </div>
        </div>
    </div>
"""
        
        # 統計資料
        total_words = 0
        total_segments = len(segments)
        confidence_sum = 0
        confidence_count = 0
        low_confidence_words = 0
        
        # 生成每個片段的內容
        for idx, segment in enumerate(segments):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '')
            speaker = segment.get('speaker', '')
            segment_confidence = segment.get('confidence', None)
            words = segment.get('words', None)
            
            html_content += f'    <div class="segment">\n'
            html_content += f'        <div class="segment-header">\n'
            html_content += f'            <span class="timestamp">{start_time:.2f}s → {end_time:.2f}s</span>\n'
            
            if enable_diarization and speaker:
                html_content += f'            <span class="speaker">{speaker}</span>\n'
            
            if segment_confidence is not None:
                html_content += f'            <span class="confidence-badge">片段信心度: {segment_confidence:.1f}%</span>\n'
            
            html_content += f'        </div>\n'
            html_content += f'        <div class="content">\n'
            
            # 如果有詞級時間戳
            if words:
                for word_info in words:
                    # word_info 現在是字典而不是 Word 物件
                    if isinstance(word_info, dict):
                        word_text = cc.convert(word_info['word'])
                        word_start = word_info['start']
                        word_end = word_info['end']
                        probability = word_info['probability']
                    else:
                        # 向後兼容：如果是 Word 物件
                        word_text = cc.convert(word_info.word)
                        word_start = word_info.start
                        word_end = word_info.end
                        probability = word_info.probability
                    
                    # 轉換為百分比
                    confidence_pct = probability * 100
                    color = confidence_to_color(probability)
                    
                    tooltip = f"信心度: {confidence_pct:.1f}% | 時間: {word_start:.2f}s-{word_end:.2f}s"
                    
                    html_content += f'            <span class="word word-tooltip" style="color: {color};" data-tooltip="{tooltip}">{word_text}</span>\n'
                    
                    total_words += 1
                    confidence_sum += probability
                    confidence_count += 1
                    
                    if probability < 0.5:
                        low_confidence_words += 1
            else:
                # 沒有詞級時間戳，顯示整個片段
                if segment_confidence is not None:
                    confidence = segment_confidence / 100.0
                    color = confidence_to_color(confidence)
                    tooltip = f"信心度: {segment_confidence:.1f}%"
                    html_content += f'            <span class="word word-tooltip" style="background-color: {color};" data-tooltip="{tooltip}">{text}</span>\n'
                    
                    confidence_sum += confidence
                    confidence_count += 1
                else:
                    # 沒有信心度資訊
                    html_content += f'            <span>{text}</span>\n'
            
            html_content += f'        </div>\n'
            html_content += f'    </div>\n\n'
        
        # 計算平均信心度
        avg_confidence = (confidence_sum / confidence_count * 100) if confidence_count > 0 else 0
        
        # 添加統計資訊
        html_content += f"""
    <div class="stats">
        <h3>📈 統計資訊</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{total_segments}</div>
                <div class="stat-label">總片段數</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_words}</div>
                <div class="stat-label">總詞數</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{avg_confidence:.1f}%</div>
                <div class="stat-label">平均信心度</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{low_confidence_words}</div>
                <div class="stat-label">低信心度詞數 (&lt;50%)</div>
            </div>
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # 寫入檔案
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ 信心度視覺化 HTML 已生成: {output_path}")

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
        model: str = 'CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32',
        # 新增進階參數
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        enable_confidence_score: bool = False,
        compute_type: Optional[str] = None,
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
            enable_confidence_score: 是否啟用信心分數輸出,
            compute_type: 計算類型 (float32, int8, float16)
        """
        try:
            # 檢查是否被取消
            if self.check_cancelled(task_id):
                return

            # 載入 Whisper 模型
            memory_manager.update_task_status(
                task_id,
                'processing',
                progress=0.0,
                current_stage='載入 Whisper 模型'
            )

            if compute_type == "float32":
                beam_size = 1
            elif compute_type == "int8":
                beam_size = 10
            elif compute_type == "float16":
                beam_size = 5
            else:
                beam_size = 5

            self.load_whisper_model(model, compute_type)

            # 檢查是否被取消
            if self.check_cancelled(task_id):
                return

            # 創建結果資料夾
            result_dir = Path(__file__).parent / "result" / task_id
            result_dir.mkdir(parents=True, exist_ok=True)

            # 轉換音訊格式
            memory_manager.update_task_status(
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
            
            # 卸載語者分離模型以節省 VRAM（保留 Whisper 模型）
            self.unload_diarization_model()

            # 語音轉文字
            memory_manager.update_task_status(
                task_id,
                'processing',
                progress=30.0,
                current_stage='語音轉文字 (ASR)',
                asr_progress=None
            )

            # 使用優化參數以避免 OOM
            # 構建 VAD 參數（faster-whisper 使用的參數名稱）
            vad_parameters = {
                "threshold": vad_onset,  # 語音檢測敏感度（faster-whisper 使用 threshold）
                "min_speech_duration_ms": int((1 - vad_offset) * 1000),  # 最小語音持續時間
                "min_silence_duration_ms": 500
            }

            print(f"📝 word_timestamps 設定: {enable_confidence_score == 1}")

            word_timestamps = enable_confidence_score == 1

            # 嘗試轉錄，如果失敗則自動降級到更兼容的計算類型
            try:
                segments, info = self.whisper_model.transcribe(
                    audio=str(converted_audio_path),
                    language=language,
                    task=task,
                    beam_size=beam_size,  # 減少 beam size 以降低記憶體使用
                    vad_filter=True,  # 啟用 VAD 過濾靜音片段
                    vad_parameters=vad_parameters,  # 使用進階 VAD 參數
                    word_timestamps=word_timestamps,  # True 即可啟用詞級時間戳
                    log_progress=True,
                )
            except Exception as e:
                error_msg = str(e)
                # 檢查是否為 cuBLAS 不支持的錯誤（int8 在某些 GPU 上不支持）
                if "cuBLAS" in error_msg or "CUBLAS_STATUS_NOT_SUPPORTED" in error_msg:
                    print(f"⚠️ {compute_type} 計算類型不支持，自動切換到 float16...")

                    # 重新載入模型並使用 float16
                    fallback_compute_type = "float16"
                    fallback_beam_size = 5
                    self.load_whisper_model(model, fallback_compute_type)

                    memory_manager.update_task_status(
                        task_id,
                        'processing',
                        progress=30.0,
                        current_stage=f'使用語音辨識 (已切換到 {fallback_compute_type})'
                    )

                    # 重試轉錄
                    segments, info = self.whisper_model.transcribe(
                        audio=str(converted_audio_path),
                        language=language,
                        task=task,
                        beam_size=fallback_beam_size,
                        vad_filter=True,
                        vad_parameters=vad_parameters,
                        word_timestamps=word_timestamps,
                        log_progress=True,
                    )
                else:
                    # 其他錯誤直接拋出
                    raise

            # 獲取音訊總時長用於進度計算
            audio_duration = info.duration if hasattr(info, 'duration') else None
            print(f"🎵 音訊總時長: {audio_duration:.2f}s" if audio_duration else "🎵 音訊總時長: 未知")
            
            timestamp_texts = []
            asr_lines = []
            partial_result = []
            confidence_map = {}  # 儲存每個 segment 的信心分數

            # 處理 segments（這是一個生成器）
            segment_count = 0
            segments_list = []  # 儲存所有 segment 供後續詞級對齊使用
            last_processed_time = 0.0  # 追蹤已處理的音訊時間

            for segment in segments:
                # 檢查是否被取消
                if self.check_cancelled(task_id):
                    return

                segment_count += 1
                converted_text = cc.convert(segment.text)
                timestamp_texts.append((Segment(segment.start, segment.end), converted_text ,segment.words))

                # 更新已處理的音訊時間
                last_processed_time = segment.end

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
                has_words = hasattr(segment, 'words') and segment.words
                if has_words:
                    # 將 Word 對象轉換為字典，確保可以 JSON 序列化
                    words_list = list(segment.words) if segment.words else []
                    seg_data['words'] = [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        }
                        for word in words_list
                    ]

                segments_list.append(seg_data)
                partial_result.append(seg_data)

                # 計算基於時間的進度百分比
                time_progress_pct = 0
                if audio_duration and audio_duration > 0:
                    time_progress_pct = min((last_processed_time / audio_duration) * 100, 100.0)

                # 每處理 5 個片段更新一次進度
                if segment_count % 1 == 0:
                    # 進度從 30% 到 55%，基於已處理的音訊時間
                    current_progress = min(30.0 + (time_progress_pct * 0.25), 55.0)

                    # 構建詳細的 ASR 進度信息
                    asr_progress_info = {
                        'processed_time': round(last_processed_time, 2),
                        'total_time': round(audio_duration, 2) if audio_duration else None,
                        'segment_count': segment_count,
                        'time_progress_pct': round(time_progress_pct, 1)
                    }

                    memory_manager.update_task_status(
                        task_id,
                        'processing',
                        progress=current_progress,
                        current_stage=f'語音轉文字 (ASR)',
                        asr_progress=asr_progress_info
                    )
                    memory_manager.update_task_result(task_id, partial_result)
            
            memory_manager.update_task_status(
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
            diarization_success = False

            if enable_diarization:
                self.unload_model()

                try:
                    # 載入語者分離模型
                    memory_manager.update_task_status(
                        task_id,
                        'processing',
                        progress=65.0,
                        current_stage='載入語者分離模型'
                    )
                    self.load_diarization_model()

                    memory_manager.update_task_status(
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

                    memory_manager.update_task_status(
                        task_id,
                        'processing',
                        progress=85.0,
                        current_stage='整合語者資訊'
                    )

                    final_result = self.diarize_text(timestamp_texts, diarization_result, confidence_map if enable_confidence_score else None)

                    # 更新部分結果（包含語者資訊）
                    partial_result = []
                    dialogue_lines = []
                    for segment, spk, sent, confidence, words in final_result:
                        result_dict = {
                            'start': segment.start,
                            'end': segment.end,
                            'speaker': spk,
                            'text': sent
                        }
                        
                        # 轉換 Word 物件為可 JSON 序列化的字典
                        if words:
                            result_dict['words'] = [
                                {
                                    'word': word.word,
                                    'start': word.start,
                                    'end': word.end,
                                    'probability': word.probability
                                }
                                for word in words
                            ]
                        
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
                    memory_manager.update_task_status(
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
            
            # 生成信心度視覺化 HTML（如果有信心度資料）
            confidence_html_path = None
            if enable_confidence_score:
                try:
                    html_path = result_dir / "confidence_visualization.html"
                    self.generate_confidence_html(
                        segments=partial_result,
                        output_path=str(html_path),
                        enable_diarization=diarization_success
                    )
                    confidence_html_path = str(html_path)
                except Exception as e:
                    print(f"⚠️ 警告：生成信心度 HTML 失敗: {str(e)}")
                    # 不影響主任務，繼續執行
            
            # 獲取任務資訊以取得郵箱和檔名
            task = memory_manager.get_task_full(task_id)
            if task:
                # 讀取轉錄結果文字
                transcript_path = result_dir / "transcript_with_speakers.txt" if diarization_success else result_dir / "transcript.txt"
                if not transcript_path.exists():
                    transcript_path = result_dir / "transcript.txt"

                transcript_text = ""
                if transcript_path.exists():
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript_text = f.read()

                # LLM 校對（如果啟用）
                corrected_text = None
                llm_comparison_html_path = None
                if task.get('enable_llm_correction', False):
                    try:
                        memory_manager.update_task_status(
                            task_id,
                            'completed',
                            progress=95.0,
                            current_stage='LLM 文本校對中'
                        )
                        print(f"🤖 開始 LLM 校對，使用模型: {task.get('llm_model', 'gemma3:4b')}")

                        # 執行校對
                        def llm_progress(msg):
                            # 任務完成 - 準備發送郵件
                            memory_manager.update_task_status(
                                task_id,
                                'processing',
                                progress=96.0,
                                current_stage=msg
                            )
                            print(f"  LLM: {msg}")

                        correction_result = ollama_service.correct_text(
                            text=transcript_text,
                            model=task.get('llm_model', 'gemma3:4b'),
                            has_diarization=diarization_success,
                            progress_callback=llm_progress
                        )

                        corrected_text = correction_result.get('corrected', transcript_text)

                        # 生成校對對比 HTML
                        llm_html_path = result_dir / "llm_correction_comparison.html"
                        llm_html_content = ollama_service.generate_comparison_html(
                            original=transcript_text,
                            corrected=corrected_text,
                            has_diarization=diarization_success
                        )
                        with open(llm_html_path, 'w', encoding='utf-8') as f:
                            f.write(llm_html_content)
                        llm_comparison_html_path = str(llm_html_path)

                        # 保存校正後的文本
                        corrected_path = result_dir / "transcript_corrected.txt"
                        with open(corrected_path, 'w', encoding='utf-8') as f:
                            f.write(corrected_text)

                        print(f"✅ LLM 校對完成")
                        # 任務完成 - 準備發送郵件
                        memory_manager.update_task_status(
                            task_id,
                            'processing',
                            progress=97.0,
                            current_stage='✅ LLM 校對完成'
                        )

                    except Exception as llm_error:
                        print(f"⚠️ LLM 校對失敗: {llm_error}")
                        corrected_text = None  # 失敗時不使用校正版本

                # 任務完成 - 準備發送郵件
                memory_manager.update_task_status(
                    task_id,
                    'completed',
                    progress=100.0,
                    current_stage='發送結果郵件'
                )
                memory_manager.update_task_result(task_id, partial_result)
                # 發送完成通知郵件
                try:
                    email_success = email_service.send_completion_email(
                        to_email=task['email'],
                        task_id=task_id,
                        filename=task['filename'],
                        transcript_text=transcript_text,
                        corrected_text=corrected_text,
                        has_diarization=diarization_success,
                        confidence_html_path=confidence_html_path,
                        llm_comparison_html_path=llm_comparison_html_path
                    )

                    if email_success:
                        print(f"✅ 任務 {task_id} 完成，結果已發送至 {task['email']}")
                    else:
                        print(f"⚠️ 任務 {task_id} 完成，但郵件發送失敗")

                except Exception as email_error:
                    print(f"⚠️ 發送郵件時發生錯誤: {email_error}")

                # 無論郵件是否成功，都清理臨時檔案
                try:
                    memory_manager.cleanup_task_files(task_id)
                    print(f"🗑️ 任務 {task_id} 的臨時檔案已清理")
                except Exception as cleanup_error:
                    print(f"⚠️ 清理檔案時發生錯誤: {cleanup_error}")

        except Exception as e:
            error_msg = str(e)
            print(f"任務 {task_id} 處理失敗: {error_msg}")
            memory_manager.update_task_status(
                task_id,
                'failed',
                error_message=error_msg
            )


# 全局任務處理器實例
task_processor = TaskProcessor()

