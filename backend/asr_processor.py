# -*- coding: utf-8 -*-
"""
Qwen3-ASR 處理器模組
封裝 Qwen3-ASR 模型的載入、轉錄和卸載邏輯
"""
import gc
from typing import Optional, List, Dict, Any

import torch

from config import (
    DEFAULT_QWEN_MODEL,
    FORCED_ALIGNER_MODEL,
    QWEN_ASR_MODELS,
)

# 嘗試導入 qwen_asr
try:
    from qwen_asr import Qwen3ASRModel
    QWEN_ASR_AVAILABLE = True
except ImportError:
    QWEN_ASR_AVAILABLE = False
    Qwen3ASRModel = None


class QwenASRProcessor:
    """Qwen ASR 處理器類別"""
    
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.enable_timestamps = False
        self.max_new_tokens = 256
    
    @staticmethod
    def is_available() -> bool:
        """檢查 Qwen ASR 套件是否可用"""
        return QWEN_ASR_AVAILABLE
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """取得可用模型列表"""
        return QWEN_ASR_MODELS
    
    def load_model(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        enable_timestamps: bool = False,
        max_new_tokens: int = 512,
        max_inference_batch_size: int = 32,
    ) -> None:
        """
        載入 Qwen ASR 模型
        
        Args:
            model_name: 模型名稱
            enable_timestamps: 是否啟用時間戳輸出
            max_new_tokens: 最大生成 token 數量
            max_inference_batch_size: 批次推論大小限制
        """
        if not QWEN_ASR_AVAILABLE:
            raise RuntimeError("qwen-asr 套件未安裝。請執行: pip install -U qwen-asr")
        
        # 如果已載入相同模型且設定相同，則跳過
        if (self.model is not None and 
            self.current_model_name == model_name and
            self.enable_timestamps == enable_timestamps):
            print(f"✓ Qwen ASR 模型已載入: {model_name}")
            return
        
        # 卸載舊模型
        self.unload_model()
        
        print(f"🔄 正在載入 Qwen ASR 模型: {model_name}")
        print(f"   設備: {self.device}")
        print(f"   時間戳: {'啟用' if enable_timestamps else '停用'}")
        
        model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": self.device,
            "max_inference_batch_size": max_inference_batch_size,
            "max_new_tokens": max_new_tokens,
        }
        
        if enable_timestamps:
            print(f"   載入時間戳對齊模型: {FORCED_ALIGNER_MODEL}")
            model_kwargs["forced_aligner"] = FORCED_ALIGNER_MODEL
            model_kwargs["forced_aligner_kwargs"] = {
                "dtype": torch.bfloat16,
                "device_map": self.device,
            }
        
        self.model = Qwen3ASRModel.from_pretrained(model_name, **model_kwargs)
        self.current_model_name = model_name
        self.enable_timestamps = enable_timestamps
        self.max_new_tokens = max_new_tokens
        
        print(f"✅ Qwen ASR 模型載入完成")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        執行語音轉錄
        
        Args:
            audio_path: 音訊檔案路徑
            language: 語言 (如 "Chinese", "English")，None 為自動偵測
            return_timestamps: 是否返回時間戳
            
        Returns:
            {
                "text": str,
                "language": str,
                "timestamps": list,
                "segments": list,
            }
        """
        if self.model is None:
            raise RuntimeError("模型尚未載入。請先呼叫 load_model()")
        
        print(f"📝 開始轉錄: {audio_path}")
        print(f"   語言: {language if language else '自動偵測'}")
        
        results = self.model.transcribe(
            audio=audio_path,
            language=language,
            return_time_stamps=return_timestamps and self.enable_timestamps,
        )
        
        result = results[0]
        
        output = {
            "text": result.text,
            "language": result.language,
            "timestamps": [],
            "segments": [],
        }
        
        # 處理時間戳
        if return_timestamps and self.enable_timestamps and hasattr(result, 'time_stamps') and result.time_stamps:
            output["timestamps"] = result.time_stamps
            output["segments"] = self._convert_to_segments(result.text, result.time_stamps)
        else:
            output["segments"] = [{
                "start": 0.0,
                "end": 0.0,
                "text": result.text,
                "words": None,
            }]
        
        print(f"✅ 轉錄完成")
        print(f"   語言: {output['language']}")
        print(f"   文字長度: {len(output['text'])} 字元")
        
        return output
    
    def _convert_to_segments(
        self,
        text: str,
        time_stamps: List[Dict]
    ) -> List[Dict]:
        """將時間戳轉換為 segments 格式"""
        if not time_stamps:
            return [{"start": 0.0, "end": 0.0, "text": text, "words": None}]
        
        words = []
        for ts in time_stamps:
            # 處理 ForcedAlignItem 物件（有 start_time, end_time 屬性）
            if hasattr(ts, "start_time"):
                words.append({
                    "word": getattr(ts, "text", ""),
                    "start": getattr(ts, "start_time", 0.0),
                    "end": getattr(ts, "end_time", 0.0),
                    "probability": getattr(ts, "confidence", 1.0),
                })
            # 處理字典格式（有 start, end 鍵）
            elif isinstance(ts, dict):
                words.append({
                    "word": ts.get("text", ts.get("word", "")),
                    "start": ts.get("start", ts.get("start_time", 0.0)),
                    "end": ts.get("end", ts.get("end_time", 0.0)),
                    "probability": ts.get("confidence", 1.0),
                })
            # 處理其他物件格式
            elif hasattr(ts, "text"):
                words.append({
                    "word": ts.text,
                    "start": getattr(ts, "start", getattr(ts, "start_time", 0.0)),
                    "end": getattr(ts, "end", getattr(ts, "end_time", 0.0)),
                    "probability": getattr(ts, "confidence", 1.0),
                })
        
        if words:
            return [{
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": text,
                "words": words,
            }]
        
        return [{"start": 0.0, "end": 0.0, "text": text, "words": None}]
    
    def unload_model(self) -> None:
        """卸載模型並釋放 VRAM"""
        if self.model is None:
            return
        
        print("🔄 正在卸載 Qwen ASR 模型...")
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024**3
            except Exception:
                memory_before = 0
        else:
            memory_before = 0
        
        try:
            del self.model
            self.model = None
            self.current_model_name = None
        except Exception as e:
            print(f"   ⚠️ 刪除模型時發生錯誤: {e}")
            self.model = None
            self.current_model_name = None
        
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
        
        print("✅ Qwen ASR 模型已卸載")


# 建立全域實例
qwen_processor = QwenASRProcessor()
