"""
音訊預處理模組
提供多種音訊增強與預處理功能
"""
import os
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class PreprocessConfig:
    """預處理配置"""
    # 降噪
    enable_denoise: bool = False
    denoise_strength: float = 0.5  # 0.0-1.0

    # 音量正規化
    enable_normalize: bool = False
    normalize_type: str = "peak"  # peak, lufs
    target_level: float = -3.0  # dB for peak, -23.0 LUFS for lufs

    # 靜音移除
    enable_silence_removal: bool = False
    silence_threshold: float = -50.0  # dB (更低的閾值，只移除真正的靜音)
    min_silence_duration: float = 1.0  # seconds (更長的時間，避免誤刪短暫停頓)

    # 人聲分離
    enable_vocal_separation: bool = False

    # 人聲增強
    enable_vocal_enhancement: bool = False
    enhancement_strength: float = 0.5  # 0.0-1.0

    # 迴聲消除
    enable_echo_removal: bool = False

    # 頻率均衡
    enable_eq: bool = False
    eq_low_gain: float = 0.0  # dB, -12 to 12
    eq_mid_gain: float = 0.0  # dB, -12 to 12
    eq_high_gain: float = 0.0  # dB, -12 to 12

    # 速度調整
    enable_speed_change: bool = False
    speed_factor: float = 1.0  # 0.5-2.0

    # 音調調整
    enable_pitch_shift: bool = False
    pitch_semitones: int = 0  # -12 to 12

    # 取樣率轉換
    enable_resample: bool = False
    target_sample_rate: int = 16000  # 16000, 44100, 48000

    # 立體聲轉單聲道
    enable_mono: bool = True

    # 動態範圍壓縮
    enable_compression: bool = False
    compression_ratio: float = 4.0  # 1.0-20.0
    compression_threshold: float = -20.0  # dB

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessConfig':
        """從字典創建配置"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            k: getattr(self, k)
            for k in self.__annotations__
        }


class AudioPreprocessor:
    """音訊預處理器"""

    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """查找 FFmpeg 可執行檔"""
        # 嘗試本地 FFmpeg
        local_ffmpeg = Path(__file__).parent.parent / "ffmpeg-7.1.1-full_build-shared" / "bin" / "ffmpeg.exe"
        if local_ffmpeg.exists():
            return str(local_ffmpeg)
        return "ffmpeg"

    def preprocess(
        self,
        input_path: str,
        output_path: str,
        config: PreprocessConfig,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        執行音訊預處理

        Args:
            input_path: 輸入音訊路徑
            output_path: 輸出音訊路徑
            config: 預處理配置
            progress_callback: 進度回調函數

        Returns:
            處理結果資訊
        """
        try:
            # 構建 FFmpeg 濾鏡鏈
            filters = []

            # 1. 降噪 (使用 afftdn - FFT Denoising)
            if config.enable_denoise:
                noise_reduction = int(config.denoise_strength * 97)  # 0-97
                filters.append(f"afftdn=nr={noise_reduction}:nf=-25")
                if progress_callback:
                    progress_callback(10, "applying_denoise")

            # 2. 高通濾波器去除低頻噪音
            if config.enable_denoise:
                filters.append("highpass=f=200")

            # 3. 迴聲消除 (使用 adeclick 去除點擊聲)
            if config.enable_echo_removal:
                # adeclick 參數 t 範圍是 1-100 (毫秒)
                filters.append("adeclick=t=2:w=85")
                filters.append("adeclip")
                if progress_callback:
                    progress_callback(20, "removing_echo")

            # 4. 頻率均衡器
            if config.enable_eq:
                # 低頻 (80-300Hz), 中頻 (300-3000Hz), 高頻 (3000-8000Hz)
                eq_filters = []
                if config.eq_low_gain != 0:
                    eq_filters.append(f"equalizer=f=150:width_type=o:width=2:g={config.eq_low_gain}")
                if config.eq_mid_gain != 0:
                    eq_filters.append(f"equalizer=f=1000:width_type=o:width=2:g={config.eq_mid_gain}")
                if config.eq_high_gain != 0:
                    eq_filters.append(f"equalizer=f=5000:width_type=o:width=2:g={config.eq_high_gain}")
                filters.extend(eq_filters)
                if progress_callback:
                    progress_callback(30, "applying_eq")

            # 5. 人聲增強 (增強 300-3000Hz 語音頻段)
            if config.enable_vocal_enhancement:
                gain = config.enhancement_strength * 6  # 0-6 dB
                filters.append(f"equalizer=f=1500:width_type=o:width=2:g={gain}")
                if progress_callback:
                    progress_callback(40, "enhancing_vocal")

            # 6. 動態範圍壓縮
            if config.enable_compression:
                filters.append(
                    f"acompressor=threshold={config.compression_threshold}dB:"
                    f"ratio={config.compression_ratio}:attack=5:release=50"
                )
                if progress_callback:
                    progress_callback(50, "compressing_dynamics")

            # 7. 靜音移除
            if config.enable_silence_removal:
                # 使用 silenceremove 濾鏡
                # stop_periods=0 只移除開始和結束的靜音，不移除中間的靜音
                # 這樣可以避免誤刪有聲音的區塊
                filters.append(
                    f"silenceremove=start_periods=1:start_duration={config.min_silence_duration}:"
                    f"start_threshold={config.silence_threshold}dB:"
                    f"stop_periods=1:stop_duration={config.min_silence_duration}:"
                    f"stop_threshold={config.silence_threshold}dB"
                )
                if progress_callback:
                    progress_callback(60, "removing_silence")

            # 8. 速度調整 (保持音調)
            if config.enable_speed_change and config.speed_factor != 1.0:
                filters.append(f"atempo={config.speed_factor}")
                if progress_callback:
                    progress_callback(70, "changing_speed")

            # 9. 音調調整
            if config.enable_pitch_shift and config.pitch_semitones != 0:
                # 使用 asetrate + atempo 組合來改變音調
                # 計算採樣率調整係數
                pitch_factor = 2 ** (config.pitch_semitones / 12)
                current_rate = 48000  # 假設原始採樣率
                new_rate = int(current_rate * pitch_factor)
                filters.append(f"asetrate={new_rate},atempo=1/{pitch_factor}")
                if progress_callback:
                    progress_callback(75, "shifting_pitch")

            # 10. 立體聲轉單聲道
            if config.enable_mono:
                filters.append("pan=mono|c0=0.5*c0+0.5*c1")

            # 11. 音量正規化 (放在最後)
            if config.enable_normalize:
                if config.normalize_type == "peak":
                    # 峰值歸一化
                    filters.append(f"volume={config.target_level}dB")
                else:  # lufs
                    # 響度歸一化
                    filters.append(f"loudnorm=I={config.target_level}:TP=-1.5:LRA=11")
                if progress_callback:
                    progress_callback(80, "normalizing_volume")

            # 12. 取樣率轉換
            if config.enable_resample:
                filters.append(f"aresample={config.target_sample_rate}")

            # 構建完整的 FFmpeg 命令
            command = [self.ffmpeg_path, "-i", input_path]

            if filters:
                filter_str = ",".join(filters)
                command.extend(["-af", filter_str])

            # 輸出設定
            command.extend([
                "-ar", str(config.target_sample_rate if config.enable_resample else 16000),
                "-ac", "1" if config.enable_mono else "2",
                "-y",  # 覆蓋輸出檔案
                output_path
            ])

            # 執行 FFmpeg
            if progress_callback:
                progress_callback(90, "processing")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg 處理失敗: {result.stderr}")

            if progress_callback:
                progress_callback(100, "completed")

            # 返回處理資訊
            return {
                "success": True,
                "output_path": output_path,
                "config": config.to_dict(),
                "filters_applied": filters
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def preprocess_vocal_separation(
        self,
        input_path: str,
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        人聲分離 (需要額外安裝 spleeter 或 demucs)
        這是一個佔位符，實際需要整合專門的人聲分離模型
        """
        # TODO: 整合 Demucs 或 Spleeter
        # 目前返回原始檔案
        return {
            "success": False,
            "error": "Vocal separation not implemented yet. Requires Demucs/Spleeter installation."
        }

    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """獲取音訊檔案資訊"""
        command = [
            self.ffmpeg_path, "-i", audio_path,
            "-hide_banner"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        # 從 stderr 解析音訊資訊
        info = {
            "duration": None,
            "sample_rate": None,
            "channels": None,
            "codec": None
        }

        for line in result.stderr.split('\n'):
            if "Duration:" in line:
                # 解析時長
                duration_str = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = duration_str.split(":")
                info["duration"] = float(h) * 3600 + float(m) * 60 + float(s)
            elif "Audio:" in line:
                # 解析音訊資訊
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if "Hz" in part:
                        info["sample_rate"] = int(part.split()[0])
                    elif "mono" in part.lower():
                        info["channels"] = 1
                    elif "stereo" in part.lower():
                        info["channels"] = 2
                    elif part.startswith("Audio:"):
                        info["codec"] = part.split(":")[1].strip().split()[0]

        return info


# 單例模式
audio_preprocessor = AudioPreprocessor()
