"""
音訊預處理任務處理器
負責異步處理音訊預處理任務
"""
import asyncio
from pathlib import Path
from typing import Dict, Any
from database import db_manager
from audio_preprocessor import audio_preprocessor, PreprocessConfig


class PreprocessProcessor:
    """預處理任務處理器（單例模式）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True

    async def process_task(self, preprocess_id: str):
        """處理預處理任務"""
        try:
            # 獲取任務詳情
            task = db_manager.get_preprocess_task(preprocess_id)
            if not task:
                print(f"找不到預處理任務: {preprocess_id}")
                return

            # 檢查任務是否已取消
            if task['status'] == 'canceled':
                print(f"預處理任務已取消: {preprocess_id}")
                return

            # 更新狀態為處理中
            db_manager.update_preprocess_status(
                preprocess_id,
                'processing',
                progress=0.0,
                current_stage='開始預處理'
            )

            # 解析配置
            config_dict = task.get('config', {})
            preprocess_config = PreprocessConfig.from_dict(config_dict)

            original_path = task['original_path']
            processed_path = task['processed_path']

            # 執行預處理（包含進度回調）
            def progress_callback(progress: float, stage: str):
                """進度更新回調"""
                # 檢查是否被取消
                current_task = db_manager.get_preprocess_task(preprocess_id)
                if current_task and current_task['status'] == 'canceled':
                    raise Exception("任務已取消")

                db_manager.update_preprocess_status(
                    preprocess_id,
                    'processing',
                    progress=progress,
                    current_stage=stage
                )

            # 階段 1: 分析原始音訊 (0-10%)
            progress_callback(5.0, '分析原始音訊')
            original_info = await asyncio.to_thread(
                audio_preprocessor.get_audio_info,
                original_path
            )

            # 階段 2: 執行預處理 (10-90%)
            progress_callback(10.0, '執行音訊預處理')
            result = await asyncio.to_thread(
                audio_preprocessor.preprocess,
                original_path,
                processed_path,
                preprocess_config,
                progress_callback  # 傳遞進度回調以檢查取消
            )

            if not result.get("success"):
                raise Exception(result.get("error", "預處理失敗"))

            # 階段 3: 分析處理後音訊 (90-100%)
            progress_callback(90.0, '分析處理後音訊')
            processed_info = await asyncio.to_thread(
                audio_preprocessor.get_audio_info,
                processed_path
            )

            # 更新結果
            db_manager.update_preprocess_result(
                preprocess_id,
                original_info,
                processed_info,
                result.get("filters_applied", [])
            )

            # 完成
            db_manager.update_preprocess_status(
                preprocess_id,
                'completed',
                progress=100.0,
                current_stage='預處理完成'
            )

            print(f"預處理任務完成: {preprocess_id}")

        except Exception as e:
            error_msg = str(e)
            print(f"預處理任務失敗: {preprocess_id}, 錯誤: {error_msg}")

            # 更新失敗狀態
            db_manager.update_preprocess_status(
                preprocess_id,
                'failed',
                error_message=error_msg
            )


# 全局預處理處理器實例
preprocess_processor = PreprocessProcessor()
