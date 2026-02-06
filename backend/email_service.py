# -*- coding: utf-8 -*-
"""
郵件服務模組
處理郵件驗證和任務完成通知
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional
import threading

from storage import email_verification


class EmailService:
    """郵件服務類別"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            # SMTP 設定（從環境變數讀取）
            self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
            self.smtp_username = os.getenv('SMTP_USERNAME', '')
            self.smtp_password = os.getenv('SMTP_PASSWORD', '')
            self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)
            self.initialized = True
    
    def generate_verification_code(self, email: str) -> str:
        """生成 6 位數驗證碼"""
        return email_verification.generate_code(email)
    
    def verify_code(self, email: str, code: str) -> bool:
        """驗證郵件驗證碼"""
        return email_verification.verify_code(email, code)
    
    def is_email_verified(self, email: str) -> bool:
        """檢查郵件是否已驗證"""
        return email_verification.is_verified(email)
    
    def send_verification_email(self, to_email: str) -> bool:
        """發送驗證郵件"""
        try:
            code = self.generate_verification_code(to_email)
            
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = 'Qwen ASR 語音轉文字 - 郵件驗證碼'
            
            body = f"""
            <html>
            <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px;">
                <div style="max-width: 500px; margin: 0 auto; background: rgba(255,255,255,0.05); border-radius: 10px; padding: 30px;">
                    <h2 style="color: #667eea;">🔐 郵件驗證</h2>
                    <p>您的驗證碼是：</p>
                    <div style="font-size: 32px; font-weight: bold; color: #764ba2; letter-spacing: 8px; text-align: center; padding: 20px; background: rgba(102,126,234,0.1); border-radius: 8px;">
                        {code}
                    </div>
                    <p style="margin-top: 20px;">此驗證碼將在 <strong>5 分鐘</strong>後過期。</p>
                    <p style="color: #8892b0; font-size: 12px;">如果您沒有請求此驗證碼，請忽略此郵件。</p>
                    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;">
                    <p style="color: #8892b0; font-size: 12px;">Qwen ASR 語音轉文字服務 V2</p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            return self._send_email(msg)
            
        except Exception as e:
            print(f"❌ 發送驗證郵件失敗: {e}")
            return False
    
    def send_completion_email(
        self,
        to_email: str,
        task_id: str,
        filename: str,
        transcript_text: str,
        has_diarization: bool = False,
        detected_language: Optional[str] = None,
    ) -> bool:
        """發送任務完成通知郵件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = f'✅ 轉錄完成 - {filename}'
            
            # 截取前 500 字作為預覽
            preview = transcript_text[:500]
            if len(transcript_text) > 500:
                preview += "..."
            
            body = f"""
            <html>
            <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; background: rgba(255,255,255,0.05); border-radius: 10px; padding: 30px;">
                    <h2 style="color: #667eea;">🎉 語音轉文字任務完成</h2>
                    
                    <p>您的音訊檔案 <strong style="color: #764ba2;">{filename}</strong> 已完成轉錄。</p>
                    
                    <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <p style="margin: 5px 0;"><strong>任務 ID:</strong> <code style="color: #667eea;">{task_id}</code></p>
                        <p style="margin: 5px 0;"><strong>偵測語言:</strong> {detected_language or '未知'}</p>
                        <p style="margin: 5px 0;"><strong>語者分離:</strong> {'✅ 已啟用' if has_diarization else '❌ 未啟用'}</p>
                    </div>
                    
                    <h3 style="color: #667eea;">📝 轉錄結果預覽：</h3>
                    <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 15px; white-space: pre-wrap; font-size: 14px; line-height: 1.6;">
{preview}
                    </div>
                    
                    <p style="margin-top: 20px;">完整的轉錄結果請見附件。</p>
                    
                    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;">
                    <p style="color: #8892b0; font-size: 12px;">Qwen ASR 語音轉文字服務 V2</p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # 附加轉錄文字檔案
            attachment = MIMEBase('text', 'plain')
            attachment.set_payload(transcript_text.encode('utf-8'))
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename="{task_id}_transcript.txt"'
            )
            msg.attach(attachment)
            
            return self._send_email(msg)
            
        except Exception as e:
            print(f"❌ 發送完成通知郵件失敗: {e}")
            return False
    
    def _send_email(self, msg: MIMEMultipart) -> bool:
        """發送郵件的內部方法"""
        try:
            if not self.smtp_username or not self.smtp_password:
                print("⚠️ SMTP 未配置，跳過郵件發送")
                return False
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                print(f"✅ 郵件已成功發送至 {msg['To']}")
            return True
            
        except Exception as e:
            print(f"❌ 發送郵件失敗: {e}")
            return False


# 全局郵件服務實例
email_service = EmailService()
