"""
郵件服務模組
處理郵件驗證和任務完成通知
"""
import os
import smtplib
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from typing import Optional, Dict
import threading


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

            # 驗證碼存儲（記憶體中）
            self.verification_codes: Dict[str, dict] = {}
            self.initialized = True

    def generate_verification_code(self, email: str) -> str:
        """生成 6 位數驗證碼"""
        code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])

        # 存儲驗證碼，5 分鐘有效
        self.verification_codes[email] = {
            'code': code,
            'expires_at': datetime.now() + timedelta(minutes=5),
            'verified': False
        }

        return code

    def verify_code(self, email: str, code: str) -> bool:
        """驗證郵件驗證碼"""
        if email not in self.verification_codes:
            return False

        stored = self.verification_codes[email]

        # 檢查是否過期
        if datetime.now() > stored['expires_at']:
            del self.verification_codes[email]
            return False

        # 檢查驗證碼是否正確
        if stored['code'] == code:
            stored['verified'] = True
            # 驗證成功後，延長有效期到 24 小時
            stored['expires_at'] = datetime.now() + timedelta(hours=24)
            return True

        return False

    def is_email_verified(self, email: str) -> bool:
        """檢查郵件是否已驗證"""
        if email not in self.verification_codes:
            return False

        stored = self.verification_codes[email]

        # 檢查是否過期
        if datetime.now() > stored['expires_at']:
            return False

        return stored.get('verified', False)

    def send_verification_email(self, to_email: str) -> bool:
        """發送驗證郵件"""
        try:
            code = self.generate_verification_code(to_email)

            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = 'Whisper 語音轉文字 - 郵件驗證碼'

            body = f"""
            <html>
            <body>
                <h2>郵件驗證</h2>
                <p>您的驗證碼是：<strong style="font-size: 24px; color: #007bff;">{code}</strong></p>
                <p>此驗證碼將在 5 分鐘後過期。</p>
                <p>如果您沒有請求此驗證碼，請忽略此郵件。</p>
                <hr>
                <p style="color: #666; font-size: 12px;">Whisper 語音轉文字服務</p>
            </body>
            </html>
            """

            msg.attach(MIMEText(body, 'html'))

            return self._send_email(msg)

        except Exception as e:
            print(f"發送驗證郵件失敗: {e}")
            return False

    def send_completion_email(
        self,
        to_email: str,
        task_id: str,
        filename: str,
        transcript_text: str,
        has_diarization: bool = False,
        confidence_html_path: Optional[str] = None
    ) -> bool:
        """發送任務完成通知郵件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = f'轉錄完成 - {filename}'

            # 截取前 500 字作為預覽
            preview = transcript_text[:500]
            if len(transcript_text) > 500:
                preview += "..."

            # 根據是否有信心分數報告調整郵件內容
            confidence_note = ""
            if confidence_html_path:
                confidence_note = """
                <p><strong>📊 信心分數報告：</strong>附件中包含詞級信心分數可視化 HTML 報告，在瀏覽器中打開可查看詳細的轉錄信心度分析。</p>
                """

            body = f"""
            <html>
            <body>
                <h2>語音轉文字任務完成</h2>
                <p>您的音訊檔案 <strong>{filename}</strong> 已完成轉錄。</p>
                <p><strong>任務 ID:</strong> {task_id}</p>
                <p><strong>語者分離:</strong> {'已啟用' if has_diarization else '未啟用'}</p>
                {confidence_note}

                <h3>轉錄結果預覽：</h3>
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap;">
{preview}
                </div>

                <p>完整的轉錄結果請見附件。</p>

                <hr>
                <p style="color: #666; font-size: 12px;">Whisper 語音轉文字服務</p>
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

            # 附加信心分數可視化 HTML（如果有）
            if confidence_html_path and os.path.exists(confidence_html_path):
                try:
                    with open(confidence_html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    html_attachment = MIMEBase('text', 'html')
                    html_attachment.set_payload(html_content.encode('utf-8'))
                    encoders.encode_base64(html_attachment)
                    html_attachment.add_header(
                        'Content-Disposition',
                        f'attachment; filename="{task_id}_confidence_report.html"'
                    )
                    msg.attach(html_attachment)
                    print(f"✓ 已附加信心分數可視化 HTML: {confidence_html_path}")
                except Exception as html_error:
                    print(f"⚠️ 附加 HTML 文件失敗: {html_error}")

            return self._send_email(msg)

        except Exception as e:
            print(f"發送完成通知郵件失敗: {e}")
            return False

    def _send_email(self, msg: MIMEMultipart) -> bool:
        """發送郵件的內部方法"""
        try:
            if not self.smtp_username or not self.smtp_password:
                print("警告：SMTP 憑證未設定，無法發送郵件")
                return False

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            return True

        except Exception as e:
            print(f"發送郵件失敗: {e}")
            return False


# 全局郵件服務實例
email_service = EmailService()
