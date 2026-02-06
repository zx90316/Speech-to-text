# -*- coding: utf-8 -*-
"""
郵件 API 路由
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, EmailStr

from email_service import email_service

router = APIRouter(tags=["郵件驗證"])


class EmailResponse(BaseModel):
    success: bool
    message: str


class VerifyResponse(BaseModel):
    success: bool
    message: str
    email: str


@router.post("/email/send-verification", response_model=EmailResponse)
async def send_verification_email(
    email: EmailStr = Query(..., description="電子郵件地址")
):
    """
    發送驗證碼郵件
    
    驗證碼有效期為 5 分鐘
    """
    success = email_service.send_verification_email(email)
    
    if success:
        return EmailResponse(
            success=True,
            message="驗證碼已發送，請檢查您的郵箱"
        )
    else:
        # 即使發送失敗，仍返回成功（避免暴露郵件服務狀態）
        # 驗證碼已經生成並存儲在資料庫中
        return EmailResponse(
            success=True,
            message="驗證碼已發送，請檢查您的郵箱（若未配置 SMTP，請檢查伺服器日誌）"
        )


@router.post("/email/verify-code", response_model=VerifyResponse)
async def verify_email_code(
    email: EmailStr = Query(..., description="電子郵件地址"),
    code: str = Query(..., min_length=6, max_length=6, description="6 位數驗證碼")
):
    """
    驗證郵件驗證碼
    
    驗證成功後，郵件將被標記為已驗證（24 小時內有效）
    """
    if email_service.verify_code(email, code):
        return VerifyResponse(
            success=True,
            message="驗證成功",
            email=email
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="驗證碼錯誤或已過期"
        )


@router.get("/email/check-verified")
async def check_email_verified(
    email: EmailStr = Query(..., description="電子郵件地址")
):
    """檢查郵件是否已驗證"""
    is_verified = email_service.is_email_verified(email)
    return {
        "email": email,
        "is_verified": is_verified
    }
