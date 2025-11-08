/**
 * 郵件驗證組件
 * 處理郵件驗證碼的發送和驗證
 */
import { useState } from 'react';
import { Mail, Check, AlertCircle, Loader } from 'lucide-react';
import { api } from '../api';

interface EmailVerificationProps {
  onVerified: (email: string) => void;
  initialEmail?: string;
}

export function EmailVerification({ onVerified, initialEmail = '' }: EmailVerificationProps) {
  const [email, setEmail] = useState(initialEmail);
  const [verificationCode, setVerificationCode] = useState('');
  const [step, setStep] = useState<'email' | 'code'>('email');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSendCode = async () => {
    setError(null);
    setSuccess(null);
    setLoading(true);

    try {
      const response = await api.sendVerificationEmail(email);
      setSuccess(response.message);
      setStep('code');
    } catch (err: any) {
      setError(err.response?.data?.detail || '發送驗證碼失敗，請檢查郵箱格式');
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyCode = async () => {
    setError(null);
    setSuccess(null);
    setLoading(true);

    try {
      const response = await api.verifyEmailCode(email, verificationCode);
      setSuccess(response.message);
      setTimeout(() => {
        onVerified(email);
      }, 500);
    } catch (err: any) {
      setError(err.response?.data?.detail || '驗證碼無效或已過期');
    } finally {
      setLoading(false);
    }
  };

  const handleResendCode = () => {
    setVerificationCode('');
    setError(null);
    setSuccess(null);
    handleSendCode();
  };

  return (
    <div className="email-verification">
      {step === 'email' ? (
        <div className="verification-step">
          <div className="step-header">
            <Mail size={24} />
            <h3>郵箱驗證</h3>
          </div>
          <p className="step-description">
            為了確保您能收到轉錄結果，請先驗證您的電子郵件地址
          </p>

          <div className="form-group">
            <label htmlFor="email">電子郵件</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your.email@example.com"
              disabled={loading}
              className="email-input"
            />
          </div>

          {error && (
            <div className="alert alert-error">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {success && (
            <div className="alert alert-success">
              <Check size={16} />
              {success}
            </div>
          )}

          <button
            onClick={handleSendCode}
            disabled={loading || !email}
            className="btn btn-primary"
          >
            {loading ? (
              <>
                <Loader size={16} className="spinner" />
                發送中...
              </>
            ) : (
              <>
                <Mail size={16} />
                發送驗證碼
              </>
            )}
          </button>
        </div>
      ) : (
        <div className="verification-step">
          <div className="step-header">
            <Check size={24} />
            <h3>輸入驗證碼</h3>
          </div>
          <p className="step-description">
            驗證碼已發送至 <strong>{email}</strong>，請檢查您的郵箱（包括垃圾郵件夾）
          </p>

          <div className="form-group">
            <label htmlFor="code">驗證碼（6位數字）</label>
            <input
              id="code"
              type="text"
              value={verificationCode}
              onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              placeholder="000000"
              disabled={loading}
              className="code-input"
              maxLength={6}
            />
          </div>

          {error && (
            <div className="alert alert-error">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {success && (
            <div className="alert alert-success">
              <Check size={16} />
              {success}
            </div>
          )}

          <div className="button-group">
            <button
              onClick={handleVerifyCode}
              disabled={loading || verificationCode.length !== 6}
              className="btn btn-primary"
            >
              {loading ? (
                <>
                  <Loader size={16} className="spinner" />
                  驗證中...
                </>
              ) : (
                <>
                  <Check size={16} />
                  驗證
                </>
              )}
            </button>

            <button
              onClick={handleResendCode}
              disabled={loading}
              className="btn btn-secondary"
            >
              重新發送驗證碼
            </button>

            <button
              onClick={() => {
                setStep('email');
                setVerificationCode('');
                setError(null);
                setSuccess(null);
              }}
              disabled={loading}
              className="btn btn-text"
            >
              更改郵箱
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
