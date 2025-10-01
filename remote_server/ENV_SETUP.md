# 環境變數設定說明

## 必要環境變數

在 `remote_server` 目錄下創建 `.env` 檔案，內容如下：

```env
# Hugging Face Token (用於下載語者分離模型)
# 請前往 https://huggingface.co/settings/tokens 獲取
HUGGINGFACE_TOKEN=your_token_here
```

## 獲取 Hugging Face Token

1. 前往 [Hugging Face](https://huggingface.co/)
2. 註冊或登入帳號
3. 前往 [Settings > Access Tokens](https://huggingface.co/settings/tokens)
4. 點擊 "New token" 創建新的 Token
5. 選擇 "Read" 權限即可
6. 複製生成的 Token
7. 將 Token 貼到 `.env` 檔案中

## 語者分離模型授權

使用語者分離功能需要同意模型使用條款：

1. 訪問 [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
2. 點擊 "Agree and access repository"
3. 閱讀並同意使用條款

完成後，您的 Token 就可以用來下載該模型了。

## 驗證設定

啟動 API 後，如果環境變數設定正確，模型會自動下載（首次需要一些時間）。

如果看到以下錯誤，表示 Token 未設定或無效：
```
Error: Cannot access gated repository
```

請確認：
1. `.env` 檔案是否在正確的位置（`remote_server/.env`）
2. Token 是否正確貼入
3. 是否已同意模型使用條款

