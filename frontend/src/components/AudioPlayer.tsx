/**
 * 音頻播放器組件
 * 支援播放、暫停、進度條拖動、時間範圍選擇
 */
import { useEffect, useRef, useState } from 'react';
import { Play, Pause, Volume2, VolumeX } from 'lucide-react';

interface AudioPlayerProps {
  file: File | null;
  onTimeRangeChange?: (start: number, end: number) => void;
}

export function AudioPlayer({ file, onTimeRangeChange }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // 時間範圍選擇
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [isSelectingRange, setIsSelectingRange] = useState(false);

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setAudioUrl(null);
      setCurrentTime(0);
      setDuration(0);
      setIsPlaying(false);
    }
  }, [file]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => {
      const time = audio.currentTime;
      setCurrentTime(time);

      // 如果正在播放且設定了時間範圍，檢查是否超出範圍
      if (isPlaying && isSelectingRange) {
        if (time >= endTime) {
          // 循環回到開始時間
          audio.currentTime = startTime;
        } else if (time < startTime) {
          // 如果手動拖動到範圍外，跳轉回開始時間
          audio.currentTime = startTime;
        }
      }
    };

    const handleDurationChange = () => {
      setDuration(audio.duration);
      setEndTime(audio.duration);
    };

    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleDurationChange);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleDurationChange);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioUrl, isPlaying, isSelectingRange, startTime, endTime]);

  useEffect(() => {
    if (onTimeRangeChange && duration > 0 && isSelectingRange) {
      onTimeRangeChange(startTime, endTime);
    }
  }, [startTime, endTime, duration, isSelectingRange, onTimeRangeChange]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      // 如果設定了時間範圍且當前時間不在範圍內，跳轉到開始時間
      if (isSelectingRange && (audio.currentTime < startTime || audio.currentTime >= endTime)) {
        audio.currentTime = startTime;
      }
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const time = parseFloat(e.target.value);
    audio.currentTime = time;
    setCurrentTime(time);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const vol = parseFloat(e.target.value);
    audio.volume = vol;
    setVolume(vol);
    setIsMuted(vol === 0);
  };

  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isMuted) {
      audio.volume = volume || 0.5;
      setIsMuted(false);
    } else {
      audio.volume = 0;
      setIsMuted(true);
    }
  };

  const formatTime = (seconds: number) => {
    if (!isFinite(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // 將秒數轉換為分鐘和秒
  const secondsToMinutesAndSeconds = (totalSeconds: number): { minutes: number; seconds: number } => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = Math.floor(totalSeconds % 60);
    return { minutes: mins, seconds: secs };
  };

  // 驗證並調整時間範圍
  const validateTimeRange = (start: number, end: number): { start: number; end: number } => {
    let validStart = start;
    let validEnd = end;

    // 確保開始時間不小於 0
    validStart = Math.max(0, validStart);

    // 確保結束時間不大於音頻時長
    validEnd = Math.min(duration, validEnd);

    // 確保結束時間大於 0
    validEnd = Math.max(0.1, validEnd);

    // 確保開始時間小於結束時間（至少相差 0.1 秒）
    if (validStart >= validEnd) {
      validStart = Math.max(0, validEnd - 0.1);
    }

    return { start: validStart, end: validEnd };
  };

  const handleStartTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    let time = parseFloat(e.target.value);
    
    // 驗證時間範圍
    const validated = validateTimeRange(time, endTime);
    setStartTime(validated.start);
    setEndTime(validated.end);
    setIsSelectingRange(true);

    // 如果正在播放且當前時間在新開始時間之前，跳轉到新開始時間
    if (audio && isPlaying && audio.currentTime < validated.start) {
      audio.currentTime = validated.start;
    }
  };

  const handleEndTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    let time = parseFloat(e.target.value);
    
    // 驗證時間範圍
    const validated = validateTimeRange(startTime, time);
    setStartTime(validated.start);
    setEndTime(validated.end);
    setIsSelectingRange(true);

    // 如果正在播放且當前時間超過新結束時間，跳轉到開始時間
    if (audio && isPlaying && audio.currentTime >= validated.end) {
      audio.currentTime = validated.start;
    }
  };

  // 處理開始時間分鐘輸入
  const handleStartMinuteInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    const mins = parseInt(e.target.value) || 0;
    const { seconds } = secondsToMinutesAndSeconds(startTime);
    const time = Math.max(0, mins * 60 + seconds);
    
    // 驗證時間範圍
    const validated = validateTimeRange(time, endTime);
    setStartTime(validated.start);
    setEndTime(validated.end);
    setIsSelectingRange(true);

    // 如果正在播放且當前時間在新開始時間之前，跳轉到新開始時間
    if (audio && isPlaying && audio.currentTime < validated.start) {
      audio.currentTime = validated.start;
    }
  };

  // 處理開始時間秒鐘輸入
  const handleStartSecondInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    let secs = parseInt(e.target.value) || 0;
    // 限制秒數範圍 0-59
    secs = Math.max(0, Math.min(59, secs));
    const { minutes } = secondsToMinutesAndSeconds(startTime);
    const time = minutes * 60 + secs;
    
    // 驗證時間範圍
    const validated = validateTimeRange(time, endTime);
    setStartTime(validated.start);
    setEndTime(validated.end);
    setIsSelectingRange(true);

    // 如果正在播放且當前時間在新開始時間之前，跳轉到新開始時間
    if (audio && isPlaying && audio.currentTime < validated.start) {
      audio.currentTime = validated.start;
    }
  };

  // 處理結束時間分鐘輸入
  const handleEndMinuteInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    const mins = parseInt(e.target.value) || 0;
    const { seconds } = secondsToMinutesAndSeconds(endTime);
    const time = Math.max(0, mins * 60 + seconds);
    
    // 驗證時間範圍
    const validated = validateTimeRange(startTime, time);
    setStartTime(validated.start);
    setEndTime(validated.end);
    setIsSelectingRange(true);

    // 如果正在播放且當前時間超過新結束時間，跳轉到開始時間
    if (audio && isPlaying && audio.currentTime >= validated.end) {
      audio.currentTime = validated.start;
    }
  };

  // 處理結束時間秒鐘輸入
  const handleEndSecondInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    let secs = parseInt(e.target.value) || 0;
    // 限制秒數範圍 0-59
    secs = Math.max(0, Math.min(59, secs));
    const { minutes } = secondsToMinutesAndSeconds(endTime);
    const time = minutes * 60 + secs;
    
    // 驗證時間範圍
    const validated = validateTimeRange(startTime, time);
    setStartTime(validated.start);
    setEndTime(validated.end);
    setIsSelectingRange(true);

    // 如果正在播放且當前時間超過新結束時間，跳轉到開始時間
    if (audio && isPlaying && audio.currentTime >= validated.end) {
      audio.currentTime = validated.start;
    }
  };

  const resetTimeRange = () => {
    setStartTime(0);
    setEndTime(duration);
    setIsSelectingRange(false);
    if (onTimeRangeChange) {
      onTimeRangeChange(0, duration);
    }
  };

  if (!audioUrl) {
    return null;
  }

  return (
    <div className="audio-player">
      <audio ref={audioRef} src={audioUrl} />

      <div className="player-controls">
        <button className="play-button" onClick={togglePlay} disabled={!audioUrl}>
          {isPlaying ? <Pause size={24} /> : <Play size={24} />}
        </button>

        <div className="time-display">{formatTime(currentTime)}</div>

        <div className="progress-container">
          <input
            type="range"
            className="progress-slider"
            min="0"
            max={duration || 0}
            step="0.1"
            value={currentTime}
            onChange={handleSeek}
            disabled={!audioUrl}
          />
          {isSelectingRange && (
            <div className="time-range-indicator">
              <div
                className="range-highlight"
                style={{
                  left: `${(startTime / duration) * 100}%`,
                  width: `${((endTime - startTime) / duration) * 100}%`
                }}
              />
            </div>
          )}
        </div>

        <div className="time-display">{formatTime(duration)}</div>

        <button className="volume-button" onClick={toggleMute}>
          {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
        </button>

        <input
          type="range"
          className="volume-slider"
          min="0"
          max="1"
          step="0.01"
          value={isMuted ? 0 : volume}
          onChange={handleVolumeChange}
        />
      </div>

      <div className="time-range-controls">
        <div className="time-range-group">
          <label>開始時間</label>
          <div className="time-control-row">
            <input
              type="range"
              min="0"
              max={duration || 0}
              step="0.1"
              value={startTime}
              onChange={handleStartTimeChange}
              disabled={!audioUrl}
              className="time-slider"
            />
            <div className="time-inputs">
              <input
                type="number"
                className="time-input-number"
                value={secondsToMinutesAndSeconds(startTime).minutes}
                onChange={handleStartMinuteInput}
                min="0"
                disabled={!audioUrl}
                placeholder="0"
              />
              <span className="time-separator">:</span>
              <input
                type="number"
                className="time-input-number"
                value={secondsToMinutesAndSeconds(startTime).seconds}
                onChange={handleStartSecondInput}
                min="0"
                max="59"
                disabled={!audioUrl}
                placeholder="00"
              />
            </div>
          </div>
        </div>

        <div className="time-range-group">
          <label>結束時間</label>
          <div className="time-control-row">
            <input
              type="range"
              min="0"
              max={duration || 0}
              step="0.1"
              value={endTime}
              onChange={handleEndTimeChange}
              disabled={!audioUrl}
              className="time-slider"
            />
            <div className="time-inputs">
              <input
                type="number"
                className="time-input-number"
                value={secondsToMinutesAndSeconds(endTime).minutes}
                onChange={handleEndMinuteInput}
                min="0"
                disabled={!audioUrl}
                placeholder="0"
              />
              <span className="time-separator">:</span>
              <input
                type="number"
                className="time-input-number"
                value={secondsToMinutesAndSeconds(endTime).seconds}
                onChange={handleEndSecondInput}
                min="0"
                max="59"
                disabled={!audioUrl}
                placeholder="00"
              />
            </div>
          </div>
        </div>

        <button className="reset-range-button" onClick={resetTimeRange} disabled={!audioUrl}>
          重置範圍
        </button>
      </div>

      {isSelectingRange && (
        <div className="range-info">
          將轉錄 {formatTime(startTime)} - {formatTime(endTime)}
          （長度: {formatTime(endTime - startTime)}）
        </div>
      )}
    </div>
  );
}
