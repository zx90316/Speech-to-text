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

  const handleStartTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    const time = parseFloat(e.target.value);
    setStartTime(time);
    setIsSelectingRange(true);

    // 如果正在播放且當前時間在新開始時間之前，跳轉到新開始時間
    if (audio && isPlaying && audio.currentTime < time) {
      audio.currentTime = time;
    }
  };

  const handleEndTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    const time = parseFloat(e.target.value);
    setEndTime(time);
    setIsSelectingRange(true);

    // 如果正在播放且當前時間超過新結束時間，跳轉到開始時間
    if (audio && isPlaying && audio.currentTime >= time) {
      audio.currentTime = startTime;
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
          <input
            type="range"
            min="0"
            max={duration || 0}
            step="0.1"
            value={startTime}
            onChange={handleStartTimeChange}
            disabled={!audioUrl}
          />
          <span className="time-value">{formatTime(startTime)}</span>
        </div>

        <div className="time-range-group">
          <label>結束時間</label>
          <input
            type="range"
            min="0"
            max={duration || 0}
            step="0.1"
            value={endTime}
            onChange={handleEndTimeChange}
            disabled={!audioUrl}
          />
          <span className="time-value">{formatTime(endTime)}</span>
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
