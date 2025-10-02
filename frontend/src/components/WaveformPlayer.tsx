/**
 * 音訊波形播放器組件
 */
import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, Volume2, VolumeX, SkipBack, SkipForward } from 'lucide-react';
import './WaveformPlayer.css';

interface WaveformPlayerProps {
  audioUrl: string;
  title: string;
  color?: string;
}

export const WaveformPlayer: React.FC<WaveformPlayerProps> = ({
  audioUrl,
  title,
  color = '#6366f1'
}) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // 載入音訊並生成波形
  useEffect(() => {
    const loadAudio = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();

        const audioContext = new AudioContext();
        const buffer = await audioContext.decodeAudioData(arrayBuffer);

        setAudioBuffer(buffer);
        setDuration(buffer.duration);
        drawWaveform(buffer);
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to load audio:', error);
        setIsLoading(false);
      }
    };

    loadAudio();
  }, [audioUrl]);

  // 繪製波形
  const drawWaveform = (buffer: AudioBuffer) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const data = buffer.getChannelData(0); // 使用第一個聲道
    const step = Math.ceil(data.length / width);
    const amp = height / 2;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#1e293b'; // 背景色
    ctx.fillRect(0, 0, width, height);

    // 繪製波形
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;

      for (let j = 0; j < step; j++) {
        const datum = data[i * step + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }

      const y1 = (1 + min) * amp;
      const y2 = (1 + max) * amp;

      if (i === 0) {
        ctx.moveTo(i, y1);
      }
      ctx.lineTo(i, y1);
      ctx.lineTo(i, y2);
    }

    ctx.stroke();

    // 繪製進度覆蓋
    drawProgress(ctx, width, height, 0);
  };

  // 繪製播放進度
  const drawProgress = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    progress: number
  ) => {
    const progressWidth = width * progress;

    // 繪製已播放部分（高亮）
    ctx.fillStyle = color + '40'; // 40% 透明度
    ctx.fillRect(0, 0, progressWidth, height);

    // 繪製進度線
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(progressWidth, 0);
    ctx.lineTo(progressWidth, height);
    ctx.stroke();
  };

  // 更新進度顯示
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioBuffer) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 重繪波形
    drawWaveform(audioBuffer);
  }, [currentTime, audioBuffer]);

  // 音訊時間更新
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);

  // 播放/暫停
  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  // 跳轉到指定時間
  const seekTo = (time: number) => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = time;
    setCurrentTime(time);
  };

  // 點擊波形跳轉
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !duration) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const progress = x / rect.width;
    seekTo(progress * duration);
  };

  // 快進/快退
  const skip = (seconds: number) => {
    seekTo(Math.max(0, Math.min(duration, currentTime + seconds)));
  };

  // 音量控制
  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
    if (newVolume === 0) {
      setIsMuted(true);
    } else if (isMuted) {
      setIsMuted(false);
    }
  };

  // 靜音切換
  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isMuted) {
      audio.volume = volume;
      setIsMuted(false);
    } else {
      audio.volume = 0;
      setIsMuted(true);
    }
  };

  // 格式化時間
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // 調整 canvas 尺寸
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;

      const rect = container.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = 120;

      if (audioBuffer) {
        drawWaveform(audioBuffer);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [audioBuffer]);

  return (
    <div className="waveform-player">
      <div className="waveform-header">
        <h4>{title}</h4>
        {isLoading && <span className="loading-indicator">載入中...</span>}
      </div>

      <div className="waveform-container" ref={containerRef}>
        <canvas
          ref={canvasRef}
          className="waveform-canvas"
          onClick={handleCanvasClick}
        />
      </div>

      <div className="waveform-controls">
        <div className="playback-controls">
          <button
            className="control-btn"
            onClick={() => skip(-5)}
            disabled={isLoading}
            title="後退 5 秒"
          >
            <SkipBack size={20} />
          </button>

          <button
            className="control-btn play-btn"
            onClick={togglePlay}
            disabled={isLoading}
          >
            {isPlaying ? <Pause size={24} /> : <Play size={24} />}
          </button>

          <button
            className="control-btn"
            onClick={() => skip(5)}
            disabled={isLoading}
            title="前進 5 秒"
          >
            <SkipForward size={20} />
          </button>
        </div>

        <div className="time-display">
          <span className="current-time">{formatTime(currentTime)}</span>
          <span className="separator">/</span>
          <span className="total-time">{formatTime(duration)}</span>
        </div>

        <div className="volume-controls">
          <button className="volume-btn" onClick={toggleMute}>
            {isMuted || volume === 0 ? (
              <VolumeX size={20} />
            ) : (
              <Volume2 size={20} />
            )}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            className="volume-slider"
          />
        </div>
      </div>

      <audio ref={audioRef} src={audioUrl} preload="metadata" />
    </div>
  );
};
