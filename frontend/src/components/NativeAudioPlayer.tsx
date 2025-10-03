/**
 * 原生 HTML5 Audio 播放器 - 適用於大文件串流播放
 * 使用原生 audio 標籤，支持 HTTP Range Requests，不會完整加載到記憶體
 */
import React from 'react';
import './NativeAudioPlayer.css';

interface NativeAudioPlayerProps {
  audioUrl: string;
  title: string;
}

export const NativeAudioPlayer: React.FC<NativeAudioPlayerProps> = ({
  audioUrl,
  title
}) => {
  return (
    <div className="native-audio-player">
      <div className="player-header">
        <span className="player-title">{title}</span>
        <span className="player-hint">大型音頻文件 - 使用串流播放</span>
      </div>
      <audio
        controls
        preload="metadata"
        className="audio-control"
      >
        <source src={audioUrl} />
        您的瀏覽器不支持音頻播放。
      </audio>
    </div>
  );
};
