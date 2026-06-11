import React from 'react';
import { cn } from './CameraView';

interface TopBarProps {
  isConnected: boolean;
  fps: number | null;
  objectsCount: number;
  latency: number | null;
}

export const TopBar: React.FC<TopBarProps> = ({ isConnected, fps, objectsCount, latency }) => {
  return (
    <div className="flex items-center justify-between px-5 py-3 glass-panel border-b min-h-[56px] shrink-0 z-50">
      <div className="flex items-center gap-2 font-bold text-base tracking-wide bg-gradient-to-r from-white to-[#8b92a5] bg-clip-text text-transparent">
        <div className="flex items-center px-2 py-1 rounded-full border border-white/10 bg-black/20 mr-2">
          <span className={cn(
            "w-2.5 h-2.5 rounded-full shadow-[0_0_10px_currentColor]",
            isConnected ? "bg-emerald-500 text-emerald-500" : "bg-red-500 text-red-500"
          )} />
        </div>
        VISION<span className="font-light text-[#8b92a5]">AI</span>
      </div>
      
      <div className="flex gap-4 font-mono text-xs text-[#8b92a5] bg-black/20 px-3 py-1.5 rounded-full border border-white/10">
        <div className="flex items-center gap-1.5 hidden sm:flex">
          <span>FPS:</span>
          <span className="text-cyan-400 font-bold">{fps ? fps.toFixed(1) : '--'}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span>OBJ:</span>
          <span className="text-cyan-400 font-bold">{objectsCount}</span>
        </div>
        <div className="flex items-center gap-1.5 text-[#8b92a5] hidden sm:flex">
          <span>{latency ? `${latency} ms` : '-- ms'}</span>
        </div>
      </div>
    </div>
  );
};
