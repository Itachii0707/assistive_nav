import React from 'react';
import { Play, Square, Mic, MicOff, Camera, Moon, Sun } from 'lucide-react';
import { cn } from './CameraView';

interface ControlPanelProps {
  isRunning: boolean;
  onStart: () => void;
  onStop: () => void;
  isMuted: boolean;
  onToggleMute: () => void;
  onFlipCamera: () => void;
  isNightMode: boolean;
  onToggleNightMode: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  isRunning,
  onStart,
  onStop,
  isMuted,
  onToggleMute,
  onFlipCamera,
  isNightMode,
  onToggleNightMode
}) => {
  return (
    <div className="flex items-center justify-center gap-4 px-5 py-4 glass-panel border-t shrink-0 z-50">
      {!isRunning ? (
        <button 
          onClick={onStart}
          className="relative flex items-center justify-center gap-2 px-6 py-3 w-full max-w-[300px] rounded-xl text-white font-bold tracking-wide overflow-hidden bg-gradient-to-br from-blue-500 to-violet-500 shadow-[0_8px_24px_rgba(99,102,241,0.3)] hover:-translate-y-0.5 hover:shadow-[0_12px_32px_rgba(99,102,241,0.4)] active:translate-y-px transition-all duration-300"
        >
          <Play size={20} fill="currentColor" />
          START NAVIGATION
        </button>
      ) : (
        <>
          <button 
            onClick={onStop}
            className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold tracking-wide bg-red-500/10 text-red-500 border border-red-500/30 hover:bg-red-500 hover:text-white hover:shadow-[0_8px_24px_rgba(239,68,68,0.4)] transition-all duration-300"
          >
            <Square size={20} fill="currentColor" />
            STOP
          </button>
          
          <button 
            onClick={onToggleMute}
            className={cn(
              "flex items-center justify-center p-3 rounded-xl border transition-all duration-300",
              isMuted 
                ? "bg-red-500/15 text-red-500 border-red-500" 
                : "bg-white/5 text-white border-white/10 hover:bg-white/10"
            )}
          >
            {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
          </button>

          <button 
            onClick={onToggleNightMode}
            className={cn(
              "flex items-center justify-center p-3 rounded-xl border transition-all duration-300",
              isNightMode
                ? "bg-emerald-500/15 text-emerald-500 border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.15)]" 
                : "bg-white/5 text-white border-white/10 hover:bg-white/10"
            )}
          >
            {isNightMode ? <Moon size={24} /> : <Sun size={24} />}
          </button>

          <button 
            onClick={onFlipCamera}
            className="flex items-center justify-center p-3 rounded-xl bg-white/5 text-white border border-white/10 hover:bg-white/10 transition-all duration-300"
            title="Flip Camera"
          >
            <Camera size={24} />
          </button>
        </>
      )}
    </div>
  );
};
