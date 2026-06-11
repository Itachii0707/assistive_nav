import React from 'react';
import { cn } from './CameraView';

interface DirectionBannerProps {
  direction: string;
}

export const DirectionBanner: React.FC<DirectionBannerProps> = ({ direction }) => {
  const getBannerStyles = (dir: string) => {
    const text = dir.toUpperCase();
    if (text.includes('STOP')) {
      return "text-red-500 bg-red-500/10 shadow-[inset_0_2px_20px_rgba(239,68,68,0.2)] animate-pulse";
    }
    if (text.includes('LEFT') || text.includes('RIGHT')) {
      return "text-amber-500 bg-amber-500/5 shadow-[inset_0_2px_20px_rgba(245,158,11,0.1)]";
    }
    return "text-emerald-500 bg-emerald-500/5 shadow-[inset_0_2px_20px_rgba(16,185,129,0.1)]";
  };

  return (
    <div className={cn(
      "p-4 text-center text-2xl sm:text-[24px] font-extrabold tracking-[3px] uppercase bg-[#181b21] border-t border-white/10 min-h-[68px] flex items-center justify-center shrink-0 relative overflow-hidden transition-all duration-500 z-50",
      getBannerStyles(direction)
    )}>
      {/* Glossy overlay */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full animate-[shimmer_2s_infinite]" />
      
      {direction}
    </div>
  );
};
