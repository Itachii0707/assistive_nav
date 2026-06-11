import React from 'react';
import { motion } from 'framer-motion';
import { cn } from './CameraView';

interface SonarRadarProps {
  isActive: boolean;
  detectsObjects: boolean;
}

export const SonarRadar: React.FC<SonarRadarProps> = ({ isActive, detectsObjects }) => {
  if (!isActive) return null;

  return (
    <div className="absolute top-[60px] right-5 w-[130px] h-[130px] rounded-full border-2 border-cyan-400/20 shadow-[0_0_20px_rgba(6,182,212,0.1),inset_0_0_20px_rgba(6,182,212,0.1)] z-10 pointer-events-none transform perspective-[1000px] rotate-x-[60deg] rotate-y-[-5deg]">
      
      {/* Radar Sweep Animation */}
      <motion.div 
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
        className="absolute w-full h-full rounded-full"
        style={{
          background: 'conic-gradient(from 0deg, rgba(6, 182, 212, 0.35), transparent 60%)'
        }}
      />
      
      {/* Pulse Ring */}
      <motion.div 
        animate={{ 
          scale: [0.8, 1.15, 0.8],
          opacity: [0.3, 0.85, 0.3]
        }}
        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
        className={cn(
          "absolute w-[70%] h-[70%] top-[15%] left-[15%] rounded-full border-[1.5px] border-dashed",
          detectsObjects ? "border-red-500/50" : "border-violet-500/30"
        )}
      />

      {/* Center Glow */}
      <div className={cn(
        "absolute w-2 h-2 rounded-full shadow-[0_0_15px_#06b6d4,0_0_30px_#06b6d4] top-[-4px] left-1/2 -translate-x-1/2",
        detectsObjects ? "bg-red-400 shadow-red-500" : "bg-cyan-400"
      )} />
      
      {/* Target Blips */}
      {detectsObjects && (
        <>
          <motion.div 
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: [0, 1, 0], scale: [0.5, 1.5, 0.5] }}
            transition={{ repeat: Infinity, duration: 1.5, delay: 0.5 }}
            className="absolute w-3 h-3 bg-red-500 rounded-full top-[30%] left-[60%] shadow-[0_0_10px_#ef4444]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: [0, 1, 0], scale: [0.5, 1.5, 0.5] }}
            transition={{ repeat: Infinity, duration: 2, delay: 1 }}
            className="absolute w-2 h-2 bg-orange-400 rounded-full top-[60%] left-[30%] shadow-[0_0_8px_#f59e0b]"
          />
        </>
      )}
    </div>
  );
};
