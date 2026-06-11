import React, { useEffect, useRef } from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface CameraViewProps {
  onFrameCapture: (canvas: HTMLCanvasElement) => void;
  drawOverlay: (canvas: HTMLCanvasElement, video: HTMLVideoElement) => void;
  isRunning: boolean;
  facingMode: 'user' | 'environment';
}

export const CameraView: React.FC<CameraViewProps> = ({ 
  onFrameCapture, 
  drawOverlay, 
  isRunning, 
  facingMode 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(null);
  const frameIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    
    const startCamera = async () => {
      if (videoRef.current) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: facingMode,
              width: { ideal: 640 },
              height: { ideal: 480 }
            },
            audio: false
          });
          videoRef.current.srcObject = stream;
        } catch (err) {
          console.error("Error accessing camera:", err);
        }
      }
    };

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [facingMode]);

  useEffect(() => {
    if (isRunning && videoRef.current && captureRef.current) {
      frameIntervalRef.current = window.setInterval(() => {
        const video = videoRef.current;
        const captureCanvas = captureRef.current;
        if (video && captureCanvas && video.readyState >= 2) {
          captureCanvas.width = video.videoWidth;
          captureCanvas.height = video.videoHeight;
          const ctx = captureCanvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            onFrameCapture(captureCanvas);
          }
        }
      }, 150); // Capture roughly 6-7 FPS to save bandwidth/compute
    } else if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }

    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    };
  }, [isRunning, onFrameCapture]);

  useEffect(() => {
    let animationFrameId: number;
    
    const renderLoop = () => {
      if (overlayRef.current && videoRef.current) {
        drawOverlay(overlayRef.current, videoRef.current);
      }
      animationFrameId = requestAnimationFrame(renderLoop);
    };
    
    renderLoop();
    
    return () => cancelAnimationFrame(animationFrameId);
  }, [drawOverlay]);

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-black overflow-hidden shadow-inner">
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted 
        className={cn("w-full h-full object-cover transition-opacity duration-500", facingMode === 'user' ? 'scale-x-[-1]' : '')}
      />
      <canvas 
        ref={overlayRef} 
        className="absolute top-0 left-0 w-full h-full pointer-events-none z-10"
      />
      <canvas ref={captureRef} className="hidden" />
      
      {/* Scanning Animation */}
      {isRunning && (
        <div className="absolute left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#06b6d4,0_0_30px_#06b6d4] opacity-60 z-20 pointer-events-none animate-scan" />
      )}
    </div>
  );
};
