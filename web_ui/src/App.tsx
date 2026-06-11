import React, { useState, useCallback } from 'react';
import { CameraView } from './components/CameraView';
import { TopBar } from './components/TopBar';
import { ControlPanel } from './components/ControlPanel';
import { DirectionBanner } from './components/DirectionBanner';
import { SonarRadar } from './components/SonarRadar';
import { useSocket, type DetectionResult } from './hooks/useSocket';

function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isNightMode, setIsNightMode] = useState(false);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [lastResult, setLastResult] = useState<DetectionResult | null>(null);

  const socketData = useSocket();

  // Draw overlay callback
  const drawOverlay = useCallback((canvas: HTMLCanvasElement, video: HTMLVideoElement) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Make canvas match video element dimensions
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!lastResult || !lastResult.detections) return;

    // Calculate scale between original frame and canvas
    const scaleX = canvas.width / lastResult.frame_width;
    const scaleY = canvas.height / lastResult.frame_height;

    lastResult.detections.forEach(det => {
      let [x1, y1, x2, y2] = det.bbox;
      
      // Handle camera flip if facing mode is user
      if (facingMode === 'user') {
        const tempX1 = x1;
        x1 = lastResult.frame_width - x2;
        x2 = lastResult.frame_width - tempX1;
      }

      x1 *= scaleX;
      x2 *= scaleX;
      y1 *= scaleY;
      y2 *= scaleY;

      const w = x2 - x1;
      const h = y2 - y1;

      // Determine colors based on risk
      let strokeColor = '#10b981'; // Green
      let bgColor = 'rgba(16, 185, 129, 0.2)';
      if (det.risk_level === 'high') {
        strokeColor = '#ef4444'; // Red
        bgColor = 'rgba(239, 68, 68, 0.2)';
      } else if (det.risk_level === 'medium') {
        strokeColor = '#f59e0b'; // Orange
        bgColor = 'rgba(245, 158, 11, 0.2)';
      }

      // Draw box
      ctx.lineWidth = 3;
      ctx.strokeStyle = strokeColor;
      ctx.fillStyle = bgColor;
      
      // Draw rounded rect manually or standard rect
      ctx.beginPath();
      ctx.rect(x1, y1, w, h);
      ctx.fill();
      ctx.stroke();

      // Draw Label Background
      const label = `${det.class_name} ${det.distance_str}`;
      ctx.font = 'bold 14px "JetBrains Mono", monospace';
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;

      ctx.fillStyle = strokeColor;
      ctx.fillRect(x1, y1 > 20 ? y1 - 22 : y1, textWidth + 12, 22);

      // Draw Label Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x1 + 6, y1 > 20 ? y1 - 6 : y1 + 16);
    });
  }, [lastResult, facingMode]);

  const handleFrameCapture = useCallback((canvas: HTMLCanvasElement) => {
    // Only send frames if running
    if (!isRunning) return;

    const dataUrl = canvas.toDataURL('image/jpeg', 0.6); // Compress to save bandwidth
    
    // Simulate pitch for now
    let pitch = 0;

    socketData.sendFrame({
      image: dataUrl,
      low_light: isNightMode,
      pitch: pitch,
      send_time: Date.now()
    });
  }, [isRunning, isNightMode, socketData]);

  // Hook up socket results
  React.useEffect(() => {
    socketData.setOnResult((result) => {
      setLastResult(result);
    });
  }, [socketData]);

  const toggleStart = () => {
    if (!isRunning && !socketData.isConnected) {
      alert("Cannot start: Not connected to server.");
      return;
    }
    setIsRunning(true);
    setLastResult(null);
  };

  const toggleStop = () => {
    setIsRunning(false);
    setLastResult(null);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (!isMuted) {
      window.speechSynthesis.cancel();
    }
  };

  const toggleCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
  };

  const toggleNightMode = () => {
    setIsNightMode(!isNightMode);
  };

  return (
    <div className="flex flex-col h-screen w-full relative">
      <TopBar 
        isConnected={socketData.isConnected}
        fps={socketData.fps}
        objectsCount={socketData.objectsCount}
        latency={socketData.latency}
      />
      
      <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-black shadow-[inset_0_0_40px_rgba(0,0,0,0.5)]">
        <CameraView 
          isRunning={isRunning}
          facingMode={facingMode}
          onFrameCapture={handleFrameCapture}
          drawOverlay={drawOverlay}
        />

        <SonarRadar 
          isActive={isRunning} 
          detectsObjects={socketData.objectsCount > 0} 
        />

        {!isRunning && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0f1115]/95 backdrop-blur-xl z-20 gap-8 p-8 text-center transition-opacity">
            <div className="relative w-[100px] h-[100px] flex items-center justify-center rounded-[30px] bg-gradient-to-br from-blue-500/20 to-violet-500/20 shadow-[0_0_40px_rgba(99,102,241,0.2)] animate-[float_6s_ease-in-out_infinite]">
              <span className="text-5xl drop-shadow-[0_0_10px_rgba(255,255,255,0.5)]">👁️</span>
            </div>
            
            <h1 className="text-2xl sm:text-3xl font-extrabold bg-gradient-to-br from-white to-indigo-200 bg-clip-text text-transparent max-w-[85%] leading-tight">
              Design and development of an assistive navigation system for the visually impaired
            </h1>
            
            <p className="text-[#8b92a5] text-[15px] max-w-[340px] leading-relaxed">
              Real-time AI obstacle detection with voice guidance and spatial sonar feedback.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 mb-2">
              <div className="text-[11px] bg-white/5 border border-white/10 px-3 py-1.5 rounded-full text-gray-300 flex items-center gap-1.5">
                ⚡ Real-time processing
              </div>
              <div className="text-[11px] bg-white/5 border border-white/10 px-3 py-1.5 rounded-full text-gray-300 flex items-center gap-1.5">
                🗣️ Voice guidance
              </div>
              <div className="text-[11px] bg-white/5 border border-white/10 px-3 py-1.5 rounded-full text-gray-300 flex items-center gap-1.5">
                🔔 Spatial Sonar
              </div>
            </div>
          </div>
        )}
      </div>

      <DirectionBanner direction={socketData.direction} />

      <div className="px-5 py-2 text-center text-sm text-[#8b92a5] bg-[#0f1115] border-t border-white/10 min-h-[40px] flex items-center justify-center shrink-0 font-light tracking-wide">
        {socketData.sceneText}
      </div>

      <ControlPanel 
        isRunning={isRunning}
        onStart={toggleStart}
        onStop={toggleStop}
        isMuted={isMuted}
        onToggleMute={toggleMute}
        onFlipCamera={toggleCamera}
        isNightMode={isNightMode}
        onToggleNightMode={toggleNightMode}
      />
    </div>
  );
}

export default App;
