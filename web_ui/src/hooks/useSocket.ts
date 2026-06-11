import { useState, useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  zone: string;
  risk_level: 'low' | 'medium' | 'high';
  proximity: string;
  area_ratio: number;
  distance_m: number;
  distance_str: string;
}

export interface DetectionResult {
  detections: Detection[];
  direction: string;
  scene: string;
  alert: string | null;
  fps: number;
  inference_ms: number;
  num_objects: number;
  frame_width: number;
  frame_height: number;
  send_time: number;
}

export function useSocket() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [fps, setFps] = useState<number | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [objectsCount, setObjectsCount] = useState(0);
  const [direction, setDirection] = useState('READY TO START');
  const [sceneText, setSceneText] = useState('System initialized. Waiting for camera...');
  
  const synth = window.speechSynthesis;

  // We use refs for callbacks to avoid re-creating socket listeners
  const onResultRef = useRef<((result: DetectionResult) => void) | undefined>(undefined);

  useEffect(() => {
    // Determine the socket URL based on the current location
    const protocol = window.location.protocol;
    const host = window.location.hostname;
    // The Flask server typically runs on the same port or 5000/8080.
    // In dev, Vite is 5173 and Flask is 5000. So we assume Flask is on port 5000 if dev.
    const port = import.meta.env.DEV ? 5000 : window.location.port;
    const socketUrl = `${protocol}//${host}:${port}`;
    
    console.log("Connecting to socket:", socketUrl);
    const newSocket = io(socketUrl, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 10,
      reconnectionDelay: 1000
    });

    newSocket.on('connect', () => {
      console.log('Connected to server');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
      setIsConnected(false);
    });

    newSocket.on('result', (result: DetectionResult) => {
      if (onResultRef.current) {
        onResultRef.current(result);
      }
      
      setFps(result.fps);
      setObjectsCount(result.num_objects);
      setDirection(result.direction);
      setSceneText(result.scene);
      
      if (result.send_time) {
        const currentLatency = Date.now() - result.send_time;
        setLatency(currentLatency);
      }

      if (result.alert && !synth.speaking) {
        const utterance = new SpeechSynthesisUtterance(result.alert);
        utterance.rate = 1.1;
        synth.speak(utterance);
      }
    });

    newSocket.on('error', (err: any) => {
      console.error('Socket error:', err);
    });

    setSocket(newSocket);

    return () => {
      newSocket.disconnect();
    };
  }, []);

  const sendFrame = useCallback((frameData: { image: string, low_light: boolean, pitch: number, send_time: number }) => {
    if (socket && isConnected) {
      socket.emit('frame', frameData);
    }
  }, [socket, isConnected]);

  const setOnResult = useCallback((callback: (result: DetectionResult) => void) => {
    onResultRef.current = callback;
  }, []);

  return {
    isConnected,
    fps,
    latency,
    objectsCount,
    direction,
    sceneText,
    sendFrame,
    setOnResult
  };
}
