"""
benchmark.py — FPS, latency, and accuracy benchmarking.
"""

import time
import numpy as np
import psutil


class Benchmark:
    """Performance benchmarking utilities."""

    def __init__(self):
        self.frame_times = []
        self.inference_times = []
        self.pipeline_times = []

    def start_frame(self):
        return time.perf_counter()

    def end_frame(self, start):
        elapsed = (time.perf_counter() - start) * 1000
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 500:
            self.frame_times = self.frame_times[-250:]

    def record_inference(self, duration_ms):
        self.inference_times.append(duration_ms)
        if len(self.inference_times) > 500:
            self.inference_times = self.inference_times[-250:]

    def record_pipeline(self, duration_ms):
        self.pipeline_times.append(duration_ms)
        if len(self.pipeline_times) > 500:
            self.pipeline_times = self.pipeline_times[-250:]

    def get_fps(self):
        if not self.frame_times:
            return 0.0
        avg_ms = np.mean(self.frame_times[-30:])
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def get_avg_inference_ms(self):
        if not self.inference_times:
            return 0.0
        return float(np.mean(self.inference_times[-30:]))

    def get_avg_pipeline_ms(self):
        if not self.pipeline_times:
            return 0.0
        return float(np.mean(self.pipeline_times[-30:]))

    def get_memory_usage(self):
        return psutil.virtual_memory().percent

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=0)

    def get_report(self):
        return {
            'fps': round(self.get_fps(), 1),
            'avg_inference_ms': round(self.get_avg_inference_ms(), 1),
            'avg_pipeline_ms': round(self.get_avg_pipeline_ms(), 1),
            'memory_percent': self.get_memory_usage(),
            'cpu_percent': self.get_cpu_usage(),
        }

    def print_report(self):
        r = self.get_report()
        print(f"[BENCH] FPS: {r['fps']} | "
              f"Inference: {r['avg_inference_ms']}ms | "
              f"Pipeline: {r['avg_pipeline_ms']}ms | "
              f"RAM: {r['memory_percent']}% | "
              f"CPU: {r['cpu_percent']}%")
        