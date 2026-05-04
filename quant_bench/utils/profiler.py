"""Profiling utilities for memory and throughput measurement."""

import time
from typing import Optional

class MemoryProfiler:
    """GPU memory profiling using nvidia-ml-py or nvidia-smi."""

    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None

    def start(self):
        """Start memory profiling."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.initial_memory = info.used / 1024**3  # Convert to GB
            print(f"Initial VRAM: {self.initial_memory:.2f} GB")
        except ImportError:
            print("Warning: pynvml not installed. Install with: pip install nvidia-ml-py3")

    def stop(self):
        """Stop profiling and return peak memory usage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            current_memory = info.used / 1024**3
            self.peak_memory = max(self.initial_memory or 0, current_memory)
            print(f"Peak VRAM: {self.peak_memory:.2f} GB")
            return self.peak_memory
        except ImportError:
            print("Warning: pynvml not installed.")
            return None

class ThroughputProfiler:
    """Measure tokens per second and latency."""

    def __init__(self):
        self.start_time = None
        self.token_count = 0

    def start(self):
        """Start throughput measurement."""
        self.start_time = time.time()

    def stop(self, num_tokens: int):
        """Stop measurement and return throughput."""
        if self.start_time is None:
            return 0
        elapsed = time.time() - self.start_time
        throughput = num_tokens / elapsed
        latency = (elapsed / num_tokens) * 1000  # ms per token
        print(f"Throughput: {throughput:.2f} tokens/sec")
        print(f"Latency: {latency:.2f} ms/token")
        return throughput, latency
