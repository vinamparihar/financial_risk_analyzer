import time
import threading
from collections import defaultdict

class RateLimitTracker:
    """
    Tracks API/model call counts and calculates rate limits for any number of services.
    Thread-safe for concurrent agent execution.
    """
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        self.lock = threading.Lock()
        self.calls = defaultdict(list)  # {service: [timestamps]}

    def log_call(self, service: str):
        now = time.time()
        with self.lock:
            self.calls[service].append(now)
            # Remove calls outside the window
            self.calls[service] = [t for t in self.calls[service] if now - t < self.window_seconds]

    def get_count(self, service: str):
        now = time.time()
        with self.lock:
            self.calls[service] = [t for t in self.calls[service] if now - t < self.window_seconds]
            return len(self.calls[service])

    def get_all_counts(self):
        now = time.time()
        with self.lock:
            for service in self.calls:
                self.calls[service] = [t for t in self.calls[service] if now - t < self.window_seconds]
            return {service: len(times) for service, times in self.calls.items()}

# Singleton instance for use across agents/tools
rate_limiter = RateLimitTracker()
