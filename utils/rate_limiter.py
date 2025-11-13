# utils/rate_limiter.py
import time
import threading
from collections import deque
from typing import Optional


class RateLimiter:
    """
    A thread-safe rate limiter that ensures no more than max_calls
    are made within a time_window (in seconds).
    """
    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        """
        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds (default: 60 seconds = 1 minute)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()
    
    def acquire(self) -> None:
        """
        Blocks until a call can be made without exceeding the rate limit.
        Call this before making an API request.
        Optimized: Minimal blocking, smarter wait calculation.
        """
        with self.lock:
            now = time.time()
            
            # Remove calls that are outside the time window
            while self.calls and self.calls[0] <= now - self.time_window:
                self.calls.popleft()
            
            # If we're at the limit, wait until the oldest call expires
            if len(self.calls) >= self.max_calls:
                # Calculate minimum wait time needed
                oldest_call = self.calls[0]
                wait_needed = oldest_call + self.time_window - now
                
                if wait_needed > 0:
                    # Add minimal buffer (0.05s instead of 0.1s)
                    sleep_time = wait_needed + 0.05
                    if sleep_time > 1.0:  # Only print if significant wait
                        print(f"    ⏳ Rate limit: waiting {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    
                    # Clean up expired calls after waiting
                    now = time.time()
                    while self.calls and self.calls[0] <= now - self.time_window:
                        self.calls.popleft()
            
            # Record this call
            self.calls.append(now)
    
    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self.lock:
            self.calls.clear()
    
    def configure(self, max_calls: int, time_window: float) -> None:
        """Update rate limiter configuration."""
        with self.lock:
            self.max_calls = max_calls
            self.time_window = time_window
            self.calls.clear()


# Global rate limiter instance for Gemini API
# Will be configured from config.yaml in main.py
gemini_rate_limiter = RateLimiter(max_calls=8, time_window=60.0)
