# utils/rate_limited_llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import Any, List, Optional, Tuple, Dict, Type
from functools import lru_cache
from utils.rate_limiter import gemini_rate_limiter
import time

# Map string message types to their respective classes
MESSAGE_TYPE_MAP: Dict[str, Type[BaseMessage]] = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
    "chat": BaseMessage, # Fallback for generic chat messages
}

def _hashable_messages(messages: List[BaseMessage]) -> Tuple:
    """Converts a list of BaseMessage objects into a hashable tuple."""
    return tuple((m.type, m.content) for m in messages)

class RateLimitedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    A wrapper around ChatGoogleGenerativeAI that enforces rate limiting
    and adds caching to avoid redundant API calls.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize cache as instance variable
        self._cache: Dict[Tuple, Any] = {}
        self._max_cache_size = 128

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Override _generate to add caching and rate limiting."""
        hashable_messages = _hashable_messages(messages)
        
        # Check cache first
        cache_key = (hashable_messages, tuple(sorted((k, v) for k, v in kwargs.items())) if kwargs else ())
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Apply rate limiting BEFORE the API call
        gemini_rate_limiter.acquire()
        
        # Make the actual API call with FASTER retry logic
        max_retries = 2  # Reduced from 3 to 2
        retry_delay = 1.0  # Reduced from 2.0 to 1.0
        
        for attempt in range(max_retries):
            try:
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                
                # Cache the result
                if len(self._cache) >= self._max_cache_size:
                    # Simple FIFO eviction
                    first_key = next(iter(self._cache))
                    del self._cache[first_key]
                
                self._cache[cache_key] = result
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Don't retry on certain errors
                if 'invalid' in error_msg or 'authentication' in error_msg or 'permission' in error_msg:
                    print(f"    ❌ Non-retryable error: {str(e)[:100]}")
                    raise
                
                if attempt < max_retries - 1:
                    print(f"    ⚠️ Retry {attempt + 1}/{max_retries} in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Gentler backoff (1.5x instead of 2x)
                else:
                    print(f"    ❌ Failed after {max_retries} attempts: {str(e)[:100]}")
                    raise
