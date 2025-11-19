# utils/rate_limited_llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import Any, List, Optional, Tuple, Dict, Type
from functools import lru_cache
from utils.rate_limiter import gemini_rate_limiter
import time
import sqlite3
import pickle
import os
import hashlib

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
    and adds persistent disk-based caching to avoid redundant API calls.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize disk-based cache
        self._cache_dir = os.path.join(os.getcwd(), ".cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        self._db_path = os.path.join(self._cache_dir, "llm_cache.db")
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database for caching."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS llm_responses (
                        cache_key TEXT PRIMARY KEY,
                        response BLOB,
                        timestamp REAL
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"⚠️ Failed to initialize cache DB: {e}")

    def _get_from_cache(self, key: str) -> Any:
        """Retrieve response from SQLite cache."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("SELECT response FROM llm_responses WHERE cache_key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return pickle.loads(row[0])
        except Exception as e:
            # Fail silently on cache read errors to allow fresh generation
            pass
        return None

    def _save_to_cache(self, key: str, response: Any):
        """Save response to SQLite cache."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_responses (cache_key, response, timestamp) VALUES (?, ?, ?)",
                    (key, pickle.dumps(response), time.time())
                )
                conn.commit()
        except Exception as e:
            print(f"⚠️ Failed to save to cache: {e}")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Override _generate to add persistent caching and rate limiting."""
        hashable_messages = _hashable_messages(messages)
        
        # Generate a stable cache key
        cache_key_tuple = (hashable_messages, tuple(sorted((k, v) for k, v in kwargs.items())) if kwargs else ())
        cache_key_bytes = pickle.dumps(cache_key_tuple)
        cache_key = hashlib.sha256(cache_key_bytes).hexdigest()
        
        # Check disk cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Apply rate limiting BEFORE the API call
        gemini_rate_limiter.acquire()
        
        # Make the actual API call with FASTER retry logic
        max_retries = 2  # Reduced from 3 to 2
        retry_delay = 1.0  # Reduced from 2.0 to 1.0
        
        for attempt in range(max_retries):
            try:
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                
                # Save to disk cache
                self._save_to_cache(cache_key, result)
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
