# utils/rate_limited_llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from typing import Any, List, Optional
from utils.rate_limiter import gemini_rate_limiter


class RateLimitedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    A wrapper around ChatGoogleGenerativeAI that enforces rate limiting
    before each API call to avoid hitting quota limits.
    """
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Override _generate to add rate limiting at the lowest level."""
        gemini_rate_limiter.acquire()
        return super()._generate(messages, stop, run_manager, **kwargs)
