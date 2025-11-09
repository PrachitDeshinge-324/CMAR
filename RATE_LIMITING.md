# Rate Limiting Solution for CMAR LangGraph

## Problem
The application was hitting Gemini's free tier rate limit of **10 requests per minute**, causing `ResourceExhausted` errors during execution.

## Solution Implemented

### 1. **Thread-Safe Rate Limiter** (`utils/rate_limiter.py`)
- Implements a token bucket-style rate limiter
- Automatically throttles API calls to stay within quota
- Thread-safe using locks for concurrent operations
- Configurable via `config.yaml`

### 2. **Rate-Limited LLM Wrapper** (`utils/rate_limited_llm.py`)
- Wraps `ChatGoogleGenerativeAI` with automatic rate limiting
- Transparently intercepts all LLM calls
- No changes needed to agent code

### 3. **Sequential Risk Assessment** 
- Changed from parallel (ThreadPoolExecutor) to sequential execution
- Reduces burst of simultaneous API calls
- Still processes all specialties, just one at a time

## Configuration

Edit `config/config.yaml` to adjust rate limits:

```yaml
gemini:
  rate_limit:
    max_calls: 8        # Max requests allowed
    time_window: 60     # Time window in seconds
```

**Default**: 8 requests per 60 seconds (conservative, below the 10/minute limit)

## How It Works

1. **Before each LLM call**: The rate limiter checks if we're within quota
2. **If at limit**: Automatically waits until a slot becomes available
3. **User feedback**: Prints wait time when throttling occurs
4. **Automatic cleanup**: Old timestamps are removed from tracking

## Expected Behavior

You'll see messages like:
```
⏳ Rate limit reached. Waiting 12.3s before next call...
```

This is **normal and expected** - the system is protecting you from quota errors!

## Benefits

✅ No more `ResourceExhausted` errors  
✅ Automatic handling - no manual intervention needed  
✅ Configurable for different API tiers  
✅ Thread-safe for concurrent operations  
✅ Transparent - works with existing code  

## Upgrading Your API Tier

If you upgrade to a paid Gemini API tier with higher limits, simply update the config:

```yaml
gemini:
  rate_limit:
    max_calls: 60       # Example for higher tier
    time_window: 60
```
