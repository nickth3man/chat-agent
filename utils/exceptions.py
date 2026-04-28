class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class PermanentLLMError(LLMError):
    """Non-retryable LLM error (e.g., 400, 401, 403)."""
    pass


class RetryableLLMError(LLMError):
    """Retryable LLM error (e.g., 500, 502, 503)."""
    pass


class RateLimitError(RetryableLLMError):
    """Rate limit error (429) — retryable after backoff."""
    pass
