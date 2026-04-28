import httpx

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """Return a singleton httpx.Client with a 30-second timeout.

    The client is created once and reused across all callers,
    avoiding connection churn and socket exhaustion.
    """
    global _client
    if _client is None:
        _client = httpx.Client(timeout=httpx.Timeout(30.0))
    return _client
