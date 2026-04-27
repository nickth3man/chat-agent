"""
Free web search utility using DuckDuckGo Instant Answer API.
No API key required. Rate-limited to ~1 req/s by default.

Returns:
  list[dict]: [{"title": str, "url": str, "snippet": str}, ...]
"""

import time
from typing import Optional

from duckduckgo_search import DDGS

_LAST_CALL = 0.0
_MIN_INTERVAL = 1.2  # seconds between calls to be polite


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo for a query and return results.

    Args:
        query: Search query string
        max_results: Maximum results (1-10, default 5)

    Returns:
        List of dicts with title, url, snippet keys
    """
    global _LAST_CALL

    elapsed = time.monotonic() - _LAST_CALL
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(max_results, 10)))
    except Exception:
        results = []

    _LAST_CALL = time.monotonic()

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in results
    ]


def search_web_raw(query: str, max_results: int = 5) -> str:
    """Search and return results as a formatted string suitable for LLM context."""
    results = search_web(query, max_results)
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['snippet']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    query = "climate change solutions 2025"
    print(f"Searching: {query}\n")
    print(search_web_raw(query, max_results=3))
