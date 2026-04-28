"""
Free web search utility using DuckDuckGo Instant Answer API.
No API key required. Rate-limited to ~1 req/s by default.

Returns:
  list[dict]: [{"title": str, "url": str, "snippet": str}, ...]
"""

import logging
import time
from typing import Optional

from ddgs import DDGS
logger = logging.getLogger(__name__)
from utils.constants import SEARCH_RATE_LIMIT_INTERVAL

_LAST_CALL = 0.0

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
    if elapsed < SEARCH_RATE_LIMIT_INTERVAL:
        time.sleep(SEARCH_RATE_LIMIT_INTERVAL - elapsed)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(max_results, 10)))
    except Exception as e:
        logger.warning("Web search failed for query=%r: %s", query, e)
        results = []
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
