"""Tests for utils/search_web.py — DuckDuckGo web search."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from utils.search_web import search_web, search_web_raw


class TestSearchWeb:
    """Tests for the search_web function."""

    @patch("utils.search_web.DDGS")
    def test_search_web_formats_results(self, mock_ddgs):
        """Should return list of dicts with title, url, snippet keys."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {
                "title": "Test Title",
                "href": "https://example.com",
                "body": "A test snippet.",
            },
            {
                "title": "Another Title",
                "href": "https://example2.com",
                "body": "Another snippet.",
            },
        ]
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        # Bypass rate limiting
        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            results = search_web("test query")
        finally:
            sw._LAST_CALL = original_last_call

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["title"] == "Test Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == "A test snippet."
        assert results[1]["title"] == "Another Title"
        assert results[1]["url"] == "https://example2.com"
        assert results[1]["snippet"] == "Another snippet."

    @patch("utils.search_web.DDGS")
    def test_search_web_converts_missing_fields(self, mock_ddgs):
        """Should use empty string for missing title/href/body fields."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {},  # no fields at all
            {"title": "Only Title"},
            {"href": "https://only-url.com"},
            {"body": "Only Body"},
        ]
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            results = search_web("test query")
        finally:
            sw._LAST_CALL = original_last_call

        assert len(results) == 4
        assert results[0] == {"title": "", "url": "", "snippet": ""}
        assert results[1] == {"title": "Only Title", "url": "", "snippet": ""}
        assert results[2] == {"title": "", "url": "https://only-url.com", "snippet": ""}
        assert results[3] == {"title": "", "url": "", "snippet": "Only Body"}

    @patch("utils.search_web.DDGS")
    def test_handles_empty_results(self, mock_ddgs):
        """Should return empty list for empty results."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            results = search_web("test query")
        finally:
            sw._LAST_CALL = original_last_call

        assert results == []

    @patch("utils.search_web.DDGS")
    def test_handles_search_error(self, mock_ddgs):
        """Should return empty list on exception."""
        mock_instance = MagicMock()
        mock_instance.text.side_effect = RuntimeError("Search failed")
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            results = search_web("test query")
        finally:
            sw._LAST_CALL = original_last_call

        assert results == []

    @patch("utils.search_web.DDGS")
    def test_respects_max_results(self, mock_ddgs):
        """Should pass min(max_results, 10) to DDGS.text."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {"title": f"Result {i}", "href": f"https://example{i}.com", "body": ""}
            for i in range(5)
        ]
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            search_web("test", max_results=3)
        finally:
            sw._LAST_CALL = original_last_call

        mock_instance.text.assert_called_once_with("test", max_results=3)

    @patch("utils.search_web.DDGS")
    def test_clamps_max_results_to_10(self, mock_ddgs):
        """Should clamp max_results to 10."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            search_web("test", max_results=20)
        finally:
            sw._LAST_CALL = original_last_call

        mock_instance.text.assert_called_once_with("test", max_results=10)


class TestSearchWebRaw:
    """Tests for the search_web_raw function."""

    @patch("utils.search_web.DDGS")
    def test_search_web_raw_formats_for_llm(self, mock_ddgs):
        """Should return formatted string with URLs and snippets."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {"title": "First", "href": "https://a.com", "body": "Snippet A"},
            {"title": "Second", "href": "https://b.com", "body": "Snippet B"},
        ]
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            result = search_web_raw("test query", max_results=2)
        finally:
            sw._LAST_CALL = original_last_call

        assert isinstance(result, str)
        assert "[1] First" in result
        assert "URL: https://a.com" in result
        assert "Snippet A" in result
        assert "[2] Second" in result
        assert "URL: https://b.com" in result
        assert "Snippet B" in result

    @patch("utils.search_web.DDGS")
    def test_search_web_raw_no_results(self, mock_ddgs):
        """Should return 'No results found.' for empty results."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            result = search_web_raw("test query")
        finally:
            sw._LAST_CALL = original_last_call

        assert result == "No results found."

    @patch("utils.search_web.DDGS")
    def test_search_web_raw_error(self, mock_ddgs):
        """Should return 'No results found.' on error."""
        mock_instance = MagicMock()
        mock_instance.text.side_effect = RuntimeError("Search failed")
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            sw._LAST_CALL = 0.0
            result = search_web_raw("test query")
        finally:
            sw._LAST_CALL = original_last_call

        assert result == "No results found."


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_variables_exist(self):
        """Rate limiting globals should be defined."""
        import utils.search_web as sw
        assert hasattr(sw, "_LAST_CALL")
        from utils.constants import SEARCH_RATE_LIMIT_INTERVAL
        assert SEARCH_RATE_LIMIT_INTERVAL > 0

    @patch("utils.search_web.DDGS")
    @patch("time.sleep", return_value=None)
    @patch("time.monotonic")
    def test_sleeps_when_interval_not_elapsed(self, mock_monotonic, mock_sleep, mock_ddgs):
        """Should sleep when _MIN_INTERVAL hasn't elapsed since last call."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            # Set _LAST_CALL to a recent time so elapsed < _MIN_INTERVAL
            sw._LAST_CALL = 99.5
            mock_monotonic.side_effect = [100.0]  # now=100, elapsed=0.5 < 1.2

            search_web("test query", max_results=1)

            # Should sleep for (1.2 - 0.5) = 0.7 seconds
            mock_sleep.assert_called_once_with(1.2 - 0.5)
        finally:
            sw._LAST_CALL = original_last_call

    @patch("utils.search_web.DDGS")
    @patch("time.sleep", return_value=None)
    @patch("time.monotonic")
    def test_skips_sleep_when_interval_elapsed(self, mock_monotonic, mock_sleep, mock_ddgs):
        """Should NOT sleep when _MIN_INTERVAL has already elapsed."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs.return_value = mock_instance

        import utils.search_web as sw
        original_last_call = sw._LAST_CALL
        try:
            # _LAST_CALL far in past, elapsed = 100 > 1.2
            sw._LAST_CALL = 0.0
            mock_monotonic.side_effect = [100.0]

            search_web("test query", max_results=1)

            mock_sleep.assert_not_called()
        finally:
            sw._LAST_CALL = original_last_call
