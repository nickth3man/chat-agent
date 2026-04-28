"""Tests for utils/call_llm_stream.py — streaming OpenRouter API client."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from utils.call_llm_stream import _get_client, call_llm_stream


def _build_sse_line(data: str | None) -> str:
    """Build a raw SSE event line from a data payload string."""
    if data is None:
        return ": heartbeat"
    return f"data: {data}"


def _mock_sse_stream(lines: list[str]):
    """Create an httpx.Response mock that returns SSE-like lines from iter_lines()."""

    class IterLinesIterator:
        def __init__(self, lines):
            self._lines = lines
            self._i = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._i >= len(self._lines):
                raise StopIteration
            line = self._lines[self._i]
            self._i += 1
            return line

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.iter_lines.return_value = IterLinesIterator(lines)
    mock_response.raise_for_status.return_value = None
    mock_response.status_code = 200
    return mock_response


class TestCallLLMStream:
    """Tests for the call_llm_stream generator function."""

    def test_yields_text_chunks(self, monkeypatch):
        """Should yield text chunks from SSE stream."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
            ),
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": ", "}}]})
            ),
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "world!"}}]})
            ),
            _build_sse_line("[DONE]"),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            generator = call_llm_stream("Hello?")
            chunks = list(generator)

            assert chunks == ["Hello", ", ", "world!"]

    def test_yields_chunks_before_done(self, monkeypatch):
        """All text chunks should be yielded before StopIteration."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "A"}}]})
            ),
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "B"}}]})
            ),
            _build_sse_line("[DONE]"),
            # This comes after DONE, should never be yielded
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "C"}}]})
            ),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            generator = call_llm_stream("Hello?")
            chunks = list(generator)

            assert chunks == ["A", "B"]
            assert "C" not in chunks

    def test_returns_full_text_via_stop_iteration(self, monkeypatch):
        """StopIteration.value should contain full accumulated text."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
            ),
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": " world"}}]})
            ),
            _build_sse_line("[DONE]"),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            generator = call_llm_stream("Hello?")
            full_text = ""
            for chunk in generator:
                full_text += chunk

            assert full_text == "Hello world"

    def test_skips_malformed_json(self, monkeypatch):
        """Should skip chunks that fail JSON parsing."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "Good"}}]})
            ),
            _build_sse_line("not-valid-json{{{"),  # malformed
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": " chunk"}}]})
            ),
            _build_sse_line("[DONE]"),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            generator = call_llm_stream("Hello?")
            chunks = list(generator)

            assert chunks == ["Good", " chunk"]

    def test_skips_empty_delta(self, monkeypatch):
        """Should skip delta entries with no content."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
            ),
            _build_sse_line(
                json.dumps({"choices": [{"delta": {}}]})  # no content key
            ),
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": ""}}]})  # empty content
            ),
            _build_sse_line("[DONE]"),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            generator = call_llm_stream("Hello?")
            chunks = list(generator)

            assert chunks == ["Hello"]

    def test_handles_empty_stream(self, monkeypatch):
        """Should handle response with only [DONE] gracefully."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            _build_sse_line("[DONE]"),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            generator = call_llm_stream("Hello?")
            chunks = list(generator)

            assert chunks == []

    def test_includes_system_prompt(self, monkeypatch):
        """System prompt should be included in request body."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "Hi"}}]})
            ),
            _build_sse_line("[DONE]"),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            list(call_llm_stream("Hello?", system="You are helpful."))

            _, call_kwargs = mock_client.stream.call_args
            body = call_kwargs["json"]
            assert len(body["messages"]) == 2
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][0]["content"] == "You are helpful."
            assert body["stream"] is True

    def test_includes_seed_when_provided(self, monkeypatch):
        """seed should be in request body when provided."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "Hi"}}]})
            ),
            _build_sse_line("[DONE]"),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            list(call_llm_stream("Hello?", seed=123))

            _, call_kwargs = mock_client.stream.call_args
            body = call_kwargs["json"]
            assert body["seed"] == 123

    def test_handles_http_error(self, monkeypatch):
        """Should raise on HTTP errors from the stream."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.iter_lines.return_value = iter([])
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                list(call_llm_stream("Hello?"))

    def test_skips_non_data_lines(self, monkeypatch):
        """Lines not starting with 'data: ' should be ignored."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        sse_lines = [
            ":comment line",  # SSE comment — skipped
            "",  # empty line — skipped
            _build_sse_line(
                json.dumps({"choices": [{"delta": {"content": "Hi"}}]})
            ),
            _build_sse_line("[DONE]"),
        ]
        mock_response = _mock_sse_stream(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__.return_value = mock_response

        with patch("utils.call_llm_stream._get_client", return_value=mock_client):
            generator = call_llm_stream("Hello?")
            chunks = list(generator)

            assert chunks == ["Hi"]


class TestGetClientStream:
    """Tests for _get_client singleton in http_client module."""

    def test_returns_same_client(self):
        """_get_client should return the same instance on repeated calls."""
        import utils.http_client as hc
        hc._client = None

        client1 = hc._get_client()
        client2 = hc._get_client()
        assert client1 is client2
