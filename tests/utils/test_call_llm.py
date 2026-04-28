"""Tests for utils/call_llm.py — OpenRouter API client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from utils.call_llm import _get_client, call_llm


def _make_mock_response(content: str = "Hello, world!") -> MagicMock:
    """Create a mock httpx.Response with given content."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    resp.raise_for_status.return_value = None
    return resp


class TestCallLLM:
    """Tests for the call_llm function."""

    def test_builds_correct_request(self, monkeypatch):
        """call_llm should POST to OpenRouter with correct headers and body."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response("Hello, world!")

        with patch("utils.call_llm._get_client", return_value=mock_client):
            result = call_llm("Hello?")

            assert result == "Hello, world!"
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
            body = call_kwargs["json"]
            assert body["model"] == "test-model"
            assert len(body["messages"]) == 1
            assert body["messages"][0]["role"] == "user"
            assert body["messages"][0]["content"] == "Hello?"
            assert body["temperature"] == 0.7
            assert body["max_tokens"] == 2048
            assert "seed" not in body

    def test_handles_http_error(self, monkeypatch):
        """call_llm should raise RetryableLLMError on 500 errors."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        from utils.exceptions import RetryableLLMError
        with patch("utils.call_llm._get_client", return_value=mock_client):
            with pytest.raises(RetryableLLMError):
                call_llm("Hello?")

    def test_handles_auth_error(self, monkeypatch):
        """call_llm should raise PermanentLLMError on 401 Unauthorized."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "bad-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        from utils.exceptions import PermanentLLMError
        with patch("utils.call_llm._get_client", return_value=mock_client):
            with pytest.raises(PermanentLLMError):
                call_llm("Hello?")

    def test_includes_system_prompt_when_provided(self, monkeypatch):
        """System prompt should be added as system role message."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response("Answer")

        with patch("utils.call_llm._get_client", return_value=mock_client):
            call_llm("Hello?", system="You are a helpful assistant.")

            body = mock_client.post.call_args.kwargs["json"]
            assert len(body["messages"]) == 2
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][0]["content"] == "You are a helpful assistant."
            assert body["messages"][1]["role"] == "user"

    def test_includes_seed_when_provided(self, monkeypatch):
        """seed parameter should be added to request body."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response("Answer")

        with patch("utils.call_llm._get_client", return_value=mock_client):
            call_llm("Hello?", seed=42)

            body = mock_client.post.call_args.kwargs["json"]
            assert body["seed"] == 42

    def test_seed_not_included_when_none(self, monkeypatch):
        """seed should NOT be in body when not provided (default None)."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response("Answer")

        with patch("utils.call_llm._get_client", return_value=mock_client):
            call_llm("Hello?")

            body = mock_client.post.call_args.kwargs["json"]
            assert "seed" not in body

    def test_uses_env_vars(self, monkeypatch):
        """Should read OPENROUTER_API_KEY and LLM_MODEL from env."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-test-key")
        monkeypatch.setenv("LLM_MODEL", "env-test-model")

        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response("Answer")

        with patch("utils.call_llm._get_client", return_value=mock_client):
            call_llm("Hello?")

            headers = mock_client.post.call_args.kwargs["headers"]
            body = mock_client.post.call_args.kwargs["json"]
            assert headers["Authorization"] == "Bearer env-test-key"
            assert body["model"] == "env-test-model"

    def test_posts_to_correct_endpoint(self, monkeypatch):
        """Should POST to the OpenRouter chat completions endpoint."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")

        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response("Answer")

        with patch("utils.call_llm._get_client", return_value=mock_client):
            call_llm("Hello?")

            url = mock_client.post.call_args.args[0]
            assert url == "https://openrouter.ai/api/v1/chat/completions"


class TestGetClient:
    """Tests for the _get_client singleton."""

    def test_returns_same_client(self):
        """_get_client should return the same instance on repeated calls."""
        import utils.call_llm as cll
        cll._client = None

        client1 = cll._get_client()
        client2 = cll._get_client()
        assert client1 is client2

    def test_creates_client_with_correct_timeout(self):
        """_get_client should create httpx.Client with 30s timeout."""
        import utils.call_llm as cll
        cll._client = None

        client = cll._get_client()
        assert isinstance(client, httpx.Client)
