import os
import logging
import json
import httpx

from utils.http_client import _get_client

logger = logging.getLogger(__name__)


def call_llm_stream(
    prompt: str,
    system: str = "",
    temperature: float = 0.8,
    max_tokens: int = 2048,
    seed: int | None = None,
):
    """Stream LLM output with configurable parameters.

    Args:
        prompt: The user message to send.
        system: Optional system prompt for role/metacognitive framing.
        temperature: Creativity control (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum output tokens (prevents silent truncation).
        seed: Optional seed for reproducible outputs.

    Yields:
        Text chunks as they arrive from the streaming API.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")

    logger.debug("LLM stream: model=%s, temp=%.2f, max_tokens=%d, prompt_len=%d",
                model, temperature, max_tokens, len(prompt))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if seed is not None:
        body["seed"] = seed

    with _get_client().stream(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
    ) as response:
        response.raise_for_status()
        full_text = ""
        for line in response.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        yield content
                except (json.JSONDecodeError, KeyError):
                    continue
        # return sets StopIteration.value — used by callers that consume the full generator
        return full_text


if __name__ == "__main__":
    gen = call_llm_stream("Tell me a short joke.")
    while True:
        try:
            chunk = next(gen)
            print(chunk, end="", flush=True)
        except StopIteration as e:
            print(f"\nFull text: {e.value}")
            break
