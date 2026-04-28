import logging
import os
import httpx
from utils.constants import RETRYABLE_HTTP_STATUSES, PERMANENT_HTTP_STATUSES
from utils.exceptions import PermanentLLMError, RetryableLLMError, RateLimitError

from utils.http_client import _get_client

logger = logging.getLogger(__name__)




def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> str:
    """Call OpenRouter LLM with configurable parameters.

    Args:
        prompt: The user message to send.
        system: Optional system prompt for role/metacognitive framing.
        temperature: Creativity control (0.0 = deterministic, 1.0 = creative).
        seed: Optional seed for reproducible outputs.

    Returns:
        The LLM's text response.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")

    logger.debug("LLM call: model=%s, temp=%.2f, prompt_len=%d",
                model, temperature, len(prompt))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    if seed is not None:
        body["seed"] = seed

    response = _get_client().post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
    )

    status = response.status_code
    if status in PERMANENT_HTTP_STATUSES:
        raise PermanentLLMError(f"Permanent HTTP {status}: {response.text[:500]}")
    if status in RETRYABLE_HTTP_STATUSES:
        if status == 429:
            raise RateLimitError(f"Rate limited (429): {response.text[:500]}")
        raise RetryableLLMError(f"Retryable HTTP {status}: {response.text[:500]}")
    # Unclassified error status - treat as permanent to avoid infinite retries
    if status >= 400:
        raise PermanentLLMError(f"Unclassified HTTP {status}: {response.text[:500]}")
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    content = data["choices"][0]["message"]["content"]
    logger.debug("LLM response: len=%d, preview=%r",
                len(content), content[:200])
    return content


if __name__ == "__main__":
    prompt = "What is the meaning of life?"
    print(call_llm(prompt))
