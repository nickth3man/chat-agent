import os
import httpx


def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    seed: int | None = None,
) -> str:
    """Call OpenRouter LLM with configurable parameters.

    Args:
        prompt: The user message to send.
        system: Optional system prompt for role/metacognitive framing.
        temperature: Creativity control (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum output tokens (prevents silent truncation).
        seed: Optional seed for reproducible outputs.

    Returns:
        The LLM's text response.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = seed

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    prompt = "What is the meaning of life?"
    print(call_llm(prompt))
