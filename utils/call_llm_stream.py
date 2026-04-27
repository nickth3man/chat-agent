import os
import json
import httpx


def call_llm_stream(prompt: str):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")

    with httpx.stream(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
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
