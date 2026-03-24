import asyncio
import os

from genai_pyo3 import ChatMessage, ChatOptions, ChatRequest, Client


def normalize_base_url(base_url: str) -> str:
    return base_url if base_url.endswith("/") else f"{base_url}/"


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "test-key")
    base_url = os.environ.get("GEMINI_BASE_URL")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    if base_url:
        client = Client.with_api_key_and_base_url(
            "gemini", api_key, normalize_base_url(base_url)
        )
    else:
        client = Client.with_api_key("gemini", api_key)

    request = ChatRequest(
        messages=[ChatMessage("user", "Reply with exactly: gemini stream ok")]
    )
    options = ChatOptions(capture_content=True, capture_usage=True)

    stream = await client.astream_chat(model, request, options)
    async for event in stream:
        if event.kind == "chunk" and event.content:
            print(event.content, end="", flush=True)
        elif event.kind == "end":
            print()


asyncio.run(main())
