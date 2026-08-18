"""Retrying the failures worth retrying, and only those."""

import asyncio
import os

from genai_pyo3 import (
    AuthData,
    ChatMessage,
    ChatRequest,
    Client,
    GenaiError,
    RateLimitError,
)


async def chat_with_retry(client, model, request, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return await client.achat(model, request)
        except RateLimitError as err:
            # The provider told us how long to wait; believe it.
            delay = err.retry_after or 2.0**attempt
            print(f"rate limited, sleeping {delay}s")
            await asyncio.sleep(delay)
        except GenaiError as err:
            # Everything else: retry only what the error says is worth it.
            if not err.is_retryable or attempt == attempts - 1:
                print(f"giving up: {type(err).__name__} status={err.status}")
                raise
            await asyncio.sleep(2.0**attempt)
    raise RuntimeError("unreachable")


async def main() -> None:
    client = (
        Client.builder()
        .adapter_kind("openai")
        .provider("openai", auth=AuthData.from_env("OPENAI_API_KEY"))
        .build()
    )
    request = ChatRequest(messages=[ChatMessage("user", "Say hello in one short sentence")])

    response = await chat_with_retry(client, os.environ.get("MODEL", "gpt-4o-mini"), request)
    print(response.text)


asyncio.run(main())
