"""Embedding one string, and a batch."""

import asyncio
import os

from genai_pyo3 import AuthData, Client, EmbedOptions


async def main() -> None:
    client = (
        Client.builder()
        .adapter_kind("openai")
        .provider("openai", auth=AuthData.from_env("OPENAI_API_KEY"))
        .build()
    )
    model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

    single = await client.aembed(model, "hello world", EmbedOptions(dimensions=256))
    print(f"one vector, {len(single.embeddings[0])} dimensions")

    batch = await client.aembed_batch(model, ["alpha", "beta", "gamma"])
    # vectors is sorted by input index; providers do not promise order.
    print(f"{len(batch)} vectors, tokens={batch.usage.prompt_tokens}")


asyncio.run(main())
