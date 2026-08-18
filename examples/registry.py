"""One client, several named endpoints, chosen per call.

A ServiceTarget states where to go, how to authenticate, and which model to
ask for — so a registry of named endpoints is a plain dict, and a single
Client serves all of them.
"""

import asyncio
import os

from genai_pyo3 import AuthData, ChatMessage, ChatRequest, Client, ModelIden, ServiceTarget

REGISTRY = {
    "flash": ServiceTarget(
        "https://openrouter.ai/api/v1/",
        AuthData.from_env("OPENROUTER_API_KEY"),
        ModelIden("openai", "deepseek-ai/DeepSeek-V4-Flash-0731"),
    ),
    "local": ServiceTarget(
        os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1/"),
        AuthData.none(),
        ModelIden("openai", os.environ.get("LOCAL_MODEL", "Qwen/Qwen3.8-27B")),
    ),
}


async def main() -> None:
    client = Client()
    name = os.environ.get("ENDPOINT", "local")
    target = REGISTRY[name]

    # Check where the call lands before spending a request on it.
    resolved = client.resolve_target(target)
    print(f"{name} -> {resolved.endpoint.base_url} ({resolved.model.model_name})")

    request = ChatRequest(messages=[ChatMessage("user", "Reply with exactly: registry ok")])
    response = await client.achat(target, request)

    print(response.text)
    print(f"stop_reason={response.stop_reason}")


asyncio.run(main())
