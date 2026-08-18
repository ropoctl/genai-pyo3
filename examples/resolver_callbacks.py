"""Resolving credentials and model names with Python callbacks.

Callbacks are called synchronously, on the thread driving the request,
while holding the GIL. Keep them fast — do the slow part up front and close
over the result, as the credential cache below does.
"""

import asyncio
import os

from genai_pyo3 import AuthData, ChatMessage, ChatRequest, Client

# Fetched once at startup rather than per request: a blocking vault call
# inside the callback would stall every other Python thread.
CREDENTIALS = {"openai": os.environ.get("OPENAI_API_KEY", "sk-missing")}

ALIASES = {"fast": "gpt-4o-mini", "smart": "gpt-4o"}


def resolve_auth(model_iden):
    """fn(ModelIden) -> AuthData | None — None means 'not mine'."""
    key = CREDENTIALS.get(model_iden.adapter_kind.name)
    return AuthData.key(key) if key else None


def map_model(model_iden):
    """fn(ModelIden) -> ModelIden | str — a bare string keeps the adapter."""
    return ALIASES.get(model_iden.model_name, model_iden.model_name)


async def main() -> None:
    client = (
        Client.builder()
        .adapter_kind("openai")
        .auth_resolver(resolve_auth)
        .model_mapper(map_model)
        .build()
    )

    print("fast ->", client.resolve_service_target("fast").model.model_name)

    request = ChatRequest(messages=[ChatMessage("user", "Say hello in one short sentence")])
    response = await client.achat("fast", request)
    print(response.text)


asyncio.run(main())
