import asyncio
import os

from genai_pyo3 import AuthData, ChatMessage, ChatRequest, Client


async def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    client = (
        Client.builder()
        .adapter_kind("openai")
        .provider("openai", auth=AuthData.key(api_key))
        .build()
    )

    request = ChatRequest(
        messages=[ChatMessage("user", "Say hello in one short sentence")]
    )
    response = await client.achat("gpt-4o-mini", request)
    print(response.text)


asyncio.run(main())
