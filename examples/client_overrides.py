import asyncio
import os

from genai_pyo3 import ChatMessage, ChatRequest, Client


async def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    client = Client.with_api_key("openai", api_key)

    request = ChatRequest(
        messages=[ChatMessage("user", "Say hello in one short sentence")]
    )
    response = await client.achat("gpt-4o-mini", request)
    print(response.text)


asyncio.run(main())
