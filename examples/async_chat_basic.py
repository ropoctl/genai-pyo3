import asyncio

from genai_pyo3 import ChatMessage, ChatOptions, ChatRequest, Client


async def main() -> None:
    client = Client()
    request = ChatRequest(
        messages=[ChatMessage("user", "Say hello in one short sentence")]
    )
    options = ChatOptions(temperature=0.2)

    response = await client.achat("gpt-4o-mini", request, options)
    print(response.text)


asyncio.run(main())
