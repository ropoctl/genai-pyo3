import asyncio

from genai_pyo3 import ChatMessage, ChatOptions, ChatRequest, Client


async def main() -> None:
    client = Client()
    request = ChatRequest(
        messages=[ChatMessage("user", "Write three short bullet points about Rust")]
    )
    options = ChatOptions(capture_content=True, capture_usage=True)

    stream = await client.astream_chat("gpt-4o-mini", request, options)
    async for event in stream:
        print(event.kind, event.content)


asyncio.run(main())
