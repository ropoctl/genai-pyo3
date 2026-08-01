# genai-pyo3

Typed Python bindings for the Rust [`genai`](https://github.com/jeremychone/rust-genai) crate, built with `pyo3` and `maturin`.

This repo uses the upstream GitHub repository for `genai` directly:

```toml
genai = { git = "https://github.com/jeremychone/rust-genai" }
```

## What It Exposes

The Python extension currently exposes:

- `Client`
- `GenaiError`
- `ChatMessage`
- `ChatRequest`
- `ChatOptions`
- `ChatResponse`
- `ChatStreamEvent`
- `StreamEnd`
- `ToolCall`
- `Usage`

The main async entry points are:

- `await client.achat(model, request, options=None)` for a full response
- `await client.achat_via_stream(model, request, options=None)` for a fully
  collected response from a streaming-only endpoint
- `await client.astream_chat(model, request, options=None)` for chunked streaming

`ChatResponse.reasoning_content` exposes captured readable reasoning, while
`ChatResponse.thought_signatures` exposes encrypted reasoning signatures that
must be retained when replaying an assistant turn. Both are also included by
`ChatResponse.to_dict()`.

For a signed-thinking tool continuation, carry both values onto the assistant
message along with the returned tool calls:

```python
assistant = ChatMessage(
    "assistant",
    "",
    tool_calls=response.tool_calls,
    thought_signatures=response.thought_signatures,
    reasoning_content=response.reasoning_content,
)
```

## Install

Editable install with `uv`:

```bash
uv pip install -e .
```

This builds the `pyo3` extension and makes `genai_pyo3` importable from the active environment.

## Quick Start

Minimal async call:

```python
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
```

## Errors and retries

Model-call failures raise `GenaiError`, a `RuntimeError` subclass with structured
metadata copied from rust-genai:

```python
from genai_pyo3 import GenaiError

try:
    response = await client.achat(model, request)
except GenaiError as error:
    print(error.kind)         # e.g. "http_error", "timeout", "web_stream"
    print(error.status_code)  # e.g. 429 or 503; None for transport failures
    print(error.retryable)    # True for transport failures, 429, and 5xx
    print(error.headers)      # response headers when rust-genai retained them
    print(error.body)         # response body when rust-genai retained it
```

The library reports retryability but does not retry calls itself. This keeps
retry budgets and backoff policy under application control. A response-body
stream decode failure generally has no HTTP error status because the server
already returned a successful status; it is exposed as a retryable transport
error instead.

Minimal async streaming call:

```python
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
```

## Examples

- `examples/async_chat_basic.py`: basic `achat` example
- `examples/async_stream_chat.py`: basic `astream_chat` example
- `examples/client_overrides.py`: explicit provider/API-key override example
- `examples/gemini_env.py`: Gemini `achat` example using env vars
- `examples/gemini_stream_env.py`: Gemini streaming example using env vars

### Run The Gemini Examples

The Gemini examples use:

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (optional, defaults to `gemini-2.0-flash`)
- `GEMINI_BASE_URL` (optional, for custom or local Gemini-compatible endpoints)

Non-streaming:

```bash
GEMINI_API_KEY=your-key python examples/gemini_env.py
```

Streaming:

```bash
GEMINI_API_KEY=your-key python examples/gemini_stream_env.py
```

With a custom endpoint:

```bash
GEMINI_API_KEY=your-key \
GEMINI_BASE_URL=http://your-host/v1beta \
python examples/gemini_env.py
```

```bash
GEMINI_API_KEY=your-key \
GEMINI_BASE_URL=http://your-host/v1beta \
python examples/gemini_stream_env.py
```

If `GEMINI_BASE_URL` is unset, the examples use the default Gemini endpoint resolved by the underlying `genai` client.

## Layout

- `Cargo.toml`: Rust extension crate
- `pyproject.toml`: Python package metadata and `maturin` config
- `python/genai_pyo3`: Python package entrypoint
- `src/lib.rs`: `pyo3` binding implementation
- `examples/`: small Python usage examples

## Local Development

Create a virtualenv and install in editable mode:

```bash
cd genai-pyo3
python -m venv .venv
source .venv/bin/activate
uv pip install -e .
```

If you want stricter reproducibility, pin that dependency to a branch, tag, or revision.
