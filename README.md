# genai-pyo3

Typed Python bindings for the Rust [`genai`](https://github.com/jeremychone/rust-genai) crate, built with `pyo3` and `maturin`.

This repo uses the upstream GitHub repository for `genai` directly:

```toml
genai = { git = "https://github.com/jeremychone/rust-genai" }
```

## What It Exposes

The Python extension currently exposes:

- `Client`
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
- `await client.astream_chat(model, request, options=None)` for chunked streaming

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
