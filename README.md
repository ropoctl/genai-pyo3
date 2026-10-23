# genai-pyo3

Typed Python bindings for the Rust [`genai`](https://github.com/jeremychone/rust-genai) crate, built with `pyo3` and `maturin`.

This release uses upstream `genai` v0.7.1. The Bedrock SigV4 feature is enabled
in the extension:

```toml
genai = { git = "https://github.com/jeremychone/rust-genai", tag = "v0.7.1", features = ["bedrock-sigv4"] }
```

## What It Exposes

**Routing** — `AdapterKind`, `ModelIden`, `Endpoint`, `AuthData`, `ServiceTarget`

**Client** — `Client`, `ClientBuilder`

**Requests** — `ChatRequest`, `ChatMessage`, `ChatOptions`, `Tool`, `ToolCall`, `JsonSpec`, `Binary`

**Responses** — `ChatResponse`, `StreamEnd`, `ChatStreamEvent`, `Usage` (with prompt/completion/cache-creation detail)

**Embeddings** — `EmbedOptions`, `EmbedResponse`, `Embedding`

**Errors** — `GenaiError` and its subclasses (see below)

Entry points:

| | |
| --- | --- |
| `await client.achat(model, request, options=None)` | a full response |
| `await client.astream_chat(model, request, options=None)` | an async iterator of events |
| `await client.achat_via_stream(model, request, options=None)` | stream under the hood, one response out |
| `client.chat(model, request, options=None)` | blocking |
| `await client.aembed(model, text, options=None)` | one embedding |
| `await client.aembed_batch(model, texts, options=None)` | many |
| `client.resolve_service_target(model)` | where a call would go, without making it |
| `await client.aall_model_names(adapter_kind)` | what a provider is serving |

`model` is `str | ModelIden | ServiceTarget` everywhere.

### Vertex and Bedrock

Provider namespaces route through the native 0.7.1 adapters:

```python
from genai_pyo3 import AuthData, ChatMessage, ChatRequest, Client

request = ChatRequest(messages=[ChatMessage("user", "Hello")])

# Vertex uses a GCP OAuth bearer token; VERTEX_PROJECT_ID and optionally
# VERTEX_LOCATION control the project and region.
vertex = Client.builder().provider("vertex", auth=AuthData.from_env("VERTEX_TOKEN")).build()
await vertex.achat("vertex::gemini-2.5-flash", request)
await vertex.achat("vertex::claude-sonnet-4-6", request)

# Bedrock API keys use BEDROCK_API_KEY (or AWS_BEARER_TOKEN_BEDROCK).
await Client().achat("bedrock_api::us.anthropic.claude-sonnet-4-6", request)

# SigV4 uses the AWS credential chain. Select a named profile per client
# with AuthData.key("profile-name") when needed.
sigv4 = Client.builder().provider("bedrock_sigv4", auth=AuthData.key("my-profile")).build()
await sigv4.achat("bedrock_sigv4::us.anthropic.claude-sonnet-4-5-20250929-v1:0", request)
```

For Vertex, refresh the OAuth token before expiry through an auth resolver;
`AuthData.from_env` reads an already available token. `ChatOptions` also accepts
`stream_header_timeout` and `stream_read_timeout` in seconds for long-running
streams. `AdapterKind.all()` includes the 0.7.1 provider names, including
Mistral, Requesty, API Route, and Gemini Interactions.

## Install

Editable install with `uv`:

```bash
uv pip install -e .
```

This builds the `pyo3` extension and makes `genai_pyo3` importable from the active environment.

## Quick Start

```python
import asyncio
from genai_pyo3 import AuthData, ChatMessage, ChatRequest, Client


async def main() -> None:
    client = (
        Client.builder()
        .adapter_kind("openai")
        .provider("openai", auth=AuthData.from_env("OPENAI_API_KEY"))
        .build()
    )
    request = ChatRequest(messages=[ChatMessage("user", "Say hello in one short sentence")])

    response = await client.achat("gpt-4o-mini", request)
    print(response.text)


asyncio.run(main())
```

## Named endpoints

A `ServiceTarget` states where to go, how to authenticate, and which model to
ask for. A registry of them is just a dict, and one client serves all of it:

```python
from genai_pyo3 import AuthData, Client, ModelIden, ServiceTarget

client = Client()
registry = {
    "flash": ServiceTarget(
        "https://openrouter.ai/api/v1/",
        AuthData.from_env("OPENROUTER_API_KEY"),
        ModelIden("openai", "deepseek-ai/DeepSeek-V4-Flash-0731"),
    ),
    "local": ServiceTarget(
        "http://localhost:8000/v1/",
        AuthData.none(),
        ModelIden("openai", "Qwen/Qwen3.8-27B"),
    ),
}

response = await client.achat(registry["flash"], request)
```

`AuthData` never reveals a key to Python and redacts it in `repr()`, so
logging a `ServiceTarget` does not log a credential.

## Errors

```text
GenaiError(RuntimeError)
├── ConfigError        malformed request/options, adapter mismatch
├── AuthError          missing or unresolvable credentials
├── ResolverError      a resolver (including a Python callback) failed
├── ApiError           the provider answered with a failure status
│   ├── BadRequestError    4xx other than 429
│   ├── RateLimitError     429
│   └── ServerError        5xx
├── TransportError     connect/timeout/socket failures, dropped streams
├── StreamError        malformed or error events mid-stream
├── ResponseError      a well-formed reply that could not be interpreted
└── UnsupportedError   feature unavailable on this adapter/model
```

Every instance carries:

| | |
| --- | --- |
| `kind` | the precise failure in rust-genai's own vocabulary — the error variant in snake_case (`"http_error"`, `"no_auth_data"`), or the provider's own error type for a mid-stream failure (`"overloaded_error"`) |
| `status` | HTTP status, when the provider answered with one |
| `status_code` | compatibility alias for `status` |
| `retryable` | conservative hint; the retry and backoff policy is yours |
| `body` | response body, when there was one |
| `headers` | response headers, when the failure carried an HTTP response |
| `retry_after` | seconds, parsed from `headers` when present |
| `model_iden` | the model the failure is about, when the error names one |

The exception class is the actionable category; `kind` is the exact cause.
Branch on the class, log the `kind`.

```python
try:
    response = await client.achat(target, request)
except RateLimitError as err:
    await asyncio.sleep(err.retry_after or 5.0)
```

## Truncation

`stop_reason` is normalised across providers — `"completed"`,
`"max_tokens"`, `"tool_call"`, `"content_filter"`, `"stop_sequence"`,
`"other"` — with the provider's own wording kept in `stop_reason_raw`.
Without it, a truncated answer is indistinguishable from a short one.

## Images and PDFs

```python
from genai_pyo3 import Binary, ChatMessage

ChatMessage("user", content_parts=["what is in this?", Binary.from_path("chart.png")])
```

## Transport

```python
(Client.builder()
    .proxy("http://proxy.corp:3128", scheme="https")   # "all" | "http" | "https"
    .timeouts(connect_seconds=10.0, read_seconds=120.0)
    .gzip(False)
    .tcp_nodelay(False)
    .build())
```

Without an explicit proxy, requests ignore whatever proxy environment the
process was started with — which on a locked-down network looks like an
unexplained connect timeout.

`sanitize_json_schema(schema, dialect)` applies a provider's
constrained-decoding normalization without making a request, for seeing
what will actually be enforced before a rejection tells you.

## Resolver callbacks

For credentials or routing that cannot be stated up front:

```python
(Client.builder()
    .auth_resolver(lambda iden: AuthData.key(cache[iden.adapter_kind.name]))
    .model_mapper(lambda iden: ALIASES.get(iden.model_name, iden.model_name))
    .service_target_resolver(lambda target: reroute(target))
    .build())
```

Callbacks must be regular `def`s. They are called synchronously while
holding the GIL, so an `async def` would never be awaited — it is rejected
at registration — and blocking inside one stalls every other Python thread.
Cache in Python rather than doing I/O in a callback.

## Development

```bash
maturin develop          # build the extension into the active venv
python -m pytest         # Python tests (no network; a local fake provider)

# Rust tests link against libpython, so they need the extension-module
# feature off and the interpreter feature on:
cargo test --no-default-features --features test-interpreter
```

`cargo build` does not link the cdylib on macOS (it lacks maturin's link
arguments); use `cargo check` or `maturin develop`.

## Migrating

See [MIGRATION.md](MIGRATION.md) for the move from the `Client.with_*`
constructors to `ClientBuilder`.
