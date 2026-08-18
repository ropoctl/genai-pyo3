# Migrating to the expanded API

This release replaces the four `Client.with_*` constructors with a builder,
and adds a large amount of previously-unexposed `rust-genai` surface. Only
the client-construction change is breaking; everything else is additive.

## Client construction

The old constructors could configure exactly one provider each, because they
smuggled the key and base URL in through resolver closures gated on a single
adapter kind. `ClientBuilder` makes provider configuration repeatable.

| Before | After |
| --- | --- |
| `Client.with_api_key("openai", key)` | `Client.builder().adapter_kind("openai").provider("openai", auth=AuthData.key(key)).build()` |
| `Client.with_base_url("openai", url)` | `Client.builder().adapter_kind("openai").provider("openai", endpoint=url).build()` |
| `Client.with_api_key_and_base_url("openai", key, url)` | `Client.builder().adapter_kind("openai").provider("openai", endpoint=url, auth=AuthData.key(key)).build()` |
| `Client.with_request_override("openai", url, headers)` | `Client.builder().adapter_kind("openai").provider("openai", auth=AuthData.request_override(url, headers)).build()` |
| `Client()` | unchanged |

Timeouts moved from per-constructor keyword arguments onto `.timeouts()`,
which replaces the whole timeout configuration at once:

```python
# before
Client.with_api_key("openai", key, connect_timeout_seconds=None, read_timeout_seconds=30.0)

# after
(Client.builder()
    .provider("openai", auth=AuthData.key(key))
    .timeouts(connect_seconds=None, read_seconds=30.0)
    .build())
```

`default_headers` likewise moved to `.default_headers({...})`.

## Other breaking changes

- **`Client.provider`** (a string) is now **`Client.adapter_kind`** (an
  `AdapterKind`, or `None` for an unconfigured client).
- **`ChatOptions.sanitize_schema`** is gone. `rust-genai` applies the
  adapter's schema normalization unconditionally, so the value was ignored
  and always read back as `True`.
- **Errors gained subclasses, and one attribute was renamed.**
  `GenaiError` still derives from `RuntimeError` and still carries `kind`,
  `retryable`, `body`, and `headers` exactly as in 0.7.0.1 — but
  `status_code` is now **`status`**, matching rust-genai's own
  `Error::status()`. New: `retry_after` (parsed from `headers`) and
  `model_iden`. And the class is now specific — `RateLimitError`,
  `ServerError`, `AuthError`, ... — so you can branch on the class and log
  the `kind`.

  ```python
  # before
  except GenaiError as err:
      if err.status_code == 429: ...

  # after
  except RateLimitError as err:
      await asyncio.sleep(err.retry_after or 5.0)
  ```

- **Two methods renamed to match rust-genai**: `resolve_target` →
  `resolve_service_target`, `alist_models` → `aall_model_names`. Both were
  introduced in this release, so nothing published used the old names.

- **`ChatResponse` gained `model_iden` / `provider_model_iden`** returning
  `ModelIden`. The flat `model_name` / `model_adapter_kind` strings stay.

## What you can do that you could not before

### One client, many providers, routed per call

```python
client = Client()
registry = {
    "flash": ServiceTarget(
        "https://openrouter.ai/api/v1/",
        AuthData.from_env("OPENROUTER_API_KEY"),
        ModelIden("openai", "deepseek-ai/DeepSeek-V4-Flash-0731"),
    ),
    "local": ServiceTarget(
        "http://localhost:8000/v1/", AuthData.none(), ModelIden("openai", "Qwen/Qwen3.8-27B")
    ),
}

await client.achat(registry["flash"], request)
```

`model` now accepts `str | ModelIden | ServiceTarget` on every call,
including embeddings.

### Retry on the right failures

```python
try:
    response = await client.achat(target, request)
except RateLimitError as err:
    await asyncio.sleep(err.retry_after or 5.0)
except GenaiError as err:
    if not err.is_retryable:
        raise
```

Every error carries `status`, `body`, `model_name`, `adapter_kind`,
`retry_after`, and `is_retryable`.

### Detect truncation

```python
if response.stop_reason == "max_tokens":
    ...  # the answer is cut off, not merely short
```

### Validate configuration at startup

```python
target = client.resolve_service_target("flash")          # no request made
print(target.endpoint.base_url, target.model.model_name)

served = await client.aall_model_names("openai")     # what is loaded right now
```

### Vision and PDF input

```python
ChatMessage("user", content_parts=["what is in this?", Binary.from_path("chart.png")])
```

### Embeddings

```python
response = await client.aembed_batch("text-embedding-3-small", ["a", "b"])
response.vectors  # sorted by input index
```

### Resolver callbacks

```python
(Client.builder()
    .auth_resolver(lambda iden: AuthData.key(vault_cache[iden.adapter_kind.name]))
    .model_mapper(lambda iden: ALIASES.get(iden.model_name, iden.model_name))
    .build())
```

Callbacks must be regular `def`s: they are called synchronously while
holding the GIL, so an `async def` would never be awaited (and is rejected
at registration), and blocking inside one stalls every other Python thread.
