//! The `Client` pyclass and its builder.

use futures::StreamExt;
use genai::ModelIden;
use genai::adapter::AdapterKind;
use genai::chat::{ChatOptions, ChatStreamEvent, StreamEnd};
use genai::embed::EmbedOptions;
use genai::resolver::{AuthData, AuthResolver, Endpoint, ModelMapper, ServiceTargetResolver};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

use crate::chat::{coerce_chat_options, coerce_chat_request};
use crate::convert::build_header_map;
use crate::embed::{coerce_embed_options, to_py_embed_response};
use crate::errors::to_py_err;
use crate::resolvers::{PyCallback, auth_resolver_fn, model_mapper_fn, service_target_resolver_fn};
use crate::response::{
    PyChatResponse, PyChatStream, stop_reason_parts, to_py_chat_response, to_py_usage,
};
use crate::target::{
    AdapterKindLike, EndpointLike, ModelLike, PyAdapterKind, PyAuthData, PyModelIden,
    PyServiceTarget,
};

/// Connect-timeout applied unless `ClientBuilder.timeouts(...)` says
/// otherwise. Thirty seconds covers any realistic TCP+TLS handshake while
/// still surfacing a hung proxy or unreachable endpoint before it eats
/// through a whole run.
const DEFAULT_CONNECT_TIMEOUT_SECONDS: f64 = 30.0;

/// Endpoint and credential overrides for one adapter.
#[derive(Clone, Default)]
struct ProviderOverride {
    endpoint: Option<Endpoint>,
    auth: Option<AuthData>,
}

/// Python-facing wrapper around `genai::Client`.
///
/// A bare `Client()` routes per call: hand `achat` a `ServiceTarget` and it
/// goes exactly there, or hand it a model name and rust-genai infers the
/// adapter and resolves credentials from the environment.
///
/// `Client.builder()` adds configuration on top — per-provider endpoints
/// and credentials, default chat options, timeouts, headers.
#[pyclass(name = "Client")]
pub(crate) struct PyClient {
    inner: genai::Client,
    /// The adapter this client is pinned to via `ClientBuilder.adapter_kind`,
    /// or `None` when routing is decided per call.
    bound_adapter_kind: Option<AdapterKind>,
    /// The declarative per-provider endpoints and credentials this client
    /// was built with.
    ///
    /// Chat routing gets these through the resolvers rust-genai installs,
    /// but `all_model_names` takes a `ProviderConfig` argument instead and
    /// never consults the service-target resolver — so without keeping the
    /// table here, listing models would quietly ignore a configured
    /// endpoint and query the adapter's public default instead.
    provider_overrides: HashMap<AdapterKind, ProviderOverride>,
}

#[pymethods]
impl PyClient {
    /// A client with no configuration: every call must carry enough
    /// routing information of its own.
    #[new]
    fn new() -> Self {
        Self {
            inner: genai::Client::default(),
            bound_adapter_kind: None,
            provider_overrides: HashMap::new(),
        }
    }

    /// Start configuring a client.
    #[staticmethod]
    fn builder() -> PyClientBuilder {
        PyClientBuilder::default()
    }

    /// The adapter this client is pinned to, if any.
    #[getter]
    fn adapter_kind(&self) -> Option<PyAdapterKind> {
        self.bound_adapter_kind.map(|inner| PyAdapterKind { inner })
    }

    fn __repr__(&self) -> String {
        match self.bound_adapter_kind {
            Some(kind) => format!("Client(adapter_kind={:?})", kind.as_lower_str()),
            None => "Client(unbound)".to_string(),
        }
    }

    /// Work out exactly where a call would go, without making it.
    ///
    /// Runs the same resolution the request path runs — adapter inference,
    /// model mapping, auth and endpoint resolvers — and hands back the
    /// resulting `ServiceTarget`. Useful for validating configuration at
    /// startup and for answering "which provider is this actually hitting?"
    /// without spending a request.
    fn aresolve_service_target<'py>(
        &self,
        py: Python<'py>,
        model: ModelLike,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model_spec = model.into_spec();
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let target = client
                .resolve_service_target(model_spec)
                .await
                .map_err(to_py_err)?;
            Ok(PyServiceTarget { inner: target })
        })
    }

    /// Blocking form of
    /// [`aresolve_service_target`](Self::aresolve_service_target).
    fn resolve_service_target(
        &self,
        py: Python<'_>,
        model: ModelLike,
    ) -> PyResult<PyServiceTarget> {
        let target =
            block_on_shared_runtime(py, self.inner.resolve_service_target(model.into_spec()))?
                .map_err(to_py_err)?;
        Ok(PyServiceTarget { inner: target })
    }

    /// Which adapter a bare model name infers to, before any mapping.
    fn default_model(&self, model_name: &str) -> PyResult<PyModelIden> {
        self.inner
            .default_model(model_name)
            .map(|inner| PyModelIden { inner })
            .map_err(to_py_err)
    }

    /// Ask a provider which models it is currently serving.
    ///
    /// Beats a hand-maintained table for local serving stacks and
    /// gateways, where the model list is whatever was loaded this morning.
    /// `endpoint` and `auth` override the client's configuration for this
    /// lookup only.
    #[pyo3(signature = (adapter_kind, *, endpoint = None, auth = None))]
    fn aall_model_names<'py>(
        &self,
        py: Python<'py>,
        adapter_kind: AdapterKindLike,
        endpoint: Option<EndpointLike>,
        auth: Option<PyAuthData>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kind = adapter_kind.resolve()?;
        let configured = self.provider_overrides.get(&kind);
        // Explicit arguments win; otherwise use whatever this client was
        // configured with for that provider, so listing models goes to the
        // same place chatting would.
        let endpoint = match endpoint {
            Some(endpoint) => Some(endpoint.resolve()?),
            None => configured.and_then(|entry| entry.endpoint.clone()),
        };
        let auth = match auth {
            Some(auth) => Some(auth.inner),
            None => configured.and_then(|entry| entry.auth.clone()),
        };
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client
                .all_model_names(kind, (endpoint, auth))
                .await
                .map_err(to_py_err)
        })
    }

    /// Embed a single string.
    #[pyo3(signature = (model, input, options = None))]
    fn aembed<'py>(
        &self,
        py: Python<'py>,
        model: ModelLike,
        input: String,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let embed_options = coerce_embed_options(options)?;
        let model_spec = model.into_spec();
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = client
                .embed(model_spec, input, embed_options.as_ref())
                .await
                .map_err(to_py_err)?;
            Ok(to_py_embed_response(response))
        })
    }

    /// Embed several strings in one request.
    ///
    /// Providers do not guarantee response order, so each `Embedding`
    /// carries the `index` of the input it belongs to; `vectors` sorts by
    /// it for you.
    #[pyo3(signature = (model, inputs, options = None))]
    fn aembed_batch<'py>(
        &self,
        py: Python<'py>,
        model: ModelLike,
        inputs: Vec<String>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let embed_options = coerce_embed_options(options)?;
        let model_spec = model.into_spec();
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = client
                .embed_batch(model_spec, inputs, embed_options.as_ref())
                .await
                .map_err(to_py_err)?;
            Ok(to_py_embed_response(response))
        })
    }

    #[pyo3(signature = (model, request, options = None))]
    fn chat<'py>(
        &self,
        py: Python<'py>,
        model: ModelLike,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyChatResponse>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options)?;

        let chat_res = block_on_shared_runtime(
            py,
            self.inner
                .exec_chat(model.into_spec(), chat_req, chat_options.as_ref()),
        )?
        .map_err(to_py_err)?;

        to_py_chat_response(py, chat_res)
    }

    #[pyo3(signature = (model, request, options = None))]
    fn achat<'py>(
        &self,
        py: Python<'py>,
        model: ModelLike,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options)?;
        let model_spec = model.into_spec();
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chat_res = client
                .exec_chat(model_spec, chat_req, chat_options.as_ref())
                .await
                .map_err(to_py_err)?;

            Python::attach(|py| to_py_chat_response(py, chat_res))
        })
    }

    #[pyo3(signature = (model, request, options = None))]
    fn astream_chat<'py>(
        &self,
        py: Python<'py>,
        model: ModelLike,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options)?;
        let model_spec = model.into_spec();
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream_res = client
                .exec_chat_stream(model_spec, chat_req, chat_options.as_ref())
                .await
                .map_err(to_py_err)?;

            Python::attach(|py| {
                Py::new(py, PyChatStream::new(stream_res.stream)).map_err(|err| {
                    PyRuntimeError::new_err(format!("failed to build stream object: {err}"))
                })
            })
        })
    }

    /// Run the request as a stream and return a fully-collected
    /// :class:`ChatResponse`. Useful for providers whose backend only
    /// accepts ``stream=true`` (e.g. OpenAI's Responses API) but whose
    /// callers still want a single synchronous result.
    ///
    /// `capture_content`, `capture_tool_calls`, and `capture_usage` are
    /// forced on so the terminal `StreamEnd` carries the aggregated
    /// content — callers do not need to configure capture themselves.
    /// `capture_reasoning_content` is NOT forced: it stays opt-in, same
    /// as rust-genai itself. On OpenAI Responses it triggers the
    /// `reasoning.summary` request parameter, which some models reject
    /// outright (gpt-5.3-codex-spark returns `unsupported_parameter`) —
    /// forcing it here made basic requests to those models fail.
    #[pyo3(signature = (model, request, options = None))]
    fn achat_via_stream<'py>(
        &self,
        py: Python<'py>,
        model: ModelLike,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let mut chat_options = coerce_chat_options(options)?.unwrap_or_default();
        chat_options.capture_content = Some(true);
        chat_options.capture_tool_calls = Some(true);
        chat_options.capture_usage = Some(true);
        let model_spec = model.into_spec();
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream_res = client
                .exec_chat_stream(model_spec, chat_req, Some(&chat_options))
                .await
                .map_err(to_py_err)?;

            let model_iden = stream_res.model_iden.clone();
            let mut stream = stream_res.stream;
            let mut end: Option<StreamEnd> = None;
            let mut streamed_thought_signatures = Vec::new();
            while let Some(event) = stream.next().await {
                let event = event.map_err(to_py_err)?;
                match event {
                    ChatStreamEvent::ThoughtSignatureChunk(chunk) => {
                        streamed_thought_signatures.push(chunk.content);
                    }
                    ChatStreamEvent::End(e) => {
                        end = Some(e);
                        break;
                    }
                    _ => {}
                }
            }

            let end = end.ok_or_else(|| {
                PyRuntimeError::new_err("stream ended without a terminal End event")
            })?;
            let mut content = end
                .captured_content
                .unwrap_or_else(genai::chat::MessageContent::default);
            let streamed_thought_signatures = normalize_streamed_thought_signatures(
                &model_iden.adapter_kind,
                streamed_thought_signatures,
            );
            preserve_streamed_thought_signatures(&mut content, streamed_thought_signatures);
            let reasoning_content = end.captured_reasoning_content;
            let usage = end.captured_usage.unwrap_or_default();
            let response_id = end.captured_response_id;
            // Carry the stream's stop reason onto the collected response so
            // this path can report truncation like the non-streaming one.
            let (stop_reason, stop_reason_raw) = match end.captured_stop_reason.as_ref() {
                Some(reason) => {
                    let (name, raw) = stop_reason_parts(reason);
                    (Some(name.to_string()), Some(raw))
                }
                None => (None, None),
            };

            Python::attach(|py| {
                Py::new(
                    py,
                    PyChatResponse {
                        inner_content: content,
                        reasoning_content,
                        model_adapter_kind: model_iden.adapter_kind.to_string(),
                        model_name: model_iden.model_name.to_string(),
                        provider_model_adapter_kind: model_iden.adapter_kind.to_string(),
                        provider_model_name: model_iden.model_name.to_string(),
                        usage: to_py_usage(&usage),
                        stop_reason,
                        stop_reason_raw,
                        response_id,
                        // Streaming has no single raw body to capture.
                        captured_raw_body: None,
                    },
                )
            })
        })
    }
}

/// Chainable configuration for a [`PyClient`].
///
/// ```python
/// client = (
///     Client.builder()
///     .provider("openai", endpoint="https://openrouter.ai/api/v1/",
///               auth=AuthData.from_env("OPENROUTER_API_KEY"))
///     .provider("ollama", endpoint="http://localhost:11434/v1/")
///     .chat_options({"temperature": 0.2})
///     .build()
/// )
/// ```
///
/// `provider(...)` is repeatable, so one client can reach every configured
/// provider. Building does not consume the builder — it can be reused to
/// produce several clients.
#[pyclass(name = "ClientBuilder")]
#[derive(Default)]
pub(crate) struct PyClientBuilder {
    adapter_kind: Option<AdapterKind>,
    providers: HashMap<AdapterKind, ProviderOverride>,
    chat_options: Option<ChatOptions>,
    connect_timeout_seconds: Option<f64>,
    read_timeout_seconds: Option<f64>,
    total_timeout_seconds: Option<f64>,
    default_headers: Option<HashMap<String, String>>,
    timeouts_configured: bool,
    embed_options: Option<EmbedOptions>,
    proxy_url: Option<String>,
    /// Which traffic the proxy handles: "all", "http", or "https". Only
    /// meaningful alongside `proxy_url`.
    proxy_scheme: String,
    gzip: Option<bool>,
    tcp_nodelay: Option<bool>,
    auth_callback: Option<PyCallback>,
    model_mapper_callback: Option<PyCallback>,
    service_target_callback: Option<PyCallback>,
}

#[pymethods]
impl PyClientBuilder {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Pin routing to a single adapter, so bare model names never need a
    /// `"provider::"` prefix. A model that resolves elsewhere is then an
    /// error rather than a silent reroute that would drop this client's
    /// configured credentials.
    fn adapter_kind<'py>(
        mut slf: PyRefMut<'py, Self>,
        adapter_kind: AdapterKindLike,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.adapter_kind = Some(adapter_kind.resolve()?);
        Ok(slf)
    }

    /// Configure one provider's endpoint and/or credential. Repeatable —
    /// call once per provider the client should be able to reach.
    #[pyo3(signature = (adapter_kind, *, endpoint = None, auth = None))]
    fn provider<'py>(
        mut slf: PyRefMut<'py, Self>,
        adapter_kind: AdapterKindLike,
        endpoint: Option<EndpointLike>,
        auth: Option<PyAuthData>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let kind = adapter_kind.resolve()?;
        let endpoint = endpoint.map(EndpointLike::resolve).transpose()?;
        let entry = slf.providers.entry(kind).or_default();
        if endpoint.is_some() {
            entry.endpoint = endpoint;
        }
        if let Some(auth) = auth {
            entry.auth = Some(auth.inner);
        }
        Ok(slf)
    }

    /// Default chat options for every call. Per-call options win.
    fn chat_options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.chat_options = coerce_chat_options(Some(options))?;
        Ok(slf)
    }

    /// Replace the whole timeout configuration.
    ///
    /// Every call sets all three, so `timeouts(connect_seconds=None)` is
    /// how the default connect timeout gets turned off, and
    /// `timeouts(read_seconds=5.0)` leaves connect at its 30s default.
    #[pyo3(signature = (
        *,
        connect_seconds = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_seconds = None,
        total_seconds = None,
    ))]
    fn timeouts(
        mut slf: PyRefMut<'_, Self>,
        connect_seconds: Option<f64>,
        read_seconds: Option<f64>,
        total_seconds: Option<f64>,
    ) -> PyRefMut<'_, Self> {
        slf.connect_timeout_seconds = connect_seconds;
        slf.read_timeout_seconds = read_seconds;
        slf.total_timeout_seconds = total_seconds;
        slf.timeouts_configured = true;
        slf
    }

    /// Resolve credentials per request: `fn(ModelIden) -> AuthData | None`.
    ///
    /// Consulted before the declarative `provider(auth=...)` table, so it
    /// can override one; returning `None` means "not mine" and falls
    /// through to that table and then to the adapter's default key
    /// environment variable.
    ///
    /// Must be a regular `def`. It is called synchronously, while holding
    /// the GIL, on whichever thread is driving the request — so blocking
    /// inside it stalls that request and every other Python thread with
    /// it. Cache in Python instead.
    fn auth_resolver<'py>(
        mut slf: PyRefMut<'py, Self>,
        callback: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.auth_callback = Some(PyCallback::new(callback, "auth_resolver")?);
        Ok(slf)
    }

    /// Rewrite model identities: `fn(ModelIden) -> ModelIden | str`.
    ///
    /// Returning a bare string renames the model but keeps the adapter,
    /// which is all an alias table needs. Same threading rules as
    /// `auth_resolver`.
    fn model_mapper<'py>(
        mut slf: PyRefMut<'py, Self>,
        callback: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.model_mapper_callback = Some(PyCallback::new(callback, "model_mapper")?);
        Ok(slf)
    }

    /// Rewrite the whole target: `fn(ServiceTarget) -> ServiceTarget`.
    ///
    /// Runs after the declarative `provider(endpoint=...)` overrides have
    /// been applied and receives the result, so it has the last word on
    /// endpoint, credential, and model alike. Same threading rules as
    /// `auth_resolver`.
    fn service_target_resolver<'py>(
        mut slf: PyRefMut<'py, Self>,
        callback: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.service_target_callback = Some(PyCallback::new(callback, "service_target_resolver")?);
        Ok(slf)
    }

    /// Default embed options for every embedding call. Per-call options win.
    fn embed_options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.embed_options = coerce_embed_options(Some(options))?;
        Ok(slf)
    }

    /// Route requests through an HTTP proxy.
    ///
    /// `scheme` picks which traffic the proxy handles: `"all"` (default),
    /// `"http"`, or `"https"`. Without this, requests ignore any proxy
    /// environment the process was started with, which in a locked-down
    /// network looks like an unexplained connect timeout.
    #[pyo3(signature = (url, *, scheme = "all"))]
    fn proxy<'py>(
        mut slf: PyRefMut<'py, Self>,
        url: String,
        scheme: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        match scheme {
            "all" | "http" | "https" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid proxy scheme {other:?}. expected one of: all, http, https"
                )));
            }
        }
        slf.proxy_url = Some(url);
        slf.proxy_scheme = scheme.to_string();
        Ok(slf)
    }

    /// Enable or disable gzip response decompression (default: enabled).
    fn gzip(mut slf: PyRefMut<'_, Self>, enabled: bool) -> PyRefMut<'_, Self> {
        slf.gzip = Some(enabled);
        slf
    }

    /// Enable or disable TCP_NODELAY (default: enabled).
    fn tcp_nodelay(mut slf: PyRefMut<'_, Self>, enabled: bool) -> PyRefMut<'_, Self> {
        slf.tcp_nodelay = Some(enabled);
        slf
    }

    /// Headers sent with every request.
    fn default_headers(
        mut slf: PyRefMut<'_, Self>,
        headers: HashMap<String, String>,
    ) -> PyRefMut<'_, Self> {
        slf.default_headers = Some(headers);
        slf
    }

    fn build(&self) -> PyResult<PyClient> {
        // Seed the ClientConfig before anything else touches it.
        // `ClientBuilder::with_config` *replaces* the config wholesale, so
        // calling it further down would silently discard the adapter
        // binding, web config, and resolvers set up in between. Everything
        // after this point amends the config in place.
        let mut builder = match self.embed_options.clone() {
            Some(embed_options) => genai::Client::builder()
                .with_config(genai::ClientConfig::default().with_embed_options(embed_options)),
            None => genai::Client::builder(),
        };

        if let Some(adapter_kind) = self.adapter_kind {
            builder = builder.with_adapter_kind(adapter_kind);
        }

        let connect_timeout_seconds = if self.timeouts_configured {
            self.connect_timeout_seconds
        } else {
            Some(DEFAULT_CONNECT_TIMEOUT_SECONDS)
        };
        if let Some(web_config) = self.web_config(connect_timeout_seconds)? {
            builder = builder.with_web_config(web_config);
        }

        if let Some(chat_options) = self.chat_options.clone() {
            builder = builder.with_chat_options(chat_options);
        }

        // rust-genai holds a single resolver of each kind, so the
        // per-provider table folds into one closure that dispatches on the
        // adapter and falls through — returning the target unchanged, or
        // `None` for auth — for adapters this client was not told about.
        let auth_by_kind: HashMap<AdapterKind, AuthData> = self
            .providers
            .iter()
            .filter_map(|(kind, entry)| entry.auth.clone().map(|auth| (*kind, auth)))
            .collect();
        if !auth_by_kind.is_empty() || self.auth_callback.is_some() {
            let callback = self.auth_callback.clone().map(auth_resolver_fn);
            builder = builder.with_auth_resolver(AuthResolver::from_resolver_fn(
                move |model_iden: ModelIden| {
                    // The callback gets first refusal so it can override a
                    // configured credential; `None` means "not mine".
                    if let Some(callback) = callback.as_ref()
                        && let Some(auth) = callback(model_iden.clone())?
                    {
                        return Ok(Some(auth));
                    }
                    Ok(auth_by_kind.get(&model_iden.adapter_kind).cloned())
                },
            ));
        }

        let endpoint_by_kind: HashMap<AdapterKind, Endpoint> = self
            .providers
            .iter()
            .filter_map(|(kind, entry)| entry.endpoint.clone().map(|ep| (*kind, ep)))
            .collect();
        if !endpoint_by_kind.is_empty() || self.service_target_callback.is_some() {
            // Declarative endpoints are applied first; the callback then
            // sees the result and has the final say.
            let apply_declarative = move |mut target: genai::ServiceTarget| {
                if let Some(endpoint) = endpoint_by_kind.get(&target.model.adapter_kind) {
                    target.endpoint = endpoint.clone();
                }
                target
            };
            let resolver =
                match self.service_target_callback.clone() {
                    Some(callback) => ServiceTargetResolver::from_resolver_fn(
                        service_target_resolver_fn(callback, apply_declarative),
                    ),
                    None => ServiceTargetResolver::from_resolver_fn(
                        move |target: genai::ServiceTarget| Ok(apply_declarative(target)),
                    ),
                };
            builder = builder.with_service_target_resolver(resolver);
        }

        if let Some(callback) = self.model_mapper_callback.clone() {
            builder =
                builder.with_model_mapper(ModelMapper::from_mapper_fn(model_mapper_fn(callback)));
        }

        Ok(PyClient {
            inner: builder.build(),
            bound_adapter_kind: self.adapter_kind,
            provider_overrides: self.providers.clone(),
        })
    }

    fn __repr__(&self) -> String {
        let mut providers: Vec<&str> = self
            .providers
            .keys()
            .map(|kind| kind.as_lower_str())
            .collect();
        providers.sort_unstable();
        format!(
            "ClientBuilder(adapter_kind={:?}, providers=[{}])",
            self.adapter_kind.map(|kind| kind.as_lower_str()),
            providers.join(", ")
        )
    }
}

/// Drive `fut` to completion on the runtime shared with the async entry
/// points, with the GIL released for the duration.
///
/// The runtime comes from `pyo3_async_runtimes` — the same multi-threaded,
/// `enable_all()` runtime that backs `achat` / `astream_chat`. Building a
/// fresh runtime per call (as this used to) spun up and tore down an entire
/// worker pool for every synchronous request.
///
/// Calling a blocking entry point from inside a running runtime — a resolver
/// callback that re-enters the client, say — would panic deep inside tokio,
/// so it is rejected up front with an actionable message.
fn block_on_shared_runtime<F>(py: Python<'_>, fut: F) -> PyResult<F::Output>
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(PyRuntimeError::new_err(
            "blocking Client.chat() called from inside a running async runtime \
             (e.g. from a resolver callback). Use the async entry points \
             (achat / astream_chat) instead.",
        ));
    }
    Ok(py.detach(|| pyo3_async_runtimes::tokio::get_runtime().block_on(fut)))
}

#[allow(clippy::too_many_arguments)]
/// Keep stream-level reasoning signatures that never reached `StreamEnd`.
///
/// Some adapters emit encrypted reasoning signatures as stream chunks
/// without copying them into the terminal event. Signatures already
/// captured by `StreamEnd` win; otherwise the streamed ones are retained in
/// the response content, since losing them breaks the next turn on
/// providers that require signed thinking.
pub(crate) fn preserve_streamed_thought_signatures(
    content: &mut genai::chat::MessageContent,
    streamed_thought_signatures: Vec<String>,
) {
    if content.thought_signatures().is_empty() && !streamed_thought_signatures.is_empty() {
        content.extend_front(
            streamed_thought_signatures
                .into_iter()
                .map(genai::chat::ContentPart::ThoughtSignature),
        );
    }
}

/// Reassemble streamed signature deltas into the units a provider expects.
///
/// The Anthropic streamer exposes signature deltas without block boundaries
/// and omits terminal signatures. An Anthropic thinking response uses one
/// signed thinking block, so the deltas concatenate into a single logical
/// signature. Other adapters already emit whole units and pass through.
pub(crate) fn normalize_streamed_thought_signatures(
    adapter_kind: &AdapterKind,
    streamed_chunks: Vec<String>,
) -> Vec<String> {
    if matches!(adapter_kind, AdapterKind::Anthropic) && !streamed_chunks.is_empty() {
        vec![streamed_chunks.concat()]
    } else {
        streamed_chunks
    }
}

impl PyClientBuilder {
    /// The `WebConfig` this builder describes, or `None` when it describes
    /// nothing worth overriding.
    fn web_config(
        &self,
        connect_timeout_seconds: Option<f64>,
    ) -> PyResult<Option<genai::WebConfig>> {
        let default_headers = self.default_headers.clone();
        if connect_timeout_seconds.is_none()
            && self.read_timeout_seconds.is_none()
            && self.total_timeout_seconds.is_none()
            && self.proxy_url.is_none()
            && self.gzip.is_none()
            && self.tcp_nodelay.is_none()
            && default_headers
                .as_ref()
                .is_none_or(|headers| headers.is_empty())
        {
            return Ok(None);
        }
        let mut web_config = genai::WebConfig::default();
        if let Some(secs) = connect_timeout_seconds {
            web_config = web_config.with_connect_timeout(Duration::from_secs_f64(secs));
        }
        if let Some(secs) = self.read_timeout_seconds {
            web_config.read_timeout = Some(Duration::from_secs_f64(secs));
        }
        if let Some(secs) = self.total_timeout_seconds {
            web_config = web_config.with_timeout(Duration::from_secs_f64(secs));
        }
        if let Some(headers) = default_headers
            && !headers.is_empty()
        {
            web_config = web_config.with_default_headers(build_header_map(&headers)?);
        }
        if let Some(url) = self.proxy_url.as_deref() {
            web_config = match self.proxy_scheme.as_str() {
                "http" => web_config.with_proxy_url(url),
                "https" => web_config.with_https_proxy_url(url),
                _ => web_config.with_all_proxy_url(url),
            }
            .map_err(|err| PyValueError::new_err(format!("invalid proxy url {url:?}: {err}")))?;
        }
        if let Some(enabled) = self.gzip {
            web_config.gzip = enabled;
        }
        if let Some(enabled) = self.tcp_nodelay {
            web_config.tcp_nodelay = enabled;
        }
        Ok(Some(web_config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genai::chat::{ContentPart, MessageContent};

    #[test]
    fn streamed_thought_signatures_fill_missing_stream_end_capture() {
        let mut content = MessageContent::from_text("answer");
        preserve_streamed_thought_signatures(
            &mut content,
            vec!["sig-one".to_string(), "sig-two".to_string()],
        );

        assert_eq!(content.thought_signatures(), ["sig-one", "sig-two"]);
        assert_eq!(content.texts(), ["answer"]);
    }

    #[test]
    fn anthropic_streamed_signature_deltas_form_one_fallback_signature() {
        let signatures = normalize_streamed_thought_signatures(
            &AdapterKind::Anthropic,
            vec!["signed-".to_string(), "fragmented".to_string()],
        );

        assert_eq!(signatures, ["signed-fragmented"]);
    }

    #[test]
    fn stream_end_thought_signatures_are_not_duplicated() {
        let mut content = MessageContent::from_parts(vec![
            ContentPart::ThoughtSignature("captured".to_string()),
            ContentPart::Text("answer".to_string()),
        ]);
        preserve_streamed_thought_signatures(&mut content, vec!["streamed".to_string()]);

        assert_eq!(content.thought_signatures(), ["captured"]);
    }
}
