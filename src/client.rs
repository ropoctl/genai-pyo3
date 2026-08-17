//! The `Client` pyclass and its builder.

use futures::StreamExt;
use genai::ModelIden;
use genai::adapter::AdapterKind;
use genai::chat::{ChatOptions, ChatStreamEvent, StreamEnd};
use genai::resolver::{AuthData, AuthResolver, Endpoint, ModelMapper, ServiceTargetResolver};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

use crate::chat::{coerce_chat_options, coerce_chat_request};
use crate::convert::build_header_map;
use crate::errors::to_py_err;
use crate::resolvers::{PyCallback, auth_resolver_fn, model_mapper_fn, service_target_resolver_fn};
use crate::response::{
    PyChatResponse, PyChatStream, stop_reason_parts, to_py_chat_response, to_py_usage,
};
use crate::target::{AdapterKindLike, EndpointLike, ModelLike, PyAdapterKind, PyAuthData};

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
            while let Some(event) = stream.next().await {
                let event = event.map_err(to_py_err)?;
                if let ChatStreamEvent::End(e) = event {
                    end = Some(e);
                    break;
                }
            }

            let end = end.ok_or_else(|| {
                PyRuntimeError::new_err("stream ended without a terminal End event")
            })?;
            let content = end
                .captured_content
                .unwrap_or_else(genai::chat::MessageContent::default);
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

    /// Headers sent with every request.
    fn default_headers(
        mut slf: PyRefMut<'_, Self>,
        headers: HashMap<String, String>,
    ) -> PyRefMut<'_, Self> {
        slf.default_headers = Some(headers);
        slf
    }

    fn build(&self) -> PyResult<PyClient> {
        let mut builder = genai::Client::builder();

        if let Some(adapter_kind) = self.adapter_kind {
            builder = builder.with_adapter_kind(adapter_kind);
        }

        let connect_timeout_seconds = if self.timeouts_configured {
            self.connect_timeout_seconds
        } else {
            Some(DEFAULT_CONNECT_TIMEOUT_SECONDS)
        };
        if let Some(web_config) = web_config_from_overrides(
            connect_timeout_seconds,
            self.read_timeout_seconds,
            self.total_timeout_seconds,
            self.default_headers.clone(),
        )? {
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

fn web_config_from_overrides(
    connect_timeout_seconds: Option<f64>,
    read_timeout_seconds: Option<f64>,
    timeout_seconds: Option<f64>,
    default_headers: Option<HashMap<String, String>>,
) -> PyResult<Option<genai::WebConfig>> {
    if connect_timeout_seconds.is_none()
        && read_timeout_seconds.is_none()
        && timeout_seconds.is_none()
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
    if let Some(secs) = read_timeout_seconds {
        web_config.read_timeout = Some(Duration::from_secs_f64(secs));
    }
    if let Some(secs) = timeout_seconds {
        web_config = web_config.with_timeout(Duration::from_secs_f64(secs));
    }
    if let Some(headers) = default_headers
        && !headers.is_empty()
    {
        web_config = web_config.with_default_headers(build_header_map(&headers)?);
    }
    Ok(Some(web_config))
}
