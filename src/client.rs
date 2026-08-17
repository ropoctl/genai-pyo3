//! The `Client` pyclass and its construction helpers.

use futures::StreamExt;
use genai::Headers;
use genai::adapter::AdapterKind;
use genai::chat::{ChatStreamEvent, StreamEnd};
use genai::resolver::{AuthData, AuthResolver, Endpoint, ServiceTargetResolver};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

use crate::chat::{coerce_chat_options, coerce_chat_request};
use crate::convert::build_header_map;
use crate::errors::to_py_err;
use crate::response::{PyChatResponse, PyChatStream, to_py_chat_response, to_py_usage};

/// Default connect-timeout applied by `PyClient::with_*` builders when the
/// caller doesn't override it. A 30-second cap is long enough to cover any
/// realistic TCP+TLS handshake and short enough to surface a hung proxy or
/// an unreachable endpoint before it eats through an entire run. Pass
/// ``connect_timeout_seconds=None`` explicitly to disable it.
const DEFAULT_CONNECT_TIMEOUT_SECONDS: f64 = 30.0;

/// Python-facing wrapper around `genai::Client`.
///
/// A `Client` built via `with_api_key` / `with_base_url` /
/// `with_api_key_and_base_url` is bound to a single provider at
/// construction time — rust-genai's `ClientBuilder::with_adapter_kind`
/// drives routing directly from that binding, so callers hand bare
/// model names to `chat`, `achat`, and friends. The zero-arg
/// `Client()` path leaves routing unbound and falls back to
/// rust-genai's model-name prefix inference.
#[pyclass(name = "Client")]
pub(crate) struct PyClient {
    inner: genai::Client,
    /// Canonical provider string the Client was built with
    /// (`"openai"`, `"openai_resp"`, `"anthropic"`, …), or `None`
    /// for a zero-arg unbound `Client()`. Exposed as a read-only
    /// `.provider` attribute so callers can introspect a Client
    /// without keeping parallel state alongside it.
    #[pyo3(get)]
    provider: Option<String>,
}

#[pymethods]
impl PyClient {
    #[new]
    fn new() -> Self {
        Self {
            inner: genai::Client::default(),
            provider: None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.provider {
            Some(provider) => format!("Client(provider={provider:?})"),
            None => "Client(unbound)".to_string(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (
        provider,
        api_key,
        *,
        connect_timeout_seconds = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds = None,
        timeout_seconds = None,
        default_headers = None,
    ))]
    fn with_api_key(
        provider: String,
        api_key: String,
        connect_timeout_seconds: Option<f64>,
        read_timeout_seconds: Option<f64>,
        timeout_seconds: Option<f64>,
        default_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        build_client_with_overrides(
            provider,
            Some(api_key),
            None,
            connect_timeout_seconds,
            read_timeout_seconds,
            timeout_seconds,
            default_headers,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (
        provider,
        api_key,
        base_url,
        *,
        connect_timeout_seconds = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds = None,
        timeout_seconds = None,
        default_headers = None,
    ))]
    fn with_api_key_and_base_url(
        provider: String,
        api_key: String,
        base_url: String,
        connect_timeout_seconds: Option<f64>,
        read_timeout_seconds: Option<f64>,
        timeout_seconds: Option<f64>,
        default_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        build_client_with_overrides(
            provider,
            Some(api_key),
            Some(base_url),
            connect_timeout_seconds,
            read_timeout_seconds,
            timeout_seconds,
            default_headers,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (
        provider,
        base_url,
        *,
        connect_timeout_seconds = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds = None,
        timeout_seconds = None,
        default_headers = None,
    ))]
    fn with_base_url(
        provider: String,
        base_url: String,
        connect_timeout_seconds: Option<f64>,
        read_timeout_seconds: Option<f64>,
        timeout_seconds: Option<f64>,
        default_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        build_client_with_overrides(
            provider,
            None,
            Some(base_url),
            connect_timeout_seconds,
            read_timeout_seconds,
            timeout_seconds,
            default_headers,
        )
    }

    #[staticmethod]
    fn with_request_override(
        provider: String,
        url: String,
        headers: HashMap<String, String>,
    ) -> PyResult<Self> {
        build_client_with_request_override(provider, url, headers)
    }

    #[pyo3(signature = (model, request, options = None))]
    fn chat<'py>(
        &self,
        py: Python<'py>,
        model: String,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyChatResponse>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options, self.provider.as_deref())?;

        let chat_res = block_on_shared_runtime(
            py,
            self.inner.exec_chat(model, chat_req, chat_options.as_ref()),
        )?
        .map_err(to_py_err)?;

        to_py_chat_response(py, chat_res)
    }

    #[pyo3(signature = (model, request, options = None))]
    fn achat<'py>(
        &self,
        py: Python<'py>,
        model: String,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options, self.provider.as_deref())?;
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chat_res = client
                .exec_chat(model, chat_req, chat_options.as_ref())
                .await
                .map_err(to_py_err)?;

            Python::attach(|py| to_py_chat_response(py, chat_res))
        })
    }

    #[pyo3(signature = (model, request, options = None))]
    fn astream_chat<'py>(
        &self,
        py: Python<'py>,
        model: String,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options, self.provider.as_deref())?;
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream_res = client
                .exec_chat_stream(model, chat_req, chat_options.as_ref())
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
        model: String,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let mut chat_options =
            coerce_chat_options(options, self.provider.as_deref())?.unwrap_or_default();
        chat_options.capture_content = Some(true);
        chat_options.capture_tool_calls = Some(true);
        chat_options.capture_usage = Some(true);
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream_res = client
                .exec_chat_stream(model, chat_req, Some(&chat_options))
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
                        response_id,
                    },
                )
            })
        })
    }
}

/// Drive `fut` to completion on the runtime shared with the async entry
/// points, with the GIL released for the duration.
///
/// The runtime comes from `pyo3_async_runtimes` — the same multi-threaded,
/// `enable_all()` runtime that backs `achat` / `astream_chat`. Building a
/// fresh runtime per call (as this used to) span up and tore down an entire
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

fn parse_adapter_kind(provider: &str) -> PyResult<AdapterKind> {
    let normalized = provider.trim().to_ascii_lowercase();
    AdapterKind::from_lower_str(&normalized).ok_or_else(|| {
		PyValueError::new_err(format!(
			"invalid provider '{provider}'. expected one of: openai, openai_resp, gemini, anthropic, fireworks, together, groq, mimo, nebius, xai, deepseek, zai, bigmodel, aliyun, cohere, ollama"
		))
	})
}

fn build_client_with_overrides(
    provider: String,
    api_key: Option<String>,
    base_url: Option<String>,
    connect_timeout_seconds: Option<f64>,
    read_timeout_seconds: Option<f64>,
    timeout_seconds: Option<f64>,
    default_headers: Option<HashMap<String, String>>,
) -> PyResult<PyClient> {
    let adapter_kind = parse_adapter_kind(&provider)?;
    // Bind the Client to the adapter up front. rust-genai's
    // `with_adapter_kind` drives routing directly from this, so every
    // `exec_chat` call hands a bare model name through without any
    // `"{provider}::{model}"` trickery on the Python side. Mismatched
    // namespaces in the caller's model string surface as
    // `Error::AdapterKindMismatch` from the Rust side.
    let mut builder = genai::Client::builder().with_adapter_kind(adapter_kind);

    if let Some(web_config) = web_config_from_overrides(
        connect_timeout_seconds,
        read_timeout_seconds,
        timeout_seconds,
        default_headers,
    )? {
        builder = builder.with_web_config(web_config);
    }

    if let Some(api_key) = api_key {
        let auth_resolver = AuthResolver::from_resolver_fn(move |model_iden: genai::ModelIden| {
            if model_iden.adapter_kind == adapter_kind {
                Ok(Some(AuthData::from_single(api_key.clone())))
            } else {
                Ok(None)
            }
        });
        builder = builder.with_auth_resolver(auth_resolver);
    }

    if let Some(base_url) = base_url {
        let service_target_resolver = ServiceTargetResolver::from_resolver_fn(
            move |mut service_target: genai::ServiceTarget| {
                if service_target.model.adapter_kind == adapter_kind {
                    service_target.endpoint = Endpoint::from_owned(base_url.clone());
                }
                Ok(service_target)
            },
        );
        builder = builder.with_service_target_resolver(service_target_resolver);
    }

    Ok(PyClient {
        inner: builder.build(),
        provider: Some(provider),
    })
}

fn build_client_with_request_override(
    provider: String,
    url: String,
    headers: HashMap<String, String>,
) -> PyResult<PyClient> {
    let adapter_kind = parse_adapter_kind(&provider)?;
    let url = url.trim().to_string();
    if url.is_empty() {
        return Err(PyValueError::new_err(
            "request override url cannot be empty",
        ));
    }

    let headers = Headers::from(headers.into_iter().collect::<Vec<_>>());
    let auth_resolver = AuthResolver::from_resolver_fn(move |model_iden: genai::ModelIden| {
        if model_iden.adapter_kind == adapter_kind {
            Ok(Some(AuthData::RequestOverride {
                url: url.clone(),
                headers: headers.clone(),
            }))
        } else {
            Ok(None)
        }
    });

    Ok(PyClient {
        inner: genai::Client::builder()
            .with_adapter_kind(adapter_kind)
            .with_auth_resolver(auth_resolver)
            .build(),
        provider: Some(provider),
    })
}
