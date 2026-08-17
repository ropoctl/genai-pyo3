//! Typed Python exceptions for rust-genai failures.
//!
//! `genai::Error` carries structured detail — HTTP status, response body,
//! the resolved model — that a flattened `RuntimeError(str)` throws away,
//! leaving callers to grep error strings to decide whether to retry. Each
//! variant is classified here into an exception class and decorated with
//! the attributes needed to act on it.
//!
//! The hierarchy is rooted at `GenaiError`, which derives from
//! `RuntimeError`: every failure this extension has ever raised was a
//! `RuntimeError`, and quietly breaking existing `except RuntimeError:`
//! handlers would fail silently at runtime rather than loudly at import.
//!
//! ```text
//! GenaiError(RuntimeError)
//! ├── ConfigError        malformed request/options, adapter/model mismatch
//! ├── AuthError          missing or unresolvable credentials
//! ├── ResolverError      a resolver (including a Python callback) failed
//! ├── ApiError           the provider answered with a failure status
//! │   ├── BadRequestError    4xx other than 429
//! │   ├── RateLimitError     429
//! │   └── ServerError        5xx
//! ├── TransportError     connect/timeout/socket failures, stream drops
//! ├── StreamError        malformed or error events mid-stream
//! ├── ResponseError      a well-formed reply we could not interpret
//! └── UnsupportedError   feature not supported by this adapter/model
//! ```
//!
//! Every instance carries `model_name`, `adapter_kind`, `status`, `body`,
//! `retry_after`, and `is_retryable` (any of which may be `None`).
//! `is_retryable` is a conservative hint — true only for statuses and
//! transport failures that are meaningfully worth retrying — and the retry
//! and backoff policy itself stays with the caller.

use genai::webc;
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;

create_exception!(
    _genai_pyo3,
    GenaiError,
    PyRuntimeError,
    "Base class for every error raised by genai-pyo3."
);
create_exception!(
    _genai_pyo3,
    ConfigError,
    GenaiError,
    "The request, options, or client configuration is invalid."
);
create_exception!(
    _genai_pyo3,
    AuthError,
    GenaiError,
    "Credentials are missing or could not be resolved."
);
create_exception!(
    _genai_pyo3,
    ResolverError,
    GenaiError,
    "An auth / model / service-target resolver failed."
);
create_exception!(
    _genai_pyo3,
    ApiError,
    GenaiError,
    "The provider answered with a failure status."
);
create_exception!(
    _genai_pyo3,
    BadRequestError,
    ApiError,
    "The provider rejected the request (4xx other than 429)."
);
create_exception!(
    _genai_pyo3,
    RateLimitError,
    ApiError,
    "The provider rate-limited the request (429)."
);
create_exception!(
    _genai_pyo3,
    ServerError,
    ApiError,
    "The provider failed to handle the request (5xx)."
);
create_exception!(
    _genai_pyo3,
    TransportError,
    GenaiError,
    "The request never completed: connect, timeout, or socket failure."
);
create_exception!(
    _genai_pyo3,
    StreamError,
    GenaiError,
    "A streaming response carried an error or malformed event."
);
create_exception!(
    _genai_pyo3,
    ResponseError,
    GenaiError,
    "The provider replied, but the response could not be interpreted."
);
create_exception!(
    _genai_pyo3,
    UnsupportedError,
    GenaiError,
    "The adapter or model does not support the requested feature."
);

/// Register the exception classes on the extension module.
pub(crate) fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("GenaiError", py.get_type::<GenaiError>())?;
    module.add("ConfigError", py.get_type::<ConfigError>())?;
    module.add("AuthError", py.get_type::<AuthError>())?;
    module.add("ResolverError", py.get_type::<ResolverError>())?;
    module.add("ApiError", py.get_type::<ApiError>())?;
    module.add("BadRequestError", py.get_type::<BadRequestError>())?;
    module.add("RateLimitError", py.get_type::<RateLimitError>())?;
    module.add("ServerError", py.get_type::<ServerError>())?;
    module.add("TransportError", py.get_type::<TransportError>())?;
    module.add("StreamError", py.get_type::<StreamError>())?;
    module.add("ResponseError", py.get_type::<ResponseError>())?;
    module.add("UnsupportedError", py.get_type::<UnsupportedError>())?;
    Ok(())
}

/// Which exception class a `genai::Error` maps onto.
#[derive(Clone, Copy)]
enum Kind {
    Config,
    Auth,
    Resolver,
    BadRequest,
    RateLimit,
    Server,
    Api,
    Transport,
    Stream,
    Response,
    Unsupported,
    Generic,
}

impl Kind {
    fn py_type(self, py: Python<'_>) -> Bound<'_, PyType> {
        match self {
            Kind::Config => py.get_type::<ConfigError>(),
            Kind::Auth => py.get_type::<AuthError>(),
            Kind::Resolver => py.get_type::<ResolverError>(),
            Kind::BadRequest => py.get_type::<BadRequestError>(),
            Kind::RateLimit => py.get_type::<RateLimitError>(),
            Kind::Server => py.get_type::<ServerError>(),
            Kind::Api => py.get_type::<ApiError>(),
            Kind::Transport => py.get_type::<TransportError>(),
            Kind::Stream => py.get_type::<StreamError>(),
            Kind::Response => py.get_type::<ResponseError>(),
            Kind::Unsupported => py.get_type::<UnsupportedError>(),
            Kind::Generic => py.get_type::<GenaiError>(),
        }
    }
}

/// The structured detail attached to every raised exception.
#[derive(Default)]
struct Detail {
    model_name: Option<String>,
    adapter_kind: Option<String>,
    status: Option<u16>,
    body: Option<String>,
    retry_after: Option<f64>,
    retryable: bool,
}

impl Detail {
    fn with_model(mut self, model_iden: &genai::ModelIden) -> Self {
        self.model_name = Some(model_iden.model_name.to_string());
        self.adapter_kind = Some(model_iden.adapter_kind.to_string());
        self
    }

    fn with_adapter(mut self, adapter_kind: genai::adapter::AdapterKind) -> Self {
        self.adapter_kind = Some(adapter_kind.to_string());
        self
    }
}

/// Statuses worth another attempt: rate limits, request timeouts, and
/// server-side failures. Everything else is the caller's problem to fix.
fn status_is_retryable(status: u16) -> bool {
    status == 429 || status == 408 || (500..600).contains(&status)
}

fn status_kind(status: u16) -> Kind {
    match status {
        429 => Kind::RateLimit,
        500..=599 => Kind::Server,
        400..=499 => Kind::BadRequest,
        _ => Kind::Api,
    }
}

/// `Retry-After` in delta-seconds form. The HTTP-date form is legal but
/// vanishingly rare from inference providers; it is left as `None` rather
/// than half-parsed.
fn retry_after_seconds(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
}

fn classify_webc(err: &webc::Error) -> (Kind, Detail) {
    match err {
        webc::Error::ResponseFailedStatus {
            status,
            body,
            headers,
        } => {
            let code = status.as_u16();
            (
                status_kind(code),
                Detail {
                    status: Some(code),
                    body: Some(body.clone()),
                    retry_after: retry_after_seconds(headers),
                    retryable: status_is_retryable(code),
                    ..Default::default()
                },
            )
        }
        webc::Error::ResponseFailedNotJson { body, .. }
        | webc::Error::ResponseFailedInvalidJson { body, .. } => (
            Kind::Response,
            Detail {
                body: Some(body.clone()),
                ..Default::default()
            },
        ),
        webc::Error::Reqwest(reqwest_err) => (
            Kind::Transport,
            Detail {
                status: reqwest_err.status().map(|s| s.as_u16()),
                // A request that never reached the provider is exactly the
                // case worth retrying: DNS, connect, TLS, or read timeout.
                retryable: true,
                ..Default::default()
            },
        ),
        webc::Error::JsonValueExt(_) => (Kind::Response, Detail::default()),
    }
}

fn classify(error: &genai::Error) -> (Kind, Detail) {
    use genai::Error as E;

    match error {
        // -- Request/config problems the caller can fix
        E::ChatReqHasNoMessages { model_iden }
        | E::LastChatMessageIsNotUser { model_iden, .. }
        | E::CacheBreakpointNoEligibleContent { model_iden, .. } => {
            (Kind::Config, Detail::default().with_model(model_iden))
        }
        E::JsonModeWithoutInstruction
        | E::VerbosityParsing { .. }
        | E::ReasoningParsingError { .. }
        | E::ServiceTierParsing { .. }
        | E::PromptCacheRetentionParsing { .. }
        | E::AdapterKindMismatch { .. } => (Kind::Config, Detail::default()),

        // -- Unsupported feature for this adapter/model
        E::MessageRoleNotSupported { model_iden, .. }
        | E::MessageContentTypeNotSupported { model_iden, .. } => {
            (Kind::Unsupported, Detail::default().with_model(model_iden))
        }
        E::AdapterNotSupported { adapter_kind, .. } => (
            Kind::Unsupported,
            Detail::default().with_adapter(*adapter_kind),
        ),

        // -- Auth
        E::RequiresApiKey { model_iden }
        | E::NoAuthResolver { model_iden }
        | E::NoAuthData { model_iden } => (Kind::Auth, Detail::default().with_model(model_iden)),

        // -- Resolvers. A missing credential env var is really an auth
        //    problem wearing a resolver's coat, so it is reported as one.
        E::Resolver {
            model_iden,
            resolver_error,
        } => {
            let kind = match resolver_error {
                genai::resolver::Error::ApiKeyEnvNotFound { .. } => Kind::Auth,
                _ => Kind::Resolver,
            };
            (kind, Detail::default().with_model(model_iden))
        }
        E::ModelMapperFailed { model_iden, .. } => {
            (Kind::Resolver, Detail::default().with_model(model_iden))
        }

        // -- Web calls: the status lives inside the webc error
        E::WebModelCall {
            model_iden,
            webc_error,
        } => {
            let (kind, detail) = classify_webc(webc_error);
            (kind, detail.with_model(model_iden))
        }
        E::WebAdapterCall {
            adapter_kind,
            webc_error,
        } => {
            let (kind, detail) = classify_webc(webc_error);
            (kind, detail.with_adapter(*adapter_kind))
        }
        E::HttpError { status, body, .. } => {
            let code = status.as_u16();
            (
                status_kind(code),
                Detail {
                    status: Some(code),
                    body: Some(body.clone()),
                    retryable: status_is_retryable(code),
                    ..Default::default()
                },
            )
        }

        // -- Streaming
        E::StreamParse { model_iden, .. } => {
            (Kind::Stream, Detail::default().with_model(model_iden))
        }
        E::ChatResponse { model_iden, body } => (
            Kind::Stream,
            Detail {
                body: Some(body.to_string()),
                ..Default::default()
            }
            .with_model(model_iden),
        ),
        E::WebStream { model_iden, .. } => (
            Kind::Transport,
            Detail {
                // A stream that died mid-flight is a transport failure; the
                // request itself was accepted, so re-issuing it is sound.
                retryable: true,
                ..Default::default()
            }
            .with_model(model_iden),
        ),

        // -- Well-formed reply we could not turn into a ChatResponse
        E::NoChatResponse { model_iden } => {
            (Kind::Response, Detail::default().with_model(model_iden))
        }
        E::ChatResponseGeneration {
            model_iden,
            response_body,
            ..
        } => (
            Kind::Response,
            Detail {
                body: Some(response_body.to_string()),
                ..Default::default()
            }
            .with_model(model_iden),
        ),
        E::InvalidJsonResponseElement { .. } => (Kind::Response, Detail::default()),

        _ => (Kind::Generic, Detail::default()),
    }
}

/// Convert a `genai::Error` into the matching Python exception.
///
/// Safe to call without the GIL held (the async paths do): the token is
/// acquired here, on the error path only.
pub(crate) fn to_py_err(error: genai::Error) -> PyErr {
    let (kind, detail) = classify(&error);
    let message = error.to_string();

    Python::attach(|py| {
        let exc_type = kind.py_type(py);
        let instance = match exc_type.call1((message,)) {
            Ok(instance) => instance,
            // Constructing the exception failed — surface that failure
            // rather than masking it with a fabricated stand-in.
            Err(err) => return err,
        };
        let _ = instance.setattr("model_name", detail.model_name);
        let _ = instance.setattr("adapter_kind", detail.adapter_kind);
        let _ = instance.setattr("status", detail.status);
        let _ = instance.setattr("body", detail.body);
        let _ = instance.setattr("retry_after", detail.retry_after);
        let _ = instance.setattr("is_retryable", detail.retryable);
        PyErr::from_value(instance)
    })
}
