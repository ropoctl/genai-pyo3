//! Typed Python exceptions for rust-genai failures.
//!
//! Two layers, deliberately kept separate:
//!
//! - **Attributes** come from [`ErrorMetadata`], which walks the
//!   `genai::Error` (and the `webc`/`reqwest` errors nested inside it) to
//!   recover `kind`, `status`, `retryable`, `body`, and response `headers`.
//!   This is the surface shipped in 0.7.0.1 and is preserved as-is.
//! - **The exception class** is chosen from the `genai::Error` variant
//!   itself rather than from the `kind` string, so a provider inventing a
//!   new `kind` (they do — `kind` can be a provider's own error type from a
//!   stream error body) still lands in the right category.
//!
//! ```text
//! GenaiError(RuntimeError)
//! ├── ConfigError        malformed request/options, adapter mismatch
//! ├── AuthError          missing or unresolvable credentials
//! ├── ResolverError      a resolver (including a Python callback) failed
//! ├── ApiError           the provider answered with a failure status
//! │   ├── BadRequestError    4xx other than 429
//! │   ├── RateLimitError     429
//! │   └── ServerError        5xx
//! ├── TransportError     connect/timeout/socket failures, dropped streams
//! ├── StreamError        malformed or error events mid-stream
//! ├── ResponseError      a well-formed reply we could not interpret
//! └── UnsupportedError   feature unavailable on this adapter/model
//! ```
//!
//! `GenaiError` derives from `RuntimeError` because that is what every
//! failure this extension has ever raised was; existing
//! `except RuntimeError:` handlers keep working.

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use reqwest::header::HeaderMap;
use std::collections::HashMap;

use crate::target::PyModelIden;

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

// region: --- Metadata (attributes)

#[derive(Debug, PartialEq)]
pub(crate) struct ErrorMetadata {
    pub(crate) kind: String,
    pub(crate) status: Option<u16>,
    pub(crate) retryable: bool,
    pub(crate) body: Option<String>,
    pub(crate) headers: HashMap<String, String>,
}

impl ErrorMetadata {
    fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            status: None,
            retryable: false,
            body: None,
            headers: HashMap::new(),
        }
    }
}

pub(crate) fn retryable_status(status: u16) -> bool {
    status == 429 || status >= 500
}

fn headers_to_hash_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect()
}

fn reqwest_error_metadata(error: &reqwest::Error) -> ErrorMetadata {
    let status = error.status().map(|status| status.as_u16());
    let kind = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else if error.is_decode() {
        "response_decode"
    } else if error.is_body() {
        "response_body"
    } else if status.is_some() {
        "http_error"
    } else {
        "transport"
    };
    ErrorMetadata {
        kind: kind.to_string(),
        status,
        retryable: status.is_none_or(retryable_status),
        body: None,
        headers: HashMap::new(),
    }
}

fn webc_error_metadata(error: &genai::webc::Error) -> ErrorMetadata {
    match error {
        genai::webc::Error::ResponseFailedStatus {
            status,
            body,
            headers,
        } => ErrorMetadata {
            kind: "http_error".to_string(),
            status: Some(status.as_u16()),
            retryable: retryable_status(status.as_u16()),
            body: Some(body.clone()),
            headers: headers_to_hash_map(headers),
        },
        genai::webc::Error::Reqwest(error) => reqwest_error_metadata(error),
        genai::webc::Error::ResponseFailedNotJson { body, .. } => {
            let mut metadata = ErrorMetadata::new("response_not_json");
            metadata.body = Some(body.clone());
            metadata
        }
        genai::webc::Error::ResponseFailedInvalidJson { body, .. } => {
            let mut metadata = ErrorMetadata::new("response_invalid_json");
            metadata.body = Some(body.clone());
            metadata
        }
        genai::webc::Error::JsonValueExt(_) => ErrorMetadata::new("json_value_ext"),
    }
}

/// A provider's own error payload, arriving as an event mid-stream. `kind`
/// is the provider's error type verbatim when it gives one — which is why
/// the exception class is picked from the Rust variant instead.
fn provider_stream_error_metadata(body: &serde_json::Value) -> ErrorMetadata {
    let provider_kind = body
        .pointer("/error/type")
        .and_then(serde_json::Value::as_str)
        .or_else(|| body.get("type").and_then(serde_json::Value::as_str));
    let retryable = matches!(
        provider_kind,
        Some("overloaded_error" | "rate_limit_error" | "internal_server_error")
    );
    let mut metadata = ErrorMetadata::new(provider_kind.unwrap_or("provider_stream_error"));
    metadata.retryable = retryable;
    metadata.body = Some(body.to_string());
    metadata
}

pub(crate) fn error_metadata(error: &genai::Error) -> ErrorMetadata {
    match error {
        genai::Error::WebAdapterCall { webc_error, .. }
        | genai::Error::WebModelCall { webc_error, .. } => webc_error_metadata(webc_error),
        genai::Error::HttpError {
            status,
            body,
            headers,
            ..
        } => ErrorMetadata {
            kind: "http_error".to_string(),
            status: Some(status.as_u16()),
            retryable: retryable_status(status.as_u16()),
            body: Some(body.clone()),
            headers: headers_to_hash_map(headers),
        },
        genai::Error::WebStream { error, .. } => {
            if let Some(error) = error.downcast_ref::<genai::Error>() {
                error_metadata(error)
            } else if let Some(error) = error.downcast_ref::<reqwest::Error>() {
                reqwest_error_metadata(error)
            } else {
                ErrorMetadata {
                    retryable: true,
                    ..ErrorMetadata::new("web_stream")
                }
            }
        }
        genai::Error::StreamParse { .. } => ErrorMetadata {
            retryable: true,
            ..ErrorMetadata::new("stream_parse")
        },
        genai::Error::ChatResponse { body, .. } => provider_stream_error_metadata(body),
        genai::Error::ChatReqHasNoMessages { .. } => ErrorMetadata::new("chat_req_has_no_messages"),
        genai::Error::LastChatMessageIsNotUser { .. } => {
            ErrorMetadata::new("last_chat_message_is_not_user")
        }
        genai::Error::MessageRoleNotSupported { .. } => {
            ErrorMetadata::new("message_role_not_supported")
        }
        genai::Error::MessageContentTypeNotSupported { .. } => {
            ErrorMetadata::new("message_content_type_not_supported")
        }
        genai::Error::JsonModeWithoutInstruction => {
            ErrorMetadata::new("json_mode_without_instruction")
        }
        genai::Error::VerbosityParsing { .. } => ErrorMetadata::new("verbosity_parsing"),
        genai::Error::ReasoningParsingError { .. } => ErrorMetadata::new("reasoning_parsing"),
        genai::Error::ServiceTierParsing { .. } => ErrorMetadata::new("service_tier_parsing"),
        genai::Error::PromptCacheRetentionParsing { .. } => {
            ErrorMetadata::new("prompt_cache_retention_parsing")
        }
        genai::Error::NoChatResponse { .. } => ErrorMetadata::new("no_chat_response"),
        genai::Error::InvalidJsonResponseElement { .. } => {
            ErrorMetadata::new("invalid_json_response_element")
        }
        genai::Error::RequiresApiKey { .. } => ErrorMetadata::new("requires_api_key"),
        genai::Error::NoAuthResolver { .. } => ErrorMetadata::new("no_auth_resolver"),
        genai::Error::NoAuthData { .. } => ErrorMetadata::new("no_auth_data"),
        genai::Error::ModelMapperFailed { .. } => ErrorMetadata::new("model_mapper_failed"),
        genai::Error::ChatResponseGeneration { .. } => {
            ErrorMetadata::new("chat_response_generation")
        }
        genai::Error::Resolver { .. } => ErrorMetadata::new("resolver"),
        genai::Error::AdapterNotSupported { .. } => ErrorMetadata::new("adapter_not_supported"),
        genai::Error::CacheBreakpointNoEligibleContent { .. } => {
            ErrorMetadata::new("cache_breakpoint_no_eligible_content")
        }
        genai::Error::AdapterKindMismatch { .. } => ErrorMetadata::new("adapter_kind_mismatch"),
        genai::Error::Internal(_) => ErrorMetadata::new("internal"),
        genai::Error::JsonValueExt(_) => ErrorMetadata::new("json_value_ext"),
        genai::Error::SerdeJson(_) => ErrorMetadata::new("serde_json"),
        #[allow(unreachable_patterns)]
        _ => ErrorMetadata::new("genai_error"),
    }
}

// endregion: --- Metadata

// region: --- Classification (exception class)

/// Which exception class a `genai::Error` maps onto.
#[derive(Clone, Copy)]
enum Kind {
    Config,
    Auth,
    Resolver,
    Api,
    Transport,
    Stream,
    Response,
    Unsupported,
    Generic,
}

impl Kind {
    /// Resolve to a concrete class, refining `Api` by HTTP status.
    fn py_type(self, py: Python<'_>, status: Option<u16>) -> Bound<'_, PyType> {
        match self {
            Kind::Config => py.get_type::<ConfigError>(),
            Kind::Auth => py.get_type::<AuthError>(),
            Kind::Resolver => py.get_type::<ResolverError>(),
            Kind::Api => match status {
                Some(429) => py.get_type::<RateLimitError>(),
                Some(500..=599) => py.get_type::<ServerError>(),
                Some(400..=499) => py.get_type::<BadRequestError>(),
                _ => py.get_type::<ApiError>(),
            },
            Kind::Transport => py.get_type::<TransportError>(),
            Kind::Stream => py.get_type::<StreamError>(),
            Kind::Response => py.get_type::<ResponseError>(),
            Kind::Unsupported => py.get_type::<UnsupportedError>(),
            Kind::Generic => py.get_type::<GenaiError>(),
        }
    }
}

fn classify(error: &genai::Error) -> Kind {
    use genai::Error as E;

    match error {
        // -- Request/config problems the caller can fix
        E::ChatReqHasNoMessages { .. }
        | E::LastChatMessageIsNotUser { .. }
        | E::CacheBreakpointNoEligibleContent { .. }
        | E::JsonModeWithoutInstruction
        | E::VerbosityParsing { .. }
        | E::ReasoningParsingError { .. }
        | E::ServiceTierParsing { .. }
        | E::PromptCacheRetentionParsing { .. }
        | E::AdapterKindMismatch { .. } => Kind::Config,

        // -- Unsupported feature for this adapter/model
        E::MessageRoleNotSupported { .. }
        | E::MessageContentTypeNotSupported { .. }
        | E::AdapterNotSupported { .. } => Kind::Unsupported,

        // -- Auth. A missing credential env var is an auth problem wearing
        //    a resolver's coat, so it is reported as one.
        E::RequiresApiKey { .. } | E::NoAuthResolver { .. } | E::NoAuthData { .. } => Kind::Auth,
        E::Resolver {
            resolver_error: genai::resolver::Error::ApiKeyEnvNotFound { .. },
            ..
        } => Kind::Auth,
        E::Resolver { .. } | E::ModelMapperFailed { .. } => Kind::Resolver,

        // -- Web calls. Whether these are an API failure or a transport
        //    failure depends on how far the request got.
        E::WebModelCall { webc_error, .. } | E::WebAdapterCall { webc_error, .. } => {
            classify_webc(webc_error)
        }
        E::HttpError { .. } => Kind::Api,

        // -- Streaming
        E::StreamParse { .. } | E::ChatResponse { .. } => Kind::Stream,
        E::WebStream { error, .. } => match error.downcast_ref::<genai::Error>() {
            Some(inner) => classify(inner),
            None => Kind::Transport,
        },

        // -- Well-formed reply we could not turn into a ChatResponse
        E::NoChatResponse { .. }
        | E::ChatResponseGeneration { .. }
        | E::InvalidJsonResponseElement { .. } => Kind::Response,

        _ => Kind::Generic,
    }
}

fn classify_webc(error: &genai::webc::Error) -> Kind {
    match error {
        genai::webc::Error::ResponseFailedStatus { .. } => Kind::Api,
        genai::webc::Error::Reqwest(reqwest_error) => match reqwest_error.status() {
            // A response with a status came back; that is an API failure.
            Some(_) => Kind::Api,
            // Nothing came back at all: DNS, connect, TLS, read timeout.
            None => Kind::Transport,
        },
        genai::webc::Error::ResponseFailedNotJson { .. }
        | genai::webc::Error::ResponseFailedInvalidJson { .. }
        | genai::webc::Error::JsonValueExt(_) => Kind::Response,
    }
}

/// The model the failure is about, when the error names one.
fn model_iden_of(error: &genai::Error) -> Option<genai::ModelIden> {
    use genai::Error as E;

    match error {
        E::ChatReqHasNoMessages { model_iden }
        | E::LastChatMessageIsNotUser { model_iden, .. }
        | E::MessageRoleNotSupported { model_iden, .. }
        | E::MessageContentTypeNotSupported { model_iden, .. }
        | E::NoChatResponse { model_iden }
        | E::RequiresApiKey { model_iden }
        | E::NoAuthResolver { model_iden }
        | E::NoAuthData { model_iden }
        | E::ModelMapperFailed { model_iden, .. }
        | E::WebModelCall { model_iden, .. }
        | E::ChatResponseGeneration { model_iden, .. }
        | E::ChatResponse { model_iden, .. }
        | E::StreamParse { model_iden, .. }
        | E::WebStream { model_iden, .. }
        | E::Resolver { model_iden, .. }
        | E::CacheBreakpointNoEligibleContent { model_iden, .. } => Some(model_iden.clone()),
        _ => None,
    }
}

/// `Retry-After` in delta-seconds. The HTTP-date form is legal but
/// vanishingly rare from inference providers; it is left as `None` rather
/// than half-parsed.
fn retry_after_seconds(headers: &HashMap<String, String>) -> Option<f64> {
    headers.get("retry-after")?.trim().parse::<f64>().ok()
}

// endregion: --- Classification

/// Convert a `genai::Error` into the matching Python exception.
///
/// Safe to call without the GIL held (the async paths do): the token is
/// acquired here, on the error path only.
pub(crate) fn to_py_err(error: genai::Error) -> PyErr {
    let message = error.to_string();
    let metadata = error_metadata(&error);
    let kind = classify(&error);
    let model_iden = model_iden_of(&error);
    let retry_after = retry_after_seconds(&metadata.headers);

    Python::attach(|py| {
        let exc_type = kind.py_type(py, metadata.status);
        let instance = match exc_type.call1((message,)) {
            Ok(instance) => instance,
            // Constructing the exception failed — surface that rather than
            // masking it with a fabricated stand-in.
            Err(err) => return err,
        };
        for result in [
            instance.setattr("kind", metadata.kind),
            instance.setattr("status", metadata.status),
            instance.setattr("status_code", metadata.status),
            instance.setattr("retryable", metadata.retryable),
            instance.setattr("body", metadata.body),
            instance.setattr("headers", metadata.headers),
            instance.setattr("retry_after", retry_after),
            instance.setattr("model_iden", model_iden.map(|inner| PyModelIden { inner })),
        ] {
            if let Err(attribute_error) = result {
                return attribute_error;
            }
        }
        PyErr::from_value(instance)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use genai::Error;
    use reqwest::StatusCode;
    use serde_json::json;

    #[test]
    fn http_error_metadata_preserves_status_and_retryability() {
        let too_many_requests = Error::HttpError {
            status: StatusCode::TOO_MANY_REQUESTS,
            canonical_reason: "Too Many Requests".to_string(),
            body: "slow down".to_string(),
            headers: Box::new(HeaderMap::new()),
        };
        let metadata = error_metadata(&too_many_requests);
        assert_eq!(metadata.kind, "http_error");
        assert_eq!(metadata.status, Some(429));
        assert_eq!(metadata.body.as_deref(), Some("slow down"));
        assert!(metadata.retryable);

        assert!(retryable_status(500));
        assert!(!retryable_status(400));
    }

    #[test]
    fn provider_stream_error_only_retries_transient_types() {
        let transient = Error::ChatResponse {
            model_iden: genai::ModelIden::from_static(
                genai::adapter::AdapterKind::Anthropic,
                "test-model",
            ),
            body: json!({"type": "error", "error": {"type": "overloaded_error"}}),
        };
        let metadata = error_metadata(&transient);
        assert_eq!(metadata.kind, "overloaded_error");
        assert!(metadata.retryable);

        let permanent = Error::ChatResponse {
            model_iden: genai::ModelIden::from_static(
                genai::adapter::AdapterKind::Anthropic,
                "test-model",
            ),
            body: json!({"type": "error", "error": {"type": "invalid_request_error"}}),
        };
        let metadata = error_metadata(&permanent);
        assert_eq!(metadata.kind, "invalid_request_error");
        assert!(!metadata.retryable);
    }
}
