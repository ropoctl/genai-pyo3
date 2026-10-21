//! Coercion helpers shared by the request, response, and client modules.
//!
//! Everything here is a pure translation between Python values, `serde_json`
//! values, and rust-genai types — no pyclasses, no I/O.

use genai::chat::{
    CacheControl, ChatRole, ContentPart, CustomPart, JsonSchemaDialect, ReasoningEffort,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::depythonize;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::collections::HashMap;

pub(crate) fn parse_chat_role(role: &str) -> PyResult<ChatRole> {
    match role {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        "tool" => Ok(ChatRole::Tool),
        _ => Err(PyValueError::new_err(format!(
            "invalid chat role '{role}'. expected one of: system, user, assistant, tool"
        ))),
    }
}

pub(crate) fn validate_role(role: &str) -> PyResult<()> {
    parse_chat_role(role).map(|_| ())
}

/// Coerce a Python value into a JSON Schema string. Accepts either a Python
/// `str` (used as-is) or any JSON-serializable object (typically a `dict`
/// such as ``pydantic.BaseModel.model_json_schema()``).
pub(crate) fn coerce_schema_to_json_string(value: Bound<'_, PyAny>) -> PyResult<String> {
    coerce_json_to_string(value, "schema")
}

pub(crate) fn coerce_custom_format_to_json_string(value: Bound<'_, PyAny>) -> PyResult<String> {
    coerce_json_to_string(value, "custom_format")
}

pub(crate) fn coerce_json_to_string(value: Bound<'_, PyAny>, value_name: &str) -> PyResult<String> {
    if let Ok(s) = value.extract::<String>() {
        return Ok(s);
    }
    let json_value: serde_json::Value = depythonize(&value).map_err(|err| {
        PyValueError::new_err(format!(
            "{value_name} must be a JSON string or a JSON-serializable object: {err}"
        ))
    })?;
    serde_json::to_string(&json_value).map_err(|err| {
        PyValueError::new_err(format!("failed to serialize {value_name} to JSON: {err}"))
    })
}

pub(crate) fn json_value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(value) => {
            Ok((*value).into_pyobject(py)?.to_owned().unbind().into_any())
        }
        serde_json::Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                Ok(integer.into_pyobject(py)?.unbind().into_any())
            } else if let Some(integer) = value.as_u64() {
                Ok(integer.into_pyobject(py)?.unbind().into_any())
            } else if let Some(float) = value.as_f64() {
                Ok(float.into_pyobject(py)?.unbind().into_any())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(value) => {
            Ok(value.clone().into_pyobject(py)?.unbind().into_any())
        }
        serde_json::Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(json_value_to_py(py, value)?)?;
            }
            Ok(list.unbind().into_any())
        }
        serde_json::Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                dict.set_item(key, json_value_to_py(py, value)?)?;
            }
            Ok(dict.unbind().into_any())
        }
    }
}

pub(crate) fn raw_content_json_to_message_content(
    raw: &str,
) -> PyResult<genai::chat::MessageContent> {
    let value = serde_json::from_str::<serde_json::Value>(raw).map_err(|err| {
        PyValueError::new_err(format!("raw_content must be JSON-serializable: {err}"))
    })?;
    let values = match value {
        serde_json::Value::Array(items) => items,
        other => vec![other],
    };
    let parts = values
        .into_iter()
        .map(|data| {
            ContentPart::Custom(CustomPart {
                model_iden: None,
                data,
            })
        })
        .collect::<Vec<_>>();
    Ok(genai::chat::MessageContent::from_parts(parts))
}

const ROLES_LOWER: [&str; 4] = ["system", "user", "assistant", "tool"];
const CONTENT_PART_KEYS_LOWER: [&str; 7] = [
    "text",
    "binary",
    "tool_call",
    "tool_response",
    "thought_signature",
    "reasoning_content",
    "custom",
];

fn title_case_role(s: &str) -> Option<&'static str> {
    match s {
        "system" => Some("System"),
        "user" => Some("User"),
        "assistant" => Some("Assistant"),
        "tool" => Some("Tool"),
        _ => None,
    }
}

fn title_case_content_part(s: &str) -> Option<&'static str> {
    match s {
        "text" => Some("Text"),
        "binary" => Some("Binary"),
        "tool_call" => Some("ToolCall"),
        "tool_response" => Some("ToolResponse"),
        "thought_signature" => Some("ThoughtSignature"),
        "reasoning_content" => Some("ReasoningContent"),
        "custom" => Some("Custom"),
        _ => None,
    }
}

/// Inverse of `title_case_content_part`: rewrite the externally-tagged
/// `ContentPart` variant names in a serialized `MessageContent` value from
/// Rust title-case to the Python-facing lowercase form. Variants we don't
/// know are left untouched (forward-compat with upstream rust-genai adding
/// new parts).
pub(crate) fn lowercase_content_part_keys(value: &mut serde_json::Value) {
    let Some(parts) = value.as_array_mut() else {
        return;
    };
    for part in parts {
        let Some(obj) = part.as_object_mut() else {
            continue;
        };
        let Some(title_key) = obj.keys().next().cloned() else {
            continue;
        };
        let lower = match title_key.as_str() {
            "Text" => "text",
            "Binary" => "binary",
            "ToolCall" => "tool_call",
            "ToolResponse" => "tool_response",
            "ThoughtSignature" => "thought_signature",
            "ReasoningContent" => "reasoning_content",
            "Custom" => "custom",
            _ => continue,
        };
        let inner = obj.remove(&title_key).unwrap();
        obj.insert(lower.to_string(), inner);
    }
}

/// Strictly require lowercase variant names and title-case them into the
/// serde form. Any other spelling (including native title-case) is a
/// ValueError — rust-genai's enum names are an implementation detail; the
/// dict shape is documented lowercase-only.
pub(crate) fn normalize_chat_request_value(value: &mut serde_json::Value) -> PyResult<()> {
    let Some(messages) = value.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return Ok(());
    };
    for (msg_idx, msg) in messages.iter_mut().enumerate() {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        if let Some(role_value) = obj.get("role")
            && let Some(role_str) = role_value.as_str()
        {
            let title = title_case_role(role_str).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "messages[{msg_idx}].role: expected one of {:?}, got {role_str:?}",
                    ROLES_LOWER
                ))
            })?;
            obj.insert("role".to_string(), serde_json::Value::String(title.into()));
        }
        let Some(parts) = obj.get_mut("content").and_then(|v| v.as_array_mut()) else {
            continue;
        };
        normalize_content_parts(parts, &format!("messages[{msg_idx}].content"))?;
    }
    Ok(())
}

/// Title-case the externally-tagged variant keys on a top-level content-part
/// list (lowercase ``text`` → ``Text``, etc.) so serde can deserialize it
/// into ``Vec<ContentPart>`` / ``MessageContent``. Errors mention the given
/// path (``messages[N].content`` for request-side, ``content`` for the
/// response-side constructor) so callers get a clear pointer on malformed
/// input.
pub(crate) fn normalize_content_parts(parts: &mut [serde_json::Value], path: &str) -> PyResult<()> {
    for (part_idx, part) in parts.iter_mut().enumerate() {
        let Some(part_obj) = part.as_object_mut() else {
            continue;
        };
        if part_obj.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "{path}[{part_idx}]: expected a single-key dict ({:?}), got {} keys",
                CONTENT_PART_KEYS_LOWER,
                part_obj.len()
            )));
        }
        let lower_key = part_obj.keys().next().cloned().unwrap();
        let inner = part_obj.remove(&lower_key).unwrap();
        let title = title_case_content_part(&lower_key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{path}[{part_idx}]: expected one of {:?}, got {:?}",
                CONTENT_PART_KEYS_LOWER, lower_key
            ))
        })?;
        let inner = if lower_key == "custom" {
            serde_json::json!({
                "model_iden": null,
                "data": inner,
            })
        } else {
            inner
        };
        part_obj.insert(title.into(), inner);
    }
    Ok(())
}

/// Depythonize a Python object (or None) into a ``MessageContent``. Accepts
/// the same lowercase-keyed content-part dict form the client emits.
pub(crate) fn coerce_message_content(
    obj: Option<Bound<'_, PyAny>>,
) -> PyResult<genai::chat::MessageContent> {
    let Some(obj) = obj else {
        return Ok(genai::chat::MessageContent::default());
    };
    if obj.is_none() {
        return Ok(genai::chat::MessageContent::default());
    }
    let mut value: serde_json::Value = depythonize(&obj).map_err(|err| {
        PyValueError::new_err(format!(
            "content must be a list of content-part dicts: {err}"
        ))
    })?;
    let parts = value
        .as_array_mut()
        .ok_or_else(|| PyValueError::new_err("content must be a list of content-part dicts"))?;
    normalize_content_parts(parts, "content")?;
    serde_json::from_value(value)
        .map_err(|err| PyValueError::new_err(format!("content list does not match schema: {err}")))
}

/// Parse a case-insensitive effort keyword into a `ReasoningEffort`.
///
/// Accepted: "none", "zero", "minimal", "low", "medium", "high", "xhigh",
/// "max", or "budget:<n>" where `<n>` is a non-negative u32 token
/// budget.
pub(crate) fn parse_reasoning_effort(raw: &str) -> Option<ReasoningEffort> {
    let trimmed = raw.trim().to_ascii_lowercase();
    if let Some(budget) = trimmed.strip_prefix("budget:") {
        return budget
            .trim()
            .parse::<u32>()
            .ok()
            .map(ReasoningEffort::Budget);
    }
    match trimmed.as_str() {
        "none" | "zero" => Some(ReasoningEffort::Zero),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

pub(crate) fn validate_reasoning_effort(raw: &str) -> PyResult<()> {
    parse_reasoning_effort(raw).map(|_| ()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "invalid reasoning_effort {raw:?}. expected one of: none, zero, \
             minimal, low, medium, high, xhigh, max, or budget:<u32>"
        ))
    })
}

/// Parse a case-insensitive cache-control keyword into a `CacheControl`.
///
/// Accepted: "ephemeral" (default 5m), "memory", "ephemeral_5m",
/// "ephemeral_1h", "ephemeral_24h". Returns `None` for unrecognized
/// values so callers get a silent no-op rather than a hard error —
/// matches the rest of ChatOptions' "everything is optional" pattern.
pub(crate) fn parse_cache_control(raw: &str) -> Option<CacheControl> {
    let trimmed = raw.trim().to_ascii_lowercase();
    match trimmed.as_str() {
        "ephemeral" => Some(CacheControl::Ephemeral),
        "memory" => Some(CacheControl::Memory),
        "ephemeral_5m" | "5m" => Some(CacheControl::Ephemeral5m),
        "ephemeral_1h" | "1h" => Some(CacheControl::Ephemeral1h),
        "ephemeral_24h" | "24h" => Some(CacheControl::Ephemeral24h),
        _ => None,
    }
}

pub(crate) fn build_header_map(headers: &HashMap<String, String>) -> PyResult<HeaderMap> {
    let mut map = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|err| PyValueError::new_err(format!("invalid header name {name:?}: {err}")))?;
        let header_value = HeaderValue::from_str(value).map_err(|err| {
            PyValueError::new_err(format!("invalid header value for {name:?}: {err}"))
        })?;
        map.insert(header_name, header_value);
    }
    Ok(map)
}

/// Apply a provider's constrained-decoding schema normalization to a JSON
/// Schema, without making a request.
///
/// rust-genai applies this automatically at the adapter boundary, so this
/// is for inspection: seeing what a provider will actually enforce before
/// discovering it in a rejected request. The rewrite can be lossy — OpenAI
/// requires every declared property to be required, and both providers
/// require `additionalProperties: false` on constrained objects — so the
/// result may accept fewer instances than the input.
#[pyfunction]
#[pyo3(signature = (schema, dialect))]
pub(crate) fn sanitize_json_schema(
    py: Python<'_>,
    schema: Bound<'_, PyAny>,
    dialect: &str,
) -> PyResult<Py<PyAny>> {
    let dialect = match dialect.trim().to_ascii_lowercase().as_str() {
        "openai_strict" => JsonSchemaDialect::OpenAiStrict,
        "openai_strict_tool" => JsonSchemaDialect::OpenAiStrictTool,
        "anthropic_structured" => JsonSchemaDialect::AnthropicStructured,
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid dialect {other:?}. expected one of: openai_strict, \
                 openai_strict_tool, anthropic_structured"
            )));
        }
    };
    let value: serde_json::Value = depythonize(&schema)
        .map_err(|err| PyValueError::new_err(format!("schema must be JSON-compatible: {err}")))?;
    json_value_to_py(py, &genai::chat::sanitize_json_schema(&value, dialect))
}
