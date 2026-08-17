//! Python bindings for the Rust [`genai`](https://github.com/jeremychone/rust-genai) crate.
//!
//! Module layout:
//! - [`convert`]  — pure Python ⇄ serde_json ⇄ rust-genai coercions
//! - [`errors`]   — rust-genai errors mapped onto Python exceptions
//! - [`chat`]     — request-side pyclasses (messages, requests, tools, options)
//! - [`response`] — response-side pyclasses (usage, responses, streaming)
//! - [`target`]   — routing types (adapter, model, endpoint, auth, target)
//! - [`client`]   — the `Client` pyclass and its construction helpers

mod chat;
mod client;
mod convert;
mod errors;
mod resolvers;
mod response;
mod target;

use pyo3::prelude::*;

#[pymodule]
fn _genai_pyo3(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(py, module)?;
    module.add_class::<client::PyClient>()?;
    module.add_class::<client::PyClientBuilder>()?;
    module.add_class::<target::PyAdapterKind>()?;
    module.add_class::<target::PyModelIden>()?;
    module.add_class::<target::PyEndpoint>()?;
    module.add_class::<target::PyAuthData>()?;
    module.add_class::<target::PyServiceTarget>()?;
    module.add_class::<chat::PyChatMessage>()?;
    module.add_class::<chat::PyChatRequest>()?;
    module.add_class::<chat::PyChatOptions>()?;
    module.add_class::<chat::PyJsonSpec>()?;
    module.add_class::<chat::PyTool>()?;
    module.add_class::<chat::PyToolCall>()?;
    module.add_class::<response::PyUsage>()?;
    module.add_class::<response::PyPromptTokensDetails>()?;
    module.add_class::<response::PyCompletionTokensDetails>()?;
    module.add_class::<response::PyCacheCreationDetails>()?;
    module.add_class::<response::PyChatResponse>()?;
    module.add_class::<response::PyStreamEnd>()?;
    module.add_class::<response::PyChatStreamEvent>()?;
    module.add_class::<response::PyChatStream>()?;
    Ok(())
}
