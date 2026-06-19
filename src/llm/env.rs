use crate::llm::{anthorpic, gemini, openai, types::LlmError};

pub fn openai_client_from_env() -> Result<openai::client::Client, LlmError> {
    let api_key: String = std::env::var("OPENAI_API_KEY").map_err(|_| LlmError::MissingEnv {
        key: "OPENAI_API_KEY",
    })?;
    let client = openai::client::Client::with_api_key(api_key);
    Ok(client)
}

pub fn gemini_client_from_env() -> Result<gemini::client::Client, LlmError> {
    let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| LlmError::MissingEnv {
        key: "GEMINI_API_KEY",
    })?;
    let mut client = gemini::client::Client::new(api_key);
    if let Ok(base_url) = std::env::var("GEMINI_BASE_URL") {
        client = client.with_base_url(base_url);
    }
    Ok(client)
}

pub fn anthorpic_client_from_env() -> Result<anthorpic::client::Client, LlmError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| LlmError::MissingEnv {
        key: "ANTHROPIC_API_KEY",
    })?;
    let mut client = anthorpic::client::Client::new(api_key);
    if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
        client = client.with_base_url(base_url);
    }
    if let Ok(v) = std::env::var("ANTHROPIC_VERSION") {
        client = client.with_anthropic_version(v);
    }
    Ok(client)
}

pub fn deepseek_client_from_env() -> Result<openai::client::Client, LlmError> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| LlmError::MissingEnv {
        key: "DEEPSEEK_API_KEY",
    })?;
    let base_url = "https://api.deepseek.com".to_string();

    Ok(openai::client::Client::with_key_and_base_url(
        api_key, base_url,
    ))
}
