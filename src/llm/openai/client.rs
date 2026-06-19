use async_openai::{Client as AsyncOpenAIClient, config::OpenAIConfig};

#[derive(Clone)]
pub struct Client {
    pub(crate) inner: AsyncOpenAIClient<OpenAIConfig>,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    pub fn new() -> Self {
        Self {
            inner: AsyncOpenAIClient::new(),
        }
    }

    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        let config: OpenAIConfig = OpenAIConfig::new().with_api_key(api_key);
        Self {
            inner: AsyncOpenAIClient::with_config(config),
        }
    }

    pub fn with_key_and_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(base_url);
        Self {
            inner: AsyncOpenAIClient::with_config(config),
        }
    }
}
