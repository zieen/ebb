use reqwest::header::{HeaderMap, HeaderValue};

use super::chat::ClaudeError;

#[derive(Clone)]
pub struct Client {
    pub(crate) http: reqwest::Client,
    api_key: String,
    base_url: String,
    anthropic_version: String,
}

impl Client {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            anthropic_version: "2023-06-01".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_anthropic_version(mut self, anthropic_version: impl Into<String>) -> Self {
        self.anthropic_version = anthropic_version.into();
        self
    }

    pub(crate) fn build_headers(&self) -> Result<HeaderMap, ClaudeError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).map_err(|_| ClaudeError::InvalidHeaderValue)?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(&self.anthropic_version)
                .map_err(|_| ClaudeError::InvalidHeaderValue)?,
        );
        Ok(headers)
    }

    pub(crate) fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url.trim_end_matches('/'))
    }
}
