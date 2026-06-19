use reqwest::header::{HeaderMap, HeaderValue};

use super::chat::GeminiError;

#[derive(Clone)]
pub struct Client {
    pub(crate) http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl Client {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub(crate) fn build_headers(&self) -> Result<HeaderMap, GeminiError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-goog-api-key",
            HeaderValue::from_str(&self.api_key).map_err(|_| GeminiError::InvalidHeaderValue)?,
        );
        Ok(headers)
    }

    pub(crate) fn generate_content_url(&self, model: &str) -> String {
        format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            model
        )
    }

    pub(crate) fn stream_generate_content_url(&self, model: &str) -> String {
        format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
            self.base_url.trim_end_matches('/'),
            model
        )
    }
}
