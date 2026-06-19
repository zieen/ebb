use futures::{Stream, StreamExt, stream::BoxStream};

use super::{
    chat::{GeminiError, GenerateContentRequest, GenerateContentResponse},
    client::Client,
};

impl GenerateContentResponse {
    pub fn text_delta(&self) -> Option<String> {
        let text = self.text();
        if text.is_empty() { None } else { Some(text) }
    }
}

impl Client {
    pub async fn stream_events<R: Into<GenerateContentRequest>>(
        &self,
        model: &str,
        req: R,
    ) -> Result<BoxStream<'static, Result<GenerateContentResponse, GeminiError>>, GeminiError> {
        let req: GenerateContentRequest = req.into();
        let url = self.stream_generate_content_url(model);
        let headers = self.build_headers()?;

        let resp: reqwest::Response = self
            .http
            .post(url)
            .headers(headers)
            .json(&req)
            .send()
            .await?;
        let status: reqwest::StatusCode = resp.status();
        if !status.is_success() {
            let body: String = resp.text().await?;
            return Err(GeminiError::UnexpectedStatus { status, body });
        }

        Ok(parse_stream_from_bytes(resp.bytes_stream()))
    }

    pub async fn stream_text<R: Into<GenerateContentRequest>>(
        &self,
        model: &str,
        req: R,
    ) -> Result<String, GeminiError> {
        let req: GenerateContentRequest = req.into();
        let url = self.stream_generate_content_url(model);
        let headers = self.build_headers()?;

        let resp: reqwest::Response = self
            .http
            .post(url)
            .headers(headers)
            .json(&req)
            .send()
            .await?;
        let status: reqwest::StatusCode = resp.status();
        if !status.is_success() {
            let body: String = resp.text().await?;
            return Err(GeminiError::UnexpectedStatus { status, body });
        }

        let is_event_stream = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains("text/event-stream"));

        if is_event_stream {
            let mut stream = parse_stream_from_bytes(resp.bytes_stream());
            let mut out = String::new();
            while let Some(evt) = stream.next().await.transpose()? {
                let text = evt.text();
                if !text.is_empty() {
                    out.push_str(&text);
                }
            }
            if out.is_empty() {
                return Err(GeminiError::MissingText);
            }
            return Ok(out);
        }

        let body = resp.text().await?;
        let body_trim = body.trim_start();
        let out = if body_trim.starts_with('[') {
            let v: Vec<GenerateContentResponse> = serde_json::from_str(&body)?;
            v.into_iter().map(|e| e.text()).collect::<Vec<_>>().concat()
        } else if body_trim.starts_with('{') {
            let v: GenerateContentResponse = serde_json::from_str(&body)?;
            v.text()
        } else {
            String::new()
        };

        if out.is_empty() {
            return Err(GeminiError::MissingText);
        }

        Ok(out)
    }
}

fn parse_sse_data(block: &str) -> Option<&str> {
    for line in block.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("data:") {
            return Some(rest.trim_start());
        }
    }
    None
}

fn bytes_to_string(bytes: bytes::Bytes) -> Result<String, GeminiError> {
    Ok(String::from_utf8(bytes.to_vec())?)
}

pub(crate) fn parse_stream_from_bytes<S>(
    bytes_stream: S,
) -> BoxStream<'static, Result<GenerateContentResponse, GeminiError>>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
{
    futures::stream::unfold(
        (bytes_stream, String::new()),
        |(mut stream, mut buf)| async move {
            loop {
                if let Some(nl_idx) = buf.find('\n') {
                    let raw_line = buf[..nl_idx].to_string();
                    let line = raw_line.trim_end_matches('\r').trim();
                    if let Some(rest) = line.strip_prefix("data:") {
                        let data = rest.trim_start().to_string();
                        buf.drain(..nl_idx + 1);
                        if data.is_empty() {
                            continue;
                        }
                        if data == "[DONE]" || data == "DONE" {
                            continue;
                        }
                        if data.starts_with('{') {
                            let evt = serde_json::from_str::<GenerateContentResponse>(&data)
                                .map_err(GeminiError::Json);
                            return Some((evt, (stream, buf)));
                        }
                        continue;
                    }

                    if line.starts_with("event:") || line.is_empty() {
                        buf.drain(..nl_idx + 1);
                        continue;
                    }
                }

                let boundary = if let Some(idx) = buf.find("\r\n\r\n") {
                    Some((idx, 4))
                } else if let Some(idx) = buf.find("\n\n") {
                    Some((idx, 2))
                } else {
                    None
                };

                if let Some((idx, sep_len)) = boundary {
                    let block: String = buf.drain(..idx + sep_len).collect();
                    let block = block.trim_end_matches('\n').trim_end_matches('\r');
                    let Some(data) = parse_sse_data(&block).or_else(|| {
                        let t = block.trim();
                        if t.starts_with('{') || t.starts_with('[') {
                            Some(t)
                        } else {
                            None
                        }
                    }) else {
                        continue;
                    };

                    if data.is_empty() {
                        continue;
                    }

                    if data == "[DONE]" || data == "DONE" {
                        continue;
                    }

                    let event = serde_json::from_str::<GenerateContentResponse>(data)
                        .map_err(GeminiError::Json);
                    return Some((event, (stream, buf)));
                }

                let trimmed = buf.trim_start();
                if trimmed.starts_with('{') {
                    let de = serde_json::Deserializer::from_str(trimmed);
                    let mut it = de.into_iter::<GenerateContentResponse>();
                    match it.next() {
                        Some(Ok(v)) => {
                            let offset = it.byte_offset();
                            let drain_len = buf.len() - trimmed.len() + offset;
                            buf.drain(..drain_len);
                            return Some((Ok(v), (stream, buf)));
                        }
                        Some(Err(e)) if e.is_eof() => {}
                        Some(Err(e)) => return Some((Err(GeminiError::Json(e)), (stream, buf))),
                        None => {}
                    }
                }

                match stream.next().await {
                    Some(Ok(chunk)) => match bytes_to_string(chunk) {
                        Ok(s) => buf.push_str(&s),
                        Err(e) => return Some((Err(e), (stream, buf))),
                    },
                    Some(Err(e)) => return Some((Err(GeminiError::Http(e)), (stream, buf))),
                    None => return None,
                }
            }
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;

    use super::*;
    use crate::llm::gemini::chat::{
        Content, FunctionDeclaration, GenerateContentRequest, Part, Tool,
    };
    use dotenv::dotenv;

    struct ReqWrap(GenerateContentRequest);

    impl From<ReqWrap> for GenerateContentRequest {
        fn from(value: ReqWrap) -> Self {
            value.0
        }
    }

    #[tokio::test]
    async fn parse_stream_from_bytes_parses_ndjson() {
        let chunks: Vec<Result<bytes::Bytes, reqwest::Error>> =
            vec![Ok(bytes::Bytes::from_static(
                br#"{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}
{"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}
"#,
            ))];

        let stream = futures::stream::iter(chunks);
        let events = parse_stream_from_bytes(stream)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let text: String = events.iter().map(|e| e.text()).collect();
        assert_eq!(text, "Hello");
    }

    #[tokio::test]
    async fn runtime_stream_text() {
        let _lock = crate::llm::chat::ENV_LOCK.lock().unwrap();
        dotenv().ok();
        let Ok(api_key) = std::env::var("GEMINI_API_KEY") else {
            return;
        };

        let client = Client::new(api_key);
        let req = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part {
                    text: Some("Reply with the single word OK.".to_string()),
                    inline_data: None,
                    function_call: None,
                    function_response: None,
                }],
            }],
            tools: None,
            generation_config: None,
        };

        let text = client
            .stream_text("gemini-3.5-flash", ReqWrap(req))
            .await
            .unwrap();
        assert!(text.to_lowercase().contains("ok"));
    }

    #[tokio::test]
    async fn runtime_stream_events_with_tool() {
        let _lock = crate::llm::chat::ENV_LOCK.lock().unwrap();
        dotenv().ok();
        let Ok(api_key) = std::env::var("GEMINI_API_KEY") else {
            return;
        };

        let client = Client::new(api_key);
        let req = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part {
                    text: Some("You are a weather bot. The user asks: What's the weather like in Tokyo? Call the get_current_weather tool to find out.".to_string()),
                    inline_data: None,
                    function_call: None,
                    function_response: None,
                }],
            }],
            tools: Some(vec![Tool {
                function_declarations: vec![FunctionDeclaration {
                    name: "get_current_weather".to_string(),
                    description: Some("Gets the current weather for a given location".to_string()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g. Tokyo"
                            }
                        },
                        "required": ["location"]
                    }),
                }],
            }]),
            generation_config: None,
        };

        let mut stream = client
            .stream_events("gemini-3.5-flash", ReqWrap(req))
            .await
            .unwrap();

        let mut has_tool_call = false;
        while let Some(Ok(resp)) = stream.next().await {
            for candidate in resp.candidates {
                if let Some(content) = candidate.content {
                    for part in content.parts {
                        if let Some(fc) = part.function_call {
                            assert_eq!(fc.name, "get_current_weather");
                            assert!(fc.args.get("location").is_some());
                            has_tool_call = true;
                        }
                    }
                }
            }
        }

        assert!(has_tool_call, "Expected a tool call but got none");
    }
}
