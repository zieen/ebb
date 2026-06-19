use futures::{Stream, StreamExt, stream::BoxStream};
use serde::Deserialize;

use super::{
    chat::{ClaudeError, ContentBlock, MessagesRequest},
    client::Client,
};

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart,
    #[serde(rename = "content_block_start")]
    ContentBlockStart { content_block: ContentBlock },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: Delta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop,
    #[serde(rename = "message_delta")]
    MessageDelta,
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Delta {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
    pub partial_json: Option<String>,
}

impl StreamEvent {
    pub fn text_delta(&self) -> Option<&str> {
        match self {
            StreamEvent::ContentBlockDelta { delta } if delta.kind == "text_delta" => {
                delta.text.as_deref()
            }
            _ => None,
        }
    }

    pub fn input_json_delta(&self) -> Option<&str> {
        match self {
            StreamEvent::ContentBlockDelta { delta } if delta.kind == "input_json_delta" => {
                delta.partial_json.as_deref()
            }
            _ => None,
        }
    }
}

impl Client {
    pub async fn stream_events<R: Into<MessagesRequest>>(
        &self,
        req: R,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ClaudeError>>, ClaudeError> {
        let req: MessagesRequest = req.into();
        let url = self.messages_url();
        let headers = self.build_headers()?;

        let resp = self
            .http
            .post(url)
            .headers(headers)
            .json(&MessagesRequest {
                stream: Some(true),
                ..req
            })
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await?;
            return Err(ClaudeError::UnexpectedStatus { status, body });
        }

        Ok(parse_stream_from_bytes(resp.bytes_stream()))
    }
}

fn parse_sse_event(block: &str) -> Option<&str> {
    for line in block.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("data:") {
            return Some(rest.trim_start());
        }
    }
    None
}

fn decode_event_json(data: &str) -> Result<StreamEvent, ClaudeError> {
    Ok(serde_json::from_str::<StreamEvent>(data)?)
}

fn bytes_to_string(bytes: bytes::Bytes) -> Result<String, ClaudeError> {
    Ok(String::from_utf8(bytes.to_vec())?)
}

pub(crate) fn parse_stream_from_bytes<S>(
    bytes_stream: S,
) -> BoxStream<'static, Result<StreamEvent, ClaudeError>>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
{
    futures::stream::unfold(
        (bytes_stream, String::new()),
        |(mut stream, mut buf)| async move {
            loop {
                if let Some(idx) = buf.find("\n\n") {
                    let block: String = buf.drain(..idx + 2).collect();
                    let block = block.trim_end_matches('\n').trim_end_matches('\r');
                    if let Some(data) = parse_sse_event(block) {
                        if data.is_empty() {
                            continue;
                        }
                        let event = decode_event_json(data);
                        return Some((event, (stream, buf)));
                    }
                    continue;
                }

                match stream.next().await {
                    Some(Ok(chunk)) => match bytes_to_string(chunk) {
                        Ok(s) => buf.push_str(&s),
                        Err(e) => return Some((Err(e), (stream, buf))),
                    },
                    Some(Err(e)) => return Some((Err(ClaudeError::Http(e)), (stream, buf))),
                    None => return None,
                }
            }
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    struct ReqWrap(MessagesRequest);

    impl From<ReqWrap> for MessagesRequest {
        fn from(value: ReqWrap) -> Self {
            value.0
        }
    }

    #[tokio::test]
    async fn stream_events_parses_text_deltas() {
        let server = MockServer::start().await;

        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\"}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "testkey"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let req = MessagesRequest {
            model: "claude-3-5-sonnet-20240620".to_string(),
            max_tokens: 16,
            messages: vec![],
            system: None,
            stream: Some(true),
            tools: None,
            tool_choice: None,
        };

        let client = Client::new("testkey").with_base_url(server.uri());
        let events = client.stream_events(ReqWrap(req)).await.unwrap();
        let text: String = events
            .try_filter_map(|e| async move { Ok(e.text_delta().map(ToString::to_string)) })
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .concat();

        assert_eq!(text, "Hello");
    }

    #[tokio::test]
    async fn stream_events_parses_tool_use_and_input_json_delta() {
        let server = MockServer::start().await;

        let sse_body = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Shanghai\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\"}\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "testkey"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let req = MessagesRequest {
            model: "claude-3-5-sonnet-20240620".to_string(),
            max_tokens: 16,
            messages: vec![],
            system: None,
            stream: Some(true),
            tools: None,
            tool_choice: None,
        };

        let client = Client::new("testkey").with_base_url(server.uri());
        let events = client
            .stream_events(ReqWrap(req))
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(
            events[0],
            StreamEvent::ContentBlockStart {
                content_block: ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({}),
                }
            }
        );
        assert_eq!(
            events[1].input_json_delta(),
            Some("{\"city\":\"Shanghai\"}")
        );
    }

    #[tokio::test]
    async fn runtime_stream_events() {
        use dotenv::dotenv;
        use futures::StreamExt;

        let _lock = crate::llm::chat::ENV_LOCK.lock().unwrap();
        dotenv().ok();
        let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
            return;
        };

        let client = Client::new(api_key);
        let req = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 32,
            messages: vec![crate::llm::anthorpic::chat::Message {
                role: "user".to_string(),
                content: crate::llm::anthorpic::chat::MessageContent::Text(
                    "Reply with the single word OK.".to_string(),
                ),
            }],
            ..Default::default()
        };

        let mut events = client.stream_events(req).await.unwrap();
        let mut text = String::new();
        let mut saw_message_stop = false;

        while let Some(evt) = events.next().await {
            let evt = evt.unwrap();
            if let Some(delta) = evt.text_delta() {
                text.push_str(delta);
            }
            if matches!(evt, StreamEvent::MessageStop) {
                saw_message_stop = true;
                break;
            }
        }

        assert!(saw_message_stop);
        assert!(text.to_lowercase().contains("ok"));
    }

    #[tokio::test]
    async fn runtime_stream_tool_use() {
        use dotenv::dotenv;
        use futures::StreamExt;

        let _lock = crate::llm::chat::ENV_LOCK.lock().unwrap();
        dotenv().ok();
        let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
            return;
        };

        let client = Client::new(api_key);
        let req: MessagesRequest = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 256,
            messages: vec![crate::llm::anthorpic::chat::Message {
                role: "user".to_string(),
                content: crate::llm::anthorpic::chat::MessageContent::Text(
                    "Use the get_weather tool with city = Shanghai.".to_string(),
                ),
            }],
            tools: Some(vec![crate::llm::anthorpic::chat::ToolDefinition {
                name: "get_weather".to_string(),
                description: Some("get weather by city".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
            }]),
            tool_choice: Some(crate::llm::anthorpic::chat::ToolChoice::Auto),
            ..Default::default()
        };

        let mut events = client.stream_events(req).await.unwrap();

        let mut tool_use_id: Option<String> = None;
        let mut tool_name: Option<String> = None;
        let mut input_json = String::new();
        let mut parsed_input: Option<serde_json::Value> = None;

        for _ in 0..500 {
            let Some(evt) = events.next().await else {
                break;
            };
            let evt = evt.unwrap();

            if let StreamEvent::ContentBlockStart { content_block } = &evt {
                if let ContentBlock::ToolUse { id, name, .. } = content_block {
                    tool_use_id = Some(id.clone());
                    tool_name = Some(name.clone());
                }
            }

            if let Some(delta) = evt.input_json_delta() {
                input_json.push_str(delta);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&input_json) {
                    parsed_input = Some(v);
                }
            }

            if tool_use_id.is_some() && parsed_input.is_some() {
                break;
            }
        }

        assert_eq!(tool_name.as_deref(), Some("get_weather"));
        assert!(tool_use_id.is_some());

        let input: serde_json::Value = parsed_input.unwrap();
        assert_eq!(input.get("city").and_then(|v| v.as_str()), Some("Shanghai"));
    }
}
