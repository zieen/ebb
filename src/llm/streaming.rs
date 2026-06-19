use futures::{StreamExt, stream::BoxStream};

use crate::llm::{
    anthorpic,
    env::*,
    gemini,
    model_list::{is_anthorpic_model, is_deepseek_model, is_gemini_model, is_openai_model},
    openai,
    types::*,
};

use tokio::sync::mpsc;

use crate::llm::openai::streaming::StreamEventExt as OpenAIStreamEventExt;

#[derive(Debug)]
pub enum StreamEvent {
    OpenAI(openai::types::chat::CreateChatCompletionStreamResponse),
    Gemini(gemini::chat::GenerateContentResponse),
    Anthorpic(anthorpic::streaming::StreamEvent),
}

impl StreamEvent {
    pub fn text_delta(&self) -> Option<String> {
        match self {
            StreamEvent::OpenAI(evt) => evt.text_delta().map(ToString::to_string),
            StreamEvent::Gemini(evt) => evt.text_delta(),
            StreamEvent::Anthorpic(evt) => evt.text_delta().map(ToString::to_string),
        }
    }

    pub fn input_json_delta(&self) -> Option<String> {
        match self {
            StreamEvent::Anthorpic(evt) => evt.input_json_delta().map(ToString::to_string),
            _ => None,
        }
    }
}

pub async fn stream_events(
    kwargs: ChatKwargs,
) -> Result<BoxStream<'static, Result<StreamEvent, LlmError>>, LlmError> {
    let model = kwargs.model.clone();

    if is_openai_model(&model) {
        let client = openai::client::Client::new();
        let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
        let stream = client.stream_events(req).await?;
        return Ok(stream
            .map(|item| {
                item.map(StreamEvent::OpenAI)
                    .map_err(openai::chat::OpenAIError::from)
                    .map_err(LlmError::from)
            })
            .boxed());
    }

    if is_gemini_model(&model) {
        let client = gemini_client_from_env()?;
        let req: gemini::chat::GenerateContentRequest = kwargs.into();
        let stream = client.stream_events(&model, req).await?;
        return Ok(stream
            .map(|item| item.map(StreamEvent::Gemini).map_err(LlmError::from))
            .boxed());
    }

    if is_anthorpic_model(&model) {
        let client = anthorpic_client_from_env()?;
        let req: anthorpic::chat::MessagesRequest = kwargs.into();
        let stream = client.stream_events(req).await?;
        return Ok(stream
            .map(|item| item.map(StreamEvent::Anthorpic).map_err(LlmError::from))
            .boxed());
    }

    if is_deepseek_model(&model) {
        let client = deepseek_client_from_env()?;
        let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
        let stream = client.stream_events(req).await?;
        return Ok(stream
            .map(|item| {
                item.map(StreamEvent::OpenAI)
                    .map_err(openai::chat::OpenAIError::from)
                    .map_err(LlmError::from)
            })
            .boxed());
    }

    Err(LlmError::UnknownModel(model))
}

pub async fn stream_text(kwargs: ChatKwargs) -> Result<String, LlmError> {
    let model = kwargs.model.clone();

    if is_openai_model(&model) {
        let client = openai::client::Client::new();
        let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
        return Ok(client.stream_text(req).await?);
    }

    if is_gemini_model(&model) {
        let client = gemini_client_from_env()?;
        let req: gemini::chat::GenerateContentRequest = kwargs.into();
        return Ok(client.stream_text(&model, req).await?);
    }

    if is_anthorpic_model(&model) {
        let mut stream = stream_events(kwargs).await?;
        let mut out = String::new();

        while let Some(evt) = stream.next().await.transpose()? {
            if let Some(delta) = evt.text_delta() {
                out.push_str(&delta);
            }
        }

        if out.is_empty() {
            return Err(anthorpic::chat::ClaudeError::MissingText.into());
        }

        return Ok(out);
    }

    Err(LlmError::UnknownModel(model))
}

pub enum StreamNextResp {
    Text(String),
    ToolCall(serde_json::Value),
}

pub async fn streaming(
    kwargs: ChatKwargs,
    output_chan: mpsc::Sender<StreamNextResp>,
) -> Result<(), LlmError> {
    let mut json_partial = String::new();
    let mut fn_buf;
    let mut args_buf = String::new();

    let mut stream = stream_events(kwargs).await?;

    while let Some(event) = stream.next().await {
        let Ok(e) = event else {
            continue;
        };

        match e {
            StreamEvent::OpenAI(evt) => {
                if let Some(delta) = evt.text_delta() {
                    if let Err(_) = output_chan
                        .send(StreamNextResp::Text(delta.to_string()))
                        .await
                    {
                        continue;
                    }
                } else {
                    for choice in &evt.choices {
                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            for tc in tool_calls {
                                if let Some(function) = &tc.function {
                                    let Some(function_name) = function.name.as_deref() else {
                                        continue;
                                    };

                                    let Some(arguments) = function.arguments.as_deref() else {
                                        continue;
                                    };

                                    args_buf.push_str(arguments);
                                    fn_buf = function_name.to_string();

                                    if let Err(_) =
                                        serde_json::from_str::<serde_json::Value>(&args_buf)
                                    {
                                        args_buf.clear();
                                        continue;
                                    }

                                    if let Err(_) = output_chan
                                        .send(StreamNextResp::ToolCall(serde_json::json!({
                                            "name": function_name,
                                            "arguments": arguments,
                                        })))
                                        .await
                                    {
                                        fn_buf.clear();
                                        args_buf.clear();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            StreamEvent::Gemini(evt) => {
                let mut is_tool_call = false;
                for candidate in &evt.candidates {
                    if let Some(content) = &candidate.content {
                        for part in &content.parts {
                            if let Some(fc) = &part.function_call {
                                is_tool_call = true;
                                if let Err(_) = output_chan
                                    .send(StreamNextResp::ToolCall(serde_json::json!({
                                        "name": fc.name,
                                        "arguments": fc.args.to_string(),
                                    })))
                                    .await
                                {
                                    continue;
                                }
                            }
                        }
                    }
                }

                if !is_tool_call {
                    if let Some(delta) = evt.text_delta() {
                        if let Err(_) = output_chan
                            .send(StreamNextResp::Text(delta.to_string()))
                            .await
                        {
                            continue;
                        }
                    }
                }
            }
            StreamEvent::Anthorpic(evt) => {
                if let Some(delta) = evt.text_delta() {
                    if let Err(_) = output_chan
                        .send(StreamNextResp::Text(delta.to_string()))
                        .await
                    {
                        continue;
                    }
                } else if let Some(json) = evt.input_json_delta() {
                    json_partial.push_str(&json);
                }

                if let Err(_) = serde_json::from_str::<serde_json::Value>(&json_partial) {
                    json_partial.clear();
                    continue;
                }

                let Ok(tool_call_json) = serde_json::from_str::<serde_json::Value>(&json_partial)
                else {
                    json_partial.clear();
                    continue;
                };

                if let Err(_) = output_chan
                    .send(StreamNextResp::ToolCall(tool_call_json))
                    .await
                {
                    json_partial.clear();
                }
            }
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    use futures::TryStreamExt;
    use tokio::sync::mpsc;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    use super::*;
    use crate::llm::{chat::ENV_LOCK, model_list::model_names};

    #[test]
    fn stream_event_text_delta_wraps_gemini_event() {
        let event = StreamEvent::Gemini(gemini::chat::GenerateContentResponse {
            candidates: vec![gemini::chat::Candidate {
                content: Some(gemini::chat::Content {
                    role: Some("model".to_string()),
                    parts: vec![gemini::chat::Part {
                        text: Some("Hello".to_string()),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                    }],
                }),
                finish_reason: None,
            }],
        });

        assert_eq!(event.text_delta().as_deref(), Some("Hello"));
        assert_eq!(event.input_json_delta(), None);
    }

    #[test]
    fn stream_event_input_json_delta_wraps_anthorpic_event() {
        let event = StreamEvent::Anthorpic(anthorpic::streaming::StreamEvent::ContentBlockDelta {
            delta: anthorpic::streaming::Delta {
                kind: "input_json_delta".to_string(),
                text: None,
                partial_json: Some("{\"city\":\"Shanghai\"}".to_string()),
            },
        });

        assert_eq!(
            event.input_json_delta().as_deref(),
            Some("{\"city\":\"Shanghai\"}")
        );
        assert_eq!(event.text_delta(), None);
    }

    #[tokio::test]
    async fn stream_text_routes_to_gemini_by_model() {
        let server = MockServer::start().await;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let prev_key = std::env::var("GEMINI_API_KEY").ok();
        let prev_base = std::env::var("GEMINI_BASE_URL").ok();

        unsafe {
            std::env::set_var("GEMINI_API_KEY", "testkey");
            std::env::set_var("GEMINI_BASE_URL", server.uri());
        }

        Mock::given(method("POST"))
            .and(path(
                "/v1beta/models/gemini-3.5-flash:streamGenerateContent",
            ))
            .and(header("x-goog-api-key", "testkey"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(vec![serde_json::json!({
                    "candidates": [{
                        "content": { "parts": [{"text":"OK"}] }
                    }]
                })]),
            )
            .mount(&server)
            .await;

        let text = stream_text(ChatKwargs {
            model: "gemini-3.5-flash".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            ..Default::default()
        })
        .await
        .unwrap();

        assert_eq!(text, "OK");

        unsafe {
            match prev_key {
                Some(v) => std::env::set_var("GEMINI_API_KEY", v),
                None => std::env::remove_var("GEMINI_API_KEY"),
            }
            match prev_base {
                Some(v) => std::env::set_var("GEMINI_BASE_URL", v),
                None => std::env::remove_var("GEMINI_BASE_URL"),
            }
        }
    }

    #[tokio::test]
    async fn stream_text_routes_to_anthorpic_by_model() {
        let server = MockServer::start().await;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let prev_key = std::env::var("ANTHROPIC_API_KEY").ok();
        let prev_base = std::env::var("ANTHROPIC_BASE_URL").ok();
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "testkey");
            std::env::set_var("ANTHROPIC_BASE_URL", server.uri());
        }

        let sse_body = concat!(
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

        let text = stream_text(ChatKwargs {
            model: model_names::CLAUDE_OPUS_4_8.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            ..Default::default()
        })
        .await
        .unwrap();

        assert_eq!(text, "Hello");

        unsafe {
            match prev_key {
                Some(v) => std::env::set_var("ANTHROPIC_API_KEY", v),
                None => std::env::remove_var("ANTHROPIC_API_KEY"),
            }
            match prev_base {
                Some(v) => std::env::set_var("ANTHROPIC_BASE_URL", v),
                None => std::env::remove_var("ANTHROPIC_BASE_URL"),
            }
        }
    }

    #[tokio::test]
    async fn stream_events_routes_to_anthorpic_by_model() {
        let server = MockServer::start().await;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let prev_key = std::env::var("ANTHROPIC_API_KEY").ok();
        let prev_base = std::env::var("ANTHROPIC_BASE_URL").ok();
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "testkey");
            std::env::set_var("ANTHROPIC_BASE_URL", server.uri());
        }

        let sse_body = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
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

        let events = stream_events(ChatKwargs {
            model: model_names::CLAUDE_OPUS_4_8.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            ..Default::default()
        })
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert_eq!(events[0].text_delta().as_deref(), Some("Hi"));

        unsafe {
            match prev_key {
                Some(v) => std::env::set_var("ANTHROPIC_API_KEY", v),
                None => std::env::remove_var("ANTHROPIC_API_KEY"),
            }
            match prev_base {
                Some(v) => std::env::set_var("ANTHROPIC_BASE_URL", v),
                None => std::env::remove_var("ANTHROPIC_BASE_URL"),
            }
        }
    }

    #[tokio::test]
    async fn streaming_routes_gemini_tool_calls() {
        let server = MockServer::start().await;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let prev_key = std::env::var("GEMINI_API_KEY").ok();
        let prev_base = std::env::var("GEMINI_BASE_URL").ok();
        unsafe {
            std::env::set_var("GEMINI_API_KEY", "testkey");
            std::env::set_var("GEMINI_BASE_URL", server.uri());
        }

        Mock::given(method("POST"))
            .and(path(
                "/v1beta/models/gemini-3.5-flash:streamGenerateContent",
            ))
            .and(header("x-goog-api-key", "testkey"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(concat!(
                        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_current_weather\",\"args\":{\"location\":\"Tokyo\"}}}]}}]}\n\n"
                    )),
            )
            .mount(&server)
            .await;

        let (tx, mut rx) = mpsc::channel(8);
        streaming(
            ChatKwargs {
                model: model_names::GEMINI_3_5_FLASH.to_string(),
                messages: vec![Message {
                    role: "user".to_string(),
                    content: "What's the weather like in Tokyo?".to_string(),
                }],
                tools: Some(vec![ChatTool {
                    name: "get_current_weather".to_string(),
                    description: Some("Gets the current weather for a given location".to_string()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": { "type": "string" }
                        },
                        "required": ["location"]
                    }),
                    handler: |args| args,
                }]),
                ..Default::default()
            },
            tx,
        )
        .await
        .unwrap();

        let event = rx.recv().await.expect("expected a streamed event");
        match event {
            StreamNextResp::ToolCall(tc) => {
                assert_eq!(tc["name"], "get_current_weather");
                assert_eq!(tc["arguments"], "{\"location\":\"Tokyo\"}");
            }
            StreamNextResp::Text(text) => panic!("expected tool call, got text: {text}"),
        }

        unsafe {
            match prev_key {
                Some(v) => std::env::set_var("GEMINI_API_KEY", v),
                None => std::env::remove_var("GEMINI_API_KEY"),
            }
            match prev_base {
                Some(v) => std::env::set_var("GEMINI_BASE_URL", v),
                None => std::env::remove_var("GEMINI_BASE_URL"),
            }
        }
    }

    #[tokio::test]
    async fn streaming_propagates_provider_setup_errors() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let prev_key = std::env::var("ANTHROPIC_API_KEY").ok();
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
        }

        let err = streaming(
            ChatKwargs {
                model: model_names::CLAUDE_OPUS_4_8.to_string(),
                messages: vec![Message {
                    role: "user".to_string(),
                    content: "hi".to_string(),
                }],
                ..Default::default()
            },
            mpsc::channel(1).0,
        )
        .await
        .expect_err("expected missing env error");

        match err {
            LlmError::MissingEnv { key } => assert_eq!(key, "ANTHROPIC_API_KEY"),
            other => panic!("unexpected error: {:?}", other),
        }

        unsafe {
            match prev_key {
                Some(v) => std::env::set_var("ANTHROPIC_API_KEY", v),
                None => std::env::remove_var("ANTHROPIC_API_KEY"),
            }
        }
    }

    #[tokio::test]
    async fn stream_text_unknown_model_errors() {
        let err = stream_text(ChatKwargs {
            model: "unknown-1".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            ..Default::default()
        })
        .await
        .unwrap_err();

        match err {
            LlmError::UnknownModel(model) => assert_eq!(model, "unknown-1"),
            _ => panic!("unexpected error"),
        }
    }

    #[tokio::test]
    async fn test_stream_events() {
        let (tx, mut rx) = mpsc::channel(32);

        use dotenv::dotenv;
        dotenv().ok();

        let req = ChatKwargs {
            model: model_names::DEEPSEEK_V4_PRO.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            ..Default::default()
        };

        tokio::spawn(async move {
            if let Ok(mut stream) = stream_events(req).await {
                while let Some(event) = stream.next().await {
                    if let Err(e) = tx.send(event).await {
                        eprintln!("Failed to send event: {:?}", e);
                    }
                }
            }
        });

        while let Some(event) = rx.recv().await {
            match event {
                Ok(resp) => {
                    println!("{:?}", resp.text_delta());
                }
                Err(e) => {
                    eprintln!("Failed to receive event: {:?}", e);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_streaming_gemini_tool_call() {
        let server = MockServer::start().await;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let prev_key = std::env::var("GEMINI_API_KEY").ok();
        let prev_base = std::env::var("GEMINI_BASE_URL").ok();
        unsafe {
            std::env::set_var("GEMINI_API_KEY", "testkey");
            std::env::set_var("GEMINI_BASE_URL", server.uri());
        }

        Mock::given(method("POST"))
            .and(path(
                "/v1beta/models/gemini-3.5-flash:streamGenerateContent",
            ))
            .and(header("x-goog-api-key", "testkey"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(concat!(
                        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_current_weather\",\"args\":{\"location\":\"Tokyo\"}}}]}}]}\n\n"
                    )),
            )
            .mount(&server)
            .await;

        let req = ChatKwargs {
            model: model_names::GEMINI_3_5_FLASH.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "You are a weather bot. The user asks: What's the weather like in Tokyo? Call the get_current_weather tool to find out.".to_string(),
            }],
            tools: Some(vec![ChatTool {
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
                handler: |args| args,
            }]),
            ..Default::default()
        };

        let (tx, mut rx) = mpsc::channel(32);

        tokio::spawn(async move {
            if let Err(e) = streaming(req, tx).await {
                eprintln!("Streaming error: {:?}", e);
            }
        });

        let mut received_tool_call = false;

        while let Some(event) = rx.recv().await {
            match event {
                StreamNextResp::Text(t) => {
                    println!("Received text: {}", t);
                }
                StreamNextResp::ToolCall(tc) => {
                    println!("Received tool call: {}", tc);
                    assert_eq!(tc["name"], "get_current_weather");
                    received_tool_call = true;
                    break;
                }
            }
        }

        assert!(received_tool_call, "expected Gemini to emit a tool call");

        unsafe {
            match prev_key {
                Some(v) => std::env::set_var("GEMINI_API_KEY", v),
                None => std::env::remove_var("GEMINI_API_KEY"),
            }
            match prev_base {
                Some(v) => std::env::set_var("GEMINI_BASE_URL", v),
                None => std::env::remove_var("GEMINI_BASE_URL"),
            }
        }

        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());

        let req = ChatKwargs {
            model: model_names::CHATGPT_5_5.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "You are a weather bot. The user asks: What's the weather like in Tokyo? Call the get_current_weather tool to find out.".to_string(),
            }],
            tools: Some(vec![ChatTool {
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
                    "required": ["location"],
                    "additionalProperties": false
                }),
                handler: |args| args,
            }]),
            ..Default::default()
        };

        let (tx, mut rx) = mpsc::channel(32);

        tokio::spawn(async move {
            if let Err(e) = streaming(req, tx).await {
                eprintln!("Streaming error: {:?}", e);
            }
        });

        let mut received_tool_call = false;

        while let Some(event) = rx.recv().await {
            match event {
                StreamNextResp::Text(t) => {
                    println!("Received text: {}", t);
                }
                StreamNextResp::ToolCall(tc) => {
                    println!("Received tool call: {}", tc);
                    assert_eq!(tc["name"], "get_current_weather");
                    received_tool_call = true;
                    break;
                }
            }
        }

        assert!(received_tool_call, "Expected a tool call but got none");
    }

    #[tokio::test]
    async fn test_streaming_anthropic_tool_call() {
        use dotenv::dotenv;
        dotenv().ok();

        if std::env::var("ANTHROPIC_API_KEY").is_err() {
            return;
        }

        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());

        let req = ChatKwargs {
            model: crate::llm::model_list::model_names::CLAUDE_OPUS_4_8.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "You are a weather bot. The user asks: What's the weather like in Tokyo? Call the get_current_weather tool to find out.".to_string(),
            }],
            tools: Some(vec![ChatTool {
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
                handler: |args| args,
            }]),
            ..Default::default()
        };

        let (tx, mut rx) = mpsc::channel(32);

        tokio::spawn(async move {
            if let Err(e) = streaming(req, tx).await {
                eprintln!("Streaming error: {:?}", e);
            }
        });

        let mut received_tool_call = false;

        while let Some(event) = rx.recv().await {
            match event {
                StreamNextResp::Text(t) => {
                    println!("Received text: {}", t);
                }
                StreamNextResp::ToolCall(tc) => {
                    println!("Received tool call: {}", tc);
                    assert_eq!(tc["name"], "get_current_weather");
                    received_tool_call = true;
                    break;
                }
            }
        }

        assert!(received_tool_call, "Expected a tool call but got none");
    }

    #[tokio::test]
    async fn runtime_streaming_gemini_tool_call() {
        use dotenv::dotenv;
        dotenv().ok();

        let Ok(_) = std::env::var("GEMINI_API_KEY") else {
            return;
        };

        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());

        let req = ChatKwargs {
            model: model_names::GEMINI_3_5_FLASH.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "You are a weather bot. The user asks: What's the weather like in Tokyo? Call the get_current_weather tool to find out.".to_string(),
            }],
            tools: Some(vec![ChatTool {
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
                handler: |args| args,
            }]),
            ..Default::default()
        };

        let (tx, mut rx) = mpsc::channel(32);

        tokio::spawn(async move {
            if let Err(e) = streaming(req, tx).await {
                eprintln!("Streaming error: {:?}", e);
            }
        });

        let mut received_tool_call = false;

        while let Some(event) = rx.recv().await {
            match event {
                StreamNextResp::Text(t) => {
                    println!("Received text: {}", t);
                }
                StreamNextResp::ToolCall(tc) => {
                    println!("Received tool call: {}", tc);
                    assert_eq!(tc["name"], "get_current_weather");
                    received_tool_call = true;
                    break;
                }
            }
        }

        assert!(
            received_tool_call,
            "expected Gemini to emit a live tool call"
        );
    }
}
