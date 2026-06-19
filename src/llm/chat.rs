use serde::de::DeserializeOwned;

use crate::llm::env::*;
use crate::llm::types::*;

use crate::llm::{anthorpic, gemini, openai};

#[cfg(test)]
pub(crate) static ENV_LOCK: once_cell::sync::Lazy<std::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(()));

pub async fn chat(kwargs: ChatKwargs) -> Result<ChatResponse, LlmError> {
    let model = kwargs.model.clone();
    let Some(provider) = provider_for_model(&model) else {
        return Err(LlmError::UnknownModel(model));
    };

    match provider {
        Provider::OpenAI => {
            let client = openai_client_from_env()?;
            let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
            Ok(ChatResponse::OpenAI(client.chat(req).await?))
        }
        Provider::Gemini => {
            let client = gemini_client_from_env()?;
            let req: gemini::chat::GenerateContentRequest = kwargs.into();
            Ok(ChatResponse::Gemini(client.chat(&model, req).await?))
        }
        Provider::Anthorpic => {
            let client = anthorpic_client_from_env()?;
            let req: anthorpic::chat::MessagesRequest = kwargs.into();
            Ok(ChatResponse::Anthorpic(client.chat(req).await?))
        }
        Provider::DeepSeek => {
            let client = deepseek_client_from_env()?;
            let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
            Ok(ChatResponse::OpenAI(client.chat(req).await?))
        }
    }
}

pub async fn chat_text(kwargs: ChatKwargs) -> Result<String, LlmError> {
    let model = kwargs.model.clone();
    let Some(provider) = provider_for_model(&model) else {
        return Err(LlmError::UnknownModel(model));
    };

    match provider {
        Provider::OpenAI => {
            let client = openai_client_from_env()?;
            let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
            Ok(client.chat_text(req).await?)
        }
        Provider::Gemini => {
            let client = gemini_client_from_env()?;
            let req: gemini::chat::GenerateContentRequest = kwargs.into();
            Ok(client.chat_text(&model, req).await?)
        }
        Provider::Anthorpic => {
            let client = anthorpic_client_from_env()?;
            let req: anthorpic::chat::MessagesRequest = kwargs.into();
            Ok(client.chat_text(req).await?)
        }
        Provider::DeepSeek => {
            let client = deepseek_client_from_env()?;
            let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
            Ok(client.chat_text(req).await?)
        }
    }
}

pub async fn chat_structure<T>(kwargs: ChatKwargs) -> Result<T, LlmError>
where
    T: DeserializeOwned + schemars::JsonSchema,
{
    let model = kwargs.model.clone();
    let Some(provider) = provider_for_model(&model) else {
        return Err(LlmError::UnknownModel(model));
    };

    match provider {
        Provider::OpenAI => {
            let client = openai_client_from_env()?;
            let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
            Ok(client.chat_structure(req).await?)
        }
        Provider::Gemini => {
            let client = gemini_client_from_env()?;
            let req: gemini::chat::GenerateContentRequest = kwargs.into();
            Ok(client.chat_structure(&model, req).await?)
        }
        Provider::Anthorpic => {
            let client = anthorpic_client_from_env()?;
            let req: anthorpic::chat::MessagesRequest = kwargs.into();
            Ok(client.chat_structure(req).await?)
        }
        Provider::DeepSeek => {
            let client = deepseek_client_from_env()?;
            let req: openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
            Ok(client.chat_structure(req).await?)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, header, method, path},
    };

    use super::*;

    #[test]
    fn provider_for_model_detects_known_prefixes() {
        assert_eq!(provider_for_model("gpt-5.5"), Some(Provider::OpenAI));
        assert_eq!(
            provider_for_model("gemini-3.5-flash"),
            Some(Provider::Gemini)
        );
        assert_eq!(
            provider_for_model("claude-sonnet-4-5-20250929"),
            Some(Provider::Anthorpic)
        );
        assert_eq!(provider_for_model("unknown-1"), None);
    }

    #[test]
    fn chat_kwargs_try_into_openai_request_maps_tools() {
        let req: crate::llm::openai::types::chat::CreateChatCompletionRequest = ChatKwargs {
            model: "gpt-5.5".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            tools: Some(vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather by city".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
                handler: |args| {
                    serde_json::json!({
                        "city": args["city"].as_str().unwrap(),
                        "temperature": 25.0,
                        "humidity": 60.0,
                    })
                },
            }]),
            ..Default::default()
        }
        .try_into()
        .unwrap();

        let value = serde_json::to_value(req).unwrap();
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            value["tools"][0]["function"]["description"],
            "Get weather by city"
        );
    }

    #[test]
    fn chat_kwargs_into_gemini_request_maps_tools() {
        let req: gemini::chat::GenerateContentRequest = ChatKwargs {
            model: "gemini-3.5-flash".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            tools: Some(vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather by city".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
                handler: |args| {
                    serde_json::json!({
                        "city": args["city"].as_str().unwrap(),
                        "temperature": 25.0,
                        "humidity": 60.0,
                    })
                },
            }]),
            ..Default::default()
        }
        .into();

        let value = serde_json::to_value(req).unwrap();
        assert_eq!(
            value["tools"][0]["functionDeclarations"][0]["name"],
            "get_weather"
        );
        assert_eq!(
            value["tools"][0]["functionDeclarations"][0]["description"],
            "Get weather by city"
        );
    }

    #[test]
    fn chat_kwargs_into_anthorpic_request_maps_tools() {
        let req: anthorpic::chat::MessagesRequest = ChatKwargs {
            model: "claude-sonnet-4-5-20250929".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            tools: Some(vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather by city".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
                handler: |args| {
                    serde_json::json!({
                        "city": args["city"].as_str().unwrap(),
                        "temperature": 25.0,
                        "humidity": 60.0,
                    })
                },
            }]),
            ..Default::default()
        }
        .into();

        let value = serde_json::to_value(req).unwrap();
        assert_eq!(value["tools"][0]["name"], "get_weather");
        assert_eq!(value["tools"][0]["description"], "Get weather by city");
    }

    #[tokio::test]
    async fn chat_text_routes_to_gemini_by_model() {
        let server = MockServer::start().await;

        let _lock = ENV_LOCK.lock().unwrap();
        let prev_key = std::env::var("GEMINI_API_KEY").ok();
        let prev_base = std::env::var("GEMINI_BASE_URL").ok();
        unsafe {
            std::env::set_var("GEMINI_API_KEY", "testkey");
            std::env::set_var("GEMINI_BASE_URL", server.uri());
        }

        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-3.5-flash:generateContent"))
            .and(header("x-goog-api-key", "testkey"))
            .and(body_json(serde_json::json!({
                "contents": [{
                    "role": "user",
                    "parts": [{"text":"hi"}]
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": { "parts": [{"text":"OK"}] }
                }]
            })))
            .mount(&server)
            .await;

        let text = chat_text(ChatKwargs {
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
    async fn chat_text_routes_to_anthorpic_by_model() {
        let server = MockServer::start().await;

        let _lock = ENV_LOCK.lock().unwrap();
        let prev_key = std::env::var("ANTHROPIC_API_KEY").ok();
        let prev_base = std::env::var("ANTHROPIC_BASE_URL").ok();
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "testkey");
            std::env::set_var("ANTHROPIC_BASE_URL", server.uri());
        }

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "testkey"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_json(serde_json::json!({
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 1024,
                "messages": [{"role":"user","content":"hi"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type":"text","text":"hello"}]
            })))
            .mount(&server)
            .await;

        let text = chat_text(ChatKwargs {
            model: "claude-3-5-sonnet-20240620".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            ..Default::default()
        })
        .await
        .unwrap();
        assert_eq!(text, "hello");

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
    async fn chat_text_unknown_model_errors() {
        let err = chat_text(ChatKwargs {
            model: "nope-1".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            ..Default::default()
        })
        .await
        .unwrap_err();

        match err {
            LlmError::UnknownModel(m) => assert_eq!(m, "nope-1"),
            _ => panic!("unexpected error"),
        }
    }

    #[tokio::test]
    async fn test_runtime_chat_with_tool() {
        let _lock = ENV_LOCK.lock().unwrap();
        dotenv::dotenv().ok();

        // Skip if API key is not set
        if std::env::var("GEMINI_API_KEY").is_err() {
            return;
        }

        let req = ChatKwargs {
            model: "gemini-3.5-flash".to_string(),
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

        let resp = chat(req).await.unwrap();

        let mut has_tool_call = false;
        match resp {
            ChatResponse::Gemini(gemini_resp) => {
                if let Some(candidate) = gemini_resp.candidates.first() {
                    if let Some(content) = &candidate.content {
                        if content.parts.iter().any(|p| p.function_call.is_some()) {
                            let part = content
                                .parts
                                .iter()
                                .find(|p| p.function_call.is_some())
                                .unwrap();
                            let fc = part.function_call.as_ref().unwrap();
                            assert_eq!(fc.name, "get_current_weather");
                            assert!(fc.args.get("location").is_some());
                            has_tool_call = true;
                        }
                    }
                }
            }
            _ => panic!("Expected Gemini response"),
        }

        assert!(has_tool_call, "Expected a tool call but got none");
    }

    #[tokio::test]
    async fn test_runtime_chat_with_tool_openai() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        dotenv::dotenv().ok();

        // Skip if API key is not set
        if std::env::var("OPENAI_API_KEY").is_err() {
            return;
        }

        let req = ChatKwargs {
            model: "gpt-4o-mini".to_string(),
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

        let resp = chat(req).await.unwrap();

        let mut has_tool_call = false;
        match resp {
            ChatResponse::OpenAI(openai_resp) => {
                if let Some(choice) = openai_resp.choices.first() {
                    if let Some(tool_calls) = &choice.message.tool_calls {
                        if let Some(crate::llm::openai::types::chat::ChatCompletionMessageToolCall { function, .. }) = tool_calls.first().cloned().and_then(|t| match t {
                            crate::llm::openai::types::chat::ChatCompletionMessageToolCalls::Function(f) => Some(f),
                            _ => None,
                        }) {
                            assert_eq!(function.name, "get_current_weather");
                            let args: serde_json::Value = serde_json::from_str(&function.arguments).unwrap();
                            assert!(args.get("location").is_some());
                            has_tool_call = true;
                        }
                    }
                }
            }
            _ => panic!("Expected OpenAI response"),
        }

        assert!(has_tool_call, "Expected a tool call but got none");
    }

    #[tokio::test]
    async fn test_runtime_chat_with_tool_anthropic() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        dotenv::dotenv().ok();

        // Skip if API key is not set
        if std::env::var("ANTHROPIC_API_KEY").is_err() {
            return;
        }

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

        let resp = chat(req).await.unwrap();

        let mut has_tool_call = false;
        match resp {
            ChatResponse::Anthorpic(anthropic_resp) => {
                for block in &anthropic_resp.content {
                    if let anthorpic::chat::ContentBlock::ToolUse { name, input, .. } = block {
                        assert_eq!(name, "get_current_weather");
                        assert!(input.get("location").is_some());
                        has_tool_call = true;
                    }
                }
            }
            _ => panic!("Expected Anthropic response"),
        }

        assert!(has_tool_call, "Expected a tool call but got none");
    }

    #[tokio::test]
    async fn test_runtime_chat_with_tool_deepseek() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        dotenv::dotenv().ok();

        // Skip if API key is not set
        if std::env::var("DEEPSEEK_API_KEY").is_err() {
            return;
        }

        let req = ChatKwargs {
            model: "gpt-4o-mini".to_string(),
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

        let resp = chat(req).await.unwrap();

        let mut has_tool_call = false;
        match resp {
            ChatResponse::OpenAI(openai_resp) => {
                if let Some(choice) = openai_resp.choices.first() {
                    if let Some(tool_calls) = &choice.message.tool_calls {
                        if let Some(crate::llm::openai::types::chat::ChatCompletionMessageToolCall { function, .. }) = tool_calls.first().cloned().and_then(|t| match t {
                            crate::llm::openai::types::chat::ChatCompletionMessageToolCalls::Function(f) => Some(f),
                            _ => None,
                        }) {
                            assert_eq!(function.name, "get_current_weather");
                            let args: serde_json::Value = serde_json::from_str(&function.arguments).unwrap();
                            assert!(args.get("location").is_some());
                            has_tool_call = true;
                        }
                    }
                }
            }
            _ => panic!("Expected OpenAI response"),
        }

        assert!(has_tool_call, "Expected a tool call but got none");
    }
}
