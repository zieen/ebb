use async_openai::types::chat::{ChatCompletionResponseStream, CreateChatCompletionRequest};
use futures::StreamExt;

use super::{chat::OpenAIError, client::Client};

pub trait StreamEventExt {
    fn text_delta(&self) -> Option<&str>;
}

impl StreamEventExt for async_openai::types::chat::CreateChatCompletionStreamResponse {
    fn text_delta(&self) -> Option<&str> {
        self.choices
            .iter()
            .filter_map(|c| c.delta.content.as_deref())
            .next()
    }

}

impl Client {
    pub async fn stream_events<T: Into<CreateChatCompletionRequest>>(
        &self,
        req: T,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        Ok(self.inner.chat().create_stream(req.into()).await?)
    }

    pub async fn stream_text<T: Into<CreateChatCompletionRequest>>(
        &self,
        req: T,
    ) -> Result<String, OpenAIError> {
        let mut stream = self.stream_events(req).await?;
        let mut out = String::new();

        while let Some(chunk) = stream.next().await.transpose()? {
            if let Some(delta) = chunk.text_delta() {
                out.push_str(delta);
            }
        }

        if out.is_empty() {
            return Err(OpenAIError::MissingText);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::FunctionObject;
    use async_openai::types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestUserMessageContent, ChatCompletionTools,
        CreateChatCompletionRequestArgs,
    };
    use dotenv::dotenv;

    use super::*;

    #[tokio::test]
    async fn runtime_stream_text() {
        dotenv().ok();
        if std::env::var("OPENAI_API_KEY").is_err() {
            return;
        }

        let client = Client::new();
        let message = ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Text(
                    "Reply with the single word OK.".to_string(),
                ))
                .build()
                .unwrap(),
        );

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-5.5")
            .messages(vec![message])
            .build()
            .unwrap();

        struct ReqWrap(CreateChatCompletionRequest);
        impl From<ReqWrap> for CreateChatCompletionRequest {
            fn from(value: ReqWrap) -> Self {
                value.0
            }
        }

        let text = client.stream_text(ReqWrap(req)).await.unwrap();
        assert!(text.to_lowercase().contains("ok"));
    }

    #[tokio::test]
    async fn runtime_stream_tool_use() {
        dotenv().ok();
        if std::env::var("OPENAI_API_KEY").is_err() {
            return;
        }

        let client = Client::new();
        let message = ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Text(
                    "Use get_weather with city = Shanghai.".to_string(),
                ))
                .build()
                .unwrap(),
        );

        let tools: Vec<ChatCompletionTools> = vec![ChatCompletionTools::Function(
            async_openai::types::chat::ChatCompletionTool {
                function: FunctionObject {
                    name: "get_weather".to_string(),
                    description: Some("Get weather for a city".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"],
                        "additionalProperties": false
                    })),
                    strict: Some(true),
                },
            },
        )];

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-5.5")
            .messages(vec![message])
            .tools(tools)
            .build()
            .unwrap();

        struct ReqWrap(CreateChatCompletionRequest);
        impl From<ReqWrap> for CreateChatCompletionRequest {
            fn from(value: ReqWrap) -> Self {
                value.0
            }
        }

        let mut stream = client.stream_events(ReqWrap(req)).await.unwrap();

        let mut saw_tool_call = false;
        let mut args_buf = String::new();

        while let Some(chunk) = stream.next().await.transpose().unwrap() {
            for choice in &chunk.choices {
                if let Some(tool_calls) = &choice.delta.tool_calls {
                    for tc in tool_calls {
                        if let Some(function) = &tc.function {
                            if function.name.as_deref() == Some("get_weather") {
                                saw_tool_call = true;
                            }
                            if let Some(arguments) = &function.arguments {
                                args_buf.push_str(arguments);
                            }
                        }
                    }
                }
            }
        }

        assert!(saw_tool_call);
        if !args_buf.is_empty() {
            let _ = serde_json::from_str::<serde_json::Value>(&args_buf).ok();
        }
    }
}
