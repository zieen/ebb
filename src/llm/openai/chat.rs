use std::{error::Error, fmt};

use async_openai::{
    error::OpenAIError as AsyncOpenAIError,
    types::chat::{
        CreateChatCompletionRequest, CreateChatCompletionResponse, ResponseFormat,
        ResponseFormatJsonSchema,
    },
};
use serde::de::DeserializeOwned;

pub use super::client::Client;

#[derive(Debug)]
pub enum OpenAIError {
    OpenAI(AsyncOpenAIError),
    Json(serde_json::Error),
    MissingText,
}

impl fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenAIError::OpenAI(e) => write!(f, "{e}"),
            OpenAIError::Json(e) => write!(f, "json error: {e}"),
            OpenAIError::MissingText => write!(f, "missing text in response"),
        }
    }
}

impl Error for OpenAIError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OpenAIError::OpenAI(e) => Some(e),
            OpenAIError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<AsyncOpenAIError> for OpenAIError {
    fn from(value: AsyncOpenAIError) -> Self {
        Self::OpenAI(value)
    }
}

impl From<serde_json::Error> for OpenAIError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

fn add_additional_properties_false(schema: &mut serde_json::Value) {
    let obj = match schema.as_object_mut() {
        Some(v) => v,
        None => return,
    };

    if obj
        .get("type")
        .and_then(|v| v.as_str())
        .is_some_and(|t| t == "object")
        && !obj.contains_key("additionalProperties")
    {
        obj.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(false),
        );
    }
}

fn sanitize_schema(schema: &mut serde_json::Value) {
    let obj = match schema.as_object_mut() {
        Some(v) => v,
        None => return,
    };

    for k in [
        "$schema",
        "title",
        "description",
        "examples",
        "default",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ] {
        obj.remove(k);
    }

    if obj
        .get("type")
        .and_then(|v| v.as_str())
        .is_some_and(|t| t == "object")
        && !obj.contains_key("additionalProperties")
    {
        obj.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(false),
        );
    }

    if let Some(props) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
        for (_, v) in props.iter_mut() {
            sanitize_schema(v);
        }
    }

    for defs_key in ["definitions", "$defs"] {
        if let Some(defs) = obj.get_mut(defs_key).and_then(|v| v.as_object_mut()) {
            for (_, v) in defs.iter_mut() {
                sanitize_schema(v);
            }
        }
    }

    if let Some(items) = obj.get_mut("items") {
        sanitize_schema(items);
    }

    for k in ["anyOf", "oneOf", "allOf"] {
        if let Some(arr) = obj.get_mut(k).and_then(|v| v.as_array_mut()) {
            for v in arr.iter_mut() {
                sanitize_schema(v);
            }
        }
    }

    if let Some(ap) = obj.get_mut("additionalProperties") {
        if ap.is_object() {
            sanitize_schema(ap);
        }
    }
}

impl Client {
    pub async fn chat<T: Into<CreateChatCompletionRequest>>(
        &self,
        req: T,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        Ok(self.inner.chat().create(req.into()).await?)
    }

    pub async fn chat_text<T: Into<CreateChatCompletionRequest>>(
        &self,
        req: T,
    ) -> Result<String, OpenAIError> {
        let resp = self.chat(req).await?;
        let text = resp
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        if text.is_empty() {
            return Err(OpenAIError::MissingText);
        }

        Ok(text)
    }

    pub async fn chat_structure<S, T: Into<CreateChatCompletionRequest>>(
        &self,
        req: T,
    ) -> Result<S, OpenAIError>
    where
        S: DeserializeOwned + schemars::JsonSchema,
    {
        let root_schema = schemars::schema_for!(S);
        let mut schema = serde_json::to_value(&root_schema.schema)?;
        if let (serde_json::Value::Object(obj), false) =
            (&mut schema, root_schema.definitions.is_empty())
        {
            obj.insert(
                "definitions".to_string(),
                serde_json::to_value(&root_schema.definitions)?,
            );
        }
        add_additional_properties_false(&mut schema);
        sanitize_schema(&mut schema);

        let resp = self
            .inner
            .chat()
            .create(CreateChatCompletionRequest {
                stream: None,
                response_format: Some(ResponseFormat::JsonSchema {
                    json_schema: ResponseFormatJsonSchema {
                        description: None,
                        name: "structured_output".to_string(),
                        schema,
                        strict: Some(true),
                    },
                }),
                ..req.into()
            })
            .await?;

        let text = resp
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        if text.is_empty() {
            return Err(OpenAIError::MissingText);
        }

        Ok(serde_json::from_str::<S>(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::{
        ChatCompletionNamedToolChoice, ChatCompletionRequestMessage,
        ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, ChatCompletionTool,
        ChatCompletionToolChoiceOption, ChatCompletionTools, CreateChatCompletionRequestArgs,
        FunctionName, FunctionObject, ImageUrl,
    };
    use dotenv::dotenv;

    use crate::llm::model_list::model_names;

use super::*;

    #[test]
    fn request_supports_image_understanding_url() {
        let message = ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: "what is in this image?".to_string(),
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: "https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg".to_string(),
                                detail: None,
                            },
                        },
                    ),
                ]))
                .build()
                .unwrap(),
        );

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o-mini")
            .messages(vec![message])
            .build()
            .unwrap();

        let v = serde_json::to_value(req).unwrap();
        assert_eq!(
            v["messages"][0]["content"][0]["type"].as_str(),
            Some("text")
        );
        assert_eq!(
            v["messages"][0]["content"][1]["type"].as_str(),
            Some("image_url")
        );
        assert_eq!(
            v["messages"][0]["content"][1]["image_url"]["url"].as_str(),
            Some(
                "https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg"
            )
        );
    }

    #[test]
    fn request_supports_image_understanding_base64() {
        let base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax3fQAAAABJRU5ErkJggg==";

        let message = ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: "what's in this image?".to_string(),
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: format!("data:image/png;base64,{base64_image}"),
                                detail: None,
                            },
                        },
                    ),
                ]))
                .build()
                .unwrap(),
        );

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o-mini")
            .messages(vec![message])
            .build()
            .unwrap();

        let v = serde_json::to_value(req).unwrap();
        assert_eq!(
            v["messages"][0]["content"][1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,"),
            true
        );
    }

    #[tokio::test]
    async fn runtime_chat_text() {
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

        let text = client.chat_text(req).await.unwrap();
        assert!(text.to_lowercase().contains("ok"));
    }

    #[tokio::test]
    async fn runtime_chat_image_understanding_url() {
        dotenv().ok();
        if std::env::var("OPENAI_API_KEY").is_err() {
            return;
        }

        let client = Client::new();
        let message = ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: "Describe what is in this image in one sentence.".to_string(),
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: "https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg".to_string(),
                                detail: None,
                            },
                        },
                    ),
                ]))
                .build()
                .unwrap(),
        );

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o-mini")
            .messages(vec![message])
            .build()
            .unwrap();

        let text = client.chat_text(req).await.unwrap();
        assert!(!text.trim().is_empty());
    }

    #[tokio::test]
    async fn runtime_chat_tool_use() {
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

        let tools = vec![ChatCompletionTools::Function(ChatCompletionTool {
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
        })];

        let tool_choice = ChatCompletionToolChoiceOption::Function(ChatCompletionNamedToolChoice {
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        });

        let req = CreateChatCompletionRequestArgs::default()
            .model(model_names::CHATGPT_5_5.to_string())
            .messages(vec![message])
            .tools(tools)
            .tool_choice(tool_choice)
            .build()
            .unwrap();

        let resp = client.chat(req).await.unwrap();
        let calls = resp
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.tool_calls)
            .unwrap_or_default();

        let mut saw = false;
        for call in calls {
            if let async_openai::types::chat::ChatCompletionMessageToolCalls::Function(f) = call {
                if f.function.name == "get_weather" {
                    saw = true;
                    let args: serde_json::Value = serde_json::from_str(&f.function.arguments)
                        .unwrap_or(serde_json::Value::Null);
                    if let Some(city) = args.get("city").and_then(|v| v.as_str()) {
                        assert_eq!(city, "Shanghai");
                    }
                }
            }
        }

        assert!(saw);
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq, Eq)]
    struct UserInfo {
        name: String,
        age: u32,
        is_student: bool,
    }

    #[test]
    fn request_serializes_json_schema_response_format() {
        use async_openai::types::chat::{ResponseFormat, ResponseFormatJsonSchema};

        let message = ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Text(
                    "Extract fields".to_string(),
                ))
                .build()
                .unwrap(),
        );

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-5.5")
            .messages(vec![message])
            .response_format(ResponseFormat::JsonSchema {
                json_schema: ResponseFormatJsonSchema {
                    description: None,
                    name: "structured_output".to_string(),
                    schema: serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
                    strict: Some(true),
                },
            })
            .build()
            .unwrap();

        let v = serde_json::to_value(req).unwrap();
        assert_eq!(v["response_format"]["type"].as_str(), Some("json_schema"));
        assert_eq!(
            v["response_format"]["json_schema"]["name"].as_str(),
            Some("structured_output")
        );
    }

    #[tokio::test]
    async fn runtime_chat_structure() {
        dotenv().ok();
        if std::env::var("OPENAI_API_KEY").is_err() {
            return;
        }

        let client = Client::new();
        let message = ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Text(
                    "Return JSON for a user: name=Alice age=30 is_student=false".to_string(),
                ))
                .build()
                .unwrap(),
        );
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-5.5")
            .messages(vec![message])
            .build()
            .unwrap();

        let parsed: UserInfo = client.chat_structure(req).await.unwrap();
        assert_eq!(
            parsed,
            UserInfo {
                name: "Alice".to_string(),
                age: 30,
                is_student: false,
            }
        );
    }
}
