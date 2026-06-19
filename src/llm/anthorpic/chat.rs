use std::{error::Error, fmt};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub use super::client::Client;

#[derive(Debug)]
pub enum ClaudeError {
    Http(reqwest::Error),
    InvalidHeaderValue,
    UnexpectedStatus {
        status: reqwest::StatusCode,
        body: String,
    },
    Json(serde_json::Error),
    Utf8(std::string::FromUtf8Error),
    MissingText,
}

impl fmt::Display for ClaudeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClaudeError::Http(e) => write!(f, "http error: {e}"),
            ClaudeError::InvalidHeaderValue => write!(f, "invalid header value"),
            ClaudeError::UnexpectedStatus { status, body } => {
                write!(f, "unexpected status {status}: {body}")
            }
            ClaudeError::Json(e) => write!(f, "json error: {e}"),
            ClaudeError::Utf8(e) => write!(f, "utf8 error: {e}"),
            ClaudeError::MissingText => write!(f, "missing text in response"),
        }
    }
}

impl Error for ClaudeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ClaudeError::Http(e) => Some(e),
            ClaudeError::Json(e) => Some(e),
            ClaudeError::Utf8(e) => Some(e),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for ClaudeError {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

impl From<serde_json::Error> for ClaudeError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<std::string::FromUtf8Error> for ClaudeError {
    fn from(value: std::string::FromUtf8Error) -> Self {
        Self::Utf8(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum MediaSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
    #[serde(rename = "url")]
    Url { url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: MediaSource },
    #[serde(rename = "document")]
    Document { source: MediaSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    pub model: Option<String>,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub role: Option<String>,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub stop_details: Option<serde_json::Value>,
    pub usage: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
struct OutputConfig {
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Serialize)]
struct OutputFormat {
    #[serde(rename = "type")]
    pub kind: OutputFormatKind,
    pub schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum OutputFormatKind {
    JsonSchema,
}

#[derive(Debug, Clone, Serialize)]
struct StructuredMessagesRequest {
    #[serde(flatten)]
    pub base: MessagesRequest,
    pub output_config: OutputConfig,
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
    pub async fn chat<R: Into<MessagesRequest>>(
        &self,
        req: R,
    ) -> Result<MessagesResponse, ClaudeError> {
        let req: MessagesRequest = req.into();
        let url = self.messages_url();
        let headers = self.build_headers()?;

        let resp: reqwest::Response = self
            .http
            .post(url)
            .headers(headers)
            .json(&MessagesRequest {
                stream: None,
                ..req
            })
            .send()
            .await?;

        let status: reqwest::StatusCode = resp.status();
        if !status.is_success() {
            let body: String = resp.text().await?;
            return Err(ClaudeError::UnexpectedStatus { status, body });
        }

        let json_resp: MessagesResponse = resp.json().await?;

        Ok(json_resp)
    }

    pub async fn chat_text<R: Into<MessagesRequest>>(&self, req: R) -> Result<String, ClaudeError> {
        let parsed = self.chat(req).await?;
        let text: String = parsed
            .content
            .into_iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text),
                _ => None,
            })
            .collect();

        if text.is_empty() {
            return Err(ClaudeError::MissingText);
        }

        Ok(text)
    }

    pub async fn chat_structure<T, R: Into<MessagesRequest>>(
        &self,
        req: R,
    ) -> Result<T, ClaudeError>
    where
        T: DeserializeOwned + schemars::JsonSchema,
    {
        let req: MessagesRequest = req.into();
        let url = self.messages_url();
        let headers = self.build_headers()?;

        let root_schema = schemars::schema_for!(T);
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

        let body: StructuredMessagesRequest = StructuredMessagesRequest {
            base: MessagesRequest {
                stream: None,
                ..req
            },
            output_config: OutputConfig {
                format: OutputFormat {
                    kind: OutputFormatKind::JsonSchema,
                    schema,
                },
            },
        };

        let resp: reqwest::Response = self
            .http
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status: reqwest::StatusCode = resp.status();
        if !status.is_success() {
            let body: String = resp.text().await?;
            return Err(ClaudeError::UnexpectedStatus { status, body });
        }

        let json_resp: MessagesResponse = resp.json().await?;
        let text: String = json_resp
            .content
            .into_iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text),
                _ => None,
            })
            .collect();

        if text.is_empty() {
            return Err(ClaudeError::MissingText);
        }

        Ok(serde_json::from_str::<T>(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, header, method, path},
    };

    struct ReqWrap(MessagesRequest);

    impl From<ReqWrap> for MessagesRequest {
        fn from(value: ReqWrap) -> Self {
            value.0
        }
    }

    #[test]
    fn request_supports_image_url_blocks() {
        let req = MessagesRequest {
            model: "claude-opus-4-8".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Blocks(vec![
                    ContentBlock::Image {
                        source: MediaSource::Url {
                            url: "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg".to_string(),
                        },
                    },
                    ContentBlock::Text {
                        text: "Describe this image.".to_string(),
                    },
                ]),
            }],
            ..Default::default()
        };

        let v = serde_json::to_value(req).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this image."
                        }
                    ]
                }]
            })
        );
    }

    #[test]
    fn request_supports_document_url_blocks() {
        let req = MessagesRequest {
            model: "claude-opus-4-8".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Blocks(vec![
                    ContentBlock::Document {
                        source: MediaSource::Url {
                            url: "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf".to_string(),
                        },
                    },
                    ContentBlock::Text {
                        text: "What are the key findings in this document?".to_string(),
                    },
                ]),
            }],
            ..Default::default()
        };

        let v = serde_json::to_value(req).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "url",
                                "url": "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf"
                            }
                        },
                        {
                            "type": "text",
                            "text": "What are the key findings in this document?"
                        }
                    ]
                }]
            })
        );
    }

    #[tokio::test]
    async fn chat_text_sends_required_headers_and_body() {
        let server = MockServer::start().await;

        let req = MessagesRequest {
            model: "claude-3-5-sonnet-20240620".to_string(),
            max_tokens: 16,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
            }],
            system: None,
            stream: None,
            tools: None,
            tool_choice: None,
        };

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "testkey"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_json(serde_json::json!({
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 16,
                "messages": [{"role":"user","content":"hi"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type":"text","text":"hello"}]
            })))
            .mount(&server)
            .await;

        let client = Client::new("testkey").with_base_url(server.uri());
        let text = client.chat_text(ReqWrap(req)).await.unwrap();
        assert_eq!(text, "hello");
    }

    #[tokio::test]
    async fn chat_supports_tool_use_blocks() {
        let server = MockServer::start().await;

        let req = MessagesRequest {
            model: "claude-3-5-sonnet-20240620".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("what is weather".to_string()),
            }],
            system: None,
            stream: None,
            tools: Some(vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: Some("get weather by city".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }),
            }]),
            tool_choice: Some(ToolChoice::Auto),
        };

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "testkey"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_json(serde_json::json!({
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 1024,
                "messages": [{"role":"user","content":"what is weather"}],
                "tools": [{
                    "name": "get_weather",
                    "description": "get weather by city",
                    "input_schema": {
                        "type":"object",
                        "properties":{"city":{"type":"string"}},
                        "required":["city"]
                    }
                }],
                "tool_choice": {"type":"auto"}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{
                    "type":"tool_use",
                    "id":"toolu_1",
                    "name":"get_weather",
                    "input":{"city":"Shanghai"}
                }]
            })))
            .mount(&server)
            .await;

        let client = Client::new("testkey").with_base_url(server.uri());
        let resp = client.chat(req).await.unwrap();
        assert_eq!(
            resp.content,
            vec![ContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({"city":"Shanghai"}),
            }]
        );
    }

    #[tokio::test]
    async fn runtime_chat() {
        use dotenv::dotenv;
        let _lock = crate::llm::chat::ENV_LOCK.lock().unwrap();
        dotenv().ok();

        let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
            return;
        };

        let messages = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("what is weather in Shanghai".to_string()),
        }];

        let req = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: messages.clone(),
            system: None,
            stream: None,
            tools: Some(vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: Some("get weather by city".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }),
            }]),
            tool_choice: Some(ToolChoice::Auto),
        };

        let client: Client = Client::new(api_key);
        let resp_result: Result<MessagesResponse, ClaudeError> = client.chat(req.clone()).await;
        let mut new_message  = messages.clone();
        let mut tool_results: Vec<ContentBlock> = vec![];
        match resp_result {
            Ok(resp) => {
                println!("{:?}", resp);
                let resp_content = &resp.content[0];
                
                let mut msg_content: Vec<ContentBlock> = vec![];

                match resp_content {
                    ContentBlock::Text { text } => {
                        msg_content.push(ContentBlock::Text { text: text.clone() });
                    },
                    ContentBlock::ToolUse { id, name, input } => {
                        println!("{:?}", (id, name, input));
                        msg_content.push(ContentBlock::ToolUse { id: id.clone(), name: name.clone(), input: input.clone() });
                        tool_results.push(ContentBlock::ToolResult { tool_use_id: id.clone(), content: serde_json::json!({"temperature": 25.0}).to_string() });
                    },
                    _ => println!("Other content block"),
                }

                new_message.push(Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(msg_content),
                });
            }
            Err(err) => {
                println!("{:?}", err);
            }
        };

        new_message.push(Message {
            role: "user".to_string(),
            content: MessageContent::Blocks(tool_results),
        });


        // let tool_resp = resp.content[0];

        let second_req = MessagesRequest {
            messages: new_message,
            ..req.clone()
        };

        match client.chat(second_req.clone()).await {
            Ok(resp) => {
                println!("{:?}", resp);
            }
            Err(err) => {
                println!("{:?}", err);
            }
        };
        
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug)]
    struct UserInfo {
        name: String,
        age: u32,
        is_student: bool,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq, Eq)]
    struct EmailInfo {
        name: String,
        email: String,
        plan_interest: String,
        demo_requested: bool,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq, Eq)]
    struct ContactInfo {
        email: String,
        phone: Option<String>,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq, Eq)]
    struct AddressInfo {
        city: String,
        country: String,
        zip: String,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq, Eq)]
    struct UserProfile {
        name: String,
        age: u32,
        is_student: bool,
        contact: ContactInfo,
        address: AddressInfo,
        tags: Vec<String>,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq, Eq)]
    enum Plan {
        Free,
        Pro,
        Enterprise,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq, Eq)]
    struct LineItem {
        sku: String,
        quantity: u32,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq, Eq)]
    struct OrderInfo {
        order_id: String,
        plan: Plan,
        expedited: bool,
        items: Vec<LineItem>,
    }

    fn try_build_client() -> Option<Client> {
        use dotenv::dotenv;
        let _lock = crate::llm::chat::ENV_LOCK.lock().unwrap();
        dotenv().ok();

        let api_key = std::env::var("ANTHROPIC_API_KEY").ok()?;
        Some(Client::new(api_key))
    }

    #[tokio::test]
    async fn test_chat_structure_user_info() {
        let Some(client) = try_build_client() else {
            return;
        };

        let req = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text(
                    "Extract structured fields. Alice is 20 years old and is not a student."
                        .to_string(),
                ),
            }],
            ..Default::default()
        };

        let resp: UserInfo = client.chat_structure(ReqWrap(req)).await.unwrap();
        assert!(resp.name.to_lowercase().contains("alice"));
        assert_eq!(resp.age, 20);
        assert!(!resp.is_student);
    }

    #[tokio::test]
    async fn test_chat_structure_email_info() {
        let Some(client) = try_build_client() else {
            return;
        };

        let req = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text(
                    "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
                        .to_string(),
                ),
            }],
            ..Default::default()
        };

        let resp: EmailInfo = client.chat_structure(ReqWrap(req)).await.unwrap();
        assert!(resp.name.to_lowercase().contains("john"));
        assert_eq!(resp.email, "john@example.com");
        assert!(resp.plan_interest.to_lowercase().contains("enterprise"));
        assert!(resp.demo_requested);
    }

    #[tokio::test]
    async fn test_chat_structure_nested_profile() {
        let Some(client) = try_build_client() else {
            return;
        };

        let req = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text(
                    "Extract structured fields from this profile: Name: Bob Lee. Age: 33. Student: false. Email: bob@example.com. City: Shanghai. Country: China. Zip: 200000. Tags: rust, graphql, tokio. No phone number provided."
                        .to_string(),
                ),
            }],
            ..Default::default()
        };

        let resp: UserProfile = client.chat_structure(ReqWrap(req)).await.unwrap();
        assert!(resp.name.to_lowercase().contains("bob"));
        assert_eq!(resp.age, 33);
        assert!(!resp.is_student);
        assert_eq!(resp.contact.email, "bob@example.com");
        assert_eq!(resp.contact.phone, None);
        assert_eq!(resp.address.city, "Shanghai");
        assert_eq!(resp.address.country, "China");
        assert_eq!(resp.address.zip, "200000");
        assert!(resp.tags.iter().any(|t| t == "rust"));
        assert!(resp.tags.iter().any(|t| t == "graphql"));
        assert!(resp.tags.iter().any(|t| t == "tokio"));
    }

    #[tokio::test]
    async fn test_chat_structure_enum_and_array() {
        let Some(client) = try_build_client() else {
            return;
        };

        let req = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text(
                    "Extract order info: order id is ORD-100. The plan is Enterprise. Expedited shipping is true. Items: SKU A1 quantity 2; SKU B2 quantity 5."
                        .to_string(),
                ),
            }],
            ..Default::default()
        };

        let resp: OrderInfo = client.chat_structure(ReqWrap(req)).await.unwrap();
        assert_eq!(resp.order_id, "ORD-100");
        assert_eq!(resp.plan, Plan::Enterprise);
        assert!(resp.expedited);
        assert!(resp.items.iter().any(|i| i.sku == "A1" && i.quantity == 2));
        assert!(resp.items.iter().any(|i| i.sku == "B2" && i.quantity == 5));
    }
}
