use crate::llm::{
    anthorpic, gemini,
    model_list::{is_anthorpic_model, is_gemini_model, is_openai_model},
    openai,
};

#[derive(Debug, Clone)]
pub struct ChatTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    pub handler: fn(serde_json::Value) -> serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ChatKwargs {
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<ChatTool>>,
}

impl Default for ChatKwargs {
    fn default() -> Self {
        Self {
            model: "gpt-5.5".to_string(),
            temperature: None,
            max_tokens: 1024,
            messages: Vec::new(),
            tools: None,
        }
    }
}

#[derive(Debug)]
pub enum LlmError {
    OpenAI(openai::chat::OpenAIError),
    Gemini(gemini::chat::GeminiError),
    Anthorpic(anthorpic::chat::ClaudeError),
    UnknownModel(String),
    MissingEnv { key: &'static str },
}

impl From<openai::chat::OpenAIError> for LlmError {
    fn from(value: openai::chat::OpenAIError) -> Self {
        Self::OpenAI(value)
    }
}

impl From<gemini::chat::GeminiError> for LlmError {
    fn from(value: gemini::chat::GeminiError) -> Self {
        Self::Gemini(value)
    }
}

impl From<anthorpic::chat::ClaudeError> for LlmError {
    fn from(value: anthorpic::chat::ClaudeError) -> Self {
        Self::Anthorpic(value)
    }
}

#[derive(Debug)]
pub enum ChatResponse {
    OpenAI(openai::types::chat::CreateChatCompletionResponse),
    Gemini(gemini::chat::GenerateContentResponse),
    Anthorpic(anthorpic::chat::MessagesResponse),
}

impl ChatResponse {
    pub fn text(&self) -> String {
        match self {
            ChatResponse::OpenAI(resp) => resp
                .choices
                .iter()
                .filter_map(|c| c.message.content.as_deref())
                .collect::<Vec<_>>()
                .concat(),
            ChatResponse::Gemini(resp) => resp.text(),
            ChatResponse::Anthorpic(resp) => resp
                .content
                .iter()
                .filter_map(|b| match b {
                    anthorpic::chat::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .concat(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
    Gemini,
    Anthorpic,
    DeepSeek,
}

pub fn provider_for_model(model: &str) -> Option<Provider> {
    if is_openai_model(model) {
        return Some(Provider::OpenAI);
    }
    if is_gemini_model(model) {
        return Some(Provider::Gemini);
    }
    if is_anthorpic_model(model) {
        return Some(Provider::Anthorpic);
    }
    None
}

fn gemini_role(role: &str) -> Option<String> {
    match role {
        "user" => Some("user".to_string()),
        "assistant" => Some("model".to_string()),
        _ => None,
    }
}

impl From<ChatKwargs> for gemini::chat::GenerateContentRequest {
    fn from(value: ChatKwargs) -> Self {
        let tools = value.tools.map(|tools| {
            vec![gemini::chat::Tool {
                function_declarations: tools
                    .into_iter()
                    .map(|tool| gemini::chat::FunctionDeclaration {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.parameters,
                    })
                    .collect(),
            }]
        });

        gemini::chat::GenerateContentRequest {
            contents: value
                .messages
                .into_iter()
                .map(|m| gemini::chat::Content {
                    // role: gemini_role(&m.role),
                    role: (|role: &str| -> Option<String> {
                        match role {
                            "user" => Some("user".to_string()),
                            "assistant" => Some("model".to_string()),
                            _ => None,
                        }
                    })(&m.role),
                    parts: vec![gemini::chat::Part {
                        text: Some(m.content),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                    }],
                })
                .collect(),
            tools,
            generation_config: None,
        }
    }
}

impl From<ChatKwargs> for anthorpic::chat::MessagesRequest {
    fn from(value: ChatKwargs) -> Self {
        let mut system_buf: Vec<String> = Vec::new();
        let mut messages: Vec<anthorpic::chat::Message> = Vec::new();
        let tools = value.tools.map(|tools| {
            tools
                .into_iter()
                .map(|tool| anthorpic::chat::ToolDefinition {
                    name: tool.name,
                    description: tool.description,
                    input_schema: tool.parameters,
                })
                .collect()
        });

        for m in value.messages {
            if m.role == "system" {
                system_buf.push(m.content);
                continue;
            }
            messages.push(anthorpic::chat::Message {
                role: m.role,
                content: anthorpic::chat::MessageContent::Text(m.content),
            });
        }

        anthorpic::chat::MessagesRequest {
            model: value.model,
            max_tokens: value.max_tokens,
            messages,
            system: if system_buf.is_empty() {
                None
            } else {
                Some(system_buf.join("\n"))
            },
            stream: None,
            tools,
            tool_choice: None,
        }
    }
}

impl TryFrom<ChatKwargs> for openai::types::chat::CreateChatCompletionRequest {
    type Error = openai::chat::OpenAIError;

    fn try_from(value: ChatKwargs) -> Result<Self, Self::Error> {
        let mut msgs: Vec<openai::types::chat::ChatCompletionRequestMessage> = Vec::new();
        for m in value.messages {
            match m.role.as_str() {
                "system" => {
                    msgs.push(openai::types::chat::ChatCompletionRequestMessage::System(
                        openai::types::chat::ChatCompletionRequestSystemMessageArgs::default()
                            .content(openai::types::chat::ChatCompletionRequestSystemMessageContent::Text(m.content))
                            .build()
                            .map_err(openai::chat::OpenAIError::from)?,
                    ));
                }
                "assistant" => {
                    msgs.push(openai::types::chat::ChatCompletionRequestMessage::Assistant(
                        openai::types::chat::ChatCompletionRequestAssistantMessageArgs::default()
                            .content(openai::types::chat::ChatCompletionRequestAssistantMessageContent::Text(m.content))
                            .build()
                            .map_err(openai::chat::OpenAIError::from)?,
                    ));
                }
                _ => {
                    msgs.push(openai::types::chat::ChatCompletionRequestMessage::User(
                        openai::types::chat::ChatCompletionRequestUserMessageArgs::default()
                            .content(
                                openai::types::chat::ChatCompletionRequestUserMessageContent::Text(
                                    m.content,
                                ),
                            )
                            .build()
                            .map_err(openai::chat::OpenAIError::from)?,
                    ));
                }
            }
        }

        let mut builder = openai::types::chat::CreateChatCompletionRequestArgs::default();
        builder.model(value.model).messages(msgs);
        if let Some(t) = value.temperature {
            builder.temperature(t as f32);
        }
        if let Some(tools) = value.tools {
            builder.tools(
                tools
                    .into_iter()
                    .map(|tool| {
                        openai::types::chat::ChatCompletionTools::Function(
                            openai::types::chat::ChatCompletionTool {
                                function: openai::types::chat::FunctionObject {
                                    name: tool.name,
                                    description: tool.description,
                                    parameters: Some(tool.parameters),
                                    strict: Some(true),
                                },
                            },
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }
        Ok(builder.build().map_err(openai::chat::OpenAIError::from)?)
    }
}
