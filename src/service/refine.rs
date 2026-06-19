use schemars::JsonSchema;
use serde::Deserialize;

use crate::llm::chat::chat_structure;
use crate::llm::types::{ChatKwargs, LlmError, Message};
use crate::service::prompts::SYSTEM_PROMPT;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RefinedResult {
    pub refined_text: String,
    pub mistakes: Vec<String>,
}

fn refine_request(text: &str) -> ChatKwargs {
    ChatKwargs {
        model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "gemini-3.5-flash".to_string()),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: format!("Refine the following text and list any mistakes:\n\n{text}"),
            },
        ],
        ..Default::default()
    }
}

pub async fn refine_text(text: &str) -> Result<RefinedResult, LlmError> {
    chat_structure(refine_request(text)).await
}

#[cfg(test)]
mod tests {
    use super::refine_request;

    #[test]
    fn refine_request_builds_system_and_user_messages() {
        let request = refine_request("i has a apple");

        assert_eq!(request.model, "gemini-3.5-flash");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, "system");
        assert!(request.messages[0].content.contains("refines English text"));
        assert_eq!(request.messages[1].role, "user");
        assert!(request.messages[1].content.contains("i has a apple"));
    }
}
