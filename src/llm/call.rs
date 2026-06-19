use crate::llm::{
    anthorpic,
    env::{anthorpic_client_from_env, gemini_client_from_env, openai_client_from_env},
    gemini,
    types::{ChatKwargs, ChatResponse, LlmError, Provider, provider_for_model},
};

pub async fn call(kwargs: ChatKwargs) -> Result<ChatResponse, LlmError> {
    let model = kwargs.model.clone();
    let Some(provider) = provider_for_model(&model) else {
        return Err(LlmError::UnknownModel(model));
    };

    match provider {
        Provider::OpenAI => {
            let client = openai_client_from_env()?;
            let req: async_openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
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
            let client = openai_client_from_env()?;
            let req: async_openai::types::chat::CreateChatCompletionRequest = kwargs.try_into()?;
            Ok(ChatResponse::OpenAI(client.chat(req).await?))
        }
    }
}
