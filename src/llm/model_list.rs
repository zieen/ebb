use indexmap::IndexMap;
use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub input_price: f64,
    pub output_price: f64,
    pub cached_price: f64,
}

pub mod model_names {
    pub const CLAUDE_OPUS_4_8: &str = "claude-opus-4-8";
    pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";
    pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

    pub const CHATGPT_5_5: &str = "gpt-5.5";
    pub const CHATGPT_5_4: &str = "gpt-5.4";
    pub const CHATGPT_5_4_MINI: &str = "gpt-5.4-mini";

    pub const GEMINI_3_1: &str = "gemini-3.1-pro-preview";
    pub const GEMINI_3_5_FLASH: &str = "gemini-3.5-flash";

    pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
    pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";
}

pub fn is_openai_model(model: &str) -> bool {
    model.starts_with("gpt-")
}

pub fn is_gemini_model(model: &str) -> bool {
    model.starts_with("gemini-")
}

pub fn is_anthorpic_model(model: &str) -> bool {
    model.starts_with("claude-")
}

pub fn is_deepseek_model(model: &str) -> bool {
    model.starts_with("deepseek-")
}

pub static MODEL_LIST: Lazy<IndexMap<String, ModelInfo>> = Lazy::new(|| {
    let mut map = IndexMap::new();

    map.insert(
        model_names::CLAUDE_OPUS_4_8.to_string(),
        ModelInfo {
            name: "Claude Opus 4-8".to_string(),
            input_price: 5.0,
            output_price: 25.0,
            cached_price: 0.5,
        },
    );

    map.insert(
        model_names::CLAUDE_SONNET_4_6.to_string(),
        ModelInfo {
            name: "Claude Sonnet 4-6".to_string(),
            input_price: 3.0,
            output_price: 15.0,
            cached_price: 0.3,
        },
    );

    map.insert(
        model_names::CLAUDE_HAIKU_4_5.to_string(),
        ModelInfo {
            name: model_names::CLAUDE_HAIKU_4_5.to_string(),
            input_price: 1.0,
            output_price: 5.0,
            cached_price: 0.1,
        },
    );

    map.insert(
        model_names::CHATGPT_5_5.to_string(),
        ModelInfo {
            name: "GPT-5.5".to_string(),
            input_price: 5.0,
            output_price: 30.0,
            cached_price: 0.5,
        },
    );

    map.insert(
        model_names::CHATGPT_5_4.to_string(),
        ModelInfo {
            name: "GPT-5.4".to_string(),
            input_price: 2.5,
            output_price: 15.0,
            cached_price: 0.25,
        },
    );

    map.insert(
        model_names::CHATGPT_5_4_MINI.to_string(),
        ModelInfo {
            name: "GPT-5.4 Mini".to_string(),
            input_price: 0.75,
            output_price: 4.5,
            cached_price: 0.075,
        },
    );

    map.insert(
        model_names::GEMINI_3_1.to_string(),
        ModelInfo {
            name: "Gemini 3.1 Pro Preview".to_string(),
            input_price: 2.0,
            output_price: 12.0,
            cached_price: 0.2,
        },
    );

    map.insert(
        model_names::GEMINI_3_5_FLASH.to_string(),
        ModelInfo {
            name: "Gemini 3.5 Flash".to_string(),
            input_price: 1.5,
            output_price: 9.0,
            cached_price: 0.15,
        },
    );

    map
});
