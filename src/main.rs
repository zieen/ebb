use std::{
    env, fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};

use dotenv::dotenv;
use service::refine::refine_text;

mod llm;
mod service;

#[tokio::main]
async fn main() {
    load_global_config().ok();
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 && args[1] == "setup" {
        if let Err(err) = run_setup() {
            eprintln!("Setup failed: {err}");
            std::process::exit(1);
        }
        return;
    }

    if args.len() < 2 {
        eprintln!("Usage: {} setup | <words...>", args[0]);
        return;
    }

    let args = &args[1..];
    let words = args.join(" ");

    match refine_text(&words).await {
        Ok(result) => {
            println!("BB: {}", result.refined_text);
            if !result.mistakes.is_empty() {
                println!();
                println!("Mistakes:");
                for mistake in result.mistakes {
                    println!("- {mistake}");
                }
            }
        }
        Err(err) => {
            eprintln!("Refine failed: {err:?}");
            std::process::exit(1);
        }
    }
}

fn load_global_config() -> io::Result<()> {
    let config_path = user_config_path()?;
    if config_path.is_file() {
        dotenv::from_path(&config_path).ok();
    } else {
        dotenv().ok();
    }
    Ok(())
}

fn run_setup() -> io::Result<()> {
    let config_path = user_config_path()?;
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    run_setup_with_io(&mut reader, &mut writer, &config_path)
}

fn run_setup_with_io<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    config_path: &Path,
) -> io::Result<()> {
    writeln!(writer, "LLM setup")?;
    writeln!(writer, "1. OpenAI")?;
    writeln!(writer, "2. Gemini")?;
    writeln!(writer, "3. Anthropic")?;
    writeln!(writer, "4. DeepSeek")?;

    let vendor = prompt_vendor(reader, writer)?;
    let api_key = prompt_value(reader, writer, "Enter API key: ")?;

    if api_key.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "API key cannot be empty",
        ));
    }

    save_setup(config_path, &vendor, api_key.trim())?;

    writeln!(writer, "Saved configuration to {}", config_path.display())?;
    Ok(())
}

fn prompt_vendor<R: BufRead, W: Write>(reader: &mut R, writer: &mut W) -> io::Result<String> {
    loop {
        let value = prompt_value(reader, writer, "Choose vendor [1-4]: ")?;
        let normalized = value.trim().to_ascii_lowercase();

        let vendor = match normalized.as_str() {
            "1" | "openai" => Some("openai"),
            "2" | "gemini" => Some("gemini"),
            "3" | "anthropic" => Some("anthropic"),
            "4" | "deepseek" => Some("deepseek"),
            _ => None,
        };

        if let Some(vendor) = vendor {
            return Ok(vendor.to_string());
        }

        writeln!(writer, "Invalid selection. Choose 1, 2, 3, or 4.")?;
    }
}

fn prompt_value<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    prompt: &str,
) -> io::Result<String> {
    write!(writer, "{prompt}")?;
    writer.flush()?;

    let mut input = String::new();
    reader.read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn save_setup(config_path: &Path, vendor: &str, api_key: &str) -> io::Result<()> {
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let existing = fs::read_to_string(config_path).unwrap_or_default();
    let env_key = vendor_api_key_var(vendor).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Unsupported vendor: {vendor}"),
        )
    })?;

    let updated = upsert_env_line(&existing, "LLM_VENDOR", vendor);
    let updated = upsert_env_line(&updated, "LLM_MODEL", vendor_default_model(vendor));
    let updated = upsert_env_line(&updated, env_key, api_key);

    fs::write(config_path, updated)
}

fn user_config_path() -> io::Result<PathBuf> {
    let home = env::var_os("HOME").map(PathBuf::from);
    let xdg = env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
    config_path_from_env(home.as_deref(), xdg.as_deref())
}

fn config_path_from_env(home_dir: Option<&Path>, xdg_config_home: Option<&Path>) -> io::Result<PathBuf> {
    let base_dir = if let Some(xdg) = xdg_config_home {
        xdg.to_path_buf()
    } else if let Some(home) = home_dir {
        home.join(".config")
    } else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Could not determine a user config directory",
        ));
    };

    Ok(base_dir.join("ebb").join(".env"))
}

fn vendor_api_key_var(vendor: &str) -> Option<&'static str> {
    match vendor.trim().to_ascii_lowercase().as_str() {
        "openai" => Some("OPENAI_API_KEY"),
        "gemini" => Some("GEMINI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "deepseek" => Some("DEEPSEEK_API_KEY"),
        _ => None,
    }
}

fn vendor_default_model(vendor: &str) -> &'static str {
    match vendor {
        "openai" => "gpt-5.5",
        "gemini" => "gemini-3.5-flash",
        "anthropic" => "claude-sonnet-4-6",
        "deepseek" => "deepseek-v4-flash",
        _ => "gpt-5.5",
    }
}

fn upsert_env_line(contents: &str, key: &str, value: &str) -> String {
    let new_line = format!("{key}=\"{}\"", escape_env_value(value));
    let target_prefix = format!("{key}=");
    let mut replaced = false;
    let mut lines = Vec::new();

    for line in contents.lines() {
        if line.trim_start().starts_with(&target_prefix) {
            if !replaced {
                lines.push(new_line.clone());
                replaced = true;
            }
            continue;
        }

        lines.push(line.to_string());
    }

    if !replaced {
        lines.push(new_line);
    }

    let mut output = lines.join("\n");
    if !output.ends_with('\n') {
        output.push('\n');
    }
    output
}

fn escape_env_value(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_api_key_var_matches_supported_vendors() {
        assert_eq!(vendor_api_key_var("openai"), Some("OPENAI_API_KEY"));
        assert_eq!(vendor_api_key_var("gemini"), Some("GEMINI_API_KEY"));
        assert_eq!(vendor_api_key_var("anthropic"), Some("ANTHROPIC_API_KEY"));
        assert_eq!(vendor_api_key_var("deepseek"), Some("DEEPSEEK_API_KEY"));
        assert_eq!(vendor_api_key_var("unknown"), None);
    }

    #[test]
    fn upsert_env_line_updates_existing_key_without_removing_others() {
        let original = "DATABASE_URL=postgres://localhost/db\nOPENAI_API_KEY=old-key\n";
        let updated = upsert_env_line(original, "OPENAI_API_KEY", "new-key");

        assert!(updated.contains("DATABASE_URL=postgres://localhost/db"));
        assert!(updated.contains("OPENAI_API_KEY=\"new-key\""));
        assert!(!updated.contains("OPENAI_API_KEY=old-key"));
    }

    #[test]
    fn upsert_env_line_appends_missing_key() {
        let updated = upsert_env_line(
            "DATABASE_URL=postgres://localhost/db\n",
            "GEMINI_API_KEY",
            "test-key",
        );

        assert!(updated.contains("DATABASE_URL=postgres://localhost/db"));
        assert!(updated.contains("GEMINI_API_KEY=\"test-key\""));
    }

    #[test]
    fn run_setup_writes_vendor_model_and_api_key() {
        let temp_dir = std::env::temp_dir().join(format!(
            "ebb-setup-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let config_path = temp_dir.join(".env");

        let mut input = std::io::Cursor::new(b"2\ntest-gemini-key\n".to_vec());
        let mut output = Vec::new();

        run_setup_with_io(&mut input, &mut output, &config_path).unwrap();

        let saved = std::fs::read_to_string(&config_path).unwrap();
        assert!(saved.contains("LLM_VENDOR=\"gemini\""));
        assert!(saved.contains("LLM_MODEL=\"gemini-3.5-flash\""));
        assert!(saved.contains("GEMINI_API_KEY=\"test-gemini-key\""));

        std::fs::remove_file(&config_path).unwrap();
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn config_path_prefers_xdg_config_home() {
        let path = config_path_from_env(
            Some(Path::new("/tmp/home")),
            Some(Path::new("/tmp/xdg-config")),
        )
        .unwrap();

        assert_eq!(path, PathBuf::from("/tmp/xdg-config/ebb/.env"));
    }

    #[test]
    fn config_path_falls_back_to_home_config_directory() {
        let path = config_path_from_env(Some(Path::new("/tmp/home")), None).unwrap();

        assert_eq!(path, PathBuf::from("/tmp/home/.config/ebb/.env"));
    }
}
