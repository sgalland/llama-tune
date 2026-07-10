//! Persisted user settings (llama.cpp path, last-launched model), stored as
//! JSON under the OS config directory.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Persistent user settings (currently just the llama.cpp executable path),
/// stored as JSON under the OS config directory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct Config {
    pub(crate) llama_cpp_path: Option<String>,
    /// model_id of the most recently launched model, for quick relaunch.
    pub(crate) last_launched_model_id: Option<String>,
}

impl Config {
    /// Load the config from disk, falling back to defaults if it doesn't
    /// exist yet or fails to parse.
    pub(crate) fn load() -> Self {
        Self::config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub(crate) fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("could not determine config directory"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("llama-tune").join("config.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests exercise serde round-tripping directly rather than
    // Config::load()/save()/config_path(), which would read/write the
    // developer's real config file under dirs::config_dir().

    #[test]
    fn default_config_has_no_persisted_state() {
        let config = Config::default();
        assert!(config.llama_cpp_path.is_none());
        assert!(config.last_launched_model_id.is_none());
    }

    #[test]
    fn serde_roundtrip_preserves_fields() {
        let config = Config {
            llama_cpp_path: Some("/usr/local/bin/llama-cli".to_string()),
            last_launched_model_id: Some("org/some-model".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.llama_cpp_path, config.llama_cpp_path);
        assert_eq!(
            restored.last_launched_model_id,
            config.last_launched_model_id
        );
    }

    #[test]
    fn serde_roundtrip_preserves_none_fields() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();
        assert!(restored.llama_cpp_path.is_none());
        assert!(restored.last_launched_model_id.is_none());
    }

    #[test]
    fn malformed_json_fails_to_parse() {
        // Mirrors the degrade-to-default behavior Config::load() relies on:
        // a corrupt config file should fail parsing, not panic or partially load.
        assert!(serde_json::from_str::<Config>("not json").is_err());
    }

    #[test]
    fn missing_fields_default_to_none() {
        let restored: Config = serde_json::from_str("{}").unwrap();
        assert!(restored.llama_cpp_path.is_none());
        assert!(restored.last_launched_model_id.is_none());
    }
}
