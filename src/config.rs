use std::{fs::File, path::Path};

use serde_json::Value;

use crate::model::{ModelError, ModelResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfigSummary {
    pub model_type: Option<String>,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerConfigSummary {
    pub model_type: Option<String>,
    pub vocab_size: Option<usize>,
}

pub fn read_model_config_summary(path: &Path) -> ModelResult<ModelConfigSummary> {
    let json = read_json(path)?;

    Ok(ModelConfigSummary {
        model_type: string_field(&json, "model_type"),
        vocab_size: usize_field(&json, "vocab_size")?,
        hidden_size: usize_field(&json, "hidden_size")?,
        num_hidden_layers: usize_field(&json, "num_hidden_layers")?,
    })
}

pub fn read_tokenizer_config_summary(path: &Path) -> ModelResult<TokenizerConfigSummary> {
    let json = read_json(path)?;
    let model = json.get("model").unwrap_or(&json);

    Ok(TokenizerConfigSummary {
        model_type: string_field(model, "type"),
        vocab_size: object_len_field(model, "vocab"),
    })
}

fn read_json(path: &Path) -> ModelResult<Value> {
    let file = File::open(path).map_err(|_| ModelError::InvalidConfig("JSON file must exist"))?;
    serde_json::from_reader(file).map_err(|_| ModelError::InvalidConfig("invalid JSON file"))
}

fn string_field(json: &Value, key: &str) -> Option<String> {
    json.get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn usize_field(json: &Value, key: &str) -> ModelResult<Option<usize>> {
    json.get(key)
        .map(value_as_usize)
        .transpose()
        .map_err(|_| ModelError::InvalidConfig("JSON integer field is out of range"))
}

fn object_len_field(json: &Value, key: &str) -> Option<usize> {
    json.get(key).and_then(Value::as_object).map(|value| value.len())
}

fn value_as_usize(value: &Value) -> Result<usize, ()> {
    value.as_u64().ok_or(())?.try_into().map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{File, create_dir_all, remove_dir_all, write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        config::{read_model_config_summary, read_tokenizer_config_summary},
        model::ModelError,
    };

    struct TempJson {
        root: PathBuf,
        path: PathBuf,
    }

    impl TempJson {
        fn new(name: &str, contents: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-json-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");
            let path = root.join("asset.json");
            write(&path, contents).expect("JSON should be written");

            Self { root, path }
        }
    }

    impl Drop for TempJson {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.root);
        }
    }

    #[test]
    fn reads_model_config_summaries() {
        let json = TempJson::new(
            "model",
            r#"{
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 32
            }"#,
        );

        let summary = read_model_config_summary(&json.path).expect("summary should parse");

        assert_eq!(summary.model_type.as_deref(), Some("llama"));
        assert_eq!(summary.vocab_size, Some(32000));
        assert_eq!(summary.hidden_size, Some(4096));
        assert_eq!(summary.num_hidden_layers, Some(32));
    }

    #[test]
    fn reads_tokenizer_config_summaries() {
        let json = TempJson::new(
            "tokenizer",
            r#"{
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "hello": 0,
                        "world": 1
                    }
                }
            }"#,
        );

        let summary = read_tokenizer_config_summary(&json.path).expect("summary should parse");

        assert_eq!(summary.model_type.as_deref(), Some("BPE"));
        assert_eq!(summary.vocab_size, Some(2));
    }

    #[test]
    fn rejects_missing_and_invalid_json_files() {
        let missing = std::env::temp_dir().join("speclative-diffusion-missing-config.json");
        let invalid = TempJson::new("invalid", "{not-json");

        assert_eq!(
            read_model_config_summary(&missing),
            Err(ModelError::InvalidConfig("JSON file must exist"))
        );
        assert_eq!(
            read_model_config_summary(&invalid.path),
            Err(ModelError::InvalidConfig("invalid JSON file"))
        );
    }

    #[test]
    fn rejects_out_of_range_integer_fields() {
        let json = TempJson::new("range", r#"{"vocab_size": 18446744073709551615}"#);

        if usize::BITS < 64 {
            assert_eq!(
                read_model_config_summary(&json.path),
                Err(ModelError::InvalidConfig(
                    "JSON integer field is out of range"
                ))
            );
        } else {
            assert_eq!(
                read_model_config_summary(&json.path)
                    .expect("summary should parse")
                    .vocab_size,
                Some(usize::MAX)
            );
        }
    }

    #[test]
    fn rejects_directories_as_json_files() {
        let temp = TempJson::new("directory", "{}");
        let directory = temp.root.join("dir");
        create_dir_all(&directory).expect("directory should be created");
        let _ = File::open(&temp.path).expect("file should exist");

        assert_eq!(
            read_model_config_summary(&directory),
            Err(ModelError::InvalidConfig("JSON file must exist"))
        );
    }
}
