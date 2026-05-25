use std::path::{Path, PathBuf};

use crate::model::{ModelError, ModelResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAssetPaths {
    pub config_file: PathBuf,
    pub tokenizer_file: PathBuf,
    pub weight_files: Vec<PathBuf>,
}

impl ModelAssetPaths {
    pub fn new(
        config_file: impl Into<PathBuf>,
        tokenizer_file: impl Into<PathBuf>,
        weight_files: Vec<PathBuf>,
    ) -> ModelResult<Self> {
        let paths = Self {
            config_file: config_file.into(),
            tokenizer_file: tokenizer_file.into(),
            weight_files,
        };
        paths.validate()?;
        Ok(paths)
    }

    pub fn validate(&self) -> ModelResult<()> {
        validate_json_file(&self.config_file, "config file must be a JSON file")?;
        validate_json_file(&self.tokenizer_file, "tokenizer file must be a JSON file")?;

        if self.weight_files.is_empty() {
            return Err(ModelError::InvalidConfig(
                "at least one weight file is required",
            ));
        }

        for weight_file in &self.weight_files {
            validate_weight_file(weight_file)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLoadRequest {
    pub target: ModelAssetPaths,
    pub draft: Option<ModelAssetPaths>,
}

impl ModelLoadRequest {
    pub fn target_only(target: ModelAssetPaths) -> Self {
        Self {
            target,
            draft: None,
        }
    }

    pub fn with_draft(target: ModelAssetPaths, draft: ModelAssetPaths) -> Self {
        Self {
            target,
            draft: Some(draft),
        }
    }

    pub fn validate(&self) -> ModelResult<()> {
        self.target.validate()?;

        if let Some(draft) = &self.draft {
            draft.validate()?;
        }

        Ok(())
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }
}

fn validate_json_file(path: &Path, message: &'static str) -> ModelResult<()> {
    if !has_extension(path, "json") {
        return Err(ModelError::InvalidConfig(message));
    }

    Ok(())
}

fn validate_weight_file(path: &Path) -> ModelResult<()> {
    if has_extension(path, "safetensors") || has_extension(path, "gguf") {
        return Ok(());
    }

    Err(ModelError::InvalidConfig(
        "weight files must be safetensors or gguf files",
    ))
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{
        loading::{ModelAssetPaths, ModelLoadRequest},
        model::ModelError,
    };

    fn valid_assets() -> ModelAssetPaths {
        ModelAssetPaths::new(
            "/models/qwen/config.json",
            "/models/qwen/tokenizer.json",
            vec![PathBuf::from("/models/qwen/model.safetensors")],
        )
        .expect("valid paths")
    }

    #[test]
    fn accepts_safetensor_assets() {
        let paths = valid_assets();

        assert_eq!(paths.config_file, PathBuf::from("/models/qwen/config.json"));
        assert_eq!(
            paths.tokenizer_file,
            PathBuf::from("/models/qwen/tokenizer.json")
        );
        assert_eq!(paths.weight_files.len(), 1);
    }

    #[test]
    fn accepts_gguf_assets() {
        let paths = ModelAssetPaths::new(
            "/models/tiny/config.JSON",
            "/models/tiny/tokenizer.JSON",
            vec![PathBuf::from("/models/tiny/model.GGUF")],
        );

        assert!(paths.is_ok());
    }

    #[test]
    fn rejects_invalid_asset_extensions() {
        assert_eq!(
            ModelAssetPaths::new(
                "/models/qwen/config.toml",
                "/models/qwen/tokenizer.json",
                vec![PathBuf::from("/models/qwen/model.safetensors")],
            ),
            Err(ModelError::InvalidConfig("config file must be a JSON file"))
        );
        assert_eq!(
            ModelAssetPaths::new(
                "/models/qwen/config.json",
                "/models/qwen/tokenizer.txt",
                vec![PathBuf::from("/models/qwen/model.safetensors")],
            ),
            Err(ModelError::InvalidConfig(
                "tokenizer file must be a JSON file"
            ))
        );
        assert_eq!(
            ModelAssetPaths::new(
                "/models/qwen/config.json",
                "/models/qwen/tokenizer.json",
                vec![PathBuf::from("/models/qwen/model.bin")],
            ),
            Err(ModelError::InvalidConfig(
                "weight files must be safetensors or gguf files"
            ))
        );
    }

    #[test]
    fn rejects_empty_weight_list() {
        assert_eq!(
            ModelAssetPaths::new(
                "/models/qwen/config.json",
                "/models/qwen/tokenizer.json",
                Vec::new(),
            ),
            Err(ModelError::InvalidConfig(
                "at least one weight file is required"
            ))
        );
    }

    #[test]
    fn distinguishes_target_only_and_draft_requests() {
        let target = valid_assets();
        let draft = ModelAssetPaths::new(
            "/models/qwen-draft/config.json",
            "/models/qwen-draft/tokenizer.json",
            vec![PathBuf::from("/models/qwen-draft/model.safetensors")],
        )
        .expect("valid draft");

        let target_only = ModelLoadRequest::target_only(target.clone());
        let with_draft = ModelLoadRequest::with_draft(target, draft);

        assert!(!target_only.has_draft());
        assert!(target_only.validate().is_ok());
        assert!(with_draft.has_draft());
        assert!(with_draft.validate().is_ok());
    }
}
