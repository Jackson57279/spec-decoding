use std::path::{Path, PathBuf};

use crate::{
    drafters::Drafter,
    model::{ModelError, ModelResult, TargetModel},
};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedModelBundle<T, D> {
    pub target: T,
    pub draft: Option<D>,
}

impl<T, D> LoadedModelBundle<T, D> {
    pub fn target_only(target: T) -> Self {
        Self {
            target,
            draft: None,
        }
    }

    pub fn with_draft(target: T, draft: D) -> Self {
        Self {
            target,
            draft: Some(draft),
        }
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }
}

pub trait ModelLoader {
    type Target: TargetModel;
    type Draft: Drafter;

    fn load_target(&mut self, assets: &ModelAssetPaths) -> ModelResult<Self::Target>;
    fn load_draft(&mut self, assets: &ModelAssetPaths) -> ModelResult<Self::Draft>;

    fn load(
        &mut self,
        request: &ModelLoadRequest,
    ) -> ModelResult<LoadedModelBundle<Self::Target, Self::Draft>> {
        request.validate()?;

        let target = self.load_target(&request.target)?;
        let draft = request
            .draft
            .as_ref()
            .map(|assets| self.load_draft(assets))
            .transpose()?;

        Ok(LoadedModelBundle { target, draft })
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
        drafters::{DraftSequence, Drafter},
        loading::{LoadedModelBundle, ModelAssetPaths, ModelLoadRequest, ModelLoader},
        model::{ModelError, ModelResult, TargetModel, TokenId},
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct FakeTarget {
        config_file: PathBuf,
    }

    impl TargetModel for FakeTarget {
        fn vocab_size(&self) -> usize {
            1
        }

        fn logits_for_prefix(&mut self, _prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
            Ok(vec![0.0])
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct FakeDraft {
        config_file: PathBuf,
    }

    impl Drafter for FakeDraft {
        fn draft(&mut self, _prefix: &[TokenId], _max_tokens: usize) -> ModelResult<DraftSequence> {
            Ok(DraftSequence::new(Vec::new()))
        }
    }

    #[derive(Debug, Default)]
    struct FakeLoader {
        target_loads: usize,
        draft_loads: usize,
    }

    impl ModelLoader for FakeLoader {
        type Target = FakeTarget;
        type Draft = FakeDraft;

        fn load_target(&mut self, assets: &ModelAssetPaths) -> ModelResult<Self::Target> {
            self.target_loads += 1;
            Ok(FakeTarget {
                config_file: assets.config_file.clone(),
            })
        }

        fn load_draft(&mut self, assets: &ModelAssetPaths) -> ModelResult<Self::Draft> {
            self.draft_loads += 1;
            Ok(FakeDraft {
                config_file: assets.config_file.clone(),
            })
        }
    }

    #[test]
    fn builds_loaded_model_bundles() {
        let target = FakeTarget {
            config_file: PathBuf::from("/models/qwen/config.json"),
        };
        let draft = FakeDraft {
            config_file: PathBuf::from("/models/qwen-draft/config.json"),
        };

        assert!(!LoadedModelBundle::<_, FakeDraft>::target_only(target.clone()).has_draft());
        assert!(LoadedModelBundle::with_draft(target, draft).has_draft());
    }

    #[test]
    fn loads_valid_target_and_draft_assets() {
        let target = valid_assets();
        let draft = ModelAssetPaths::new(
            "/models/qwen-draft/config.json",
            "/models/qwen-draft/tokenizer.json",
            vec![PathBuf::from("/models/qwen-draft/model.safetensors")],
        )
        .expect("valid draft");
        let request = ModelLoadRequest::with_draft(target.clone(), draft.clone());
        let mut loader = FakeLoader::default();

        let bundle = loader.load(&request).expect("load should succeed");

        assert_eq!(bundle.target.config_file, target.config_file);
        assert_eq!(
            bundle.draft.expect("draft should load").config_file,
            draft.config_file
        );
        assert_eq!(loader.target_loads, 1);
        assert_eq!(loader.draft_loads, 1);
    }

    #[test]
    fn validates_requests_before_loading() {
        let request = ModelLoadRequest::target_only(ModelAssetPaths {
            config_file: PathBuf::from("/models/qwen/config.toml"),
            tokenizer_file: PathBuf::from("/models/qwen/tokenizer.json"),
            weight_files: vec![PathBuf::from("/models/qwen/model.safetensors")],
        });
        let mut loader = FakeLoader::default();

        let result = loader.load(&request);

        assert_eq!(
            result,
            Err(ModelError::InvalidConfig("config file must be a JSON file"))
        );
        assert_eq!(loader.target_loads, 0);
        assert_eq!(loader.draft_loads, 0);
    }
}
