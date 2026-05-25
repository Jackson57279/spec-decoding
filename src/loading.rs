use std::path::{Path, PathBuf};

use crate::{
    drafters::Drafter,
    model::{ModelError, ModelResult, TargetModel, Tokenizer},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAssetPaths {
    pub config_file: PathBuf,
    pub tokenizer_file: PathBuf,
    pub weight_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    SafeTensors,
    Gguf,
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

        self.weight_format()?;

        Ok(())
    }

    pub fn weight_format(&self) -> ModelResult<WeightFormat> {
        let mut format = None;

        for weight_file in &self.weight_files {
            let current = weight_format_for_file(weight_file)?;
            if format.is_some_and(|expected| expected != current) {
                return Err(ModelError::InvalidConfig(
                    "weight files must use one format per model",
                ));
            }

            format = Some(current);
        }

        format.ok_or(ModelError::InvalidConfig(
            "at least one weight file is required",
        ))
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
pub struct LoadedModel<M, Tok> {
    pub model: M,
    pub tokenizer: Tok,
}

impl<M, Tok> LoadedModel<M, Tok> {
    pub fn new(model: M, tokenizer: Tok) -> Self {
        Self { model, tokenizer }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedModelBundle<T, TT, D, DT> {
    pub target: LoadedModel<T, TT>,
    pub draft: Option<LoadedModel<D, DT>>,
}

impl<T, TT, D, DT> LoadedModelBundle<T, TT, D, DT> {
    pub fn target_only(target: T, target_tokenizer: TT) -> Self {
        Self {
            target: LoadedModel::new(target, target_tokenizer),
            draft: None,
        }
    }

    pub fn with_draft(target: T, target_tokenizer: TT, draft: D, draft_tokenizer: DT) -> Self {
        Self {
            target: LoadedModel::new(target, target_tokenizer),
            draft: Some(LoadedModel::new(draft, draft_tokenizer)),
        }
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }
}

pub trait ModelLoader {
    type Target: TargetModel;
    type TargetTokenizer: Tokenizer;
    type Draft: Drafter;
    type DraftTokenizer: Tokenizer;

    fn load_target(&mut self, assets: &ModelAssetPaths) -> ModelResult<Self::Target>;
    fn load_target_tokenizer(
        &mut self,
        assets: &ModelAssetPaths,
    ) -> ModelResult<Self::TargetTokenizer>;
    fn load_draft(&mut self, assets: &ModelAssetPaths) -> ModelResult<Self::Draft>;
    fn load_draft_tokenizer(
        &mut self,
        assets: &ModelAssetPaths,
    ) -> ModelResult<Self::DraftTokenizer>;

    fn load(
        &mut self,
        request: &ModelLoadRequest,
    ) -> ModelResult<
        LoadedModelBundle<Self::Target, Self::TargetTokenizer, Self::Draft, Self::DraftTokenizer>,
    > {
        request.validate()?;

        let target = self.load_target(&request.target)?;
        let target_tokenizer = self.load_target_tokenizer(&request.target)?;
        let draft = if let Some(assets) = &request.draft {
            Some(LoadedModel::new(
                self.load_draft(assets)?,
                self.load_draft_tokenizer(assets)?,
            ))
        } else {
            None
        };

        Ok(LoadedModelBundle {
            target: LoadedModel::new(target, target_tokenizer),
            draft,
        })
    }
}

fn validate_json_file(path: &Path, message: &'static str) -> ModelResult<()> {
    if !has_extension(path, "json") {
        return Err(ModelError::InvalidConfig(message));
    }

    Ok(())
}

fn weight_format_for_file(path: &Path) -> ModelResult<WeightFormat> {
    if has_extension(path, "safetensors") {
        return Ok(WeightFormat::SafeTensors);
    }

    if has_extension(path, "gguf") {
        return Ok(WeightFormat::Gguf);
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
        loading::{
            LoadedModel, LoadedModelBundle, ModelAssetPaths, ModelLoadRequest, ModelLoader,
            WeightFormat,
        },
        model::{ModelError, ModelResult, TargetModel, TokenId, TokenSequence, Tokenizer},
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
        assert_eq!(paths.weight_format(), Ok(WeightFormat::SafeTensors));
    }

    #[test]
    fn accepts_gguf_assets() {
        let paths = ModelAssetPaths::new(
            "/models/tiny/config.JSON",
            "/models/tiny/tokenizer.JSON",
            vec![PathBuf::from("/models/tiny/model.GGUF")],
        );

        assert_eq!(
            paths.expect("valid gguf assets").weight_format(),
            Ok(WeightFormat::Gguf)
        );
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
    fn rejects_mixed_weight_formats() {
        assert_eq!(
            ModelAssetPaths::new(
                "/models/qwen/config.json",
                "/models/qwen/tokenizer.json",
                vec![
                    PathBuf::from("/models/qwen/model-00001.safetensors"),
                    PathBuf::from("/models/qwen/model-00002.gguf"),
                ],
            ),
            Err(ModelError::InvalidConfig(
                "weight files must use one format per model"
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct FakeTokenizer {
        tokenizer_file: PathBuf,
    }

    impl Tokenizer for FakeTokenizer {
        fn vocab_size(&self) -> usize {
            1
        }

        fn encode(&self, _text: &str) -> ModelResult<TokenSequence> {
            Ok(TokenSequence::new(Vec::new()))
        }

        fn decode(&self, _tokens: &[TokenId]) -> ModelResult<String> {
            Ok(String::new())
        }
    }

    #[derive(Debug, Default)]
    struct FakeLoader {
        target_loads: usize,
        target_tokenizer_loads: usize,
        draft_loads: usize,
        draft_tokenizer_loads: usize,
    }

    impl ModelLoader for FakeLoader {
        type Target = FakeTarget;
        type TargetTokenizer = FakeTokenizer;
        type Draft = FakeDraft;
        type DraftTokenizer = FakeTokenizer;

        fn load_target(&mut self, assets: &ModelAssetPaths) -> ModelResult<Self::Target> {
            self.target_loads += 1;
            Ok(FakeTarget {
                config_file: assets.config_file.clone(),
            })
        }

        fn load_target_tokenizer(
            &mut self,
            assets: &ModelAssetPaths,
        ) -> ModelResult<Self::TargetTokenizer> {
            self.target_tokenizer_loads += 1;
            Ok(FakeTokenizer {
                tokenizer_file: assets.tokenizer_file.clone(),
            })
        }

        fn load_draft(&mut self, assets: &ModelAssetPaths) -> ModelResult<Self::Draft> {
            self.draft_loads += 1;
            Ok(FakeDraft {
                config_file: assets.config_file.clone(),
            })
        }

        fn load_draft_tokenizer(
            &mut self,
            assets: &ModelAssetPaths,
        ) -> ModelResult<Self::DraftTokenizer> {
            self.draft_tokenizer_loads += 1;
            Ok(FakeTokenizer {
                tokenizer_file: assets.tokenizer_file.clone(),
            })
        }
    }

    #[test]
    fn builds_loaded_models_and_bundles() {
        let target = FakeTarget {
            config_file: PathBuf::from("/models/qwen/config.json"),
        };
        let target_tokenizer = FakeTokenizer {
            tokenizer_file: PathBuf::from("/models/qwen/tokenizer.json"),
        };
        let draft = FakeDraft {
            config_file: PathBuf::from("/models/qwen-draft/config.json"),
        };
        let draft_tokenizer = FakeTokenizer {
            tokenizer_file: PathBuf::from("/models/qwen-draft/tokenizer.json"),
        };

        assert_eq!(
            LoadedModel::new(target.clone(), target_tokenizer.clone()).tokenizer,
            target_tokenizer
        );
        assert!(
            !LoadedModelBundle::<_, _, FakeDraft, FakeTokenizer>::target_only(
                target.clone(),
                FakeTokenizer {
                    tokenizer_file: PathBuf::from("/models/qwen/tokenizer.json"),
                },
            )
            .has_draft()
        );
        assert!(
            LoadedModelBundle::with_draft(target, target_tokenizer, draft, draft_tokenizer)
                .has_draft()
        );
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

        assert_eq!(bundle.target.model.config_file, target.config_file);
        assert_eq!(
            bundle.target.tokenizer.tokenizer_file,
            target.tokenizer_file
        );
        let loaded_draft = bundle.draft.expect("draft should load");
        assert_eq!(loaded_draft.model.config_file, draft.config_file);
        assert_eq!(loaded_draft.tokenizer.tokenizer_file, draft.tokenizer_file);
        assert_eq!(loader.target_loads, 1);
        assert_eq!(loader.target_tokenizer_loads, 1);
        assert_eq!(loader.draft_loads, 1);
        assert_eq!(loader.draft_tokenizer_loads, 1);
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
        assert_eq!(loader.target_tokenizer_loads, 0);
        assert_eq!(loader.draft_loads, 0);
        assert_eq!(loader.draft_tokenizer_loads, 0);
    }
}
