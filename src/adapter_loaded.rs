use crate::{
    adapters::{AdapterKind, AdapterLoadPreflight, AdapterLoaderShell, AdapterModelPreflight},
    config::{ModelAssetSummary, ModelConfigSummary, TokenizerConfigSummary},
    loading::ModelLoadRequest,
    model::{ModelError, ModelResult, TargetModel, TokenId, TokenSequence, Tokenizer},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterTargetPlaceholder {
    summary: ModelConfigSummary,
    tokenizer_vocab_size: Option<usize>,
}

impl AdapterTargetPlaceholder {
    pub fn from_summaries(summary: ModelConfigSummary, tokenizer: &TokenizerConfigSummary) -> Self {
        Self {
            summary,
            tokenizer_vocab_size: tokenizer.vocab_size,
        }
    }

    pub fn summary(&self) -> &ModelConfigSummary {
        &self.summary
    }

    pub fn model_type(&self) -> Option<&str> {
        self.summary.model_type.as_deref()
    }
}

impl TargetModel for AdapterTargetPlaceholder {
    fn vocab_size(&self) -> usize {
        self.summary
            .vocab_size
            .or(self.tokenizer_vocab_size)
            .unwrap_or(0)
    }

    fn logits_for_prefix(&mut self, _prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        Err(ModelError::InvalidConfig(
            "metadata target cannot produce logits",
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterTokenizerPlaceholder {
    summary: TokenizerConfigSummary,
}

impl AdapterTokenizerPlaceholder {
    pub fn from_summary(summary: TokenizerConfigSummary) -> Self {
        Self { summary }
    }

    pub fn summary(&self) -> &TokenizerConfigSummary {
        &self.summary
    }

    pub fn model_type(&self) -> Option<&str> {
        self.summary.model_type.as_deref()
    }
}

impl Tokenizer for AdapterTokenizerPlaceholder {
    fn vocab_size(&self) -> usize {
        self.summary.vocab_size.unwrap_or(0)
    }

    fn encode(&self, _text: &str) -> ModelResult<TokenSequence> {
        Err(ModelError::InvalidConfig(
            "metadata tokenizer cannot encode text",
        ))
    }

    fn decode(&self, _tokens: &[TokenId]) -> ModelResult<String> {
        Err(ModelError::InvalidConfig(
            "metadata tokenizer cannot decode tokens",
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterLoadedModelMetadata {
    pub kind: AdapterKind,
    pub model: AdapterTargetPlaceholder,
    pub tokenizer: AdapterTokenizerPlaceholder,
    pub summary: ModelAssetSummary,
}

impl AdapterLoadedModelMetadata {
    pub fn from_preflight(preflight: AdapterModelPreflight) -> Self {
        let tokenizer =
            AdapterTokenizerPlaceholder::from_summary(preflight.summary.tokenizer.clone());
        let model = AdapterTargetPlaceholder::from_summaries(
            preflight.summary.model.clone(),
            &preflight.summary.tokenizer,
        );

        Self {
            kind: preflight.plan.kind,
            model,
            tokenizer,
            summary: preflight.summary,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterLoadedMetadataBundle {
    pub target: AdapterLoadedModelMetadata,
    pub draft: Option<AdapterLoadedModelMetadata>,
}

impl AdapterLoadedMetadataBundle {
    pub fn from_preflight(preflight: AdapterLoadPreflight) -> Self {
        Self {
            target: AdapterLoadedModelMetadata::from_preflight(preflight.target),
            draft: preflight
                .draft
                .map(AdapterLoadedModelMetadata::from_preflight),
        }
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }
}

impl AdapterLoaderShell {
    pub fn load_target_metadata(
        self,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterLoadedMetadataBundle> {
        self.preflight_target_only(request)
            .map(AdapterLoadedMetadataBundle::from_preflight)
    }

    pub fn load_with_draft_metadata(
        self,
        draft_kind: AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterLoadedMetadataBundle> {
        self.preflight_with_draft(draft_kind, request)
            .map(AdapterLoadedMetadataBundle::from_preflight)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{File, create_dir_all, remove_dir_all, write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        adapters::{AdapterKind, AdapterLoaderShell},
        loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat},
        model::{ModelError, TargetModel, Tokenizer},
    };

    struct TempAssets {
        root: PathBuf,
        config: PathBuf,
        tokenizer: PathBuf,
        weights: PathBuf,
    }

    impl TempAssets {
        fn new(name: &str, weight_name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-loaded-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");

            let config = root.join("config.json");
            let tokenizer = root.join("tokenizer.json");
            let weights = root.join(weight_name);

            write(&config, r#"{"model_type":"llama","vocab_size":32000}"#)
                .expect("config should be written");
            write(
                &tokenizer,
                r#"{"model":{"type":"BPE","vocab":{"hello":0,"world":1}}}"#,
            )
            .expect("tokenizer should be written");
            File::create(&weights).expect("weights should be created");

            Self {
                root,
                config,
                tokenizer,
                weights,
            }
        }

        fn paths(&self) -> ModelAssetPaths {
            ModelAssetPaths::new(
                self.config.clone(),
                self.tokenizer.clone(),
                vec![self.weights.clone()],
            )
            .expect("asset paths should be valid")
        }
    }

    impl Drop for TempAssets {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.root);
        }
    }

    #[test]
    fn loader_shell_returns_target_metadata_placeholder() {
        let target = TempAssets::new("target", "model.safetensors");
        let request = ModelLoadRequest::target_only(target.paths());

        let loaded = AdapterLoaderShell::new(AdapterKind::Candle)
            .load_target_metadata(&request)
            .expect("metadata should load");

        assert_eq!(loaded.target.kind, AdapterKind::Candle);
        assert_eq!(
            loaded.target.summary.model.model_type.as_deref(),
            Some("llama")
        );
        assert_eq!(loaded.target.model.model_type(), Some("llama"));
        assert_eq!(TargetModel::vocab_size(&loaded.target.model), 32000);

        let mut target_model = loaded.target.model.clone();
        assert_eq!(
            target_model.logits_for_prefix(&[0, 1]),
            Err(ModelError::InvalidConfig(
                "metadata target cannot produce logits"
            ))
        );

        assert_eq!(loaded.target.tokenizer.model_type(), Some("BPE"));
        assert_eq!(loaded.target.tokenizer.vocab_size(), 2);
        assert_eq!(
            loaded.target.tokenizer.encode("hello"),
            Err(ModelError::InvalidConfig(
                "metadata tokenizer cannot encode text"
            ))
        );
        assert_eq!(
            loaded.target.tokenizer.decode(&[0]),
            Err(ModelError::InvalidConfig(
                "metadata tokenizer cannot decode tokens"
            ))
        );
        assert_eq!(
            loaded.target.summary.weight_format,
            WeightFormat::SafeTensors
        );
        assert!(!loaded.has_draft());
    }

    #[test]
    fn loader_shell_returns_target_and_draft_metadata_placeholders() {
        let target = TempAssets::new("target-draft", "model.safetensors");
        let draft = TempAssets::new("draft", "model.gguf");
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());

        let loaded = AdapterLoaderShell::new(AdapterKind::Candle)
            .load_with_draft_metadata(AdapterKind::Gguf, &request)
            .expect("metadata should load");

        let loaded_draft = loaded.draft.expect("draft metadata should exist");
        assert_eq!(loaded.target.kind, AdapterKind::Candle);
        assert_eq!(loaded_draft.kind, AdapterKind::Gguf);
        assert_eq!(loaded_draft.summary.weight_format, WeightFormat::Gguf);
    }
}
