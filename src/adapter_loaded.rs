#[cfg(feature = "tokenizers")]
use std::path::Path;

use crate::{
    adapter_weight_preflight::{
        AdapterLoadWeightPreflight, AdapterModelWeightPreflight, AdapterWeightFilePreflight,
    },
    adapters::{AdapterKind, AdapterLoadPreflight, AdapterLoaderShell, AdapterModelPreflight},
    config::{ModelAssetSummary, ModelConfigSummary, TokenizerConfigSummary},
    loading::ModelLoadRequest,
    model::{
        ModelError, ModelResult, TargetModel, TokenId, TokenSequence, Tokenizer,
        TokenizerDecodeOptions, TokenizerEncodeOptions,
    },
};

#[cfg(feature = "tokenizers")]
use crate::loading::{LoadedModel, LoadedModelBundle};

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

    fn encode_with_options(
        &self,
        _text: &str,
        _options: TokenizerEncodeOptions,
    ) -> ModelResult<TokenSequence> {
        Err(ModelError::InvalidConfig(
            "metadata tokenizer cannot encode text",
        ))
    }

    fn decode_with_options(
        &self,
        _tokens: &[TokenId],
        _options: TokenizerDecodeOptions,
    ) -> ModelResult<String> {
        Err(ModelError::InvalidConfig(
            "metadata tokenizer cannot decode tokens",
        ))
    }
}

#[cfg(feature = "tokenizers")]
pub struct AdapterJsonTokenizer {
    inner: tokenizers::Tokenizer,
}

#[cfg(feature = "tokenizers")]
impl AdapterJsonTokenizer {
    pub fn from_file(path: &Path) -> ModelResult<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|_| ModelError::InvalidConfig("tokenizer JSON cannot be loaded"))?;

        Ok(Self { inner })
    }
}

#[cfg(feature = "tokenizers")]
impl Tokenizer for AdapterJsonTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    fn encode_with_options(
        &self,
        text: &str,
        options: TokenizerEncodeOptions,
    ) -> ModelResult<TokenSequence> {
        let encoding = self
            .inner
            .encode(text, options.add_special_tokens())
            .map_err(|_| ModelError::InvalidConfig("tokenizer backend failed to encode text"))?;

        Ok(TokenSequence::new(encoding.get_ids().to_vec()))
    }

    fn decode_with_options(
        &self,
        tokens: &[TokenId],
        options: TokenizerDecodeOptions,
    ) -> ModelResult<String> {
        self.inner
            .decode(tokens, options.skip_special_tokens_enabled())
            .map_err(|_| ModelError::InvalidConfig("tokenizer backend failed to decode tokens"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterLoadedModelMetadata {
    pub kind: AdapterKind,
    pub model: AdapterTargetPlaceholder,
    pub tokenizer: AdapterTokenizerPlaceholder,
    pub summary: ModelAssetSummary,
    pub weights: Vec<AdapterWeightFilePreflight>,
}

impl AdapterLoadedModelMetadata {
    pub fn from_preflight(preflight: AdapterModelPreflight) -> Self {
        Self::from_parts(preflight, Vec::new())
    }

    pub fn from_weight_preflight(preflight: AdapterModelWeightPreflight) -> Self {
        Self::from_parts(preflight.model, preflight.weights)
    }

    fn from_parts(
        preflight: AdapterModelPreflight,
        weights: Vec<AdapterWeightFilePreflight>,
    ) -> Self {
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
            weights,
        }
    }

    #[cfg(feature = "tokenizers")]
    pub fn with_json_tokenizer(
        &self,
        tokenizer_file: &Path,
    ) -> ModelResult<LoadedModel<AdapterTargetPlaceholder, AdapterJsonTokenizer>> {
        Ok(LoadedModel::new(
            self.model.clone(),
            AdapterJsonTokenizer::from_file(tokenizer_file)?,
        ))
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

    pub fn from_weight_preflight(preflight: AdapterLoadWeightPreflight) -> Self {
        Self {
            target: AdapterLoadedModelMetadata::from_weight_preflight(preflight.target),
            draft: preflight
                .draft
                .map(AdapterLoadedModelMetadata::from_weight_preflight),
        }
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }

    #[cfg(feature = "tokenizers")]
    pub fn with_json_tokenizers(
        &self,
        request: &ModelLoadRequest,
    ) -> ModelResult<
        LoadedModelBundle<
            AdapterTargetPlaceholder,
            AdapterJsonTokenizer,
            AdapterTargetPlaceholder,
            AdapterJsonTokenizer,
        >,
    > {
        let target = self
            .target
            .with_json_tokenizer(&request.target.tokenizer_file)?;

        let draft = match (&self.draft, &request.draft) {
            (Some(metadata), Some(assets)) => {
                Some(metadata.with_json_tokenizer(&assets.tokenizer_file)?)
            }
            (None, None) => None,
            _ => {
                return Err(ModelError::InvalidConfig(
                    "loaded metadata and request draft shape must match",
                ));
            }
        };

        Ok(LoadedModelBundle { target, draft })
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

    pub fn load_target_metadata_with_weight_preflight(
        self,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterLoadedMetadataBundle> {
        self.preflight_target_only_with_weight_metadata(request)
            .map(AdapterLoadedMetadataBundle::from_weight_preflight)
    }

    pub fn load_with_draft_metadata_with_weight_preflight(
        self,
        draft_kind: AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterLoadedMetadataBundle> {
        self.preflight_with_draft_weight_metadata(draft_kind, request)
            .map(AdapterLoadedMetadataBundle::from_weight_preflight)
    }

    #[cfg(feature = "tokenizers")]
    pub fn load_target_with_json_tokenizer(
        self,
        request: &ModelLoadRequest,
    ) -> ModelResult<
        LoadedModelBundle<
            AdapterTargetPlaceholder,
            AdapterJsonTokenizer,
            AdapterTargetPlaceholder,
            AdapterJsonTokenizer,
        >,
    > {
        self.load_target_metadata(request)?
            .with_json_tokenizers(request)
    }

    #[cfg(feature = "tokenizers")]
    pub fn load_with_draft_json_tokenizers(
        self,
        draft_kind: AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<
        LoadedModelBundle<
            AdapterTargetPlaceholder,
            AdapterJsonTokenizer,
            AdapterTargetPlaceholder,
            AdapterJsonTokenizer,
        >,
    > {
        self.load_with_draft_metadata(draft_kind, request)?
            .with_json_tokenizers(request)
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
        gguf_parse::test_gguf_bytes,
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

        fn write_gguf(&self) {
            write(
                &self.weights,
                test_gguf_bytes(Some("llama"), "token_embd.weight", &[4096, 32000]),
            )
            .expect("gguf should be written");
        }

        #[cfg(feature = "safetensors")]
        fn write_safetensors(&self) {
            let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
            let mut bytes = Vec::new();
            bytes.extend((header.len() as u64).to_le_bytes());
            bytes.extend(header);
            bytes.extend([0_u8; 8]);
            write(&self.weights, bytes).expect("safetensors should be written");
        }

        #[cfg(feature = "tokenizers")]
        fn write_word_level_tokenizer(&self) {
            write(
                &self.tokenizer,
                r#"{
                    "version": "1.0",
                    "truncation": null,
                    "padding": null,
                    "added_tokens": [],
                    "normalizer": null,
                    "pre_tokenizer": {
                        "type": "Whitespace"
                    },
                    "post_processor": null,
                    "decoder": null,
                    "model": {
                        "type": "WordLevel",
                        "vocab": {
                            "[UNK]": 0,
                            "hello": 1,
                            "world": 2
                        },
                        "unk_token": "[UNK]"
                    }
                }"#,
            )
            .expect("tokenizer should be written");
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

    #[test]
    fn loader_shell_returns_metadata_after_weight_preflight() {
        let target = TempAssets::new("target-weight", "model.safetensors");
        #[cfg(feature = "safetensors")]
        target.write_safetensors();
        let request = ModelLoadRequest::target_only(target.paths());

        let loaded = AdapterLoaderShell::new(AdapterKind::Candle)
            .load_target_metadata_with_weight_preflight(&request)
            .expect("metadata should load after weight preflight");

        assert_eq!(loaded.target.kind, AdapterKind::Candle);
        assert_eq!(loaded.target.weights.len(), 1);
        assert_eq!(loaded.target.weights[0].format, WeightFormat::SafeTensors);

        #[cfg(feature = "safetensors")]
        assert_eq!(
            loaded.target.weights[0]
                .safetensors
                .as_ref()
                .expect("safetensors metadata")
                .tensor_count(),
            1
        );
    }

    #[test]
    fn loader_shell_returns_draft_metadata_after_weight_preflight() {
        let target = TempAssets::new("target-draft-weight", "model.safetensors");
        let draft = TempAssets::new("draft-weight", "model.gguf");
        draft.write_gguf();
        #[cfg(feature = "safetensors")]
        target.write_safetensors();
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());

        let loaded = AdapterLoaderShell::new(AdapterKind::Candle)
            .load_with_draft_metadata_with_weight_preflight(AdapterKind::Gguf, &request)
            .expect("metadata should load after weight preflight");
        let loaded_draft = loaded.draft.expect("draft metadata should exist");

        assert_eq!(loaded.target.weights[0].format, WeightFormat::SafeTensors);
        assert_eq!(loaded_draft.weights[0].format, WeightFormat::Gguf);
        assert_eq!(
            loaded_draft.weights[0]
                .gguf
                .as_ref()
                .expect("gguf metadata")
                .tensor_count,
            1
        );
    }

    #[cfg(feature = "tokenizers")]
    #[test]
    fn adapter_json_tokenizer_loads_hf_tokenizer_files() {
        let temp = TempAssets::new("json-tokenizer", "model.safetensors");
        temp.write_word_level_tokenizer();

        let tokenizer = crate::adapter_loaded::AdapterJsonTokenizer::from_file(&temp.tokenizer)
            .expect("tokenizer should load");
        let encoded = tokenizer.encode("hello world").expect("text should encode");

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(encoded.as_slice(), &[1, 2]);
        assert_eq!(
            tokenizer
                .encode_with_options(
                    "hello world",
                    crate::model::TokenizerEncodeOptions::with_special_tokens(),
                )
                .expect("text should encode")
                .as_slice(),
            &[1, 2]
        );
        assert_eq!(
            tokenizer.decode(encoded.as_slice()),
            Ok(String::from("hello world"))
        );
        assert_eq!(
            tokenizer.decode_with_options(
                encoded.as_slice(),
                crate::model::TokenizerDecodeOptions::preserve_special_tokens(),
            ),
            Ok(String::from("hello world"))
        );
    }

    #[cfg(feature = "tokenizers")]
    #[test]
    fn loader_shell_can_return_json_tokenizer_runtime() {
        let target = TempAssets::new("runtime-target", "model.safetensors");
        target.write_word_level_tokenizer();
        let request = ModelLoadRequest::target_only(target.paths());

        let loaded = AdapterLoaderShell::new(AdapterKind::Candle)
            .load_target_with_json_tokenizer(&request)
            .expect("runtime tokenizer should load");
        let encoded = loaded
            .target
            .tokenizer
            .encode("hello world")
            .expect("text should encode");

        assert_eq!(loaded.target.model.model_type(), Some("llama"));
        assert_eq!(encoded.as_slice(), &[1, 2]);
        assert!(loaded.draft.is_none());
    }

    #[cfg(feature = "tokenizers")]
    #[test]
    fn loader_shell_can_return_target_and_draft_json_tokenizer_runtime() {
        let target = TempAssets::new("runtime-target-draft", "model.safetensors");
        let draft = TempAssets::new("runtime-draft", "model.gguf");
        target.write_word_level_tokenizer();
        draft.write_word_level_tokenizer();
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());

        let loaded = AdapterLoaderShell::new(AdapterKind::Candle)
            .load_with_draft_json_tokenizers(AdapterKind::Gguf, &request)
            .expect("runtime tokenizers should load");

        assert_eq!(loaded.target.tokenizer.vocab_size(), 3);
        assert_eq!(
            loaded
                .draft
                .expect("draft runtime should load")
                .tokenizer
                .vocab_size(),
            3
        );
    }
}
