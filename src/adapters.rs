use crate::{
    asset_files::validate_model_asset_files,
    config::{ModelAssetSummary, read_model_asset_summary},
    loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat},
    model::{ModelError, ModelResult},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    Candle,
    Gguf,
}

impl AdapterKind {
    pub fn expected_weight_format(self) -> WeightFormat {
        match self {
            Self::Candle => WeightFormat::SafeTensors,
            Self::Gguf => WeightFormat::Gguf,
        }
    }

    pub fn validate_assets(self, assets: &ModelAssetPaths) -> ModelResult<()> {
        let actual = assets.weight_format()?;
        if actual != self.expected_weight_format() {
            return Err(ModelError::InvalidConfig(
                "adapter does not support this weight format",
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterModelPlan {
    pub kind: AdapterKind,
    pub weight_format: WeightFormat,
    pub weight_files: usize,
}

impl AdapterModelPlan {
    pub fn new(kind: AdapterKind, assets: &ModelAssetPaths) -> ModelResult<Self> {
        kind.validate_assets(assets)?;

        Ok(Self {
            kind,
            weight_format: assets.weight_format()?,
            weight_files: assets.weight_files.len(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterLoadPlan {
    pub target: AdapterModelPlan,
    pub draft: Option<AdapterModelPlan>,
}

impl AdapterLoadPlan {
    pub fn target_only(target_kind: AdapterKind, request: &ModelLoadRequest) -> ModelResult<Self> {
        request.validate()?;

        if request.has_draft() {
            return Err(ModelError::InvalidConfig(
                "target-only adapter plan must not include draft assets",
            ));
        }

        Ok(Self {
            target: AdapterModelPlan::new(target_kind, &request.target)?,
            draft: None,
        })
    }

    pub fn with_draft(
        target_kind: AdapterKind,
        draft_kind: AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<Self> {
        request.validate()?;

        let draft = request.draft.as_ref().ok_or(ModelError::InvalidConfig(
            "draft adapter plan requires draft assets",
        ))?;

        Ok(Self {
            target: AdapterModelPlan::new(target_kind, &request.target)?,
            draft: Some(AdapterModelPlan::new(draft_kind, draft)?),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterModelPreflight {
    pub plan: AdapterModelPlan,
    pub summary: ModelAssetSummary,
}

impl AdapterModelPreflight {
    pub fn new(kind: AdapterKind, assets: &ModelAssetPaths) -> ModelResult<Self> {
        validate_model_asset_files(assets)?;

        Ok(Self {
            plan: AdapterModelPlan::new(kind, assets)?,
            summary: read_model_asset_summary(assets)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterLoadPreflight {
    pub target: AdapterModelPreflight,
    pub draft: Option<AdapterModelPreflight>,
}

impl AdapterLoadPreflight {
    pub fn target_only(target_kind: AdapterKind, request: &ModelLoadRequest) -> ModelResult<Self> {
        if request.has_draft() {
            return Err(ModelError::InvalidConfig(
                "target-only adapter preflight must not include draft assets",
            ));
        }

        Ok(Self {
            target: AdapterModelPreflight::new(target_kind, &request.target)?,
            draft: None,
        })
    }

    pub fn with_draft(
        target_kind: AdapterKind,
        draft_kind: AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<Self> {
        let draft = request.draft.as_ref().ok_or(ModelError::InvalidConfig(
            "draft adapter preflight requires draft assets",
        ))?;

        Ok(Self {
            target: AdapterModelPreflight::new(target_kind, &request.target)?,
            draft: Some(AdapterModelPreflight::new(draft_kind, draft)?),
        })
    }
}

#[cfg(feature = "candle")]
pub mod candle {
    use crate::adapters::AdapterKind;

    pub const KIND: AdapterKind = AdapterKind::Candle;
}

#[cfg(feature = "gguf")]
pub mod gguf {
    use crate::adapters::AdapterKind;

    pub const KIND: AdapterKind = AdapterKind::Gguf;
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{File, create_dir_all, remove_dir_all, write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        adapters::{AdapterKind, AdapterLoadPlan, AdapterLoadPreflight, AdapterModelPlan},
        loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat},
        model::{ModelError, ModelResult},
    };

    fn assets(weight_file: &str) -> ModelResult<ModelAssetPaths> {
        ModelAssetPaths::new(
            "/models/qwen/config.json",
            "/models/qwen/tokenizer.json",
            vec![PathBuf::from(weight_file)],
        )
    }

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
                "speclative-diffusion-adapter-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");
            let config = root.join("config.json");
            let tokenizer = root.join("tokenizer.json");
            let weights = root.join(weight_name);

            write(
                &config,
                r#"{
                    "model_type": "llama",
                    "vocab_size": 32000,
                    "hidden_size": 4096,
                    "num_hidden_layers": 32
                }"#,
            )
            .expect("config should be written");
            write(
                &tokenizer,
                r#"{
                    "model": {
                        "type": "BPE",
                        "vocab": {
                            "hello": 0,
                            "world": 1
                        }
                    }
                }"#,
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
    fn validates_candle_safetensor_assets() {
        let paths = assets("/models/qwen/model.safetensors").expect("valid assets");

        assert_eq!(AdapterKind::Candle.validate_assets(&paths), Ok(()));
    }

    #[test]
    fn validates_gguf_assets() {
        let paths = assets("/models/qwen/model.gguf").expect("valid assets");

        assert_eq!(AdapterKind::Gguf.validate_assets(&paths), Ok(()));
    }

    #[test]
    fn rejects_adapter_weight_format_mismatch() {
        let paths = assets("/models/qwen/model.gguf").expect("valid assets");

        assert_eq!(
            AdapterKind::Candle.validate_assets(&paths),
            Err(ModelError::InvalidConfig(
                "adapter does not support this weight format"
            ))
        );
    }

    #[test]
    fn builds_adapter_model_plans() {
        let paths = assets("/models/qwen/model.safetensors").expect("valid assets");

        assert_eq!(
            AdapterModelPlan::new(AdapterKind::Candle, &paths),
            Ok(AdapterModelPlan {
                kind: AdapterKind::Candle,
                weight_format: WeightFormat::SafeTensors,
                weight_files: 1,
            })
        );
    }

    #[test]
    fn builds_target_only_adapter_load_plans() {
        let request =
            ModelLoadRequest::target_only(assets("/models/qwen/model.safetensors").unwrap());

        let plan = AdapterLoadPlan::target_only(AdapterKind::Candle, &request).unwrap();

        assert_eq!(plan.target.kind, AdapterKind::Candle);
        assert_eq!(plan.target.weight_format, WeightFormat::SafeTensors);
        assert_eq!(plan.draft, None);
    }

    #[test]
    fn builds_target_and_draft_adapter_load_plans() {
        let request = ModelLoadRequest::with_draft(
            assets("/models/qwen/model.safetensors").unwrap(),
            assets("/models/qwen-draft/model.gguf").unwrap(),
        );

        let plan =
            AdapterLoadPlan::with_draft(AdapterKind::Candle, AdapterKind::Gguf, &request).unwrap();

        assert_eq!(plan.target.kind, AdapterKind::Candle);
        assert_eq!(plan.draft.expect("draft plan").kind, AdapterKind::Gguf);
    }

    #[test]
    fn rejects_adapter_load_plan_shape_mismatches() {
        let target_only =
            ModelLoadRequest::target_only(assets("/models/qwen/model.safetensors").unwrap());
        let with_draft = ModelLoadRequest::with_draft(
            assets("/models/qwen/model.safetensors").unwrap(),
            assets("/models/qwen-draft/model.gguf").unwrap(),
        );

        assert_eq!(
            AdapterLoadPlan::with_draft(AdapterKind::Candle, AdapterKind::Gguf, &target_only),
            Err(ModelError::InvalidConfig(
                "draft adapter plan requires draft assets"
            ))
        );
        assert_eq!(
            AdapterLoadPlan::target_only(AdapterKind::Candle, &with_draft),
            Err(ModelError::InvalidConfig(
                "target-only adapter plan must not include draft assets"
            ))
        );
    }

    #[test]
    fn builds_adapter_load_preflight_with_asset_summaries() {
        let target = TempAssets::new("target", "model.safetensors");
        let draft = TempAssets::new("draft", "model.gguf");
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());

        let preflight =
            AdapterLoadPreflight::with_draft(AdapterKind::Candle, AdapterKind::Gguf, &request)
                .expect("preflight should pass");

        assert_eq!(preflight.target.plan.kind, AdapterKind::Candle);
        assert_eq!(
            preflight.target.summary.model.model_type.as_deref(),
            Some("llama")
        );
        assert_eq!(
            preflight
                .draft
                .expect("draft preflight")
                .summary
                .weight_format,
            WeightFormat::Gguf
        );
    }

    #[test]
    fn rejects_adapter_preflight_missing_files_and_shape_mismatches() {
        let target = TempAssets::new("target-only", "model.safetensors");
        let draft = TempAssets::new("draft-shape", "model.gguf");
        let target_only = ModelLoadRequest::target_only(target.paths());
        let with_draft = ModelLoadRequest::with_draft(target.paths(), draft.paths());
        let missing = ModelLoadRequest::target_only(
            ModelAssetPaths::new(
                "/missing/config.json",
                "/missing/tokenizer.json",
                vec![PathBuf::from("/missing/model.safetensors")],
            )
            .expect("missing paths still have valid extensions"),
        );

        assert_eq!(
            AdapterLoadPreflight::with_draft(AdapterKind::Candle, AdapterKind::Gguf, &target_only),
            Err(ModelError::InvalidConfig(
                "draft adapter preflight requires draft assets"
            ))
        );
        assert_eq!(
            AdapterLoadPreflight::target_only(AdapterKind::Candle, &with_draft),
            Err(ModelError::InvalidConfig(
                "target-only adapter preflight must not include draft assets"
            ))
        );
        assert_eq!(
            AdapterLoadPreflight::target_only(AdapterKind::Candle, &missing),
            Err(ModelError::InvalidConfig("config file must exist"))
        );
    }
}
