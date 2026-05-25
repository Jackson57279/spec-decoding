use std::path::PathBuf;

use crate::{
    adapter_loaded::{AdapterLoadedMetadataBundle, AdapterLoadedModelMetadata},
    adapter_weight_preflight::AdapterWeightFilePreflight,
    adapters::{AdapterKind, AdapterLoaderShell},
    loading::{ModelLoadRequest, WeightFormat},
    model::{ModelError, ModelResult},
    weight_metadata::{GgufFileMetadata, SafeTensorsFileMetadata},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterRuntimeSafeTensorsPlan {
    pub tensor_count: usize,
    pub user_metadata_fields: usize,
    pub data_bytes: usize,
}

impl AdapterRuntimeSafeTensorsPlan {
    fn from_metadata(metadata: &SafeTensorsFileMetadata) -> Self {
        Self {
            tensor_count: metadata.tensor_count(),
            user_metadata_fields: metadata.user_metadata_fields,
            data_bytes: metadata.data_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterRuntimeGgufPlan {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub header_bytes: usize,
    pub architecture: Option<String>,
    pub parsed_tensor_count: usize,
}

impl AdapterRuntimeGgufPlan {
    fn from_metadata(metadata: &GgufFileMetadata) -> Self {
        Self {
            version: metadata.version,
            tensor_count: metadata.tensor_count,
            metadata_kv_count: metadata.metadata_kv_count,
            header_bytes: metadata.header_bytes,
            architecture: metadata.architecture.clone(),
            parsed_tensor_count: metadata.tensors.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterRuntimeWeightFilePlan {
    pub path: PathBuf,
    pub format: WeightFormat,
    pub safetensors: Option<AdapterRuntimeSafeTensorsPlan>,
    pub gguf: Option<AdapterRuntimeGgufPlan>,
}

impl AdapterRuntimeWeightFilePlan {
    fn from_preflight(preflight: &AdapterWeightFilePreflight) -> ModelResult<Self> {
        match preflight.format {
            WeightFormat::SafeTensors if preflight.gguf.is_some() => {
                return Err(ModelError::InvalidConfig(
                    "safetensors runtime weight must not include gguf metadata",
                ));
            }
            WeightFormat::Gguf if preflight.safetensors.is_some() => {
                return Err(ModelError::InvalidConfig(
                    "gguf runtime weight must not include safetensors metadata",
                ));
            }
            _ => {}
        }

        Ok(Self {
            path: preflight.path.clone(),
            format: preflight.format,
            safetensors: preflight
                .safetensors
                .as_ref()
                .map(AdapterRuntimeSafeTensorsPlan::from_metadata),
            gguf: preflight
                .gguf
                .as_ref()
                .map(AdapterRuntimeGgufPlan::from_metadata),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterTargetRuntimePlan {
    pub kind: AdapterKind,
    pub model_type: Option<String>,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub tokenizer_model_type: Option<String>,
    pub tokenizer_vocab_size: Option<usize>,
    pub weight_format: WeightFormat,
    pub weights: Vec<AdapterRuntimeWeightFilePlan>,
}

impl AdapterTargetRuntimePlan {
    pub fn from_loaded_metadata(metadata: &AdapterLoadedModelMetadata) -> ModelResult<Self> {
        if metadata.weights.is_empty() {
            return Err(ModelError::InvalidConfig(
                "adapter runtime plan requires weight preflight metadata",
            ));
        }

        let weights = metadata
            .weights
            .iter()
            .map(AdapterRuntimeWeightFilePlan::from_preflight)
            .collect::<ModelResult<Vec<_>>>()?;

        if weights
            .iter()
            .any(|weight| weight.format != metadata.summary.weight_format)
        {
            return Err(ModelError::InvalidConfig(
                "runtime weight format must match model summary",
            ));
        }

        Ok(Self {
            kind: metadata.kind,
            model_type: metadata.summary.model.model_type.clone(),
            vocab_size: metadata.summary.model.vocab_size,
            hidden_size: metadata.summary.model.hidden_size,
            num_hidden_layers: metadata.summary.model.num_hidden_layers,
            tokenizer_model_type: metadata.summary.tokenizer.model_type.clone(),
            tokenizer_vocab_size: metadata.summary.tokenizer.vocab_size,
            weight_format: metadata.summary.weight_format,
            weights,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterRuntimePlanBundle {
    pub target: AdapterTargetRuntimePlan,
    pub draft: Option<AdapterTargetRuntimePlan>,
}

impl AdapterRuntimePlanBundle {
    pub fn from_loaded_metadata(metadata: &AdapterLoadedMetadataBundle) -> ModelResult<Self> {
        Ok(Self {
            target: AdapterTargetRuntimePlan::from_loaded_metadata(&metadata.target)?,
            draft: metadata
                .draft
                .as_ref()
                .map(AdapterTargetRuntimePlan::from_loaded_metadata)
                .transpose()?,
        })
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }
}

impl AdapterLoaderShell {
    pub fn load_target_runtime_plan_bundle(
        self,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterRuntimePlanBundle> {
        let loaded = self.load_target_metadata_with_weight_preflight(request)?;
        AdapterRuntimePlanBundle::from_loaded_metadata(&loaded)
    }

    pub fn load_with_draft_runtime_plan_bundle(
        self,
        draft_kind: AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterRuntimePlanBundle> {
        let loaded = self.load_with_draft_metadata_with_weight_preflight(draft_kind, request)?;
        AdapterRuntimePlanBundle::from_loaded_metadata(&loaded)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{create_dir_all, remove_dir_all, write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        adapter_runtime_plan::{AdapterRuntimePlanBundle, AdapterTargetRuntimePlan},
        adapters::{AdapterKind, AdapterLoaderShell},
        gguf_parse::test_gguf_bytes,
        loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat},
        model::ModelError,
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
                "speclative-diffusion-runtime-plan-{name}-{}-{unique}",
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
                r#"{"model":{"type":"BPE","vocab":{"hello":0,"world":1}}}"#,
            )
            .expect("tokenizer should be written");
            write(
                &weights,
                test_gguf_bytes(Some("llama"), "token_embd.weight", &[4096, 32000]),
            )
            .expect("weights should be written");

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
    fn builds_target_runtime_plan_from_weight_checked_metadata() {
        let target = TempAssets::new("target", "model.gguf");
        let request = ModelLoadRequest::target_only(target.paths());
        let loaded = AdapterLoaderShell::new(AdapterKind::Gguf)
            .load_target_metadata_with_weight_preflight(&request)
            .expect("loaded metadata should be built");

        let plan = AdapterTargetRuntimePlan::from_loaded_metadata(&loaded.target)
            .expect("runtime plan should be built");

        assert_eq!(plan.kind, AdapterKind::Gguf);
        assert_eq!(plan.model_type.as_deref(), Some("llama"));
        assert_eq!(plan.vocab_size, Some(32000));
        assert_eq!(plan.hidden_size, Some(4096));
        assert_eq!(plan.num_hidden_layers, Some(32));
        assert_eq!(plan.tokenizer_model_type.as_deref(), Some("BPE"));
        assert_eq!(plan.tokenizer_vocab_size, Some(2));
        assert_eq!(plan.weight_format, WeightFormat::Gguf);
        assert_eq!(
            plan.weights[0]
                .gguf
                .as_ref()
                .expect("gguf plan")
                .tensor_count,
            1
        );
        assert_eq!(
            plan.weights[0]
                .gguf
                .as_ref()
                .expect("gguf plan")
                .architecture
                .as_deref(),
            Some("llama")
        );
    }

    #[test]
    fn builds_target_and_draft_runtime_plan_bundle() {
        let target = TempAssets::new("target-draft", "model.gguf");
        let draft = TempAssets::new("draft", "draft.gguf");
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());
        let loaded = AdapterLoaderShell::new(AdapterKind::Gguf)
            .load_with_draft_metadata_with_weight_preflight(AdapterKind::Gguf, &request)
            .expect("loaded metadata should be built");

        let bundle = AdapterRuntimePlanBundle::from_loaded_metadata(&loaded)
            .expect("runtime plan bundle should be built");

        assert!(bundle.has_draft());
        assert_eq!(bundle.target.weight_format, WeightFormat::Gguf);
        assert_eq!(
            bundle.draft.expect("draft runtime plan").weights[0]
                .gguf
                .as_ref()
                .expect("gguf metadata")
                .metadata_kv_count,
            1
        );
    }

    #[test]
    fn loader_shell_builds_target_runtime_plan_bundle() {
        let target = TempAssets::new("shell-target", "model.gguf");
        let request = ModelLoadRequest::target_only(target.paths());

        let bundle = AdapterLoaderShell::new(AdapterKind::Gguf)
            .load_target_runtime_plan_bundle(&request)
            .expect("runtime plan bundle should be built");

        assert!(!bundle.has_draft());
        assert_eq!(bundle.target.kind, AdapterKind::Gguf);
        assert_eq!(bundle.target.model_type.as_deref(), Some("llama"));
        assert_eq!(
            bundle.target.weights[0]
                .gguf
                .as_ref()
                .expect("gguf metadata")
                .tensor_count,
            1
        );
    }

    #[test]
    fn loader_shell_builds_target_and_draft_runtime_plan_bundle() {
        let target = TempAssets::new("shell-target-draft", "model.gguf");
        let draft = TempAssets::new("shell-draft", "draft.gguf");
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());

        let bundle = AdapterLoaderShell::new(AdapterKind::Gguf)
            .load_with_draft_runtime_plan_bundle(AdapterKind::Gguf, &request)
            .expect("runtime plan bundle should be built");

        assert!(bundle.has_draft());
        assert_eq!(bundle.target.weight_format, WeightFormat::Gguf);
        assert_eq!(
            bundle.draft.expect("draft runtime plan").weights[0]
                .gguf
                .as_ref()
                .expect("gguf metadata")
                .metadata_kv_count,
            1
        );
    }

    #[test]
    fn rejects_loaded_metadata_without_weight_preflight() {
        let target = TempAssets::new("no-weight-preflight", "model.gguf");
        let request = ModelLoadRequest::target_only(target.paths());
        let loaded = AdapterLoaderShell::new(AdapterKind::Gguf)
            .load_target_metadata(&request)
            .expect("loaded metadata should be built");

        assert_eq!(
            AdapterTargetRuntimePlan::from_loaded_metadata(&loaded.target),
            Err(ModelError::InvalidConfig(
                "adapter runtime plan requires weight preflight metadata"
            ))
        );
    }
}
