#[cfg(feature = "candle")]
pub mod candle {
    use std::path::PathBuf;

    use crate::{
        adapter_runtime_plan::{AdapterRuntimePlanBundle, AdapterTargetRuntimePlan},
        adapter_runtime_target::AdapterRuntimeTargetPlaceholder,
        adapter_runtime_weight_check::validate_candle_runtime_weights,
        adapters::{AdapterKind, AdapterLoaderShell},
        loading::ModelLoadRequest,
        model::{ModelError, ModelResult, TargetModel, TokenId},
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct CandleRuntimeTarget {
        inner: AdapterRuntimeTargetPlaceholder,
        weight_paths: Vec<PathBuf>,
    }

    impl CandleRuntimeTarget {
        pub fn from_runtime_plan(plan: &AdapterTargetRuntimePlan) -> ModelResult<Self> {
            if plan.kind != AdapterKind::Candle {
                return Err(ModelError::InvalidConfig(
                    "candle backend requires candle runtime plan",
                ));
            }

            let inner = AdapterRuntimeTargetPlaceholder::from_runtime_plan(plan)?;
            validate_candle_runtime_weights(plan)?;

            Ok(Self {
                inner,
                weight_paths: weight_paths(plan),
            })
        }

        pub fn model_type(&self) -> &str {
            self.inner.model_type()
        }

        pub fn weight_file_count(&self) -> usize {
            self.inner.weight_file_count()
        }

        pub fn weight_paths(&self) -> &[PathBuf] {
            &self.weight_paths
        }
    }

    impl TargetModel for CandleRuntimeTarget {
        fn vocab_size(&self) -> usize {
            TargetModel::vocab_size(&self.inner)
        }

        fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
            self.inner.validate_prefix(prefix)?;
            Err(ModelError::InvalidConfig(
                "candle backend cannot produce logits yet",
            ))
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct CandleRuntimeTargetBundle {
        pub target: CandleRuntimeTarget,
        pub draft: Option<CandleRuntimeTarget>,
    }

    impl CandleRuntimeTargetBundle {
        pub fn from_runtime_plan_bundle(bundle: &AdapterRuntimePlanBundle) -> ModelResult<Self> {
            Ok(Self {
                target: CandleRuntimeTarget::from_runtime_plan(&bundle.target)?,
                draft: bundle
                    .draft
                    .as_ref()
                    .map(CandleRuntimeTarget::from_runtime_plan)
                    .transpose()?,
            })
        }

        pub fn has_draft(&self) -> bool {
            self.draft.is_some()
        }
    }

    impl AdapterLoaderShell {
        pub fn load_candle_runtime_backend_bundle(
            self,
            request: &ModelLoadRequest,
        ) -> ModelResult<CandleRuntimeTargetBundle> {
            let plan = self.load_target_runtime_plan_bundle(request)?;
            CandleRuntimeTargetBundle::from_runtime_plan_bundle(&plan)
        }

        pub fn load_with_draft_candle_runtime_backend_bundle(
            self,
            request: &ModelLoadRequest,
        ) -> ModelResult<CandleRuntimeTargetBundle> {
            let plan = self.load_with_draft_runtime_plan_bundle(AdapterKind::Candle, request)?;
            CandleRuntimeTargetBundle::from_runtime_plan_bundle(&plan)
        }
    }

    fn weight_paths(plan: &AdapterTargetRuntimePlan) -> Vec<PathBuf> {
        plan.weights
            .iter()
            .map(|weight| weight.path.clone())
            .collect()
    }

    pub fn runtime_target(plan: &AdapterTargetRuntimePlan) -> ModelResult<CandleRuntimeTarget> {
        CandleRuntimeTarget::from_runtime_plan(plan)
    }
}

#[cfg(feature = "gguf")]
pub mod gguf {
    use std::path::PathBuf;

    use crate::{
        adapter_runtime_plan::{AdapterRuntimePlanBundle, AdapterTargetRuntimePlan},
        adapter_runtime_target::AdapterRuntimeTargetPlaceholder,
        adapter_runtime_weight_check::validate_gguf_runtime_weights,
        adapters::{AdapterKind, AdapterLoaderShell},
        loading::ModelLoadRequest,
        model::{ModelError, ModelResult, TargetModel, TokenId},
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct GgufRuntimeTarget {
        inner: AdapterRuntimeTargetPlaceholder,
        weight_paths: Vec<PathBuf>,
    }

    impl GgufRuntimeTarget {
        pub fn from_runtime_plan(plan: &AdapterTargetRuntimePlan) -> ModelResult<Self> {
            if plan.kind != AdapterKind::Gguf {
                return Err(ModelError::InvalidConfig(
                    "gguf backend requires gguf runtime plan",
                ));
            }

            let inner = AdapterRuntimeTargetPlaceholder::from_runtime_plan(plan)?;
            validate_gguf_runtime_weights(plan)?;

            Ok(Self {
                inner,
                weight_paths: weight_paths(plan),
            })
        }

        pub fn model_type(&self) -> &str {
            self.inner.model_type()
        }

        pub fn weight_file_count(&self) -> usize {
            self.inner.weight_file_count()
        }

        pub fn weight_paths(&self) -> &[PathBuf] {
            &self.weight_paths
        }
    }

    impl TargetModel for GgufRuntimeTarget {
        fn vocab_size(&self) -> usize {
            TargetModel::vocab_size(&self.inner)
        }

        fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
            self.inner.validate_prefix(prefix)?;
            Err(ModelError::InvalidConfig(
                "gguf backend cannot produce logits yet",
            ))
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct GgufRuntimeTargetBundle {
        pub target: GgufRuntimeTarget,
        pub draft: Option<GgufRuntimeTarget>,
    }

    impl GgufRuntimeTargetBundle {
        pub fn from_runtime_plan_bundle(bundle: &AdapterRuntimePlanBundle) -> ModelResult<Self> {
            Ok(Self {
                target: GgufRuntimeTarget::from_runtime_plan(&bundle.target)?,
                draft: bundle
                    .draft
                    .as_ref()
                    .map(GgufRuntimeTarget::from_runtime_plan)
                    .transpose()?,
            })
        }

        pub fn has_draft(&self) -> bool {
            self.draft.is_some()
        }
    }

    impl AdapterLoaderShell {
        pub fn load_gguf_runtime_backend_bundle(
            self,
            request: &ModelLoadRequest,
        ) -> ModelResult<GgufRuntimeTargetBundle> {
            let plan = self.load_target_runtime_plan_bundle(request)?;
            GgufRuntimeTargetBundle::from_runtime_plan_bundle(&plan)
        }

        pub fn load_with_draft_gguf_runtime_backend_bundle(
            self,
            request: &ModelLoadRequest,
        ) -> ModelResult<GgufRuntimeTargetBundle> {
            let plan = self.load_with_draft_runtime_plan_bundle(AdapterKind::Gguf, request)?;
            GgufRuntimeTargetBundle::from_runtime_plan_bundle(&plan)
        }
    }

    fn weight_paths(plan: &AdapterTargetRuntimePlan) -> Vec<PathBuf> {
        plan.weights
            .iter()
            .map(|weight| weight.path.clone())
            .collect()
    }

    pub fn runtime_target(plan: &AdapterTargetRuntimePlan) -> ModelResult<GgufRuntimeTarget> {
        GgufRuntimeTarget::from_runtime_plan(plan)
    }
}

#[cfg(test)]
#[cfg(any(feature = "candle", feature = "gguf"))]
mod tests {
    use std::{
        fs::{create_dir_all, remove_dir_all, write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        adapter_runtime_plan::AdapterTargetRuntimePlan,
        adapters::{AdapterKind, AdapterLoaderShell},
        loading::{ModelAssetPaths, ModelLoadRequest},
        model::{ModelError, TargetModel},
    };

    struct TempAssets {
        root: PathBuf,
        config: PathBuf,
        tokenizer: PathBuf,
        weights: PathBuf,
    }

    impl TempAssets {
        fn gguf(name: &str, config: &str) -> Self {
            Self::new(name, "model.gguf", config, gguf_bytes(12))
        }

        fn safetensors(name: &str, config: &str, shape: &[usize]) -> Self {
            Self::new(
                name,
                "model.safetensors",
                config,
                safetensors_bytes("model.embed_tokens.weight", shape),
            )
        }

        fn new(name: &str, weight_name: &str, config: &str, weights: Vec<u8>) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-runtime-backend-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");

            let config_path = root.join("config.json");
            let tokenizer = root.join("tokenizer.json");
            let weights_path = root.join(weight_name);

            write(&config_path, config).expect("config should be written");
            write(&tokenizer, r#"{"model":{"type":"BPE"}}"#).expect("tokenizer should be written");
            write(&weights_path, weights).expect("weights should be written");

            Self {
                root,
                config: config_path,
                tokenizer,
                weights: weights_path,
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

        fn runtime_plan(&self, kind: AdapterKind) -> AdapterTargetRuntimePlan {
            let request = ModelLoadRequest::target_only(self.paths());
            AdapterLoaderShell::new(kind)
                .load_target_runtime_plan_bundle(&request)
                .expect("runtime plan bundle should be built")
                .target
        }
    }

    impl Drop for TempAssets {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.root);
        }
    }

    fn valid_config() -> &'static str {
        r#"{
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32
        }"#
    }

    fn incomplete_config() -> &'static str {
        r#"{"model_type":"llama","vocab_size":32000}"#
    }

    fn gguf_bytes(tensor_count: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(tensor_count.to_le_bytes());
        bytes.extend(4_u64.to_le_bytes());
        bytes
    }

    fn safetensors_bytes(tensor_name: &str, shape: &[usize]) -> Vec<u8> {
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{tensor_name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[0,8]}}}}"#
        );
        let mut bytes = Vec::new();
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend([0_u8; 8]);
        bytes
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn gguf_backend_builds_from_runtime_plan() {
        let assets = TempAssets::gguf("gguf-target", valid_config());
        let plan = assets.runtime_plan(AdapterKind::Gguf);

        let target =
            crate::adapter_runtime_backend::gguf::GgufRuntimeTarget::from_runtime_plan(&plan)
                .expect("gguf backend should build");

        assert_eq!(target.model_type(), "llama");
        assert_eq!(TargetModel::vocab_size(&target), 32000);
        assert_eq!(target.weight_file_count(), 1);
        assert!(target.weight_paths().contains(&assets.weights));
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn gguf_backend_rejects_wrong_kind_and_incomplete_shape() {
        let candle_assets =
            TempAssets::safetensors("gguf-wrong-kind", valid_config(), &[32000, 4096]);
        let incomplete_assets = TempAssets::gguf("gguf-incomplete", incomplete_config());
        let candle_plan = candle_assets.runtime_plan(AdapterKind::Candle);
        let incomplete_plan = incomplete_assets.runtime_plan(AdapterKind::Gguf);

        assert_eq!(
            crate::adapter_runtime_backend::gguf::GgufRuntimeTarget::from_runtime_plan(
                &candle_plan
            ),
            Err(ModelError::InvalidConfig(
                "gguf backend requires gguf runtime plan"
            ))
        );
        assert_eq!(
            crate::adapter_runtime_backend::gguf::GgufRuntimeTarget::from_runtime_plan(
                &incomplete_plan
            ),
            Err(ModelError::InvalidConfig("hidden size is required"))
        );
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn gguf_backend_rejects_empty_weight_headers() {
        let assets = TempAssets::new(
            "gguf-empty",
            "model.gguf",
            valid_config(),
            gguf_bytes(0),
        );
        let plan = assets.runtime_plan(AdapterKind::Gguf);

        assert_eq!(
            crate::adapter_runtime_backend::gguf::GgufRuntimeTarget::from_runtime_plan(&plan),
            Err(ModelError::InvalidConfig(
                "gguf backend weight file must contain tensors"
            ))
        );
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn gguf_loader_builds_target_and_draft_backend_bundle() {
        let target = TempAssets::gguf("gguf-loader-target", valid_config());
        let draft = TempAssets::gguf("gguf-loader-draft", valid_config());
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());

        let bundle = crate::adapters::gguf::loader()
            .load_with_draft_gguf_runtime_backend_bundle(&request)
            .expect("gguf backend bundle should build");

        assert!(bundle.has_draft());
        assert_eq!(bundle.target.model_type(), "llama");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn candle_backend_builds_from_runtime_plan() {
        let assets = TempAssets::safetensors("candle-target", valid_config(), &[32000, 4096]);
        let plan = assets.runtime_plan(AdapterKind::Candle);

        let target =
            crate::adapter_runtime_backend::candle::CandleRuntimeTarget::from_runtime_plan(&plan)
                .expect("candle backend should build");

        assert_eq!(target.model_type(), "llama");
        assert_eq!(TargetModel::vocab_size(&target), 32000);
        assert_eq!(target.weight_file_count(), 1);
        assert!(target.weight_paths().contains(&assets.weights));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn candle_backend_rejects_wrong_kind_and_incomplete_shape() {
        let gguf_assets = TempAssets::gguf("candle-wrong-kind", valid_config());
        let incomplete_assets =
            TempAssets::safetensors("candle-incomplete", incomplete_config(), &[32000, 4096]);
        let gguf_plan = gguf_assets.runtime_plan(AdapterKind::Gguf);
        let incomplete_plan = incomplete_assets.runtime_plan(AdapterKind::Candle);

        assert_eq!(
            crate::adapter_runtime_backend::candle::CandleRuntimeTarget::from_runtime_plan(
                &gguf_plan
            ),
            Err(ModelError::InvalidConfig(
                "candle backend requires candle runtime plan"
            ))
        );
        assert_eq!(
            crate::adapter_runtime_backend::candle::CandleRuntimeTarget::from_runtime_plan(
                &incomplete_plan
            ),
            Err(ModelError::InvalidConfig("hidden size is required"))
        );
    }

    #[cfg(all(feature = "candle", feature = "safetensors"))]
    #[test]
    fn candle_backend_rejects_embedding_shape_mismatch() {
        let assets = TempAssets::safetensors("candle-shape", valid_config(), &[100, 4096]);
        let plan = assets.runtime_plan(AdapterKind::Candle);

        assert_eq!(
            crate::adapter_runtime_backend::candle::CandleRuntimeTarget::from_runtime_plan(&plan),
            Err(ModelError::InvalidConfig(
                "candle backend weight tensor shape does not match config"
            ))
        );
    }

}
