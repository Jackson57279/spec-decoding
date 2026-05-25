use crate::{
    adapter_runtime_plan::{AdapterRuntimePlanBundle, AdapterTargetRuntimePlan},
    adapters::{AdapterKind, AdapterLoaderShell},
    loading::{ModelLoadRequest, WeightFormat},
    model::{ModelError, ModelResult, TargetModel, TokenId},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterRuntimeTargetPlaceholder {
    kind: AdapterKind,
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    weight_format: WeightFormat,
    weight_file_count: usize,
}

impl AdapterRuntimeTargetPlaceholder {
    pub fn from_runtime_plan(plan: &AdapterTargetRuntimePlan) -> ModelResult<Self> {
        Self::validate_runtime_plan(plan)?;

        Ok(Self {
            kind: plan.kind,
            model_type: required_text(plan.model_type.as_deref(), "model type is required")?
                .to_owned(),
            vocab_size: required_positive_usize(plan.vocab_size, "vocab size is required")?,
            hidden_size: required_positive_usize(plan.hidden_size, "hidden size is required")?,
            num_hidden_layers: required_positive_usize(
                plan.num_hidden_layers,
                "hidden layer count is required",
            )?,
            weight_format: plan.weight_format,
            weight_file_count: plan.weights.len(),
        })
    }

    pub fn validate_runtime_plan(plan: &AdapterTargetRuntimePlan) -> ModelResult<()> {
        if plan.weights.is_empty() {
            return Err(ModelError::InvalidConfig(
                "runtime target requires at least one weight file",
            ));
        }

        if plan.kind.expected_weight_format() != plan.weight_format {
            return Err(ModelError::InvalidConfig(
                "runtime target weight format must match adapter kind",
            ));
        }

        required_text(plan.model_type.as_deref(), "model type is required")?;
        let vocab_size = required_positive_usize(plan.vocab_size, "vocab size is required")?;
        required_positive_usize(plan.hidden_size, "hidden size is required")?;
        required_positive_usize(plan.num_hidden_layers, "hidden layer count is required")?;

        if matches!(plan.tokenizer_vocab_size, Some(tokenizer_vocab_size) if tokenizer_vocab_size != vocab_size)
        {
            return Err(ModelError::InvalidConfig(
                "model and tokenizer vocab sizes must match",
            ));
        }

        Ok(())
    }

    pub fn kind(&self) -> AdapterKind {
        self.kind
    }

    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    pub fn weight_format(&self) -> WeightFormat {
        self.weight_format
    }

    pub fn weight_file_count(&self) -> usize {
        self.weight_file_count
    }

    pub fn validate_prefix(&self, prefix: &[TokenId]) -> ModelResult<()> {
        for (index, token) in prefix.iter().copied().enumerate() {
            if token as usize >= self.vocab_size {
                return Err(ModelError::TokenOutOfRange { index });
            }
        }

        Ok(())
    }
}

impl TargetModel for AdapterRuntimeTargetPlaceholder {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn logits_for_prefix(&mut self, prefix: &[TokenId]) -> ModelResult<Vec<f32>> {
        self.validate_prefix(prefix)?;

        Err(ModelError::InvalidConfig(
            "runtime target cannot produce logits",
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterRuntimeTargetBundle {
    pub target: AdapterRuntimeTargetPlaceholder,
    pub draft: Option<AdapterRuntimeTargetPlaceholder>,
}

impl AdapterRuntimeTargetBundle {
    pub fn from_runtime_plan_bundle(bundle: &AdapterRuntimePlanBundle) -> ModelResult<Self> {
        Ok(Self {
            target: AdapterRuntimeTargetPlaceholder::from_runtime_plan(&bundle.target)?,
            draft: bundle
                .draft
                .as_ref()
                .map(AdapterRuntimeTargetPlaceholder::from_runtime_plan)
                .transpose()?,
        })
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }
}

impl AdapterLoaderShell {
    pub fn load_target_runtime_target_bundle(
        self,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterRuntimeTargetBundle> {
        let plan = self.load_target_runtime_plan_bundle(request)?;
        AdapterRuntimeTargetBundle::from_runtime_plan_bundle(&plan)
    }

    pub fn load_with_draft_runtime_target_bundle(
        self,
        draft_kind: AdapterKind,
        request: &ModelLoadRequest,
    ) -> ModelResult<AdapterRuntimeTargetBundle> {
        let plan = self.load_with_draft_runtime_plan_bundle(draft_kind, request)?;
        AdapterRuntimeTargetBundle::from_runtime_plan_bundle(&plan)
    }
}

fn required_text<'a>(value: Option<&'a str>, message: &'static str) -> ModelResult<&'a str> {
    match value {
        Some(value) if !value.is_empty() => Ok(value),
        _ => Err(ModelError::InvalidConfig(message)),
    }
}

fn required_positive_usize(value: Option<usize>, message: &'static str) -> ModelResult<usize> {
    match value {
        Some(value) if value > 0 => Ok(value),
        _ => Err(ModelError::InvalidConfig(message)),
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
        adapter_runtime_plan::AdapterTargetRuntimePlan,
        adapter_runtime_target::AdapterRuntimeTargetPlaceholder,
        adapters::{AdapterKind, AdapterLoaderShell},
        gguf_parse::test_gguf_bytes,
        loading::{ModelAssetPaths, ModelLoadRequest, WeightFormat},
        model::{ModelError, TargetBatch, TargetModel, TokenSequence},
    };

    struct TempAssets {
        root: PathBuf,
        config: PathBuf,
        tokenizer: PathBuf,
        weights: PathBuf,
    }

    impl TempAssets {
        fn new(name: &str, config: &str, tokenizer: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-runtime-target-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");

            let config_path = root.join("config.json");
            let tokenizer_path = root.join("tokenizer.json");
            let weights = root.join("model.gguf");

            write(&config_path, config).expect("config should be written");
            write(&tokenizer_path, tokenizer).expect("tokenizer should be written");
            write(
                &weights,
                test_gguf_bytes(Some("llama"), "token_embd.weight", &[4096, 32000]),
            )
            .expect("weights should be written");

            Self {
                root,
                config: config_path,
                tokenizer: tokenizer_path,
                weights,
            }
        }

        fn valid(name: &str) -> Self {
            Self::new(
                name,
                r#"{
                    "model_type": "llama",
                    "vocab_size": 32000,
                    "hidden_size": 4096,
                    "num_hidden_layers": 32
                }"#,
                r#"{"model":{"type":"BPE"}}"#,
            )
        }

        fn paths(&self) -> ModelAssetPaths {
            ModelAssetPaths::new(
                self.config.clone(),
                self.tokenizer.clone(),
                vec![self.weights.clone()],
            )
            .expect("asset paths should be valid")
        }

        fn runtime_plan(&self) -> AdapterTargetRuntimePlan {
            let request = ModelLoadRequest::target_only(self.paths());
            let loaded = AdapterLoaderShell::new(AdapterKind::Gguf)
                .load_target_metadata_with_weight_preflight(&request)
                .expect("loaded metadata should be built");

            AdapterTargetRuntimePlan::from_loaded_metadata(&loaded.target)
                .expect("runtime plan should be built")
        }

    }

    impl Drop for TempAssets {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.root);
        }
    }

    #[test]
    fn builds_runtime_target_from_weight_checked_plan() {
        let assets = TempAssets::valid("target");
        let plan = assets.runtime_plan();

        let target = AdapterRuntimeTargetPlaceholder::from_runtime_plan(&plan)
            .expect("target should be built");

        assert_eq!(target.kind(), AdapterKind::Gguf);
        assert_eq!(target.model_type(), "llama");
        assert_eq!(TargetModel::vocab_size(&target), 32000);
        assert_eq!(target.hidden_size(), 4096);
        assert_eq!(target.num_hidden_layers(), 32);
        assert_eq!(target.weight_format(), WeightFormat::Gguf);
        assert_eq!(target.weight_file_count(), 1);
    }

    #[test]
    fn loader_shell_builds_target_runtime_target_bundle() {
        let assets = TempAssets::valid("shell-target");
        let request = ModelLoadRequest::target_only(assets.paths());

        let bundle = AdapterLoaderShell::new(AdapterKind::Gguf)
            .load_target_runtime_target_bundle(&request)
            .expect("runtime target bundle should be built");

        assert!(!bundle.has_draft());
        assert_eq!(bundle.target.model_type(), "llama");
        assert_eq!(TargetModel::vocab_size(&bundle.target), 32000);
        assert_eq!(bundle.target.weight_file_count(), 1);
    }

    #[test]
    fn runtime_target_rejects_missing_architecture_fields() {
        let assets = TempAssets::new(
            "missing-arch",
            r#"{"model_type":"llama","vocab_size":32000}"#,
            r#"{"model":{"type":"BPE"}}"#,
        );
        let plan = assets.runtime_plan();

        assert_eq!(
            AdapterRuntimeTargetPlaceholder::from_runtime_plan(&plan),
            Err(ModelError::InvalidConfig("hidden size is required"))
        );
    }

    #[test]
    fn loader_shell_runtime_target_bundle_rejects_incomplete_config() {
        let assets = TempAssets::new(
            "shell-missing-arch",
            r#"{"model_type":"llama","vocab_size":32000}"#,
            r#"{"model":{"type":"BPE"}}"#,
        );
        let request = ModelLoadRequest::target_only(assets.paths());

        assert_eq!(
            AdapterLoaderShell::new(AdapterKind::Gguf).load_target_runtime_target_bundle(&request),
            Err(ModelError::InvalidConfig("hidden size is required"))
        );
    }

    #[test]
    fn runtime_target_rejects_zero_or_missing_vocab_size() {
        let zero_vocab = TempAssets::new(
            "zero-vocab",
            r#"{
                "model_type": "llama",
                "vocab_size": 0,
                "hidden_size": 4096,
                "num_hidden_layers": 32
            }"#,
            r#"{"model":{"type":"BPE"}}"#,
        );
        let missing_vocab = TempAssets::new(
            "missing-vocab",
            r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32
            }"#,
            r#"{"model":{"type":"BPE"}}"#,
        );

        assert_eq!(
            AdapterRuntimeTargetPlaceholder::from_runtime_plan(&zero_vocab.runtime_plan()),
            Err(ModelError::InvalidConfig("vocab size is required"))
        );
        assert_eq!(
            AdapterRuntimeTargetPlaceholder::from_runtime_plan(&missing_vocab.runtime_plan()),
            Err(ModelError::InvalidConfig("vocab size is required"))
        );
    }

    #[test]
    fn runtime_target_rejects_tokenizer_vocab_mismatch() {
        let assets = TempAssets::new(
            "vocab-mismatch",
            r#"{
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 32
            }"#,
            r#"{"model":{"type":"BPE","vocab":{"hello":0,"world":1}}}"#,
        );

        assert_eq!(
            AdapterRuntimeTargetPlaceholder::from_runtime_plan(&assets.runtime_plan()),
            Err(ModelError::InvalidConfig(
                "model and tokenizer vocab sizes must match"
            ))
        );
    }

    #[test]
    fn runtime_target_rejects_out_of_range_tokens_before_logits() {
        let assets = TempAssets::valid("range");
        let plan = assets.runtime_plan();
        let mut target = AdapterRuntimeTargetPlaceholder::from_runtime_plan(&plan)
            .expect("target should be built");

        assert_eq!(
            target.logits_for_prefix(&[32000]),
            Err(ModelError::TokenOutOfRange { index: 0 })
        );
    }

    #[test]
    fn runtime_target_fails_logits_with_explicit_message() {
        let assets = TempAssets::valid("logits");
        let plan = assets.runtime_plan();
        let mut target = AdapterRuntimeTargetPlaceholder::from_runtime_plan(&plan)
            .expect("target should be built");

        assert_eq!(
            target.logits_for_prefix(&[0, 1]),
            Err(ModelError::InvalidConfig(
                "runtime target cannot produce logits"
            ))
        );
    }

    #[test]
    fn builds_target_and_draft_runtime_target_bundle() {
        let target = TempAssets::valid("bundle-target");
        let draft = TempAssets::valid("bundle-draft");
        let request = ModelLoadRequest::with_draft(target.paths(), draft.paths());
        let target_bundle = AdapterLoaderShell::new(AdapterKind::Gguf)
            .load_with_draft_runtime_target_bundle(AdapterKind::Gguf, &request)
            .expect("target bundle should be built");

        assert!(target_bundle.has_draft());
        assert_eq!(target_bundle.target.model_type(), "llama");
        assert_eq!(
            target_bundle
                .draft
                .expect("draft target should be built")
                .weight_file_count(),
            1
        );
    }

    #[test]
    fn batched_fallback_preserves_logits_failure() {
        let assets = TempAssets::valid("batch");
        let plan = assets.runtime_plan();
        let mut target = AdapterRuntimeTargetPlaceholder::from_runtime_plan(&plan)
            .expect("target should be built");
        let batch =
            TargetBatch::new(vec![TokenSequence::new(vec![0])]).expect("batch should be valid");

        assert_eq!(
            crate::model::BatchedTargetModel::logits_for_prefixes(&mut target, &batch),
            Err(ModelError::InvalidConfig(
                "runtime target cannot produce logits"
            ))
        );
    }
}
