use crate::{
    adapter_runtime_plan::AdapterTargetRuntimePlan,
    model::{ModelError, ModelResult},
    weight_metadata::read_gguf_file_metadata,
};

#[cfg(feature = "safetensors")]
const EMBEDDING_TENSOR_NAMES: &[&str] = &[
    "model.embed_tokens.weight",
    "model.model.embed_tokens.weight",
    "transformer.wte.weight",
];

pub fn validate_gguf_runtime_weights(plan: &AdapterTargetRuntimePlan) -> ModelResult<()> {
    for weight in &plan.weights {
        let metadata = read_gguf_file_metadata(&weight.path)?;
        if metadata.tensor_count == 0 {
            return Err(ModelError::InvalidConfig(
                "gguf backend weight file must contain tensors",
            ));
        }

        if let Some(planned) = &weight.gguf {
            if metadata.tensor_count != planned.tensor_count {
                return Err(ModelError::InvalidConfig(
                    "gguf backend weight metadata changed after preflight",
                ));
            }
        }
    }

    Ok(())
}

#[cfg(feature = "safetensors")]
pub fn validate_candle_runtime_weights(plan: &AdapterTargetRuntimePlan) -> ModelResult<()> {
    let vocab_size = required_usize(plan.vocab_size, "vocab size is required")?;
    let hidden_size = required_usize(plan.hidden_size, "hidden size is required")?;
    let mut saw_tensor = false;
    let mut saw_embedding = false;

    for weight in &plan.weights {
        let metadata = crate::weight_metadata::read_safetensors_file_metadata(&weight.path)?;
        if metadata.tensor_count() == 0 {
            return Err(ModelError::InvalidConfig(
                "candle backend weight file must contain tensors",
            ));
        }
        saw_tensor = true;

        for tensor in &metadata.tensors {
            if EMBEDDING_TENSOR_NAMES.contains(&tensor.name.as_str()) {
                saw_embedding = true;
                if tensor.shape.as_slice() != [vocab_size, hidden_size] {
                    return Err(ModelError::InvalidConfig(
                        "candle backend weight tensor shape does not match config",
                    ));
                }
            }
        }
    }

    if !saw_tensor {
        return Err(ModelError::InvalidConfig(
            "candle backend weight file must contain tensors",
        ));
    }

    if !saw_embedding {
        return Err(ModelError::InvalidConfig(
            "candle backend missing required weight tensor",
        ));
    }

    Ok(())
}

#[cfg(not(feature = "safetensors"))]
pub fn validate_candle_runtime_weights(_plan: &AdapterTargetRuntimePlan) -> ModelResult<()> {
    Err(ModelError::InvalidConfig(
        "candle backend requires safetensors weight metadata",
    ))
}

#[cfg(feature = "safetensors")]
fn required_usize(value: Option<usize>, message: &'static str) -> ModelResult<usize> {
    value.ok_or(ModelError::InvalidConfig(message))
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
        adapter_runtime_weight_check::validate_gguf_runtime_weights,
        adapters::{AdapterKind, AdapterLoaderShell},
        loading::{ModelAssetPaths, ModelLoadRequest},
        model::ModelError,
    };

    #[cfg(feature = "safetensors")]
    use crate::adapter_runtime_weight_check::validate_candle_runtime_weights;

    struct TempAssets {
        root: PathBuf,
        config: PathBuf,
        tokenizer: PathBuf,
        weights: PathBuf,
    }

    impl TempAssets {
        fn new(name: &str, weight_name: &str, config: &str, weights: Vec<u8>) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-runtime-weight-check-{name}-{}-{unique}",
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

        fn plan(&self, kind: AdapterKind) -> AdapterTargetRuntimePlan {
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
            "vocab_size": 2,
            "hidden_size": 4,
            "num_hidden_layers": 32
        }"#
    }

    fn gguf_bytes(tensor_count: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(tensor_count.to_le_bytes());
        bytes.extend(4_u64.to_le_bytes());
        bytes
    }

    #[cfg(feature = "safetensors")]
    fn safetensors_bytes(tensor_name: &str, shape: &[usize]) -> Vec<u8> {
        let data_bytes = shape.iter().product::<usize>() * 4;
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{tensor_name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[0,{data_bytes}]}}}}"#
        );
        let mut bytes = Vec::new();
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend(vec![0_u8; data_bytes]);
        bytes
    }

    #[test]
    fn validates_gguf_weight_headers() {
        let assets = TempAssets::new("gguf-valid", "model.gguf", valid_config(), gguf_bytes(12));

        assert_eq!(
            validate_gguf_runtime_weights(&assets.plan(AdapterKind::Gguf)),
            Ok(())
        );
    }

    #[test]
    fn rejects_empty_gguf_weight_headers() {
        let assets = TempAssets::new("gguf-empty", "model.gguf", valid_config(), gguf_bytes(0));

        assert_eq!(
            validate_gguf_runtime_weights(&assets.plan(AdapterKind::Gguf)),
            Err(ModelError::InvalidConfig(
                "gguf backend weight file must contain tensors"
            ))
        );
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn validates_candle_embedding_shape() {
        let assets = TempAssets::new(
            "candle-valid",
            "model.safetensors",
            valid_config(),
            safetensors_bytes("model.embed_tokens.weight", &[2, 4]),
        );

        assert_eq!(
            validate_candle_runtime_weights(&assets.plan(AdapterKind::Candle)),
            Ok(())
        );
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn rejects_candle_embedding_shape_mismatch() {
        let assets = TempAssets::new(
            "candle-mismatch",
            "model.safetensors",
            valid_config(),
            safetensors_bytes("model.embed_tokens.weight", &[3, 4]),
        );

        assert_eq!(
            validate_candle_runtime_weights(&assets.plan(AdapterKind::Candle)),
            Err(ModelError::InvalidConfig(
                "candle backend weight tensor shape does not match config"
            ))
        );
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn rejects_candle_missing_embedding_tensor() {
        let assets = TempAssets::new(
            "candle-missing",
            "model.safetensors",
            valid_config(),
            safetensors_bytes("other.weight", &[2, 4]),
        );

        assert_eq!(
            validate_candle_runtime_weights(&assets.plan(AdapterKind::Candle)),
            Err(ModelError::InvalidConfig(
                "candle backend missing required weight tensor"
            ))
        );
    }
}
